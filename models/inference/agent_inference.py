import torch
import torch.distributed as dist
from emulator.keys import CLASS_TO_KEY
from safetensors.torch import load_file
import math
from models.dataclass import DataArguments, TrainingArguments, ModelArguments, InferenceArguments, IDMArguments
from models.train.train_agent import create_dataset, init_model, train_with_rollback, AgentObjectiveManager
from models.train.train_idm import train_idm_best_checkpoint
from transformers import AutoImageProcessor, AutoModel
from models.util.misc import local_model_map
import pickle
import os
    
class ASHOnlineAgent:
    def __init__(self,
                model_args: ModelArguments,
                training_args: TrainingArguments,
                data_args: DataArguments,
                inference_args: InferenceArguments,
                idm_args: IDMArguments,
                ):
        
        self.model_args: ModelArguments = model_args
        self.training_args: TrainingArguments = training_args
        self.data_args: DataArguments = data_args
        self.inference_args: InferenceArguments = inference_args
        self.idm_args: IDMArguments = idm_args

        assert inference_args.model_checkpoint is None, "Use model_args.load_path"
         
        self.sampling_strategy = inference_args.sampling_strategy
        self.context_len = inference_args.context_length
        self.actions_per_second = inference_args.actions_per_second
        self.model_fps = inference_args.agent_fps
        self.buffersize = self.context_len // self.model_fps * self.actions_per_second
        self.stride = self.actions_per_second // self.model_fps
        self.temperature = inference_args.temperature

        self.num_objectives = self.data_args.num_objectives

        self.model, self.idm, self.processor, self.device = init_model(self.model_args, self.training_args)
        print(f"****************{model_args.load_path}")
        if model_args.load_path is not None:
            self.model.load_state_dict(load_file(model_args.load_path))
        self.model.to(self.device).eval()
        self.agent_frames = torch.zeros(self.buffersize, 3, 160, 240, dtype=torch.uint8, device=self.device)
        self.input_ids = torch.zeros(1, self.buffersize, dtype=torch.long)
        self.idx = 0
        self.total_steps = 0

        class _Unpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == '__main__' and name == 'AgentObjectiveManager':
                    return AgentObjectiveManager
                return super().find_class(module, name)

        with open(model_args.objective_load_path, 'rb') as f:
            self.objective_manager = _Unpickler(f).load()

        dino_id: str = local_model_map(model_args.dino_name_or_path)
        self.dino_processor = AutoImageProcessor.from_pretrained(dino_id, use_fast=True)
        self.dino_model = AutoModel.from_pretrained(dino_id).eval().to(self.device)
        print("[AGENT] Initialized agent")
     
    def train_agent(self, intervals: str, bootstrap: int, query_embeds: list, video_frames: list):
        train_ds, eval_ds, objective_manager = create_dataset(intervals, self.processor, bootstrap, split = self.inference_args.train_eval_split, query_embeds=query_embeds, video_frames=video_frames, data_args=self.data_args)
        self.model.train()
        train_with_rollback(self.model, self.training_args, train_ds=train_ds, eval_ds=eval_ds)
        self.model.eval()
        self.objective_manager = objective_manager

    def train_idm(self, data_sources: "str | list[str]"):
        self.idm.train()
        train_idm_best_checkpoint(self.idm, self.idm_args, data_sources, split = self.inference_args.train_eval_split)
        self.idm.eval()

    def broadcast_agent_state(self, src=1):
        if dist.is_initialized():
            dist.broadcast(self.agent_frames, src=src)
            input_ids_device = self.input_ids.to(self.device)
            dist.broadcast(input_ids_device, src=src)
            self.input_ids = input_ids_device.to('cpu')

    def clear_memory(self):
        import gc
        self.objective_manager = None
        gc.collect()
        dist.barrier()

    @torch.no_grad()
    def infer_action(self, frame: torch.Tensor): # (C, H, W)

        frame = frame.to(self.device)

        if self.idx == self.buffersize - 1:
            self.agent_frames = torch.cat((self.agent_frames[1:], frame.unsqueeze(0)), dim=0)
        else:
            self.agent_frames[self.idx] = frame

        images = self.agent_frames[self.idx % self.stride::self.stride]

        self.total_steps += 1
        if self.total_steps % self.context_len == 0:
            dino_inputs = self.dino_processor(images=images.cpu(), return_tensors="pt")
            dino_inputs = {k: v.to(self.device) for k, v in dino_inputs.items()}
            dino_outputs = self.dino_model(**dino_inputs)
            embeds = dino_outputs.last_hidden_state[:, 0]  # CLS token, (S, D)
            embeds = embeds / embeds.norm(p=2, dim=-1, keepdim=True)
            self.objective_manager.mine_and_add_objectives(
                [e.cpu().numpy() for e in embeds],
                images.cpu()
            )
            print(f"[GPU {dist.get_rank()}] has achieved {len(self.objective_manager.achieved_objectives)} objectives")

        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device).unsqueeze(0)  # (1, S, C, H, W)

        objectives_list = self.objective_manager.retrieve_last_n_objectives(self.num_objectives)
        n_pad = self.num_objectives - len(objectives_list)
        if objectives_list:
            obj_pixels = self.processor(images=[o.cpu() for o in objectives_list], return_tensors="pt")["pixel_values"]
            if n_pad > 0:
                obj_pixels = torch.cat([obj_pixels, torch.zeros(n_pad, *obj_pixels.shape[1:])], dim=0)
        else:
            obj_pixels = torch.zeros(self.num_objectives, *pixel_values.shape[2:])
        objectives = obj_pixels.to(self.device).unsqueeze(0)  # (1, num_objectives, C, H, W)

        input_ids_dev = self.input_ids.to(self.device)[:, self.idx % self.stride::self.stride]

        output = self.model(input_ids=input_ids_dev, pixel_values=pixel_values, objectives=objectives)
        logits = output["logits"][0, math.floor(self.idx / self.stride)]                     # (num_classes,)

        if self.sampling_strategy == "default":
            cls = torch.argmax(logits, dim=-1)
        elif self.sampling_strategy == "temperature":
            probs = torch.softmax(logits / self.temperature, dim=-1)       # temperature sampling
            cls = torch.multinomial(probs, num_samples=1).squeeze(-1)      # sample an index

        if self.idx == 0: # Go right
            cls = torch.tensor(7, dtype=torch.long, device=probs.device)

        if self.idx == self.buffersize - 1:
            self.input_ids = torch.cat((self.input_ids[:, 1:], cls.view(1, 1).to('cpu')), dim=1)
        else:
            self.input_ids[0, self.idx] = cls.to('cpu')

        if self.idx < self.buffersize - 1:
            self.idx += 1

        key = CLASS_TO_KEY[cls.item()]
    
        if "LZ_MODE" in os.environ and key == "start":
            key = "none"

        return key 
    

        return CLASS_TO_KEY[cls.item()]