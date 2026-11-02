# ASH: Agents that Self-Hone in Long-Horizon Worlds

[![Paper](https://img.shields.io/badge/arXiv-2605.14211-b31b1b.svg)](https://arxiv.org/abs/2605.14211)
[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://toolbox-hq.github.io/ASH/)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Toolbox--HQ%2FASH-yellow.svg)](https://huggingface.co/Toolbox-HQ/ASH)

[Benjamin Schneider](https://benjaminschneider.ca/)<sup>1</sup>, Xavier Schneider<sup>1</sup>, [Victor Zhong](https://www.victorzhong.com/)<sup>1</sup>, Sun Sun<sup>2</sup>

<sup>1</sup>University of Waterloo &nbsp;&nbsp; <sup>2</sup>National Research Council Canada

**NeurIPS 2026**

ASH is an agentic system that learns a long-horizon policy from unlabeled, noisy internet video, without reward shaping or expert annotation. It plays *Pokémon Emerald* and *The Legend of Zelda: The Minish Cap* for hours at a time, and when it gets stuck it finds helpful gameplay video on the internet, labels it with its own Inverse Dynamics Model (IDM), and learns from it.

## Overview

Long-horizon tasks are hard for current methods because they rely on hand-engineered rewards or action-labeled demonstrations, and neither scales. Behavioral cloning needs massive annotated datasets and cannot model dependencies that span hours. Reinforcement learning needs a dense reward for every intermediate milestone.

ASH is analogous to a human who plays a game, gets stuck, and looks up a video. The system follows a **dynamic bootstrap loop**:

- **Self-improvement from internet video.** When ASH gets stuck, it trains an IDM on its own trajectories and uses that IDM to extract action supervision from relevant, observation-only internet video.
- **Key moments as long-term memory.** ASH uses unsupervised learning (HDBSCAN clustering over DINOv2 embeddings of retrieved video) to identify key moments in the environment. Recent key moments form a long-term memory that lets the policy make decisions that hinge on information from many steps ago.

<p align="center">
  <img src="https://toolbox-hq.github.io/ASH/static/images/teaser.png" alt="ASH teaser" width="90%">
</p>

## How ASH works

ASH deploys *N* agents in parallel under a single shared policy π, conditioned on per-agent short-term and long-term memory. A key-moment classifier derived from internet data decides whether an observation is a new key moment. A new key moment is appended to that agent's long-term memory and resets a per-agent stuck-timer. When any timer exceeds a threshold Δ, ASH bootstraps.

1. **Inference.** Run the *N* agents under π. On each step, detect new key moments and add them to that agent's long-term memory. Halt when any agent fails to find a new key moment for Δ steps.
2. **Retrieval.** Embed the agents' trajectories with DINOv2. Retrieve the top-*k* visually similar videos from the internet corpus for each trajectory and union them into a retrieved dataset.
3. **Bootstrapping.** Update the key-moment classifier on the retrieved videos to discover new key moments. Update the IDM on the agents' trajectories. Update π on the retrieved videos, using the IDM for pseudo-actions and the key moments to construct long-term memory. Return to step 1.

**Dual memory.** Short-term memory holds the most recent observation-action pairs for reactive control. Long-term memory holds the most recent key-moment observations. Together they let ASH disambiguate situations whose recent observations look identical but whose correct next action depends on what happened long ago, such as which exit to take from Oldale Town depending on whether the rival has already been met.

## Results

Agents are evaluated with **milestones**, human-defined markers of progress derived from online walkthroughs, over an 8-hour playthrough. Baselines are Voyager+Opus 5, VPT, Offline BC, and a Retrieval-augmented Embodied Agent (RAEA).



<p align="center">
  <img src="https://toolbox-hq.github.io/ASH/static/images/Milestone_Figure.png" alt="Milestone progress" width="90%">
</p>


**Takeaways:** 

- **Self-improvement is necessary for sustained progression.** ASH reaches milestone 12 in both games. No baseline exceeds milestone 8 in Pokémon or 6 in Zelda. VPT and offline BC plateau once the games introduce dynamics underrepresented in their training data.
- **ASH adapts to changing dynamics.** At the Zelda castle entrance the visuals shift abruptly and frozen policies collapse into button-mashing. ASH's policy also falters, but recovers after a bootstrap. The same pattern appears in Pokémon when the agent first meets the battle interface.
- **Bootstrapping evolves new skills.** ASH's initial checkpoint cannot battle at all. After hour 4 it wins 87% of the Pokémon battles it enters, and in Zelda it kills enemies with its sword in 75% of trajectories.

## Running ASH

Instructions coming soon.

## Released models

Weights and assets are hosted at [Toolbox-HQ/ASH](https://huggingface.co/Toolbox-HQ/ASH) on Hugging Face.

```
ASH/
├── assets/
│   ├── Qwen3-1.7B/                 # language backbone of the policy
│   ├── siglip-base-patch16-224/    # vision encoder of the policy
│   ├── dinov2-base/                # retrieval and key-moment embeddings
│   ├── rnd_idm_model.pt            # IDM initialization
│   └── truck_start.state           # Pokémon Emerald start-of-game save state
├── checkpoints/
│   ├── checkpoint-1/               # earliest bootstrap
│   │   ├── agent.safetensors       # policy weights
│   │   ├── idm.pt                  # IDM weights
│   │   └── objective_manager.pkl   # key-moment classifier (HDBSCAN) and key moments
│   ├── ...
│   └── checkpoint-10/              # latest bootstrap
└── rom/                            # place your own Pokémon Emerald ROM here (not distributed)
```

`checkpoint-1` through `checkpoint-10` are successive bootstraps of a single Pokémon Emerald run. Each one is a start-of-game checkpoint: the same policy, IDM, and key-moment classifier the run had at that point, with the long-term memory emptied, so the agent can be launched from the beginning of the game with any of them.

## Repository layout

```
main.py          # online self-improvement loop (inference → retrieval → bootstrapping)
models/
├── model/       # policy (LMObjectiveAgent) and IDM architectures
├── train/       # policy, IDM, and key-moment training
├── inference/   # online agent, DINOv2 embedding, video retrieval
├── dataclass/   # configuration dataclasses
└── util/        # distributed, reproducibility, and misc helpers
emulator/        # mGBA emulator bindings (frame capture, input, save states, video)
config/          # YAML run configurations
script/          # SLURM / container launch scripts
dconfig/         # Apptainer and Docker definitions
```

## Citation

```bibtex
@misc{schneider2026ashagentsselfhonelonghorizon,
      title={ASH: Agents that Self-Hone in Long-Horizon Worlds}, 
      author={Benjamin Schneider and Xavier Schneider and Victor Zhong and Sun Sun},
      year={2026},
      eprint={2605.14211},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2605.14211}, 
}
```
