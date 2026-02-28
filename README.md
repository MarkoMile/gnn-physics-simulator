# Graph Neural Networks for Simulating Physics

![WCSPH vs GNN Comparison](media/wcsph-gnn-comparison.gif)

A PyTorch Geometric implementation of [Learning to Simulate Complex Physics with Graph Networks](https://arxiv.org/abs/2002.09405) (Sanchez-Gonzalez et al., 2020). Trains a Graph Neural Network to predict particle accelerations from local interactions, reproducing DeepMind's architecture for simulating fluids, with additional fine-tuning on a custom Weakly Compressible Smoothed Particle Hydrodynamics (WCSPH) dataset and support for optimized dataset generation.

This project was developed as an optional assignment for the course **Numerical Algorithms and Numerical Software (E231)** at the Faculty of Technical Sciences, University of Novi Sad. It contrasts classic physically grounded numerical integrators against deep learning Message-Passing approaches.

## Project Structure

```text
gnn-physics-simulator/
├── configs/                 # Configuration files (model, training, dataset, simulation)
│   ├── config.yaml          # Central configuration for default training
│   └── wcsph_transfer.yaml  # WCSPH transfer learning / fine-tuning config
├── models.py                # GNNSimulator: Encoder → Processor → Decoder architecture
├── dataset.py               # ParticleDataset: NPZ loading, noise injection, graph building
├── train.py                 # Training loop: normalized loss, LR decay, checkpointing
├── simulation.py            # WCSPH & N-Body simulators with dataset export
├── integrators.py           # Symplectic Euler, RK4, Forward Euler (Numba JIT optimized)
├── scripts/
│   ├── generate_dataset.py  # CLI to generate synthetic fluid datasets
│   └── download_dataset.py  # Download DeepMind datasets (WaterDrop, etc.)
├── presentation/            # Interactive project presentation and generation scripts
│   ├── Prezentacija.ipynb   # Presentation notebook (in Serbian) for the final presentation
│   └── utils.py             # MP4 matplotlib animation generation utilities
├── notebooks/
│   ├── TestSim.ipynb        # Pure numerical WCSPH simulation testing and analysis
│   ├── Visualize.ipynb      # Rollout visualization for DeepMind WaterDrop dataset
│   └── Visualize_wcsph.ipynb# Rollout visualization for custom WCSPH dataset
├── utils/
│   ├── train_kaggle.ipynb           # Kaggle notebook: train on full WaterDrop dataset
│   └── train_generated_kaggle.ipynb # Kaggle notebook: generate + train synthetic data
├── data/                    # Dataset directory 
├── checkpoints/             # Saved model weights and training states (.pt files)
│   ├── best_model_wcsph.pt  # Model weights after WCSPH fine-tuning
│   └── best_model.pt        # Model weights after pre-training on WaterDrop dataset
└── media/                   # Saved animations and comparison GIFs
```

## Core Features & Methodology

The simulator utilizes a multi-stage approach to learning and simulating fluid dynamics:

1. **Physical Simulation (WCSPH):** Developed a custom *Weakly Compressible Smoothed Particle Hydrodynamics* (WCSPH) simulator with boundary particles handling and a highly stable **Symplectic Euler** numerical integrator optimized with **Numba JIT**.
2. **Pre-Training:** The GNN model undergoes pre-training on DeepMind's extensive WaterDrop dataset to learn general fluid dynamics and container interactions.
3. **Fine-Tuning:** Using the custom WCSPH numerical simulation (`scripts/generate_dataset.py`), generated a proprietary dataset. The model is fine-tuned on this dataset to adapt to the specific SPH formulation.
4. **GNN Architecture:** An Encode-Process-Decode approach using `InteractionNetwork` message passing blocks, transforming kinetic data into a Radius Graph for prediction, as described in the original paper.

## Installation

```bash
git clone https://github.com/MarkoMile/gnn-physics-simulator.git
cd gnn-physics-simulator

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

## Quick Start

### Train on DeepMind's WaterDrop Dataset

```bash
# 1. Download the WaterDrop dataset
python scripts/download_dataset.py

# 2. Train the model
python train.py --config configs/config.yaml

# With Weights & Biases logging:
python train.py --config configs/config.yaml --wandb
```

Or use the provided Kaggle-ready notebook [train_kaggle.ipynb](utils/train_kaggle.ipynb).

### Generate and Fine-Tune on Synthetic WCSPH Data

```bash
# Generate a WCSPH Fluid dataset
python scripts/generate_dataset.py --config configs/wcsph_transfer.yaml --output data/generated

# Fine-tune the pre-trained model on the generated data
python train.py --config configs/wcsph_transfer.yaml --load checkpoints/best_model.pt
```

Or use the provided Kaggle-ready notebook [finetune_wcsph_kaggle.ipynb](utils/finetune_wcsph_kaggle.ipynb).

### Viewing the Presentation

The core results, algorithms, and 60fps rollout comparisons are structured into an interactive [notebook presentation](presentation/Prezentacija.ipynb) (in Serbian).

Visualizations for the WCSPH simulator, GNN trained on WaterDrop, and GNN fine-tuned for WCSPH are also available in separate notebooks in the [notebooks/](notebooks/) directory.

## Configuration

All hyper-parameters, training details, and simulation setup variables are centralized in the `configs/` directory. See [configs/config.yaml](configs/config.yaml) and [configs/wcsph_transfer.yaml](configs/wcsph_transfer.yaml) for specific fields.

## References

- [Learning to Simulate Complex Physics with Graph Networks](https://arxiv.org/abs/2002.09405) — Sanchez-Gonzalez et al., 2020
- [Interaction Networks for Learning about Objects, Relations and Physics](https://arxiv.org/abs/1612.00222) — Battaglia et al., 2016
- [Simulating Free Surface Flows with SPH](https://doi.org/10.1006/jcph.1994.1034) — Monaghan, 1994
- [Weakly compressible SPH for free surface flows](https://diglib.eg.org/server/api/core/bitstreams/df5651a7-e578-4db9-a056-c8c36e925067/content) — Becker & Teschner, 2007
- [Versatile Rigid-Fluid Coupling for Incompressible SPH](https://dl.acm.org/doi/10.1145/2185520.2185558) — Akinci et al., 2012
- [Particle-Based Fluid Simulation for Interactive Applications](https://matthias-research.github.io/pages/publications/sca03.pdf) — Müller et al., 2003

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
