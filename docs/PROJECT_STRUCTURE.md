# Project Structure

```
gnn-physics-simulator/
├── configs/                    # Configuration files
│   ├── default.yaml           # Default configuration
│   ├── experiment_euler.yaml  # Euler integrator experiment
│   └── experiment_rk4.yaml    # RK4 integrator experiment
├── notebooks/                  # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_integrator_analysis.ipynb
│   ├── 03_model_development.ipynb
│   ├── 04_training_experiments.ipynb
│   └── 05_results_analysis.ipynb
├── scripts/                    # Executable scripts
│   ├── generate_dataset.py    # Generate N-body simulation data
│   ├── preprocess_data.py     # Preprocess raw data
│   ├── train.py               # Train GNN model
│   ├── evaluate.py            # Evaluate trained model
│   ├── compare_integrators.py # Compare numerical integrators
│   └── visualize.py           # Generate visualizations
├── src/                        # Source code
│   ├── data/                  # Data loading and processing
│   │   ├── dataset.py         # PyTorch dataset classes
│   │   ├── preprocessing.py   # Data preprocessing utilities
│   │   └── graph_utils.py     # Graph construction utilities
│   ├── models/                # Neural network models
│   │   ├── gnn_simulator.py   # Main GNN simulator
│   │   ├── layers.py          # Message passing layers
│   │   └── encoder_decoder.py # Encoder/decoder networks
│   ├── simulation/            # Physics simulation
│   │   ├── integrators.py     # Numerical integrators
│   │   ├── nbody.py           # N-body simulation
│   │   └── data_generation.py # Dataset generation
│   ├── training/              # Training utilities
│   │   ├── trainer.py         # Training loop
│   │   ├── losses.py          # Loss functions
│   │   └── schedulers.py      # Learning rate schedulers
│   ├── evaluation/            # Evaluation utilities
│   │   ├── metrics.py         # Evaluation metrics
│   │   ├── rollout.py         # Rollout evaluation
│   │   └── visualization.py   # Visualization utilities
│   └── utils/                 # General utilities
│       ├── config.py          # Configuration loading
│       ├── logging.py         # Logging utilities
│       └── helpers.py         # Helper functions
├── tests/                      # Unit tests
│   ├── test_integrators.py
│   ├── test_models.py
│   ├── test_data.py
│   ├── test_training.py
│   └── test_evaluation.py
├── docs/                       # Documentation
│   └── PROJECT_STRUCTURE.md   # This file
├── data/                       # Data directory (gitignored)
│   ├── raw/                   # Raw simulation data
│   └── processed/             # Processed data
├── checkpoints/                # Model checkpoints (gitignored)
├── logs/                       # Training logs (gitignored)
├── results/                    # Evaluation results (gitignored)
├── requirements.txt            # Python dependencies
├── setup.py                    # Package setup
├── pyproject.toml              # Project configuration
├── README.md                   # Project overview
└── LICENSE                     # License file
```

## Directory Details

### `configs/`

YAML configuration files for experiments. The `default.yaml` contains base settings that can be overridden by experiment-specific configs.

### `notebooks/`

Jupyter notebooks for interactive exploration, analysis, and visualization. Numbered for suggested order of exploration.

### `scripts/`

Command-line scripts for the main workflow: data generation → preprocessing → training → evaluation.

### `src/`

Main source code organized by functionality:

- **data/**: Dataset classes and data processing utilities
- **models/**: GNN architecture and neural network layers
- **simulation/**: Physics simulation and numerical integrators
- **training/**: Training loop, losses, and schedulers
- **evaluation/**: Metrics, rollout evaluation, and visualization
- **utils/**: Configuration, logging, and helper functions

### `tests/`

Unit tests using pytest. Run with `pytest tests/`.

### `data/`, `checkpoints/`, `logs/`, `results/`

Runtime directories for data, model checkpoints, training logs, and evaluation results. These are typically gitignored.
