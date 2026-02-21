import argparse
import yaml
import os
import sys

# Add parent directory to path so we can import simulation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation import generate_dataset

def main():
    parser = argparse.ArgumentParser(description="Generate N-Body dataset from YAML config")
    parser.add_argument("--config", "-c", type=str, default="configs/default.yaml",
                        help="Path to YAML configuration file")
    parser.add_argument("--output", "-o", type=str, default="data/processed",
                        help="Path to save the generated dataset")
    args = parser.parse_args()
    
    config_path = args.config
    # Fallback if scripts are run from root
    if not os.path.exists(config_path) and os.path.exists(os.path.basename(config_path)):
        config_path = os.path.basename(config_path)
        
    print(f"Loading configuration from: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    if "simulation" not in config:
        raise ValueError(f"No 'simulation' block found in {config_path}")
        
    sim_cfg = config["simulation"]
    
    print(f"Generating {sim_cfg['num_trajectories']} trajectories...")
    print(f"Particles: {sim_cfg['num_particles']} | Time: {sim_cfg['total_time']}s | dt: {sim_cfg['dt']}")
    
    generate_dataset(
        num_trajectories=sim_cfg['num_trajectories'],
        num_particles=sim_cfg['num_particles'],
        total_time=sim_cfg['total_time'],
        dt=sim_cfg['dt'],
        save_every=sim_cfg.get('save_every', 10),
        integrator=sim_cfg.get('integrator', 'rk4'),
        save_path=args.output,
        gravitational_constant=sim_cfg.get('gravitational_constant', 1.0),
        softening_length=sim_cfg.get('softening_length', 0.1),
        position_scale=sim_cfg.get('position_scale', 2.0),
        velocity_scale=sim_cfg.get('velocity_scale', 0.5),
        mass_range=(sim_cfg.get('mass_range_min', 0.8), sim_cfg.get('mass_range_max', 1.2))
    )
    
    print(f"\nDataset successfully generated and saved to {args.output}")

if __name__ == "__main__":
    main()
