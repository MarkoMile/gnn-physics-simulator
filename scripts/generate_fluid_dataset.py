"""
Generate a WCSPH fluid simulation dataset for fine-tuning a GNN trained on DeepMind's WaterDrop.

Produces .npz files + metadata.json in the exact format expected by dataset.py's
load_raw_data() / ParticleDataset, aligned with the WaterDrop conventions:
  - bounds:  [[0.1, 0.9], [0.1, 0.9]]
  - dt:      0.0025 (effective frame delta)
  - h:       0.015  (smoothing length = connectivity radius)
  - particle_type: 5  (DeepMind's fluid type)
  - 1000-frame trajectories
"""

import argparse
import json
import os
import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulation import FluidSimulation


# ─── WaterDrop-aligned defaults ────────────────────────────────────────────────
WATERDROP_BOUNDS = [[0.1, 0.9], [0.1, 0.9]]
WATERDROP_DT = 0.0025          # Effective frame-to-frame dt
INTEGRATION_DT = 0.0005        # Sub-step dt for WCSPH stability
SAVE_EVERY = 5                 # INTEGRATION_DT * SAVE_EVERY = WATERDROP_DT
SMOOTHING_LENGTH = 0.015       # = connectivity_radius
SEQUENCE_LENGTH = 1000         # Frames per trajectory
POSITION_SCALE = 0.8           # Container size: [0, 0.8], shifted by +0.1 → [0.1, 0.9]
OFFSET = 0.1                   # Coordinate shift to align with WaterDrop bounds
PARTICLE_TYPE_FLUID = 5        # DeepMind's encoding for fluid particles


def run_trajectory(num_particles: int, seed: int) -> dict:
    """Run a single WCSPH trajectory and return it."""
    rng = np.random.RandomState(seed)
    
    # Total simulation time to produce SEQUENCE_LENGTH frames
    total_time = (SEQUENCE_LENGTH - 1) * SAVE_EVERY * INTEGRATION_DT

    sim = FluidSimulation(
        num_particles=num_particles,
        gravitational_constant=9.81,
        softening_length=SMOOTHING_LENGTH,
        integrator='symplectic_euler',
        position_scale=POSITION_SCALE,
        rest_density=1000.0,
        stiffness=1000.0,
        exponent=7.0,
        viscosity=0.3,
    )

    # Randomized initial drop center within the inner region of the container.
    # Leave margins so the drop grid fits inside [0, POSITION_SCALE].
    margin = 0.15 * POSITION_SCALE
    cx = rng.uniform(margin, POSITION_SCALE - margin)
    cy = rng.uniform(0.3 * POSITION_SCALE, POSITION_SCALE - margin)

    sim.initialize_random_state(
        position_scale=POSITION_SCALE,
        velocity_scale=0.0,
        mass_range=(1.0, 1.0),
        start=(cx, cy),
    )

    positions, velocities, _ = sim.simulate(
        total_time=total_time,
        dt=INTEGRATION_DT,
        save_every=SAVE_EVERY,
    )

    # Trim to exactly SEQUENCE_LENGTH if the simulator produced extra frames
    positions = positions[:SEQUENCE_LENGTH]
    velocities = velocities[:SEQUENCE_LENGTH]

    # Shift coordinates by OFFSET so [0, 0.8] → [0.1, 0.9]
    positions = positions + OFFSET

    return {
        'position': positions.astype(np.float32),
        'velocity': velocities.astype(np.float32),
        'particle_type': np.full(num_particles, PARTICLE_TYPE_FLUID, dtype=np.int64),
        'mass': np.ones(num_particles, dtype=np.float32),
    }


def generate_dataset(
    output_dir: str,
    num_train: int,
    num_valid: int,
    num_test: int,
    min_particles: int,
    max_particles: int,
    seed: int,
):
    """Generate full train/valid/test dataset."""
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.RandomState(seed)

    total = num_train + num_valid + num_test
    # Pre-sample particle counts and per-trajectory seeds
    particle_counts = rng.randint(min_particles, max_particles + 1, size=total)
    traj_seeds = rng.randint(0, 2**31, size=total)

    trajectories = []
    from tqdm.auto import tqdm
    for i in tqdm(range(total), desc="Generating WCSPH trajectories"):
        n = int(particle_counts[i])
        print(f"\n[Trajectory {i+1}/{total}] particles={n}, seed={traj_seeds[i]}")
        traj = run_trajectory(n, int(traj_seeds[i]))
        trajectories.append(traj)
        print(f"  → pos range: [{traj['position'].min():.4f}, {traj['position'].max():.4f}]")

    # Split
    splits = {
        'train': trajectories[:num_train],
        'valid': trajectories[num_train:num_train + num_valid],
        'test':  trajectories[num_train + num_valid:],
    }

    # Compute global velocity and acceleration statistics across ALL trajectories
    all_vels = []
    all_accs = []
    dim = 2
    for traj in trajectories:
        vel = traj['velocity']          # [T, N, D]
        acc = vel[1:] - vel[:-1]        # [T-1, N, D]
        all_vels.append(vel.reshape(-1, dim))
        all_accs.append(acc.reshape(-1, dim))

    all_vels = np.concatenate(all_vels, axis=0)
    all_accs = np.concatenate(all_accs, axis=0)

    metadata = {
        "bounds": WATERDROP_BOUNDS,
        "sequence_length": SEQUENCE_LENGTH,
        "default_connectivity_radius": SMOOTHING_LENGTH,
        "dim": dim,
        "dt": WATERDROP_DT,
        "vel_mean": all_vels.mean(axis=0).tolist(),
        "vel_std":  all_vels.std(axis=0).tolist(),
        "acc_mean": all_accs.mean(axis=0).tolist(),
        "acc_std":  all_accs.std(axis=0).tolist(),
    }

    # Save metadata
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved metadata → {meta_path}")
    print(f"  vel_mean={metadata['vel_mean']}, vel_std={metadata['vel_std']}")
    print(f"  acc_mean={metadata['acc_mean']}, acc_std={metadata['acc_std']}")

    # Save splits as .npz (format matches load_raw_data's npz loader)
    for split_name, split_trajs in splits.items():
        if len(split_trajs) == 0:
            continue
        save_dict = {f"trajectory_{i}": traj for i, traj in enumerate(split_trajs)}
        out_path = os.path.join(output_dir, f"{split_name}.npz")
        np.savez_compressed(out_path, **save_dict)
        print(f"Saved {split_name} ({len(split_trajs)} trajectories) → {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate WCSPH fluid dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output", "-o", type=str, default="data/wcsph_transfer",
                        help="Output directory for the dataset")
    parser.add_argument("--num-train", type=int, default=8,
                        help="Number of training trajectories")
    parser.add_argument("--num-valid", type=int, default=1,
                        help="Number of validation trajectories")
    parser.add_argument("--num-test", type=int, default=1,
                        help="Number of test trajectories")
    parser.add_argument("--min-particles", type=int, default=200,
                        help="Minimum particles per trajectory")
    parser.add_argument("--max-particles", type=int, default=400,
                        help="Maximum particles per trajectory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    args = parser.parse_args()

    print("=" * 60)
    print("WCSPH Dataset Generator")
    print("=" * 60)
    print(f"  Output:         {args.output}")
    print(f"  Trajectories:   {args.num_train} train / {args.num_valid} valid / {args.num_test} test")
    print(f"  Particles:      {args.min_particles}-{args.max_particles}")
    print(f"  Sequence length: {SEQUENCE_LENGTH} frames")
    print(f"  Effective dt:   {WATERDROP_DT}")
    print(f"  Smoothing h:    {SMOOTHING_LENGTH}")
    print(f"  Bounds:         {WATERDROP_BOUNDS}")
    print(f"  Particle type:  {PARTICLE_TYPE_FLUID}")
    print(f"  Seed:           {args.seed}")
    print("=" * 60)

    generate_dataset(
        output_dir=args.output,
        num_train=args.num_train,
        num_valid=args.num_valid,
        num_test=args.num_test,
        min_particles=args.min_particles,
        max_particles=args.max_particles,
        seed=args.seed,
    )

    print("\nDataset generation complete!")


if __name__ == "__main__":
    main()
