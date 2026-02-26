from integrators import get_integrator

"""
N-body gravitational simulation.

This module implements the N-body problem for generating
training data for the GNN simulator.
"""

import numpy as np
import os
from scipy.spatial import cKDTree
import json
from typing import Tuple, Optional, Dict, Any


class NBodySimulation:
    """
    N-body gravitational simulation.
    
    Simulates N particles interacting via gravitational forces.
    """
    
    def __init__(
        self,
        num_particles: int,
        gravitational_constant: float = 1.0,
        softening_length: float = 0.01,
        integrator: str = 'rk4'
    ):
        """
        Initialize N-body simulation.
        
        Args:
            num_particles: Number of particles
            gravitational_constant: Gravitational constant G
            softening_length: Softening parameter to avoid singularities
            integrator: Integration method name
        """
        self.num_particles = num_particles
        self.gravitational_constant = gravitational_constant
        self.softening_length = softening_length
        self.integrator = get_integrator(integrator)
    def initialize_random_state(
        self,
        position_scale: float = 1.0,
        velocity_scale: float = 0.1,
        mass_range: Tuple[float, float] = (0.5, 1.5)
    ):
        """
        Initialize particles with random positions, velocities, and masses.
        
        Args:
            position_scale: Scale for initial positions
            velocity_scale: Scale for initial velocities
            mass_range: Range for random masses (min, max)
        """
        self.positions = np.random.uniform(-position_scale, position_scale, (self.num_particles, 2))
        self.velocities = np.random.uniform(-velocity_scale, velocity_scale, (self.num_particles, 2))
        self.masses = np.random.uniform(mass_range[0], mass_range[1], self.num_particles)
    
    def compute_gravitational_forces(self, positions: np.ndarray, masses: np.ndarray) -> np.ndarray:
        """
        Compute gravitational forces between all particles.
        
        Args:
             positions: Particle positions [N, D]
            masses: Particle masses [N]
            
        Returns:
             Accelerations [N, D]
        """
        # Displacement vectors between all pairs [N, N, D]
        # r[i, j, :] is the vector from particle i to particle j
        r = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]
        
        # Squared distances [N, N]
        r_sq = np.sum(r**2, axis=-1)
        
        # Add softening length squared to avoid singularity (and division by zero)
        r_sq_softened = r_sq + self.softening_length**2
        
        # Inverse distance cubed [N, N]
        inv_r3 = r_sq_softened ** (-1.5)
        
        # Acceleration computation: 
        # masses array is [N], we reshape to [1, N, 1] to broadcast over interacting particles (j) and 3D coords
        # sum over axis=1 which represents the contributions of all particles j to particle i
        accelerations = self.gravitational_constant * np.sum(
            masses[np.newaxis, :, np.newaxis] * inv_r3[:, :, np.newaxis] * r,
            axis=1
        )
        
        return accelerations
    
    def simulate(
        self,
        total_time: float,
        dt: float,
        save_every: int = 1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run the simulation.
        
        Args:
            total_time: Total simulation time
            dt: Time step
            save_every: Save state every N steps
            
        Returns:
            Tuple of (positions, velocities, times) arrays
        """
        if not hasattr(self, 'positions') or not hasattr(self, 'velocities') or not hasattr(self, 'masses'):
            raise ValueError("Simulation state not initialized. Call initialize_random_state() first.")
            
        total_steps = int(total_time / dt)
        
        # Arrays to store the history
        num_saved = (total_steps // save_every) + 1
        dim = self.positions.shape[1]
        history_positions = np.zeros((num_saved, self.num_particles, dim))
        history_velocities = np.zeros((num_saved, self.num_particles, dim))
        history_times = np.zeros(num_saved)
        
        # Extract current state to iterate over
        pos = self.positions.copy()
        vel = self.velocities.copy()
        
        # to pass to the integrator taking just the positions array
        def force_fn(positions_array):
            return self.compute_gravitational_forces(positions_array, self.masses)
            
        save_idx = 0
        from tqdm.auto import tqdm
        for step in tqdm(range(total_steps + 1), desc="N-Body Simulation Progress", leave=False):
            # Save current state
            if step % save_every == 0:
                history_positions[save_idx] = pos
                history_velocities[save_idx] = vel
                history_times[save_idx] = step * dt
                save_idx += 1
                
            # Perform integration step (skip on the very last step since we just wanted to save it)
            if step < total_steps:
                pos, vel = self.integrator.step((pos, vel), dt, force_fn)
                
        # Update internal class state
        self.positions = pos
        self.velocities = vel
        
        # Truncate arrays if total_steps wasn't perfectly divisible by save_every
        history_positions = history_positions[:save_idx]
        history_velocities = history_velocities[:save_idx]
        history_times = history_times[:save_idx]
        
        return history_positions, history_velocities, history_times
    
    def compute_total_energy(self) -> float:
        """
        Compute total energy (kinetic + potential).
        
        Returns:
            Total energy of the system
        """
        if not hasattr(self, 'positions') or not hasattr(self, 'velocities') or not hasattr(self, 'masses'):
            raise ValueError("Simulation state not initialized. Call initialize_random_state() first.")
            
        # Kinetic Energy: K = 0.5 * sum(m * v^2)
        v_sq = np.sum(self.velocities**2, axis=-1)
        kinetic_energy = 0.5 * np.sum(self.masses * v_sq)
        
        # Potential Energy: U = -G * sum_{i < j} (m_i * m_j / sqrt(r_{ij}^2 + eps^2))
        r = self.positions[np.newaxis, :, :] - self.positions[:, np.newaxis, :]
        r_sq = np.sum(r**2, axis=-1)
        
        # Add softening length squared to distance calculation
        r_sq_softened = r_sq + self.softening_length**2
        
        # Computes interactions matrix
        potential_matrix = -self.gravitational_constant * (
            self.masses[:, np.newaxis] * self.masses[np.newaxis, :]
        ) / np.sqrt(r_sq_softened)
        
        # We only take the upper triangle (k=1 excludes diagonal) 
        # to ensure we don't double count pairs and avoid self-interaction
        potential_energy = np.sum(np.triu(potential_matrix, k=1))
        
        return float(kinetic_energy + potential_energy)

class FluidSimulation:
    """
    Fluid dynamics simulation using Weakly Compressible Smoothed Particle Hydrodynamics (WCSPH).
    Simulates water drop splash interactions using Poly6 (density), Spiky (pressure), and 
    Laplacian (viscosity) kernels for fluid dynamics natively in 2D.
    """
    
    def __init__(
        self,
        num_particles: int,
        gravitational_constant: float = 9.81,
        softening_length: float = 0.1, # Acts as the SPH Smoothing Length (h)
        integrator: str = 'symplectic_euler', # Crucial: SPH stability depends on Symplectic Euler
        position_scale: float = 1.0,
        # WCSPH specific parameters
        rest_density: float = 1000.0,
        stiffness: float = 35000.0, # Tait's equation gas constant (k)
        viscosity: float = 0.05,
        exponent: float = 7.0
    ):
        self.num_particles = num_particles
        self.gravitational_constant = gravitational_constant
        self.smoothing_length = softening_length
        self.integrator = get_integrator(integrator)
        self.position_scale = position_scale
        
        self.rest_density = rest_density
        self.stiffness = stiffness
        self.viscosity = viscosity
        self.exponent = exponent
        
        # 2D Cubic Spline Kernel constants
        h = self.smoothing_length
        self.kernel_k = 40.0 / (7.0 * np.pi * h**2)
        self.kernel_l = 240.0 / (7.0 * np.pi * h**2)
        self.kernel_0 = self.kernel_k * 1.0  # Value at r=0 (q=0)
        
    def cubic_kernel_2d(self, r_dist: np.ndarray) -> np.ndarray:
        """2D Cubic Spline Kernel evaluation"""
        q = r_dist / self.smoothing_length
        res = np.zeros_like(q)
        
        # q <= 0.5
        mask1 = q <= 0.5
        res[mask1] = self.kernel_k * (6.0 * q[mask1]**3 - 6.0 * q[mask1]**2 + 1.0)
        
        # 0.5 < q <= 1.0
        mask2 = (q > 0.5) & (q <= 1.0)
        res[mask2] = self.kernel_k * (2.0 * (1.0 - q[mask2])**3)
        return res
        
    def cubic_kernel_2d_gradient(self, r: np.ndarray, r_dist: np.ndarray) -> np.ndarray:
        """2D Cubic Spline Kernel Gradient evaluation"""
        q = r_dist / self.smoothing_length
        res = np.zeros_like(r)
        
        mask_valid = (q <= 1.0) & (r_dist > 1e-12)
        if not np.any(mask_valid):
            return res
            
        # q <= 0.5
        mask1 = mask_valid & (q <= 0.5)
        if np.any(mask1):
            q1 = q[mask1]
            factor1 = self.kernel_l * q1 * (3.0 * q1 - 2.0)
            gradq_x1 = r[mask1, 0] / (r_dist[mask1] * self.smoothing_length)
            gradq_y1 = r[mask1, 1] / (r_dist[mask1] * self.smoothing_length)
            res[mask1, 0] = factor1 * gradq_x1
            res[mask1, 1] = factor1 * gradq_y1
            
        # 0.5 < q <= 1.0
        mask2 = mask_valid & (q > 0.5)
        if np.any(mask2):
            q2 = q[mask2]
            factor2 = -self.kernel_l * (1.0 - q2)**2
            gradq_x2 = r[mask2, 0] / (r_dist[mask2] * self.smoothing_length)
            gradq_y2 = r[mask2, 1] / (r_dist[mask2] * self.smoothing_length)
            res[mask2, 0] = factor2 * gradq_x2
            res[mask2, 1] = factor2 * gradq_y2
            
        return res

    def _generate_boundary_particles(self):
        """Generate static boundary particles along all 4 container walls.
        
        Uses 2 layers of particles (Akinci et al. 2012) at the same spacing as
        fluid particles to ensure proper kernel support near walls. Boundary
        particles contribute density/pressure to fluid but are never moved.
        """
        spacing = self.smoothing_length / 2.0
        limit = self.position_scale
        num_layers = 2
        
        boundary_pts = []
        
        # Bottom and top walls (extend past limit to avoid corner holes)
        x_vals = np.arange(-limit - num_layers * spacing, limit + num_layers * spacing + 1e-5, spacing)
        for layer in range(num_layers):
            offset = (layer + 0.5) * spacing  # half-spacing offset from wall
            # Bottom wall
            for x in x_vals:
                boundary_pts.append([x, -limit - offset])
            # Top wall
            for x in x_vals:
                boundary_pts.append([x, limit + offset])
        
        # Left and right walls
        y_vals = np.arange(-limit, limit + 1e-5, spacing)
        for layer in range(num_layers):
            offset = (layer + 0.5) * spacing
            # Left wall
            for y in y_vals:
                boundary_pts.append([-limit - offset, y])
            # Right wall
            for y in y_vals:
                boundary_pts.append([limit + offset, y])
        
        self.boundary_positions = np.array(boundary_pts)
        self.num_boundary = len(self.boundary_positions)
        
        # Compute pseudo-mass (psi) for boundary particles (Akinci et al. 2012)
        # psi_i = rest_density / sum_j W(r_ij)
        from scipy.spatial import cKDTree
        tree = cKDTree(self.boundary_positions)
        pairs = tree.query_pairs(r=self.smoothing_length)
        
        deltas = np.full(self.num_boundary, self.kernel_0)
        if pairs:
            pairs_arr = np.array(list(pairs))
            i_idx = pairs_arr[:, 0]
            j_idx = pairs_arr[:, 1]
            r_vec = self.boundary_positions[i_idx] - self.boundary_positions[j_idx]
            r_dist = np.linalg.norm(r_vec, axis=1)
            
            valid = r_dist > 1e-12
            r_dist = r_dist[valid]
            i_idx = i_idx[valid]
            j_idx = j_idx[valid]
            
            if len(r_dist) > 0:
                w = self.cubic_kernel_2d(r_dist)
                np.add.at(deltas, i_idx, w)
                np.add.at(deltas, j_idx, w)
                
        self.boundary_masses = self.rest_density / deltas

    def initialize_random_state(
        self,
        position_scale: float = 1.0,
        velocity_scale: float = 0.1,
        mass_range: Tuple[float, float] = (1.0, 1.0) # Uniform mass in fluid
    ):
        """Initializes a uniform clustered 'Water Drop' layout near the top of the box."""
        # Calculate optimal uniform spacing to satisfy rest_density mathematically
        # In SPH, fluid particles must spawn at a specific spacing relative to h 
        # to ensure the initial density evaluates to rest_density (not overlapping, not a void).
        spacing = self.smoothing_length / 2.0
        
        # Enforce maximum particle volume safety threshold
        # Evaluate how much area the set of particles will logically occupy
        # based on geometry vs maximum available container space.
        required_area = self.num_particles * (spacing ** 2)
        total_container_area = (2.0 * position_scale) ** 2
        max_allowed_area = 0.90 * total_container_area
        
        if required_area > max_allowed_area:
            raise ValueError(
                f"Fluid Initialization Error: Particles require {required_area:.2f} fluid area based on h={self.smoothing_length}, "
                f"which exceeds the 90% container limit of {max_allowed_area:.2f}. "
                f"Reduce num_particles or decrease softening_length."
            )
        
        # Calculate grid dimensions ensuring N particles fit
        cols = int(np.sqrt(self.num_particles))
        rows = int(np.ceil(self.num_particles / cols))
        
        self.positions = np.zeros((self.num_particles, 2))
        
        # Center the drop horizontally and place vertically near the top
        start_x = - (cols * spacing) / 2.0
        start_y = position_scale * 0.9 - (rows * spacing)
        
        for i in range(self.num_particles):
            row = i // cols
            col = i % cols
            # Add a microscopic jitter (1e-4) to prevent perfectly symmetric grid locking issues
            jitter_x = np.random.uniform(-1e-4, 1e-4)
            jitter_y = np.random.uniform(-1e-4, 1e-4)
            self.positions[i, 0] = start_x + col * spacing + jitter_x
            self.positions[i, 1] = start_y + row * spacing + jitter_y
            
        self.velocities = np.zeros((self.num_particles, 2))

        # mass computation : m = spacing^2 * rest_density
        m0 = (spacing**2) * self.rest_density
        self.masses = np.full(self.num_particles, m0)
        
        # Generate static boundary particles along container walls
        self._generate_boundary_particles()

    def compute_forces(self, positions: np.ndarray, velocities: np.ndarray, masses: np.ndarray, dt: float) -> np.ndarray:
        """Compute SPH forces including boundary particle interactions.
        
        Boundary particles contribute density and pressure to fluid particles
        but are static (zero velocity, fixed density = rest_density, pressure = 0).
        Returns accelerations for fluid particles only.
        """
        N_fluid = self.num_particles
        N_bound = self.num_boundary
        N_total = N_fluid + N_bound
        h = self.smoothing_length
        
        # Concatenate fluid + boundary positions/masses for unified KDTree search
        all_positions = np.concatenate([positions, self.boundary_positions], axis=0)
        all_masses = np.concatenate([masses, self.boundary_masses], axis=0)
        # Boundary velocities are zero
        all_velocities = np.concatenate([velocities, np.zeros((N_bound, 2))], axis=0)
        
        accelerations = np.zeros((N_fluid, 2))  # Only fluid particles get accelerated
        
        # 0. O(N log N) KDTree Neighborhood Search over ALL particles
        tree = cKDTree(all_positions)
        pairs = tree.query_pairs(r=h)
        
        # 1. Density Computation — fluid self-density initialized, boundary fixed
        fluid_densities = np.full(N_fluid, masses[0] * self.kernel_0)
        
        if not pairs:
            accelerations[:, 1] -= self.gravitational_constant
            return accelerations
        
        # Convert pairs to arrays for vectorized computation
        pairs = np.array(list(pairs))
        i_idx = pairs[:, 0]
        j_idx = pairs[:, 1]
        
        # Compute distances for interacting pairs
        r_vec = all_positions[j_idx] - all_positions[i_idx]  # vector from i to j
        r_dist = np.linalg.norm(r_vec, axis=1)
        
        # Filter zero distances
        valid = r_dist > 1e-12
        i_idx = i_idx[valid]
        j_idx = j_idx[valid]
        r_vec = r_vec[valid]
        r_dist = r_dist[valid]
        
        if len(r_dist) == 0:
            accelerations[:, 1] -= self.gravitational_constant
            return accelerations
        
        w = self.cubic_kernel_2d(r_dist)
        
        # Classify pair types using boolean masks
        i_is_fluid = i_idx < N_fluid
        j_is_fluid = j_idx < N_fluid
        
        # --- Density contributions to fluid particles ---
        # Case: both fluid (symmetric)
        ff_mask = i_is_fluid & j_is_fluid
        if np.any(ff_mask):
            fi = i_idx[ff_mask]
            fj = j_idx[ff_mask]
            w_ff = w[ff_mask]
            np.add.at(fluid_densities, fi, all_masses[fj] * w_ff)
            np.add.at(fluid_densities, fj, all_masses[fi] * w_ff)
        
        # Case: fluid-boundary (only fluid gets density contribution)
        fb_mask = i_is_fluid & ~j_is_fluid
        if np.any(fb_mask):
            fi = i_idx[fb_mask]
            bj = j_idx[fb_mask]
            w_fb = w[fb_mask]
            np.add.at(fluid_densities, fi, all_masses[bj] * w_fb)
        
        # Case: boundary-fluid (only fluid j gets density contribution)
        bf_mask = ~i_is_fluid & j_is_fluid
        if np.any(bf_mask):
            fj = j_idx[bf_mask]
            bi = i_idx[bf_mask]
            w_bf = w[bf_mask]
            np.add.at(fluid_densities, fj, all_masses[bi] * w_bf)
        
        # (boundary-boundary pairs: ignored — boundary density is fixed)
        
        # 2. Pressure Computation (Tait's Equation) — fluid only
        fluid_densities = np.maximum(fluid_densities, self.rest_density)
        fluid_pressures = self.stiffness * ((fluid_densities / self.rest_density)**self.exponent - 1.0)
        
        # Build full density/pressure arrays for indexing convenience
        # Boundary: density = rest_density, pressure = 0
        all_densities = np.concatenate([fluid_densities, np.full(N_bound, self.rest_density)])
        all_pressures = np.concatenate([fluid_pressures, np.zeros(N_bound)])
        
        # 3. Pressure & Viscosity Accelerations
        # Viscosity: Monaghan's Artificial Viscosity (Monaghan, J. J. (1992))
        grad_w = self.cubic_kernel_2d_gradient(r_vec, r_dist)
        
        # Compute per-pair pressure term and viscosity for ALL pairs
        p_term = (all_pressures[i_idx] / all_densities[i_idx]**2) + \
                 (all_pressures[j_idx] / all_densities[j_idx]**2)
        
        cs = np.sqrt((self.stiffness * self.exponent) / self.rest_density)
        epsilon = 0.01 * (self.smoothing_length ** 2)


        # --- Fluid-Fluid pairs: symmetric forces ---
        if np.any(ff_mask):
            fi = i_idx[ff_mask]
            fj = j_idx[ff_mask]
            pt = p_term[ff_mask]
            gw = grad_w[ff_mask]
            r_vec_ff = r_vec[ff_mask]
            r_dist_ff = r_dist[ff_mask]
            
            # 1. Pressure
            a_press_ij = all_masses[fj][:, np.newaxis] * pt[:, np.newaxis] * gw
            a_press_ji = all_masses[fi][:, np.newaxis] * pt[:, np.newaxis] * -gw
            
            # 2. Viscosity
            v_rel = all_velocities[fj] - all_velocities[fi]
            v_dot_r = np.sum(v_rel * r_vec_ff, axis=1)
            
            # THE FIX: Clamp positive values to 0.0. No boolean masking needed!
            v_dot_r_clipped = np.minimum(v_dot_r, 0.0)
            
            mu_ij = (self.smoothing_length * v_dot_r_clipped) / (r_dist_ff**2 + epsilon)
            rho_ij = 0.5 * (all_densities[fi] + all_densities[fj])
            
            pi_ij = (-self.viscosity * cs * mu_ij) / rho_ij
            
            a_visc_ij = all_masses[fj][:, np.newaxis] * pi_ij[:, np.newaxis] * gw
            a_visc_ji = all_masses[fi][:, np.newaxis] * pi_ij[:, np.newaxis] * -gw
            
            np.add.at(accelerations, fi, a_press_ij + a_visc_ij)
            np.add.at(accelerations, fj, a_press_ji + a_visc_ji)
        
        # --- Fluid-Boundary pairs: force on fluid i from boundary j ---
        if np.any(fb_mask):
            fi = i_idx[fb_mask]
            bj = j_idx[fb_mask]
            r_vec_fb = r_vec[fb_mask]
            r_dist_fb = r_dist[fb_mask]
            
            # Pressure
            pt_fb = (all_pressures[fi] / all_densities[fi]**2)
            gw = grad_w[fb_mask]
            a_press = all_masses[bj][:, np.newaxis] * pt_fb[:, np.newaxis] * gw
            
            # Viscosity
            boundary_friction = 1.0 
            v_rel = all_velocities[bj] - all_velocities[fi]
            v_dot_r = np.sum(v_rel * r_vec_fb, axis=1)
            
            v_dot_r_clipped = np.minimum(v_dot_r, 0.0)
            
            mu_ij = (self.smoothing_length * v_dot_r_clipped) / (r_dist_fb**2 + epsilon)
            rho_ij = 0.5 * (all_densities[fi] + all_densities[bj])
            
            pi_ij = (-(self.viscosity * boundary_friction) * cs * mu_ij) / rho_ij
            a_visc = all_masses[bj][:, np.newaxis] * pi_ij[:, np.newaxis] * gw
                
            np.add.at(accelerations, fi, a_press + a_visc)
        
        # --- Boundary-Fluid pairs: force on fluid j from boundary i ---
        if np.any(bf_mask):
            fj = j_idx[bf_mask]
            bi = i_idx[bf_mask]
            r_vec_bf = r_vec[bf_mask]
            r_dist_bf = r_dist[bf_mask]
            
            # Pressure
            pt_bf = (all_pressures[fj] / all_densities[fj]**2)
            gw = grad_w[bf_mask]
            a_press = all_masses[bi][:, np.newaxis] * pt_bf[:, np.newaxis] * -gw
            
            # Viscosity
            boundary_friction = 1.0 
            v_rel = all_velocities[fj] - all_velocities[bi]
            v_dot_r = np.sum(v_rel * r_vec_bf, axis=1)
            
            v_dot_r_clipped = np.minimum(v_dot_r, 0.0)
            
            mu_ij = (self.smoothing_length * v_dot_r_clipped) / (r_dist_bf**2 + epsilon)
            rho_ij = 0.5 * (all_densities[bi] + all_densities[fj])
            
            pi_ij = (-(self.viscosity * boundary_friction) * cs * mu_ij) / rho_ij
            a_visc = all_masses[bi][:, np.newaxis] * pi_ij[:, np.newaxis] * -gw
                
            np.add.at(accelerations, fj, a_press + a_visc)
        
        # 4. External Forces (Gravity) — fluid only
        accelerations[:, 1] -= self.gravitational_constant
        
        return accelerations

    def apply_boundaries(self, positions: np.ndarray, velocities: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Safety-net hard clamp for particles that somehow escape past boundary particles.
        
        With Akinci boundary particles providing pressure repulsion, this should
        rarely activate. Keeps particles strictly inside the container.
        """
        limit = self.position_scale
        spacing = self.smoothing_length / 2.0
        safe_limit = limit - 0.5 * spacing
        
        for axis in range(2):
            mask_low = positions[:, axis] < -safe_limit
            mask_high = positions[:, axis] > safe_limit
            
            # Hard clamp position, soft bounce velocity component
            positions[mask_low, axis] = -safe_limit
            velocities[mask_low, axis] *= -0.5
            
            positions[mask_high, axis] = safe_limit
            velocities[mask_high, axis] *= -0.5
            
        return positions, velocities

    def simulate(self, total_time: float, dt: float, save_every: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Formally validate Courant-Friedrichs-Lewy (CFL) stability for acoustic waves
        c_s = np.sqrt((self.stiffness * self.exponent) / self.rest_density)
        max_dt = 0.4 * (self.smoothing_length / c_s)
        
        if dt > max_dt:
            print(f"WARNING: Provided dt={dt} violates the CFL condition (max_dt={max_dt:.5f} for h={self.smoothing_length}, cs={c_s:.2f}).")
            print("The simulation may be numerically unstable and explode. Consider drastically lowering dt or stiffness.")
            
        total_steps = int(total_time / dt)
        num_saved = (total_steps // save_every) + 1
        dim = self.positions.shape[1]
        history_positions = np.zeros((num_saved, self.num_particles, dim))
        history_velocities = np.zeros((num_saved, self.num_particles, dim))
        history_times = np.zeros(num_saved)
        
        pos = self.positions.copy()
        vel = self.velocities.copy()
        
        # Closure capturing current state dependencies and dt integration delta
        def force_fn(positions_array, dt_step):
            return self.compute_forces(positions_array, vel, self.masses, dt_step)
            
        save_idx = 0
        from tqdm.auto import tqdm
        for step in tqdm(range(total_steps + 1), desc="Fluid Simulation Progress", leave=False):
            if step % save_every == 0:
                history_positions[save_idx] = pos
                history_velocities[save_idx] = vel
                history_times[save_idx] = step * dt
                save_idx += 1
                
            if step < total_steps:
                pos, vel = self.integrator.step((pos, vel), dt, force_fn)
                pos, vel = self.apply_boundaries(pos, vel)
                
        self.positions = pos
        self.velocities = vel
        
        return history_positions[:save_idx], history_velocities[:save_idx], history_times[:save_idx]

    def compute_total_energy(self) -> float:
        """
        Compute total energy (kinetic + potential gravity).
        Internal pressure energy is omitted as WCSPH equations of state are highly non-conservative.
        
        Returns:
            Total tracked energy of the system
        """
        if not hasattr(self, 'positions') or not hasattr(self, 'velocities') or not hasattr(self, 'masses'):
            raise ValueError("Simulation state not initialized. Call initialize_random_state() first.")
            
        # Kinetic Energy: K = 0.5 * sum(m * v^2)
        v_sq = np.sum(self.velocities**2, axis=-1)
        kinetic_energy = 0.5 * np.sum(self.masses * v_sq)
        
        # Potential Energy (Gravity): U = m * g * y
        potential_energy_gravity = np.sum(self.masses * self.gravitational_constant * self.positions[:, 1])
            
        return float(kinetic_energy + potential_energy_gravity)


"""
Dataset generation utilities.

This module provides functions for generating and saving
simulation datasets for training the GNN.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any


def generate_dataset(
    num_trajectories: int,
    num_particles: int,
    total_time: float,
    dt: float,
    integrator: str = 'rk4',
    save_path: str = None,
    **simulation_kwargs
) -> Dict[str, Any]:
    """
    Generate a dataset of N-body trajectories matching DeepMind's sequence structure.
    
    Args:
        num_trajectories: Number of trajectories to generate
        num_particles: Number of particles per trajectory
        total_time: Total simulation time per trajectory
        dt: Time step
        integrator: Integration method
        save_path: Optional path to save dataset
        **simulation_kwargs: Additional simulation parameters
        
    Returns:
        Dictionary containing trajectories with positions, types, masses, etc.
    """
    from tqdm.auto import tqdm
    
    simulation_type = simulation_kwargs.get('type', 'n_body')
    gravitational_constant = simulation_kwargs.get('gravitational_constant', 1.0)
    softening_length = simulation_kwargs.get('softening_length', 0.01)
    connectivity_radius = simulation_kwargs.get('connectivity_radius', 0.015)
    
    position_scale = simulation_kwargs.get('position_scale', 1.0)
    velocity_scale = simulation_kwargs.get('velocity_scale', 0.1)
    mass_range = simulation_kwargs.get('mass_range', (0.5, 1.5))
    save_every = simulation_kwargs.get('save_every', 10)
    
    trajectories = []
    
    for _ in tqdm(range(num_trajectories), desc=f"Generating {simulation_type} Trajectories"):
        if simulation_type == 'fluid':
            sim = FluidSimulation(
                num_particles=num_particles,
                gravitational_constant=gravitational_constant,
                softening_length=softening_length,
                integrator=integrator,
                position_scale=position_scale,
                rest_density=simulation_kwargs.get('rest_density', 1000.0),
                stiffness=simulation_kwargs.get('stiffness', 35000.0),
                viscosity=simulation_kwargs.get('viscosity', 0.05)
            )
        else:
            sim = NBodySimulation(
                num_particles=num_particles,
                gravitational_constant=gravitational_constant,
                softening_length=softening_length,
                integrator=integrator
            )
            
        sim.initialize_random_state(
            position_scale=position_scale,
            velocity_scale=velocity_scale,
            mass_range=mass_range
        )
        
        history_positions, history_velocities, _ = sim.simulate(
            total_time=total_time,
            dt=dt,
            save_every=save_every
        )
        
        # Save trajectory in a format mimicking DeepMind's TFRecords
        traj_data = {
            'particle_type': np.zeros(num_particles, dtype=np.int64),  # Type 0 for all gravitational particles
            'position': history_positions.astype(np.float32),
            'velocity': history_velocities.astype(np.float32),
            'mass': sim.masses.astype(np.float32)
        }
        trajectories.append(traj_data)
        
    dataset = {'trajectories': trajectories}
    
    # Determine correct bounds based on simulation type
    if simulation_type == 'fluid':
        bounds = [[-position_scale, position_scale] for _ in range(2)]
    else:
        bounds = None  # N-Body has no walls
    
    if save_path:
        save_dataset(dataset, save_path, dt * save_every,
                     bounds=bounds, connectivity_radius=connectivity_radius)
        
    return dataset


def save_dataset(dataset: Dict[str, Any], save_path: str, save_dt: float,
                 bounds: list = None, connectivity_radius: float = 0.015):
    """
    Save dataset to disk in NPZ format along with metadata.json.
    
    Args:
        dataset: Dataset dictionary containing list of trajectories
        save_path: Path to save directory
        save_dt: The time delta between saved frames (dt * save_every)
        bounds: Simulation domain bounds per axis, or None for unbounded sims
        connectivity_radius: Graph connectivity radius for the model
    """
    os.makedirs(save_path, exist_ok=True)
    
    all_vels = []
    all_accs = []
    
    trajectories = dataset['trajectories']
    dim = trajectories[0]['position'].shape[-1]
    
    # Compute global kinematic stats using the actual stored velocities
    for traj in trajectories:
        vel = traj['velocity']  # Use real integrator velocities, not position deltas
        acc = vel[1:] - vel[:-1]
        
        all_vels.append(vel.reshape(-1, dim))
        all_accs.append(acc.reshape(-1, dim))
        
    all_vels = np.concatenate(all_vels, axis=0)
    all_accs = np.concatenate(all_accs, axis=0)
    
    metadata = {
        "bounds": bounds,  # None for N-Body (no walls), [[lo, hi], ...] for Fluid
        "sequence_length": trajectories[0]['position'].shape[0],
        "default_connectivity_radius": connectivity_radius,
        "dim": dim,
        "dt": save_dt,
        "vel_mean": all_vels.mean(axis=0).tolist(),
        "vel_std": all_vels.std(axis=0).tolist(),
        "acc_mean": all_accs.mean(axis=0).tolist(),
        "acc_std": all_accs.std(axis=0).tolist()
    }
    
    with open(os.path.join(save_path, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
        
    # Split trajectories
    num_traj = len(trajectories)
    train_idx = int(0.8 * num_traj)
    valid_idx = int(0.9 * num_traj)
    
    splits = {
        'train': trajectories[:train_idx],
        'valid': trajectories[train_idx:valid_idx],
        'test': trajectories[valid_idx:]
    }
    
    for split_name, split_trajs in splits.items():
        if len(split_trajs) > 0:
            # We compress them as a sequence of dicts to a single .npz
            save_dict = {f"trajectory_{i}": traj for i, traj in enumerate(split_trajs)}
            np.savez_compressed(os.path.join(save_path, f"{split_name}.npz"), **save_dict)


def load_dataset(load_path: str) -> Dict[str, Any]:
    """
    Load dataset from disk.
    
    Args:
        load_path: Path to dataset directory
        
    Returns:
        Dataset dictionary with splits and metadata
    """
    dataset = {}
    with open(os.path.join(load_path, "metadata.json"), "r") as f:
        dataset['metadata'] = json.load(f)
        
    dataset['splits'] = {}
    for split in ['train', 'valid', 'test']:
        split_path = os.path.join(load_path, f"{split}.npz")
        if os.path.exists(split_path):
            data = np.load(split_path, allow_pickle=True)
            dataset['splits'][split] = [data[k].item() for k in data.files]
            
    return dataset


