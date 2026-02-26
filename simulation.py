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
import numba
from numba import njit


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

@njit
def cubic_kernel_2d_jit(r_dist, smoothing_length, kernel_k):
    """JIT-accelerated 2D Cubic Spline Kernel evaluation"""
    q = r_dist / smoothing_length
    if q <= 0.5:
        return kernel_k * (6.0 * q**3 - 6.0 * q**2 + 1.0)
    elif q <= 1.0:
        return kernel_k * (2.0 * (1.0 - q)**3)
    else:
        return 0.0

@njit
def cubic_kernel_2d_gradient_jit(r_x, r_y, r_dist, smoothing_length, kernel_l):
    """JIT-accelerated 2D Cubic Spline Kernel Gradient evaluation"""
    q = r_dist / smoothing_length
    if q > 1.0 or r_dist <= 1e-12:
        return 0.0, 0.0
        
    if q <= 0.5:
        factor = kernel_l * q * (3.0 * q - 2.0)
    else:
        factor = -kernel_l * (1.0 - q)**2
        
    gradq_x = r_x / (r_dist * smoothing_length)
    gradq_y = r_y / (r_dist * smoothing_length)
    return factor * gradq_x, factor * gradq_y

@njit
def compute_densities_jit(
    pairs, positions, masses, densities,
    smoothing_length, kernel_k, kernel_0, N_fluid
):
    """JIT-accelerated density accumulation using neighbor pairs."""
    # Densities should be pre-initialized with (mass * kernel_0) for self-contribution
    for idx in range(len(pairs)):
        i = pairs[idx, 0]
        j = pairs[idx, 1]
        
        dx = positions[j, 0] - positions[i, 0]
        dy = positions[j, 1] - positions[i, 1]
        r_dist = np.sqrt(dx*dx + dy*dy)
        
        if r_dist < smoothing_length:
            w = cubic_kernel_2d_jit(r_dist, smoothing_length, kernel_k)
            # Both particles get contribution if fluid
            if i < N_fluid:
                densities[i] += masses[j] * w
            if j < N_fluid:
                densities[j] += masses[i] * w

@njit
def compute_forces_jit(
    pairs, positions, velocities, masses, densities, pressures,
    smoothing_length, kernel_l, viscosity, stiffness, rest_density, exponent,
    gravitational_constant, N_fluid
):
    """JIT-accelerated pressure and viscosity force computation."""
    accelerations = np.zeros((N_fluid, 2))
    cs = np.sqrt((stiffness * exponent) / rest_density)
    epsilon = 0.01 * (smoothing_length**2)
    boundary_friction = 1.0
    
    for idx in range(len(pairs)):
        i = pairs[idx, 0]
        j = pairs[idx, 1]
        
        # At least one must be fluid to receive force
        if i >= N_fluid and j >= N_fluid:
            continue
            
        dx = positions[j, 0] - positions[i, 0]
        dy = positions[j, 1] - positions[i, 1]
        r_dist = np.sqrt(dx*dx + dy*dy)
        
        if r_dist <= 1e-12 or r_dist > smoothing_length:
            continue
            
        gx, gy = cubic_kernel_2d_gradient_jit(dx, dy, r_dist, smoothing_length, kernel_l)
        
        # Pressure terms
        p_i = pressures[i] / (densities[i]**2)
        p_j = pressures[j] / (densities[j]**2)
        p_term = p_i + p_j
        
        # Viscosity (Artificial Viscosity)
        vx = velocities[j, 0] - velocities[i, 0]
        vy = velocities[j, 1] - velocities[i, 1]
        v_dot_r = vx * dx + vy * dy
        
        v_dot_r_clipped = v_dot_r if v_dot_r < 0.0 else 0.0
        mu_ij = (smoothing_length * v_dot_r_clipped) / (r_dist*r_dist + epsilon)
        rho_ij = 0.5 * (densities[i] + densities[j])
        
        # Correct classification for friction
        fric = boundary_friction if (i >= N_fluid or j >= N_fluid) else 1.0
        pi_ij = (-viscosity * fric * cs * mu_ij) / rho_ij
        
        # Combined force term
        f_x = (p_term + pi_ij) * gx
        f_y = (p_term + pi_ij) * gy
        
        # Apply forces (symmetric if both fluid, one-way if one is boundary)
        if i < N_fluid:
            accelerations[i, 0] += masses[j] * f_x
            accelerations[i, 1] += masses[j] * f_y
            
        if j < N_fluid:
            accelerations[j, 0] -= masses[i] * f_x
            accelerations[j, 1] -= masses[i] * f_y
            
    # Add gravity
    for i in range(N_fluid):
        accelerations[i, 1] -= gravitational_constant
        
    return accelerations

    return accelerations

@njit
def _pos_to_cell(pos, min_pos, inv_cell_size, grid_dim):
    """Map position to a 1D cell index."""
    ix = int((pos[0] - min_pos) * inv_cell_size)
    iy = int((pos[1] - min_pos) * inv_cell_size)
    # Clip to grid boundaries
    ix = max(0, min(ix, grid_dim - 1))
    iy = max(0, min(iy, grid_dim - 1))
    return ix * grid_dim + iy

@njit
def build_grid_jit(positions, min_pos, inv_cell_size, grid_dim, cell_starts, cell_ends, sorted_indices):
    """Build the uniform grid by counting particles per cell and sorting indices."""
    num_particles = positions.shape[0]
    num_cells = grid_dim * grid_dim
    
    # Reset grid arrays
    cell_starts.fill(-1)
    cell_ends.fill(-1)
    
    # Count particles per cell
    cell_counts = np.zeros(num_cells, dtype=np.int32)
    particle_cells = np.zeros(num_particles, dtype=np.int32)
    
    for i in range(num_particles):
        c = _pos_to_cell(positions[i], min_pos, inv_cell_size, grid_dim)
        particle_cells[i] = c
        cell_counts[c] += 1
        
    # Compute cumulative offsets (cell_starts)
    current_offset = 0
    for c in range(num_cells):
        if cell_counts[c] > 0:
            cell_starts[c] = current_offset
            cell_ends[c] = current_offset + cell_counts[c]
            current_offset += cell_counts[c]
            
    # Fill sorted_indices
    # Reuse cell_counts to track current insertion point for each cell
    insertion_offsets = cell_starts.copy()
    for i in range(num_particles):
        c = particle_cells[i]
        pos_in_sorted = insertion_offsets[c]
        sorted_indices[pos_in_sorted] = i
        insertion_offsets[c] += 1

@njit
def find_pairs_ugs_jit(positions, smoothing_length, min_pos, inv_cell_size, grid_dim, cell_starts, cell_ends, sorted_indices):
    """Find neighbor pairs using the uniform grid."""
    num_particles = positions.shape[0]
    # Max pairs estimate - in 2D SPH, typically ~30-50 neighbors per particle
    # We allocate a reasonably large buffer and return a slice.
    max_pairs = num_particles * 64 
    pairs = np.zeros((max_pairs, 2), dtype=np.int64)
    pair_count = 0
    
    h_sq = smoothing_length * smoothing_length
    
    for cx in range(grid_dim):
        for cy in range(grid_dim):
            c_idx = cx * grid_dim + cy
            if cell_starts[c_idx] == -1:
                continue
                
            # Check cell and 8 neighbors (only looking forward/current to avoid duplicates)
            # Actually, standard query_pairs style: check all neighbors for i < j or similar.
            # For simplicity and to match cKDTree.query_pairs exactly:
            # We'll check the cell itself and all 8 neighbors, keeping i < j.
            
            for dx in range(-1, 2):
                nx = cx + dx
                if nx < 0 or nx >= grid_dim: continue
                for dy in range(-1, 2):
                    ny = cy + dy
                    if ny < 0 or ny >= grid_dim: continue
                    
                    n_idx = nx * grid_dim + ny
                    if cell_starts[n_idx] == -1:
                        continue
                        
                    # Particle i in cell c_idx, particle j in cell n_idx
                    for p_i_idx in range(cell_starts[c_idx], cell_ends[c_idx]):
                        i = sorted_indices[p_i_idx]
                        
                        for p_j_idx in range(cell_starts[n_idx], cell_ends[n_idx]):
                            j = sorted_indices[p_j_idx]
                            
                            # Avoid double counting and self-interaction
                            if i < j:
                                dist_sq = (positions[i, 0] - positions[j, 0])**2 + \
                                          (positions[i, 1] - positions[j, 1])**2
                                if dist_sq < h_sq:
                                    if pair_count < max_pairs:
                                        pairs[pair_count, 0] = i
                                        pairs[pair_count, 1] = j
                                        pair_count += 1
                                        
    return pairs[:pair_count]

@njit
def _accumulate_boundary_densities_jit(pairs, positions, deltas, h, k):
    """JIT helper for boundary particle constant density."""
    for idx in range(len(pairs)):
        i = pairs[idx, 0]
        j = pairs[idx, 1]
        dx = positions[j, 0] - positions[i, 0]
        dy = positions[j, 1] - positions[i, 1]
        r = np.sqrt(dx*dx + dy*dy)
        if r < h:
            w = cubic_kernel_2d_jit(r, h, k)
            deltas[i] += w
            deltas[j] += w

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

        # Pre-allocate unified memory pools (Fluid + Boundary)
        self.num_boundary = 0 # Will be updated in _generate_boundary_particles
        self.all_positions = np.zeros((0, 2))
        self.all_velocities = np.zeros((0, 2))
        self.all_masses = np.zeros(0)
        self.all_densities = np.zeros(0)
        self.all_pressures = np.zeros(0)
        
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
        N_total = self.num_particles + self.num_boundary
        
        # Grid parameters for UGS
        h = self.smoothing_length
        limit = self.position_scale
        self.grid_min = -limit - (num_layers + 1) * spacing # buffer for boundaries
        self.grid_inv_size = 1.0 / h
        # Total span / h
        span = 2 * limit + 2 * (num_layers + 1) * spacing
        self.grid_dim = int(np.ceil(span * self.grid_inv_size)) + 1
        
        # Initialize/Resize Pre-allocated memory pools
        self.all_positions = np.zeros((N_total, 2))
        self.all_velocities = np.zeros((N_total, 2))
        self.all_masses = np.zeros(N_total)
        self.all_densities = np.zeros(N_total)
        self.all_pressures = np.zeros(N_total)
        
        # Buffers for UGS
        num_cells = self.grid_dim * self.grid_dim
        self.cell_starts = np.full(num_cells, -1, dtype=np.int32)
        self.cell_ends = np.full(num_cells, -1, dtype=np.int32)
        self.sorted_indices = np.zeros(N_total, dtype=np.int32)
        
        # Fill Boundary Slices (Static)
        self.all_positions[self.num_particles:] = self.boundary_positions
        # Boundary velocities remain zero by default
        
        # Compute pseudo-mass (psi) for boundary particles (Akinci et al. 2012)
        # psi_i = rest_density / sum_j W(r_ij)
        
        # Build grid for boundary particles ONLY to find boundary-boundary pairs
        # (This is a one-time setup)
        temp_starts = np.full(num_cells, -1, dtype=np.int32)
        temp_ends = np.full(num_cells, -1, dtype=np.int32)
        temp_indices = np.zeros(self.num_boundary, dtype=np.int32)
        
        build_grid_jit(
            self.boundary_positions, self.grid_min, self.grid_inv_size, self.grid_dim,
            temp_starts, temp_ends, temp_indices
        )
        
        pairs = find_pairs_ugs_jit(
            self.boundary_positions, self.smoothing_length, self.grid_min, self.grid_inv_size, self.grid_dim,
            temp_starts, temp_ends, temp_indices
        )
        
        deltas = np.full(self.num_boundary, self.kernel_0)
        if len(pairs) > 0:
            # Use a specialized JIT helper for boundary density to avoid np.add.at
            _accumulate_boundary_densities_jit(
                pairs, self.boundary_positions, deltas, 
                self.smoothing_length, self.kernel_k
            )
                
        self.boundary_masses = self.rest_density / deltas
        # Fill Boundary Mass Slice
        self.all_masses[self.num_particles:] = self.boundary_masses
        # Pre-initialize boundary densities as fixed
        self.all_densities[self.num_particles:] = self.rest_density

    def initialize_random_state(
        self,
        position_scale: Optional[float] = None,
        velocity_scale: float = 0.1,
        mass_range: Tuple[float, float] = (1.0, 1.0), # Uniform mass in fluid
        start: Tuple[float,float] = (0.0, 0.5) # Starting position of the drop center
    ):
        """Initializes a uniform clustered 'Water Drop' layout near the top of the box."""
        if position_scale is not None:
            self.position_scale = position_scale
        
        # Use local variable for convenience in formulas
        pos_scale = self.position_scale

        # Calculate optimal uniform spacing to satisfy rest_density mathematically
        spacing = self.smoothing_length / 2.0
        
        required_area = self.num_particles * (spacing ** 2)
        total_container_area = (2.0 * pos_scale) ** 2
        max_allowed_area = 0.90 * total_container_area
        
        if required_area > max_allowed_area:
            raise ValueError(
                f"Fluid Initialization Error: Particles require {required_area:.2f} fluid area based on h={self.smoothing_length}, "
                f"which exceeds the 90% container limit of {max_allowed_area:.2f}. "
                f"Reduce num_particles or decrease softening_length."
            )
        
        cols = int(np.sqrt(self.num_particles))
        rows = int(np.ceil(self.num_particles / cols))
        
        # Ensure boundary particles exist before setting fluid views
        if self.num_boundary == 0:
            self._generate_boundary_particles()

        # Set up views into memory pools for the fluid slice
        self.positions = self.all_positions[:self.num_particles]
        self.velocities = self.all_velocities[:self.num_particles]
        self.masses = self.all_masses[:self.num_particles]
        
        # Calculate grid boundaries
        grid_width = (cols - 1) * spacing
        grid_height = (rows - 1) * spacing

        start_x = start[0] - grid_width / 2.0
        start_y = start[1] - grid_height / 2.0

        # Clip to ensure the entire grid fits inside [-pos_scale, pos_scale]
        start_x = np.clip(start_x, -pos_scale, pos_scale - grid_width)
        start_y = np.clip(start_y, -pos_scale, pos_scale - grid_height)
        
        # Fill positions
        for i in range(self.num_particles):
            row = i // cols
            col = i % cols
            jitter_x = np.random.uniform(-1e-4, 1e-4)
            jitter_y = np.random.uniform(-1e-4, 1e-4)
            self.positions[i, 0] = start_x + col * spacing + jitter_x
            self.positions[i, 1] = start_y + row * spacing + jitter_y
            
        self.velocities.fill(0.0)

        # mass computation : m = spacing^2 * rest_density
        m0 = (spacing**2) * self.rest_density
        self.all_masses[:self.num_particles] = m0

    def compute_forces(self, positions: np.ndarray, velocities: np.ndarray, masses: np.ndarray, dt: float) -> np.ndarray:
        """High-performance SPH force computation using Numba JIT and Uniform Grid Search."""
        # 1. Update fluid slices in unified arrays
        self.all_positions[:self.num_particles] = positions
        self.all_velocities[:self.num_particles] = velocities
        
        # 2. Neighborhood Search (Uniform Grid Search)
        build_grid_jit(
            self.all_positions, self.grid_min, self.grid_inv_size, self.grid_dim,
            self.cell_starts, self.cell_ends, self.sorted_indices
        )
        
        pairs = find_pairs_ugs_jit(
            self.all_positions, self.smoothing_length, self.grid_min, self.grid_inv_size, self.grid_dim,
            self.cell_starts, self.cell_ends, self.sorted_indices
        )
        
        if len(pairs) == 0:
            accel = np.zeros((self.num_particles, 2))
            accel[:, 1] -= self.gravitational_constant
            return accel
            
        # 3. Density Accumulation (JIT)
        self.all_densities[:self.num_particles] = self.all_masses[:self.num_particles] * self.kernel_0
        
        compute_densities_jit(
            pairs, self.all_positions, self.all_masses, self.all_densities,
            self.smoothing_length, self.kernel_k, self.kernel_0, self.num_particles
        )
        
        # 4. Pressure Update (Tait's Equation)
        # Clamping and pressure calculation can be vectorized or JIT-ed.
        # Boundary pressures are 0 by default.
        self.all_densities[:self.num_particles] = np.maximum(self.all_densities[:self.num_particles], self.rest_density)
        self.all_pressures[:self.num_particles] = self.stiffness * (
            (self.all_densities[:self.num_particles] / self.rest_density)**self.exponent - 1.0
        )
        
        # 5. Force Accumulation (JIT)
        accelerations = compute_forces_jit(
            pairs, self.all_positions, self.all_velocities, self.all_masses,
            self.all_densities, self.all_pressures,
            self.smoothing_length, self.kernel_l, self.viscosity, self.stiffness,
            self.rest_density, self.exponent, self.gravitational_constant, self.num_particles
        )
        
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


