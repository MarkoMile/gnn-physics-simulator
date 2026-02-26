"""
Numerical integrators for N-body simulation.

This module implements classical numerical integration methods
for solving ordinary differential equations in physics simulations.
"""

import numpy as np
from abc import ABC, abstractmethod
from numba import njit


class Integrator(ABC):
    """
    Abstract base class for numerical integrators.
    """
    
    @abstractmethod
    def step(self, state, dt, force_fn):
        """
        Perform one integration step.
        
        Args:
            state: Current state (positions, velocities)
            dt: Time step
            force_fn: Function to compute forces/accelerations
            
        Returns:
            Updated state after time step
        """
        return state


@njit
def euler_step_jit(positions, velocities, accelerations, dt):
    """JIT-optimized Euler step logic."""
    new_positions = positions + velocities * dt
    new_velocities = velocities + accelerations * dt
    return new_positions, new_velocities

class EulerIntegrator(Integrator):
    """
    Forward Euler integrator.
    
    First-order explicit method:
    x(t+dt) = x(t) + v(t) * dt
    v(t+dt) = v(t) + a(t) * dt
    """
    
    def step(self, state, dt, force_fn):
        """Perform one Euler integration step."""
        positions, velocities = state
        accelerations = force_fn(positions, dt)
        return euler_step_jit(positions, velocities, accelerations, dt)


@njit
def rk4_step_jit(positions, velocities, k1_x, k2_x, k3_x, k4_x, k1_v, k2_v, k3_v, k4_v, dt):
    """JIT-optimized RK4 final state combination."""
    new_positions = positions + (k1_x + 2 * k2_x + 2 * k3_x + k4_x) * dt / 6
    new_velocities = velocities + (k1_v + 2 * k2_v + 2 * k3_v + k4_v) * dt / 6
    return new_positions, new_velocities

class RK4Integrator(Integrator):
    """
    4th-order Runge-Kutta integrator.
    
    Classical RK4 method with higher accuracy than Euler.
    """
    
    def step(self, state, dt, force_fn):
        """Perform one RK4 integration step."""
        positions, velocities = state

        # k1 = f(y_n)
        k1_v = force_fn(positions, dt)
        k1_x = velocities

        # k2 = f(y_n + dt/2 * k1)
        k2_v = force_fn(positions + k1_x * dt / 2, dt)
        k2_x = velocities + k1_v * dt / 2

        # k3 = f(y_n + dt/2 * k2)
        k3_v = force_fn(positions + k2_x * dt / 2, dt)
        k3_x = velocities + k2_v * dt / 2

        # k4 = f(y_n + dt * k3)
        k4_v = force_fn(positions + k3_x * dt, dt)
        k4_x = velocities + k3_v * dt

        return rk4_step_jit(positions, velocities, k1_x, k2_x, k3_x, k4_x, k1_v, k2_v, k3_v, k4_v, dt)

class LeapfrogIntegrator(Integrator):
    """
    Leapfrog (Störmer-Verlet) integrator.
    
    Symplectic integrator that conserves energy well
    for Hamiltonian systems.
    """
    
    def step(self, state, dt, force_fn):
        """Perform one Leapfrog integration step."""
        # TODO: Implement Leapfrog integration
        raise NotImplementedError("LeapfrogIntegrator not implemented yet")


class VerletIntegrator(Integrator):
    """
    Velocity Verlet integrator.
    
    Symplectic integrator commonly used in molecular dynamics.
    """
    
    def step(self, state, dt, force_fn):
        """Perform one Verlet integration step."""
        # TODO: Implement Verlet integration
        raise NotImplementedError("VerletIntegrator not implemented yet")


@njit
def symplectic_euler_step_jit(positions, velocities, accelerations, dt):
    """JIT-optimized Symplectic Euler step logic."""
    # 1. Update velocities FIRST
    new_velocities = velocities + accelerations * dt
    # 2. Update positions using NEW velocities
    new_positions = positions + new_velocities * dt
    return new_positions, new_velocities

class SymplecticEulerIntegrator(Integrator):
    """
    Symplectic Euler (Semi-Implicit) integrator.
    
    First-order symplectic method mathematically required for SPH stability 
    so that particle pressure dependencies compute with updated velocities:
    v(t+dt) = v(t) + a(t) * dt
    x(t+dt) = x(t) + v(t+dt) * dt
    """
    
    def step(self, state, dt, force_fn):
        """Perform one Symplectic Euler integration step."""
        positions, velocities = state
        
        # 1. Evaluate forces using CURRENT positions
        accelerations = force_fn(positions, dt)
        
        return symplectic_euler_step_jit(positions, velocities, accelerations, dt)


def get_integrator(name: str) -> Integrator:
    """
    Factory function to get integrator by name.
    
    Args:
        name: Integrator name ('euler', 'rk4', 'leapfrog', 'verlet', 'symplectic_euler')
        
    Returns:
        Integrator instance
    """
    integrators = {
        'euler': EulerIntegrator,
        'rk4': RK4Integrator,
        'leapfrog': LeapfrogIntegrator,
        'verlet': VerletIntegrator,
        'symplectic_euler': SymplecticEulerIntegrator,
    }
    
    if name.lower() not in integrators:
        raise ValueError(f"Unknown integrator: {name}")
    
    return integrators[name.lower()]()

