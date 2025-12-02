#!/usr/bin/env python
"""
Honeycomb lattice Heisenberg model with phonon coupling.

This example demonstrates iPEPS simulation of the antiferromagnetic
Heisenberg model on a honeycomb lattice coupled to local phonon modes
(Holstein model).

The Hamiltonian is:
    H = J Σ_{<ij>} S_i · S_j + ω Σ_i b†_i b_i + g Σ_i S^z_i (b†_i + b_i)

where:
    - J: Heisenberg exchange coupling (J > 0 for AFM)
    - ω: phonon frequency  
    - g: spin-phonon coupling strength
    - b†, b: phonon creation/annihilation operators

Usage:
    python honeycomb_spin_phonon.py [--D BOND_DIM] [--chi ENV_BOND] 
                                    [--J EXCHANGE] [--omega PHONON_FREQ] 
                                    [--g COUPLING] [--steps N_STEPS]
"""

import numpy as np
import argparse
from pathlib import Path
import time

# Add parent to path for development
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ipeps.core.ipeps_state import iPEPSState
from ipeps.lattice.honeycomb import HoneycombLattice
from ipeps.models.spin_boson import SpinBosonHamiltonian
from ipeps.algorithms.simple_update import SimpleUpdate
from ipeps.algorithms.ctmrg import CTMRG
from ipeps.hpc.checkpointing import CheckpointManager, AutoCheckpointer
from ipeps.hpc.profiling import Profiler


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Honeycomb lattice spin-phonon iPEPS simulation"
    )
    
    # iPEPS parameters
    parser.add_argument(
        "--D", type=int, default=3,
        help="iPEPS bond dimension (default: 3)"
    )
    parser.add_argument(
        "--chi", type=int, default=20,
        help="Environment bond dimension for CTMRG (default: 20)"
    )
    
    # Model parameters
    parser.add_argument(
        "--J", type=float, default=1.0,
        help="Heisenberg exchange coupling (default: 1.0)"
    )
    parser.add_argument(
        "--omega", type=float, default=1.0,
        help="Phonon frequency (default: 1.0)"
    )
    parser.add_argument(
        "--g", type=float, default=0.5,
        help="Spin-phonon coupling (default: 0.5)"
    )
    parser.add_argument(
        "--n_bosons", type=int, default=3,
        help="Maximum boson number (Fock space truncation, default: 3)"
    )
    
    # Simulation parameters
    parser.add_argument(
        "--steps", type=int, default=1000,
        help="Number of imaginary time steps (default: 1000)"
    )
    parser.add_argument(
        "--dt", type=float, default=0.01,
        help="Imaginary time step (default: 0.01)"
    )
    parser.add_argument(
        "--tol", type=float, default=1e-8,
        help="Convergence tolerance (default: 1e-8)"
    )
    
    # I/O parameters
    parser.add_argument(
        "--checkpoint_dir", type=str, default="./checkpoints",
        help="Directory for checkpoints (default: ./checkpoints)"
    )
    parser.add_argument(
        "--output", type=str, default="results.npz",
        help="Output file for results (default: results.npz)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from checkpoint if available"
    )
    
    return parser.parse_args()


def create_hamiltonian(J: float, omega: float, g: float, n_bosons: int):
    """Create the spin-phonon Hamiltonian."""
    return SpinBosonHamiltonian(
        spin=0.5,
        n_bosons=n_bosons,
        model_type="holstein",
        J=J,
        omega=omega,
        g=g,
    )


def compute_observables(state, env, hamiltonian):
    """Compute physical observables."""
    observables = {}
    
    # Get reduced density matrices
    rdms = env.compute_rdms(state)
    
    # Local observables
    spin_ops = hamiltonian.spin_operators
    boson_ops = hamiltonian.boson_operators
    
    # <Sz> on each sublattice
    Sz = spin_ops.Sz
    Id_boson = np.eye(hamiltonian.n_bosons)
    Sz_full = np.kron(Sz, Id_boson)
    
    Sz_A = 0.0
    Sz_B = 0.0
    n_A = 0
    n_B = 0
    
    for key, rdm in rdms.items():
        if hasattr(key, 'sublattice'):
            Sz_val = np.trace(rdm @ Sz_full).real
            if key.sublattice == 0:
                Sz_A += Sz_val
                n_A += 1
            else:
                Sz_B += Sz_val
                n_B += 1
    
    if n_A > 0:
        observables['Sz_A'] = Sz_A / n_A
    if n_B > 0:
        observables['Sz_B'] = Sz_B / n_B
    
    # Staggered magnetization
    observables['m_stag'] = abs(observables.get('Sz_A', 0) - observables.get('Sz_B', 0)) / 2
    
    # <n> phonon number
    n_op = boson_ops.n
    Id_spin = np.eye(2)
    n_full = np.kron(Id_spin, n_op)
    
    n_phon = 0.0
    count = 0
    for rdm in rdms.values():
        n_phon += np.trace(rdm @ n_full).real
        count += 1
    
    if count > 0:
        observables['n_phonon'] = n_phon / count
    
    # Energy per site
    observables['energy'] = compute_energy(state, env, hamiltonian)
    
    return observables


def compute_energy(state, env, hamiltonian):
    """Compute energy per site using CTMRG environment."""
    H_bond = hamiltonian.get_bond_hamiltonian()
    
    # Contract environment with Hamiltonian
    lattice = state.unit_cell
    energy = 0.0
    n_bonds = 0
    
    for bond in lattice.get_bonds():
        # Get reduced density matrix for bond
        rdm2 = env.compute_rdm_two_site(state, bond)
        
        # Compute <H>
        e = np.trace(rdm2 @ H_bond).real
        energy += e
        n_bonds += 1
    
    if n_bonds > 0:
        energy /= n_bonds
    
    # Add local terms
    H_local = hamiltonian.get_local_hamiltonian()
    rdms = env.compute_rdms(state)
    
    e_local = 0.0
    n_sites = 0
    for rdm in rdms.values():
        e_local += np.trace(rdm @ H_local).real
        n_sites += 1
    
    if n_sites > 0:
        e_local /= n_sites
    
    return energy + e_local


def main():
    """Main simulation loop."""
    args = parse_args()
    
    print("=" * 60)
    print("Honeycomb Lattice Spin-Phonon iPEPS Simulation")
    print("=" * 60)
    print(f"Bond dimension D = {args.D}")
    print(f"Environment χ = {args.chi}")
    print(f"Heisenberg J = {args.J}")
    print(f"Phonon ω = {args.omega}")
    print(f"Coupling g = {args.g}")
    print(f"Max bosons = {args.n_bosons}")
    print("=" * 60)
    
    # Initialize profiler
    profiler = Profiler()
    
    # Create lattice
    with profiler.profile("initialization"):
        lattice = HoneycombLattice(lx=1, ly=1)
        
        # Create Hamiltonian
        hamiltonian = create_hamiltonian(
            J=args.J,
            omega=args.omega,
            g=args.g,
            n_bosons=args.n_bosons,
        )
        
        phys_dim = hamiltonian.local_dim
        print(f"Physical dimension = {phys_dim}")
        
        # Initialize checkpointing
        ckpt_manager = CheckpointManager(
            checkpoint_dir=args.checkpoint_dir,
            max_checkpoints=3,
        )
        auto_ckpt = AutoCheckpointer(ckpt_manager, interval=100)
        
        # Try to resume from checkpoint
        state = None
        env = None
        start_iter = 0
        
        if args.resume:
            result = ckpt_manager.load_latest()
            if result is not None:
                state, env, metadata = result
                start_iter = metadata.iteration
                print(f"Resumed from iteration {start_iter}")
        
        # Create new state if needed
        if state is None:
            state = iPEPSState.random(
                unit_cell=lattice,
                bond_dim=args.D,
                phys_dim=phys_dim,
            )
            state.normalize()
        
        # Initialize CTMRG environment if needed
        if env is None:
            ctmrg = CTMRG(chi=args.chi, max_iter=100, tol=1e-10)
            env, _ = ctmrg.run(state)
        
        # Initialize Simple Update
        su = SimpleUpdate(
            state=state,
            dt=args.dt,
            max_bond=args.D,
        )
        
        # Set Hamiltonian gates
        gate = hamiltonian.get_two_site_gate(dt=args.dt)
        su.set_hamiltonian(gate)
    
    # Storage for results
    energies = []
    magnetizations = []
    phonon_numbers = []
    times = []
    
    print("\nStarting optimization...")
    print("-" * 60)
    
    prev_energy = float('inf')
    
    for iteration in range(start_iter, args.steps):
        with profiler.profile("iteration"):
            # Simple Update step
            with profiler.profile("simple_update"):
                su.step()
            
            # Periodically run CTMRG and compute observables
            if iteration % 10 == 0:
                with profiler.profile("ctmrg"):
                    ctmrg = CTMRG(chi=args.chi, max_iter=50, tol=1e-8)
                    env, converged = ctmrg.run(state)
                
                with profiler.profile("observables"):
                    obs = compute_observables(state, env, hamiltonian)
                
                energy = obs['energy']
                m_stag = obs.get('m_stag', 0)
                n_phon = obs.get('n_phonon', 0)
                
                energies.append(energy)
                magnetizations.append(m_stag)
                phonon_numbers.append(n_phon)
                times.append(time.time())
                
                # Check convergence
                delta_e = abs(energy - prev_energy)
                prev_energy = energy
                
                print(
                    f"Iter {iteration:5d}: E = {energy:+.8f}, "
                    f"m_stag = {m_stag:.6f}, "
                    f"<n> = {n_phon:.4f}, "
                    f"ΔE = {delta_e:.2e}"
                )
                
                if delta_e < args.tol:
                    print(f"\nConverged at iteration {iteration}!")
                    break
            
            # Checkpoint
            auto_ckpt.maybe_checkpoint(
                ipeps_state=state,
                environment=env,
                iteration=iteration,
                energy=prev_energy,
                extra_data={'model_type': 'holstein'},
            )
    
    print("-" * 60)
    print("\nFinal Results:")
    print(f"  Energy per site: {energies[-1]:+.8f}")
    print(f"  Staggered magnetization: {magnetizations[-1]:.6f}")
    print(f"  Average phonon number: {phonon_numbers[-1]:.4f}")
    
    # Save results
    np.savez(
        args.output,
        energies=np.array(energies),
        magnetizations=np.array(magnetizations),
        phonon_numbers=np.array(phonon_numbers),
        times=np.array(times),
        parameters={
            'D': args.D,
            'chi': args.chi,
            'J': args.J,
            'omega': args.omega,
            'g': args.g,
            'n_bosons': args.n_bosons,
        },
    )
    print(f"\nResults saved to {args.output}")
    
    # Print profiling report
    print("\n" + profiler.report())


if __name__ == "__main__":
    main()
