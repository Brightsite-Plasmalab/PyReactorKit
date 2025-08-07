from types import NoneType
import numpy as np
import matplotlib.pyplot as plt
import cantera as ct
from typing import Tuple
from pyreactorkit.display import print_species

from pyreactorkit.models.base import (
    HeatedMixingPlugFlowReactor,
    ConvergingModel,
    HeatedPlugFlowReactor,
)


class SimpleRecirculatingReactor(ConvergingModel):
    reactor_edge: HeatedMixingPlugFlowReactor
    reactor_recirculation: HeatedPlugFlowReactor

    X_end_previous_iteration: np.ndarray = None
    T_end_previous_iteration: float = None

    min_iterations: int
    max_iterations: int

    def __init__(
        self, reactor_edge, reactor_recirculation, min_iterations=5, max_iterations=100
    ):
        self.reactor_edge = reactor_edge
        self.reactor_recirculation = reactor_recirculation

        self.min_iterations = min_iterations
        self.max_iterations = max_iterations

        assert np.array_equal(
            reactor_edge.z, reactor_recirculation.z[::-1]
        ), "Axial coordinates must match and be in oposite direction."

    def get_iteration_results(self):
        # Simulate recirculating reactor first
        results_recirc = self.reactor_recirculation.simulate()
        solarr_recirc, _, _ = results_recirc

        X_recirc_end = solarr_recirc.X[-1]
        T_recirc_end = solarr_recirc.T[-1]
        print_species(
            solarr_recirc[-1], prefix="  core end mass fractions: ", concat=", "
        )
        print(
            f"  core temperature: peak {np.max(solarr_recirc.T):.0f}K, end {T_recirc_end:.0f}K"
        )

        # Simulate the edge reactor, with the recirculating reactor outlet as the mixing profile
        self.reactor_edge.set_mixing_profile(T_mix=T_recirc_end, X_mix=X_recirc_end)
        results_edge = self.reactor_edge.simulate()

        # print formatted results
        solarr, _, _ = results_edge
        print_species(solarr[-1], prefix="  edge end mass fractions: ", concat=", ")
        print(f"  edge end temperature: {solarr.T[-1]:.0f}K")

        return (results_edge, results_recirc)

    def is_converged(self, i, parameters: Tuple, results: Tuple):
        if i < self.min_iterations:
            return False
        if i >= self.max_iterations:
            return True

        results_edge, results_recirc = results

        # Check if the composition of the major species has converged
        X_end = results_edge[0].X[-1]

        if self.X_end_previous_iteration is None:
            self.X_end_previous_iteration = X_end
            self.T_end_previous_iteration = results_edge[0].T[-1]
            return False

        # Calculate the relative change
        X_end[X_end == 0] = np.nan
        relchange = np.abs((X_end - self.X_end_previous_iteration) / X_end)
        relchange[np.isnan(relchange)] = 0

        # Identify major species (mass fraction > 1%)
        idx_majorspecies = results_edge[0].Y[-1] > 0.01
        relchange = relchange[idx_majorspecies]

        # Prepare for next iteration
        self.X_end_previous_iteration = X_end
        self.T_end_previous_iteration = results_edge[0].T[-1]

        max_relchange = np.max(relchange)
        delta_T = results_edge[0].T[-1] - self.T_end_previous_iteration
        print(f"  max relative change: {max_relchange:.3e} (max allowed: 1e-3)")
        return (max_relchange < 1e-3) and (delta_T < 20)

    def adjust_for_next_iteration(self, parameters: Tuple, results: Tuple):
        if results is None:
            return (parameters,)

        # Adjust the starting composition of the recirculating core to be equal to the outlet of the edge reactor of the previous simulation

        results_edge, results_recirc = results
        solarr, _, _ = results_edge

        T_edge_prev = solarr.T[-1]
        X_edge_prev = dict(zip(solarr.species_names, solarr.X[-1]))

        self.reactor_recirculation.set(
            T0=T_edge_prev,
            X0=X_edge_prev,
        )

        return (T_edge_prev, X_edge_prev)
