from types import NoneType
import numpy as np
import matplotlib.pyplot as plt
import cantera as ct

from pyreactorkit.models.base import (
    MyReactor,
    LossyPlugFlowReactor,
    HeatedMixingPlugFlowReactor,
    ConvergingModel,
)


class RecirculatingReactorIteration(MyReactor):
    """
    Simulates one iteration of the recirculating reactor model:
    - Input the edge reactor and the recirculation reactor parameters.
    - Simulate the edge reactor.
    - Use the output of the edge reactor to set the mixing profile of the recirculation reactor.
    - Simulate the recirculation reactor.
    """

    reactor_edge: LossyPlugFlowReactor
    reactor_recirculation: HeatedMixingPlugFlowReactor

    def __init__(self, reactor_edge, reactor_recirculation):
        self.reactor_edge = reactor_edge
        self.reactor_recirculation = reactor_recirculation

        assert np.array_equal(
            reactor_edge.z, reactor_recirculation.z[::-1]
        ), "Axial coordinates must match and be in oposite direction."

    def prepare_simulation(self):
        pass

    def simulate(self):
        print("Simulating edge reactor")
        solarr_edge, _, _ = self.reactor_edge.simulate()
        X_edge_end = solarr_edge.X[-1]
        T_end = solarr_edge.T[-1]
        p_end = solarr_edge.P[-1]

        self.reactor_recirculation.set_mixing_profile(
            T_end,
            p_end,
            X_edge_end,
            mixing_profile_kgps=self.reactor_edge.loss_profile_kgps[::-1],
        )

        print("Simulating recirculation reactor")
        results = self.reactor_recirculation.simulate()

        plt.figure()
        plt.plot(solarr_edge.z, solarr_edge.T)
        plt.xlabel("z [m]")
        plt.ylabel("Temperature [kg/s]")
        plt.show()

        return results


class RecirculatingReactor(ConvergingModel):
    recirculating_reactor: RecirculatingReactorIteration
    X0_last_iteration: dict = None

    def __init__(self, reactor_edge, reactor_recirculation):
        self.recirculating_reactor = RecirculatingReactorIteration(
            reactor_edge, reactor_recirculation
        )
        super().__init__(self.recirculating_reactor, 1e-2)

    def calculate_residuals(self, solarr: ct.SolutionArray):
        X0_previous = self.X0_last_iteration
        X0_new = dict(zip(solarr.species_names, solarr.X[-1]))

        self.X0_last_iteration = X0_new

        if X0_previous is None:
            return np.ones_like(solarr.X[-1])

        # Get species of both the new and previous iteration
        species = list(set(X0_previous.keys()) | set(X0_new.keys()))

        # Append missing species to the X0's
        relative_change = {}
        for s in species:
            if s not in X0_previous:
                X0_previous[s] = 0
            if s not in X0_new:
                X0_new[s] = 0
            relative_change[s] = (X0_new[s] - X0_previous[s]) / X0_new[s]

        X0_new = np.array([X0_new[s] for s in species])
        X0_previous = np.array([X0_previous[s] for s in species])

        X0_new[X0_new < 1e-2] = 0
        X0_previous[X0_previous < 1e-2] = 0

        X0_new[X0_new == 0] = np.nan
        relative_change = np.abs((X0_new - X0_previous) / X0_new)
        relative_change[np.isnan(relative_change)] = 0
        relative_change[np.isinf(relative_change)] = 1

        return relative_change

    def get_initial_parameters(self):
        return {
            "T_recirc": self.recirculating_reactor.reactor_recirculation.T0,
            "X_recirc": self.recirculating_reactor.reactor_edge.X0,
            "flowrate_kgps": 1,
        }

    def get_new_parameters(
        self,
        old_parameters: np.ndarray,
        solarr: ct.SolutionArray,
        residuals: np.ndarray | NoneType,
    ) -> np.ndarray:
        """
        For the next iteration of this converging model, use the outlet of the recirculating reactor to set the mixing composition of the edge reactor.
        """

        print(f"Peak temperature {np.max(solarr.T):.0f}K")

        return {
            "T_recirc": solarr.T[-1],
            "X_recirc": dict(zip(solarr.species_names, solarr.X[-1])),
            "flowrate_kgps": solarr.flowrate[-1],
        }

    def apply_parameters(self, parameters: dict):
        """Apply the new parameters to the reactor for the next iteration.

        In this case, the parameters are the recirculating composition and total flow rate.

        Args:
            parameters (dict): _description_
        """
        X_new = parameters["X_recirc"]
        T_new = parameters["T_recirc"]
        mdot_tot = parameters["flowrate_kgps"]
        p_new = self.recirculating_reactor.reactor_edge.pressure

        mixing_profile_new = self.recirculating_reactor.reactor_edge.mixing_profile_kgps
        mixing_profile_new *= mdot_tot / np.max(mixing_profile_new)

        self.recirculating_reactor.reactor_edge.set_mixing_profile(
            T_mix=T_new,
            p_mix=p_new,
            X_mix=X_new,
            mixing_profile_kgps=mixing_profile_new,
        )

    def format_parameters(self, parameters: np.ndarray):
        X_new = parameters["X_recirc"]
        mdot_tot = parameters["flowrate_kgps"]
        T_new = parameters["T_recirc"]

        threshold = 1e-3 * np.max(np.array(list(X_new.values())))
        composition_filtered = {
            k: f"{v:.2f}" for k, v in X_new.items() if v > threshold
        }

        return f"temperature: {T_new}, composition: {composition_filtered}, flowrate: {mdot_tot:.2f} SLM"

    def format_residuals(self, residuals: np.ndarray):
        max_residual = np.max(np.abs(residuals))

        return f"Max relative composition change: {max_residual*100:.1f}%"
