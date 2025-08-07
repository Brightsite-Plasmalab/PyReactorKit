from pyreactorkit.models.base import ConvergingModel, HeatedPlugFlowReactor
import numpy as np
import cantera as ct
import toddler
from typing import Tuple


class ContractionReactor(ConvergingModel, HeatedPlugFlowReactor):
    """
    This reactor iterates the fraction of applied power in order to reach a target temperature.
    """

    P_original: float
    T_max_target: float
    T_tolerance: float

    def __init__(self, P_original, T_max_target, T_tolerance=50, **kwargs):
        super(ConvergingModel, self).__init__(**kwargs)
        super(HeatedPlugFlowReactor, self).__init__(**kwargs)
        self.T_tolerance = T_tolerance
        self.T_max_target = T_max_target
        self.P_original = P_original

    def get_iteration_results(self):
        solarr, _, _ = self.simulate()
        T_max = np.max(solarr.T)
        print(
            f" resulted in temperature {T_max:.0f}K (target {self.T_max_target:.0f}K)"
        )
        return (solarr,)

    def is_converged(self, i, parameters: Tuple, results: Tuple):
        if results is None:
            return False

        solarr = results[0]

        return abs(np.max(solarr.T) - self.T_max_target) < self.T_tolerance

    def adjust_for_next_iteration(self, parameters: Tuple, results: Tuple):
        if results is None:
            contraction_factor = self._get_initial_parameters()
        else:
            contraction_factor_old = parameters[0]
            solarr = results[0]

            adj = (
                np.max(solarr.T) / self.T_max_target
            ) ** 1.3  # Factor that the current temperature is off from the target. >1 is too high, <1 is too low

            contraction_factor = contraction_factor_old / adj

        # Adjust total power for next iteration
        self.set_heating_profile(P=self.P_original * contraction_factor)

        print(f" contraction fraction: {contraction_factor*100:.1f}% (% power in core)")

        parameters = (contraction_factor,)
        return parameters

    def _get_initial_parameters(self):
        P = np.sum(self.heating_profile)  # [W] total power

        # Calculate enthalpy of initial mixture
        gas = ct.Solution(self.cell.chemistry_set)
        gas.TPX = self.T0, self.pressure, self.X0
        h0 = gas.enthalpy_mass  # [J/kg]

        # Calculate enthalpy of that mixture at target temperature
        gas.TP = self.T_max_target, self.pressure
        h_target = gas.enthalpy_mass  # [J/kg]

        # Calculate the flowrate to get to the needed SEI
        SEI_needed = h_target - h0
        flowrate_kgps = toddler.physics.flow.slm_to_massflux(
            self.mass_flow_rate_slm, gas.mean_molecular_weight
        )
        SEI_Jpkg = P / flowrate_kgps
        contraction_factor = SEI_needed / SEI_Jpkg

        return contraction_factor
