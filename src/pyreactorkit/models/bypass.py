from pyreactorkit.models.base import ConvergingModel, HeatedPlugFlowReactor
import numpy as np
import cantera as ct
import toddler
from typing import Tuple


class BypassReactor(ConvergingModel, HeatedPlugFlowReactor):
    mass_flow_rate_original: float
    T_max_target: float
    T_tolerance: float

    def __init__(self, T_max_target, T_tolerance=50, **kwargs):
        super(ConvergingModel, self).__init__(**kwargs)
        super(HeatedPlugFlowReactor, self).__init__(**kwargs)
        self.T_tolerance = T_tolerance
        self.T_max_target = T_max_target
        self.mass_flow_rate_original = self.mass_flow_rate_slm

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
            bypass_fraction = self._get_initial_parameters()
        else:
            bypass_old = parameters[0]
            solarr = results[0]

            adj = (np.max(solarr.T) / self.T_max_target) ** 1.1
            treated_flow_fraction_old = 1 - bypass_old
            treated_flow_fraction = treated_flow_fraction_old * adj

            bypass_fraction = 1 - treated_flow_fraction

        self.mass_flow_rate_slm = self.mass_flow_rate_original * (1 - bypass_fraction)

        print(f" bypass fraction: {bypass_fraction*100:.1f}%")

        parameters = (bypass_fraction,)
        return parameters

    def _get_initial_parameters(self):
        P = np.sum(self.heating_profile)  # [W] total power

        # Calculate enthalpy of initial mixture
        gas = ct.Solution(self.cell.chemistry_set)
        gas.TPX = self.T0, self.pressure, self.X0
        h0 = gas.enthalpy_mass

        # Calculate enthalpy of that mixture at target temperature
        gas.TP = self.T_max_target, self.pressure
        h_target = gas.enthalpy_mass

        # Calculate the flowrate to get to the needed SEI
        SEI_needed = h_target - h0
        flowrate_kgps = toddler.physics.flow.slm_to_moles(self.mass_flow_rate_slm) / 1e3
        SEI_Jpkg = P / flowrate_kgps
        bypass_fraction = 1 - SEI_needed / SEI_Jpkg

        return bypass_fraction
