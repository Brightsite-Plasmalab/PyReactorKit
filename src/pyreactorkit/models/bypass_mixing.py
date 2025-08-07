from pyreactorkit.models.base import ConvergingModel, HeatedMixingPlugFlowReactor
import numpy as np
import cantera as ct
import toddler
from typing import Tuple


class BypassMixingReactor(ConvergingModel, HeatedMixingPlugFlowReactor):
    mass_flow_rate_original: float
    T_max_target: float
    T_tolerance: float

    def __init__(self, T_max_target, T_tolerance=50, **kwargs):
        super(ConvergingModel, self).__init__(**kwargs)
        super(HeatedMixingPlugFlowReactor, self).__init__(**kwargs)
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

            adj = (np.max(solarr.T) / self.T_max_target) ** 2
            treated_flow_fraction_old = 1 - bypass_old
            treated_flow_fraction = treated_flow_fraction_old * adj

            bypass_fraction = 1 - treated_flow_fraction

        self.mass_flow_rate_slm = self.mass_flow_rate_original * (1 - bypass_fraction)
        self.set_mixing_profile(
            mixing_slm_total=self.mass_flow_rate_original
            * bypass_fraction
            # mixing_slm_total=0
        )

        # coreflow_
        r_plug = np.sqrt(bypass_fraction) * 13e-3
        r_plug = 13e-3
        self.L_circumference = 2 * np.pi * r_plug
        self.A_crosssection = np.pi * r_plug**2

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
