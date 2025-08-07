import logging
from typing import Tuple
import toddler
import collomech
import numpy as np
import cantera as ct

from pyreactorkit.models.base import ConvergingModel


class AbstractReactorCell:
    """
    This is a utility class to simulate a reactor cell with two gas inlets and one outlet. It is aimed at making it easy to re-use the same cell for different simulations,
    such as chains of continuous stirred tank reactors (CSTRs) or plug flow reactors (PFRs).
    """

    L_circumference: float
    A_crosssection: float

    reactor: ct.IdealGasReactor

    gas_inlet: list[ct.Solution] = []
    res_inlet: list[ct.Reservoir] = []
    mfc_inlet: list[ct.MassFlowController] = []

    gas_reactor: ct.Solution
    res_downstream: ct.Reservoir
    mfc_out: ct.PressureController

    gas_amb: ct.Solution
    res_amb: ct.Reservoir
    wall_tube: ct.Wall

    sim: ct.ReactorNet
    chemistry_set: str

    def __init__(self, diameter=26e-3, chemistry_set=None, N_inlets=1):
        if chemistry_set is None:
            chemistry_set = "gri30.yaml"
        self.chemistry_set = chemistry_set

        self.L_circumference = np.pi * diameter
        self.A_crosssection = 0.25 * np.pi * diameter**2

        self.gas_reactor = ct.Solution(chemistry_set)
        self.reactor = ct.IdealGasReactor(self.gas_reactor)
        # self.reactor = ct.IdealGasConstPressureReactor(self.gas_reactor)

        # Define inlets
        self.gas_inlet = []
        self.res_inlet = []
        self.mfc_inlet = []
        for _ in range(N_inlets):
            gas_inlet_i = ct.Solution(chemistry_set)
            res_inlet_i = ct.Reservoir(gas_inlet_i)
            mfc_inlet_i = ct.MassFlowController(res_inlet_i, self.reactor, mdot=0)

            self.gas_inlet.append(gas_inlet_i)
            self.res_inlet.append(res_inlet_i)
            self.mfc_inlet.append(mfc_inlet_i)

        # Define outlet
        if N_inlets >= 1:
            self.res_downstream = ct.Reservoir(self.gas_reactor)
            # self.mfc_out = ct.Valve(self.reactor, self.res_downstream, K=1e-5)
            self.mfc_out = ct.PressureController(
                self.reactor, self.res_downstream, primary=self.mfc_inlet[0], K=1e-5
            )

        # Define ambient gas for cooling
        self.gas_amb = ct.Solution(chemistry_set)
        self.res_amb = ct.Reservoir(self.gas_amb)
        self.res_amb.thermo.TP = 300, 1e5
        self.wall_tube = ct.Wall(self.res_amb, self.reactor, U=0, Q=0, A=1)

        self.sim = ct.ReactorNet([self.reactor])

    def set_inlet(
        self,
        i,
        T,
        pressure,
        composition,
        mass_flow_rate_slm: float = None,
        mass_flow_rate_kgps: float = None,
    ):
        """Sets the upstream conditions for the reactor.

        Sets the temperature, pressure, and composition of the gas inlet.
        Synchronizes the state of the upstream reservoir.
        Converts mass flow rate from SLM to kg/s if provided.
        Sets the mass flow coefficient for the mass flow controller.

        Args:
            i (int): Index of the inlet.
            T (float): Temperature in Kelvin.
            pressure (float): Pressure in Pascals.
            composition (dict): Composition of the gas mixture in the form {"A": 1, "B": 1}.
            mass_flow_rate_slm (float): Mass flow rate in standard liters per minute.
            mass_flow_rate_kgps (float): Mass flow rate in kilograms per second.
        """

        self.gas_inlet[i].TPX = T, pressure, composition

        if mass_flow_rate_slm is not None:
            mass_flow_rate_kgps = toddler.physics.flow.slm_to_massflux(
                mass_flow_rate_slm,
                self.gas_inlet[i].mean_molecular_weight,
            )
        self.mfc_inlet[i].mass_flow_rate = mass_flow_rate_kgps
        self.res_inlet[i].syncState()

        self.gas_reactor.TPX = T, pressure, composition
        self.reactor.syncState()
        self.res_downstream.syncState()

        self.sim.reinitialize()

        return

    def set_inlets(
        self,
        T: float,
        pressure: float,
        composition,
        mass_flow_rate_slm: float = None,
        mass_flow_rate_kgps: float = None,
    ):
        """Sets the upstream conditions for the reactor.

        Sets the temperature, pressure, and composition of the gas inlet.
        Synchronizes the state of the upstream reservoir.
        Converts mass flow rate from SLM to kg/s if provided.
        Sets the mass flow coefficient for the mass flow controller.

        Args:
            T (float): Temperature in Kelvin.
            pressure (float): Pressure in Pascals.
            composition (dict): Composition of the gas mixture in the form {"A": 1, "B": 1}.
            mass_flow_rate_slm (float, optional): Mass flow rate in standard liters per minute.
            mass_flow_rate_kgps (float, optional): Mass flow rate in kilograms per second.
        """
        if type(mass_flow_rate_slm) != list:
            mass_flow_rate_slm = [mass_flow_rate_slm] * len(self.gas_inlet)
        if type(mass_flow_rate_kgps) != list:
            mass_flow_rate_kgps = [mass_flow_rate_kgps] * len(self.gas_inlet)
        if type(composition) != list:
            composition = [composition] * len(self.gas_inlet)
        if type(T) != list:
            T = [T] * len(self.gas_inlet)

        for i in range(len(self.gas_inlet)):
            self.set_inlet(
                i,
                T[i],
                pressure,
                composition[i],
                mass_flow_rate_slm[i],
                mass_flow_rate_kgps[i],
            )

    def set_reactor(self, V: float, A: float):
        """
        Sets the volume and area of the reactor.
        Args:
            V (float): Volume of the reactor in cubic meters.
            A (float): Area of the reactor in square meters.
        """
        self.reactor.volume = V
        self.wall_tube.area = A

    def set_reactor_heating(self, P: float, U_heatloss: float = 1e2):
        self.wall_tube.heat_flux = P / self.wall_tube.area
        self.wall_tube.heat_transfer_coeff = U_heatloss

    def get_result(self):
        """Evaluate the reactor cell and return the state and the time step."""
        self.sim.reinitialize()
        self.sim.initial_time = 0

        try:
            self.sim.advance_to_steady_state()
        except ct.CanteraError as e:
            print("Cantera error")
            print("See https://cantera.org/dev/userguide/faq.html#reactor-networks")
            raise e

        if self.mfc_out.mass_flow_rate == 0:
            logging.warning("Mass flow rate is zero")

        dt = self.reactor.mass / (np.float32(self.mfc_out.mass_flow_rate))
        return self.reactor.thermo.state, dt


class SelfMixingReactor:
    N_R: int = 1
    cells: list[AbstractReactorCell]
    solarr: list[ct.SolutionArray]
    r: list[float]
    kgps: list[float]
    z: np.ndarray
    dz: float
    f_mix: float

    P: list[np.ndarray[float]]
    U_heatloss: list[float]

    INLET_AXIAL = 0
    INLET_MIX = 1
    INLET_QUENCH = 2

    CELL_CORE = 0
    CELL_SHELL = 1

    def __init__(
        self,
        N_R: int = 1,
        chemistry_set: str = "gri30",
        L_mix: float = 0.1,
        z_max=100e-3,
        N_z=100,
        **kwargs,
    ):

        self.N_R = N_R
        self.cells = [
            AbstractReactorCell(chemistry_set=chemistry_set, N_inlets=2)
            for _ in range(self.N_R)
        ]

        self.z = np.linspace(0, z_max, N_z)
        self.dz = self.z[1] - self.z[0]

        self.f_mix = self.dz / L_mix

    def set_heating_profiles(
        self, heating_profile: list[np.ndarray[float]], U_heatloss: list[float] = [0]
    ):
        self.P = heating_profile
        self.U_heatloss = U_heatloss

    def set_initial_conditions(
        self,
        T: float,
        p: float,
        composition: dict,
        mass_flow_rate_slm: float,
        f_flow: np.ndarray,
    ):
        self.T0 = T
        self.p = p
        self.X0 = composition
        f_flow = np.array(f_flow, dtype=np.float32)
        f_flow /= np.sum(f_flow)

        self.mass_flow_rate_slm = f_flow * mass_flow_rate_slm

    def prepare_simulation(self):
        # For each radial dimension, set the reactor cell
        for i, celli in enumerate(self.cells):
            celli.set_reactor(
                np.pi * 13e-3**2 * self.dz, 2 * np.pi * 13e-3 * self.dz
            )  # TODO: base on mass flow fractions?
            celli.set_inlets(
                self.T0,
                self.p,
                self.X0,
                mass_flow_rate_slm=[self.mass_flow_rate_slm[i], 0],
            )

    def get_state(self, cells: list[AbstractReactorCell] = None):
        if cells is None:
            return ["flowrate", "flowrates_in", "A_cross", "A_shell", "L", "V"]

        return [
            {
                "flowrate": celli.mfc_out.mass_flow_rate,
                "flowrates_in": [mfci.mass_flow_rate for mfci in celli.mfc_inlet],
                "A_cross": celli.A_crosssection,
                "A_shell": celli.wall_tube.area,
                "L": celli.L_circumference,
                "V": celli.reactor.volume,
            }
            for celli in cells
        ]

    def prepare_step(self, i):
        if i == 0:
            return

        flow_mix_i = self.solarr[self.CELL_CORE][0].flowrate * self.f_mix

        # core -> core
        self.cells[self.CELL_CORE].set_inlet(
            self.INLET_AXIAL,
            self.solarr[self.CELL_CORE][i - 1].T,
            self.solarr[self.CELL_CORE][i - 1].P,
            self.solarr[self.CELL_CORE][i - 1].X,
            mass_flow_rate_kgps=self.solarr[self.CELL_CORE][i - 1].flowrate
            - flow_mix_i,
        )
        # core -> shell
        self.cells[self.CELL_SHELL].set_inlet(
            self.INLET_MIX,
            self.solarr[self.CELL_CORE][i - 1].T,
            self.solarr[self.CELL_CORE][i - 1].P,
            self.solarr[self.CELL_CORE][i - 1].X,
            mass_flow_rate_kgps=flow_mix_i,
        )
        # shell -> core
        self.cells[self.CELL_CORE].set_inlet(
            self.INLET_MIX,
            self.solarr[self.CELL_SHELL][i - 1].T,
            self.solarr[self.CELL_SHELL][i - 1].P,
            self.solarr[self.CELL_SHELL][i - 1].X,
            mass_flow_rate_kgps=flow_mix_i,
        )
        # shell -> shell
        self.cells[self.CELL_SHELL].set_inlet(
            self.INLET_AXIAL,
            self.solarr[self.CELL_SHELL][i - 1].T,
            self.solarr[self.CELL_SHELL][i - 1].P,
            self.solarr[self.CELL_SHELL][i - 1].X,
            mass_flow_rate_kgps=self.solarr[self.CELL_SHELL][i - 1].flowrate
            - flow_mix_i,
        )

        for j, cellj in enumerate(self.cells):
            cellj.set_reactor_heating(self.P[j][i], U_heatloss=self.U_heatloss[j])

        return

    def simulate(self):
        dt = np.zeros_like(self.z)

        self.solarr = [
            ct.SolutionArray(
                self.cells[0].gas_inlet[0], extra=["t", "z", *self.get_state()]
            )
            for _ in range(self.N_R)
        ]

        self.prepare_simulation()
        for i, zi in enumerate(self.z):
            self.prepare_step(i)

            for cellj in self.cells:
                _, _ = cellj.get_result()
            for j, cellj in enumerate(self.cells):
                statej = self.get_state(self.cells)
                self.solarr[j].append(
                    cellj.reactor.thermo.state, z=zi, t=np.sum(dt), **statej[j]
                )

        return self.solarr


class SelfMixingConvergingReactor(ConvergingModel, SelfMixingReactor):
    mass_flow_rate_original: float
    T_max_target: float
    T_tolerance: float

    def __init__(self, T_max_target, T_tolerance=50, **kwargs):
        # super(ConvergingModel, self).__init__(**kwargs)
        # super(SelfMixingReactor, self).__init__(**kwargs)
        super().__init__(**kwargs)
        self.set_target(T_max_target, T_tolerance)

    def set_target(self, T_max_target: float, T_tolerance: float = 50):
        """Set the target temperature and tolerance for convergence."""
        self.T_max_target = T_max_target
        self.T_tolerance = T_tolerance

    def get_iteration_results(self):
        solarr_core, solarr_edge = self.simulate()
        T_max = np.max(solarr_core.T)
        print(
            f" resulted in temperature {T_max:.0f}K (target {self.T_max_target:.0f}K)"
        )
        return (solarr_core, solarr_edge)

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
            solarr_core = results[0]

            adj = (np.max(solarr_core.T) / self.T_max_target) ** 1.1
            treated_flow_fraction_old = 1 - bypass_old
            treated_flow_fraction = treated_flow_fraction_old * adj

            bypass_fraction = 1 - treated_flow_fraction

        self.mass_flow_rate_slm = np.sum(self.mass_flow_rate_slm) * np.array(
            [1 - bypass_fraction, bypass_fraction]
        )
        self.prepare_simulation()

        print(f" bypass fraction: {bypass_fraction*100:.1f}%")

        parameters = (bypass_fraction,)
        return parameters

    def _get_initial_parameters(self):
        self.mass_flow_rate_original = self.mass_flow_rate_slm

        # Calculate enthalpy of initial mixture
        gas = ct.Solution(self.cells[0].chemistry_set)
        gas.TPX = self.T0, self.p, self.X0
        gas.equilibrate("TP")
        h0 = gas.enthalpy_mass

        # Calculate enthalpy of that mixture at target temperature
        gas.TP = self.T_max_target, self.p
        h_target = gas.enthalpy_mass

        # Calculate the needed SEI to reach the target temperature
        SEI_needed = h_target - h0  # [J/kg]

        # Calculate the global SEI
        flowrate_kgps = (
            toddler.physics.flow.slm_to_moles(self.mass_flow_rate_slm)
            / 1e3
            * gas.mean_molecular_weight
        )
        SEI_Jpkg = np.sum(self.P) / flowrate_kgps[0]  # [J/kg]

        # Calculate the fraction of flow that can reach the target temperature
        bypass_fraction = 1 - SEI_needed / SEI_Jpkg

        return bypass_fraction
