from types import NoneType
from typing import Tuple, Union
import numpy as np
import cantera as ct
import toddler

import matplotlib.pyplot as plt
from pyreactorkit.util import kgps_to_slm
import logging


class MyReactorCell:
    """
    This is a utility class to simulate a reactor cell with two gas inlets and one outlet. It is aimed at making it easy to re-use the same cell for different simulations,
    such as chains of continuous stirred tank reactors (CSTRs) or plug flow reactors (PFRs).
    """

    L_circumference: float
    A_crosssection: float

    reactor: ct.IdealGasReactor

    gas_inlet: ct.Solution
    res_upstream: ct.Reservoir
    mfc_in: ct.MassFlowController

    gas_mix: ct.Solution
    res_mix: ct.Reservoir
    mfc_mix: ct.MassFlowController

    # gas_exchange: ct.Solution
    # res_exchange: ct.Reservoir
    # mfc_exchange: ct.MassFlowController

    gas_outlet: ct.Solution
    res_downstream: ct.Reservoir
    mfc_out: ct.PressureController

    gas_amb: ct.Solution
    res_amb: ct.Reservoir
    wall_tube: ct.Wall

    chemistry_set: str
    sim: ct.ReactorNet

    def __init__(self, diameter=26e-3, chemistry_set=None):
        if chemistry_set is None:
            chemistry_set = "gri30.yaml"
        self.chemistry_set = chemistry_set

        self.L_circumference = np.pi * diameter
        self.A_crosssection = 0.25 * np.pi * diameter**2

        self.gas_outlet = ct.Solution(chemistry_set)
        self.reactor = ct.IdealGasReactor(self.gas_outlet)
        # self.reactor = ct.IdealGasConstPressureReactor(self.gas_outlet)

        self.gas_inlet = ct.Solution(chemistry_set)
        self.res_upstream = ct.Reservoir(self.gas_inlet)
        self.mfc_in = ct.MassFlowController(self.res_upstream, self.reactor, mdot=0)

        self.gas_amb = ct.Solution(chemistry_set)
        self.res_amb = ct.Reservoir(self.gas_amb)
        self.res_amb.thermo.TP = 300, 1e5
        self.wall_tube = ct.Wall(self.res_amb, self.reactor, U=0, Q=0, A=1)

        self.gas_mix = ct.Solution(chemistry_set)
        self.res_mix = ct.Reservoir(self.gas_mix)
        self.mfc_mix = ct.MassFlowController(self.res_mix, self.reactor, mdot=0)

        # self.mfc_exchange = ct.MassFlowController(self.res_mix, self.reactor, mdot=0)

        self.res_downstream = ct.Reservoir(self.gas_outlet)
        # self.mfc_out = ct.Valve(self.reactor, self.res_downstream, K=1e-5)
        self.mfc_out = ct.PressureController(
            self.reactor, self.res_downstream, primary=self.mfc_in, K=1e-5
        )

        self.sim = ct.ReactorNet([self.reactor])

    def set_upstream(
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

        self.gas_inlet.TPX = T, pressure, composition
        self.gas_outlet.TPX = T, pressure, composition
        self.reactor.syncState()
        self.res_upstream.syncState()
        self.res_downstream.syncState()
        self.res_mix.syncState()

        self.sim.reinitialize()

        if mass_flow_rate_slm is not None:
            mass_flow_rate_kgps = toddler.physics.flow.slm_to_massflux(
                mass_flow_rate_slm, self.res_upstream.thermo.mean_molecular_weight
            )

        self.mfc_in.mass_flow_coeff = mass_flow_rate_kgps

    def set_mix(
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

        self.gas_mix.TPX = T, pressure, composition
        self.res_mix.syncState()

        if mass_flow_rate_slm is not None:
            mass_flow_rate_kgps = toddler.physics.flow.slm_to_massflux(
                mass_flow_rate_slm, self.res_mix.thermo.mean_molecular_weight
            )

        self.mfc_mix.mass_flow_coeff = mass_flow_rate_kgps

        if np.isnan(self.mfc_mix.mass_flow_rate):
            print("Mass flow rate is NaN")

    def set_reactor(self, V: float, A: float):
        """Sets the reactor conditions for the next step.

        Sets the volume of the reactor and the heat flux on the wall.
        The volume is calculated from the cross-sectional area and the height dz.
        The heat flux is calculated from the power P and the circumference of the reactor.

        Args:
            dz (float): Axial length of the reactor cell.
            P (float): Power input to the reactor cell.
        """
        self.reactor.volume = V
        self.wall_tube.area = A

    def set_reactor_heating(self, P: float, U_heatloss: float = 1e2):
        self.wall_tube.heat_flux = P / self.wall_tube.area
        self.wall_tube.heat_transfer_coeff = U_heatloss

    def downstream_to_upstream(self):
        """
        Sets the downstream conditions as the upstream conditions for the next step.
        """
        mdot_tot = self.mfc_out.mass_flow_rate
        if mdot_tot == 0:
            logging.warning("Mass flow rate is zero")

        self.set_upstream(
            self.reactor.thermo.T,
            self.reactor.thermo.P,
            self.reactor.thermo.mole_fraction_dict(),
            mass_flow_rate_kgps=mdot_tot,
        )

    def get_result(self):
        """Evaluate the reactor cell and return the state and the time step."""
        # self.res_upstream.syncState()
        # self.res_mix.syncState()
        # self.reactor.syncState()
        self.sim.reinitialize()
        self.sim.initial_time = 0

        # V_start = self.reactor.volume

        try:
            self.sim.advance_to_steady_state()
        except ct.CanteraError as e:
            print("Cantera error")
            print("See https://cantera.org/dev/userguide/faq.html#reactor-networks")
            raise e

        if self.mfc_out.mass_flow_rate == 0:
            logging.warning("Mass flow rate is zero")

        # print(f"Growfac: {self.reactor.volume / V_start:.3f}")
        # print(f"New pressure: {self.reactor.thermo.P:.3f} Pa")
        dt = self.reactor.mass / (self.mfc_out.mass_flow_rate)

        return self.reactor.thermo.state, dt


class Reactor0D:
    chemistry_set: str
    gas: ct.Solution
    r: ct.IdealGasConstPressureMoleReactor
    sim: ct.ReactorNet

    def __init__(self, chemistry_set=None):
        if chemistry_set is None:
            chemistry_set = "gri30.yaml"
        self.chemistry_set = chemistry_set

        self.gas = ct.Solution(self.chemistry_set)

        self.r = ct.IdealGasConstPressureMoleReactor(self.gas)

        self.sim = ct.ReactorNet([self.r])
        self.sim.verbose = False

    def run(
        self,
        T,
        p,
        X0,
        fix_T=True,
        dt_max=1e-6,
        t_max=1,
        N_max=1e5,
        dn_rel_max=1e-5,
        verbose=False,
    ):
        self.gas.TPX = T, p, X0
        self.r.energy_enabled = not fix_T
        self.sim.initial_time = 0
        self.r.syncState()
        self.sim.reinitialize()
        # self.sim.initialize()

        # limit advance when temperature difference is exceeded
        # delta_T_max = 20.0
        # self.r.set_advance_limit("temperature", delta_T_max)

        states = ct.SolutionArray(self.gas, extra=["t", "V"])

        i = 0
        while self.sim.time < t_max:
            self.sim.advance(self.sim.time + dt_max)
            states.append(self.r.thermo.state, t=self.sim.time, V=self.r.volume)

            if i >= N_max - 1:
                print("Iteration limit reached")
                break

            i += 1
            if verbose and i % 100 == 0:
                print(i)

        return states

    def to_pfr(solarr, A=np.pi * 13e-3**2):
        pass  # TODO


class MyReactor:
    def prepare_simulation(self):
        pass

    def simulate(self):
        raise NotImplementedError


class PlugFlowReactor(MyReactor):
    pressure: float  # Pa
    T0: float  # K
    X0: dict  # mole fractions
    mass_flow_rate_slm: float = None  # SLM
    mass_flow_rate_kgps: float = None  # kg/s
    z: np.ndarray  # m
    dz: float  # m

    A_crosssection: float = np.pi * 13e-3**2  # m2
    L_circumference: float = 2 * np.pi * 13e-3  # m

    cell: MyReactorCell

    def __init__(
        self,
        T0,
        pressure,
        X0,
        z,
        dz=None,
        mass_flow_rate_slm=None,
        mass_flow_rate_kgps=None,
        chemistry_set=None,
        **kwargs,
    ):
        super(MyReactor, self).__init__(**kwargs)

        if dz is None:
            dz = z[1] - z[0]

        self.set(
            T0=T0,
            pressure=pressure,
            mass_flow_rate_slm=mass_flow_rate_slm,
            mass_flow_rate_kgps=mass_flow_rate_kgps,
            X0=X0,
            z=z,
            dz=dz,
        )

        self.cell = MyReactorCell(chemistry_set=chemistry_set)

    def set(
        self,
        T0=None,
        pressure=None,
        mass_flow_rate_slm=None,
        mass_flow_rate_kgps=None,
        X0=None,
        z=None,
        dz=None,
    ):
        if T0 is not None:
            self.T0 = T0
        if pressure is not None:
            self.pressure = pressure
        if mass_flow_rate_slm is not None:
            self.mass_flow_rate_slm = mass_flow_rate_slm
        if mass_flow_rate_kgps is not None:
            self.mass_flow_rate_kgps = mass_flow_rate_kgps
        if X0 is not None:
            self.X0 = X0
        if z is not None:
            self.z = z
        if dz is not None:
            self.dz = dz

    def prepare_simulation(self):
        self.cell.set_upstream(
            self.T0,
            self.pressure,
            self.X0,
            mass_flow_rate_slm=self.mass_flow_rate_slm,
            mass_flow_rate_kgps=self.mass_flow_rate_kgps,
        )

    def prepare_cell(self, i):
        if i > 0:
            self.cell.downstream_to_upstream()

        A_crosssection_i = (
            self.A_crosssection[i]
            if isinstance(self.A_crosssection, np.ndarray)
            else self.A_crosssection
        )
        L_circumference_i = (
            self.L_circumference[i]
            if isinstance(self.L_circumference, np.ndarray)
            else self.L_circumference
        )

        # self.cell.sim.reinitialize()
        self.cell.set_reactor(
            self.dz * A_crosssection_i,
            self.dz * L_circumference_i,
        )

    def simulate(self):
        dt = np.zeros_like(self.z)

        solarr = ct.SolutionArray(
            self.cell.gas_inlet, extra=["t", "z", "flowrate", "flowrate_mix"]
        )

        self.prepare_simulation()
        for i, zi in enumerate(self.z):
            self.prepare_cell(i)

            state, dt[i] = self.cell.get_result()
            solarr.append(
                state,
                z=zi,
                t=np.sum(dt),
                flowrate=self.cell.mfc_out.mass_flow_rate,  # [kg/s]
                flowrate_mix=self.cell.mfc_mix.mass_flow_rate,  # [kg/s]
            )

        return solarr, self.cell.reactor.thermo, state


class HeatedPlugFlowReactor(PlugFlowReactor):
    heating_profile = None  # W
    U_heatloss = None  # W/m2

    def get_total_power(self):
        return np.sum(self.heating_profile)

    def set_heating_profile(self, heating_profile=None, U_heatloss=None, P=None):
        if heating_profile is not None:
            self.heating_profile = heating_profile
        if U_heatloss is not None:
            self.U_heatloss = U_heatloss
        if P is not None:
            self.heating_profile = (
                self.heating_profile * P / np.sum(self.heating_profile)
            )

    def prepare_simulation(self):
        super().prepare_simulation()
        assert (self.heating_profile is not None) and (
            self.U_heatloss is not None
        ), "Heating profile and heat loss coefficient must be set for a heated plug flow reactor. Use set_heating_profile() method."

    def prepare_cell(self, i):
        super().prepare_cell(i)
        self.cell.set_reactor_heating(
            self.heating_profile[i],
            self.U_heatloss,
        )


class MixingPlugFlowReactor(PlugFlowReactor):
    T_mix: float  # K
    p_mix: float  # Pa
    X_mix: dict  # mole fractions
    mixing_profile_slm: np.ndarray = None  # slm
    mixing_profile_kgps: np.ndarray = None  # kg/s

    def set_mixing_profile(
        self,
        T_mix=None,
        p_mix=None,
        X_mix=None,
        mixing_profile_slm=None,
        mixing_profile_kgps=None,
        mixing_slm_total=None,
        mixing_kgps_total=None,
    ):
        if T_mix is not None:
            self.T_mix = T_mix
        if p_mix is not None:
            self.p_mix = p_mix
        if X_mix is not None:
            self.X_mix = X_mix

        if mixing_profile_kgps is not None or mixing_profile_slm is not None:
            self.mixing_profile_slm = mixing_profile_slm
            self.mixing_profile_kgps = mixing_profile_kgps

        if mixing_slm_total is not None:
            if self.mixing_profile_slm is not None:
                if mixing_slm_total == 0:
                    self.mixing_profile_slm = np.zeros_like(self.mixing_profile_slm)
                else:
                    self.mixing_profile_slm *= mixing_slm_total / np.sum(
                        self.mixing_profile_slm
                    )
            else:
                raise ValueError(
                    "mixing_profile_slm must be set if mixing_slm_total is set"
                )
        elif mixing_kgps_total is not None:
            if mixing_profile_kgps is not None:
                if mixing_kgps_total == 0:
                    self.mixing_profile_kgps = np.zeros_like(self.mixing_profile_slm)
                else:
                    self.mixing_profile_kgps *= mixing_kgps_total / np.sum(
                        self.mixing_profile_kgps
                    )
            else:
                raise ValueError(
                    "mixing_profile_kgps must be set if mixing_kgps_total is set"
                )

    def prepare_cell(self, i):
        super().prepare_cell(i)

        self.cell.set_mix(
            self.T_mix,
            self.p_mix,
            self.X_mix,
            mass_flow_rate_slm=(
                self.mixing_profile_slm[i]
                if self.mixing_profile_slm is not None
                else None
            ),
            mass_flow_rate_kgps=(
                self.mixing_profile_kgps[i]
                if self.mixing_profile_kgps is not None
                else None
            ),
        )

        # if (
        #     self.cell.mfc_mix.mass_flow_rate == 0
        #     and self.cell.mfc_in.mass_flow_rate == 0
        # ):
        #     logging.warning("Mass flow rate is zero")

    def prepare_simulation(self):
        super().prepare_simulation()
        assert (self.mixing_profile_slm is not None) or (
            self.mixing_profile_kgps is not None
        ), "Mixing profile must be set for a mixing plug flow reactor. Use set_mixing_profile() method."

    # def get_heatflow(self, flowrate_kgps):
    #  TODO
    #     gas = ct.Solution(self.cell.chemistry_set)
    #     gas.X = self.X0
    #     h_main = gas.enthalpy_mass

    #     gas.X = self.X_mix
    #     h_mix = gas.enthalpy_mass


class HeatedMixingPlugFlowReactor(HeatedPlugFlowReactor, MixingPlugFlowReactor):
    def prepare_cell(self, i):
        HeatedPlugFlowReactor.prepare_cell(self, i)
        MixingPlugFlowReactor.prepare_cell(self, i)

    def prepare_simulation(self):
        HeatedPlugFlowReactor.prepare_simulation(self)
        MixingPlugFlowReactor.prepare_simulation(self)


class LossyPlugFlowReactor(HeatedMixingPlugFlowReactor):
    loss_profile_kgps: np.ndarray = None  # [slm] differential loss profile

    def set_mass_flow_rate_loss(self, loss_profile_kgps):
        self.loss_profile_kgps = loss_profile_kgps

        total_loss = np.sum(loss_profile_kgps)
        print(f"Total mass flow rate loss: {total_loss:.3f} kgps")

    def prepare_cell(self, i):
        super().prepare_cell(i)

        remaining_mass_flow = (
            self.cell.mfc_in.mass_flow_rate - self.loss_profile_kgps[i]
        )

        if remaining_mass_flow < 0:
            logging.warning("Mass flow rate is negative")

        self.cell.mfc_in.mass_flow_rate = remaining_mass_flow

    def prepare_simulation(self):
        super().prepare_simulation()
        assert (
            self.loss_profile_kgps is not None
        ), "Loss profile must be set for a lossy plug flow reactor. Use set_mass_flow_rate_loss() method."


class ConvergingModel:
    """
    A converging model is a model that iteratively adjusts its parameters to converge a simulation to a desired state.
    Examples could be to adjust a flow rate to reach a target peak temperature, or to recirculate flow until the output does not change anymore.

    The model should implement the following methods:
    - get_iteration_results() -> Tuple(results)
    - is_converged() -> True/False
    - adjust_for_next_iteration(Tuple(parameters), Tuple(results)) -> Tuple(parameters)
    """

    def __init__(self, **kwargs):
        # super(ConvergingModel, self).__init__(kwargs)
        super().__init__(**kwargs)

    def get_iteration_results(self) -> Tuple:
        raise NotImplementedError

    def is_converged(self, Tuple) -> bool:
        raise NotImplementedError

    def adjust_for_next_iteration(self, parameters: Tuple, results: Tuple) -> Tuple:
        raise NotImplementedError

    def converge(self):
        results = None
        results_all = []
        parameters = None

        i = 0

        while True:
            print(f"Iteration {i}")

            parameters = self.adjust_for_next_iteration(parameters, results)
            results = self.get_iteration_results()
            results_all.append(results)

            if self.is_converged(i, parameters, results):
                break

            i += 1

        print(f"Converged after {i} iterations")

        return results, results_all, parameters
