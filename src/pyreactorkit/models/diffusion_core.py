import numpy as np
import cantera as ct
import toddler
from src.pyreactorkit.main import SimResults


def get_thermal_enthalpy(gas, gas_ref):
    Ti = gas.T
    Hi = (
        gas.enthalpy_mass
    )  # Enthalpy at Ti, consisting of thermal and chemical enthalpy
    gas.TP = 300, gas.P
    Hf = gas.enthalpy_mass  # Enthalpy at 300 K ("thermal" enthalpy)
    Htherm = (Hi - Hf) - (gas.enthalpy_mass - gas_ref.enthalpy_mass)
    gas.TP = Ti, gas.P
    return Htherm  # [J/kg]


def get_outlet_composition(
    P,
    pressure,
    slm,
    T_core,
    volume,
    initial_composition,
    treated_flow_fraction_initial=1.0,
    initial_temp=300,
    heatflux_multiplier=0,  # WIP: disable heatflux out of the system
):

    # Define the species we want to keep
    species_names = initial_composition.keys()

    # Create a new Solution object with only the selected species
    # Initialize the gas phase with GRI-MECH 3.0
    gas_inlet = ct.Solution("gri30.yaml", species=species_names)
    gas_inlet.TPX = initial_temp, pressure, initial_composition
    gas_core = ct.Solution("gri30.yaml", species=species_names)
    gas_core.TPX = initial_temp, pressure, initial_composition

    massflux = toddler.physics.flow.slm_to_massflux(
        slm, gas_inlet.mean_molecular_weight
    )

    # Set up the two cells
    cell_plasma = ct.IdealGasReactor(gas_core, volume=volume)

    # Set up the reservoir for constant CO2 inflow
    res_inlet = ct.Reservoir(gas_inlet)

    # Set up the outlet (assumed to be at ambient pressure)
    res_outlet_plasma = ct.Reservoir(gas_core)

    # Add a heat source to cell2
    res_heat = ct.Reservoir(gas_inlet)
    heat_wall2 = ct.Wall(res_heat, cell_plasma, A=1, Q=P, K=0, U=0)  # 1000 W of heat

    # Mass flow controllers for inflow and outflow
    mfc_in = ct.MassFlowController(res_inlet, cell_plasma, mdot=massflux)  # kg/s
    # v_reactor_out = ct.Valve(cell_plasma, res_outlet_plasma, K=1e-5)
    mfc_out = ct.PressureController(
        cell_plasma, res_outlet_plasma, master=mfc_in, K=1e-5
    )

    # Create the reactor network
    network = ct.ReactorNet([cell_plasma])

    # Set tolerances for the solver
    network.rtol = 1e-6
    network.atol = 1e-15

    # Find the treated flow fraction that results in the measured core temperature
    treated_flow_fraction = treated_flow_fraction_initial
    i = 0
    while True:
        i += 1
        if i > 500:
            raise ValueError(
                f"Could not find solution in 500 iterations. Initial parameters: {P:.0f}W, {slm:.0f}slm, {T_core}K, {initial_composition}. Final parameters: T {Ti:.0f}K, {treated_flow_fraction:.2f}."
            )
        mfc_in.mass_flow_rate = massflux * treated_flow_fraction

        # Emulate losing the mass through diffusion by adding a heat source equal to the thermal enthalpy lost
        thermal_heatflux = (
            get_thermal_enthalpy(gas_core, gas_inlet) * massflux * treated_flow_fraction
        )
        heat_wall2.heat_flux = P + thermal_heatflux * (1 - heatflux_multiplier)

        network.reinitialize()
        network.advance_to_steady_state()

        Ti = gas_core.T

        dT = T_core - Ti

        if abs(dT) > 50:
            treated_flow_fraction -= dT * 1e-5
            if treated_flow_fraction < 0 or treated_flow_fraction > 1:
                raise ValueError(
                    f"Treated flow fraction invalid ({treated_flow_fraction:.2f}), could not find solution. Current temperature: {Ti:.0f}K, target temperature: {T_core}K."
                )
        else:
            break

    return (
        cell_plasma,
        Ti,
        treated_flow_fraction,
    )
