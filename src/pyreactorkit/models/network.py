import numpy as np
import matplotlib.pyplot as plt
import cantera as ct
import cantera as ct
from typing_extensions import Any
from pyreactorkit.util import mass_frac_dict, combined_mass_fracs


def discretize(
    N_axial,
    z: np.ndarray,
    massflow_tot: float,
    massflow_core_neg: np.ndarray,
    massflow_core_pos: np.ndarray,
    massflow_core_edge_dl,
):
    # Now discretize that data to N axial points
    N_bin = len(z) // N_axial

    z = z[::N_bin]
    dz = np.abs(np.diff(z)[0])

    # Find the index where z is closest to 100 mm
    idx_attach = np.argmin(np.abs(z[::N_bin] - -90e-3))

    # Mass flow is 0 before the attachment point, and constant after that
    flow_axial_tot = (
        np.ones_like(z)
        * massflow_tot
        * (np.arange(np.size(z)) >= idx_attach).astype(float)
    )

    # Bin the mass flow data
    flow_axial_core_neg = massflow_core_neg[::N_bin]
    flow_axial_core_pos = massflow_core_pos[::N_bin]

    # Calculate the total axial flow in the core
    flow_axial_core = flow_axial_core_pos + flow_axial_core_neg
    flow_axial_edge = flow_axial_tot - flow_axial_core

    # Net radial flow resutls from changes in axial flow
    flow_radial_net = -np.diff(flow_axial_core, prepend=0.0)

    # Calculate the core-to-edge mass flow from the CFD data
    #   Take the mean average radial massflux for each step [kg/m/s]
    flow_radial_core_edge = (
        np.diff(np.cumsum(massflow_core_edge_dl * dz)[::N_bin], append=np.nan) / N_bin
    )
    flow_radial_core_edge[-1] = 0

    # Balance radial flows such that changes in core flow are compensated
    flow_radial_edge_core = flow_radial_core_edge - flow_radial_net

    return (
        z,
        dz,
        idx_attach,
        flow_axial_tot,
        flow_axial_core,
        flow_axial_edge,
        flow_axial_core_neg,
        flow_axial_core_pos,
        flow_radial_core_edge,
        flow_radial_edge_core,
    )


def run(
    mechanism: Any,
    X0: dict,
    pressure: float,
    z: np.ndarray,
    idx_attach: int,
    massflow_tot: float,
    P_core: np.ndarray,
    flow_axial_core: np.ndarray,
    flow_axial_edge: np.ndarray,
    flow_radial_core_edge: np.ndarray,
    flow_radial_edge_core: np.ndarray,
):
    dz = np.diff(z)[0]

    # Create a gas object and set initial conditions
    gas = ct.Solution(mechanism)
    gas.TPX = 300.0, pressure, X0

    # Create a reservoir as the H2 source
    inlet_reservoir = ct.Reservoir(gas, name="inlet_reservoir")

    amb = ct.Solution(mechanism)
    amb.TP = 300.0, ct.one_atm
    amb_res = ct.Reservoir(amb, name="amb_reservoir")

    # Create 10 consecutive constant-volume reactors
    N_axial = np.size(flow_axial_core)

    reactors_core = [
        ct.IdealGasReactor(gas, energy="on", name=f"reactor_core_{i+1}")
        for i in range(N_axial)
    ]  # Core reactors
    reactors_edge = [
        ct.IdealGasReactor(gas, energy="on", name=f"reactor_edge_{i+1}")
        for i in range(N_axial)
    ]  # Core reactors
    reactors = [*reactors_core, *reactors_edge]

    for i in range(N_axial):
        reactors_core[i].volume = 6e-3**2 * np.pi * dz  # Volume of the core reactors
        reactors_edge[i].volume = (
            (13e-3**2 - 6.5e-3**2) * np.pi * dz
        )  # Volume of the edge reactors

        # Add heat loss to the edge reactor
        ct.Wall(reactors_edge[i], amb_res, A=13e-3 * np.pi * 2 * dz, U=1e1)

        # Add conductive heat transfer from core to edge
        # ct.Wall(reactors_core[i], reactors_edge[i], A=6.5e-3*np.pi*2*dz, U=1e2)  # Wall between core and edge reactors

        # Add power input to the core reactors
        if P_core[i] > 0:
            ct.Wall(
                amb_res,
                reactors_core[i],
                A=1,
                U=0,
                Q=P_core[i],
                name=f"wall_core_{i+1}",
            )

    # Connect the inlet reservoir to the first reactor with a MassFlowController
    mfc_in_core = ct.MassFlowController(
        upstream=inlet_reservoir, downstream=reactors_edge[idx_attach]
    )
    mfc_in_core.mass_flow_rate = massflow_tot
    # mfc_in_core = ct.MassFlowController(upstream=inlet_reservoir, downstream=reactors_edge[idx_attach])
    # mfc_in_core.mass_flow_rate = flow_axial_core[-1]
    # mfc_in_edge = ct.MassFlowController(upstream=inlet_reservoir, downstream=reactors_edge[idx_attach])
    # mfc_in_edge.mass_flow_rate = flow_axial_edge[-1]

    # Connect the reactors in series with MassFlowControllers
    for i in range(N_axial):
        # Connect core to edge reactors with radial flow

        if flow_radial_edge_core[i] > 0:
            mfc_radial_edge_core = ct.MassFlowController(
                upstream=reactors_edge[i],
                downstream=reactors_core[i],
                name=f"mfc_edge_core_{i+1}",
            )
            mfc_radial_edge_core.mass_flow_rate = flow_radial_edge_core[i]

        if flow_radial_core_edge[i] > 0:
            mfc_radial_core_edge = ct.MassFlowController(
                upstream=reactors_core[i],
                downstream=reactors_edge[i],
                name=f"mfc_core_edge_{i+1}",
            )
            mfc_radial_core_edge.mass_flow_rate = flow_radial_core_edge[i]

        if i >= N_axial - 1:
            continue

        if flow_axial_core[i] > 0:
            mfc_core_i = ct.MassFlowController(
                upstream=reactors_core[i], downstream=reactors_core[i + 1]
            )
            mfc_core_i.mass_flow_rate = flow_axial_core[i]
        elif flow_axial_core[i] < 0:
            mfc_core_i = ct.MassFlowController(
                upstream=reactors_core[i + 1], downstream=reactors_core[i]
            )
            mfc_core_i.mass_flow_rate = -flow_axial_core[i]

        mfc_edge_i = ct.MassFlowController(
            upstream=reactors_edge[i], downstream=reactors_edge[i + 1]
        )
        mfc_edge_i.mass_flow_rate = flow_axial_edge[i]

    # Create a reservoir as the exhaust
    exhaust_core = ct.Reservoir(gas)
    exhaust_edge = ct.Reservoir(gas)
    # mfc_out_core = ct.PressureController(upstream=reactors_core[-1], downstream=exhaust_core, primary=mfc_in_core)
    # mfc_out_edge = ct.PressureController(upstream=reactors_core[-1], downstream=exhaust_core, primary=mfc_in_edge)
    mfc_out_core = ct.MassFlowController(
        upstream=reactors_core[-1], downstream=exhaust_core
    )
    mfc_out_core.mass_flow_rate = flow_axial_core[-1]
    mfc_out_edge = ct.MassFlowController(
        upstream=reactors_edge[-1], downstream=exhaust_edge
    )
    mfc_out_edge.mass_flow_rate = flow_axial_edge[-1]

    # Create the reactor network
    network = ct.ReactorNet([*reactors_core, *reactors_edge])
    network.reinitialize()

    # g = network.draw()  # Returns graphviz.graphs.BaseGraph
    # g.render("output_graph", format="png", view=True)
    # return

    def fix_pressure():
        """Fix the pressure of all reactors to a specified value."""
        P_diff = np.ptp([r.thermo.P for r in reactors])
        print(f"Pressure difference: {P_diff:.2f} Pa, homogenizing...")
        for r in reactors:
            r.thermo.TP = r.thermo.T, pressure
            r.syncState()

    # Integrate the network in time
    # network.advance_to_steady_state()
    # for n in range(1):
    #     fix_pressure()
    #     network.advance(network.time + 1e-4)
    #     print(
    #         f"Time: {network.time:.4f} s, exit temperature: {reactors_core[-1].T:.1f}K (core) {reactors_edge[-1].T:.1f}K (edge)",
    #         flush=True,
    #     )

    while True:
        fix_pressure()
        network.advance_to_steady_state(max_steps=1e5)
        print(
            f"Time: {network.time:.4f} s, exit temperature: {reactors_core[-1].T:.1f}K (core) {reactors_edge[-1].T:.1f}K (edge)",
            flush=True,
        )

        P_diff = np.ptp([r.thermo.P for r in reactors])
        if P_diff < 50:
            break

    solarr_core = ct.SolutionArray(gas, extra=["z", "V"])
    solarr_edge = ct.SolutionArray(gas, extra=["z", "V"])
    for i in range(len(z)):
        solarr_edge.append(
            reactors_edge[i].thermo.state, z=z[i], V=reactors_edge[i].volume
        )
        solarr_core.append(
            reactors_core[i].thermo.state, z=z[i], V=reactors_core[i].volume
        )

    return (
        solarr_core,
        solarr_edge,
        reactors_core,
        reactors_edge,
        combined_mass_fracs(
            (solarr_core[-1], flow_axial_core[-1]),
            (solarr_edge[-1], flow_axial_edge[-1]),
        ),
    )
