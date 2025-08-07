from pyreactorkit.models import HeatedPlugFlowReactor, HeatedMixingPlugFlowReactor
from pyreactorkit.util import slm_to_kgps
import numpy as np


def simulate_split_reactor(
    P, p, X0, massflow_total, chemistry_set, massflow_core_frac=0.15
):
    z = np.linspace(0, 30e-3, 30)

    # Part one: simulate the core
    massflow_total_kgps = slm_to_kgps(massflow_total, X0, chemistry_set=chemistry_set)

    reactor = HeatedPlugFlowReactor(
        T0=300,
        pressure=p,
        mass_flow_rate_kgps=massflow_total_kgps * massflow_core_frac,
        X0=X0,
        z=z,
        dz=np.diff(z)[0],
        chemistry_set=chemistry_set,
    )
    # reactor.A_crosssection = (np.pi * 5e-3**2)

    heating_profile = (z < 10e-3) * 1.0
    # heating_profile = np.exp(-(z-5e-3)**2 / 3e-3**2)
    # heating_profile = np.ones_like(z) * (z < 5e-3)
    heating_profile *= P / np.sum(heating_profile)
    reactor.set_heating_profile(heating_profile=heating_profile, U_heatloss=0)

    solarr, _, _ = reactor.simulate()
    print(f"Max temperature: {np.max(solarr.T):.0f} K")

    # Alternative: simulate mixing the other way
    z2 = np.linspace(30e-3, 300e-3, 50)
    reactor2 = HeatedMixingPlugFlowReactor(
        T0=solarr[-1].T,
        pressure=p,
        mass_flow_rate_kgps=solarr[-1].flowrate,
        X0=solarr[-1].X,
        z=z2,
        dz=np.diff(z2)[0],
        chemistry_set=chemistry_set,
    )

    mixing_profile_kgps = (z2 < 100e-3) * 1.0
    mixing_profile_kgps *= (massflow_total_kgps - solarr[-1].flowrate) / np.sum(
        mixing_profile_kgps
    )

    reactor2.set_mixing_profile(300, p, X0, mixing_profile_kgps=mixing_profile_kgps)
    reactor2.set_heating_profile(np.zeros_like(z2), U_heatloss=1e2)

    solarr2, _, _ = reactor2.simulate()

    return solarr, solarr2


def simulate_split_reactor2(P, p, X0, massflow_total, chemistry_set):
    z = np.linspace(0, 30e-3, 30)

    # Part one: simulate the core
    massflow_total = 20
    massflow_core_frac = 0.15
    massflow_total_kgps = slm_to_kgps(massflow_total, X0, chemistry_set=chemistry_set)

    reactor = HeatedPlugFlowReactor(
        T0=300,
        pressure=p,
        mass_flow_rate_kgps=massflow_total_kgps * massflow_core_frac,
        X0=X0,
        z=z,
        dz=np.diff(z)[0],
        chemistry_set=chemistry_set,
    )
    # reactor.A_crosssection = (np.pi * 5e-3**2)

    heating_profile = (z < 10e-3) * 1.0
    # heating_profile = np.exp(-(z-5e-3)**2 / 3e-3**2)
    # heating_profile = np.ones_like(z) * (z < 5e-3)
    heating_profile *= P / np.sum(heating_profile)
    reactor.set_heating_profile(heating_profile=heating_profile, U_heatloss=0)

    solarr, _, _ = reactor.simulate()
    print(f"Max temperature: {np.max(solarr.T):.0f} K")

    z2 = np.linspace(30e-3, 300e-3, 50)
    reactor2 = HeatedMixingPlugFlowReactor(
        T0=300,
        pressure=p,
        mass_flow_rate_kgps=massflow_total_kgps - solarr[-1].flowrate,
        X0=X0,
        z=z2,
        dz=np.diff(z2)[0],
        chemistry_set=chemistry_set,
    )

    mixing_profile_kgps = (z2 < 100e-3) * 1.0
    mixing_profile_kgps *= (solarr[-1].flowrate) / np.sum(mixing_profile_kgps)

    reactor2.set_mixing_profile(
        solarr[-1].T, p, solarr[-1].X, mixing_profile_kgps=mixing_profile_kgps
    )
    reactor2.set_heating_profile(np.zeros_like(z2), U_heatloss=1e2)

    solarr2, _, _ = reactor2.simulate()

    return solarr, solarr2
