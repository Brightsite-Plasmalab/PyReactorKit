import matplotlib.pyplot as plt
import numpy as np
import cantera as ct
from pyreactorkit.display import *


def get_equilibrium_composition(X0, pressure, T, T_margin=1e-2):
    gas = ct.Solution("gri30.yaml")
    gas.basis = "molar"

    # Calculate roomtemperature enthalpy
    gas.TPX = 300, pressure, X0
    h_roomtemp = gas.enthalpy_mole

    # Iterate to find equilibrium composition
    dT = 0
    while True:
        gas.TPX = gas.T + dT / 2, gas.P, gas.X
        gas.equilibrate("TP")

        # Check if temperature is close enough
        dT = T - gas.T
        if abs(dT) < T_margin:
            break

    h_final = gas.enthalpy_mole
    dh = h_final - h_roomtemp
    return gas, dh


def get_temperature(X0, X, pressure, h, mode="mass"):
    gas = ct.Solution("gri30.yaml")
    gas.basis = mode
    gas.TPX = 300, pressure, X0
    h_roomtemp = gas.h

    while True:
        dh = h - (gas.h - h_roomtemp)
        if abs(dh) / h < 1e-2 or h == 0:
            break

        dT = dh / gas.cp
        gas.TPX = gas.T + dT, gas.P, X

    return gas.T


def interpolate(X_ref, X, T, pressure, h_target, mode="mass"):
    """
    Find a composition in X with an enthalpy closest to h_target, relative to the reference composition X_ref at the given (T, pressure).
    """
    gas = ct.Solution("gri30.yaml")
    gas.basis = mode
    gas.TPX = 300, pressure, X_ref
    h_roomtemp = gas.h

    parametrized = [x for x in X.values() if type(x) == np.ndarray][0]
    h = np.zeros_like(parametrized)

    for i in range(len(parametrized)):
        Xi = X.copy()

        # Loop over all key-value pairs in Xi
        # if the value is an array, replace it with the ith element
        for key, value in Xi.items():
            if type(value) == np.ndarray:
                Xi[key] = value[i]

        # Calculate enthalpy
        gas.TPX = T, pressure, Xi
        h[i] = gas.h

    h = h - h_roomtemp

    id_closest = np.argmin(np.abs(h - h_target))

    X_closest = X.copy()
    for key, value in X_closest.items():
        if type(value) == np.ndarray:
            X_closest[key] = value[id_closest]

    return X_closest


if __name__ == "__main__":
    X0 = {"H2": 30, "Ar": 0, "CH4": 5}
    X0 = {"H2": 0, "Ar": 1, "CH4": 0}
    gas = ct.Solution("gri30.yaml")
    gas.X = X0

    print(
        get_temperature(
            X0,
            X0,
            200e2,
            toddler.physics.energy.SEI_J_per_g(1000, 35, gas.mean_molecular_weight)
            * 1e3,
        )
    )
    # X0 = {"H2": 1, "Ar": 1}
    # pressure = 200e2
    # T = 3500
    # gas, dh = get_equilibrium_composition(X0, pressure, T)
    # h = dh / 1e6  # [kJ/mol]
    # h_slm = h / 100 * 130  # W/slm
    # print(f"{h:.2e} kJ/mol = {h_slm:.0f} W/slm")
    # plot_yield(gas)
    # plt.show()
    diss = np.linspace(0, 1, 100)
    X_ref = {"H2": 1, "Ar": 20}
    X = {"H2": X_ref["H2"] * (1 - diss), "Ar": X_ref["Ar"], "H": 2 * diss}
    pressure = 200e2
    T = 300

    gas = ct.Solution("gri30.yaml")
    gas.X = X_ref
    h_mass = (
        toddler.physics.energy.SEI_J_per_g(1000, 30, gas.mean_molecular_weight) * 1e3
    )

    X_closest = interpolate(
        X_ref,
        X,
        T,
        pressure,
        h_mass,
    )
    print(X_closest)
