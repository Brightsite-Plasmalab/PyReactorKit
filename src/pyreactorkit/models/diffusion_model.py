import numpy as np
import cantera as ct


class Cell:
    V: float  # [m^3]
    A: float  # [m^2]
    L: float  # [m]
    gas: ct.Solution
    gas_adj: ct.Solution

    def __init__(self, V, A, L, gas, gas_adj):
        self.V = V
        self.A = A
        self.L = L
        self.gas = gas
        self.gas_adj = gas_adj

    def calculate_production_rate(self):
        ri = self.gas.net_production_rates  # [kmol/m^3/s]
        dNi = ri * self.V  # [kmol/s]

        return dNi

    def calculate_diffusion_rate(self):
        Dij = self.gas.binary_diff_coeffs  # [m^2/s]

        id_i_eq_j = np.eye(Dij.shape[0], dtype=np.bool)
        Dij[id_i_eq_j] = 0
        Di = (1 - self.gas.X) / np.nansum(self.gas_adj.X / Dij, axis=1)  # [m^2/s]

        # Di = np.nansum(self.gas_adj.X * Dij, axis=1)  # [m^2/s]

        gradient = (self.gas.X - self.gas_adj.X) / self.L  # [m^-1]
        gradient_mole = gradient * self.gas.density_mole  # [kmol/m^4]
        flux = -Di * gradient_mole  # [kmol/m^2/s]

        dNi = flux * self.A  # [kmol/s]

        return dNi

    def calculate_residual_rate(self, production, diffusion, convection):
        # Balance a loss of number of moles in the cell
        residual_rate = -np.sum(production + diffusion + convection)  # [kmol/s]

        residual_balanced = (
            residual_rate * self.gas_adj.X / np.sum(self.gas_adj.X)
        )  # [kmol/s]
        return residual_balanced

    def calculate_convection_rate(self):
        # Calculate the convection rate
        dNi = self.gas.X * 0
        return dNi

    def calculate_rate(self):
        production = self.calculate_production_rate()
        diffusion = self.calculate_diffusion_rate()
        convection = self.calculate_convection_rate()
        residual = self.calculate_residual_rate(production, diffusion, convection)

        dNi_tot = production + diffusion + convection + residual
        return dNi_tot

    def update(self, dt):
        N = self.gas.X * self.V
        N += self.calculate_rate() * dt
        self.gas.X = N / self.V

    def converge(self, dt, tol=1e-6):
        dNi_tot = self.calculate_rate()
        while np.max(np.abs(dNi_tot)) > tol:
            self.update(dt)
            dNi_tot = self.calculate_rate()
