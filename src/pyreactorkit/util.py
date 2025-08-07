import cantera as ct
import toddler
import numpy as np
import matplotlib.pyplot as plt


def homogenize_dicts(dict1: dict, dict2: dict):
    keys = set(dict1.keys()).union(set(dict2.keys()))

    dict1 = {k: dict1.get(k, 0) for k in keys}
    dict2 = {k: dict2.get(k, 0) for k in keys}

    return dict1, dict2


def kgps_to_slm(kgps, X, chemistry_set):
    """Convert kg/s to SLM."""
    solution = ct.Solution(chemistry_set)
    solution.X = X

    return toddler.physics.flow.massflux_to_slm(kgps, solution.mean_molecular_weight)


def slm_to_kgps(slm, X, chemistry_set):
    """Convert SLM to kg/s."""
    solution = ct.Solution(chemistry_set)
    solution.X = X

    return toddler.physics.flow.slm_to_massflux(slm, solution.mean_molecular_weight)


def calculate_conversion(Y0, Y):
    return (Y0 - Y) / Y0


def calculate_conversion_from_states(s0, s1, species_name):
    idx = s0.species_index(species_name)
    return calculate_conversion(s0.Y[idx], s1.Y[idx])


def calculate_elements(state, element):
    element_number = np.array(
        [
            state.species(i).composition.get(element, 0)
            for i in range(len(state.species()))
        ]
    )

    return element_number


def mass_frac_dict(state):
    """Return a dictionary of mass fractions for each species in the state."""
    return {s: state.Y[i] for i, s in enumerate(state.species_names)}


def combined_mass_fracs(*states_and_weights):
    """Combine mass fractions from multiple states with given weights."""
    combined = {}
    total_weight = 0

    for state, weight in states_and_weights:
        mass_fracs = mass_frac_dict(state)
        for species, frac in mass_fracs.items():
            combined[species] = combined.get(species, 0) + frac * weight
        total_weight += weight

    # Normalize the combined mass fractions
    if total_weight > 0:
        for species in combined:
            combined[species] /= total_weight

    return combined


def calculate_selectivity(
    state, element="C", hide_species=[], exclude_species=["CH4"], threshold=1e-3
):
    species_names = state.species_names

    # Calculate the number of atoms of the element in each species
    element_number = calculate_elements(state, element)

    # Calculate the selectivity of the element
    element_yield = state.X * element_number

    idx_threshold = element_yield > threshold
    idx_exclude = [species_names.index(species) for species in exclude_species]
    idx_exclude = np.isin(np.arange(element_yield.size), idx_exclude)

    element_yield = element_yield[~idx_exclude & idx_threshold]
    element_yield /= np.sum(element_yield)
    species_names = np.array(species_names)[~idx_exclude & idx_threshold]

    idx_hide = np.isin(species_names, hide_species)
    element_yield = element_yield[~idx_hide]
    species_names = species_names[~idx_hide]

    return {s: y for s, y in zip(species_names, element_yield)}


def cos_profile(z, z0, fwhm, y0=1, sum=None):
    y = y0 * np.cos((z - z0) / fwhm * np.pi / 4) ** 2
    y[np.abs(z - z0) > 2 * fwhm] = 0
    if sum is not None:
        y *= sum / np.sum(y)

    return y


def cos_profile_binned(z, z0, fwhm, y0=1, sum=None):
    """
    Same as cos_profile, but works for data where dz >> fwhm.
    """
    assert np.std(np.diff(z)) / np.mean(np.diff(z)) < 1e-7, "z must be evenly spaced"
    dz = np.mean(np.diff(z))
    N_subsample = 100
    dz_subsample = dz / 100
    z_subsample = np.arange(z[0] - dz / 2, z[-1] + dz / 2, dz_subsample)[
        : (N_subsample) * z.size
    ]
    y_subsample = cos_profile(z_subsample, z0, fwhm, y0=y0, sum=sum)

    y = np.sum(np.reshape(y_subsample, (-1, N_subsample)), axis=1)

    return y


def gauss_profile(z, z0, fwhm, y0=1):
    y = y0 * np.exp(-((z - z0) ** 2) / (2 * fwhm**2))

    return y


if __name__ == "__main__":
    z = np.linspace(0, 10, 1000)
    y = cos_profile(z, 4, 1)

    plt.figure()
    plt.plot(z, y)
    plt.show()
