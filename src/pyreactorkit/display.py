import distinctipy
import matplotlib.pyplot as plt
import numpy as np
import re
import cantera as ct
import toddler
from pyreactorkit.util import calculate_elements


def print_species(solution, threshold_Y=1e-3, concat=", ", prefix="mass fractions:"):
    idx_significant = solution.Y > threshold_Y
    fracs = zip(solution.species_names, solution.Y)

    fracs = {k: f"{v:.3f}" for i, (k, v) in enumerate(fracs) if idx_significant[i]}

    print(f"{prefix}{concat.join( f'{k}: {v}' for k, v in fracs.items())}")


def species_subscript(species):
    return re.sub(r"([0-9]+)", r"$_{\1}$", species)


def format_reaction(reaction):
    reaction_label = reaction.replace("<=>", r"\Leftrightarrow")  # Make nice <=>
    reaction_label = reaction_label.replace("=>", r"\Rightarrow")  # Make nice <=>
    reaction_label = reaction_label.replace("<=", r"\Leftarrow")  # Make nice <=>
    # Replace numbers preceeded by letters with subscripts
    reaction_label = re.sub(r"([a-zA-Z]+)([0-9]+)", r"\1_{\2}", reaction_label)
    # Add math mode
    reaction_label = rf"${reaction_label}$"
    return reaction_label


def reverse_reaction(reaction):
    # Swap the reactants and products
    reaction = re.sub(r"(.+)\s(<=>|=>|<=)\s(.+)", r"\3 \2 \1", reaction)
    # Swap the arrow direction
    reaction = re.sub(
        r"(?<!<|>)(<=|=>)(?!<|>)",
        lambda m: "<=" if m.group(1) == "=>" else "=>",
        reaction,
    )

    return reaction


def get_x_axis(solarr, xaxis=None):
    if xaxis is None:
        xaxis = "t"
    if xaxis == "z":
        xaxis_label = "z [mm]"
        xaxis_multiplier = 1e3
        x = solarr.z * xaxis_multiplier
    elif xaxis == "t":
        xaxis_label = "t [ms]"
        xaxis_multiplier = 1e3
        x = solarr.t * xaxis_multiplier
    elif xaxis == "i":
        xaxis_label = ""
        x = np.arange(solarr.size)
    else:
        xaxis_label = None
        x = getattr(solarr, xaxis)
    return x, xaxis_label


def get_y_axis(solarr, yaxis=None):
    if yaxis is None:
        yaxis = "mole"

    if yaxis == "mole":
        yaxis_label = r"Mole fraction [$\endash$]"
        y = solarr.X
    elif yaxis == "molarflow":
        yaxis_label = r"Molar flow rate [mmol/s]"

        flowrate_molps = solarr.flowrate / (
            solarr.mean_molecular_weight / 1e6
        )  # [mol/s]
        flowrate_molps = np.reshape(
            flowrate_molps, (solarr.size, 1) if solarr.size > 1 else (solarr.size,)
        )

        y = solarr.X * flowrate_molps
    elif yaxis == "mass":
        yaxis_label = r"Mass fraction [$\endash$]"
        y = solarr.Y
    elif yaxis == "massflow":
        flowrate_kgps = np.reshape(
            solarr.flowrate, (solarr.size, 1) if solarr.size > 1 else (solarr.size,)
        )  # [kg/s]
        yaxis_label = r"Massflow [g/s]"
        y = solarr.Y * flowrate_kgps * 1e3
    elif yaxis == "yield":
        flowrate_kgps = np.reshape(
            solarr.flowrate, (solarr.size, 1) if solarr.size > 1 else (solarr.size,)
        )  # [kg/s]
        yaxis_label = r"Yield [$\endash$]"
        y = solarr.Y * flowrate_kgps * 1e3
        y /= np.sum(y)
    elif yaxis == "yield_c":
        yaxis_label = r"Yield [$\endash$]"
        y = solarr.Y
        C_atoms = calculate_elements(solarr, "C")
        y *= C_atoms
        y /= np.sum(y)
    elif yaxis == "molefrac":
        yaxis_label = r"Mole fraction [$\endash$]"
        y = solarr.X
    else:
        raise ValueError(f"Unknown yaxis: {yaxis}")
    return y, yaxis_label


def plot_species(
    solarr,
    threshold=1e-4,
    xaxis=None,
    ax=None,
    yaxis="molefrac",
    fig_opts={},
    legend_opts={},
    species=None,
):
    x, xaxis_label = get_x_axis(solarr, xaxis)
    y, yaxis_label = get_y_axis(solarr, yaxis)

    if ax is None:
        plt.figure(**fig_opts)
        ax = plt.gca()

    y_threshold = np.max(y) * threshold
    idx_display = np.max(y, axis=0) > y_threshold
    idx_color = np.cumsum(idx_display) - 1

    colors = distinctipy.get_colors(np.sum(idx_display), rng=0)

    if type(species) == str:
        species = [species]

    for i, species_i in enumerate(solarr.species_names):
        y_i = y[:, i]

        if not idx_display[i]:
            continue

        if not species is None:
            if species_i not in species:
                continue

        # print(f"{species}: {np.max(molefrac):.2e} ")
        ax.plot(x, y_i, label=species_subscript(species_i), color=colors[idx_color[i]])

    ax.set_ylim(np.max(y) * threshold, np.max(y))
    # ax.set_yscale("log")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), **legend_opts)
    ax.set_xlabel(xaxis_label)
    ax.set_ylabel(yaxis_label)


def plot_temperature(solarr, xaxis="z", ax=None, **plot_opts):
    x, xaxis_label = get_x_axis(solarr, xaxis)

    if ax is None:
        plt.figure()
        ax = plt.gca()

    ax.plot(x, solarr.T, **plot_opts)
    ax.set_xlabel(xaxis_label)
    ax.set_ylabel("Temperature [K]")


def plot_production_rates(
    solarr,
    rate_threshold_rel=1e-3,
    xaxis=None,
    ax=None,
    species=None,
    cum=False,
    dV=None,
):
    # Pass multiple parallel SolutionArrays in a list
    if type(solarr) == list or type(solarr) == tuple:
        rates = [si.net_production_rates for si in solarr]
        if type(dV) == list or type(dV) == tuple:
            rates = [rates[i] * dV[i] for i in range(len(solarr))]
        rates = np.sum(rates, axis=0)  # Sum over all SolutionArrays
        solarr = solarr[0]  # Use the first SolutionArray for x-axis and species names
    elif dV is not None:
        rates = solarr.net_production_rates * dV
    else:
        rates = solarr.net_production_rates

    x, xaxis_label = get_x_axis(solarr, xaxis)
    if cum:
        rates = np.cumsum(rates, axis=0)

    rate_max = np.max(np.abs(rates))
    rate_threshold = rate_max * rate_threshold_rel

    idx_display = np.max(np.abs(rates), axis=0) > rate_threshold
    idx_color = np.cumsum(idx_display) - 1
    colors = distinctipy.get_colors(np.sum(idx_display), rng=0)

    if ax is None:
        plt.figure()
        ax = plt.gca()
    ax.set_prop_cycle(color=plt.cm.tab20.colors)

    if type(species) == str:
        species = [species]

    for i, species_i in enumerate(solarr.species_names):
        production_rate = rates[:, i]

        if not idx_display[i]:
            continue

        if not species is None:
            if species_i not in species:
                continue

        # print(f"{species}: {np.max(molefrac):.2e} ")
        ax.plot(
            x,
            production_rate,
            label=species_subscript(species_i),
            color=colors[idx_color[i]],
        )

    # ax.set_ylim(np.min(rates), np.max(rates))
    # ax.set_yscale("log")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_xlabel(xaxis_label)
    if not cum:
        ax.set_ylabel("Production rates [$kmol/m^3/s$]")
    else:
        ax.set_ylabel("Cumulative production rates [$kmol/s$]")

    return {
        species_name: rates[:, i]
        for i, species_name in enumerate(solarr.species_names)
        if idx_display[i]
    }


def plot_reactions(
    solarr,
    rate_threshold_rel=1e-3,
    xaxis=None,
    ax=None,
    species=None,
    cum=False,
    dV=None,
    legend_opts={},
):
    # Pass multiple parallel SolutionArrays in a list
    if type(solarr) == list or type(solarr) == tuple:
        rates = np.sum(
            [si.net_rates_of_progress for si in solarr], axis=0
        )  # Sum over all SolutionArrays
        solarr = solarr[0]  # Use the first SolutionArray for x-axis and species names
    else:
        rates = solarr.net_rates_of_progress

    x, xaxis_label = get_x_axis(solarr, xaxis)

    if cum:
        rates = np.cumsum(rates, axis=0)
        if dV is not None:
            rates *= dV

    rate_max = np.max(np.abs(rates))
    rate_threshold = rate_max * rate_threshold_rel

    idx_display = np.max(np.abs(rates), axis=0) > rate_threshold
    idx_color = np.cumsum(idx_display) - 1
    colors = distinctipy.get_colors(np.sum(idx_display), rng=0)

    if ax is None:
        plt.figure()
        ax = plt.gca()
    ax.set_prop_cycle(color=plt.cm.tab20.colors)

    if type(species) == str:
        species = [species]

    for i, reaction in enumerate(solarr.reaction_equations()):
        reaction_rate = rates[:, i]

        if not idx_display[i]:
            continue

        # Filter species
        # species_i = re.findall(
        #     r"(?:(?<=^)|(?<=\s|\+))([a-zA-Z]+[a-zA-Z0-9]*)(?=[\s)])", reaction
        # )
        species_i = re.findall(
            r"(?:(?<=^)|(?<=\s|\+))([a-zA-Z]+[a-zA-Z0-9]*)", reaction
        )
        if species is not None:
            if not any([s in species_i for s in species]):
                continue

        if np.max(reaction_rate) < np.max(-reaction_rate):
            # Flip the reaction for easier display
            reaction = reverse_reaction(reaction)
            reaction_rate = -reaction_rate

        reaction_label = format_reaction(reaction)

        # Plot forward
        reaction_rate_forward = np.copy(reaction_rate)
        reaction_rate_forward[reaction_rate_forward < 0] = np.nan
        line_i = ax.plot(
            x,
            reaction_rate_forward,
            label=None,
            linestyle="-",
            color=colors[idx_color[i]],
        )
        ax.plot(
            x[:2],
            [rate_threshold, np.nan],
            label=reaction_label,
            linestyle="-",
            color=line_i[0].get_color(),
        )

        # Plot reverse
        reaction_rate_backward = -np.copy(reaction_rate)
        reaction_rate_backward[reaction_rate_backward < 0] = np.nan
        ax.plot(
            x,
            reaction_rate_backward,
            label=None,
            linestyle="--",
            color=line_i[0].get_color(),
        )

    ax.set_yscale("log")
    ax.set_ylim(rate_max * rate_threshold_rel, rate_max)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), **legend_opts)

    ax.set_xlabel(xaxis_label)
    if not cum:
        ax.set_ylabel("Reaction rates [$kmol/m^3/s$]")
    else:
        ax.set_ylabel("Cumulative reaction rates [$kmol/s$]")


def plot_yield(
    state, threshold=1e-2, ax=None, xaxis=None, yaxis=None, exclude_species=None
):
    # x, xaxis_label = get_x_axis(state, xaxis)
    y, yaxis_label = get_y_axis(state, yaxis)

    if ax is None:
        plt.figure()
        ax = plt.gca()
    if exclude_species is None:
        exclude_species = []

    species_names = state.species_names
    idx_excluded = np.isin(
        np.arange(len(species_names)),
        np.array(
            [i for i, species in enumerate(species_names) if species in exclude_species]
        ),
    )
    y_excluded = np.sum(y[idx_excluded])
    y_max = np.max(y)
    y_fac = 1 / (1 - y_excluded)

    for i, species in enumerate(state.species_names):
        if species in exclude_species:
            continue
        y_i = y[i]
        if y_i < np.max(y[~idx_excluded]) * threshold:
            continue
        y_i *= y_fac

        bar = ax.bar(species, y_i, label=species)

        label_above = y_i < y_max / 20
        ax.bar_label(
            bar,
            label_type="center" if not label_above else "edge",
            color="white" if not label_above else "k",
            fmt="%.2f",
        )

    # ax.set_xlabel(xaxis_label)
    ax.set_ylabel(yaxis_label)


def calculate_yield(
    state, element="C", threshold=1e-2, exclude_species=[], hide_species=[]
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

    return species_names, element_yield


def bar_yield(
    state,
    element="C",
    threshold=1e-2,
    exclude_species=[],
    hide_species=[],
    x=" ",
    colors=None,
):
    if colors is None:
        colors = {}
    if type(state) == ct.composite.SolutionArray and state.size != 1:
        raise AssertionError("Only single state supported")
    elif type(state) not in [ct.Solution, ct.composite.SolutionArray]:
        raise ValueError(f"Unknown state type { type(state) }")

    species_names, element_yield = calculate_yield(
        state, element, threshold, exclude_species, hide_species
    )

    element_yield *= 100  # Convert to percentage

    if len(colors) == 0:  # Todo: replace with more advanced method of extending colors
        colors = dict(
            zip(species_names, distinctipy.get_colors(len(species_names), rng=0))
        )

    # Stacked bar chart of the element yields
    for i, species in enumerate(species_names):
        color_i = colors.get(species, None)
        bar = plt.bar(
            x,
            height=element_yield[i],
            bottom=np.cumsum(element_yield)[i - 1] if i > 0 else 0,
            label=species,
            color=color_i,
        )

        if color_i is None:
            colors[species] = bar.patches[0].get_facecolor()
    plt.ylabel(f"{element} yield [%]")

    plt.legend()

    return colors
