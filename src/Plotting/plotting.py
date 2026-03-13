import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import seaborn as sns


def plot_heat_map(ns, ms, error_matrix, norm_f0):
    df = pd.DataFrame(error_matrix, index=ns, columns=ms)
    #Flips axis
    df = df.iloc[::-1]

    sns.set_theme(style="white")
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "figure.dpi": 300
    })

    color_map = cm.get_cmap("viridis_r").copy()
    color_map.set_over("darkred")

    plt.figure(figsize=(8,6))
    sns.heatmap(
        df,
        cmap=color_map,
        vmin=0,
        vmax=norm_f0,
        cbar_kws={
            "label": "Relative reconstruction Error",
            "extend": "max"
    })

    plt.xlabel("Oversampling ratio (m/n)")
    plt.ylabel("Dimension (n)")

    plt.tight_layout()
    plt.show()

def plot_heat_map_norm(norms, ms, error_matrix, norm_f0):
    df = pd.DataFrame(error_matrix, index=norms, columns=ms)
    #Flips axis
    df = df.iloc[::-1]

    sns.set_theme(style="white")
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "figure.dpi": 300
    })

    color_map = cm.get_cmap("viridis_r").copy()
    color_map.set_over("darkred")

    plt.figure(figsize=(8,6))
    sns.heatmap(
        df,
        cmap=color_map,
        vmin=0,
        vmax=1.2,
        cbar_kws={
            "label": "Relative reconstruction Error",
            "extend": "max"
    })

    plt.xlabel("Oversampling ratio (m/n)")
    plt.ylabel("Signal intensity (|f_0|)")

    plt.tight_layout()
    plt.show()

def plot_errors(errors: list[float]):
    plt.hist(errors, bins=200)
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Reconstruction Errors')
    plt.show()
    return

def plot_4_errors(errors: list[list[float]]):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    sns.histplot(errors[0], bins=200, kde=True, ax=axes[0, 0])
    axes[0, 0].set_title('norm f_0 is 1')

    sns.histplot(errors[1], bins=200, kde=True, ax=axes[0, 1])
    axes[0, 1].set_title('norm f_0 is 5')

    sns.histplot(errors[2], bins=200, kde=True, ax=axes[1, 0])
    axes[1, 0].set_title('norm f_0 is 10')

    sns.histplot(errors[3], bins=200, kde=True, ax=axes[1, 1])
    axes[1, 1].set_title('norm f_0 is 50')

    plt.tight_layout()
    plt.show()

def plot_4_ranges(ranges):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    if ranges[0]:
        min0, max0 = zip(*ranges[0])
    else:
        max0 = []

    sns.histplot(max0, bins=20, kde=True, ax=axes[0,0], legend=False)
    axes[0, 0].set_title('intensity is 1')

    min1, max1 = zip(*ranges[1])

    sns.histplot(max1, bins=20, kde=True, ax=axes[0,1], legend=False)
    axes[0, 1].set_title('Intensity is 5')

    min2,max2 = zip(*ranges[2])

    sns.histplot(max2, bins=20, kde=True, ax=axes[1,0], legend=False)
    axes[1, 0].set_title('Intensity is 10')

    min3, max3 = zip(*ranges[3])

    sns.histplot(max3, bins=20, kde=True, ax=axes[1,1], legend=False)
    axes[1, 1].set_title('Intensity is 50')

    plt.tight_layout()
    plt.show()

def plot_ranges(ranges):
    min_quotient, max_quotient = zip(*ranges)

    sns.histplot(list(filter(lambda x: x < 2, min_quotient)), bins=200, kde=True, label='Min Error', color='blue', alpha=0.5)
    sns.histplot(max_quotient, bins=200, kde=True, label='Max Error', color='red', alpha=0.5)

    plt.xlabel('Logarithm ratio')
    plt.title('Distribution of Min and Max of logarithm inside')
    plt.legend()
    plt.show()
    return


def plot_average_sim(ns: list[int], ms: list[int], errors: list[list[float]]):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()  # makes indexing easier: axes[0], axes[1], etc.

    for idx, n in enumerate(ns):
        axes[idx].plot(ms, errors[idx])
        axes[idx].set_title(f'n = {n}')
        axes[idx].set_xlabel('m')
        axes[idx].set_ylabel('Error')

    plt.tight_layout()
    plt.show()