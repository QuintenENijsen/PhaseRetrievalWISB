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
        "figure.dpi": 300,
    })

    color_map = cm.get_cmap("viridis_r").copy()
    color_map.set_over("darkred")

    plt.figure(figsize=(8,6))
    sns.heatmap(
        df,
        cmap=color_map,
        vmin=0,
        vmax=1,
        cbar_kws={
            "label": "Relative reconstruction Error",
            "extend": "max"
    })

    plt.xlabel(r"Oversampling ratio $\frac{m}{n}$")
    plt.ylabel(r"Dimension ($n$)")

    plt.tight_layout()
    plt.savefig("truncwf.png")

def plot_heat_map_norm(norms, ms, error_matrix, norm_f0):
    df = pd.DataFrame(error_matrix, index=norms, columns=ms)
    #Flips axis
    df = df.iloc[::-1]

    sns.set_theme(style="white")
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "figure.dpi": 300,
        "text.usetex": True,
        "font.family": "Serif",
    })

    color_map = cm.get_cmap("viridis_r").copy()
    color_map.set_over("darkred")

    plt.figure(figsize=(8,6))
    sns.heatmap(
        df,
        cmap=color_map,
        vmin=0,
        vmax=norm_f0 * 1.1,
        cbar_kws={
            "label": r"Relative reconstruction Error",
            "extend": r"max"
    })

    plt.xlabel(r"Oversampling ratio $\frac{m}{n}$")
    plt.ylabel(r"Signal intensity $|f_0|$")

    plt.tight_layout()
    plt.show()

def plot_heat_map_truncate(norms, alpha_fs, error_matrix):
    df = pd.DataFrame(error_matrix, index=norms, columns=alpha_fs)
    #Flips axis
    df = df.iloc[::-1]

    sns.set_theme(style="white")
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "figure.dpi": 300,
    })

    color_map = cm.get_cmap("viridis_r").copy()
    #color_map.set_over("darkred")

    plt.figure(figsize=(8,6))
    sns.heatmap(
        df,
        cmap=color_map,
        vmin=0,
        vmax=1,
        cbar_kws={
            "label": r"truncation rate",
            "extend": r"max"
    })

    plt.xlabel(r"Oversampling ratio $\frac{m}{n}$")
    plt.ylabel(r"Signal intensity ($|f_0|$)")

    plt.tight_layout()
    plt.savefig('spectralTrunc.png')

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


def plot_heat_map_genmodel(ks, ms, error_matrix, norm_f0):
    df = pd.DataFrame(error_matrix, index=ks, columns=ms)
    #Flips axis
    df = df.iloc[::-1]

    sns.set_theme(style="white")
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "figure.dpi": 300,
    })

    color_map = cm.get_cmap("viridis_r").copy()
    color_map.set_over("darkred")

    plt.figure(figsize=(8,6))
    sns.heatmap(
        df,
        cmap=color_map,
        vmin=0,
        vmax=1.5,
        cbar_kws={
            "label": "Relative reconstruction Error",
            "extend": "max"
    })

    plt.xlabel(r"Oversampling ratio $\frac{m}{k}$")
    plt.ylabel(r"Latent dimension $k$")

    plt.tight_layout()
    plt.savefig("genmodel.png")


def plot_error_per_dim(k: int, ms: list[float], st_devs: list[float], errors: list[float]):
    plt.figure(figsize=(8, 6))

    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "figure.dpi": 300,
    })

    plt.errorbar(
        ms, errors,
        yerr=st_devs,
        marker="o", linestyle="-", linewidth=2, markersize=6,
        capsize=4, capthick=1.5, elinewidth=1.5,
    )

    plt.xlabel(r"Oversampling ratio $\frac{m}{n}$")
    plt.ylabel("Relative reconstruction Error")

    plt.tight_layout()
    plt.savefig(f"error_dim_{k}.png")

if __name__ == "__main__":
    plot_heat_map_genmodel([112, 224, 336, 448, 560, 672, 784], [1, 2, 3, 4, 5, 6, 7, 8],
                            [[1.3083185923099518, 1.1039867236614227, 0.9321447387039662, 0.5859903173446656, 0.4115332080423832, 0.3170000152885914, 0.2855080816447735, 0.2666618231534958],
                             [1.4816092915534973, 1.2106402312517166, 1.0388724119067192, 0.547244853079319, 0.24086210653185844, 0.18111578917503357, 0.16720361071825027, 0.16405471546947956],
                             [1.6035872986316682, 1.2879575021266938, 1.116690017580986, 0.6208424338251353, 0.2029219942688942, 0.12124735492467881, 0.10398362569510937, 0.09838844271749259],
                             [1.7162069187164306, 1.3477535381317138, 1.15727478992939, 0.7834133118912577, 0.14347237111255526, 0.08678772544488311, 0.06635093323886394, 0.05717792579159141],
                             [1.8178040475845336, 1.3964150795936585, 1.1988041439056396, 0.8985263225845993, 0.22057270063459875, 0.07426118329539895, 0.045610473200678826,0.043616054270416496],
                             [1.9134050116539, 1.434678510427475, 1.2367589644193648, 1.0866459393203258, 0.24037968300655485, 0.08696303408965468, 0.06303918077424168, 0.03984391060099006],
                             [2.0182446670532226, 1.4669103610515595, 1.2631011241674424, 1.1330535722076893, 0.40240978986024856, 0.0824063800983131, 0.05624764955043793, 0.04353999085351825]]
                           ,1)