import matplotlib.pyplot as plt
import seaborn as sns


def set_style():
    sns.set_theme(style="ticks")
    sns.set_context("paper")
    sns.set_palette("Dark2")
    plt.rcParams["figure.figsize"] = [5.4, 3.6]
    plt.rcParams["figure.dpi"] = 200
    plt.rcParams["font.size"] = 14
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 12
    plt.rcParams["legend.title_fontsize"] = 14
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["axes.grid"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.bottom"] = True
    plt.rcParams["axes.spines.left"] = True
    plt.rcParams["axes.axisbelow"] = True
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["xtick.minor.visible"] = True
    plt.rcParams["ytick.minor.visible"] = True
    plt.rcParams["lines.linewidth"] = 1.5
    plt.rcParams["lines.markersize"] = 10
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.transparent"] = False
