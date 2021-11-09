import os
import os.path as osp
from copy import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
from mpl_toolkits.mplot3d import Axes3D


# set default plot configuration
plt.style.use('seaborn-paper')
plt.rc('font', family='Times New Roman')

COLOR_SET1 = plt.cm.Set1(np.linspace(0, 0.8, 8))  # color set 1
COLOR_SET2 = plt.cm.Set2(np.linspace(0, 1, 8))  # color set2
COLOR_DARK2 = plt.cm.Dark2(np.linspace(0, 1, 8))  # color Dark2

FIGSIZE = (12, 9)
FONT = 20
LABELPAD = 20


# Painter
def smooth(series, smooth_w=100):
    return pd.Series(copy(series)).rolling(smooth_w,
                                           min_periods=smooth_w).mean()


def get_ticks(t_start, t_end, num_ticks=6):
    """
    Get a number `num_ticks` of time ticks and labels
    given a closed time range [`t_start`, `t_end`]
    """
    time_scale = np.linspace(t_start, t_end, num_ticks + 1)
    ticks = (time_scale - t_start) * 3600
    labels = [
        f'{(h) % 24:02.0f}:{m * 60:02.0f}'
        for h, m in zip(time_scale // 1, time_scale % 1)
    ]
    return ticks, labels


def get_path():
    return "Temp"


def save_file(fig, save, filename):
    if save:
        path = get_path()
        if path is not None:
            fig.savefig(osp.join(path, filename), dpi=440, transparent=True)


def plot_demand_station(env,
                        save=False,
                        figsize=(12, 9),
                        fontsize=20,
                        filename='demand-station.pdf'):
    """
    3D ploting of (start_station, time, demand_rate)
    """
    from env import ODshape
    xticks, xlabels = get_ticks(env.t_start, env.t_end)
    fontsize = fontsize or FONT
    fontsmall = fontsize - 3
    TDD = np.repeat(ODshape(env.tvd) / 60, 60, axis=0)

    Z = TDD.sum(axis=2).T       # Z: demand rate (nodes, time)
    Y = np.arange(Z.shape[0])   # Y: service node
    X = np.arange(Z.shape[1])   # X: Time range

    # make the data continuous otherwise lead to incorrect result
    Z = np.array([smooth(Z[i, :], 2).values for i in range(Y.size)])
    Z = np.nan_to_num(Z)
    Z[:, -1] = 0

    # plot
    fig = plt.figure(figsize=figsize or FIGSIZE)
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    verts = [list(zip(X, Z[i])) for i in range(Y.size)]

    poly = PolyCollection(verts,
                          facecolors=['whitesmoke'] * 4,
                          edgecolors=['black'] * 4,
                          alpha=0.6
                          )
    ax.add_collection3d(poly, zs=Y, zdir='y')

    # label axises
    ax.set_xlabel('Time', fontsize=fontsize, labelpad=LABELPAD)
    ax.set_ylabel('Service Node', fontsize=fontsize, labelpad=LABELPAD)
    ax.set_zlabel('Demand Rate (Persons/s)',
                  fontsize=fontsize,
                  labelpad=LABELPAD)

    # limit of axises
    ax.set_xlim3d(0, np.max(X))
    ax.set_ylim3d(np.min(Y), np.max(Y))
    ax.set_zlim3d(0, np.max(Z) * 1.1)

    # tick scales and labels
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=fontsmall)
    ax.set_yticks(np.arange(env.num_plat))
    ax.set_yticklabels(env.routes, fontsize=fontsmall - 5)
    ax.set_zticks(np.arange(0, Z.max() * 1.1 + 0.1, 0.5))
    ax.set_zticklabels(np.arange(0,
                                 Z.max() * 1.1 + 0.1, 0.5),
                       fontsize=fontsize - 3)

    # save figure
    save_file(fig, save, filename)


def plot_demand_overall(env,
                        save=False,
                        peak_factor=0,
                        mode=1,
                        cmap='plasma',
                        figsize=(12, 9),
                        line_width=1.5,
                        fontsize=25,
                        filename='demand-overall.pdf'):

    assert mode in [1, 2], 'WRONG MODE (please input 1 or 2)'
    fontsize = fontsize or FONT

    # y: overall demand at time `t`
    y = np.repeat(env.tvd / 60, 60, axis=0).sum(2).sum(1)
    x = np.arange(len(y))

    fig, axs = plt.subplots(figsize=figsize or FIGSIZE)

    if 0 < peak_factor < 1:
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        Peak = y.max() * peak_factor
        axs.axhline(y=Peak,
                    color='darkgrey',
                    linestyle='dashed',
                    alpha=0.8,
                    label='Peak Time')
        if mode == 1:  # Continuous norm
            norm = plt.Normalize(y.min(), y.max())
            lc = LineCollection(segments, cmap=cmap, norm=norm)

        if mode == 2:  # To revise
            cmap = ListedColormap(["#0072B2", "#D55E00"])
            norm = BoundaryNorm([np.min(y), Peak, np.max(y)], cmap.N)
            lc = LineCollection(segments, cmap=cmap, norm=norm)

        # Set the values used for colormapping
        lc.set_array(y)
        lc.set_linewidth(line_width)
        line = axs.add_collection(lc)
        fig.colorbar(line, ax=axs)
    else:
        plt.plot(y, 'black', label='Passenger arrival')

    axs.fill_between(x, y, interpolate=True, color='whitesmoke', alpha=0.5)

    # set axis limits
    axs.set_xlim(0, x.max())
    axs.set_ylim(0, y.max() * 1.05)

    # labeling
    plt.xlabel("Time", fontsize=fontsize, labelpad=LABELPAD)
    plt.ylabel("Demand Rate (Persons/s)", fontsize=fontsize, labelpad=LABELPAD)

    # tick scales and labels
    plt.yticks(np.arange(np.ceil(y.max() * 1.1)).astype(float),
               fontsize=fontsize - 3)
    plt.xticks(*get_ticks(env.t_start, env.t_end), fontsize=fontsize - 3)

    # save figure
    save_file(fig, save, filename)


def plot_demand_dif(demand, demand_cp, t_start, t_end, num_station, figsize=(12, 9), fontsize=25, filename='demand-dif.pdf', save=False):

    fontsize = fontsize or FONT
    fontsmall = fontsize - 3

    def shift_d(d):
        return np.repeat(d[
            (t_start - 5) * 60:(t_end - 5) *
            60, :int(num_station), :][..., :int(num_station)] / 60, 60, axis=0).sum(1).sum(1)

    demand = shift_d(demand)
    demand_cp = shift_d(demand_cp)

    fig, ax = plt.subplots(figsize=figsize or FIGSIZE)
    ax.plot(demand_cp, color='grey', lw=1.5, zorder=0, linestyle='--')
    ax.plot(demand, color='black', lw=1.5, zorder=5)
    x = np.arange(len(demand))
    ax.fill_between(x, demand_cp, interpolate=True, color='whitesmoke', alpha=0.5)

    ax.set_xlabel("Time", fontsize=fontsize, labelpad=LABELPAD)
    ax.set_ylabel("Demand Rate (Persons/s)", fontsize=fontsize, labelpad=LABELPAD)

    ax.set_xlim(0, x.max() * 1.001)
    ax.set_ylim(0, demand.max() * 1.1)

    plt.yticks(np.arange(np.ceil(demand.max() * 1.1)).astype(float), fontsize=fontsmall)
    plt.xticks(*get_ticks(t_start, t_end), fontsize=fontsmall)

    # save figure
    save_file(fig, save, filename)


def plot_timetable(env,
                   timetable,
                   save=True,
                   figsize=(20, 8),
                   fontsize=20,
                   filename='timetable.pdf'):
    fig = plt.figure(figsize=figsize or FIGSIZE)
    ax1 = fig.add_subplot(111)

    fontsize = fontsize or FONT
    fontsmall = fontsize - 3

    # station index for timetable
    station = np.repeat(
        np.append(np.arange(env.num_stat),
                  np.arange(env.num_stat)[::-1]), 2)

    color_set = plt.cm.plasma(np.linspace(0, 0.8, env.stock_size))
    for i in range(len(timetable)):
        ax1.plot(timetable[i],
                 station,
                 color=color_set[i % env.stock_size])

    # ax1.axvline((env.t_end - env.t_start) * 3600,
    #             color='r',
    #             ls='dashed',
    #             label='End of passenger arrival')

    # labeling
    station_name = [x[:3] for x in env.routes[:env.num_stat]]

    last = np.ceil(timetable.max() / 1800) / 2  # operation time lasts since the start time
    xticks, xlabels = get_ticks(env.t_start,
                                env.t_start + last,
                                int(last * 2))
    ax1.set_xlim(0, xticks.max())
    ax1.set_ylim(0, env.num_stat - 0.8)
    ax1.set_ylabel('Station', fontsize=fontsize, labelpad=LABELPAD)
    ax1.set_xlabel('Time', fontsize=fontsize, labelpad=LABELPAD)

    plt.xticks(xticks, xlabels, fontsize=fontsmall)
    plt.yticks(range(len(station_name)),
               station_name,
               fontsize=fontsmall)  # Rotation

    # save figure
    save_file(fig, save, filename)


def plot_headway(env, timetable, save=True, figsize=(12, 9), filename='headway.pdf', fontsize=18):
    fontsize = fontsize or FONT
    fontsmall = fontsize - 3

    TTB = timetable[:, 0]
    HDW = np.diff(TTB, prepend=0)
    y = np.repeat(env.tvd / 60, 60, axis=0).sum(2).sum(1)
    x = np.arange(len(y))

    fig = plt.figure(figsize=figsize or FIGSIZE)
    ax2 = fig.add_subplot(111)
    l2 = ax2.plot(y, "black", lw=1, label='Demand Rate')
    ax2.fill_between(
        x, y, interpolate=False, color='whitesmoke', alpha=0.4)
    ax2.set_yticks(np.arange(np.ceil(max(y)) + 1))
    ax2.set_yticklabels(np.arange(np.ceil(max(y)) + 1).astype(int), fontsize=fontsmall)
    ax2.set_xlim(0, TTB.max() + 30)
    ax2.set_ylim(0, y.max() * 1.05)

    xticks, xlabels = get_ticks(env.t_start, env.t_end)
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xlabels, fontsize=fontsmall)

    ax1 = ax2.twinx()
    l1 = ax1.plot(TTB, HDW, '#ed553b', lw=1.5, label='Headway')

    ax1.set_ylim(env.var_min[0], env.var_max[0] + 0.2 * env.var_int[0])
    ax1.set_yticks(np.ogrid[env.var_min[0]:env.var_max[0] +
                            env.var_int[0]:env.var_int[0]])
    ax1.set_yticklabels(np.ogrid[env.var_min[0]:env.var_max[0] +
                                 env.var_int[0]:env.var_int[0]],
                        fontsize=fontsmall)

    ax1.set_ylabel('Headway (s)', fontsize=fontsize, labelpad=LABELPAD)
    ax2.set_ylabel('Demand Rate (Persons/s)', fontsize=fontsize, labelpad=LABELPAD)
    ax2.set_xlabel('Time', fontsize=fontsize, labelpad=LABELPAD)

    lns = l1 + l2
    labs = [l.get_label() for l in lns]
    plt.legend(lns, labs, loc=0, frameon=False, fontsize=fontsmall)

    last = np.ceil(timetable.max() / 1800) / 2
    xticks, xlabels = get_ticks(env.t_start,
                                env.t_start + last,
                                int(last * 2))
    # plt.xticks(xticks, xlabels, fontsize=fontsmall)
    # plt.xlim(0, TTB.max() + 30)

    # save figure
    save_file(fig, save, filename)


def plot_headway_diff(env0,
                      env1,
                      ttb0,
                      ttb1,
                      save=True,
                      figsize=(12, 9),
                      filename='headway_diff.pdf',
                      fontsize=18):

    fontsize = fontsize or FONT
    fontsmall = fontsize - 3

    dsp0, dsp1 = ttb0[:, 0], ttb1[:, 0]
    p_max = max(dsp0.max(), dsp1.max())

    hdw0, hdw1 = np.diff(dsp0, prepend=0), np.diff(dsp1, prepend=0)
    dmd0 = np.repeat(env0.tvd / 60, 60, axis=0).sum(2).sum(1)
    dmd1 = np.repeat(env1.tvd / 60, 60, axis=0).sum(2).sum(1)
    d_max = max(dmd0.max(), dmd1.max())

    x = np.arange(len(dmd0))

    fig = plt.figure(figsize=figsize or FIGSIZE)
    ax1 = fig.add_subplot(111)
    ld0 = ax1.plot(dmd0, "grey", lw=1, label='Original Demand Rate', linestyle='--', alpha=0.8)
    ld1 = ax1.plot(dmd1, "black", lw=1.2, label='Shifted Demand Rate')

    ax1.fill_between(x, dmd0, interpolate=False, color='whitesmoke', alpha=0.3)

    # x axis of demand plotting
    xticks, xlabels = get_ticks(env0.t_start, env0.t_end)
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xlabels, fontsize=fontsmall)
    ax1.set_xlim(0, p_max + 1)
    ax1.set_xlabel('Time', fontsize=fontsize, labelpad=LABELPAD)

    # y axis of demand plotting
    ax1.set_yticks(np.arange(np.ceil(d_max) + 1))
    ax1.set_yticklabels(np.arange(np.ceil(d_max) + 1, dtype=int), fontsize=fontsmall)
    ax1.set_ylim(0, d_max * 1.1)
    ax1.set_ylabel('Demand Rate (Persons/s)', fontsize=fontsize, labelpad=LABELPAD)

    # headway plotting
    ax2 = ax1.twinx()
    hd0 = ax2.plot(dsp0, hdw0, '#046ec4', lw=1.2, label='Original Headway', linestyle='--')
    hd1 = ax2.plot(dsp1, hdw1, '#ed553b', lw=1.5, label='Shifted Headway')

    # y axis of headway plotting
    ax2.set_ylim(env0.var_min[0], env0.var_max[0] + 0.2 * env0.var_int[0])
    ax2.set_yticks(np.ogrid[env0.var_min[0]:env0.var_max[0] +
                            env0.var_int[0]:env0.var_int[0]])
    ax2.set_yticklabels(np.ogrid[env0.var_min[0]:env0.var_max[0] +
                                 env0.var_int[0]:env0.var_int[0]],
                        fontsize=fontsmall)
    ax2.set_ylabel('Headway (s)', fontsize=fontsize, labelpad=LABELPAD)

    lns = ld0 + ld1 + hd0 + hd1
    plt.legend(lns, [l.get_label() for l in lns], loc=0, fontsize=fontsmall)

    # save figure
    save_file(fig, save, filename)


def plot_evaluation(evals,
                    save=True,
                    figsize=(12, 9),
                    fontsize=18,
                    filename='policy-evaluation.pdf'):

    fontsize = fontsize or FONT
    fontsmall = fontsize - 3

    fig = plt.figure(figsize=size or FIGSIZE)
    evals = np.asarray(evals)
    eval_counts = len(evals)

    evals[0, :] = np.nan
    avg, max_, min_ = evals.mean(axis=1), evals.max(axis=1), evals.min(axis=1)

    plt.plot(avg, color='black', label='Mean')
    if not np.all(max_ == min_):
        plt.plot(max_, color='grey', alpha=0.5, label='Maximum')
        plt.plot(min_, color='grey', alpha=0.5, label='Minimum')
        plt.fill_between(range(eval_counts),
                         max_,
                         min_,
                         color='grey',
                         alpha=0.1)

    plt.xticks(np.arange(0, eval_counts + 1, 5, dtype=int), fontsize=fontsmall)
    # yxs = np.arange(min_[1:].min() // 10000, max_[1:].max() // 10000 + 1, dtype=int)
    plt.yticks(fontsize=fontsmall)

    plt.xlim(0, eval_counts - 1)
    # plt.ylim(0)

    plt.xlabel('Test Point', fontsize=fontsize, labelpad=LABELPAD)
    plt.ylabel('Test Performance Score', fontsize=fontsize, labelpad=LABELPAD)

    plt.legend(loc=0, fontsize=fontsmall)
    # plt.grid()

    # save figure
    save_file(fig, save, filename)


def plot_left_map(data, labels=None, figsize=(20, 4), fontsize=20, save=True, filename='left_map.pdf'):
    fontsize = fontsize or FONT
    fontsmall = fontsize - 3

    fig, ax = plt.subplots(figsize=figsize or FIGSIZE)
    im = ax.imshow(data, cmap='binary', vmin=0, vmax=200)
    cbar = ax.figure.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))

    ax.set_yticklabels(labels)
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False, size=fontsize + 3)

    # save figure
    save_file(fig, save, filename)


def plot_left_map_combined(data,
                  labels=None,
                  figsize=(20, 10),
                  fontsize=20,
                  save=True,
                  filename='left-map-combined.pdf'):
    fontsize = fontsize or FONT
    fontsmall = fontsize - 3
    n_plot = len(data.items())

    fig, axes = plt.subplots(figsize=figsize or FIGSIZE, nrows=2)
    lbs = ['a', 'b']
    for n, (algo, data) in enumerate(data.items()):
        ax = axes.flat[n]
        im = ax.imshow(data, cmap='binary', vmin=0, vmax=200)

        ax.set_xticks(np.arange(data.shape[1]))
        ax.set_yticks(np.arange(data.shape[0]))

        ax.set_yticklabels(labels)

        ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

        for edge, spine in ax.spines.items():
            spine.set_visible(False)

        ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
        ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=1.5)
        ax.tick_params(which="minor", bottom=False, left=False, size=fontsize + 3)

        ax.set_xlabel(f'({lbs[n]}) Resultant Heat map of the number of left passengers for {algo} Implementation', fontsize=fontsmall)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist())

    # save figure
    save_file(fig, save, filename)


def plot_learning_curve_coarse(q_seq,
                               *,
                               save=False,
                               fontsize=20,
                               figsize=(10, 8),
                               filename='learning-curve-corse.pdf'):
    fig = plt.figure(figsize=figsize or FIGSIZE)
    fontsize = fontsize or FONT
    fontsmall = fontsize - 3

    q_seq = np.asarray(q_seq, dtype=np.float32)
    plt.plot(q_seq, c='black')
    plt.xlim(0, len(q_seq))
    plt.xticks(fontsize=fontsmall)
    plt.yticks(fontsize=fontsmall)

    plt.ylabel('Performance Score', fontsize=fontsmall, labelpad=LABELPAD)
    plt.xlabel('Epoch', fontsize=fontsize, labelpad=LABELPAD)

    # save figure
    save_file(fig, save, filename)


def plot_noise(noise_seq, save=False, figsize=(10, 8), fontsize=None, filename='noise.pdf'):
    fontsize = fontsize or FONT
    fontsmall = fontsize - 3

    fig = plt.figure(figsize=figsize or FIGSIZE)
    noise_seq = np.asarray(noise_seq, dtype=float)
    plt.plot(noise_seq, 'black')
    plt.fill_between(np.arange(len(noise_seq)),
                     0,
                     noise_seq,
                     color='whitesmoke',
                     alpha=0.5)

    plt.xlim(0, len(noise_seq))
    plt.ylim(0, 0.2)

    # tick size
    plt.xticks(fontsize=fontsmall)
    plt.yticks(fontsize=fontsmall)

    # labeling
    plt.ylabel(r"Noise Scale ($\epsilon$)", usetex=True, fontsize=fontsize, labelpad=LABELPAD)
    plt.xlabel('Epoch', fontsize=fontsize, labelpad=LABELPAD)

    # save figure
    save_file(fig, save, filename)


def plot_violin(simu_dict,
                save=True,
                figsize=None,
                fontsize=None,
                filename='violin.pdf'):
    import seaborn as sns
    vdict = {}
    for k, v in sorted(simu_dict.items()):
        algo, seed, random_f = k.split("-")
        if algo.endswith('DDPG'):
            algo += '(' + random_f + ')'
        vdict[algo] = v

    df = pd.DataFrame(vdict)

    fontsize = fontsize or FONT
    fontsmall = fontsize - 3

    fig = plt.figure(figsize=figsize or FIGSIZE)
    sns.violinplot(data=df, scale_hue=False, palette='Set3')

    plt.xticks(fontsize=fontsmall)
    plt.yticks(fontsize=fontsmall)
    plt.xlabel('Method', fontsize=fontsize, labelpad=LABELPAD)
    plt.ylabel("Performance Score", fontsize=fontsize, labelpad=LABELPAD)

    # save figure
    save_file(fig, save, filename)


if __name__ == '__main__':
    from configs import get_env_kwargs
    from env import TubeEnv

    env_kwargs = get_env_kwargs()
    env = TubeEnv(**env_kwargs)

    # plot_demand_station(env, False)
    # plot_demand_overall(env, True)
    # plot_timetable(env, info['timetable'])
    # plot_timetable(env, info['timetable'])

    plt.show()
