import os
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib import colors
from tqdm import tqdm_notebook

from tags import build_df_tags
from data import training_path
from data import training_tasks

# TODO: global var `tasks` that user modifies

cmap = colors.ListedColormap(
    ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
norm = colors.Normalize(vmin=0, vmax=9)


def annotate_pixels(ax, pixmap):
    nx, ny = pixmap.shape
    for x in range(nx):
        for y in range(ny):
            if pixmap[x][y] == 0:
                ax.annotate(f'{pixmap[x][y]}', (y, x),
                            xytext=(y - 0.1, x + 0.15), color='white')
                continue
            ax.annotate(f'{pixmap[x][y]}', (y, x), xytext=(y - 0.1, x + 0.15))


def plot_one_ax(ax, task, **kwargs):
    cmap = kwargs.get('cmap')
    norm = kwargs.get('norm')
    title = kwargs.get('title')
    annotate = kwargs.get('annotate', False)

    ax.imshow(task, cmap=cmap, norm=norm)
    ax.set_title(title + f' | {task.shape}')
    ax.set_yticklabels(list(range(task.shape[0])))
    ax.set_xticklabels(list(range(task.shape[1])))
    ax.set_yticks([x - 0.5 for x in range(1 + len(task))])
    ax.set_xticks([x - 0.5 for x in range(1 + len(task[0]))])
    ax.grid(True, which='both', color='lightgrey', linewidth=0.5)
    if annotate:
        annotate_pixels(ax, task)


def plot_one_sample(axs, task, split, **kwargs):
    for i, t in enumerate(task):
        t_in, t_out = np.array(t["input"]), np.array(t["output"])
        ax = axs[0][i]
        plot_one_ax(ax, t_in, cmap=cmap, norm=norm, title=f'{split}-{i} in', **kwargs)
        ax = axs[1][i]
        plot_one_ax(ax, t_out, cmap=cmap, norm=norm, title=f'{split}-{i} out', **kwargs)


# TODO: plot train and test in different figures.
def plot_task(task, filename, **kwargs):
    patterns = kwargs.get('patterns')
    export_to = kwargs.get('export_to')
    task_id = kwargs.get('task_id')
    only_train = kwargs.get('only_train', False)

    n_train = len(task["train"])
    n_test = len(task["test"]) if not only_train else 0

    n = n_train + n_test

    fig, axs = plt.subplots(2, n, figsize=(6 * n, 10), dpi=72)
    plt.subplots_adjust(wspace=0, hspace=0)
    plot_one_sample(axs, task['train'], 'Train', **kwargs)
    if not only_train:
        plot_one_sample(axs[:, len(task['train']):], task['test'], 'Test', **kwargs)
    plt.suptitle(f'Task {task_id} [{filename[:-5]}]', y=1.05, fontsize=20)
    if patterns is not None:
        plt.suptitle(f'Task {filename[:-5]}\nPatterns: {patterns}', y=1.05, fontsize=20)
    plt.tight_layout()
    if export_to is not None:
        os.makedirs(export_to, exist_ok=True)
        plt.savefig(export_to + f'/{len(os.listdir(export_to))}',
                    dpi=72, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def load_task(tasks_path, task_fn, plot=False, **kwargs):
    task_file = str(tasks_path / task_fn)

    with open(task_file, 'r') as f:
        task = json.load(f)

    if plot:
        plot_task(task, task_fn, **kwargs)
        return task
    else:
        return task


def plot_pred_and_target(pred, target, filename, sample_id, is_pred=True):
    if pred is None or target is None:
        return
    pred_title = 'Predicted'
    if not is_pred:
        pred_title = 'Input'
    title = "{} {}_{}".format(pred_title, filename[:-5], sample_id)
    fig, axs = plt.subplots(1, 2, figsize=(12, 10), dpi=72)
    plot_one_ax(axs[0], pred, cmap=cmap,
                norm=norm, title=title)
    title = "{} {}_{}".format('Target', filename[:-5], sample_id)
    plot_one_ax(axs[1], target, cmap=cmap,
                norm=norm, title=title)
    plt.tight_layout()
    plt.show()


def plot_task_and_pred(task, pred, filename, sample_id):
    fname = filename[:-5]
    pixmap_in, pixmap_out = nth_sample(task, sample_id)
    title = "{} {}_{}".format('Input', fname, sample_id)
    fig, axs = plt.subplots(1, 3, figsize=(12, 10), dpi=72)
    plot_one_ax(axs[0], pixmap_in, cmap=cmap, norm=norm, title=title)
    title = "{} {}_{}".format('Output', fname, sample_id)
    plot_one_ax(axs[1], pixmap_out, cmap=cmap, norm=norm, title=title)
    title = "{} {}_{}".format('Prediction', fname, sample_id)
    plot_one_ax(axs[2], pred, cmap=cmap, norm=norm, title=title)
    plt.tight_layout()
    plt.show()


def just_plot(pixmap):
    fig, ax = plt.subplots(1, 1, figsize=(6, 10), dpi=72)
    plot_one_ax(ax, pixmap, cmap=cmap,
                norm=norm, title='A simple plot')
    plt.tight_layout()
    plt.show()


def nth_sample(task, n):
    if n < 0 or n >= len(task):
        return None, None
    pixmap_in = np.array(task[n]['input'])
    pixmap_out = np.array(task[n]['output'])
    return pixmap_in, pixmap_out


def save_imgs_with_pattern(pattern=None):
    # TODO: make searching for image with certain patterns easy (PR in ARC?)

    df = build_df_tags()

    if pattern is not None:
        selection = df[df[pattern] == 1]
    else:
        pattern = df.columns[df.iloc[:, 1:].sum(axis=0).values.argmax() + 1]
        selection = df[df[pattern] == 1]
    # sorted by # patterns in task
    # TODO: add tasks with same patterns in order
    selection = df.iloc[selection.iloc[:, 1:].sum(axis=1).sort_values().index, :]

    for name in tqdm_notebook(selection.output_id):
        tid = training_tasks.index(name)
        patterns = selection.columns[selection[selection.output_id == name].values[:, 1:].nonzero()[1] + 1].values
        patterns = ', '.join(patterns)
        load_and_plot(training_path, training_tasks[tid], annotate=True,
                      patterns=patterns, export_to=f'patterns/{pattern}')