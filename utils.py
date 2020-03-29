import os
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from matplotlib import colors
from tqdm import tqdm_notebook
from pathlib import Path


data_path = Path('arc')
training_path = data_path / 'training'
evaluation_path = data_path / 'evaluation'
test_path = data_path / 'test'

training_tasks = sorted(os.listdir(training_path))
evaluation_tasks = sorted(os.listdir(evaluation_path))
test_tasks = sorted(os.listdir(test_path))

cmap = colors.ListedColormap(
    ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
norm = colors.Normalize(vmin=0, vmax=9)


def flattener(pred):
    str_pred = str([row for row in pred])
    str_pred = str_pred.replace(', ', '')
    str_pred = str_pred.replace('[[', '|')
    str_pred = str_pred.replace('][', '|')
    str_pred = str_pred.replace(']]', '|')
    return str_pred


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
    ax.set_yticks([x-0.5 for x in range(1+len(task))])
    ax.set_xticks([x-0.5 for x in range(1+len(task[0]))])
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


def plot_task(task, filename, **kwargs):
    patterns = kwargs.get('patterns')
    export_to = kwargs.get('export_to')
    task_id = kwargs.get('task_id')

    n = len(task["train"]) + len(task["test"])
    fig, axs = plt.subplots(2, n, figsize=(6*n, 10), dpi=72)
    plt.subplots_adjust(wspace=0, hspace=0)
    plot_one_sample(axs, task['train'], 'Train', **kwargs)
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


def load_and_plot(tasks_path, task_fn, return_what=0, **kwargs):
    task_file = str(tasks_path / task_fn)

    with open(task_file, 'r') as f:
        task = json.load(f)

    if return_what == 0:
        plot_task(task, task_fn, **kwargs)
    elif return_what == 1:
        return task
    elif return_what == 2:
        plot_task(task, task_fn, **kwargs)
        return task


def plot_pred_and_target(pred, target, filename, sample_id, is_pred=True):
    if pred is None or target is None:
        return
    pred_title = 'Predicted'
    if not is_pred:
        pred_title = 'Input'
    title = "{} {}_{}".format(pred_title, filename, sample_id)
    fig, axs = plt.subplots(1, 2, figsize=(12, 10), dpi=72)
    plot_one_ax(axs[0], pred, cmap=cmap,
                norm=norm, title=title)
    plot_one_ax(axs[1], target, cmap=cmap,
                norm=norm, title=f'Target {filename}')
    plt.tight_layout()
    plt.show()


def just_plot(pixmap):
    fig, ax = plt.subplots(1, 1, figsize=(6, 10), dpi=72)
    plot_one_ax(ax, pixmap, cmap=cmap,
                norm=norm, title='A simple plot')
    plt.tight_layout()
    plt.show()


def build_df_tags():

    # from https://www.kaggle.com/c/abstraction-and-reasoning-challenge/discussion/131238#760044
    task_classification = pd.read_csv('task_classification.csv')
    filenames = task_classification.output_id

    # clean NaN
    tags = task_classification.Tagging.values
    tags = [tag for tag in tags if type(tag) != float]

    # utility functions
    def clean_string(s):
        return s.strip().replace("'", '')

    def split_list(ls):
        return ls.replace('[', '').replace(']', '').split(',')

    # extract Tagging column
    # this series will have repeated value for the tasks with more than
    # one pattern
    tags_by_index = task_classification.Tagging.dropna()
    tags_by_index = tags_by_index.map(split_list).explode().map(clean_string)

    # build tasks with its tags as a sparse matrix
    tasks_with_tags = pd.concat([filenames[tags_by_index.index],
                                 pd.get_dummies(tags_by_index)],
                                axis=1).reset_index(drop=True)

    # sum rows for the same task
    df = pd.DataFrame([], columns=tasks_with_tags.columns)

    for group in tasks_with_tags.groupby('output_id').groups:
        group_df = tasks_with_tags.groupby('output_id').get_group(group)
        tags = group_df.iloc[:, 1:].sum().values
        row = np.concatenate([[group], tags])
        df = df.append(pd.DataFrame(row.reshape(1, -1), columns=df.columns),
                       ignore_index=True)

    # cast tags (0 or 1 values) to int
    df[df.columns[1:]] = df.iloc[:, 1:].apply(lambda x: x.astype(np.int8))

    return df


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