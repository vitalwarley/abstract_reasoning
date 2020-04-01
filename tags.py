import numpy as np
import pandas as pd


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

