import numpy as np


def group_by_color_unlifted(pixmap):
    nb_colors = int(pixmap.max()) + 1
    splited = [(pixmap == i) * i for i in range(1, nb_colors)]
    return [x for x in splited if np.any(x)]


def crop_to_content_unlifted(pixmap):
    true_points = np.argwhere(pixmap)
    if len(true_points) == 0:
        return []
    top_left = true_points.min(axis=0)
    bottom_right = true_points.max(axis=0)
    pixmap = pixmap[top_left[0]:bottom_right[0] + 1,
                    top_left[1]:bottom_right[1] + 1]
    return pixmap


def negative_by_max_color_unlifted(pixmap):
    negative = np.logical_not(pixmap).astype(int)
    color = max(pixmap.max(), 1)
    return negative * color


def negative_by_most_frequent_color_unlifted(pixmap):
    negative = np.logical_not(pixmap).astype(int)
    # count color frequency, drop 0 count, argmax, then +1 to account 0
    color = np.argmax(np.bincount(pixmap.ravel())[1:]) + 1
    return negative * color