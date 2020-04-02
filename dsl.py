import numpy as np
import itertools
from numpy.lib.stride_tricks import as_strided
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.feature import corner_harris, corner_peaks
from skimage import measure
from skimage import feature

from typing import Callable, List, Union
from nptyping import Array


"""
Mostly inspired by: https://www.kaggle.com/zenol42/dsl-and-genetic-algorithm-applied-to-arc
"""

# DSL


def group_by_color_unlifted(pixmap: Array[int]) -> List[Array[int]]:
    nb_colors = int(pixmap.max()) + 1
    splited = [(pixmap == i) * i for i in range(1, nb_colors)]
    return [x for x in splited if np.any(x)]


def crop_to_content_unlifted(pixmap: Array[int]) -> Array[int]:
    true_points = np.argwhere(pixmap)
    if len(true_points) == 0:
        return []
    top_left = true_points.min(axis=0)
    bottom_right = true_points.max(axis=0)
    pixmap = pixmap[top_left[0]:bottom_right[0] + 1,
                    top_left[1]:bottom_right[1] + 1]
    return [pixmap]


def split_h_unlifted(pixmap: Array[int]) -> List[Array[int]]:
    h = pixmap.shape[0]
    pivot = h // 2
    if h % 2 == 1:
        return [pixmap[:pivot, :], pixmap[pivot + 1:, :]]
    else:
        return [pixmap[:pivot, :], pixmap[pivot:, :]]


def split_v_unlifted(pixmap: Array[int]) -> List[Array[int]]:
    w = pixmap.shape[0]
    pivot = w // 2
    if w % 2 == 1:
        return [pixmap[:pivot, :], pixmap[pivot + 1:, :]]
    else:
        return [pixmap[:pivot, :], pixmap[pivot:, :]]


def negative_by_max_unlifted(pixmap: Array[int]) -> List[Array[int]]:
    """Turn pixmap into its negative with the color being that which is the max."""
    negative = np.logical_not(pixmap).astype(int)
    color = max(pixmap.max(), 1)
    return [negative * color]


def negative_by_frequecy_unlifted(pixmap: Array[int]) -> List[Array[int]]:
    """Turn pixmap into its negative with the color being that which is the most frequent."""
    negative = np.logical_not(pixmap).astype(int)
    # Count color frequency, drop 0 count, argmax, then +1 to account 0
    color = np.argmax(np.bincount(pixmap.ravel())[1:]) + 1
    return [negative * color]


def identity(x: Array[int]) -> List[Array[int]]:
    return x


def tail(x: List[Array[int]]) -> List[Array[int]]:
    if len(x) > 1:
        return x[1:]
    else:
        return x


def head(x: List[Array[int]]) -> List[Array[int]]:
    if len(x) > 1:
        return x[:1]
    else:
        return x


def _check_shape_is_consistent(x: List[Array[int]]) -> List[Array[int]]:
    if len(x) < 2:
        return x
    first_shape = tuple(x[0].shape)
    ok_shapes = [first_shape == tuple(pixmap.shape) for pixmap in x[1:]]
    return ok_shapes.all()


def union(x: List[Array[int]]) -> List[Array[int]]:
    if _check_shape_is_consistent(x):
        return [np.bitwise_or.reduce(np.array(x).astype(int))]
    else:
        return []


def intersect(x: List[Array[int]]) -> List[Array[int]]:
    if _check_shape_is_consistent(x):
        return [(np.prod(np.array(x), axis=0) > 0).astype(int)]
    else:
        return []


def sort_by_color(xs: List[Array[int]]) -> List[Array[int]]:
    xs = [x for x in xs if len(x.reshape(-1)) > 0]
    return list(sorted(xs, key=lambda x: x.max()))


def sort_by_weight(xs: List[Array[int]]) -> List[Array[int]]:
    xs = [x for x in xs if len(x.reshape(-1)) > 0]
    return list(sorted(xs, key=lambda x: (x > 0).sum()))


def reverse(x):
    return x[::-1]


# Composition of functions:
#   we need to generate a lifted version of the Array[int] -> List[Array[int]] functions
def lift(fct: Callable[[Array[int]], List[Array[int]]]
         ) -> Callable[[List[Array[int]]], List[Array[int]]]:
    """Turn a function f: np.array -> [np.array] to f: [np.array] -> [np.array]"""
    # Lift the function
    def lifted_function(xs: List[Array[int]]):
        # Map function to each x in xs
        results = [fct(x) for x in xs]
        # Flatten list
        return list(itertools.chain(*results))
    # Rename lifted functions
    import re
    lifted_function.__name__ = re.sub('_unlifted$', '', fct.__name__)
    return lifted_function


crop_to_content = lift(crop_to_content_unlifted)
group_by_color = lift(group_by_color_unlifted)
split_h = lift(split_h_unlifted)
split_v = lift(split_v_unlifted)
negative_by_max = lift(negative_by_max_unlifted)
negative_by_frequency = lift(negative_by_frequecy_unlifted)


def pattern_repetition(pixmap: Array[int]) -> Array[int]:
    """
    In contrast with `fractal_repetition`,
    the pattern repeats in a deterministic way.
    """
    # Crop code (ideally, called before this method)
    pattern = crop_to_content_unlifted(pixmap)

    # Pattern rep code
    out_sh = (1, 2)  # can be learned if one looks to output
    output = np.tile(pattern, out_sh)
    return output


def block_indices_from_pixmap(pixmap_shape, block_shape, stride):

    xs, ys = np.indices(pixmap_shape, dtype=np.int8)

    blocks_shape = ((pixmap_shape[0] - block_shape[0]) // stride + 1,
                    (pixmap_shape[1] - block_shape[1]) // stride + 1)

    # each block with block_shape
    shape = blocks_shape + block_shape
    new_shape = (blocks_shape[0] * blocks_shape[1],) + block_shape

    strides = (pixmap_shape[1] * stride,      # bytes to next vertical block
               stride,                   # bytes to next horizontal block
               pixmap_shape[1],               # bytes to next vertical element
               1)                        # bytes to next horizontal element

    xs_strided = as_strided(xs, shape=shape, strides=strides).reshape(*new_shape)
    ys_strided = as_strided(ys, shape=shape, strides=strides).reshape(*new_shape)

    return xs_strided, ys_strided


def fractal_repetition(pixmap, stride=3):
    """
    Maps each pixel in the pattern to a
    window with pattern shape in the output pixmap.
    """

    # TODO: maybe won't work if pattern.shape ** 2 != pixmap.shape

    # Crop code (ideally, called before this method)
    pattern = crop_to_content_unlifted(pixmap)
    ptrn_sh = pattern.shape
    pixm_sh = pixmap.shape

    # Fractal repetition code
    out_sh = ((pixm_sh[0] - ptrn_sh[0]) // stride + 1,
              (pixm_sh[1] - ptrn_sh[1]) // stride + 1)
    output = np.tile(pattern, out_sh)

    # indices in pixmap
    xs_strided, ys_strided = block_indices_from_pixmap(pixm_sh,
                                                       ptrn_sh,
                                                       stride)
    n_blocks = xs_strided.shape[0]

    # repeat pattern
    for i in range(n_blocks):  # num of blocks in pixmap
        pixel = pattern[i // ptrn_sh[0], i % ptrn_sh[0]]
        output[xs_strided[i], ys_strided[i]] = pattern * (pixel != 0)

    return output


def find_rectangle(pixmap, stride=1, shape=(3, 3), position=1):
    pixm_sh = pixmap.shape
    xs_strided, ys_strided = block_indices_from_pixmap(pixm_sh,
                                                       shape,
                                                       stride)
    return pixmap[xs_strided[position], ys_strided[position]].copy()


def retrieve_objects(pixmap, stride=3, obj_shape=(3, 3), filter_color=0):
    """
    This function is similar to `retrieve_rectangles`.

    I just modified the to filter rectangles.
    """
    pixm_sh = pixmap.shape

    # generalization of previous definition
    xs_strided, ys_strided = block_indices_from_pixmap(pixm_sh,
                                                       obj_shape,
                                                       stride)

    n_blocks = xs_strided.shape[0]
    outputs = []
    for i in range(n_blocks):
        rect = pixmap[xs_strided[i], ys_strided[i]]
        uniques_in_rect = np.unique(rect)
        if (uniques_in_rect.size == 2
                and filter_color not in uniques_in_rect):
            outputs.append(rect.copy())
    return outputs


def reflect_image(pixmap: Array[int]) -> Array[int]:
    output = np.flip(pixmap, axis=1)
    return output


def crop_unique_colors_orthogonal_to_color_continuity(pixmap):
    # select
    row = pixmap[0]
    col = pixmap[:, 0]
    row_uniques = np.unique(row)
    # define return
    return_row = row_uniques.shape[0] != 1
    output = row if return_row else col
    # delete repeated and contiguous elements
    del_idxs = np.nonzero(output[:-1] == output[1:])
    output = np.delete(output, del_idxs)
    # define return shape based on what it will return
    return_shape = (1, -1) if return_row else (-1, 1)
    return output.reshape(*return_shape)


def unique_colors(pixmap):
    unique = np.unique(pixmap.ravel())
    return unique


def select_row(pixmap):
    i = 0  # can be learned
    row = pixmap[np.newaxis, i, :]
    return row.copy()


def select_col(pixmap):
    i = 0
    col = pixmap[:, np.newaxis, i]
    return col.copy()


def unique_rows(pixmap):
    del_rows = np.nonzero((pixmap[:-1, :] == pixmap[1:, :]).any(axis=1))
    output = np.delete(pixmap, del_rows, axis=0)
    return output


def unique_cols(pixmap):
    del_cols = np.nonzero((pixmap[:, :-1] == pixmap[:, 1:]).any(axis=0))
    output = np.delete(pixmap, del_cols, axis=1)
    return output


def reshape_to_row(pixmap):
    shape = (1, -1)  # can be learned
    return pixmap.reshape(*shape)


def reshape_to_col(pixmap):
    shape = (-1, 1)  # can be learned
    return pixmap.reshape(*shape)


def longer_contour(pixmap):
    contours = measure.find_contours(pixmap, level=0)
    # get longer contour
    contours_size = np.array([c.size for c in contours])
    coords = contours[np.argmax(contours_size)].astype(np.int8)
    return coords


def find_rectangle_by_entropy(pixmap):
    pm_entropy = entropy((pixmap / 9), disk(1))
    pm_entropy = np.floor(pm_entropy)
    # low entropy, low values
    pm_entropy[pm_entropy != 0] = 1  # binary image
    # draw contour, then crop region
    contour = np.zeros_like(pm_entropy)
    for x, y in longer_contour(pm_entropy):
        contour[x, y] = 1
    region = crop_to_content_unlifted(contour)
    h, w = region.shape
    coords = match_coords(contour, region)
    x, y = next(zip(*coords))
    # main rectangle
    return pixmap[x:x + h, y:y + w].copy()


def draw_straight_lines_from_seeds(pixmap):
    # find color
    count = np.bincount(pixmap.ravel())
    color_to_draw = np.argpartition(count, -2)[-2]
    # find where they are
    coords = np.argwhere(pixmap == color_to_draw)
    coords = zip(coords[:, 0], coords[:, 1])
    # draw
    for x, y in coords:
        pixmap[x, :] = color_to_draw
        pixmap[:, y] = color_to_draw
    return pixmap


def match_coords(pixmap, obj):
    result = feature.match_template(pixmap, obj)
    ij = np.unravel_index(np.argpartition(result.ravel(), -3)[-3:], result.shape)  # first peak => its 2D index
    max_res = np.round(result[ij], decimals=3) == 1.
    return ij[0][max_res], ij[1][max_res]


def infect(pixmap_in):
    pixmap = pixmap_in.copy()
    # count colors frequency
    count = np.bincount(pixmap.ravel())
    # find intruder location
    intruder = np.argmax(count == np.trim_zeros(np.sort(count))[0])
    intruder_coords = np.argwhere(pixmap == intruder)
    # find dominant color (ASSUMPTION: the intruder will be the least dominant, aside from the foreground)
    dominant = np.argmax(count == np.sort(count)[-2])
    # infect
    xs, ys = np.indices((3, 3))
    dx, dy = (intruder_coords - 1).reshape(-1,)
    while True:  # this may be not a good idea in the GP algorithm
        # center in a 3x3 block in intruder_location
        block = pixmap[xs + dx, ys + dy]
        # next one to infect
        coords = zip(*np.nonzero(block == dominant))
        try:
            next_x, next_y = next(coords)  # need only the first coord here (infect one per block, horizontally)
        except StopIteration:
            break  # this means that there is not more pixels to be infected
        # (dx, dy) is a vector from (0, 0) in pixmap to block (0, 0)
        # (next_x, next_y) is a vector from (0, 0) in block to where next_to_infect is
        coords = (next_x + dx, next_y + dy)
        pixmap[coords] = intruder
        # next block will center in this infected pixel
        dx, dy = coords  # coords in the block center
        dx, dy = dx - 1, dy - 1  # displacement to origin (0, 0)

    # rationale? just to solve; need to eval in other similar cases.
    x, y = intruder_coords[0]
    pixmap[x, y] = 0  # FIXME?: this maybe won't generalize to other similar cases

    return pixmap


def normalize_color_by_dominant(pixmap_in):
    pixmap = pixmap_in.copy()
    count = np.bincount(pixmap.ravel())
    dominant = np.argmax(count == np.sort(count)[-2])
    weak = np.argmax(count == np.trim_zeros(np.sort(count))[0])
    weak_idxs = np.nonzero(np.logical_and(pixmap == weak, pixmap != 0))
    pixmap[weak_idxs] = dominant
    return pixmap


def filter_by_corners(pixmap):
    outputs = []
    groups = group_by_color_unlifted(pixmap)
    for group in groups:
        # can't detect > 2 corners in small rectangles
        # maybe peak_local_max?
        coords = corner_peaks(corner_harris(group), min_distance=1)
        if len(coords) <= 4:
            outputs.append(group)
    return outputs


# TODO: refactor to generalize to rectangle size
def rectangle_extremities_from_contour(coords):
    walk = sum(coords[:-1, 0] == coords[0][0]) - 1
    ll = coords[walk]  # walk from lower right to lower left
    ll[1] -= 1
    walk += sum(coords[:-1, 1] == ll[1]) - 1
    ul = coords[walk]  # walk to upper left
    ul[0] -= 1
    walk += sum(coords[:-1, 0] == ul[0]) - 1
    ur = coords[walk]  # walk to upper right
    ur[1] += 1
    ll += (-1, 1)
    ul += (1, 1)
    ur += (1, -1)
    return ll, ul, ur


# TODO: refactor. how to generalize to both tasks 232 (crop with borders?) and task 28 (crop only inside)
def extract_closure_from_rect(pixmap, filter_by_corners=True, only_inside=True):
    if filter_by_corners:
        # filter content
        filtered = filter_by_corners(pixmap)
        contours = [longer_contour(f) for f in filtered]
        coords = contours[np.argmax([c.size for c in contours])]
    else:
        # coords by contours
        coords = longer_contour(pixmap)
    # walk to extract points
    ll, ul, ur = rectangle_extremities_from_contour(coords)
    # return closure
    if only_inside:
        return pixmap[ul[0] + 1:ll[0], ul[1] + 1:ur[1]].copy()
    else:
        return pixmap[ul[0]:ll[0] + 1, ul[1]:ur[1] + 1].copy()


def match_and_fit_patterns(pixmap):

    def mask_obj(obj, main_color, secondary_color, ret_nd_color=True):
        mask = obj == main_color
        flip_mask = np.logical_not(mask)
        obj[mask] = secondary_color
        non_dominant = obj[flip_mask][0]
        obj[flip_mask] = main_color
        if ret_nd_color:
            return obj, non_dominant
        return obj

    def pixels_around(pixmap, x, y, s, c):
        if x > 0 and y > 0:
            x, y = x - 1, y - 1
        elif x > 0:
            x, y = x - 1, y
        elif y > 0:
            x, y = x, y - 1
        if x == y == 3:
            print(pixmap[x:x + s, y:y + s])
            from plot_utils import just_plot
            just_plot(pixmap)
        return (pixmap[x:x + s, y:y + s] == background).any()

    k = 3  # rect shape
    main_color = 2  # can be learned
    background = 0  # can be learned?
    objects = retrieve_objects(pixmap, stride=1)
    masked_objs = [mask_obj(obj, main_color, background) for obj in objects]
    extracted_pixmap = extract_closure_from_rect(pixmap, filter_by_corners=False, only_inside=False)

    for obj, nd_color in masked_objs:
        rotated_objs = [np.rot90(obj, k=i + 1).copy() for i in range(4)]
        for robj in rotated_objs:
            # possible matches
            xs, ys = match_coords(extracted_pixmap, robj)
            for x, y in zip(xs, ys):
                # extract matched part
                part = extracted_pixmap[x:x + k, y:y + k].copy()
                # unmask: retrieve obj original colors
                unmasked_obj = mask_obj(robj, main_color, nd_color, ret_nd_color=False)
                # check if match part wasn't replaced before
                if (part == background).any():
                    extracted_pixmap[x:x + k, y:y + k] = unmasked_obj
                # check if there are colors other 0s around
                if pixels_around(extracted_pixmap, x, y, k, background):
                    extracted_pixmap[x:x + k, y:y + k] = part

    return extracted_pixmap


def filter_by_color_frequency(pixmaps: List[Array[int]]):
    freqs = []
    for pixmap in pixmaps:
        count = np.bincount(pixmap.ravel())
        freqs.append(count[-1])  # drop 0
    return pixmaps[np.argmin(freqs)]


def detect_infected_object(pixmap):
    groups = group_by_color_unlifted(pixmap)
    intruder = filter_by_color_frequency(groups)
    intruder_coords = np.argwhere(intruder)
    # finder infected object origin
    padded_pixmap = np.pad(pixmap, 1)  # for edge cases
    x, y = intruder_coords[0, 0] + 1, intruder_coords[0, 1] + 1
    check_around_intruder = [padded_pixmap[x + dx, y + dy] != 0 for dx, dy in zip([-1, 0, 0, 1], [0, -1, +1, 0])]
    origins = np.array([[x - 3, y - 1],  # above
                        [x - 1, y - 3],  # left
                        [x - 1, y + 1],  # right
                        [x + 1, y - 1]])  # below
    pixel_infected = np.argmax(check_around_intruder)
    object_infected_coords = origins[pixel_infected, :]
    xs, ys = np.indices((3, 3))  # can be learned?
    return padded_pixmap[xs + object_infected_coords[0],
                         ys + object_infected_coords[1]]