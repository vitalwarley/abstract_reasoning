import logging
import numpy as np
from typing import Callable, List
from nptyping import Array

logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.INFO)


def program_description(program: List[Callable[[List[Array[int]]], List[Array[int]]]]) -> str:
    """ Create a human readable description of a program.

    Parameters
    ----------
    program : list of programs

    Returns
    -------
    str
        a human readable string describing the program by the functions call order
    """
    desc = [x.__name__ for x in program]
    return(" >> ".join(desc))


def evaluate(program: List[Callable[[List[Array[int]]], List[Array[int]]]],
             input_image: Array[int]) -> List[Array[int]]:
    # Mae sure the input is of the right type
    assert isinstance(input_image, Array[int, ...])

    # Apply each function on the image
    image_list = [input_image]
    for fct in program:
        logging.info(f"Applying {fct.__name__}...")
        # Apply the function
        image_list = fct(image_list)
        # Filter out empty images
        image_list = [img for img in image_list if img.shape[0] > 0 and img.shape[1] > 0]
        # Apply the function
        # Break if there is no data
        if image_list == []:
            return []
    return image_list


def are_two_images_equals(a, b):
    if tuple(a.shape) == tuple(b.shape):
        if (np.abs(b - a) < 1).all():
            return True
    return False


def is_solution(program, task):
    # For each pair input/output
    for sample in task:
        i = np.array(sample['input'])
        o = np.array(sample['output'])

        # Evaluate the program on the input
        images = evaluate(program, i)
        if len(images) < 1:
            return False

        # The solution should be in the 3 first outputs
        images = images[:3]

        # Check if the output is in the 3 images produced
        is_program_of_for_sample = any([are_two_images_equals(x, o) for x in images])
        if not is_program_of_for_sample:
            return False

    return True
