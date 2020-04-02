"""
Program evaluation.
"""
import logging
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
        # Break if there is no data
        if image_list == []:
            return []
    return image_list