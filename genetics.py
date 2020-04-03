import math
import random

import numpy as np
from dsl import identity
from dsl import tail, head, union, intersect
from dsl import sort_by_color, sort_by_weight, reverse
from dsl import crop_to_content, group_by_color, split_v, split_h
from dsl import negative_by_frequency, negative_by_max
from dsl import reflect_image
from evaluation import evaluate, program_description, is_solution

# Inspired by https://www.kaggle.com/zenol42/dsl-and-genetic-algorithm-applied-to-arc


def width_fitness(predicted, expected_output):
    """ How close the predicted image is to have the right width. Less is better."""
    return np.abs(predicted.shape[0] - expected_output.shape[0])


def height_fitness(predicted, expected_output):
    """ How close the predicted image is to have the right height. Less is better."""
    return np.abs(predicted.shape[1] - expected_output.shape[1])


def activated_pixels_fitness(p, e):
    """ How close the predicted image to have the right pixels. Less is better."""
    shape = (max(p.shape[0], e.shape[0]), max(p.shape[1], e.shape[1]))
    diff = np.zeros(shape, dtype=int)
    diff[0:p.shape[0], 0:p.shape[1]] = (p > 0).astype(int)
    diff[0:e.shape[0], 0:e.shape[1]] -= (e > 0).astype(int)

    return (diff != 0).sum()


def colors_fitness(p, e):
    p_colors = np.unique(p)
    e_colors = np.unique(e)

    nb_inter = len(np.intersect1d(p_colors, e_colors))

    return (len(p_colors) - nb_inter) + (len(e_colors) - nb_inter)


fitness_functions = [colors_fitness, activated_pixels_fitness, height_fitness, width_fitness]


def evaluate_fitness(program, task):
    """ Take a program and a task, and return its fitness score as a tuple."""
    raw_fitness = np.zeros((len(fitness_functions)))

    # For each sample
    for sample in task:
        i = np.array(sample['input'])
        o = np.array(sample['output'])

        # For each fitness function
        for index, fitness_function in enumerate(fitness_functions):
            images = evaluate(program, i)
            if images == []:
                # Penalize no prediction!
                raw_fitness[index] *= 1.5
            else:
                # Take only the score of the first output
                raw_fitness[index] = fitness_function(images[0], o)

    adjusted_fitness = sum([score / (1 + score) for score in raw_fitness]) / len(raw_fitness)
    return adjusted_fitness


def build_candidates(allowed_nodes=[identity], best_candidates=None, nb_candidates=200, **kwargs):
    """ Create a poll of fresh candidates using the `allowed_nodes`.

    The pool contain a mix of new single instructions programs
    and mutations of the best candidates.
    """
    new_candidates = []
    length_limit = kwargs.get('program_size', 4)  # Maximal length of a program

    def random_node():
        return random.choice(allowed_nodes)

    # Until we have enougth new candidates
    while(len(new_candidates) < nb_candidates):
        # Add 10 new programs
        for i in range(5):
            new_candidates += [[random_node()]]

        # Create new programs based on each best candidate
        for best_program in best_candidates:
            # Add one op on its right but limit the length of the program
            if len(best_program) < length_limit - 1:
                new_candidates += [[random_node()] + best_program]
            # Add one op on its left but limit the length of the program
            if len(best_program) < length_limit - 1:
                new_candidates += [best_program + [random_node()]]
            # Mutate one instruction of the existing program
            new_candidates += [list(best_program)]
            new_candidates[-1][random.randrange(0, len(best_program))] = random_node()

    # Truncate if we have too many candidates
    np.random.shuffle(new_candidates)
    return new_candidates[:nb_candidates]


def build_model(task, max_iterations=20, verbose=True, **kwargs):
    candidates_nodes = [
        tail, head, union, intersect,
        sort_by_color, sort_by_weight, reverse,
        crop_to_content, group_by_color, split_h, split_v,
        negative_by_frequency, negative_by_max,
        reflect_image
    ]

    if verbose:
        print("Candidates nodes are:\n", [program_description([n]) for n in candidates_nodes])
        print()

    # A dictionary of {key:(score, candidate)}
    best_candidates = {}
    for i in range(max_iterations):
        if verbose:
            print("Iteration ", i + 1)
            print("-" * 10)

        # Create a list of candidates
        previous_best_candidates = [candidate for score, candidate in best_candidates.values()]
        candidates = build_candidates(candidates_nodes, previous_best_candidates, **kwargs)

        # Keep candidates with best fitness.
        # They will be stored in the `best_candidates` dictionary
        # where the key of each program is its fitness score.
        for key, candidate in enumerate(candidates):
            score = evaluate_fitness(candidate, task)
            best_candidates[key] = (score, candidate)

        # best_candidates = {k: v for k, v in sorted(best_candidates.items(), key=lambda item: item[1][0])}
        # Normalized fitness
        population_fitness = math.fsum([score for score, candidate in best_candidates.values()])
        best_candidates = {key: (score / population_fitness, candidate) for key, (score, candidate) in best_candidates.items()}

        # For each best candidate, we look if we have an answer
        for score, program in best_candidates.values():
            if is_solution(program, task):
                return (score, program)

        # Give some informations by selecting a random candidate
        if verbose:
            print("Best candidates lenght:", len(best_candidates))
            best_score, best_candidate = sorted(best_candidates.items(), key=lambda item: item[1][0], reverse=True)[0][1]
            print("Best candidate score:", best_score)
            print("Best candidate implementation:", program_description(best_candidate))
            print()
    return None
