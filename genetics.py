import math
import random
from typing import Callable, List, Tuple

import numpy as np
from nptyping import Array
from dsl import identity
from dsl import tail, head, union, intersect
from dsl import sort_by_color, sort_by_weight, reverse
from dsl import crop_to_content, group_by_color, split_v, split_h
from dsl import negative_by_frequency, negative_by_max
from dsl import reflect_image
from evaluation import evaluate, program_description, is_solution

from type_aliases import Program

CANDIDATES_NODES = [
    tail, head, union, intersect,
    sort_by_color, sort_by_weight, reverse,
    crop_to_content, group_by_color, split_h, split_v,
    negative_by_frequency, negative_by_max,
    reflect_image
]


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
                for img in images:
                    raw_fitness[index] += fitness_function(img, o)
                raw_fitness[index] /= len(images)

    adjusted_fitness = sum([score / (1 + score) for score in raw_fitness]) / len(raw_fitness)
    return adjusted_fitness


def _random_node(allowed_nodes):
    return random.choice(allowed_nodes)


def crossover(parent_programs: Tuple[Program, Program]) -> Tuple[Program, Program]:
    """Takes parents and perform a random permutation between them."""
    assert len(parent_programs) == 2

    prog_a, prog_b = parent_programs
    prog_a_size = len(prog_a)
    prog_b_size = len(prog_b)

    # programs have variable but limited length
    min_prog = min(prog_a_size, prog_b_size)
    crossover_point = random.randrange(0, min_prog)

    offspring_a = prog_a[:crossover_point] + prog_b[crossover_point:]
    offspring_b = prog_a[crossover_point:] + prog_b[:crossover_point]

    return offspring_a, offspring_b


def mutate(program: Program, allowed_nodes: List[Program]) -> Program:
    if (random.random() * 100) <= 1:
        program[random.randrange(0, len(program))] = _random_node(allowed_nodes)
    return program


def tournament_selection(candidates: dict, tournament_size: int) -> Program:
    selection = np.random.choice(len(candidates), size=tournament_size, replace=False)
    selected_candidates = {key: candidates[key] for key in selection}
    # (score, program)
    best_candidate = sorted(selected_candidates.values(), key=lambda value: value[0], reverse=True)[0]
    return best_candidate[1]


def init_population(allowed_nodes, **kwargs):
    population_size = kwargs.get('population_size', 0)
    length_limit = kwargs.get('candidate_size', 0)  # Maximal length of a program
    if population_size == 0 or length_limit == 0:
        return []
    # programs with variable sizes but with max `length_limit`
    population = [[_random_node(allowed_nodes) for _ in range(random.randrange(1, length_limit + 1))] for _ in range(population_size)]
    return population


def build_model(task, population_size=200, candidate_size=5, tournament_ratio=0.25, max_iterations=20, verbose=False):

    if verbose:
        print("Candidates nodes are:\n", [program_description([n]) for n in CANDIDATES_NODES])
        print()

    # A dictionary of {key:(score, candidate)}; initial random population
    best_candidates = {}
    candidates = init_population(CANDIDATES_NODES, population_size=population_size, candidate_size=candidate_size)
    for i in range(max_iterations):
        if verbose:
            print("Iteration ", i + 1)
            print("-" * 10)

        # Keep candidates with best fitness.
        # They will be stored in the `best_candidates` dictionary
        # where the key of each program is its fitness score.
        for key, candidate in enumerate(candidates):
            score = evaluate_fitness(candidate, task)
            best_candidates[key] = (score, candidate)

        # Normalized fitness
        population_fitness = math.fsum([score for score, candidate in best_candidates.values()])
        best_candidates = {key: (score / population_fitness, candidate) for key, (score, candidate) in best_candidates.items()}

        # For each best candidate, we look if we have an answer
        for score, program in best_candidates.values():
            if is_solution(program, task):
                return (score, program)

        # Don't have an answer yet
        # Evolve population
        candidates.clear()
        for _ in range(population_size // 2):
            # Select best candidates for next generation
            tournament_size = int(population_size * tournament_ratio)
            candidate_a = tournament_selection(best_candidates, tournament_size=tournament_size)
            candidate_b = tournament_selection(best_candidates, tournament_size=tournament_size)
            offsprings = crossover((candidate_a, candidate_b))

            for child in offsprings:
                mutate(child, allowed_nodes=CANDIDATES_NODES)
                candidates.append(child)

        # Give some informations by selecting a random candidate
        if verbose:
            best_score, best_candidate = sorted(best_candidates.items(), key=lambda item: item[1][0], reverse=True)[0][1]
            print("Best candidate score:", best_score)
            print("Best candidate implementation:", program_description(best_candidate))
            print("Best candidate lenght:", len(best_candidate))
            print()

    return None
