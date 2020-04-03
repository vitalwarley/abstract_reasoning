import logging
import argparse
from data import training_path, training_tasks
from plot_utils import load_task
from evaluation import program_description
from genetics import build_model

logging.disable()

parser = argparse.ArgumentParser(description='Run GP on ARC.')
parser.add_argument('task_name', help='the task filename which we are going to solve')
parser.add_argument('--program-size', help='length of each candidate program',
                    type=int, default=4)
parser.add_argument('--max-it', help='max iterations to run when searching for the intended program',
                    type=int, default=20)
parser.add_argument('--verbose', help='increase output verbosity', action='store_true')

args = parser.parse_args()

fname = args.task_name + '.json'
program_size = args.program_size
max_iters = args.max_it
verbose = args.verbose

tid = training_tasks.index(fname)
task_dict = load_task(training_path, training_tasks[tid], task_id=tid, only_train=True)
sample_id = 0
task = task_dict['train']

result = build_model(task, max_iterations=max_iters, verbose=True, program_size=program_size)

if result is None:
    print("No program was found")
else:
    score, program = result
    print("Found program:", program_description(program))
    print("Score:", score)