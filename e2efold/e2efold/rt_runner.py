import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import io
import numpy as np
from tensorflow import flags
import pdb
import math

import tensorflow as tf

import randomized_telescope_runner as runner # TBC

import torch
import torch.nn.functional as F
from torch import nn
import torch.distribution as D

FLAGS = flags.FLAGS

flags.DEFINE_folat('meta_lr', None, 'meta optimization learning rate')
flags.DEFINE_float('beta1', 0.9, 'adam beta1')
flags.DEFINE_float('beta2', 0.999, 'adam beta2')
flags.DEFINE_float('adam_eps', 1e-8, 'adam eps')

flags.DEFINE_integer('train_horizon', 5, 'truncated horizon of problem')
flags.DEFINE_integer('test_horizon', 5, 'full horizon of problem'
flags.DEFINE_integer('budget', 250000, 'multiple of test_horizon we run for')

def make_problem():



def rt_runner():
    true_params, params, train_loss_fn, make_state_fn, eval_fn, make_plot_from_batch = make_problem()
    runner.run_experiment(
        params = params,
        train_loss_fn = train_loss_fn,
        make_state_fn = make_state_fn,
        eval_fn = eval_fn
    )