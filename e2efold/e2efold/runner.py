import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from tensorflow import flags
import logging
import os
from randomized_telescope import GeometricRandomizedTelescope, ShuffleRandomizedTelescope # TBC
from adaptive_randomized_telescope import NoTelescope, CollapsedFixedTelescope # TBC
import pdb
import random
from tensorboard_logger import Logger as TFLogger
import optimize_sampling_greedy
import optimize_samoling_greedy_roulette

from timer import Timer

from setproctitle import setproctitle

import io
import math
import time

import copy
FLAGS = flags.FLAGS

flags.DEFINE_string('name', "exp", 'name of experiment')

class RunningNorms(object):
    def __init__(self, horizon):
        self.estimated_norms = np.zeros([horizon+1, horizon])
        self.ts = np.zeros([horizon+1, horizon])

    def update(self, sq_norms):
        self.estimated_norms[:sq_norms.shape[0], :sq_norms.shape[1]] = (
            FLAGS.exp_decay * self.estimated_norms[:sq_norms.shape[0], :sq_norms.shape[1]] +
            (1. - FLAGS.exp_decay) * sq_norms)
        self.ts[:sq_norms.shape[0], :sq_norms.shape[1]] += 1.

    def get_norms(self):
        return self.estimated_norms / (1. - FLAGS.exp_decay**self.ts)

def make_logger():

    if FLAGS.name is None:
        raise Exception("Set name.")
    
    params_name = str(FLAGS_seed) +"_" + str(time.time())

    if not os.path.isdir(FLAGS.results_dir):
        try:
            os.mkdir(FLAGS.results_dir)
        except Exception as e:
            if not os.path.isdir(FLAGS.result_dir)：
                raise Exception(e)
    
    name = FLAGS.name
    if not os.path.isdir(os.path.join(FLAGS.results_dir, name)):
        os.mkdir(os.path.join(FLAGS.results_dir, name))

    path = os.path.join(FLAGS.results_dir, name, params_name)
    if not os.path.isdir(path):
        os.mkdir(path)

    if os.path.exists(os.path.join(path, 'out.log')):
        os.remove(os.path.join(path, 'out.log'))

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if FLAGS.verbose:
        logger.addHandler(logging.StreamHandler())

    logger.addHandler(
        logging.FileHandler(os.path.join(path, 'out.log')))
    
    if FLAGS.use_tflogger:
        tflogger = TFLogger(path)
    else:
        class TFLoggerDummy(object):
            def __init__(self):
                pass
            def log_scalar(self, *args, **kwargs):
                pass
            def log_images(self, *args, **kwargs):
                pass
        tflogger = TFLoggerDummy()

    logger.info(str(FLAGS.flag_values_dict()))

    return logger, tflogger

def run_experiment(params, train_loss_fn, eval_fn, make_state_fn):
    setproctitle(FLAGS.name)

    logger, tf_logger = make_logger()
    
    running_dnorm = RunningNorms(FLAGS.train_horizon+1)

    weight_decay = 0. if not hasattr(FLAGS, 'weight_decay') else FLAGS.weight_decay

    if FLAGS.optimizer == 'admm':
        optimizer = torch.optim.Adam(params, lr=FLAGS.meta_lr,
                                    betas=(FLAGS.beta1, FLAG.beta2),
                                    eps=FLAGS.adam_eps,
                                    weight_decay = weight_decay)

    total_losses = []
    total_compute = []

    if hasattr(FLAGS, 'lr_drop_computes') and not FLAGS.drop_lr and FLAGS.lr_drop_computes is not None:
        lr_drop_computes = [int(s) for s in FLAGS.lr_drop_computes.split(',')]
    else:
        lr_drop_computes = [] 
    
    test_param_accumulators = []
    for p in params:
        test_param_accumulators.append(p.data.clone())
    test_counter = 0
    convergence_counter = 0
    total_loss = 0.
    compute = 0
    idx_counter = 0

    best_loss_so_far = None
    drop_counter = 0

    last_good_params = copy.deepcopy(params)

    max_norm = None

    step = 0 
    test_stats = None

    eval_headers = None

    telescope = NoTelescope(FLAGS.train_horizon+1)
    weight_fn = None

    if FLAGS.test_frequency is None:
        FLAGS.test_frequency = 1
    
    tflogger.log_scalar('meta_lr_by_compute',
                        optimizer.param_groups[0]['lr'],
                        int(compute/(2**FLAGS.train_horizon+1))) # defined in rt_runner.py
    logger.info("Running until compute > {}".format(
                2**FLAGS.test_horizon * FLAGS.budget))
    
    while idx_counter < 2**FLAGS.test_horizon * FLAGS.budget: #budget in rt_runner.py
        if FLAGS.linear_schedule:
            assert not FLAGS.rt
            n_increments = FLAGS.train_horizon + 1 - 2
            increment_size = float(2**FLAGS.test_horizon * FLAGS.budget) / n_increments
            increment_number = int(math.floor(idx_counter / increment_size)) + 2
            telescope = NoTelescope(increment_number + 1)
        
        if len(lr_drop_computes) > 0 and idx_counter > lr_drop_computes[0]:
            for pg in optimizer.param_groups:
                pg['lr'] /= 10 # learning rate decay
                logger.info("Dropping lr to {}".format(optimizer.param_groups[0]['lr']))
                tflogger.log_scalar('meta_lr_compute',
                                    optimizer.param_groups[0]['lr'],
                                    int(compute/(2**FLAGS.train_horizon+1)))
                if len(lr_drop_computes) > 1:
                    lr_drop_computes = lr_drop_computes[1:]
                else:
                    lr_drop_computes = []
        
        if idx_counter >= test_counter:
            with Timer() as t:
                # logger.info("Evaluating...")
                if FLAGS.averaged_test: #TBC
                    testparams = test_param_accumulators
                else:
                    testparams = params

                test_stats = eval_fn(
                    testparams, 2**(FLAGS.test_horizon) + 1,
                    tflogger, step)

                state = make_state_fn(2**FLAGS.test_horizon+1)
                test_horizon_loss, _ = train_loss_fn(state, params,
                                                    2 ** FLAGS.test_horizon+1)
                test_horizon_loss = test_horizon_loss.item()
                optimizer.zero_grad()

                if eval_headers is None:
                    eval_headers = sorted(list(test_stats.keys()))
                test_counter += (2**FLAGS.test_horizon +1) * FLAGS.test_frequency
                for k in eval_headers:
                    v = test_stats[k]
                    tflogger.log_scalar(k, v, step)
                    tflogger.log_scalar(k + '_by_compute', v,
                                        int(compute / (2**FLAGS.test_horizon + 1)))
                    if not np.isfinite(v):
                        logger.info("NAN test stat, reverting params")
                        params = last_good_params
                last_good_params = copy.deepcopy(params)
            logger.info("Test time: {}".format(t.interval))
        
        if idx_counter >= convergence_counter:
            with Timer() as t:
                losses = []
                grads_torch = []
                grads = []
                computes = []
                with Timer() as t2:
                    state = make_state_fn(2**FLAGS.train_horizon+1)
                    for j in range(FLAGS.train_horizon + 1):
                        l, g, c = loss_and_grads(
                            train_loss_fn, state,
                            params, optimizer, 2**(j) + 1)
                        if j > 0 and
                    

