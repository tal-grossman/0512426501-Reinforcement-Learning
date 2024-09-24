import gym
import torch.optim as optim

import os
import argparse
from dqn_model import DQN
from gym.wrappers import Monitor
from dqn_learn import OptimizerSpec, dqn_learing
from utils.gym import get_env, get_wrapper_by_name
from utils.schedule import LinearSchedule

BATCH_SIZE = 32
GAMMA = 0.99
REPLAY_BUFFER_SIZE = 1000000
LEARNING_STARTS = 50000
LEARNING_FREQ = 4
FRAME_HISTORY_LEN = 4
TARGER_UPDATE_FREQ = 10000
LEARNING_RATE = 0.00025
ALPHA = 0.95
EPS = 0.01

# we going to run it for these values of gamma
# gamma_1_low = 0.5
# gamma_2_mid = 0.9
# gamma_3_high = 0.99999

def main(env, num_timesteps, run_output_dir, gamma):

    def stopping_criterion(env):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps

    optimizer_spec = OptimizerSpec(
        constructor=optim.RMSprop,
        kwargs=dict(lr=LEARNING_RATE, alpha=ALPHA, eps=EPS),
    )

    exploration_schedule = LinearSchedule(1000000, 0.1)

    dqn_learing(
        run_output_dir,
        env=env,
        q_func=DQN,
        optimizer_spec=optimizer_spec,
        exploration=exploration_schedule,
        stopping_criterion=stopping_criterion,
        replay_buffer_size=REPLAY_BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        gamma=gamma,
        learning_starts=LEARNING_STARTS,
        learning_freq=LEARNING_FREQ,
        frame_history_len=FRAME_HISTORY_LEN,
        target_update_freq=TARGER_UPDATE_FREQ,
    )

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, help='run_name name', required=True)
    parser.add_argument('--output_dir', type=str, default='results/',
                        help='Output directory', required=False)
    parser.add_argument('--gamma', type=float, default=GAMMA)

    return parser.parse_args()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    args = parse_args()

    run_name = args.run_name
    output_dir = args.output_dir
    gamma = args.gamma
    run_output_dir = os.path.join(output_dir, run_name)
    # create output dir and change the env output dir to that one
    os.makedirs(run_output_dir, exist_ok=True)

    # Get Atari games.
    benchmark = gym.benchmark_spec('Atari40M')

    # Change the index to select a different game.
    task = benchmark.tasks[3]

    # Run training
    seed = 0 # Use a seed of zero (you may want to randomize the seed!)
    
    env = get_env(task, seed, run_output_dir)

    main(env, task.max_timesteps, run_output_dir, gamma)
