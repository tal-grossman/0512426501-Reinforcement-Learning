import gym
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
from argparse import ArgumentParser

env = gym.make("CartPole-v1")
observation, info = env.reset(seed=42)

# According to section 2: init weight vector randomly [-1, 1]
INIT_WEIGHT_VEC = np.random.uniform(-1, 1, size=4)
# Maximum number of steps in an episode (according to section 3)
MAX_STEPS = 200
# Maximum number of episodes to train the agent (according to section 4)
MAX_EPISODES = 10000
# Number of runs for evaluation (according to section 5)
NUM_OF_RUNS_FOR_EVALUATION = 1000


def section2_agent_action(observation: np.ndarray, weight_vec: np.ndarray) -> int:
    """
    Agent action function for section 2.
    :param observation: The observation from the environment. shape=(4,)
    :param weight_vec: The weight vector. shape=(4,)
    """
    dot_product = np.dot(observation, weight_vec)
    action = 1 if dot_product >= 0 else 0
    return action


def run_single_episode(env, weight_vec) -> Tuple[int, dict]:
    """
    Run a single episode with the given environment and weight vector.
    :param env: The environment.
    :param weight_vec: The weight vector.
    :return: The total reward of the episode and the logging of the episode.
    """
    observation, info = env.reset()
    total_reward = 0
    log = {"terminated": False, "truncated": False, "info": None, "step": 0}
    for step in range(1, MAX_STEPS + 1):
        action = section2_agent_action(observation, weight_vec)
        observation, reward, terminated, truncated, info = env.step(action)
        log["step"] = step
        log["terminated"] = terminated
        log["truncated"] = truncated
        log["info"] = info
        log["observation"] = observation
        total_reward += reward
        if terminated or truncated:
            break
    return total_reward, log


def section_4_random_search(env, num_episodes: int) -> Tuple[np.ndarray, int]:
    """
    Train the agent using random search to find the best weight vector.
    :param env: The environment.
    :param num_episodes: The number of episodes to train the agent.
    :return: The best weight vector and the total reward of the best episode, also the episode number.
    """
    best_weight_vec = None
    best_total_reward = -np.inf
    for episode_num in range(1, num_episodes + 1):
        weight_vec = np.random.uniform(-1, 1, size=4)
        total_reward, _ = run_single_episode(env, weight_vec)
        if total_reward > best_total_reward:
            best_total_reward = total_reward
            best_weight_vec = weight_vec
        if total_reward == MAX_STEPS:
            break
    return best_weight_vec, best_total_reward, episode_num


def secion_5_random_search_evaluation(env, num_of_runs: int) -> float:
    """
    Evaluate the agent using random search.
    It runs the random search for the given number of runs and calculates the average number of episodes to reach the max reward.
    :param env: The environment.
    :param num_of_runs: The number of runs to evaluate the agent.
    :return: The average number of episodes to reach the max reward.
    """
    # get histogram of the num of episodes to reach the max reward of 200
    num_of_episodes_to_max_reward = []
    for run in range(1, num_of_runs + 1):
        best_weight_vec, best_total_reward, episode_num = section_4_random_search(
            env, num_episodes=MAX_EPISODES)
        if best_total_reward == MAX_STEPS:
            num_of_episodes_to_max_reward.append(episode_num)
    
    # calc the average number of episodes to reach the max reward
    avg_num_of_episodes_to_max_reward = np.mean(num_of_episodes_to_max_reward)

    # plot histogram of the num of episodes to reach the max reward of 200
    plt.hist(num_of_episodes_to_max_reward)
    plt.title(f"Histogram of the number of episodes to reach the max reward out of {num_of_runs} search runs", fontsize=16)
    plt.xlabel(f"Episode number to reach the max reward", fontsize=14)
    plt.ylabel('Histogram count', fontsize=14)
    plt.axvline(avg_num_of_episodes_to_max_reward, color='r', linestyle='dashed', linewidth=1)
    # add legend
    plt.legend([f'Average: {avg_num_of_episodes_to_max_reward}'], fontsize=14)
    plt.show()

    return avg_num_of_episodes_to_max_reward



def main():
    # section 3 - evaluate the for a single episode
    single_episode_reward, episode_log = run_single_episode(
        env, INIT_WEIGHT_VEC)
    # print the reward the logging of the episode
    print("### Section 3 - evaluate the for a single episode")
    print(f"Total reward: {single_episode_reward}")
    print(f"Episode logging: {episode_log}")

    # section 4 - train the agent using random search
    best_weight_vec, best_total_reward, episode_num = section_4_random_search(
        env, num_episodes=MAX_EPISODES)
    print("### Section 4 - train the agent using random search")
    print(f"Best weight vector: {best_weight_vec}")
    print(f"Best total reward: {best_total_reward}")
    print(f"Episode number: {episode_num}")

    # section 5 - evaluate the agent using random search
    avg_num_of_episodes_to_max_reward = secion_5_random_search_evaluation(env, num_of_runs=NUM_OF_RUNS_FOR_EVALUATION)
    print("### Section 5 - evaluate the agent using random search")
    print(f"Average number of episodes to reach the max reward: {avg_num_of_episodes_to_max_reward}")



def _get_parameters():
    parser = ArgumentParser()
    return parser.parse_args()


if __name__ == "__main__":

    args = _get_parameters()
    main()
