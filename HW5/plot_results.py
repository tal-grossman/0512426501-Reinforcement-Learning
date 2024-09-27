import os
import copy
import pickle
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def plot_results(names_list, pkl_files_paths, output_dir):
    data = []
    for name, pkl_file_path in zip(names_list, pkl_files_paths):
        # load the pkl file
        with open(pkl_file_path, 'rb') as f:
            data.append(pickle.load(f))

    # plot the data so we have to plots in the same figure
    # 1. for mean_episode_rewards
    # 2. for best_mean_episode_rewards
    # each plot should have all the data from the pkl files labeled with the name in the same idx in names_list

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    for name, d in zip(names_list, data):
        mean_episode_rewards = d['mean_episode_rewards']
        best_mean_episode_rewards = d['best_mean_episode_rewards']
        time = range(len(mean_episode_rewards))
        max_mean_episode_rewards = max(mean_episode_rewards)
        max_best_mean_episode_rewards = max(best_mean_episode_rewards)
        ax1.plot(time, mean_episode_rewards, label=f"{name}")
        ax2.plot(time, best_mean_episode_rewards, label=f"{name}, max: {max_best_mean_episode_rewards:.1f}")

    ax1.set_ylabel("Mean Episode Rewards")
    ax1.set_title("Mean Episode Rewards Over Steps")
    ax2.set_ylabel("Best Mean Episode Rewards")
    ax2.set_title("Best Mean Episode Rewards Over Steps")
    for ax in [ax1, ax2]:
        ax.legend()
        ax.legend(loc='lower right')
        ax.set_xlabel("steps [1m]")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x / 1e6):,}'))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(5))  # Set major ticks every 5 units on the y-axis
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        ax.axhline(y=10, color='black', linestyle='--', linewidth=0.5)
        ax.axhline(y=15, color='black', linestyle='--', linewidth=0.5)
        ax.axhline(y=17.5, color='black', linestyle='--', linewidth=0.5)
        ax.axhline(y=20, color='black', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    # plt.show()  
    output_path = os.path.join(output_dir, 'results_plots.png')
    plt.savefig(output_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--names_list', type=str, default=[], help='names list as list', nargs='+')
    parser.add_argument('--pkl_files_paths', type=str, default=[], help='pkl files paths as list',
                        nargs='+')
    parser.add_argument('--output_dir', type=str, default='results/', help='Output directory', required=False)

    return parser.parse_args()


if __name__ == '__main__':
    NAME2FOLDER = {
        "gamma=1": "gamma_1",
        "gamma=0.97": "gamma_0.97",
        "gamma=0.99": "base_14_hours_training",
        "gamma=0.95": "gamma_0.95",
        "gamma=0.9": "gamma_0.9",
        "gamma=0.5": "gamma_0.5"
    }
    parser = argparse.ArgumentParser()
    args = parse_args()
    args.names_list = ["gamma=1", "gamma=0.97", "gamma=0.99", "gamma=0.95", "gamma=0.9", "gamma=0.5"]
    args.pkl_files_paths = [os.path.join(r'C:\Users\moshey\Downloads\HW5', NAME2FOLDER[folder], 'statistics.pkl')
                            for folder in args.names_list]
    names_list = args.names_list
    pkl_files_paths = args.pkl_files_paths
    output_dir = args.output_dir
    assert len(names_list) == len(pkl_files_paths), "names_list and pkl_files_paths must have the same length"

    plot_results(names_list, pkl_files_paths, output_dir)
