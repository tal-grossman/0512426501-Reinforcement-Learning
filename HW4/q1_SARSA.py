import numpy as np
import copy
import random
from q1_TD0 import BlackjackTD

class BlackjackSARSA:
    def __init__(self, alpha=0.1, gamma=1.0):
        self.alpha = alpha
        self.gamma = gamma
        self.Q = np.zeros((22, 2))  # States are from 2 to 21 (0-terminal for bust), actions are 0 (hit) and 1 (stand)
        self.policy = np.zeros(22)
        self.wins = 0
        self.episodes = 0

    def draw_card(self):
        card = np.random.randint(1, 14)
        return min(card, 10) if card != 1 else 11  # Ace is 11, face cards are 10

    def deal(self):
        card1, card2 = self.draw_card(), self.draw_card()
        while card1 + card2 > 21:
            card1, card2 = self.draw_card(), self.draw_card()
        return card1 + card2

    def simulate_episode(self, gambler_sum, house_sum):
        trajectory = []
        while True:
            a = np.argmax(self.Q[gambler_sum]) if random.random() > 0.2 else np.random.choice([0, 1])  # epsilon-greedy
            next_gambler_sum = gambler_sum + self.draw_card() if a == 0 else gambler_sum
            next_gambler_sum = 0 if next_gambler_sum > 21 else next_gambler_sum  # Gambler busts - terminal state
            next_a = np.argmax(self.Q[next_gambler_sum]) if random.random() > 0.2 else np.random.choice([0, 1])
            if next_gambler_sum == 0:
                reward = -1
                trajectory.append((gambler_sum, a, reward, next_gambler_sum, next_a))
                return trajectory

            # House's turn
            while house_sum < 16:
                house_sum += self.draw_card()
                if house_sum > 21:
                    reward = 1  # House busts
                    trajectory.append((gambler_sum, a, reward, next_gambler_sum, next_a))
                    return trajectory
            if next_gambler_sum > house_sum:
                reward = 1
                trajectory.append((gambler_sum, a, reward, next_gambler_sum, next_a))
                gambler_sum = next_gambler_sum
            elif next_gambler_sum < house_sum:
                reward = -1
                trajectory.append((gambler_sum, a, reward, next_gambler_sum, next_a))
                gambler_sum = next_gambler_sum
            else:
                reward = 0
                trajectory.append((gambler_sum, a, reward, next_gambler_sum, next_a))
                gambler_sum = next_gambler_sum

    def add_terminal_state(self, trajectory):
        last_reward, last_action, last_state = trajectory[-1], trajectory[-2], trajectory[-3]
        if last_reward == -1 and last_action == 'h':  # Gambler busts
            terminal = 0
            trajectory += [terminal]
        else:  # gambler wins, draw or gambler loses not due to busting
            trajectory += [last_state]
        return trajectory

    def sarsa_update(self, s, a, reward, next_s, next_a):
        self.Q[s, a] += self.alpha * (reward + self.gamma * self.Q[next_s, next_a] - self.Q[s, a])

    def train(self, episodes=100000):
        self.episodes = episodes
        for _ in range(episodes):
            flag = False
            gambler_sum = self.deal()
            house_sum = self.deal()
            trajectory = self.simulate_episode(gambler_sum, house_sum)
            for (s, a, r, next_s, next_a) in trajectory:
                if r == 1:
                    flag = True
                self.sarsa_update(s, a, r, next_s, next_a)
            if flag:
                self.wins += 1

    def compute_optimal_policy(self):
        self.policy = np.argmax(self.Q, axis=1)


    def get_win_probability(self, episodes=100000):
        for _ in range(episodes):
            gambler_sum = self.deal()
            house_sum = self.deal()
            trajectory = self.simulate_episode(gambler_sum, house_sum)
            trajectory = self.add_terminal_state(trajectory)
            while len(trajectory) > 3:
                state, action, reward, next_state = trajectory[:4]
                if reward == 1:
                    self.wins += 1
                self.td_update(state, reward, next_state)
                trajectory = trajectory[3:]

def optimal_policy_to_dict(policy):
    return {state+4: 'hit' if action == 0 else 'stand' for state, action in enumerate(policy[4:])}

if __name__ == "__main__":
    blackjack_sarsa = BlackjackSARSA(alpha=0.01, gamma=0.8)
    blackjack_sarsa.train(episodes=300000)
    blackjack_sarsa.compute_optimal_policy()
    print(f'The optimal policy is: {optimal_policy_to_dict(blackjack_sarsa.policy)}')
    blackjack_td = BlackjackTD(alpha=0.01, gamma=0.8)
    blackjack_td.policy = blackjack_sarsa.policy
    blackjack_td.train(episodes=300000)
    print(f"Probability of winning with the optimal policy: {blackjack_td.get_win_probability()}")
