import numpy as np
import copy

class BlackjackTD:
    def __init__(self, alpha=0.1, gamma=1.0):
        self.alpha = alpha
        self.gamma = gamma
        self.values = np.zeros(22)  # States are from 2 to 21, index 0 and 1 are unused
        self.policy = self.define_policy()
        self.wins = 0
        self.episodes = 0

    def define_policy(self):
        policy = np.zeros(22, dtype=str)
        policy[2:18] = 'h'
        policy[18:22] = 's'
        return policy

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
        # Gambler's turn
        while self.policy[gambler_sum] == 'h':
            trajectory += [gambler_sum, 'h', None]
            gambler_sum += self.draw_card()
            if gambler_sum > 21:
                trajectory[-1] = -1
                return trajectory # Gambler busts
            trajectory[-1] = 0
        trajectory += [gambler_sum, 's', None]
        # House's turn
        while house_sum < 16:
            house_sum += self.draw_card()
            if house_sum > 21:
                trajectory[-1] = 1  # House busts
                return trajectory

        if gambler_sum > house_sum:
            trajectory[-1] = 1  # Gambler wins
            return trajectory
        elif gambler_sum < house_sum:
            trajectory[-1] = -1
            return trajectory
        else:
            trajectory[-1] = 0
            return trajectory

    def add_terminal_state(self, trajectory):
        trajectory += [21]
        return trajectory
    def td_update(self, state, reward, next_state):
        self.values[state] += self.alpha * (reward + self.gamma * self.values[next_state] - self.values[state])

    def train(self, episodes=100000):
        self.episodes = episodes
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


    def get_win_probability(self):
        return self.wins / self.episodes

if __name__ == "__main__":
    blackjack_td = BlackjackTD(alpha=0.1, gamma=0.8)
    blackjack_td.train(episodes=100000)
    print(f"Probability of winning: {blackjack_td.get_win_probability()}")
