import numpy as np


class BlackjackTD:
    def __init__(self, alpha=0.1, gamma=1.0):
        self.alpha = alpha
        self.gamma = gamma
        self.values = np.zeros(22)  # States are from 2 to 21, index 0 and 1 are unused
        self.policy = self.define_policy()

    def define_policy(self):
        policy = np.zeros(22, dtype=str)
        policy[2:18] = 'h'
        policy[18:22] = 's'
        return policy

    def draw_card(self):
        card = np.random.randint(1, 14)
        return min(card, 10) if card != 1 else 11  # Ace is 11, face cards are 10

    def deal(self):
        return self.draw_card() + self.draw_card()

    def simulate_episode(self):
        gambler_sum = self.draw_card() + self.draw_card()
        house_sum = self.draw_card() + self.draw_card()

        # Gambler's turn
        while self.policy[gambler_sum] == 'h':
            gambler_sum += self.draw_card()
            if gambler_sum > 21:
                return gambler_sum, house_sum, -1  # Gambler busts

        # House's turn
        while house_sum < 16:
            house_sum += self.draw_card()
            if house_sum > 21:
                return gambler_sum, house_sum, 1  # House busts

        if gambler_sum > house_sum:
            return gambler_sum, house_sum, 1  # Gambler wins
        elif gambler_sum < house_sum:
            return gambler_sum, house_sum, -1  # Gambler loses
        else:
            return gambler_sum, house_sum, 0  # Draw

    def td_update(self, state, reward, next_state):
        self.values[state] += self.alpha * (reward + self.gamma * self.values[next_state] - self.values[state])

    def train(self, episodes=100000):
        for _ in range(episodes):
            gambler_sum = self.deal()
            house_sum = self.deal()
            while state < 22:
                gambler_sum, house_sum, reward = self.simulate_episode()
                next_state = gambler_sum if gambler_sum < 22 else 21
                self.td_update(state, reward, next_state)
                state = next_state

    def get_win_probability(self):
        win_states = np.sum(self.values[18:22])
        total_states = np.sum(self.values[2:22])
        return win_states / total_states


if __name__ == "__main__":
    blackjack_td = BlackjackTD(alpha=0.1, gamma=0.8)
    blackjack_td.train(episodes=100)
    print(f"Probability of winning: {blackjack_td.get_win_probability()}")
