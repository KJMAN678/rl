import numpy as np
from tqdm import tqdm


class Environment:
    def __init__(self, grid_size=5, reward_location=(3, 3), reward_amount=10):
        # 環境の初期化
        self.grid_size = grid_size
        self.reward_location = reward_location
        self.reward_amount = reward_amount
        self.state = (grid_size - 1, grid_size - 1)
        self.actions = ["up", "down", "left", "right"]

    def step(self, action):
        # 環境に対するアクションの結果を返す
        if action == "up":
            self.state = (max(self.state[0] - 1, 0), self.state[1])
        elif action == "down":
            self.state = (min(self.state[0] + 1, self.grid_size - 1), self.state[1])
        elif action == "left":
            self.state = (self.state[0], max(self.state[1] - 1, 0))
        elif action == "right":
            self.state = (self.state[0], min(self.state[1] + 1, self.grid_size - 1))

        reward = self.reward_amount if self.state == self.reward_location else -1

        if self.state == self.reward_location:
            done = True
        else:
            done = False

        # ここでは状態、報酬、終了フラグを返す
        return self.state, reward, done

    def reset(self):
        self.state = (self.grid_size - 1, self.grid_size - 1)
        return self.state


class Agent:
    def __init__(self, env, learning_rate=0.5, discount_factor=0.9, exploration_rate=1.0):
        # エージェントの初期化
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.q_table = np.zeros((env.grid_size, env.grid_size, len(env.actions)))

    def policy(self, state):
        # 観測に基づいてアクションを決定するポリシー
        if np.random.uniform(0, 1) < self.exploration_rate:
            action = np.random.choice(self.env.actions)
        else:
            action = np.argmax(self.q_table[state[0], state[1]])
        return action

    def update(self, state, action, reward, next_state):
        # 学習アルゴリズムに基づいてエージェントを更新する
        old_value = self.q_table[state[0], state[1], self.env.actions.index(action)]
        future_max_value = np.max(self.q_table[next_state[0], next_state[1]])

        new_value = (1 - self.learning_rate) * old_value + self.learning_rate * (
            reward + self.discount_factor * future_max_value
        )
        self.q_table[state[0], state[1], self.env.actions.index(action)] = new_value


def main():
    num_episodes = 50

    # 環境とエージェントの初期化
    env = Environment()
    agent = Agent(env)
    done = False

    # エピソードのループ
    for episode in tqdm(range(num_episodes)):
        state = env.reset()

        # ステップのループ
        with tqdm() as pbar:
            while not done:
                pbar.update(1)

                action = agent.policy(state)
                next_state, reward, done = env.step(action)
                agent.update(state, action, reward, next_state)
                state = next_state


if __name__ == "__main__":
    main()
