import numpy as np
import gymnasium as gym
from gymnasium.utils import EzPickle
from gymnasium.spaces import Box, Discrete


class EnergyModel(gym.Env, EzPickle):
    def __init__(self, background_power_demands, renewable_generations, renewable_availability=False, task=4, power=1.5,
                 day_scope=[1, 30], task_slot=[0, 8]):
        super().__init__()
        # self.action_space = 2
        self.state_space = 4  # background_power_demand, time, remaining_task, renewable generation
        self.price = [36.62, 15.18]  # day price, night price
        self.task = task
        self.task_slot = task_slot
        self.background_power_demands = background_power_demands
        self.renewable_generations = renewable_generations
        self.renewable_availability = renewable_availability
        self.hour_scope = [(day_scope[0] - 1) * 24, (day_scope[1]) * 24]
        self.time = self.hour_scope[0]
        self.power = power
        self.observation_space = Box(
            low=np.zeros(4),
            high=np.array([100, 24, 24, 100]),
            dtype=np.float32,
        )
        self.action_space = Discrete(2)

        self.reward_space = Box(low=np.array([-1000, 0]), high=np.array([0, 120]), dtype=np.float32)
        print(f"day_scope: {day_scope}")

    def bill_calculate(self):
        bill = 0
        for i in range(len(self.background_power_demands)):
            time = i % 24
            if 8 <= time < 23:
                price = self.price[0]
            else:
                price = self.price[1]

            if self.renewable_availability:
                bill += price * max(self.background_power_demands[i] - self.renewable_generations[i], 0)
            else:
                bill += price * self.background_power_demands[i]
        return bill / 100

    def reset(self):
        self.time = self.hour_scope[0]
        obs = np.array([self.background_power_demands[self.hour_scope[0]],
                        self.time % 24,
                        self.task,
                        self.renewable_generations[self.hour_scope[0]]])
        return obs, {}


    def play_rule_based_episode(self):
        self.reset()
        truncated = False
        bill = 0
        while not truncated:
            action = 0
            if 4 <= self.time % 24 < 8:
                action = 1
            n_obs, v_reward, _, truncated, _ = self.step(action)
            bill += v_reward[0]
        print(f"run rule-based agent - get bill:£{bill}")

    def step(self, action):
        if self.time % 24 == 0:
            self.task = 4
        terminated = False
        truncated = False

        comfort = 0
        background_power_demand = self.background_power_demands[self.time]
        renewable_generation = self.renewable_generations[self.time]

        if self.renewable_availability:  # if there is renewable generation
            agg_power = max(background_power_demand - renewable_generation + self.power * action, 0)
        else:
            agg_power = background_power_demand + self.power * action

        #  If there are tasks left to complete, decrease the task hours when action equals 1,
        #  with the minimum task hours being 0.
        self.task -= action
        # self.task = max(self.task, 0)

        if 8 <= self.time % 24 < 23:
            price = self.price[0]  # day price
        else:
            price = self.price[1]  # night price

        hourly_bill = -price * agg_power / 100  # convert pennies to pounds

        # Running any of the 4 hours from 0:00 to 8:00 will earn comfort reward
        if self.task_slot[0] <= self.time % 24 < self.task_slot[1]:
            if self.task >= 0:
                comfort += 1 * action
        v_reward = np.array([hourly_bill, comfort])  # reward vector consists of hourly bill and comfort.

        self.time += 1  # increment the hours

        # when the hours reach the limit of the scope, truncate or terminate the episode
        if self.time >= self.hour_scope[1]:
            next_background_power_demand = background_power_demand
            next_renewable_generation = renewable_generation
            n_obs = np.array([next_background_power_demand, self.time % 24, self.task, next_renewable_generation])
            truncated = True
            if self.time > 365 * 24:
                terminated = True
        else:  # go to the next time step
            next_background_power_demand = self.background_power_demands[self.time]
            next_renewable_generation = self.renewable_generations[self.time]
            n_obs = np.array([next_background_power_demand, self.time % 24, self.task, next_renewable_generation])
        return n_obs, v_reward, terminated, truncated, {}  # obs: back_pd, clock, remaining_task, renew

    def render(self, mode="human"):
        print(f"current state 0 ")

    def close(self):
        pass


if __name__ == '__main__':
    background_power_demands = np.load("data/excluded_power_no_renew_kWh_14_15.npy")
    agg_background_power_demands = np.load("data/new_agg_no_renew_kWh_14_15.npy")
    renewable_generations = np.load("data/renewable_generation_kW_14_15_London.npy")
    energy_model = EnergyModel(background_power_demands=agg_background_power_demands,
                               renewable_generations=renewable_generations)

    print(f"£{energy_model.bill_calculate():.2f} bill issued")
    print(energy_model.step(1))
    energy_model.play_rule_based_episode()
