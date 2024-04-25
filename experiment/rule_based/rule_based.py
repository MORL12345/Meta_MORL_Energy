from simulator.residential_model import EnergyModel
import numpy as np
from algorithm.common.weights import equally_spaced_weights
import wandb

weights = equally_spaced_weights(2, n=100)
rules = [[0, 4],
         [4, 8],
         [8, 12],
         [12, 16],
         [16, 20]
         ]  # slot-based rules
if __name__ == '__main__':
    solutions = []
    EUs = []
    for rule in rules:
        background_power_demands = np.load("../../simulator/data/excluded_power_no_renew_kWh_14_15.npy")
        agg_background_power_demands = np.load("../../simulator/data/new_agg_no_renew_kWh_14_15.npy")
        renewable_generations = np.load("../../simulator/data/renewable_generation_kW_14_15_London.npy")
        energy_model = EnergyModel(background_power_demands=agg_background_power_demands,
                                   renewable_generations=renewable_generations,
                                   renewable_availability=True,
                                   task=4,
                                   power=1.5,
                                   day_scope=[0, 365],
                                   task_slot=[0, 8])

        bill = 0
        comfort = 0
        truncated = False
        obs, _ = energy_model.reset()
        clock = obs[1]
        utilities = []
        rew_vecs = []
        while not truncated:
            # print(clock)
            if rule[0] <= clock < rule[1]:
                action = 1
            else:
                action = 0
            n_obs, v_reward, terminated, truncated, _ = energy_model.step(action)
            # print(f"v_rew:{v_reward}")
            obs = n_obs
            clock = obs[1]
            hourly_bill = v_reward[0]
            bill += hourly_bill
            hourly_comfort = v_reward[1]
            comfort += hourly_comfort
        solutions.append((bill, comfort))
        for w in weights:
            u = np.dot(w, np.array([bill, comfort]))
            utilities.append(u)
        print(f"Rule-based agent @ rule:{rule} -- bill:£{bill:.2f} \t comfort:{comfort:.2f} were issued, EU:{np.mean(utilities)}")
        EUs.append(np.mean(utilities))
    # for i_EU in range(len(EUs)):
    #     name = "rule-based " + str(rules[i_EU])
    #     print(name)
    #     wandb.init(project='MO_Energy', entity='junlinlu16', name=name)
    #     EU = EUs[i_EU]
    #     rew_v = solutions[i_EU]
    #     for i in range(1000, 100001, 1000):
    #         wandb.log({'expected utility': EU,
    #                    'expected bill return': rew_v[0],
    #                    'expected comfort return': rew_v[1]}, step=i)
    #     wandb.finish()
    # np.save("../../simulator/solutions_points/rule_based_solutions.npy", solutions)
