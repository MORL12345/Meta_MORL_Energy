from gymnasium.envs.registration import register
import gymnasium as gym
from experiment.conditional_DQN.conditioned_DQN import ConditionedDQNAgent
import numpy as np
import random
from experiment.Meta_GPI_PD.meta_gpi_pd import GPIPD
from algorithm.common.weights import equally_spaced_weights
import torch
import argparse
from matplotlib import pyplot as plt

if __name__ == '__main__':  # seed used should be: 42, 11, 3, 50, 23
    seeds = [3,11,23,42,50]

    background_power_demands = np.load("simulator/data/excluded_power_no_renew_kWh_14_15.npy")
    agg_background_power_demands = np.load("simulator/data/new_agg_no_renew_kWh_14_15.npy")
    renewable_generations = np.load("simulator/data/renewable_generation_kW_14_15_London.npy")
    # plt.plot(renewable_generations)
    # plt.show()
    eval_weights = equally_spaced_weights(2, n=100)

    parser = argparse.ArgumentParser(description='Process some arguments.')
    # parser.add_argument('--seed', type=int, help='random seed', default=42)
    parser.add_argument('--meta', type=int, help='1 = meta; 0 = no meta', default=1)
    parser.add_argument('--gpi_pd', type=int, help='1 = gpi_pd; 0 = gpi_ls', default=1)
    parser.add_argument('--start_day',type=int, help='start day', default=0)
    parser.add_argument('--end_day', type=int, help='end day', default=365)
    parser.add_argument('--checkpoint', type=int, help='checkpoint', default=40000)
    parser.add_argument('--local', type=int, help='local running', default=0)
    # parser.add_argument('--gpi_pd', type=bool, help='whether use GPI_PD or stay with GPI_LS', default=True)
    args = parser.parse_args()

    for seed in seeds:
    # seed = args.seed
        meta = args.meta
        gpi_pd = args.gpi_pd
        start_day = args.start_day
        end_day = args.end_day
        day_scope = [start_day, end_day]
        check_point = args.checkpoint
        local_running = args.local
        if gpi_pd==1:
            baseline = "GPI_PD"
            experiment_name = "GPI_PD_" + str(seed)+"_"
        else:
            baseline = "GPI_LS"
            experiment_name = "GPI_LS_" + str(seed)+"_"

        if meta == 1:
            meta_prefix = "meta"
            model_path_prefix = ""
        else:
            meta_prefix = "no_meta"
            model_path_prefix = "plain_gpi_model"
        experiment_name = meta_prefix+"_"+experiment_name
        # check_point_path = model_path_prefix+"/"+baseline+"/"+str(seed)+"_True/models/"+str(check_point)+".pth"
        check_point_path = "agent_model/" + baseline + "/" + str(seed) + "_True/models/" + str(check_point) + ".pth"
        if local_running==0:
            check_point_path="/home/username/MO_Energy/"+check_point_path

        print(f"Experiment {experiment_name} @ {day_scope} starts--->\tbaseline:{baseline}\tseed:{seed}\tcheckpoint:{check_point}")
        register(
            id='EnergyModel-v0',
            entry_point='simulator.residential_model:EnergyModel',
        )
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

        print(f"Annual result:\tbaseline:{baseline}\tseed:{seed}\tmeta={meta_prefix}\tday_scope:{day_scope}")
        env = gym.make('EnergyModel-v0',
                       background_power_demands=agg_background_power_demands,
                       renewable_generations=renewable_generations,
                       renewable_availability=True,
                       task=4,
                       power=1.5,
                       day_scope=day_scope,
                       task_slot=[0, 8])

        eval_env = gym.make('EnergyModel-v0',
                            background_power_demands=agg_background_power_demands,
                            renewable_generations=renewable_generations,
                            renewable_availability=True,
                            task=4,
                            power=1.5,
                            day_scope=day_scope,
                            task_slot=[0, 8])
        # if baseline == "GPI_PD":

        agent = GPIPD(
                env,
                num_nets=2,
                max_grad_norm=None,
                learning_rate=3e-4,
                gamma=1,
                batch_size=128,
                net_arch=[256, 256, 256, 256],
                buffer_size=int(2e5),
                initial_epsilon=1.0,
                final_epsilon=0.05,
                epsilon_decay_steps=80000,
                learning_starts=100,
                alpha_per=0.6,
                min_priority=0.01,
                per=True,
                gpi_pd=gpi_pd==1,
                use_gpi=False,
                gradient_updates=1,
                target_net_update_freq=200,
                tau=1,
                dyna=gpi_pd==1,
                dynamics_uncertainty_threshold=1.5,
                dynamics_net_arch=[256, 256, 256],
                dynamics_buffer_size=int(1e5),
                dynamics_rollout_batch_size=25000,
                dynamics_train_freq=lambda t: 250,
                dynamics_rollout_freq=250,
                dynamics_rollout_starts=100,
                dynamics_rollout_len=1,
                real_ratio=0.5,
                log=False,
                experiment_name=experiment_name,
                seed=seed,
                wandb_entity="username",
                project_name="MO_Energy"
            )



        EU = 0
        agent.q_nets = torch.load(check_point_path, map_location=agent.device)
        for i in range(len(eval_weights)):
            w = eval_weights[i]
            print(f"normal evaluating for w:{w}")

            cumulative_rews = agent.play_a_episode_(weight=w,
                                                    eval_env=eval_env)
            print(f"w:{w}\tcumulative_rews:{cumulative_rews}")
            EU += np.dot(w, cumulative_rews)

        print(f"EU:{EU/len(eval_weights)}")