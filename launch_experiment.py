from gymnasium.envs.registration import register
import gymnasium as gym
import numpy as np
import random
from experiment.Meta_GPI_PD.meta_gpi_pd import GPIPD
import torch
import argparse

seeds = [
    3,
    11,
    23,
    42,
    50
]
if __name__ == '__main__':  # seed used should be: 42, 11, 3, 50, 23
    parser = argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('--local',type=int, default=0, help="Local running")
    parser.add_argument('--baseline', type=str, help='Baseline name')
    parser.add_argument('--total_step', type=str, help='Total Step')
    parser.add_argument('--eval_step', type=str, help='Eval Step')
    parser.add_argument('--save_per', type=str, help='Save Per')
    parser.add_argument('--log', type=int, help='Log')
    parser.add_argument('--meta', type=int, help='Meta')
    parser.add_argument('--start_day', type=int, help='start day', default=0)
    parser.add_argument('--end_day', type=int, help='end day', default=365)
    parser.add_argument('--joint_train', type=int, help='joint_train?', default=0)
    args = parser.parse_args()
    baseline = args.baseline
    meta = args.meta
    total_steps = int(args.total_step)
    if meta == 1:
        total_steps = 100
    eval_step = int(args.eval_step)
    save_per = int(args.save_per)
    log = int(args.log)==1
    start_day = args.start_day
    end_day = args.end_day
    day_scope = [start_day, end_day]
    local =int(args.local)==1
    joint_train = int(args.joint_train)==1
    wandb_entity = "username"
    project_name ="MO_Energy"
    if local:
        prefix = "meta_learning_model_local/"
    else:
        prefix = "/home/username/MO_Energy/agent/meta_learning/model/"
    for seed in seeds:
        print(f"Experiment {baseline} @ {day_scope} starts--->\tbaseline:{baseline}\tseed:{seed}\tday_scope:{day_scope}\tmeta:{meta}")
        save_to = "agent_model/" + baseline + "/" + str(seed)+"_"+str(joint_train) + "/models/"
        register(
            id='EnergyModel-v0',
            entry_point='simulator.residential_model:EnergyModel',
        )
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

        background_power_demands = np.load("simulator/data/excluded_power_no_renew_kWh_14_15.npy")
        agg_background_power_demands = np.load("simulator/data/new_agg_no_renew_kWh_14_15.npy")
        renewable_generations = np.load("simulator/data/renewable_generation_kW_14_15_London.npy")
        print(f"{background_power_demands}")
        anomaly_weeks = [0, 4, 6, 8, 10, 12, 16, 23, 29, 33, 38, 51]
        if joint_train:
            background = []
            agg_back = []
            renewable = []
            for week in anomaly_weeks:
                background.append(background_power_demands[week*7:week*7+24])
                agg_back.append(agg_background_power_demands[week * 7:week * 7 + 24])
                renewable.append(renewable_generations[week * 7:week * 7 + 24])
            agg_background_power_demands = np.array(agg_back).flatten()
            renewable_generations = np.array(renewable).flatten()
        # print(f"back:{renewable_generations}")
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


        if "GPI" in baseline:
            if baseline == "GPI_PD":
                gpi_pd = True
                agent_name = "GPI_PD"
                experiment_name = "GPI_PD_" + str(seed)
            else:
                gpi_pd = False
                agent_name = "GPI_LS"
                experiment_name = "GPI_LS_" + str(seed)

            print(f"experiment name: {experiment_name}")
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
                epsilon_decay_steps=int(total_steps*0.8),
                learning_starts=100,
                alpha_per=0.6,
                min_priority=0.01,
                per=gpi_pd,
                gpi_pd=gpi_pd,
                use_gpi=True,
                gradient_updates=1,
                target_net_update_freq=200,
                tau=1,
                dyna=gpi_pd,
                dynamics_uncertainty_threshold=1.5,
                dynamics_net_arch=[256, 256, 256],
                dynamics_buffer_size=int(1e5),
                dynamics_rollout_batch_size=25000,
                dynamics_train_freq=lambda t: 250,
                dynamics_rollout_freq=250,
                dynamics_rollout_starts=100,
                dynamics_rollout_len=1,
                real_ratio=0.5,
                log=log,
                experiment_name=experiment_name,
                seed=seed,
                wandb_entity=wandb_entity,
                project_name=project_name
            )

            if meta == 1:
                anomaly_weeks = [0, 4, 6, 8, 10, 12, 16, 23, 29, 33, 38, 51]
                day_scopes = []
                for w in anomaly_weeks:
                    day_scopes.append([w * 7, w * 7 + 1])
                agent.reptile_train(agg_background_power_demands=agg_background_power_demands,
                                    renewable_generations=renewable_generations,
                                    day_scopes=day_scopes,
                                    inner_episodes=3,
                                    n_iterations=3,
                                    n_task=10,
                                    outer_learning_rate=3e-4,
                                    save_to=prefix + baseline + "/" + str(seed) + "/",
                                    save_per=100)
            if meta == 2: #CRL
                # plain_fine_tune_path = "/home/username/MO_Energy/plain_gpi_model/"+baseline+"/"+str(seed)+"/models/39600.pth"
                meta_fine_tune_path = "/home/username/MO_Energy/agent/meta_learning/model/" + baseline + "/" + str(
                                               seed) + "/final/"+str(seed)+"_2.pth"
                meta_fine_tune_path = "/home/username/MO_Energy/agent/meta_learning/model/" + baseline + "/" + str(
                    seed) + "/final/" + str(seed) + "_2.pth"
                solutions = agent.continual_rl_train(seed=seed,
                                                     agg_background_power_demands=agg_background_power_demands,
                                                     renewable_generations=renewable_generations,
                                                     original_model_path=meta_fine_tune_path)
            else:
                agent.train(
                    total_timesteps=total_steps,
                    save_per=save_per,
                    eval_env=eval_env,
                    ref_point=np.array([0.0, 0.0, -200.0]),
                    weight_selection_algo="gpi-ls",
                    num_eval_episodes_for_front=1,
                    timesteps_per_iter=720,
                    save_to=save_to,
                    eval_freq=eval_step,
                    reward_dim=2,
                    granularity=100,
                )



