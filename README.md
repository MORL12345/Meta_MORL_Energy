# A Meta-Learning Approach for Multi-Objective Reinforcement Learning in Sustainable Home Environments

This is the implementation repository for review.

Effective residential appliance scheduling is crucial for sustainable living. While multi-objective reinforcement learning (MORL) has proven effective in balancing user preferences in appliance scheduling, traditional MORL struggles with limited data in non-stationary residential settings characterized by renewable generation variations.  Significant context shifts that can invalidate previously learned policies. To address these challenges, we extend state-of-the-art MORL algorithms with the meta-learning paradigm, enabling rapid, few-shot adaptation to shifting contexts. Additionally, we employ an auto-encoder (AE)-based unsupervised method to detect environment context changes. We have also developed a residential energy environment to evaluate our method using real-world data from London residential settings. This study not only assesses the application of MORL in residential appliance scheduling but also underscores the effectiveness of meta-learning in energy management. Our top-performing method significantly surpasses the best baseline,  while the trained model saves 3.28\% on electricity bills, a 2.74\% increase in user comfort, and a 5.9\% improvement in expected utility. Additionally, it reduces the sparsity of solutions by 62.44\%. Remarkably, these gains were accomplished using 96.71\% less training data and 61.1\% fewer training steps.

We upload the general version of the implementation. The user need to tailor it to the specific experiment they want to do. The dependencies are exported to M_GPI.yml To reconstruct the running environment on your machine: Please do:

conda env create -f M_GPI.yml

The main algorithm is in \experiment\Meta_GPI_PD\meta_gpi_pd.py . You can go to lauch_experiment.py to run the experiment. We give a example of the arguments for main function of Joint Training GPI-PD experiment. 

--baseline
GPI_PD
--total_step
40000
--eval_step
5000
--save_per
5000
--log
0
--meta
0
--joint_train
1
--start_day
1
--end_day
12
--local
1


This implementation is adapted from the work of Alegre et al. "Sample-Efficient Multi-Objective Learning via Generalized Policy Improvement Prioritization" Lucas N. Alegre, Ana L. C. Bazzan, Diederik M. Roijers, Ann Nowé, Bruno C. da Silva AAMAS 2023 Paper: https://arxiv.org/abs/2301.07784
Also please refer to this implementation for evaluations for hypervolume, sparsity etc.
