##octopus_arm_env
    octopus arm envrionment was used to make the octopus arm 

change the cloned folder name from octopus_arm to octopus_arm_env 

# Train
python3 octopus_arm_rl_env.py --mode train --timesteps 300000

# Resume from checkpoint
python3 octopus_arm_rl_env.py --mode train --load checkpoints/octopus_sac_100000_steps

# Evaluate
python3 octopus_arm_rl_env.py --mode eval \
    --load checkpoints/best/best_model \
    --vecnorm checkpoints/vec_normalize.pkl

#publish fruit postion so that RL can be trained
ros2 run octopus_arm_env fruit_position_publisher

#launch gazebo
ros2 launch octopus_arm_env octopus_arm_gazebo.launch.py