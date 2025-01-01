import gymnasium as gym

gym.register(
    id="NaoStandup-v1",
    entry_point="nao_env:NaoStandup",
    max_episode_steps=2500,
)

config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 1000000,
    "env_name": "NaoStandup-v1",
    "render_mode": "rgb_array",
}

env = gym.make(config["env_name"], render_mode=config["render_mode"])

# print("Observation space:")
# print(env.observation_space, "\n")
#
# print("Observation space shape:")
# print(env.observation_space.shape, "\n")
#
# print("Action space:")
# print(env.action_space, "\n", "\n")
#
#
# print("Metadata:")
# print(env.metadata, "\n", "\n")

action = env.action_space.sample()
env.reset()
env.step(action)


# joints = [
#     "HeadPitch",
#     "HeadYaw",
#     "LAnklePitch",
#     "LAnkleRoll",
#     "LElbowRoll",
#     "LElbowYaw",
#     "LHipPitch",
#     "LHipRoll",
#     "LHipYawPitch",
#     "LKneePitch",
#     "LShoulderPitch",
#     "LShoulderRoll",
#     "LWristYaw",
#     "RAnklePitch",
#     "RAnkleRoll",
#     "RElbowRoll",
#     "RElbowYaw",
#     "RHipPitch",
#     "RHipRoll",
#     "RHipYawPitch",
#     "RKneePitch",
#     "RShoulderPitch",
#     "RShoulderRoll",
#     "RWristYaw",
# ]
# print(data.qpos.flat[7:])
