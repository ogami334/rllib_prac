from ray import train, tune, air
from ray.rllib.algorithms.ppo import PPOConfig
import gymnasium as gym
from ray.rllib.algorithms.algorithm import Algorithm
from ray.air.integrations.wandb import WandbLoggerCallback


config = PPOConfig()
        .environment(env="CartPole-v1")  # 環境に関する設定を上書き
        .rollouts(num_rollout_workers=6)  # 訓練に関する設定を上書き
        .training(train_batch_size=20000)  # 経験収集に関する設定を上書き


tuner = tune.Tuner(
    "PPO",
    run_config=train.RunConfig(
        stop={"episode_reward_mean": 480},  # 学習の終了条件
        # callbacks=[WandbLoggerCallback(project="Wandb_nearme")]  # コメントアウトを外すとwandbにログが送信される(https://docs.wandb.ai/quickstart)
    ),
    param_space=config,
)
results = tuner.fit()
best_result = results.get_best_result(metric="episode_reward_mean", mode="max")
best_checkpoint = best_result.checkpoint
algo = Algorithm.from_checkpoint(best_checkpoint)

env = gym.make("CartPole-v1", render_mode="human")
episode_reward = 0
terminated = truncated = False

obs, info = env.reset()
while not terminated and not truncated:
    env.render()
    action = algo.compute_single_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    episode_reward += reward
print(f"Episode reward :{episode_reward}")
