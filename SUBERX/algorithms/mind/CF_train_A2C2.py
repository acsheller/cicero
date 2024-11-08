import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='torch')
import argparse
import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3 import A2C
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import TensorDict
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
from typing import Callable, Tuple
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor


from algorithms.wrappers import StableBaselineWrapperNum
from environment.mind.configs import get_enviroment_from_args, get_base_parser
from environment import load_LLM
from algorithms.logging_config import get_logger




logger = get_logger("suber_logger")

# Set environment variable to avoid tokenizer parallelism issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Define arguments
def parse_args():
    parser = get_base_parser()
    parser.add_argument("--model_device", type=str, default="cuda:0")
    parser.add_argument("--gamma", type=float, default=0.975)
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.01)

    parser.add_argument(
        '--news_dataset', 
        choices=['mind_dataset', 'small_mind_dataset'], 
        help='Specify the news dataset to use',
        default='small_mind_dataset'
    )
    # TODO Parser arguments should be here -- need to consider as most of them are 
    # in config.py

    args = parser.parse_args()
    return args

# Linear schedule function
def linear_schedule(initial_value: float):
    def func(progress_remaining: float) -> float:
        current_value = initial_value * progress_remaining
        #print(f"Linear schedule called: progress_remaining={progress_remaining}, learning_rate={current_value}")
        #logger.info(f"Linear schedule called: progress_remaining={progress_remaining}, learning_rate={current_value}")
        return current_value
    return func

class CombinedCallback(BaseCallback):
    def __init__(self, save_freq=50000, log_freq=5000, save_path="./tmp/models/", name_prefix="rl_model", verbose=0):
        super(CombinedCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.log_freq = log_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.metrics = []

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            rewards = self.locals['rewards']
            episode_length = self.locals.get('episode_lengths', None)
            value_loss = self.locals.get('value_loss', None)
            policy_loss = self.locals.get('policy_loss', None)
            if value_loss is not None:
                log_message = {
                    "step": self.num_timesteps,
                    "reward": rewards,
                    "episode_length": episode_length,
                    "value_loss": value_loss,
                    "policy_loss": policy_loss
                }
                self.metrics.append(log_message)
                self.logger.info(log_message)


        if self.n_calls % self.save_freq == 0:
            model_path = f"{self.save_path}/{self.name_prefix}_{self.num_timesteps}_steps"
            self.model.save(model_path)
            if self.verbose > 0:
                print(f"Saving model checkpoint to {model_path}")
        return True

class Net(nn.Module):
    def __init__(self, obs_space: gym.spaces.Space, num_users: int, num_items: int, learning_rate: float = 0.001):
        super().__init__()
        embedding_dim = args.embedding_dim
        self.latent_dim_pi = embedding_dim * 2
        self.latent_dim_vf = embedding_dim * 2

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)

        self.policy_net = nn.Sequential(
            nn.Linear(self.user_embedding.embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(embedding_dim * 4, num_items)
        )

        self.value_net = nn.Sequential(
            nn.Linear(self.user_embedding.embedding_dim + num_items, self.latent_dim_vf * 2),
            nn.ReLU(),
            nn.Linear(self.latent_dim_vf * 2, embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(embedding_dim * 4, self.latent_dim_vf),
            nn.ReLU()
        )

    def forward(self, features: TensorDict) -> Tuple[torch.Tensor, torch.Tensor]:
        user_id = features["user_id"].squeeze(1)
        news_seen = features["items_interact"]

        user_embedding = self.user_embedding(user_id)
        user_embedding_value = torch.cat([user_embedding, news_seen], dim=1)
        user_bias = self.user_bias(user_id)

        mask = features["items_interact"].to(dtype=torch.bool)
        logits = self.policy_net(user_embedding) + user_bias
        logits[mask] = -torch.inf
        return logits, self.value_net(user_embedding_value)

    def forward_actor(self, features: TensorDict) -> torch.Tensor:
        user_id = features["user_id"].squeeze(1)
        user_embedding = self.user_embedding(user_id)
        user_bias = self.user_bias(user_id)

        mask = features["items_interact"].to(dtype=torch.bool)
        logits = self.policy_net(user_embedding) + user_bias
        logits[mask] = -torch.inf
        return logits

    def forward_critic(self, features: TensorDict) -> torch.Tensor:
        user_id = features["user_id"].squeeze(1)
        news_seen = features["items_interact"]

        user_embedding = self.user_embedding(user_id)
        user_embedding_value = torch.cat([user_embedding, news_seen], dim=1)
        return self.value_net(user_embedding_value)

class DistributionUseLogitsDirectly(CategoricalDistribution):
    def __init__(self, action_dim: int):
        super().__init__(action_dim)

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        return nn.Identity(latent_dim)

class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, observation_space: spaces.Space, action_space: spaces.Space, lr_schedule: Callable[[float], float], *args, **kwargs):
        kwargs["ortho_init"] = True
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

        self.action_dist = DistributionUseLogitsDirectly(action_space.n)
        self._build(lr_schedule)

    def _build_mlp_extractor(self) -> None:
        default_lr = 0.01
        num_users = train_env.get_wrapper_attr('num_users')
        num_items = train_env.get_wrapper_attr('num_items')
        self.mlp_extractor = Net(self.observation_space, num_users, num_items, learning_rate=default_lr)

class ExtractPass(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space) -> None:
        super().__init__(observation_space, get_flattened_obs_dim(observation_space))
        self.flatten = nn.Flatten()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        observations["user_id"] = observations["user_id"].int()
        return observations

if __name__ == "__main__":
    args = parse_args()
    llm = load_LLM(args.llm_model)

    dir_name = f"{args.llm_model}___{args.llm_rater}___{args.items_retrieval}___{args.user_dataset}___{args.news_dataset}___{args.perturbator}___{args.reward_shaping}___{args.seed}___{args.model_device}___{args.gamma}___{args.embedding_dim}___{args.learning_rate}"
    sanitized_dir_name = dir_name.replace('/', '_').replace(':', '_').replace('.', '_')
    save_path = f"./tmp/models/{sanitized_dir_name}"
    wandb_path = f"./tmp/wandb"
    os.makedirs(save_path, exist_ok=True)

    train_env = get_enviroment_from_args(llm, args)
    test_env = get_enviroment_from_args(llm, args, seed=args.seed + 600)

    policy_kwargs = dict(features_extractor_class=ExtractPass)
    train_env = StableBaselineWrapperNum(train_env)
    test_env = Monitor(StableBaselineWrapperNum(test_env))

    check_env(train_env)
    check_env(test_env)

    model = A2C(
        CustomActorCriticPolicy,
        train_env,
        verbose=1,
        policy_kwargs=policy_kwargs,
        device=args.model_device,
        learning_rate=linear_schedule(args.learning_rate),
        tensorboard_log=save_path,
        gamma=args.gamma,
        ent_coef=0.001,
    )

    combined_callback = CombinedCallback(save_freq=2500, log_freq=500, save_path=save_path, name_prefix="rl_model", verbose=1)
    callback = CallbackList([combined_callback])

    logger.info("Model starts learning")

    model.learn(total_timesteps=10000, progress_bar=True, callback=callback, tb_log_name="t_logs")

    logger.info("Model Ends Learning")
    
    logger.info("Evaluating the Policy")
    mean_reward, std_reward = evaluate_policy(model, test_env, n_eval_episodes=50)
    logger.info(f"Mean reward: {mean_reward} +/- {std_reward}")

    reward_file_path = os.path.join(save_path, f"reward_{mean_reward:.2f}.txt")
    with open(reward_file_path, 'w') as file:
        file.write(f"Mean reward: {mean_reward} +/- {std_reward}\n")

    print(f"Reward information saved to {reward_file_path}")

    print(f"Mean Reward: {mean_reward} +/- {std_reward}")
    logger.info(f"Mean Reward: {mean_reward} +/- {std_reward}")
