"""
FinRL Stable Baselines3 エージェント用ハイパーパラメータ最適化モジュール

このモジュールは、Optunaを使用して各強化学習アルゴリズムのハイパーパラメータを
最適化するためのサンプリング関数を提供します。

主な機能:
- PPO、TRPO、A2C、SAC、TD3、DDPG、DQN等の各アルゴリズム用サンプリング関数
- 学習率、バッチサイズ、ネットワーク構造、ノイズパラメータ等の最適化範囲設定
- HER (Hindsight Experience Replay) やTQC (Truncated Quantile Critics) 等の拡張手法対応
"""

from __future__ import annotations

from typing import Any
from typing import Dict

import numpy as np
import optuna
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.utils import linear_schedule
from torch import nn as nn


def sample_ppo_params(trial: optuna.Trial) -> dict[str, Any]:
    """
    PPO (Proximal Policy Optimization) アルゴリズム用ハイパーパラメータサンプラー
    
    PPOは方策勾配法の一種で、方策の更新幅を制限することで安定した学習を実現します。
    
    :param trial: Optunaのトライアルオブジェクト
    :return: PPO用ハイパーパラメータ辞書
    """
    # バッチサイズ: 一度に処理するサンプル数
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512])
    # ステップ数: 環境から収集するステップ数
    n_steps = trial.suggest_categorical(
        "n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    )
    # 割引率: 将来の報酬をどの程度重視するか
    gamma = trial.suggest_categorical(
        "gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999]
    )
    # 学習率: パラメータ更新の大きさ
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1)
    lr_schedule = "constant"
    # 学習率スケジュール有効化時のコメントアウト
    # lr_schedule = trial.suggest_categorical('lr_schedule', ['linear', 'constant'])
    # エントロピー係数: 探索を促進するためのボーナス
    ent_coef = trial.suggest_loguniform("ent_coef", 0.00000001, 0.1)
    # クリッピング範囲: 方策更新の制限範囲
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
    # エポック数: 同じデータでの学習回数
    n_epochs = trial.suggest_categorical("n_epochs", [1, 5, 10, 20])
    # GAE λ: Advantage推定のパラメータ
    gae_lambda = trial.suggest_categorical(
        "gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0]
    )
    # 勾配クリッピング: 勾配爆発を防ぐ
    max_grad_norm = trial.suggest_categorical(
        "max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5]
    )
    # 価値関数係数: 価値関数損失の重み
    vf_coef = trial.suggest_uniform("vf_coef", 0, 1)
    # ネットワーク構造: 小さいか中程度
    net_arch = trial.suggest_categorical("net_arch", ["small", "medium"])
    # gSDE (State Dependent Exploration) 用パラメータ（連続行動空間）
    # log_std_init = trial.suggest_uniform("log_std_init", -4, 1)
    # sde_sample_freq = trial.suggest_categorical("sde_sample_freq", [-1, 8, 16, 32, 64, 128, 256])
    # 直交初期化: パラメータの初期化方法
    ortho_init = False
    # ortho_init = trial.suggest_categorical('ortho_init', [False, True])
    # 活性化関数: ニューラルネットワークの非線形変換
    # activation_fn = trial.suggest_categorical('activation_fn', ['tanh', 'relu', 'elu', 'leaky_relu'])
    activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])

    # バッチサイズがステップ数を超えないよう調整
    if batch_size > n_steps:
        batch_size = n_steps

    # 線形学習率スケジュールの適用
    if lr_schedule == "linear":
        learning_rate = linear_schedule(learning_rate)

    # ネットワーク構造の定義（方策ネットワークと価値ネットワークを独立に設定）
    # 画像以外のデータでは独立ネットワークが通常最適
    net_arch = {
        "small": [dict(pi=[64, 64], vf=[64, 64])],
        "medium": [dict(pi=[256, 256], vf=[256, 256])],
    }[net_arch]

    # 活性化関数の選択
    activation_fn = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU,
    }[activation_fn]

    return {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "clip_range": clip_range,
        "n_epochs": n_epochs,
        "gae_lambda": gae_lambda,
        "max_grad_norm": max_grad_norm,
        "vf_coef": vf_coef,
        # "sde_sample_freq": sde_sample_freq,
        "policy_kwargs": dict(
            # log_std_init=log_std_init,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
        ),
    }


def sample_trpo_params(trial: optuna.Trial) -> dict[str, Any]:
    """
    TRPO (Trust Region Policy Optimization) アルゴリズム用ハイパーパラメータサンプラー
    
    TRPOは方策の更新を信頼領域内に制限することで安定した学習を実現します。
    
    :param trial: Optunaのトライアルオブジェクト
    :return: TRPO用ハイパーパラメータ辞書
    """
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512])
    n_steps = trial.suggest_categorical(
        "n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    )
    gamma = trial.suggest_categorical(
        "gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999]
    )
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1)
    lr_schedule = "constant"
    # 学習率スケジュール有効化時のコメントアウト
    # lr_schedule = trial.suggest_categorical('lr_schedule', ['linear', 'constant'])
    # 直線探索収縮因子
    # line_search_shrinking_factor = trial.suggest_categorical("line_search_shrinking_factor", [0.6, 0.7, 0.8, 0.9])
    # 批評家ネットワークの更新回数
    n_critic_updates = trial.suggest_categorical(
        "n_critic_updates", [5, 10, 20, 25, 30]
    )
    # 共役勾配法の最大ステップ数
    cg_max_steps = trial.suggest_categorical("cg_max_steps", [5, 10, 20, 25, 30])
    # 共役勾配法のダンピングパラメータ
    # cg_damping = trial.suggest_categorical("cg_damping", [0.5, 0.2, 0.1, 0.05, 0.01])
    # 目標KLダイバージェンス: 信頼領域のサイズ
    target_kl = trial.suggest_categorical(
        "target_kl", [0.1, 0.05, 0.03, 0.02, 0.01, 0.005, 0.001]
    )
    gae_lambda = trial.suggest_categorical(
        "gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0]
    )
    net_arch = trial.suggest_categorical("net_arch", ["small", "medium"])
    # gSDE用パラメータ
    # log_std_init = trial.suggest_uniform("log_std_init", -4, 1)
    # sde_sample_freq = trial.suggest_categorical("sde_sample_freq", [-1, 8, 16, 32, 64, 128, 256])
    ortho_init = False
    # ortho_init = trial.suggest_categorical('ortho_init', [False, True])
    # activation_fn = trial.suggest_categorical('activation_fn', ['tanh', 'relu', 'elu', 'leaky_relu'])
    activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])

    # バッチサイズ調整
    if batch_size > n_steps:
        batch_size = n_steps

    if lr_schedule == "linear":
        learning_rate = linear_schedule(learning_rate)

    # ネットワーク構造設定
    net_arch = {
        "small": [dict(pi=[64, 64], vf=[64, 64])],
        "medium": [dict(pi=[256, 256], vf=[256, 256])],
    }[net_arch]

    activation_fn = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU,
    }[activation_fn]

    return {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "gamma": gamma,
        # "cg_damping": cg_damping,
        "cg_max_steps": cg_max_steps,
        # "line_search_shrinking_factor": line_search_shrinking_factor,
        "n_critic_updates": n_critic_updates,
        "target_kl": target_kl,
        "learning_rate": learning_rate,
        "gae_lambda": gae_lambda,
        # "sde_sample_freq": sde_sample_freq,
        "policy_kwargs": dict(
            # log_std_init=log_std_init,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
        ),
    }


def sample_a2c_params(trial: optuna.Trial) -> dict[str, Any]:
    """
    A2C (Advantage Actor-Critic) アルゴリズム用ハイパーパラメータサンプラー
    
    A2Cは俳優批評家法の一種で、方策と価値関数を同時に学習します。
    
    :param trial: Optunaのトライアルオブジェクト
    :return: A2C用ハイパーパラメータ辞書
    """
    gamma = trial.suggest_categorical(
        "gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999]
    )
    # Advantage正規化: 安定性向上のため
    normalize_advantage = trial.suggest_categorical(
        "normalize_advantage", [False, True]
    )
    max_grad_norm = trial.suggest_categorical(
        "max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5]
    )
    # PyTorch版RMSProp使用切り替え（TensorFlow版とは異なる）
    use_rms_prop = trial.suggest_categorical("use_rms_prop", [False, True])
    gae_lambda = trial.suggest_categorical(
        "gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0]
    )
    n_steps = trial.suggest_categorical(
        "n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    )
    lr_schedule = trial.suggest_categorical("lr_schedule", ["linear", "constant"])
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1)
    ent_coef = trial.suggest_loguniform("ent_coef", 0.00000001, 0.1)
    vf_coef = trial.suggest_uniform("vf_coef", 0, 1)
    # gSDE用パラメータ
    # log_std_init = trial.suggest_uniform("log_std_init", -4, 1)
    ortho_init = trial.suggest_categorical("ortho_init", [False, True])
    net_arch = trial.suggest_categorical("net_arch", ["small", "medium"])
    # SDE用ネットワーク構造
    # sde_net_arch = trial.suggest_categorical("sde_net_arch", [None, "tiny", "small"])
    # full_std = trial.suggest_categorical("full_std", [False, True])
    # activation_fn = trial.suggest_categorical('activation_fn', ['tanh', 'relu', 'elu', 'leaky_relu'])
    activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])

    if lr_schedule == "linear":
        learning_rate = linear_schedule(learning_rate)

    net_arch = {
        "small": [dict(pi=[64, 64], vf=[64, 64])],
        "medium": [dict(pi=[256, 256], vf=[256, 256])],
    }[net_arch]

    # SDE用ネットワーク構造（コメントアウト）
    # sde_net_arch = {
    #     None: None,
    #     "tiny": [64],
    #     "small": [64, 64],
    # }[sde_net_arch]

    activation_fn = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU,
    }[activation_fn]

    return {
        "n_steps": n_steps,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "normalize_advantage": normalize_advantage,
        "max_grad_norm": max_grad_norm,
        "use_rms_prop": use_rms_prop,
        "vf_coef": vf_coef,
        "policy_kwargs": dict(
            # log_std_init=log_std_init,
            net_arch=net_arch,
            # full_std=full_std,
            activation_fn=activation_fn,
            # sde_net_arch=sde_net_arch,
            ortho_init=ortho_init,
        ),
    }


def sample_sac_params(trial: optuna.Trial) -> dict[str, Any]:
    """
    SAC (Soft Actor-Critic) アルゴリズム用ハイパーパラメータサンプラー
    
    SACは最大エントロピー強化学習を用いたoff-policyアルゴリズムで、
    探索と活用のバランスを自動調整します。
    
    :param trial: Optunaのトライアルオブジェクト
    :return: SAC用ハイパーパラメータ辞書
    """
    gamma = trial.suggest_categorical(
        "gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999]
    )
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1)
    batch_size = trial.suggest_categorical(
        "batch_size", [16, 32, 64, 128, 256, 512, 1024, 2048]
    )
    # リプレイバッファサイズ: 過去の経験を保存する容量
    buffer_size = trial.suggest_categorical(
        "buffer_size", [int(1e4), int(1e5), int(1e6)]
    )
    # 学習開始タイミング: 最初に何ステップ経験を蓄積するか
    learning_starts = trial.suggest_categorical(
        "learning_starts", [0, 1000, 10000, 20000]
    )
    # 訓練頻度: 何ステップごとに学習するか
    # train_freq = trial.suggest_categorical('train_freq', [1, 10, 100, 300])
    train_freq = trial.suggest_categorical(
        "train_freq", [1, 4, 8, 16, 32, 64, 128, 256, 512]
    )
    # Polyak係数: ターゲットネットワークの更新率
    tau = trial.suggest_categorical("tau", [0.001, 0.005, 0.01, 0.02, 0.05, 0.08])
    # 勾配ステップ数: 1回の学習で何回勾配更新するか
    # gradient_steps = trial.suggest_categorical('gradient_steps', [1, 100, 300])
    gradient_steps = train_freq
    # エントロピー係数: 自動調整を使用
    # ent_coef = trial.suggest_categorical('ent_coef', ['auto', 0.5, 0.1, 0.05, 0.01, 0.0001])
    ent_coef = "auto"
    # gSDE使用時のlog標準偏差初期値
    log_std_init = trial.suggest_uniform("log_std_init", -4, 1)
    # ネットワーク構造（HER調整時は"verybig"も追加）
    net_arch = trial.suggest_categorical("net_arch", ["small", "medium", "big"])
    # activation_fn = trial.suggest_categorical('activation_fn', [nn.Tanh, nn.ReLU, nn.ELU, nn.LeakyReLU])

    # ネットワーク構造の定義
    net_arch = {
        "small": [64, 64],
        "medium": [256, 256],
        "big": [400, 300],
        # HER調整用（コメントアウト）
        # "large": [256, 256, 256],
        # "verybig": [512, 512, 512],
    }[net_arch]

    # 目標エントロピー: 自動調整
    target_entropy = "auto"
    # if ent_coef == 'auto':
    #     # target_entropy = trial.suggest_categorical('target_entropy', ['auto', 5, 1, 0, -1, -5, -10, -20, -50])
    #     target_entropy = trial.suggest_uniform('target_entropy', -10, 10)

    hyperparams = {
        "gamma": gamma,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "learning_starts": learning_starts,
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
        "ent_coef": ent_coef,
        "tau": tau,
        "target_entropy": target_entropy,
        "policy_kwargs": dict(log_std_init=log_std_init, net_arch=net_arch),
    }

    # HER (Hindsight Experience Replay) 使用時のパラメータ追加
    if trial.using_her_replay_buffer:
        hyperparams = sample_her_params(trial, hyperparams)

    return hyperparams


def sample_td3_params(trial: optuna.Trial) -> dict[str, Any]:
    """
    TD3 (Twin Delayed Deep Deterministic Policy Gradient) アルゴリズム用ハイパーパラメータサンプラー
    
    TD3はDDPGの改良版で、価値関数の過推定を軽減し、より安定した学習を実現します。
    
    :param trial: Optunaのトライアルオブジェクト
    :return: TD3用ハイパーパラメータ辞書
    """
    gamma = trial.suggest_categorical(
        "gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999]
    )
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1)
    batch_size = trial.suggest_categorical(
        "batch_size", [16, 32, 64, 100, 128, 256, 512, 1024, 2048]
    )
    buffer_size = trial.suggest_categorical(
        "buffer_size", [int(1e4), int(1e5), int(1e6)]
    )
    # Polyak係数
    tau = trial.suggest_categorical("tau", [0.001, 0.005, 0.01, 0.02, 0.05, 0.08])

    train_freq = trial.suggest_categorical(
        "train_freq", [1, 4, 8, 16, 32, 64, 128, 256, 512]
    )
    gradient_steps = train_freq

    # ノイズタイプ: 探索用ノイズの種類
    noise_type = trial.suggest_categorical(
        "noise_type", ["ornstein-uhlenbeck", "normal", None]
    )
    # ノイズの標準偏差
    noise_std = trial.suggest_uniform("noise_std", 0, 1)

    # ネットワーク構造（HER調整時は"verybig"も追加）
    net_arch = trial.suggest_categorical("net_arch", ["small", "medium", "big"])
    # activation_fn = trial.suggest_categorical('activation_fn', [nn.Tanh, nn.ReLU, nn.ELU, nn.LeakyReLU])

    net_arch = {
        "small": [64, 64],
        "medium": [256, 256],
        "big": [400, 300],
        # HER調整用（コメントアウト）
        # "verybig": [256, 256, 256],
    }[net_arch]

    hyperparams = {
        "gamma": gamma,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
        "policy_kwargs": dict(net_arch=net_arch),
        "tau": tau,
    }

    # 探索ノイズの設定
    if noise_type == "normal":
        hyperparams["action_noise"] = NormalActionNoise(
            mean=np.zeros(trial.n_actions), sigma=noise_std * np.ones(trial.n_actions)
        )
    elif noise_type == "ornstein-uhlenbeck":
        hyperparams["action_noise"] = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(trial.n_actions), sigma=noise_std * np.ones(trial.n_actions)
        )

    if trial.using_her_replay_buffer:
        hyperparams = sample_her_params(trial, hyperparams)

    return hyperparams


def sample_ddpg_params(trial: optuna.Trial) -> dict[str, Any]:
    """
    DDPG (Deep Deterministic Policy Gradient) アルゴリズム用ハイパーパラメータサンプラー
    
    DDPGは連続行動空間向けのoff-policyアルゴリズムで、
    決定論的方策勾配法を用いています。
    
    :param trial: Optunaのトライアルオブジェクト
    :return: DDPG用ハイパーパラメータ辞書
    """
    gamma = trial.suggest_categorical(
        "gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999]
    )
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1)
    batch_size = trial.suggest_categorical(
        "batch_size", [16, 32, 64, 100, 128, 256, 512, 1024, 2048]
    )
    buffer_size = trial.suggest_categorical(
        "buffer_size", [int(1e4), int(1e5), int(1e6)]
    )
    # Polyak係数
    tau = trial.suggest_categorical("tau", [0.001, 0.005, 0.01, 0.02, 0.05, 0.08])

    train_freq = trial.suggest_categorical(
        "train_freq", [1, 4, 8, 16, 32, 64, 128, 256, 512]
    )
    gradient_steps = train_freq

    noise_type = trial.suggest_categorical(
        "noise_type", ["ornstein-uhlenbeck", "normal", None]
    )
    noise_std = trial.suggest_uniform("noise_std", 0, 1)

    # ネットワーク構造（HERのTD3参照）
    net_arch = trial.suggest_categorical("net_arch", ["small", "medium", "big"])
    # activation_fn = trial.suggest_categorical('activation_fn', [nn.Tanh, nn.ReLU, nn.ELU, nn.LeakyReLU])

    net_arch = {"small": [64, 64], "medium": [256, 256], "big": [400, 300]}[net_arch]

    hyperparams = {
        "gamma": gamma,
        "tau": tau,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
        "policy_kwargs": dict(net_arch=net_arch),
    }

    # 探索ノイズの設定
    if noise_type == "normal":
        hyperparams["action_noise"] = NormalActionNoise(
            mean=np.zeros(trial.n_actions), sigma=noise_std * np.ones(trial.n_actions)
        )
    elif noise_type == "ornstein-uhlenbeck":
        hyperparams["action_noise"] = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(trial.n_actions), sigma=noise_std * np.ones(trial.n_actions)
        )

    if trial.using_her_replay_buffer:
        hyperparams = sample_her_params(trial, hyperparams)

    return hyperparams


def sample_dqn_params(trial: optuna.Trial) -> dict[str, Any]:
    """
    DQN (Deep Q-Network) アルゴリズム用ハイパーパラメータサンプラー
    
    DQNは価値ベースの強化学習アルゴリズムで、離散行動空間で使用されます。
    
    :param trial: Optunaのトライアルオブジェクト
    :return: DQN用ハイパーパラメータ辞書
    """
    gamma = trial.suggest_categorical(
        "gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999]
    )
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1)
    batch_size = trial.suggest_categorical(
        "batch_size", [16, 32, 64, 100, 128, 256, 512]
    )
    buffer_size = trial.suggest_categorical(
        "buffer_size", [int(1e4), int(5e4), int(1e5), int(1e6)]
    )
    # ε-greedy探索の最終値
    exploration_final_eps = trial.suggest_uniform("exploration_final_eps", 0, 0.2)
    # 探索期間の割合
    exploration_fraction = trial.suggest_uniform("exploration_fraction", 0, 0.5)
    # ターゲットネットワーク更新間隔
    target_update_interval = trial.suggest_categorical(
        "target_update_interval", [1, 1000, 5000, 10000, 15000, 20000]
    )
    learning_starts = trial.suggest_categorical(
        "learning_starts", [0, 1000, 5000, 10000, 20000]
    )

    train_freq = trial.suggest_categorical("train_freq", [1, 4, 8, 16, 128, 256, 1000])
    # サブサンプリングステップ数
    subsample_steps = trial.suggest_categorical("subsample_steps", [1, 2, 4, 8])
    gradient_steps = max(train_freq // subsample_steps, 1)

    net_arch = trial.suggest_categorical("net_arch", ["tiny", "small", "medium"])

    net_arch = {"tiny": [64], "small": [64, 64], "medium": [256, 256]}[net_arch]

    hyperparams = {
        "gamma": gamma,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
        "exploration_fraction": exploration_fraction,
        "exploration_final_eps": exploration_final_eps,
        "target_update_interval": target_update_interval,
        "learning_starts": learning_starts,
        "policy_kwargs": dict(net_arch=net_arch),
    }

    if trial.using_her_replay_buffer:
        hyperparams = sample_her_params(trial, hyperparams)

    return hyperparams


def sample_her_params(
    trial: optuna.Trial, hyperparams: dict[str, Any]
) -> dict[str, Any]:
    """
    HER (Hindsight Experience Replay) 用ハイパーパラメータサンプラー
    
    HERは目標条件付き強化学習において、失敗した経験も学習に活用する手法です。
    
    :param trial: Optunaのトライアルオブジェクト
    :param hyperparams: 既存のハイパーパラメータ辞書
    :return: HER用パラメータを追加したハイパーパラメータ辞書
    """
    her_kwargs = trial.her_kwargs.copy()
    # サンプリングする目標の数
    her_kwargs["n_sampled_goal"] = trial.suggest_int("n_sampled_goal", 1, 5)
    # 目標選択戦略
    her_kwargs["goal_selection_strategy"] = trial.suggest_categorical(
        "goal_selection_strategy", ["final", "episode", "future"]
    )
    # オンラインサンプリングの使用
    her_kwargs["online_sampling"] = trial.suggest_categorical(
        "online_sampling", [True, False]
    )
    hyperparams["replay_buffer_kwargs"] = her_kwargs
    return hyperparams


def sample_tqc_params(trial: optuna.Trial) -> dict[str, Any]:
    """
    TQC (Truncated Quantile Critics) アルゴリズム用ハイパーパラメータサンプラー
    
    TQCはSAC + 分布強化学習の組み合わせで、価値関数の分布を学習します。
    
    :param trial: Optunaのトライアルオブジェクト
    :return: TQC用ハイパーパラメータ辞書
    """
    # TQCはSAC + 分布強化学習
    hyperparams = sample_sac_params(trial)

    # 分位点の数
    n_quantiles = trial.suggest_int("n_quantiles", 5, 50)
    # ネットワークごとに削除する上位分位点の数
    top_quantiles_to_drop_per_net = trial.suggest_int(
        "top_quantiles_to_drop_per_net", 0, n_quantiles - 1
    )

    hyperparams["policy_kwargs"].update({"n_quantiles": n_quantiles})
    hyperparams["top_quantiles_to_drop_per_net"] = top_quantiles_to_drop_per_net

    return hyperparams


def sample_qrdqn_params(trial: optuna.Trial) -> dict[str, Any]:
    """
    QR-DQN (Quantile Regression DQN) アルゴリズム用ハイパーパラメータサンプラー
    
    QR-DQNはDQN + 分布強化学習の組み合わせです。
    
    :param trial: Optunaのトライアルオブジェクト
    :return: QR-DQN用ハイパーパラメータ辞書
    """
    # QR-DQNはDQN + 分布強化学習
    hyperparams = sample_dqn_params(trial)

    n_quantiles = trial.suggest_int("n_quantiles", 5, 200)
    hyperparams["policy_kwargs"].update({"n_quantiles": n_quantiles})

    return hyperparams


def sample_ars_params(trial: optuna.Trial) -> dict[str, Any]:
    """
    ARS (Augmented Random Search) アルゴリズム用ハイパーパラメータサンプラー
    
    ARSは進化戦略に基づくシンプルで効果的なアルゴリズムです。
    
    :param trial: Optunaのトライアルオブジェクト
    :return: ARS用ハイパーパラメータ辞書
    """
    # 評価エピソード数
    # n_eval_episodes = trial.suggest_categorical("n_eval_episodes", [1, 2])
    # 摂動の数（方向数）
    n_delta = trial.suggest_categorical("n_delta", [4, 8, 6, 32, 64])
    # 学習率
    # learning_rate = trial.suggest_categorical("learning_rate", [0.01, 0.02, 0.025, 0.03])
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1)
    # 摂動の標準偏差
    delta_std = trial.suggest_categorical(
        "delta_std", [0.01, 0.02, 0.025, 0.03, 0.05, 0.1, 0.2, 0.3]
    )
    # 上位何割の摂動を使用するか
    top_frac_size = trial.suggest_categorical(
        "top_frac_size", [0.1, 0.2, 0.3, 0.5, 0.8, 0.9, 1.0]
    )
    # ゼロ方策から開始するか
    zero_policy = trial.suggest_categorical("zero_policy", [True, False])
    # 上位N個の摂動を使用
    n_top = max(int(top_frac_size * n_delta), 1)

    # ネットワーク構造（線形方策のみの場合はコメントアウト）
    # net_arch = trial.suggest_categorical("net_arch", ["linear", "tiny", "small"])

    # 線形方策として使用する場合はバイアスを除去し、出力をsquashしない
    # 線形方策のみでハイパーパラメータ探索する場合はコメントアウト
    # net_arch = {
    #     "linear": [],
    #     "tiny": [16],
    #     "small": [32],
    # }[net_arch]

    # TODO: alive_bonus_offsetも最適化対象に含める

    return {
        # "n_eval_episodes": n_eval_episodes,
        "n_delta": n_delta,
        "learning_rate": learning_rate,
        "delta_std": delta_std,
        "n_top": n_top,
        "zero_policy": zero_policy,
        # "policy_kwargs": dict(net_arch=net_arch),
    }


# 各アルゴリズムのハイパーパラメータサンプラー辞書
# 新しいアルゴリズムを追加する場合は、ここに対応するサンプラー関数を追加
HYPERPARAMS_SAMPLER = {
    "a2c": sample_a2c_params,
    "ars": sample_ars_params,
    "ddpg": sample_ddpg_params,
    "dqn": sample_dqn_params,
    "qrdqn": sample_qrdqn_params,
    "sac": sample_sac_params,
    "tqc": sample_tqc_params,
    "ppo": sample_ppo_params,
    "td3": sample_td3_params,
    "trpo": sample_trpo_params,
}
