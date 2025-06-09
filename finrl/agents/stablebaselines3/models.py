"""
FinRL Stable Baselines3 深層強化学習エージェント実装モジュール

このモジュールは、金融データでの強化学習モデルの訓練・予測・アンサンブル戦略を
実装するクラスを提供します。

主な機能:
- 単一モデルの訓練・予測 (DRLAgent)
- 複数モデルのアンサンブル戦略 (DRLEnsembleAgent)  
- 動的なモデル選択とリバランシング
- 市場ボラティリティに基づく適応的制御
- Tensorboard統合による学習監視
"""

# DRL models from Stable Baselines 3
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from stable_baselines3 import A2C
from stable_baselines3 import DDPG
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv

from finrl import config
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.preprocessors import data_split

# サポートする強化学習モデル辞書
MODELS = {"a2c": A2C, "ddpg": DDPG, "td3": TD3, "sac": SAC, "ppo": PPO}

# 各モデルのデフォルトパラメータ（config.pyから取得）
MODEL_KWARGS = {x: config.__dict__[f"{x.upper()}_PARAMS"] for x in MODELS.keys()}

# 探索用ノイズタイプ辞書（連続行動空間用）
NOISE = {
    "normal": NormalActionNoise,  # 正規ノイズ
    "ornstein_uhlenbeck": OrnsteinUhlenbeckActionNoise,  # Ornstein-Uhlenbeckプロセス
}


class TensorboardCallback(BaseCallback):
    """
    Tensorboard用カスタムコールバック
    
    学習中の追加情報をTensorboardにログ出力するためのコールバッククラス。
    報酬の統計情報（最小値、平均値、最大値）を記録します。
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        """
        各ステップ後に呼び出されるメソッド
        
        現在の報酬をTensorboardに記録します。
        """
        try:
            # 報酬情報の記録（複数の形式に対応）
            self.logger.record(key="train/reward", value=self.locals["rewards"][0])

        except BaseException as error:
            try:
                # 別の形式での報酬取得を試行
                self.logger.record(key="train/reward", value=self.locals["reward"][0])

            except BaseException as inner_error:
                # どちらの形式でも取得できない場合はNoneを記録
                self.logger.record(key="train/reward", value=None)
                # デバッグ用エラー情報出力
                print("Original Error:", error)
                print("Inner Error:", inner_error)
        return True

    def _on_rollout_end(self) -> bool:
        """
        ロールアウト終了時に呼び出されるメソッド
        
        報酬の統計情報（最小値、平均値、最大値）をTensorboardに記録します。
        """
        try:
            # ロールアウトバッファから報酬を取得
            rollout_buffer_rewards = self.locals["rollout_buffer"].rewards.flatten()
            self.logger.record(
                key="train/reward_min", value=min(rollout_buffer_rewards)
            )
            self.logger.record(
                key="train/reward_mean", value=statistics.mean(rollout_buffer_rewards)
            )
            self.logger.record(
                key="train/reward_max", value=max(rollout_buffer_rewards)
            )
        except BaseException as error:
            # エラー時はNoneを記録
            self.logger.record(key="train/reward_min", value=None)
            self.logger.record(key="train/reward_mean", value=None)
            self.logger.record(key="train/reward_max", value=None)
            print("Logging Error:", error)
        return True


class DRLAgent:
    """
    深層強化学習エージェントクラス
    
    単一の強化学習モデルの訓練、予測、評価を行うためのクラス。
    複数のStable Baselines3アルゴリズムをサポートし、金融取引環境に特化した
    機能を提供します。

    Attributes:
        env: gym環境クラス - ユーザー定義の取引環境

    メソッド:
        get_model(): DRLアルゴリズムの設定
        train_model(): 訓練データでDRLアルゴリズムを訓練し、訓練済みモデルを出力
        DRL_prediction(): テストデータで予測を実行し結果を取得
    """

    def __init__(self, env):
        """
        DRLエージェントの初期化
        
        Args:
            env: 取引環境オブジェクト
        """
        self.env = env

    def get_model(
        self,
        model_name,
        policy="MlpPolicy",
        policy_kwargs=None,
        model_kwargs=None,
        verbose=1,
        seed=None,
        tensorboard_log=None,
    ):
        """
        指定されたアルゴリズムの強化学習モデルを作成
        
        Args:
            model_name: アルゴリズム名 ('a2c', 'ddpg', 'td3', 'sac', 'ppo')
            policy: 方策タイプ (デフォルト: 'MlpPolicy')
            policy_kwargs: 方策の追加パラメータ
            model_kwargs: モデルの追加パラメータ
            verbose: 詳細度レベル
            seed: ランダムシード
            tensorboard_log: Tensorboardログディレクトリ
            
        Returns:
            初期化された強化学習モデル
            
        Raises:
            ValueError: サポートされていないモデル名が指定された場合
        """
        if model_name not in MODELS:
            raise ValueError(
                f"Model '{model_name}' not found in MODELS."
            )  # NotImplementedErrorより情報豊富

        # デフォルトパラメータの設定
        if model_kwargs is None:
            model_kwargs = MODEL_KWARGS[model_name]

        # 行動ノイズの設定（連続行動空間のアルゴリズム用）
        if "action_noise" in model_kwargs:
            n_actions = self.env.action_space.shape[-1]
            model_kwargs["action_noise"] = NOISE[model_kwargs["action_noise"]](
                mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
            )
        print(model_kwargs)
        
        # モデルの作成と返却
        return MODELS[model_name](
            policy=policy,
            env=self.env,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            policy_kwargs=policy_kwargs,
            seed=seed,
            **model_kwargs,
        )

    @staticmethod
    def train_model(
        model,
        tb_log_name,
        total_timesteps=5000,
        callbacks: Type[BaseCallback] = None,
    ):  
        """
        強化学習モデルの訓練
        
        staticメソッドとして実装されているため、クラスのインスタンス化なしで呼び出し可能。
        
        Args:
            model: 訓練対象のモデル
            tb_log_name: Tensorboardログ名
            total_timesteps: 訓練ステップ数 (デフォルト: 5000)
            callbacks: 追加のコールバック関数リスト
            
        Returns:
            訓練済みモデル
        """
        # Tensorboardコールバックと追加コールバックを統合
        model = model.learn(
            total_timesteps=total_timesteps,
            tb_log_name=tb_log_name,
            callback=(
                CallbackList(
                    [TensorboardCallback()] + [callback for callback in callbacks]
                )
                if callbacks is not None
                else TensorboardCallback()
            ),
        )
        return model

    @staticmethod
    def DRL_prediction(model, environment, deterministic=True):
        """
        訓練済みモデルを使用した予測・取引実行
        
        Args:
            model: 訓練済み強化学習モデル
            environment: 取引環境
            deterministic: 決定論的行動の選択 (デフォルト: True)
            
        Returns:
            tuple: (口座記録, 行動記録)
        """
        # 環境の準備
        test_env, test_obs = environment.get_sb_env()
        # メモリ最適化のための初期化
        account_memory = None  # 不要なリスト作成を回避
        actions_memory = None  # メモリ消費を最適化
        # state_memory=[] # 状態を保存するメモリプール

        test_env.reset()
        max_steps = len(environment.df.index.unique()) - 1

        # 取引期間全体での予測実行
        for i in range(len(environment.df.index.unique())):
            # モデルによる行動予測
            action, _states = model.predict(test_obs, deterministic=deterministic)
            # account_memory = test_env.env_method(method_name="save_asset_memory")
            # actions_memory = test_env.env_method(method_name="save_action_memory")
            # 環境での行動実行
            test_obs, rewards, dones, info = test_env.step(action)

            # 最終ステップで記録を保存（メモリ効率化）
            if (
                i == max_steps - 1
            ):  # 早期終了条件をより分かりやすく記述
                account_memory = test_env.env_method(method_name="save_asset_memory")
                actions_memory = test_env.env_method(method_name="save_action_memory")
            # 現在の状態を状態メモリに追加
            # state_memory=test_env.env_method(method_name="save_state_memory")

            # エピソード終了判定
            if dones[0]:
                print("hit end!")
                break
        return account_memory[0], actions_memory[0]

    @staticmethod
    def DRL_prediction_load_from_file(model_name, environment, cwd, deterministic=True):
        """
        ファイルから読み込んだモデルを使用した予測
        
        Args:
            model_name: モデル名
            environment: 取引環境
            cwd: モデルファイルパス
            deterministic: 決定論的行動の選択
            
        Returns:
            list: エピソード総資産リスト
            
        Raises:
            ValueError: モデル読み込みに失敗した場合
        """
        if model_name not in MODELS:
            raise ValueError(
                f"Model '{model_name}' not found in MODELS."
            )  # NotImplementedErrorより情報豊富
        try:
            # モデルの読み込み
            model = MODELS[model_name].load(cwd)
            print("Successfully load model", cwd)
        except BaseException as error:
            raise ValueError(f"Failed to load agent. Error: {str(error)}") from error

        # テスト環境での評価
        state = environment.reset()
        episode_returns = []  # 累積リターン / 初期口座
        episode_total_assets = [environment.initial_total_asset]
        done = False
        
        # エピソード実行
        while not done:
            action = model.predict(state, deterministic=deterministic)[0]
            state, reward, done, _ = environment.step(action)

            # 総資産の計算
            total_asset = (
                environment.amount
                + (environment.price_ary[environment.day] * environment.stocks).sum()
            )
            episode_total_assets.append(total_asset)
            episode_return = total_asset / environment.initial_total_asset
            episode_returns.append(episode_return)

        print("episode_return", episode_return)
        print("Test Finished!")
        return episode_total_assets


class DRLEnsembleAgent:
    """
    深層強化学習アンサンブルエージェントクラス
    
    複数の強化学習アルゴリズム（A2C, PPO, DDPG, SAC, TD3）を組み合わせ、
    動的にモデル選択を行うアンサンブル戦略を実装します。
    
    各リバランシング期間で複数モデルを訓練・評価し、最も性能の良いモデルを
    選択して取引を実行します。市場のボラティリティに応じた適応的制御も行います。
    """
    
    @staticmethod
    def get_model(
        model_name,
        env,
        policy="MlpPolicy",
        policy_kwargs=None,
        model_kwargs=None,
        seed=None,
        verbose=1,
    ):
        """
        アンサンブル用モデルの作成
        
        Args:
            model_name: アルゴリズム名
            env: 取引環境
            policy: 方策タイプ
            policy_kwargs: 方策パラメータ
            model_kwargs: モデルパラメータ
            seed: ランダムシード
            verbose: 詳細度
            
        Returns:
            初期化されたモデル
        """
        if model_name not in MODELS:
            raise ValueError(
                f"Model '{model_name}' not found in MODELS."
            )  # NotImplementedErrorより情報豊富

        # パラメータの準備
        if model_kwargs is None:
            temp_model_kwargs = MODEL_KWARGS[model_name]
        else:
            temp_model_kwargs = model_kwargs.copy()

        # 行動ノイズの設定
        if "action_noise" in temp_model_kwargs:
            n_actions = env.action_space.shape[-1]
            temp_model_kwargs["action_noise"] = NOISE[
                temp_model_kwargs["action_noise"]
            ](mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        print(temp_model_kwargs)
        
        # モデル作成（Tensorboardログ設定含む）
        return MODELS[model_name](
            policy=policy,
            env=env,
            tensorboard_log=f"{config.TENSORBOARD_LOG_DIR}/{model_name}",
            verbose=verbose,
            policy_kwargs=policy_kwargs,
            seed=seed,
            **temp_model_kwargs,
        )

    @staticmethod
    def train_model(
        model,
        model_name,
        tb_log_name,
        iter_num,
        total_timesteps=5000,
        callbacks: Type[BaseCallback] = None,
    ):
        """
        アンサンブル用モデルの訓練と保存
        
        Args:
            model: 訓練対象モデル
            model_name: モデル名
            tb_log_name: Tensorboardログ名
            iter_num: イテレーション番号
            total_timesteps: 訓練ステップ数
            callbacks: 追加コールバック
            
        Returns:
            訓練済みモデル
        """
        # モデル訓練
        model = model.learn(
            total_timesteps=total_timesteps,
            tb_log_name=tb_log_name,
            callback=(
                CallbackList(
                    [TensorboardCallback()] + [callback for callback in callbacks]
                )
                if callbacks is not None
                else TensorboardCallback()
            ),
        )
        # 訓練済みモデルの保存
        model.save(
            f"{config.TRAINED_MODEL_DIR}/{model_name.upper()}_{total_timesteps // 1000}k_{iter_num}"
        )
        return model

    @staticmethod
    def get_validation_sharpe(iteration, model_name):
        """
        検証結果に基づくSharpe ratioの計算
        
        Args:
            iteration: イテレーション番号
            model_name: モデル名
            
        Returns:
            float: 計算されたSharpe ratio
        """
        # 検証結果の読み込み
        df_total_value = pd.read_csv(
            f"results/account_value_validation_{model_name}_{iteration}.csv"
        )
        # エージェントが取引を行わなかった場合
        if df_total_value["daily_return"].var() == 0:
            if df_total_value["daily_return"].mean() > 0:
                return np.inf
            else:
                return 0.0
        else:
            # 四半期換算Sharpe ratio
            return (
                (4**0.5)
                * df_total_value["daily_return"].mean()
                / df_total_value["daily_return"].std()
            )

    def __init__(
        self,
        df,
        train_period,
        val_test_period,
        rebalance_window,
        validation_window,
        stock_dim,
        hmax,
        initial_amount,
        buy_cost_pct,
        sell_cost_pct,
        reward_scaling,
        state_space,
        action_space,
        tech_indicator_list,
        print_verbosity,
    ):
        """
        アンサンブルエージェントの初期化
        
        Args:
            df: 金融データのDataFrame
            train_period: 訓練期間 [開始日, 終了日]
            val_test_period: 検証・テスト期間 [開始日, 終了日]
            rebalance_window: リバランシング期間（日数）
            validation_window: 検証期間（日数）
            stock_dim: 銘柄数
            hmax: 最大保有株数
            initial_amount: 初期資金
            buy_cost_pct: 買い手数料率
            sell_cost_pct: 売り手数料率
            reward_scaling: 報酬スケーリング
            state_space: 状態空間サイズ
            action_space: 行動空間サイズ
            tech_indicator_list: テクニカル指標リスト
            print_verbosity: 出力詳細度
        """
        self.df = df
        self.train_period = train_period
        self.val_test_period = val_test_period

        # 取引期間の日付リスト
        self.unique_trade_date = df[
            (df.date > val_test_period[0]) & (df.date <= val_test_period[1])
        ].date.unique()
        self.rebalance_window = rebalance_window
        self.validation_window = validation_window

        # 環境パラメータ
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        self.print_verbosity = print_verbosity
        self.train_env = None  # train_validation()関数で定義

    def DRL_validation(self, model, test_data, test_env, test_obs):
        """
        モデルの検証プロセス
        
        Args:
            model: 検証対象モデル
            test_data: テストデータ
            test_env: テスト環境
            test_obs: テスト観測
        """
        # 検証期間全体での実行
        for _ in range(len(test_data.index.unique())):
            action, _states = model.predict(test_obs)
            test_obs, rewards, dones, info = test_env.step(action)

    def DRL_prediction(
        self, model, name, last_state, iter_num, turbulence_threshold, initial
    ):
        """
        訓練済みモデルを使用した予測・取引
        
        Args:
            model: 予測に使用するモデル
            name: モデル名
            last_state: 前回の最終状態
            iter_num: イテレーション番号
            turbulence_threshold: ボラティリティ閾値
            initial: 初期状態フラグ
            
        Returns:
            最終状態
        """
        # 取引データの準備
        trade_data = data_split(
            self.df,
            start=self.unique_trade_date[iter_num - self.rebalance_window],
            end=self.unique_trade_date[iter_num],
        )
        
        # 取引環境の構築
        trade_env = DummyVecEnv(
            [
                lambda: StockTradingEnv(
                    df=trade_data,
                    stock_dim=self.stock_dim,
                    hmax=self.hmax,
                    initial_amount=self.initial_amount,
                    num_stock_shares=[0] * self.stock_dim,
                    buy_cost_pct=[self.buy_cost_pct] * self.stock_dim,
                    sell_cost_pct=[self.sell_cost_pct] * self.stock_dim,
                    reward_scaling=self.reward_scaling,
                    state_space=self.state_space,
                    action_space=self.action_space,
                    tech_indicator_list=self.tech_indicator_list,
                    turbulence_threshold=turbulence_threshold,
                    initial=initial,
                    previous_state=last_state,
                    model_name=name,
                    mode="trade",
                    iteration=iter_num,
                    print_verbosity=self.print_verbosity,
                )
            ]
        )

        trade_obs = trade_env.reset()

        # 取引期間での予測・実行
        for i in range(len(trade_data.index.unique())):
            action, _states = model.predict(trade_obs)
            trade_obs, rewards, dones, info = trade_env.step(action)
            if i == (len(trade_data.index.unique()) - 2):
                # 最終状態の取得
                last_state = trade_env.envs[0].render()

        # 最終状態をCSVで保存
        df_last_state = pd.DataFrame({"last_state": last_state})
        df_last_state.to_csv(f"results/last_state_{name}_{i}.csv", index=False)
        return last_state

    def _train_window(
        self,
        model_name,
        model_kwargs,
        sharpe_list,
        validation_start_date,
        validation_end_date,
        timesteps_dict,
        i,
        validation,
        turbulence_threshold,
    ):
        """
        単一ウィンドウでのモデル訓練
        
        Args:
            model_name: モデル名
            model_kwargs: モデルパラメータ
            sharpe_list: Sharpe ratioリスト
            validation_start_date: 検証開始日
            validation_end_date: 検証終了日  
            timesteps_dict: 各モデルの訓練ステップ数辞書
            i: イテレーション番号
            validation: 検証データ
            turbulence_threshold: ボラティリティ閾値
            
        Returns:
            tuple: (訓練済みモデル, 更新されたSharpe ratioリスト, Sharpe ratio値)
        """
        # モデルパラメータが未設定の場合はスキップ
        if model_kwargs is None:
            return None, sharpe_list, -1

        print(f"======{model_name} Training========")
        # モデル作成
        model = self.get_model(
            model_name, self.train_env, policy="MlpPolicy", model_kwargs=model_kwargs
        )
        # モデル訓練
        model = self.train_model(
            model,
            model_name,
            tb_log_name=f"{model_name}_{i}",
            iter_num=i,
            total_timesteps=timesteps_dict[model_name],
        )  # 100_000
        print(
            f"======{model_name} Validation from: ",
            validation_start_date,
            "to ",
            validation_end_date,
        )
        
        # 検証環境の構築
        val_env = DummyVecEnv(
            [
                lambda: StockTradingEnv(
                    df=validation,
                    stock_dim=self.stock_dim,
                    hmax=self.hmax,
                    initial_amount=self.initial_amount,
                    num_stock_shares=[0] * self.stock_dim,
                    buy_cost_pct=[self.buy_cost_pct] * self.stock_dim,
                    sell_cost_pct=[self.sell_cost_pct] * self.stock_dim,
                    reward_scaling=self.reward_scaling,
                    state_space=self.state_space,
                    action_space=self.action_space,
                    tech_indicator_list=self.tech_indicator_list,
                    turbulence_threshold=turbulence_threshold,
                    iteration=i,
                    model_name=model_name,
                    mode="validation",
                    print_verbosity=self.print_verbosity,
                )
            ]
        )
        val_obs = val_env.reset()
        
        # 検証実行
        self.DRL_validation(
            model=model,
            test_data=validation,
            test_env=val_env,
            test_obs=val_obs,
        )
        
        # Sharpe ratio計算
        sharpe = self.get_validation_sharpe(i, model_name=model_name)
        print(f"{model_name} Sharpe Ratio: ", sharpe)
        sharpe_list.append(sharpe)
        return model, sharpe_list, sharpe

    def run_ensemble_strategy(
        self,
        A2C_model_kwargs,
        PPO_model_kwargs,
        DDPG_model_kwargs,
        SAC_model_kwargs,
        TD3_model_kwargs,
        timesteps_dict,
    ):
        """
        アンサンブル戦略の実行
        
        複数のアルゴリズム（A2C, PPO, DDPG, SAC, TD3）を組み合わせ、
        各期間で最もパフォーマンスの良いモデルを動的に選択します。
        
        Args:
            A2C_model_kwargs: A2Cモデルパラメータ
            PPO_model_kwargs: PPOモデルパラメータ  
            DDPG_model_kwargs: DDPGモデルパラメータ
            SAC_model_kwargs: SACモデルパラメータ
            TD3_model_kwargs: TD3モデルパラメータ
            timesteps_dict: 各モデルの訓練ステップ数辞書
            
        Returns:
            pd.DataFrame: アンサンブル結果の要約DataFrame
        """
        # モデルパラメータ辞書
        kwargs = {
            "a2c": A2C_model_kwargs,
            "ppo": PPO_model_kwargs,
            "ddpg": DDPG_model_kwargs,
            "sac": SAC_model_kwargs,
            "td3": TD3_model_kwargs,
        }
        
        # 各モデルのSharpe ratio管理辞書
        model_dct = {k: {"sharpe_list": [], "sharpe": -1} for k in MODELS.keys()}

        print("============Start Ensemble Strategy============")
        # アンサンブルモデルでは前のモデルの最終状態を
        # 現在のモデルの初期状態として渡すことが必要
        last_state_ensemble = []

        # 結果記録用リスト
        model_use = []
        validation_start_date_list = []
        validation_end_date_list = []
        iteration_list = []

        # インサンプルボラティリティ閾値の計算
        insample_turbulence = self.df[
            (self.df.date < self.train_period[1])
            & (self.df.date >= self.train_period[0])
        ]
        # 90%分位点をボラティリティ閾値として設定
        insample_turbulence_threshold = np.quantile(
            insample_turbulence.turbulence.values, 0.90
        )

        start = time.time()
        
        # リバランシング期間ごとのループ
        for i in range(
            self.rebalance_window + self.validation_window,
            len(self.unique_trade_date),
            self.rebalance_window,
        ):
            # 検証期間の設定
            validation_start_date = self.unique_trade_date[
                i - self.rebalance_window - self.validation_window
            ]
            validation_end_date = self.unique_trade_date[i - self.rebalance_window]

            validation_start_date_list.append(validation_start_date)
            validation_end_date_list.append(validation_end_date)
            iteration_list.append(i)

            print("============================================")
            # 初期状態判定
            if i - self.rebalance_window - self.validation_window == 0:
                # 初期状態
                initial = True
            else:
                # 前回状態を継承
                initial = False

            # 過去データに基づくボラティリティ調整
            # ボラティリティ期間は四半期（63日）
            end_date_index = self.df.index[
                self.df["date"]
                == self.unique_trade_date[
                    i - self.rebalance_window - self.validation_window
                ]
            ].to_list()[-1]
            start_date_index = end_date_index - 63 + 1

            # 過去ボラティリティデータの取得
            historical_turbulence = self.df.iloc[
                start_date_index : (end_date_index + 1), :
            ]

            historical_turbulence = historical_turbulence.drop_duplicates(
                subset=["date"]
            )

            historical_turbulence_mean = np.mean(
                historical_turbulence.turbulence.values
            )

            # ボラティリティ閾値の動的調整
            if historical_turbulence_mean > insample_turbulence_threshold:
                # 過去データの平均がインサンプルの90%分位点より高い場合
                # 現在の市場がボラタイルと仮定し、
                # インサンプルの90%分位点を閾値として設定
                turbulence_threshold = insample_turbulence_threshold
            else:
                # 過去データの平均がインサンプルの90%分位点より低い場合
                # ボラティリティ閾値を上げてリスクを下げる
                turbulence_threshold = np.quantile(
                    insample_turbulence.turbulence.values, 1
                )

            # 99%分位点に設定（保守的なアプローチ）
            turbulence_threshold = np.quantile(
                insample_turbulence.turbulence.values, 0.99
            )
            print("turbulence_threshold: ", turbulence_threshold)

            # 環境設定開始
            # 訓練環境
            train = data_split(
                self.df,
                start=self.train_period[0],
                end=self.unique_trade_date[
                    i - self.rebalance_window - self.validation_window
                ],
            )
            self.train_env = DummyVecEnv(
                [
                    lambda: StockTradingEnv(
                        df=train,
                        stock_dim=self.stock_dim,
                        hmax=self.hmax,
                        initial_amount=self.initial_amount,
                        num_stock_shares=[0] * self.stock_dim,
                        buy_cost_pct=[self.buy_cost_pct] * self.stock_dim,
                        sell_cost_pct=[self.sell_cost_pct] * self.stock_dim,
                        reward_scaling=self.reward_scaling,
                        state_space=self.state_space,
                        action_space=self.action_space,
                        tech_indicator_list=self.tech_indicator_list,
                        print_verbosity=self.print_verbosity,
                    )
                ]
            )

            # 検証データ
            validation = data_split(
                self.df,
                start=self.unique_trade_date[
                    i - self.rebalance_window - self.validation_window
                ],
                end=self.unique_trade_date[i - self.rebalance_window],
            )
            # 環境設定終了

            # 訓練・検証開始
            print(
                "======Model training from: ",
                self.train_period[0],
                "to ",
                self.unique_trade_date[
                    i - self.rebalance_window - self.validation_window
                ],
            )
            
            # 各モデルの訓練
            for model_name in MODELS.keys():
                # モデル訓練実行
                model, sharpe_list, sharpe = self._train_window(
                    model_name,
                    kwargs[model_name],
                    model_dct[model_name]["sharpe_list"],
                    validation_start_date,
                    validation_end_date,
                    timesteps_dict,
                    i,
                    validation,
                    turbulence_threshold,
                )
                # モデルのSharpe ratio記録、モデル自体の保存
                model_dct[model_name]["sharpe_list"] = sharpe_list
                model_dct[model_name]["model"] = model
                model_dct[model_name]["sharpe"] = sharpe

            print(
                "======Best Model Retraining from: ",
                self.train_period[0],
                "to ",
                self.unique_trade_date[i - self.rebalance_window],
            )
            
            # 最初の取引日までのモデル再訓練用環境設定
            # train_full = data_split(self.df, start=self.train_period[0],
            # end=self.unique_trade_date[i - self.rebalance_window])
            # self.train_full_env = DummyVecEnv([lambda: StockTradingEnv(train_full,
            #                                               self.stock_dim,
            #                                               self.hmax,
            #                                               self.initial_amount,
            #                                               self.buy_cost_pct,
            #                                               self.sell_cost_pct,
            #                                               self.reward_scaling,
            #                                               self.state_space,
            #                                               self.action_space,
            #                                               self.tech_indicator_list,
            #                                              print_verbosity=self.print_verbosity
            # )])
            
            # Sharpe ratioに基づくモデル選択
            # MODELS順序: {"a2c": A2C, "ddpg": DDPG, "td3": TD3, "sac": SAC, "ppo": PPO}
            sharpes = [model_dct[k]["sharpe"] for k in MODELS.keys()]
            # 最高Sharpe ratioのモデルを選択
            max_mod = list(MODELS.keys())[np.argmax(sharpes)]
            model_use.append(max_mod.upper())
            model_ensemble = model_dct[max_mod]["model"]
            # 訓練・検証終了

            # 取引開始
            print(
                "======Trading from: ",
                self.unique_trade_date[i - self.rebalance_window],
                "to ",
                self.unique_trade_date[i],
            )
            # print("Used Model: ", model_ensemble)
            last_state_ensemble = self.DRL_prediction(
                model=model_ensemble,
                name="ensemble",
                last_state=last_state_ensemble,
                iter_num=i,
                turbulence_threshold=turbulence_threshold,
                initial=initial,
            )
            # 取引終了

        end = time.time()
        print("Ensemble Strategy took: ", (end - start) / 60, " minutes")

        # 結果要約DataFrame作成
        df_summary = pd.DataFrame(
            [
                iteration_list,
                validation_start_date_list,
                validation_end_date_list,
                model_use,
                model_dct["a2c"]["sharpe_list"],
                model_dct["ppo"]["sharpe_list"],
                model_dct["ddpg"]["sharpe_list"],
                model_dct["sac"]["sharpe_list"],
                model_dct["td3"]["sharpe_list"],
            ]
        ).T
        df_summary.columns = [
            "Iter",
            "Val Start",
            "Val End",
            "Model Used",
            "A2C Sharpe",
            "PPO Sharpe",
            "DDPG Sharpe",
            "SAC Sharpe",
            "TD3 Sharpe",
        ]

        return df_summary
