"""
FinRL Stable Baselines3 エージェント用Optuna ハイパーパラメータチューニングモジュール

このモジュールは、Optunaを使用して金融データでの強化学習モデルの
ハイパーパラメータを自動最適化するクラスを提供します。

主な機能:
- TPE (Tree Parzen Estimator) とHyperbandプルーナーを使った効率的な最適化
- Sharpe ratioを目的関数とした金融指標ベースの最適化
- 早期停止機能による計算時間の短縮
- 最適化後のモデルを使った自動バックテスト
"""

from __future__ import annotations

import datetime

import joblib
import optuna
import pandas as pd
from stable_baselines3 import A2C
from stable_baselines3 import DDPG
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3 import TD3

import finrl.agents.stablebaselines3.hyperparams_opt as hpt
from finrl import config
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.main import check_and_make_directories
from finrl.plot import backtest_stats


class LoggingCallback:
    """
    Optuna最適化の早期停止用コールバッククラス
    
    Sharpe ratioの改善が一定期間停滞した場合に最適化を停止し、
    無駄な計算時間を削減します。
    
    Attributes:
        threshold: int - Sharpe ratio改善の許容閾値
        trial_number: int - 早期停止判定を開始する最小試行回数
        patience: int - 閾値を下回る状態を許容する連続回数
        cb_list: list - 閾値に達した試行番号のリスト
    """

    def __init__(self, threshold: int, trial_number: int, patience: int):
        """
        早期停止コールバックの初期化
        
        Args:
            threshold: int - Sharpe ratio改善の許容閾値
            trial_number: int - 最小試行回数後に早期停止判定開始
            patience: int - 閾値を下回る連続回数の許容値
        """
        self.threshold = threshold
        self.trial_number = trial_number
        self.patience = patience
        self.cb_list = []  # 閾値に達した試行のリスト

    def __call__(self, study: optuna.study, frozen_trial: optuna.Trial):
        """
        各試行後に呼び出される早期停止判定メソッド
        
        Args:
            study: Optuna study オブジェクト
            frozen_trial: 完了した試行オブジェクト
        """
        # 現在の試行での最良値を設定
        study.set_user_attr("previous_best_value", study.best_value)

        # 最小試行回数を超えているかチェック
        if frozen_trial.number > self.trial_number:
            previous_best_value = study.user_attrs.get("previous_best_value", None)
            # 前回と今回の目的値が同じ符号かチェック（共に正または共に負）
            if previous_best_value * study.best_value >= 0:
                # 改善幅が閾値以下かチェック
                if abs(previous_best_value - study.best_value) < self.threshold:
                    self.cb_list.append(frozen_trial.number)
                    # 連続してpatience回数閾値を下回った場合に停止
                    if len(self.cb_list) > self.patience:
                        print("The study stops now...")
                        print(
                            "With number",
                            frozen_trial.number,
                            "and value ",
                            frozen_trial.value,
                        )
                        print(
                            "The previous and current best values are {} and {} respectively".format(
                                previous_best_value, study.best_value
                            )
                        )
                        study.stop()


class TuneSB3Optuna:
    """
    Stable Baselines3 エージェントのOptunaハイパーパラメータチューニングクラス
    
    このクラスは金融データでの強化学習モデルのハイパーパラメータを
    Sharpe ratioを目的関数として自動最適化します。

    Attributes:
        env_train: 訓練用環境
        model_name: str - 使用する強化学習アルゴリズム名
        env_trade: テスト用取引環境  
        logging_callback: 早期停止用コールバック
        total_timesteps: int - 各試行での学習ステップ数
        n_trials: int - 最適化試行回数
        
    注意:
        デフォルトのサンプラーとプルーナーは
        Tree Parzen Estimator と Hyperband Scheduler です。
    """

    def __init__(
        self,
        env_train,
        model_name: str,
        env_trade,
        logging_callback,
        total_timesteps: int = 50000,
        n_trials: int = 30,
    ):
        """
        ハイパーパラメータチューニングクラスの初期化
        
        Args:
            env_train: 訓練用環境
            model_name: 強化学習アルゴリズム名 ('a2c', 'ddpg', 'td3', 'sac', 'ppo')
            env_trade: テスト用取引環境
            logging_callback: 早期停止用コールバック
            total_timesteps: 各試行での学習ステップ数 (デフォルト: 50000)
            n_trials: 最適化試行回数 (デフォルト: 30)
        """
        self.env_train = env_train
        self.agent = DRLAgent(env=env_train)
        self.model_name = model_name
        self.env_trade = env_trade
        self.total_timesteps = total_timesteps
        self.n_trials = n_trials
        self.logging_callback = logging_callback
        # サポートするStable Baselines3モデル辞書
        self.MODELS = {"a2c": A2C, "ddpg": DDPG, "td3": TD3, "sac": SAC, "ppo": PPO}

        # 必要なディレクトリの作成
        check_and_make_directories(
            [
                config.DATA_SAVE_DIR,
                config.TRAINED_MODEL_DIR,
                config.TENSORBOARD_LOG_DIR,
                config.RESULTS_DIR,
            ]
        )

    def default_sample_hyperparameters(self, trial: optuna.Trial):
        """
        指定されたモデルのデフォルトハイパーパラメータをサンプリング
        
        Args:
            trial: Optunaトライアルオブジェクト
            
        Returns:
            dict: サンプリングされたハイパーパラメータ辞書
        """
        if self.model_name == "a2c":
            return hpt.sample_a2c_params(trial)
        elif self.model_name == "ddpg":
            return hpt.sample_ddpg_params(trial)
        elif self.model_name == "td3":
            return hpt.sample_td3_params(trial)
        elif self.model_name == "sac":
            return hpt.sample_sac_params(trial)
        elif self.model_name == "ppo":
            return hpt.sample_ppo_params(trial)

    def calculate_sharpe(self, df: pd.DataFrame):
        """
        ポートフォリオのSharpe ratioを計算
        
        Sharpe ratioは投資収益率のリスク調整後パフォーマンス指標で、
        年率換算した平均リターンを標準偏差で割った値です。
        
        Args:
            df: 口座価値を含むDataFrame
            
        Returns:
            float: 計算されたSharpe ratio
        """
        # 日次リターン率を計算
        df["daily_return"] = df["account_value"].pct_change(1)
        if df["daily_return"].std() != 0:
            # 年率換算Sharpe ratio (252営業日換算)
            sharpe = (252**0.5) * df["daily_return"].mean() / df["daily_return"].std()
            return sharpe
        else:
            # 標準偏差が0の場合（リターンが一定）
            return 0

    def objective(self, trial: optuna.Trial):
        """
        Optuna最適化の目的関数
        
        各試行で新しいハイパーパラメータでモデルを訓練し、
        テスト環境でのSharpe ratioを評価します。
        
        Args:
            trial: Optunaトライアルオブジェクト
            
        Returns:
            float: 最大化対象のSharpe ratio
        """
        # ハイパーパラメータのサンプリング
        hyperparameters = self.default_sample_hyperparameters(trial)
        policy_kwargs = hyperparameters["policy_kwargs"]
        del hyperparameters["policy_kwargs"]
        
        # モデルの作成
        model = self.agent.get_model(
            self.model_name, policy_kwargs=policy_kwargs, model_kwargs=hyperparameters
        )
        
        # モデルの訓練
        trained_model = self.agent.train_model(
            model=model,
            tb_log_name=self.model_name,
            total_timesteps=self.total_timesteps,
        )
        
        # 訓練済みモデルの保存
        trained_model.save(
            f"./{config.TRAINED_MODEL_DIR}/{self.model_name}_{trial.number}.pth"
        )
        
        # テスト環境での予測実行
        df_account_value, _ = DRLAgent.DRL_prediction(
            model=trained_model, environment=self.env_trade
        )
        
        # Sharpe ratioの計算（最大化目標）
        sharpe = self.calculate_sharpe(df_account_value)

        return sharpe

    def run_optuna(self):
        """
        Optunaハイパーパラメータ最適化の実行
        
        TPE (Tree Parzen Estimator) サンプラーとHyperbandプルーナーを使用し、
        効率的にハイパーパラメータ空間を探索します。
        
        Returns:
            optuna.Study: 完了した最適化スタディオブジェクト
        """
        # TPEサンプラーの設定（再現性のためのシード設定）
        sampler = optuna.samplers.TPESampler(seed=42)
        
        # Studyの作成（最大化方向、Hyperbandプルーナー使用）
        study = optuna.create_study(
            study_name=f"{self.model_name}_study",
            direction="maximize",  # Sharpe ratioの最大化
            sampler=sampler,
            pruner=optuna.pruners.HyperbandPruner(),  # 性能の悪い試行を早期終了
        )

        # 最適化の実行
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            catch=(ValueError,),  # ValueError例外をキャッチして継続
            callbacks=[self.logging_callback],  # 早期停止コールバック
        )

        # 最適化結果の保存
        joblib.dump(study, f"{self.model_name}_study.pkl")
        return study

    def backtest(
        self, final_study: optuna.Study
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        最適化されたモデルでのバックテスト実行
        
        最良の試行で得られたハイパーパラメータで訓練されたモデルを使用し、
        テスト環境での詳細なパフォーマンス分析を実行します。
        
        Args:
            final_study: 完了したOptuna study
            
        Returns:
            tuple: (口座価値DataFrame, 行動履歴DataFrame, パフォーマンス統計DataFrame)
        """
        print("Hyperparameters after tuning", final_study.best_params)
        print("Best Trial", final_study.best_trial)

        # 最良試行のモデルを読み込み
        tuned_model = self.MODELS[self.model_name].load(
            f"./{config.TRAINED_MODEL_DIR}/{self.model_name}_{final_study.best_trial.number}.pth",
            env=self.env_train,
        )

        # 最適化モデルでの予測実行
        df_account_value_tuned, df_actions_tuned = DRLAgent.DRL_prediction(
            model=tuned_model, environment=self.env_trade
        )

        print("==============Get Backtest Results===========")
        # 現在時刻でファイル名生成
        now = datetime.datetime.now().strftime("%Y%m%d-%Hh%M")

        # バックテスト統計の計算
        perf_stats_all_tuned = backtest_stats(account_value=df_account_value_tuned)
        perf_stats_all_tuned = pd.DataFrame(perf_stats_all_tuned)
        
        # 結果をCSVで保存
        perf_stats_all_tuned.to_csv(
            "./" + config.RESULTS_DIR + "/perf_stats_all_tuned_" + now + ".csv"
        )

        return df_account_value_tuned, df_actions_tuned, perf_stats_all_tuned
