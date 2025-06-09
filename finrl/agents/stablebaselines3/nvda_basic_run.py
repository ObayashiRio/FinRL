"""
NVDA基本動作確認スクリプト

NVIDIAの株価データを使って、FinRL + Stable Baselines3の
基本的な強化学習取引を実行します。

使用モデル: PPO (デフォルト設定)
"""

import warnings
warnings.filterwarnings('ignore')
import statistics

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# FinRL imports
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl import config
from finrl.main import check_and_make_directories

def get_nvda_data(start_date="2020-01-01", end_date="2023-12-31"):
    """
    NVDAの株価データを取得・前処理
    
    Args:
        start_date: 開始日
        end_date: 終了日
        
    Returns:
        pd.DataFrame: 前処理済みデータ
    """
    print("📈 NVDAデータ取得中...")
    
    # yfinanceでNVDAデータ取得
    nvda = yf.download('NVDA', start=start_date, end=end_date, auto_adjust=True)
    
    # 列名がMultiIndexの場合、フラット化
    if isinstance(nvda.columns, pd.MultiIndex):
        nvda.columns = nvda.columns.get_level_values(0)
    
    # DataFrameの整形
    nvda.reset_index(inplace=True)
    
    # 列名を明示的に設定（yfinanceの出力形式に合わせて）
    expected_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    for col in expected_columns:
        if col not in nvda.columns:
            print(f"警告: 列 '{col}' が見つかりません")
            print(f"実際の列名: {list(nvda.columns)}")
    
    nvda['tic'] = 'NVDA'
    nvda['day'] = nvda['Date'].dt.dayofweek
    nvda['date'] = nvda['Date'].dt.strftime('%Y-%m-%d')
    
    # 必要な列名に変更
    nvda.rename(columns={
        'Open': 'open',
        'High': 'high', 
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    }, inplace=True)
    
    # 必要な列のみ選択
    nvda = nvda[['date', 'open', 'high', 'low', 'close', 'volume', 'tic', 'day']]
    
    print(f"✅ データ取得完了: {len(nvda)}行")
    print(f"期間: {nvda['date'].min()} ~ {nvda['date'].max()}")
    
    return nvda

def add_technical_indicators(df):
    """
    テクニカル指標を追加
    
    Args:
        df: 株価データ
        
    Returns:
        pd.DataFrame: テクニカル指標付きデータ
    """
    print("🔧 テクニカル指標追加中...")
    
    try:
        # FeatureEngineerを使用してテクニカル指標追加
        fe = FeatureEngineer(
            use_technical_indicator=True,
            tech_indicator_list=[
                'macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30',
                'close_30_sma', 'close_60_sma'
            ],
            use_vix=False,
            use_turbulence=True,
            user_defined_feature=False
        )
        
        processed = fe.preprocess_data(df)
        
        # リスト内の各DataFrameを結合
        if isinstance(processed, list):
            processed = pd.concat(processed, ignore_index=True)
            
    except Exception as e:
        print(f"⚠️ FeatureEngineer エラー: {e}")
        print("🔧 シンプルなテクニカル指標を手動追加します...")
        
        # 手動でシンプルなテクニカル指標を追加
        processed = df.copy()
        
        # 移動平均
        processed['close_30_sma'] = processed['close'].rolling(window=30).mean()
        processed['close_60_sma'] = processed['close'].rolling(window=60).mean()
        
        # RSI（簡易版）
        delta = processed['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        processed['rsi_30'] = 100 - (100 / (1 + rs))
        
        # ボリンジャーバンド
        processed['boll_ub'] = processed['close_30_sma'] + 2 * processed['close'].rolling(window=30).std()
        processed['boll_lb'] = processed['close_30_sma'] - 2 * processed['close'].rolling(window=30).std()
        
        # MACD（簡易版）
        exp1 = processed['close'].ewm(span=12).mean()
        exp2 = processed['close'].ewm(span=26).mean()
        processed['macd'] = exp1 - exp2
        
        # その他の指標は0で初期化（環境動作確認のため）
        processed['cci_30'] = 0
        processed['dx_30'] = 0
        processed['turbulence'] = 0
        
        # NaN値を前方向補完
        processed = processed.fillna(method='ffill').fillna(0)
    
    print(f"✅ テクニカル指標追加完了: {processed.shape[1]}列")
    
    return processed

def create_trading_env(df, mode='train', initial_amount=1000000):
    """
    取引環境を作成
    
    Args:
        df: 処理済みデータ
        mode: 'train' or 'trade'
        initial_amount: 初期資金
        
    Returns:
        StockTradingEnv: 取引環境
    """
    stock_dimension = len(df.tic.unique())
    
    # データに実際に存在するテクニカル指標を特定
    basic_columns = ['date', 'tic', 'close', 'open', 'high', 'low', 'volume', 'day']
    tech_columns = [f for f in df.columns if f not in basic_columns]
    
    # データに存在する指標のみを使用
    available_tech_indicators = []
    desired_indicators = ['macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30', 'close_30_sma', 'close_60_sma']
    
    for indicator in desired_indicators:
        if indicator in tech_columns:
            available_tech_indicators.append(indicator)
    
    # state_spaceを正確に計算
    # StockTradingEnvの状態空間: 1 + 2*stock_dim + len(tech_indicators)*stock_dim
    state_space = 1 + (2 * stock_dimension) + (len(available_tech_indicators) * stock_dimension)
    
    print(f"🏢 {mode}環境作成中...")
    print(f"銘柄数: {stock_dimension}")
    print(f"利用可能テクニカル指標: {available_tech_indicators} (計{len(available_tech_indicators)}個)")
    print(f"計算した状態空間サイズ: {state_space}")
    print(f"データの全列: {list(df.columns)}")
    
    # 環境を作成
    env = StockTradingEnv(
        df=df,
        stock_dim=stock_dimension,
        hmax=100,  # 最大保有株数
        initial_amount=initial_amount,
        num_stock_shares=[0] * stock_dimension,
        buy_cost_pct=[0.001] * stock_dimension,  # 0.1%の取引手数料
        sell_cost_pct=[0.001] * stock_dimension,
        reward_scaling=1e-4,
        state_space=state_space,
        action_space=stock_dimension,
        tech_indicator_list=available_tech_indicators,
        print_verbosity=10
    )
    
    # 環境作成後に実際の状態空間サイズを表示
    try:
        # 環境をリセットして実際の観測値サイズを確認
        test_obs = env.reset()
        actual_state_space = len(test_obs)
        print(f"✅ 実際の状態空間サイズ: {actual_state_space}")
    except Exception as e:
        print(f"⚠️ 状態空間サイズ確認エラー: {e}")
    
    return env

def main():
    """
    メイン実行関数
    """
    print("🚀 NVDA強化学習取引 - 基本動作確認開始")
    print("=" * 50)
    
    # 必要なディレクトリ作成
    check_and_make_directories([
        config.DATA_SAVE_DIR,
        config.TRAINED_MODEL_DIR,
        config.TENSORBOARD_LOG_DIR,
        config.RESULTS_DIR,
    ])
    
    # 1. データ取得
    raw_data = get_nvda_data()
    
    # 2. テクニカル指標追加
    processed_data = add_technical_indicators(raw_data)
    
    # 3. 訓練・テスト期間に分割
    train_end_date = '2022-12-31'
    test_start_date = '2023-01-01'
    
    train_data = data_split(processed_data, start='2020-01-01', end=train_end_date)
    test_data = data_split(processed_data, start=test_start_date, end='2023-12-31')
    
    print(f"📊 訓練データ: {len(train_data)}行")
    print(f"📊 テストデータ: {len(test_data)}行")
    
    # 4. 環境作成
    print("=" * 30)
    train_env = create_trading_env(train_data, mode='train')
    print("=" * 30)
    test_env = create_trading_env(test_data, mode='trade')
    print("=" * 30)
    
    # 5. エージェント作成・訓練
    print("🤖 PPOエージェント訓練開始...")
    
    agent = DRLAgent(env=train_env)
    
    # デフォルト設定を使用
    model = agent.get_model(model_name='ppo')
    
    # 訓練実行（軽めの設定）
    trained_model = agent.train_model(
        model=model,
        tb_log_name='nvda_ppo_basic',
        total_timesteps=50000  # とりあえず5万ステップ
    )
    
    # 6. モデル保存
    model_path = f"{config.TRAINED_MODEL_DIR}/nvda_ppo_basic.zip"
    trained_model.save(model_path)
    print(f"💾 モデル保存: {model_path}")
    
    # 7. テスト取引実行
    print("📈 テスト取引実行中...")
    print(f"テスト環境データ長: {len(test_data)}")
    print(f"テスト期間: {test_data['date'].min()} ~ {test_data['date'].max()}")
    
    df_account_value, df_actions = DRLAgent.DRL_prediction(
        model=trained_model,
        environment=test_env
    )
    
    # デバッグ情報を表示
    print(f"🔍 デバッグ情報:")
    print(f"df_account_value形状: {df_account_value.shape}")
    print(f"df_actions形状: {df_actions.shape}")
    print(f"account_value列: {df_account_value.columns.tolist()}")
    print(f"最初の5行:")
    print(df_account_value.head())
    print(f"最後の5行:")
    print(df_account_value.tail())
    
    # 8. 結果表示
    if len(df_account_value) > 0 and 'account_value' in df_account_value.columns:
        initial_value = df_account_value['account_value'].iloc[0]
        final_value = df_account_value['account_value'].iloc[-1]
        total_return = (final_value - initial_value) / initial_value * 100
    else:
        print("⚠️ account_valueデータが空です")
        initial_value = 1000000.0
        final_value = 1000000.0
        total_return = 0.0
    
    print("=" * 50)
    print("📊 取引結果")
    print(f"初期資金: ${initial_value:,.2f}")
    print(f"最終資金: ${final_value:,.2f}")
    print(f"総収益率: {total_return:.2f}%")
    
    # NVDA本体のパフォーマンスと比較
    nvda_start_price = test_data['close'].iloc[0]
    nvda_end_price = test_data['close'].iloc[-1]
    nvda_return = (nvda_end_price - nvda_start_price) / nvda_start_price * 100
    
    print(f"NVDA買い持ち: {nvda_return:.2f}%")
    print(f"AI vs NVDA: {total_return - nvda_return:.2f}%の差")
    
    # 9. 結果保存
    df_account_value.to_csv(f"{config.RESULTS_DIR}/nvda_account_value.csv", index=False)
    df_actions.to_csv(f"{config.RESULTS_DIR}/nvda_actions.csv", index=False)
    
    print("✅ 実行完了！結果は results/ フォルダに保存されました")
    
    return df_account_value, df_actions

if __name__ == "__main__":
    main() 