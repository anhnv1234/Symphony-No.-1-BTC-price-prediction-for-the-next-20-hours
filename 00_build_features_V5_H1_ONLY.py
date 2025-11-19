import pandas as pd
import pandas_ta as ta
import os
import logging
import sys 
import numpy as np # (Cần cho FVG)

# --- CẤU HÌNH V5 (H1-ONLY "Cách Ly") ---
# (Input: Vẫn là 53 món gốc)
MASTER_FILE_PATH = os.path.join('02_Master_Data', 'btcusdt_master_data.parquet') 

# (SỬA V5: Output "Thức Ăn" H1 "Cách Ly")
OUTPUT_FILE_PATH = os.path.join('02_Master_Data', 'btcusdt_master_data_features_V5_H1_ONLY.parquet') 
LOG_FILE = "log_00_build_features_V5.log" # (Log file V5)

# --- CẤU HÌNH LOGGING (UTF-8) ---
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass
    
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - [BuildFeatures_V5_H1] - %(message)s',
                    handlers=[
                        logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'), 
                        logging.StreamHandler(sys.stdout)
                    ])

# === (MÓN 1: "ĐỘ" FVG) ===
def calculate_fvg(df, high_col, low_col):
    """
    "Chế" FVG (Fair Value Gaps / Imbalance)
    (Giữ nguyên logic V3)
    """
    logging.info("Đang 'nấu' Món Mới: FVG (Smart Money)...")
    
    # 1. FVG "Bò" (Bullish): Low[i-2] > High[i]
    fvg_bull = df[low_col].shift(2) - df[high_col]
    
    # 2. FVG "Gấu" (Bearish): High[i-2] < Low[i]
    fvg_bear = df[high_col].shift(2) - df[low_col]

    # "Dọn rác"
    df['H1_fvg_bull_gap'] = fvg_bull.where(fvg_bull > 0, 0)
    df['H1_fvg_bear_gap'] = fvg_bear.where(fvg_bear < 0, 0) # (Bearish gap là số âm)

    # 3. FVG Signal
    df['H1_fvg_signal'] = 0
    df.loc[df['H1_fvg_bull_gap'] > 0, 'H1_fvg_signal'] = 1
    df.loc[df['H1_fvg_bear_gap'] < 0, 'H1_fvg_signal'] = -1
    
    logging.info(f"Đã 'nấu' xong FVG. (Bull: {len(df[df['H1_fvg_signal'] == 1])}, Bear: {len(df[df['H1_fvg_signal'] == -1])})")
    return df
# === (HẾT MÓN 1) ===


def build_features_V5_H1_ONLY(df_h1_base):
    """
    "Nấu" features V5 (Chỉ "nấu" 20 món H1 + 3 món FVG)
    (Input của hàm này BẮT BUỘC chỉ là 5 cột H1 "cách ly")
    """
    logging.info("Đang bắt đầu 'nấu' features V5 (H1_RSI, H1_MACD, H1_FVG)...")
    
    # "Copy" 5 món H1 gốc để "độ"
    df = df_h1_base.copy()
    
    # === (XÁC ĐỊNH TÊN CỘT GỐC H1) ===
    col_open = 'H1_Open'
    col_high = 'H1_High'
    col_low = 'H1_Low'
    col_close = 'H1_Close'
    col_volume = 'H1_Volume'
    # ===================
    
    # --- 1. Momentum (Động lượng) (6 món) ---
    logging.info("Đang 'nấu' món 1: Momentum (RSI, STOCH, MACD)...")
    df.ta.rsi(close=col_close, length=14, append=True) # (Tên cột: RSI_14)
    df.ta.stoch(high=col_high, low=col_low, close=col_close, k=14, d=3, smooth_k=3, append=True)
    macd = df.ta.macd(close=col_close, fast=12, slow=26, signal=9, append=False)
    df['H1_MACD_12_26_9'] = macd['MACD_12_26_9']
    df['H1_MACDh_12_26_9'] = macd['MACDh_12_26_9']
    df['H1_MACDs_12_26_9'] = macd['MACDs_12_26_9']
    
    # --- 2. Volatility (Biến động) (5 món) ---
    logging.info("Đang 'nấu' món 2: Volatility (ATR, BBands)...")
    df.ta.atr(high=col_high, low=col_low, close=col_close, length=14, append=True) # (Tên cột: ATRr_14)
    bbands = df.ta.bbands(close=col_close, length=20, std=2, append=False)
    df['H1_BBL_20_2'] = bbands['BBL_20_2.0']
    df['H1_BBM_20_2'] = bbands['BBM_20_2.0']
    df['H1_BBU_20_2'] = bbands['BBU_20_2.0']
    df['H1_BBB_20_2'] = bbands['BBB_20_2.0']
    
    # --- 3. Trend (Xu hướng) (4 món) ---
    logging.info("Đang 'nấu' món 3: Trend (SMA, EMA)...")
    df['H1_SMA_20'] = df.ta.sma(close=col_close, length=20, append=False)
    df['H1_SMA_50'] = df.ta.sma(close=col_close, length=50, append=False)
    df['H1_SMA_200'] = df.ta.sma(close=col_close, length=200, append=False)
    df['H1_EMA_20'] = df.ta.ema(close=col_close, length=20, append=False)
    
    # --- 4. Volume (Khối lượng) (1 món) ---
    logging.info("Đang 'nấu' món 4: Volume (OBV)...")
    df.ta.obv(close=col_close, volume=col_volume, append=True) # (Tên cột: OBV)
    
    # --- 5. "Chế" (Quan hệ) (4 món) ---
    logging.info("Đang 'chế' món 5: Features 'quan hệ' H1...")
    df['H1_price_vs_sma50'] = df[col_close] - df['H1_SMA_50']
    df['H1_price_vs_sma200'] = df[col_close] - df['H1_SMA_200']
    df['H1_price_vs_bbu'] = df[col_close] - df['H1_BBU_20_2']
    df['H1_price_vs_bbl'] = df[col_close] - df['H1_BBL_20_2']
    
    # --- 6. (MÓN MỚI V3): FVG (Smart Money) (3 món) ---
    # (Nó sẽ "ăn" H1_High, H1_Low và "nhả" ra 3 cột FVG mới)
    df = calculate_fvg(df, col_high, col_low)
    
    logging.info(f"Đã 'nấu' xong V5! Tổng số 'món ăn' (cột) hiện tại: {len(df.columns)}")
    
    # Dọn dẹp rác (NaN) do 'nấu'
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    df.fillna(0, inplace=True) # "Trám" 0 vào nốt những lỗ
    
    return df

if __name__ == "__main__":
    logging.info(f"=== BẮT ĐẦU 'LÒ NẤU THỨC ĂN' V5 (H1-ONLY 'Cách Ly') ===")
    
    try:
        logging.info(f"Đang tải 'nguyên liệu thô' (53 món) từ: {MASTER_FILE_PATH}...")
        df_53_mon = pd.read_parquet(MASTER_FILE_PATH)
        logging.info(f"Tải 'nguyên liệu' (gốc {df_53_mon.shape[1]} cột) thành công. Shape: {df_53_mon.shape}")
        
        # === (SỬA V5: "CÁCH LY" 5 CỘT H1) ===
        base_cols = ['H1_Open', 'H1_High', 'H1_Low', 'H1_Close', 'H1_Volume']
        if not all(col in df_53_mon.columns for col in base_cols):
             logging.error(f"LỖI CHÍ MẠNG: 53 món 'thô' bị thiếu 5 cột 'cơ sở' (H1_Open...H1_Volume)")
             logging.error("Không thể 'nấu' món H1 'cách ly'.")
             sys.exit(1)

        logging.info(f"Đã 'cách ly' 5 cột 'cơ sở' (H1 OHLCV) để 'nấu'...")
        df_h1_base_only = df_53_mon[base_cols]
        # === (HẾT SỬA V5) ===

        # "Nấu" features V5 (Chỉ truyền 5 cột "cách ly" vào)
        df_features_V5 = build_features_V5_H1_ONLY(df_h1_base_only.copy())
        
        if df_features_V5 is None:
             logging.error("Dừng 'Lò V5' do 'nguyên liệu' H1 bị 'lỗi'.")
             sys.exit(1)
             
        # Ghi ra file "xịn" V5
        logging.info(f"Đang lưu 'thức ăn' H1 'cách ly' (V5) vào: {OUTPUT_FILE_PATH}...")
        df_features_V5.to_parquet(OUTPUT_FILE_PATH, index=True)
        
        logging.info(f"\n{'='*70}\n === HOÀN TẤT V5 (H1 'Cách Ly')! 'THỨC ĂN XỊN' ĐÃ SẴN SÀNG TẠI: {OUTPUT_FILE_PATH} ===\n{'='*70}")
        logging.info(f"Shape cuối cùng của 'thức ăn' V5: {df_features_V5.shape} (~28 món H1 'tinh khiết')")
        logging.info("Các 'lò' (XGBoost V4, Transformer V1) từ giờ sẽ 'ăn' file này.")

    except FileNotFoundError:
        logging.error(f"LỖI CHÍ MẠNG: Không tìm thấy file 'nguyên liệu thô' {MASTER_FILE_PATH}")
        logging.error("Đại ca đã chạy 'Lò' (data_service.py) [Chế độ 2] chưa?")
    except Exception as e:
        logging.error(f"LỖI 'NẤU' THỨC ĂN V5: {e}")