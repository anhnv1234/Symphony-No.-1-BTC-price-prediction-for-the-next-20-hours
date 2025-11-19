import pandas as pd
import numpy as np
import os
import logging
import joblib
import argparse
import schedule
import time
import sys # (Thêm sys để "bịt miệng" stdout)
import matplotlib
matplotlib.use('Agg') # <<<--- QUAN TRỌNG: Chế độ "vẽ" không cần màn hình
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec # (SỬA V53: Dùng "lưới" "xịn")
import torch
import torch.nn as nn
import math # (Cần cho "Não" Transformer)
import warnings

# --- "ĐỘ" HÀNG KHỦNG "SOI" QUÁ KHỨ (stumpy) ---
try:
    import stumpy
except ImportError:
    print("LỖI: Thiếu 'linh kiện' stumpy (Hàng Khủng).")
    print("Đại ca vui lòng chạy: pip install stumpy")
    exit()

# --- (FIX "NẾN") "ĐỘ" HÀNG VẼ "NẾN" (mplfinance) ---
try:
    import mplfinance as mpf
except ImportError:
    print("LỖI: Thiếu 'linh kiện' mplfinance (Hàng \"Nến\" Xịn).")
    print("Đại ca vui lòng chạy: pip install mplfinance")
    exit()

# --- SỬA LỖI IMPORT (THEO FILE CỦA ĐẠI CA) ---
try:
    # (Dùng "Lò" Data Service V23 (file 11:24 AM) của đại ca)
    from data_service import MasterDataServiceV23
except ImportError:
    print("LỖI: Không tìm thấy file 'data_service.py' (chứa MasterDataServiceV23).")
    print("Vui lòng đảm bảo file đó ở cùng thư mục.")
    exit()
# ----------------------------------------

# --- (MÓN 1) "BỊT MIỆNG" TOÀN BỘ CẢNH BÁO "RÁC" ---
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- Cấu hình logging (ĐÃ SỬA LỖI LOGGER) ---
# (Sửa lỗi UTF-8 cho Windows)
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass
    
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - [LiveBot_V53_ALL] - %(message)s',
                    handlers=[
                        logging.FileHandler("log_05_live_bot_V53_ALL.log", mode='w', encoding='utf-8'), 
                        logging.StreamHandler(sys.stdout) # (Vẫn "phun" ra console)
                    ])

# --- Hằng số "Vẽ" V53 ---
CHART_FILENAME = 'live_prediction_chart_V53_ALL.png' 
LOOKFORWARD = 20 # (Mặc định của CVAE/TCVAE)

# --- (SỬA V53i) "SIẾT CỔ" TIMEGAN (0.05 = 5% độ biến động) ---
TIMEGAN_DAMPING_FACTOR = 0.05 

# --- (SỬA V53) CẤU HÌNH "BỘ NÃO" (Phải "khớp" 100% lúc "luyện") ---
# (CVAE-LSTM & TimeGAN-GRU)
LATENT_DIM_CVAE = 32 
HIDDEN_DIM_TIMEGAN = 24 
NUM_FEATURES_GOC = 53 # (Số món "gốc")
# (TCVAE)
D_MODEL = 64      
N_HEAD = 4        
NUM_ENC_LAYERS = 2 
NUM_DEC_LAYERS = 2 

# --- (SỬA V53) BIẾN TOÀN CỤC (Load 6 Não + 1 Scaler) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCALER_V23 = None # (Scaler "chí mạng" 53 món)
MASTER_DATA_COLUMNS = None 
H1_COL_INDICES = {} # (Vị trí "móc" H1_Close)
# (Biến "toàn cục" để "lấy" thông số "unscale" H1_Close - Sẽ được "móc" (get) lúc load_all_brains)
SCALER_CLOSE_IDX = -1
SCALER_CLOSE_MIN = 0.0
SCALER_SCALE_CLOSE = 1.0

# (6 "Não")
CVAE_LSTM_50 = None
CVAE_LSTM_168 = None
TIMEGAN_G_50 = None
TIMEGAN_R_50 = None
TIMEGAN_G_168 = None
TIMEGAN_R_168 = None
TCVAE_50 = None
TCVAE_168 = None

# =========================================================================
# 💡 BƯỚC 1: "BÊ" (COPY) CÁC "KHUÔN NÃO" (CLASSES) TỪ 3 FILE "LÒ"
# =========================================================================

# --- "KHUÔN NÃO" 1: CVAE-LSTM (Từ file train_cvae_V11.py) ---
def sampling(args):
    z_mean, z_log_var = args
    batch = z_mean.shape[0]; dim = z_mean.shape[1]
    epsilon = torch.randn(size=(batch, dim)).to(device)
    return z_mean + torch.exp(0.5 * z_log_var) * epsilon

class CVAE_LSTM_Decoder(nn.Module):
    def __init__(self, lookback, lookforward, num_features, latent_dim, num_heads=4):
        super(CVAE_LSTM_Decoder, self).__init__()
        self.lookforward = lookforward; self.num_features = num_features; self.latent_dim = latent_dim 
        self.lstm_past_1 = nn.LSTM(num_features, 64, batch_first=True, num_layers=1)
        self.lstm_past_2 = nn.LSTM(64, 64, batch_first=True, num_layers=1)
        self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=num_heads, batch_first=True) 
        self.z_to_query_upscaler = nn.Linear(latent_dim, 64)
        self.dense_combine = nn.Linear(latent_dim + 64 + 64, 64 * lookforward)
        self.lstm_gen = nn.LSTM(64, 128, batch_first=True, num_layers=1)
        self.time_dist_dense = nn.Linear(128, num_features)
        self.relu = nn.ReLU(); self.sigmoid = nn.Sigmoid()
        
    def forward(self, condition_input, latent_input):
        h_past_seq, (last_hidden_past, _) = self.lstm_past_1(condition_input)
        _, (last_hidden_past_2, _) = self.lstm_past_2(h_past_seq)
        cond_features = self.relu(last_hidden_past_2.squeeze(0)) 
        z_query_upscaled = self.relu(self.z_to_query_upscaler(latent_input)) 
        query_vector = self.relu(cond_features + z_query_upscaled) 
        query_vector = query_vector.unsqueeze(1) 
        context_vector, attn_weights = self.attention(
            query=query_vector, key=h_past_seq, value=h_past_seq
        )
        context_vector = context_vector.squeeze(1) 
        combined = torch.cat([latent_input, cond_features, context_vector], dim=1)
        x = self.relu(self.dense_combine(combined))
        x = x.view(-1, self.lookforward, 64)
        x, _ = self.lstm_gen(x)
        x = self.time_dist_dense(x)
        reconstruction = self.sigmoid(x)
        return reconstruction, attn_weights
# --- (Hết "khuôn" CVAE-LSTM) ---

# --- "KHUÔN NÃO" 2: TIMEGAN-GRU (Từ file train_timegan_V4.py) ---
class BaseGRU(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers=2):
        super().__init__()
        self.rnn = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        output, _ = self.rnn(x); return self.output_layer(output)

class TimeGAN_GRU_Generator(BaseGRU):
    def __init__(self, hidden_dim):
        super().__init__(hidden_dim, hidden_dim, hidden_dim)

class TimeGAN_GRU_Recovery(BaseGRU):
    def __init__(self, num_features, hidden_dim):
        super().__init__(hidden_dim, num_features, hidden_dim)
# --- (Hết "khuôn" TimeGAN-GRU) ---

# --- "KHUÔN NÃO" 3: CVAE-Transformer (Từ file 04_train_transformer_cvae_V1.py) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.permute(1, 0, 2) # (Shape: 1, max_len, d_model) (Sửa cho batch_first)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ x: Shape (Batch, Seq_Len, d_model) """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class CVAE_Trans_Decoder(nn.Module):
    def __init__(self, lookback, lookforward, num_features, d_model, n_head, num_enc_layers, num_dec_layers, latent_dim):
        super(CVAE_Trans_Decoder, self).__init__()
        self.lookforward = lookforward; self.d_model = d_model
        self.embed_past = nn.Linear(num_features, d_model)
        self.z_embed = nn.Linear(latent_dim, d_model)
        self.past_feature_embed = nn.Linear(d_model * lookback, d_model)
        self.pos_encoder_past = PositionalEncoding(d_model, max_len=lookback)
        self.pos_encoder_future_query = PositionalEncoding(d_model, max_len=lookforward)
        encoder_layer_past = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, batch_first=True)
        self.transformer_encoder_past = nn.TransformerEncoder(encoder_layer_past, num_layers=num_enc_layers)
        self.dense_upsample = nn.Linear(d_model + d_model, d_model * lookforward)
        decoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, batch_first=True)
        self.transformer_decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_dec_layers)
        self.output_layer = nn.Linear(d_model, num_features)
        self.relu = nn.ReLU(); self.sigmoid = nn.Sigmoid()
        
    def forward(self, condition_input, latent_input):
        x_past = self.embed_past(condition_input)
        x_past = self.pos_encoder_past(x_past)
        past_features_seq = self.transformer_encoder_past(x_past)
        past_features_flat = past_features_seq.mean(dim=1) 
        z_features = self.relu(self.z_embed(latent_input)) 
        combined = torch.cat([past_features_flat, z_features], dim=1)
        upsampled_features = self.relu(self.dense_upsample(combined))
        future_query = upsampled_features.view(-1, self.lookforward, self.d_model)
        future_query = self.pos_encoder_future_query(future_query)
        reconstruction_features = self.transformer_decoder(future_query)
        reconstruction = self.sigmoid(self.output_layer(reconstruction_features))
        return reconstruction, None 
# --- (Hết "khuôn" CVAE-Transformer) ---


# =========================================================================
# 💡 BƯỚC 2: "HỒI SINH" 6 "NÃO" VÀ 1 "SCALER"
# =========================================================================

def load_all_brains():
    """Load 6 file "Não Bộ" (50, 168) và 1 Scaler (V23) vào RAM."""
    global SCALER_V23
    global CVAE_LSTM_50, CVAE_LSTM_168
    global TIMEGAN_G_50, TIMEGAN_R_50, TIMEGAN_G_168, TIMEGAN_R_168
    global TCVAE_50, TCVAE_168
    global MASTER_DATA_COLUMNS, H1_COL_INDICES
    global SCALER_CLOSE_IDX, SCALER_CLOSE_MIN, SCALER_SCALE_CLOSE # (Thêm)
    
    logging.info(f"Đang nạp 'Hội Đồng Não Bộ' (6 Não 53 Món) và Scaler V23...")
    
    # --- Tên file "chí mạng" ---
    scaler_file = os.path.join('01_Processed_Data', 'cvae_scaler_V23.gz')
    
    cvae_lstm_50_file = os.path.join('03_Models', f'cvae_decoder_V11_100PCT_50_{LOOKFORWARD}.pth')
    cvae_lstm_168_file = os.path.join('03_Models', f'cvae_decoder_V11_100PCT_168_{LOOKFORWARD}.pth')
    
    timegan_50_file = os.path.join('03_Models', 'advanced_tsgan_model_50_final.pth')
    timegan_168_file = os.path.join('03_Models', 'advanced_tsgan_model_168_final.pth')
    
    tcvae_50_file = os.path.join('03_Models', f'transformer_cvae_decoder_V13_50_{LOOKFORWARD}_best.pth')
    # (Đại ca 'sửa' tên file này (nếu cần) cho 'khớp' với file 'resume' đã 'rename' nhé)
    tcvae_168_file = os.path.join('03_Models', 'transformer_cvae_model_V13_168_resume.pth')
    
    try:
        # 1. "Hồi sinh" Scaler V23 (53 món)
        SCALER_V23 = joblib.load(scaler_file)
        logging.info(f"Nạp Scaler V23 (53 món) thành công.")
        
        # "Móc" (Get) 53 "tên món" (features) và "vị trí" (index) H1_Close
        try:
            MASTER_DATA_COLUMNS = list(SCALER_V23.feature_names_in_)
        except AttributeError:
             logging.warning("Scaler V23 'cũ', không có 'feature_names_in_'. Đang 'móc' H1_Close 'thủ công'...")
             df_temp = pd.read_parquet(os.path.join('02_Master_Data', 'btcusdt_master_data.parquet'))
             MASTER_DATA_COLUMNS = df_temp.columns.tolist()
             
        H1_COL_INDICES['Close'] = MASTER_DATA_COLUMNS.index('H1_Close')
        H1_COL_INDICES['Open'] = MASTER_DATA_COLUMNS.index('H1_Open')
        H1_COL_INDICES['High'] = MASTER_DATA_COLUMNS.index('H1_High')
        H1_COL_INDICES['Low'] = MASTER_DATA_COLUMNS.index('H1_Low')
        H1_COL_INDICES['Volume'] = MASTER_DATA_COLUMNS.index('H1_Volume') 
        logging.info(f"Đã 'móc' vị trí 5 cột H1 (Close={H1_COL_INDICES['Close']})")
        
        # (SỬA V53) "Móc" (Get) thông số "unscale" H1_Close
        SCALER_CLOSE_IDX = H1_COL_INDICES['Close']
        SCALER_CLOSE_MIN = SCALER_V23.min_[SCALER_CLOSE_IDX]
        SCALER_SCALE_CLOSE = SCALER_V23.scale_[SCALER_CLOSE_IDX]
        logging.info(f"Đã 'móc' thông số Unscale (Min: {SCALER_CLOSE_MIN}, Scale: {SCALER_SCALE_CLOSE})")
        
        # 2. "Hồi sinh" "Não" CVAE-LSTM (V11)
        CVAE_LSTM_50 = CVAE_LSTM_Decoder(50, LOOKFORWARD, NUM_FEATURES_GOC, LATENT_DIM_CVAE).to(device)
        CVAE_LSTM_50.load_state_dict(torch.load(cvae_lstm_50_file, map_location=device))
        CVAE_LSTM_50.eval()
        CVAE_LSTM_168 = CVAE_LSTM_Decoder(168, LOOKFORWARD, NUM_FEATURES_GOC, LATENT_DIM_CVAE).to(device)
        CVAE_LSTM_168.load_state_dict(torch.load(cvae_lstm_168_file, map_location=device))
        CVAE_LSTM_168.eval()
        logging.info(f"Nạp 'Não 1' (CVAE-LSTM V11 x2) thành công.")
        
        # 3. "Hồi sinh" "Não" TimeGAN (V4)
        checkpoint_50 = torch.load(timegan_50_file, map_location=device)
        TIMEGAN_G_50 = TimeGAN_GRU_Generator(HIDDEN_DIM_TIMEGAN).to(device)
        TIMEGAN_R_50 = TimeGAN_GRU_Recovery(NUM_FEATURES_GOC, HIDDEN_DIM_TIMEGAN).to(device)
        TIMEGAN_G_50.load_state_dict(checkpoint_50['G_state_dict']); TIMEGAN_G_50.eval()
        TIMEGAN_R_50.load_state_dict(checkpoint_50['R_state_dict']); TIMEGAN_R_50.eval()
        
        checkpoint_168 = torch.load(timegan_168_file, map_location=device)
        TIMEGAN_G_168 = TimeGAN_GRU_Generator(HIDDEN_DIM_TIMEGAN).to(device)
        TIMEGAN_R_168 = TimeGAN_GRU_Recovery(NUM_FEATURES_GOC, HIDDEN_DIM_TIMEGAN).to(device)
        TIMEGAN_G_168.load_state_dict(checkpoint_168['G_state_dict']); TIMEGAN_G_168.eval()
        TIMEGAN_R_168.load_state_dict(checkpoint_168['R_state_dict']); TIMEGAN_R_168.eval()
        logging.info(f"Nạp 'Não 2' (TimeGAN-GRU V4 x2) thành công.")
        
        # 4. "Hồi sinh" "Não" TCVAE (V1)
        
        # --- (NÃO 50 - Giữ nguyên - File "plot" chạy OK) ---
        TCVAE_50 = CVAE_Trans_Decoder(50, LOOKFORWARD, NUM_FEATURES_GOC, D_MODEL, N_HEAD, NUM_ENC_LAYERS, NUM_DEC_LAYERS, LATENT_DIM_CVAE).to(device)
        TCVAE_50.load_state_dict(torch.load(tcvae_50_file, map_location=device))
        TCVAE_50.eval()
        
        # --- (NÃO 168 - *SỬA LỖI "VẠCH BALO"*) ---
        TCVAE_168 = CVAE_Trans_Decoder(168, LOOKFORWARD, NUM_FEATURES_GOC, D_MODEL, N_HEAD, NUM_ENC_LAYERS, NUM_DEC_LAYERS, LATENT_DIM_CVAE).to(device)
        
        logging.info("Đang 'vạch Balo' (unpack) Não TCVAE 168 (do lỗi state_dict)...")
        checkpoint_168 = torch.load(tcvae_168_file, map_location=device)
        TCVAE_168.load_state_dict(checkpoint_168['decoder_state_dict']) 
        
        TCVAE_168.eval()
        
        logging.info(f"Nạp 'Não 3' (TCVAE V1 x2) thành công.")
        
        return True
        
    except Exception as e:
        logging.critical(f"LỖI CHÍ MẠNG: Không nạp được 'Não Bộ'! Lỗi: {e}")
        logging.critical("Đại ca đã chạy 'ĐÚNG THÚ TỰ' 3 Lò (V11 -> V4 -> TCVAE V1) chưa?")
        return False

# =========================================================================
# 💡 BƯỚC 3: "HẬU CẦN" (LẤY MỒI, "SOI" QUÁ KHỨ, VẼ)
# =========================================================================

# --- (*BẮT ĐẦU SỬA LỖI V53d: "CAMERA AN NINH"*) ---
def get_current_window_scaled_from_df(df_master_full, lookback):
    """
    "Cắt" (slice) "mồi" (past window) 50 hoặc 168 nến (đã "chuẩn hóa")
    LƯU Ý: Hàm này "giả định" df_master_full ĐÃ ĐƯỢC "VỆ SINH" SẠCH SẼ
    """
    past_window_df = df_master_full.iloc[-lookback:]
            
    if len(past_window_df) < lookback:
        logging.warning(f"Không đủ {lookback} nến. Chờ thêm...")
        return None, None
        
    # --- "CAMERA" SỐ 1: "Mồi" "Thô" (trước khi "ép dẻo") ---
    # (TẮT "CAMERA" 1)
    nan_count = past_window_df.isna().sum().sum()
    if nan_count > 0:
        logging.error(f"[V53p] LỖI NGHIÊM TRỌNG: 'MỒI' 'THÔ' VẪN CÒN {nan_count} 'NaN'!!!")
    # --- (Hết "Camera" 1) ---

    # Lấy "thức ăn" (53 món)
    window_scaled = SCALER_V23.transform(past_window_df)
    
    # --- "CAMERA" SỐ 2: "Mồi" "Ép Dẻo" (sau khi "scale") ---
    # (TẮT "CAMERA" 2)
    scaled_close_col = window_scaled[:, SCALER_CLOSE_IDX]
    nan_count_scaled = np.isnan(window_scaled).sum()
    if nan_count_scaled > 0:
         logging.error(f"[V53p] LỖI NGHIÊM TRỌNG: 'MỒI' 'ÉP DẺO' BỊ 'NaN' ({nan_count_scaled} 'lỗi')!!!")
    elif np.all(scaled_close_col == 0):
        logging.warning("[V53p] CẢNH BÁO: 'Mồi' H1_Close 'ÉP DẺO' 'toàn' 'số' 0! (Đây có thể là 'thủ phạm')")
    # --- (Hết "Camera" 2) ---

    window_scaled_gpu = torch.tensor(window_scaled, dtype=torch.float32).unsqueeze(0).to(device) 
    
    # Lấy "hàng" (OHLCV) "thô" (để "vẽ" nến)
    past_ohlcv = past_window_df[['H1_Open', 'H1_High', 'H1_Low', 'H1_Close', 'H1_Volume']]
    
    return window_scaled_gpu, past_ohlcv
# --- (*KẾT THÚC SỬA LỖI V53d*) ---

def get_mean_scenario(decoder_model, window_scaled_gpu):
    """
    "Vẽ" 1 kịch bản "Dự Kiến" (Mean Z-vector)
    """
    with torch.no_grad():
        z_noise = torch.zeros(1, LATENT_DIM_CVAE).to(device)
        future_fake_scaled, _ = decoder_model(window_scaled_gpu, z_noise)
        return future_fake_scaled.cpu().numpy()

# --- (SỬA LỖI V53b: "VÊNH CHÂN" TIMEGAN) ---
def get_timegan_scenario(g_model, r_model, lookback):
    """
    "Vẽ" 1 kịch bản "Tự Bịa" của TimeGAN
    """
    with torch.no_grad():
        # (SỬA LỖI: Phải "bịa" 20 (LOOKFORWARD) nến, không phải "lookback" nến!)
        z_noise = torch.zeros(1, LOOKFORWARD, HIDDEN_DIM_TIMEGAN).to(device) # (Z=0)
        h_fake_scaled = g_model(z_noise)
        x_fake_scaled = r_model(h_fake_scaled)
        return x_fake_scaled.cpu().numpy()

# --- (SỬA LỖI V53e: "CÔNG THỨC UN SCALE") ---
def unscale_h1_close(scaled_data_np):
    """
    Hàm "thần thánh": "Unscale" chỉ riêng cột H1_Close
    """
    global SCALER_CLOSE_IDX, SCALER_CLOSE_MIN, SCALER_SCALE_CLOSE
    
    try:
        if scaled_data_np.ndim == 3:
            scaled_close = scaled_data_np[0, :, SCALER_CLOSE_IDX]
        else:
            scaled_close = scaled_data_np[:, SCALER_CLOSE_IDX]
            
        # --- (SỬA V53e: SỬA CÔNG THỨC TOÁN) ---
        # (Công thức CŨ: (scaled * Scale) + Min)
        unscaled_close = (scaled_close - SCALER_CLOSE_MIN) / SCALER_SCALE_CLOSE
        # --- (HẾT SỬA V53e) ---
            
        return unscaled_close
        
    except Exception as e:
        logging.error(f"Lỗi 'Unscale': {e}")
        if scaled_data_np.ndim == 3:
            return np.zeros(scaled_data_np.shape[1]) 
        else:
            return np.zeros(LOOKFORWARD)

# --- (SỬA V53j: THUẬT TOÁN TÌM KIẾM THÔNG MINH - TRÁNH TRÙNG LẶP) ---
def find_top_3_similar_patterns(current_window_raw_close, all_historical_close_series, lookback, lookforward):
    """
    Dùng "chiêu" stumpy.mass để "quét" Top 3 "Anh Em Song Sinh" (ĐÃ NÂNG CẤP V53j)
    """
    logging.info(f"(LB={lookback}) Đang 'quét' {len(all_historical_close_series)} nến quá khứ để tìm 'Top 3 Anh Em'...")
    
    # 1. "Mẫu" (Query) (đã Z-Score)
    query_window = (current_window_raw_close - np.mean(current_window_raw_close)) / (np.std(current_window_raw_close) + 1e-9)
    
    # 2. "Lịch Sử" (Search)
    history_to_search_series = all_historical_close_series.iloc[:-(lookback + lookforward)]
    history_to_search_values = history_to_search_series.values
    
    # 3. "Quét"
    try:
        distance_profile = stumpy.mass(query_window, history_to_search_values)
    except Exception as e:
        logging.error(f"(LB={lookback}) Lỗi khi 'quét' stumpy.mass: {e}.")
        return "Lỗi: Không thể 'quét' quá khứ.", []
        
    # --- (BẮT ĐẦU NÂNG CẤP V53j: CƠ CHẾ "VÙNG CẤM ĐỊA") ---
    top_3_matches = []
    dist_profile_copy = distance_profile.copy() # Copy để "phá" mà không ảnh hưởng gốc
    
    # Bán kính vùng cấm (để 2 pattern không bị "dính" nhau quá gần)
    exclusion_zone = lookback // 2 
    
    for _ in range(3): # Tìm 3 thằng
        # Tìm thằng nhỏ nhất (giống nhất) hiện tại
        idx = np.argmin(dist_profile_copy)
        score = dist_profile_copy[idx]
        
        # Nếu "hết hàng" (toàn vô cực) thì nghỉ
        if score == np.inf: break

        timestamp = history_to_search_series.index[idx]
        top_3_matches.append({'index': idx, 'timestamp': timestamp, 'score': score})

        # "Khoanh vùng cấm địa" (Đặt distance xung quanh idx thành Vô Cực)
        # Để lần lặp sau không tìm thấy nó nữa
        start_ex = max(0, idx - exclusion_zone)
        end_ex = min(len(dist_profile_copy), idx + exclusion_zone)
        dist_profile_copy[start_ex:end_ex] = np.inf
        
    # --- (KẾT THÚC NÂNG CẤP V53j) ---
    
    similarity_text = f"Top 3 Tương Đồng (LB={lookback}):"
    if top_3_matches:
        similarity_text += f"\n  1. {top_3_matches[0]['timestamp'].strftime('%Y-%m-%d %H:00')} (Score: {top_3_matches[0]['score']:.2f})"
        logging.info(f"--- (LB={lookback}) {similarity_text} ---")
            
    return similarity_text, top_3_matches

# --- (SỬA V53L: GIAO DIỆN OVERLAY VOLUME) ---
def draw_super_chart(data_lb50, data_lb168, df_master_full):
    """
    VẼ BIỂU ĐỒ "SIÊU CẤP" (2 PHẦN: 50 vs 168) - TradingView Style (Volume Overlay)
    """
    logging.info(f"Đang 'vẽ' biểu đồ 'Siêu Cấp' (TradingView Style) -> {CHART_FILENAME}...")
    
    # V53L: Chỉ dùng 2 hàng, không tách volume riêng nữa
    fig = plt.figure(figsize=(40, 16)) 
    gs = GridSpec(2, 4, figure=fig, width_ratios=[3, 1, 1, 1])
    
    # --- Tạo Axes (Chỉ cần 1 ax cho mỗi chart) ---
    axes_50 = [fig.add_subplot(gs[0, i]) for i in range(4)]
    axes_168 = [fig.add_subplot(gs[1, i]) for i in range(4)]
    
    plot_chart_section(axes_50, data_lb50, 50, df_master_full)
    plot_chart_section(axes_168, data_lb168, 168, df_master_full)
    
    fig.suptitle(f"BOT LIVE V53p (Final No Gap) - {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M:%S UTC')}", fontsize=20, weight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    plt.savefig(CHART_FILENAME)
    plt.close(fig) 
    logging.info(f"Đã 'vẽ' (TradingView Style) và lưu xong biểu đồ tại: {CHART_FILENAME}")

# --- (SỬA V53m/p: VÁ LỖI "KHE HỞ" + "CẮT ĐUÔI THỪA") ---
def plot_chart_section(axes, data, lookback, df_master_full):
    """
    Hàm "Vẽ Con" V53L: Vẽ Giá & Volume (Overlay) trên cùng 1 ô
    """
    past_ohlcv = data['past_ohlcv']
    scenarios = data['scenarios']
    similarity_text = data['similarity_text']
    top_3_matches = data['top_3_matches']
    
    # === 1. VẼ MAIN CHART (Hiện tại) ===
    ax_main = axes[0] 
    
    # (FIX V53p: Dọn sạch dữ liệu rác NaN trước khi vẽ để len() chuẩn xác)
    past_ohlcv = past_ohlcv.dropna(subset=['H1_Close', 'H1_Open', 'H1_High', 'H1_Low'])
    
    # Đổi tên cột cho mplfinance
    past_data_mpf = past_ohlcv.rename(columns={
        'H1_Open': 'Open', 'H1_High': 'High', 'H1_Low': 'Low', 'H1_Close': 'Close', 'H1_Volume': 'Volume'
    })
    
    # (A) Vẽ Nến (Giá)
    mpf.plot(past_data_mpf, type='candle', ax=ax_main, style='yahoo', volume=False, show_nontrading=False)
    
    # (B) Vẽ Volume Overlay
    ax_vol = ax_main.twinx()
    vol_colors = np.where(past_data_mpf['Close'] >= past_data_mpf['Open'], 'green', 'red')
    ax_vol.bar(np.arange(len(past_data_mpf)), past_data_mpf['Volume'], color=vol_colors, alpha=0.3, width=0.6)
    max_vol = past_data_mpf['Volume'].max()
    if max_vol > 0:
        ax_vol.set_ylim(0, max_vol * 4)
    ax_vol.axis('off') 
    ax_main.set_zorder(2) 
    ax_main.patch.set_visible(False)
    ax_vol.set_zorder(1)

    # (C) Vẽ 3 Kịch bản Dự báo (V53i Logic + V53m Fix Gap)
    last_close_price = past_data_mpf['Close'].iloc[-1]
    
    # --- (SỬA V53m: TÍNH TOÁN ĐIỂM BẮT ĐẦU VẼ DỰA TRÊN DATA THỰC TẾ) ---
    len_data = len(past_data_mpf)
    plot_x = np.arange(len_data - 1, len_data - 1 + (LOOKFORWARD + 1)) 
    # ---------------------------------------------------
    
    colors = {'CVAE-LSTM': 'blue', 'TCVAE': 'red', 'TimeGAN': 'green'}
    
    for name, sim_scaled in scenarios.items():
        sim_unscaled = unscale_h1_close(sim_scaled)
        if len(sim_unscaled) != LOOKFORWARD: continue 
             
        try:
            if name == 'TimeGAN':
                sim_series = pd.Series(sim_unscaled)
                sim_unscaled = sim_series.ewm(span=3, adjust=False).mean().values 
                
            start_price_sim = sim_unscaled[0]
            if start_price_sim == 0: start_price_sim = 1e-9
            growth_factors = sim_unscaled / start_price_sim
            
            if name == 'TimeGAN':
                growth_factors = 1.0 + (growth_factors - 1.0) * TIMEGAN_DAMPING_FACTOR 
            
            sim_unscaled = last_close_price * growth_factors
            
        except Exception: pass
             
        plot_line = np.insert(sim_unscaled, 0, last_close_price)
        
        if len(plot_x) == len(plot_line):
            ax_main.plot(plot_x, plot_line, color=colors[name], linestyle='--', linewidth=2, label=f'KB {name}')
    
    ax_main.set_title(f"PHẦN {lookback} (Hiện Tại & Dự Báo)\n{similarity_text}", fontsize=14, loc='left')
    ax_main.legend(loc='upper left'); ax_main.grid(True)
    
    # === 2. VẼ 3 CHART PHỤ (TOP MATCHES) ===
    for i, match in enumerate(top_3_matches):
        ax_sub = axes[i+1] 
        
        start_idx = match['index']
        end_idx = match['index'] + lookback + LOOKFORWARD 
        segment_data_raw = df_master_full.iloc[start_idx:end_idx]
        segment_data = segment_data_raw[['H1_Open', 'H1_High', 'H1_Low', 'H1_Close', 'H1_Volume']]
        segment_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        v_line_time = segment_data.index[lookback-1]
        mpf.plot(segment_data, type='candle', ax=ax_sub, style='yahoo', volume=False, 
                 vlines=dict(vlines=[v_line_time], linestyle='--', colors='b', linewidths=2),
                 show_nontrading=False)
        
        ax_sub_vol = ax_sub.twinx()
        vol_colors_sub = np.where(segment_data['Close'] >= segment_data['Open'], 'green', 'red')
        ax_sub_vol.bar(np.arange(len(segment_data)), segment_data['Volume'], color=vol_colors_sub, alpha=0.3, width=0.6)
        
        max_vol_sub = segment_data['Volume'].max()
        if max_vol_sub > 0:
            ax_sub_vol.set_ylim(0, max_vol_sub * 4) 
        ax_sub_vol.axis('off')
        ax_sub.set_zorder(2)
        ax_sub.patch.set_visible(False)
        ax_sub_vol.set_zorder(1)

        ax_sub.set_title(f"Top {i+1} (Score: {match['score']:.2f})\n{match['timestamp'].strftime('%Y-%m-%d')}", fontsize=12)
        ax_sub.yaxis.tick_right()
        
# ---------------------------------------------------


# --- (SỬA V53_Update_3H) HÀM CHÍNH (Chạy 6 Não + Soi lại 3h) ---
def run_hourly_update_and_predict(data_service):
    """
    Hàm CHÍNH: Chạy toàn bộ 6 "Não" mỗi giờ.
    (Đã độ thêm tính năng refresh data 3h trước khi vẽ)
    """
    logging.info("=== BẮT ĐẦU CHU KỲ 1H MỚI (V53 - 6 NÃO) ===")
    
    # --- Bước 0: Tính toán mốc "Hồi Quy" (3 Giờ Trước) ---
    # Mục đích: Ép bot tải lại nến của 3 tiếng gần nhất để trám lỗ hổng (nếu có)
    timestamp_3h_str = None
    try:
        now_utc = pd.Timestamp.now(tz='UTC')
        time_3h_ago = now_utc - pd.Timedelta(hours=3)
        # Chuyển thành timestamp ms (dành cho ccxt/binance)
        timestamp_3h_ms = int(time_3h_ago.timestamp() * 1000)
        timestamp_3h_str = str(timestamp_3h_ms)
        logging.info(f"Bot đang yêu cầu làm mới dữ liệu từ: {time_3h_ago.strftime('%H:%M')} UTC (3h trước)...")
    except Exception as e:
        logging.error(f"Lỗi tính giờ hồi quy: {e}. Sẽ chạy mặc định.")

    # --- Bước 1 & 2: Cập nhật và Gộp "Thức ăn" ---
    logging.info("Bước 1&2: Đang cập nhật & tái tạo 'thức ăn' master (53 món)...")
    try:
        # (NEW) Cố gắng gọi download với tham số start_str
        if timestamp_3h_str:
            try:
                # Hy vọng data_service của đại ca khôn, biết ăn tham số 'start_str'
                data_service.run_download_klines(start_str=timestamp_3h_str)
                logging.info("✅ Đã gọi download lại 3 giờ gần nhất thành công!")
            except TypeError:
                # Nếu data_service "ngu" không nhận tham số -> Chạy kiểu cũ
                logging.warning("⚠️ DataService không nhận tham số 'start_str'. Chạy chế độ mặc định...")
                data_service.run_download_klines()
            except Exception as e:
                logging.error(f"⚠️ Lỗi khi force download 3h: {e}. Thử chạy mặc định...")
                data_service.run_download_klines()
        else:
            # Không tính được giờ thì chạy như cũ
            data_service.run_download_klines()

        data_service.run_fetch_bitstamp_backfill() # (Chạy Lò 1.5)
        data_service.run_create_master_file() # (Chạy Lò 2)
    except Exception as e:
        logging.error(f"LỖI \"HÚT\" DATA: {e}. Bot sẽ \"ăn\" data \"cũ\" (nếu có).")

    # --- Bước 3: Đọc "Thức ăn" 1 LẦN DUY NHẤT ---
    logging.info("Bước 3: Đã có 'thức ăn' mới nhất. Bắt đầu \"ăn\" (load)...")
    
    try:
        df_master_full = pd.read_parquet(os.path.join('02_Master_Data', 'btcusdt_master_data.parquet'))
    except Exception as e:
        logging.error(f"Lỗi khi đọc file 'thức ăn' master 'btcusdt_master_data.parquet': {e}")
        return

    # --- (SỬA LỖI V53c: "VỆ SINH" "THỨC ĂN") ---
    # (Bê nguyên "bài" 'vệ sinh' từ file "plot" sang)
    logging.info("Bước 3.5: Đang \"vệ sinh\" (interpolate/fillna) toàn bộ 'thức ăn'...")
    df_master_full.interpolate(method='time', inplace=True)
    df_master_full.fillna(method='ffill', inplace=True)
    df_master_full.fillna(method='bfill', inplace=True)
    df_master_full.fillna(0, inplace=True) # "Trám" 0 (nếu vẫn còn)
    logging.info("Đã \"vệ sinh\" 'thức ăn' xong. 'Mồi' 'bây giờ' 'siêu' 'sạch'!")
    # --- (Hết Lỗi V53c) ---

    # === (SỬA V53) Bước 4: "Chạy" "Phần" 1 (Lookback 50) ===
    logging.info("--- Bắt đầu \"Phần 1\" (Lookback 50) ---")
    data_lb50 = {}
    
    # 4.1. "Móc mồi" 50 nến (Bây giờ "mồi" đã "sạch")
    window_scaled_50, past_ohlcv_50 = get_current_window_scaled_from_df(df_master_full, 50)
    
    if window_scaled_50 is None:
        logging.warning("Không thể chuẩn bị dữ liệu LB=50. Bỏ qua chu kỳ này.")
        return
        
    data_lb50['past_ohlcv'] = past_ohlcv_50
    data_lb50['scenarios'] = {}

    # 4.2. "Vẽ" (Generate) 3 "não" 50
    data_lb50['scenarios']['CVAE-LSTM'] = get_mean_scenario(CVAE_LSTM_50, window_scaled_50)
    data_lb50['scenarios']['TimeGAN'] = get_timegan_scenario(TIMEGAN_G_50, TIMEGAN_R_50, 50)
    data_lb50['scenarios']['TCVAE'] = get_mean_scenario(TCVAE_50, window_scaled_50)
    logging.info("(LB=50) Đã \"vẽ\" xong 3 kịch bản.")

    # 4.3. "Soi" (Scan) Top 3 "Anh Em" 50
    all_historical_close = df_master_full['H1_Close']
    current_window_close_50 = past_ohlcv_50['H1_Close'].values
    sim_text_50, top_3_50 = find_top_3_similar_patterns(current_window_close_50, all_historical_close, 50, LOOKFORWARD)
    data_lb50['similarity_text'] = sim_text_50
    data_lb50['top_3_matches'] = top_3_50
    
    # === (SỬA V53) Bước 5: "Chạy" "Phần" 2 (Lookback 168) ===
    logging.info("--- Bắt đầu \"Phần 2\" (Lookback 168) ---")
    data_lb168 = {}
    
    # 5.1. "Móc mồi" 168 nến (Bây giờ "mồi" đã "sạch")
    window_scaled_168, past_ohlcv_168 = get_current_window_scaled_from_df(df_master_full, 168)
    
    if window_scaled_168 is None:
        logging.warning("Không thể chuẩn bị dữ liệu LB=168. Bỏ qua chu kỳ này.")
        return
        
    data_lb168['past_ohlcv'] = past_ohlcv_168
    data_lb168['scenarios'] = {}

    # 5.2. "Vẽ" (Generate) 3 "não" 168
    data_lb168['scenarios']['CVAE-LSTM'] = get_mean_scenario(CVAE_LSTM_168, window_scaled_168)
    data_lb168['scenarios']['TimeGAN'] = get_timegan_scenario(TIMEGAN_G_168, TIMEGAN_R_168, 168)
    data_lb168['scenarios']['TCVAE'] = get_mean_scenario(TCVAE_168, window_scaled_168)
    logging.info("(LB=168) Đã \"vẽ\" xong 3 kịch bản.")

    # 5.3. "Soi" (Scan) Top 3 "Anh Em" 168
    current_window_close_168 = past_ohlcv_168['H1_Close'].values
    sim_text_168, top_3_168 = find_top_3_similar_patterns(current_window_close_168, all_historical_close, 168, LOOKFORWARD)
    data_lb168['similarity_text'] = sim_text_168
    data_lb168['top_3_matches'] = top_3_168
    
    # === Bước 6: "Vẽ" (Plot) "Siêu Ảnh" ===
    draw_super_chart(data_lb50, data_lb168, df_master_full)
    
    logging.info("=== Hoàn tất chu kỳ ===")

# --- HÀM MAIN ĐỂ CHẠY FILE ---
if __name__ == "__main__":
    
    # (Bỏ Argparser - Bot V53 "tự động" "ăn" 50 và 168)
    
    # 1. Khởi tạo "Hậu cần" (Data Service V23)
    data_service = MasterDataServiceV23(symbol='BTCUSDT')
    
    # 2. Load "Não Bộ" 1 LẦN DUY NHẤT (6 "Não")
    if not load_all_brains():
        logging.critical("Không thể khởi động bot. Dừng lại.")
        exit()
        
    # 3. Lập Lịch Chạy
    logging.info("Bot khởi động. Lập lịch chạy...")
    
    # 3.1. Chạy 1 lần ngay lúc đầu
    run_hourly_update_and_predict(data_service)
    
    # 3.2. Lập lịch chạy vào phút thứ 2 của mỗi giờ
    logging.info("Đã lập lịch chạy vào phút :02 mỗi giờ...")
    schedule.every().hour.at(":02").do(
        run_hourly_update_and_predict, 
        data_service=data_service
    )
    
    # 4. Vòng lặp "Sống"
    while True:
        schedule.run_pending()
        time.sleep(1)