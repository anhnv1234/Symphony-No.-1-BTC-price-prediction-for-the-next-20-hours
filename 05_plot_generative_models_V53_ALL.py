import pandas as pd
import numpy as np
import os
import logging
import joblib
from sklearn.preprocessing import MinMaxScaler
import warnings 
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import sys
import math # (Cần cho "Não" Transformer)

# --- (MÓN 1) "BỊT MIỆNG" CẢNH BÁO "RÁC" ---
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- CẤU HÌNH LOGGING (UTF-8) ---
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass
    
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - [PlotGenerative_V53_ALL] - %(message)s',
                    handlers=[
                        logging.FileHandler("log_05_plot_generative_V53_ALL.log", mode='w', encoding='utf-8'), 
                        logging.StreamHandler(sys.stdout)
                    ])

# --- CẤU HÌNH "PHÒNG TRIỂN LÃM" V53 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Đang dùng thiết bị: {device} (để 'vẽ')")

# --- (A) CẤU HÌNH "THỨC ĂN" (DÙNG HÀNG "THÔ" 53 MÓN VÀ SCALER "V23") ---
MASTER_FILE_PATH_THO = os.path.join('02_Master_Data', 'btcusdt_master_data.parquet')
SCALER_FILENAME_GOC = os.path.join('01_Processed_Data', 'cvae_scaler_V23.gz')
CHART_SAVE_DIR = "05_Charts_Generative_V53_ALL" # (Thư mục "trưng" ảnh VẼ 53 Món)
os.makedirs(CHART_SAVE_DIR, exist_ok=True)

# --- (B) CẤU HÌNH "NÃO" (Tên file "não xịn" 53 Món) ---
DIR_MODELS = "03_Models"
CVAE_LSTM_MODEL_FILE = "cvae_decoder_V11_100PCT_{lb}_{lf}.pth" # (Từ "Lò" 1a)
TIMEGAN_GRU_MODEL_FILE = "advanced_tsgan_model_{lb}_final.pth" # (Từ "Lò" 1b)
CVAE_TRANS_MODEL_FILE = "transformer_cvae_decoder_V13_{lb}_{lf}_best.pth" # (Từ "Lò" 3)

# --- (C) CẤU HÌNH "BỘ NÃO" (Phải "khớp" 100% lúc "luyện") ---
LOOKBACK = 50       
LOOKFORWARD_CVAE = 20  
LATENT_DIM_CVAE = 32 
HIDDEN_DIM_TIMEGAN = 24 
NUM_FEATURES_GOC = 53 

# (Cấu hình "Não" TCVAE)
D_MODEL = 64      
N_HEAD = 4        
NUM_ENC_LAYERS = 2 
NUM_DEC_LAYERS = 2 

# (Biến "toàn cục" để "lấy" thông số "unscale" H1_Close)
SCALER_CLOSE_IDX = -1
SCALER_CLOSE_MIN = 0.0
SCALER_SCALE_CLOSE = 1.0

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
# 💡 BƯỚC 2: "DỰNG" CÁC HÀM "HẬU CẦN" (LOAD DATA, UNSCALE, LẤY MỒI)
# =========================================================================

def load_data_and_set_unscaler(num_features):
    """
    Tải Scaler "V23" (53 món) và Data "thô" (53 món).
    Quan trọng: Tìm thông số "unscale" của H1_Close
    """
    global SCALER_CLOSE_IDX, SCALER_CLOSE_MIN, SCALER_CLOSE_SCALE
    
    logging.info(f"Đang tải Scaler 'gốc' V23 (53 món): {SCALER_FILENAME_GOC}...")
    try:
        scaler = joblib.load(SCALER_FILENAME_GOC)
        
        # "Móc" thông số "unscale" của H1_Close ra
        try:
            feature_names = list(scaler.feature_names_in_)
        except AttributeError:
             logging.warning("Scaler V23 'cũ', không có 'feature_names_in_'. Đang 'móc' H1_Close 'thủ công'...")
             df_temp = pd.read_parquet(MASTER_FILE_PATH_THO)
             feature_names = df_temp.columns.tolist()
             
        SCALER_CLOSE_IDX = feature_names.index('H1_Close')
        SCALER_CLOSE_MIN = scaler.min_[SCALER_CLOSE_IDX]
        SCALER_SCALE_CLOSE = scaler.scale_[SCALER_CLOSE_IDX]
        
        logging.info(f"Đã 'móc' thông số unscale (H1_Close Idx: {SCALER_CLOSE_IDX})")
        
    except FileNotFoundError:
        logging.error(f"LỖI: Không tìm thấy Scaler 'gốc' {SCALER_FILENAME_GOC}")
        logging.error("Đại ca đã chạy 'Lò' (train_cvae_V11.py) (file 'ăn' 53 món) [Bước 1] chưa?")
        return None
    except Exception as e:
        logging.error(f"LỖI: Không tải/đọc được Scaler 'gốc'. Lỗi: {e}")
        return None

    logging.info(f"Đang tải Data 'thô' (53 món): {MASTER_FILE_PATH_THO}...")
    try:
        df = pd.read_parquet(MASTER_FILE_PATH_THO)
        
        if num_features != df.shape[1]:
            logging.error(f"LỖI 'KHỚP' NÃO: 'Lò' (V11/V4) 'luyện' {num_features} món,")
            logging.error(f"nhưng file 'thô' ({MASTER_FILE_PATH_THO}) lại 'có' {df.shape[1]} món.")
            return None
        
        df.interpolate(method='time', inplace=True)
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        df.fillna(0, inplace=True) # "Trám" 0 
        
        data_scaled = scaler.transform(df)
        
        logging.info(f"Đã tải và 'chuẩn hóa' {data_scaled.shape[0]} nến H1 'thô' (53 món).")
        return data_scaled
        
    except Exception as e:
        logging.error(f"LỖI: Không tải/xử lý được Data 'thô'. Lỗi: {e}")
        return None

def get_real_sample(data_scaled, lookback, sample_idx=-1000):
    """
    "Móc" 1 mẩu "quá khứ" (lookback) làm "mồi"
    """
    try:
        real_sample_np = data_scaled[sample_idx - lookback : sample_idx]
        real_future_np = data_scaled[sample_idx : sample_idx + LOOKFORWARD_CVAE]
        
        if real_sample_np.shape[0] != lookback or real_future_np.shape[0] != LOOKFORWARD_CVAE:
            logging.error("LỖI 'MÓC MỒI': Không đủ data để 'móc' (sample_idx quá gần cuối).")
            return None, None
            
        real_sample_gpu = torch.tensor(real_sample_np, dtype=torch.float32).unsqueeze(0).to(device)
        
        return real_sample_gpu, real_future_np
        
    except Exception as e:
        logging.error(f"LỖI 'MÓC MỒI': {e}")
        return None, None

def unscale_h1_close(scaled_data_np):
    """
    Hàm "thần thánh": "Unscale" chỉ riêng cột H1_Close
    """
    try:
        if scaled_data_np.ndim == 3:
            scaled_close = scaled_data_np[0, :, SCALER_CLOSE_IDX]
        else:
            scaled_close = scaled_data_np[:, SCALER_CLOSE_IDX]
        unscaled_close = (scaled_close * SCALER_SCALE_CLOSE) + SCALER_CLOSE_MIN
        return unscaled_close
    except Exception as e:
        logging.error(f"Lỗi 'Unscale': {e}")
        return np.zeros(scaled_data_np.shape[1]) 

# =========================================================================
# 💡 BƯỚC 3: "DỰNG" 3 "PHÒNG TRIỂN LÃM" (CVAE, TIMEGAN, TCVAE)
# =========================================================================

def plot_cvae_scenarios(decoder_model, real_past_gpu, real_future_np, lookback, lookforward, num_scenarios, model_name, file_name_suffix):
    """
    Hàm "Vẽ" CHUNG (Dùng cho CVAE-LSTM và CVAE-Transformer)
    """
    logging.info(f"--- Đang 'vẽ' {num_scenarios} kịch bản {model_name} (LB={lookback}) ---")
    
    decoder_model.eval()
    
    past_unscaled = unscale_h1_close(real_past_gpu.cpu().numpy())
    future_unscaled_real = unscale_h1_close(real_future_np)
    
    x_past = np.arange(lookback)
    x_future = np.arange(lookback, lookback + lookforward)
    
    plt.figure(figsize=(20, 8))
    plt.plot(x_past, past_unscaled, 'r-', linewidth=3, label=f"Quá Khứ (Mồi {lookback} nến)")
    plt.plot(x_future, future_unscaled_real, 'g--', linewidth=3, label=f"Tương Lai (Thật {lookforward} nến)")
    
    for i in range(num_scenarios):
        with torch.no_grad():
            z_noise = torch.randn(1, LATENT_DIM_CVAE).to(device)
            future_fake_scaled, _ = decoder_model(real_past_gpu, z_noise)
            future_unscaled_fake = unscale_h1_close(future_fake_scaled.cpu().numpy())
            plt.plot(x_future, future_unscaled_fake, 'b-', alpha=0.3, label=f'Kịch bản {model_name} {i+1}' if i < 1 else None)
            
    plt.title(f"PHÒNG TRIỂN LÃM {model_name} (53 Món): {num_scenarios} Kịch Bản Tương Lai (LB={lookback})", fontsize=16)
    plt.ylabel("Giá H1_Close (USDT)", fontsize=12)
    plt.xlabel("Nến H1", fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True)
    
    save_path = os.path.join(CHART_SAVE_DIR, f"{file_name_suffix}_Scenarios_LB{lookback}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logging.info(f"Đã 'vẽ' và lưu ảnh {model_name} -> {save_path}")
    plt.close()

def plot_timegan_scenarios(g_model, r_model, lookback, hidden_dim, num_scenarios=10):
    """
    "Vẽ" 10 kịch bản "Quá Khứ" (tự bịa) của Thầy TimeGAN (V4 - 53 món)
    """
    logging.info(f"--- Đang 'vẽ' {num_scenarios} kịch bản TimeGAN V4 (LB={lookback}) ---")
    
    g_model.eval()
    r_model.eval()
    
    x_axis = np.arange(lookback)
    
    plt.figure(figsize=(20, 8))
    
    for i in range(num_scenarios):
        with torch.no_grad():
            z_noise = torch.randn(1, lookback, hidden_dim).to(device)
            h_fake_scaled = g_model(z_noise)
            x_fake_scaled = r_model(h_fake_scaled)
            x_unscaled_fake = unscale_h1_close(x_fake_scaled.cpu().numpy())
            plt.plot(x_axis, x_unscaled_fake, 'g-', alpha=0.4, label=f'Kịch bản TimeGAN {i+1}' if i < 1 else None)

    plt.title(f"PHÒNG TRIỂN LÃM TIMEGAN V4 (53 Món): {num_scenarios} Kịch Bản 'Tự Bịa' (LB={lookback})", fontsize=16)
    plt.ylabel("Giá H1_Close (USDT)", fontsize=12)
    plt.xlabel(f"{lookback} Nến H1 (Giả)", fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True)
    
    save_path = os.path.join(CHART_SAVE_DIR, f"TimeGAN_V4_Scenarios_LB{lookback}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logging.info(f"Đã 'vẽ' và lưu ảnh TimeGAN V4 -> {save_path}")
    plt.close()


# =========================================================================
# 💡 BƯỚC 4: "MỞ CỬA" PHÒNG TRIỂN LÃM (CẢ 3 NÃO)
# =========================================================================
if __name__ == "__main__":
    logging.info(f"=== BẮT ĐẦU 'MỞ CỬA' PHÒNG TRIỂN LÃM (CẢ 3 NÃO VẼ) (53 Món) ===")
    
    # 1. "NẠP THỨC ĂN" (53 Món) VÀ "SCALER V23"
    data_scaled = load_data_and_set_unscaler(num_features=NUM_FEATURES_GOC)
    
    if data_scaled is None:
        logging.error("Dừng 'Phòng Triển Lãm' vì không có 'thức ăn' hoặc 'scaler'.")
        sys.exit(1)
        
    # 2. "MÓC MỒI" (Lấy 50 nến "thật" gần cuối làm "mồi")
    # (Dùng chung "mồi" (sample_idx=-100) cho CVAE-LSTM và CVAE-Transformer)
    real_past_gpu, real_future_np = get_real_sample(data_scaled, LOOKBACK, sample_idx=-100)
    
    if real_past_gpu is None:
        logging.error("Dừng 'Phòng Triển Lãm' vì không 'móc mồi' được.")
        sys.exit(1)

    # 3. "TRIỂN LÃM" TRANH CỦA THẦY 1: CVAE-LSTM (V11)
    try:
        logging.info("--- Đang 'hồi sinh' NÃO 1: CVAE-LSTM (V11) (53 Món) ---")
        cvae_model_path = os.path.join(DIR_MODELS, CVAE_LSTM_MODEL_FILE.format(lb=LOOKBACK, lf=LOOKFORWARD_CVAE))
        
        cvae_decoder_lstm = CVAE_LSTM_Decoder(LOOKBACK, LOOKFORWARD_CVAE, NUM_FEATURES_GOC, LATENT_DIM_CVAE).to(device)
        cvae_decoder_lstm.load_state_dict(torch.load(cvae_model_path, map_location=device))
        
        logging.info(f"Đã 'hồi sinh' não CVAE-LSTM V11 từ: {cvae_model_path}")
        
        # "Vẽ" (Dùng hàm "Vẽ" chung)
        plot_cvae_scenarios(
            cvae_decoder_lstm, real_past_gpu, real_future_np, 
            LOOKBACK, LOOKFORWARD_CVAE, 
            num_scenarios=10, 
            model_name="CVAE-LSTM (V11)",
            file_name_suffix="CVAE_LSTM_V11"
        )
        
    except FileNotFoundError:
        logging.error(f"LỖI: Không tìm thấy 'Não CVAE V11' tại: {cvae_model_path}")
        logging.error("Đại ca đã chạy 'Lò' (train_cvae_V11.py) (file 'ăn' 53 món) [Bước 1] chưa?")
    except Exception as e:
        logging.error(f"LỖI 'HỒI SINH' NÃO CVAE V11: {e}")

    # 4. "TRIỂN LÃM" TRANH CỦA THẦY 2: TIMEGAN-GRU (V4)
    try:
        logging.info("--- Đang 'hồi sinh' NÃO 2: TIMEGAN-GRU (V4) (53 Món) ---")
        timegan_model_path = os.path.join(DIR_MODELS, TIMEGAN_GRU_MODEL_FILE.format(lb=LOOKBACK))
        
        timegan_G = TimeGAN_GRU_Generator(HIDDEN_DIM_TIMEGAN).to(device)
        timegan_R = TimeGAN_GRU_Recovery(NUM_FEATURES_GOC, HIDDEN_DIM_TIMEGAN).to(device)
        
        checkpoint = torch.load(timegan_model_path, map_location=device)
        timegan_G.load_state_dict(checkpoint['G_state_dict'])
        timegan_R.load_state_dict(checkpoint['R_state_dict'])
        
        logging.info(f"Đã 'hồi sinh' não TimeGAN V4 (G và R) từ: {timegan_model_path}")
        
        # "Vẽ"
        plot_timegan_scenarios(timegan_G, timegan_R, LOOKBACK, HIDDEN_DIM_TIMEGAN, num_scenarios=10)

    except FileNotFoundError:
        logging.error(f"LỖI: Không tìm thấy 'Não TimeGAN V4' tại: {timegan_model_path}")
        logging.error("Đại ca đã chạy 'Lò' (train_timegan_V4.py) (file 'ăn' 53 món) [Bước 2] chưa?")
    except Exception as e:
        logging.error(f"LỖI 'HỒI SINH' NÃO TIMEGAN V4: {e}")

    # 5. (MỚI) "TRIỂN LÃM" TRANH CỦA THẦY 3: CVAE-TRANSFORMER (TCVAE V1)
    try:
        logging.info("--- Đang 'hồi sinh' NÃO 3: CVAE-Transformer (V13) (53 Món) ---")
        cvae_trans_model_path = os.path.join(DIR_MODELS, CVAE_TRANS_MODEL_FILE.format(lb=LOOKBACK, lf=LOOKFORWARD_CVAE))
        
        cvae_decoder_trans = CVAE_Trans_Decoder(
            LOOKBACK, LOOKFORWARD_CVAE, NUM_FEATURES_GOC, 
            D_MODEL, N_HEAD, NUM_ENC_LAYERS, NUM_DEC_LAYERS, LATENT_DIM_CVAE
        ).to(device)
        
        cvae_decoder_trans.load_state_dict(torch.load(cvae_trans_model_path, map_location=device))
        
        logging.info(f"Đã 'hồi sinh' não CVAE-Transformer V13 từ: {cvae_trans_model_path}")
        
        # "Vẽ" (Dùng hàm "Vẽ" chung)
        plot_cvae_scenarios(
            cvae_decoder_trans, real_past_gpu, real_future_np, 
            LOOKBACK, LOOKFORWARD_CVAE, 
            num_scenarios=10, 
            model_name="CVAE-Transformer (V13)",
            file_name_suffix="CVAE_TRANS_V13"
        )
        
    except FileNotFoundError:
        logging.error(f"LỖI: Không tìm thấy 'Não TCVAE V13' tại: {cvae_trans_model_path}")
        logging.error("Đại ca đã chạy 'Lò' (04_train_transformer_cvae_V1...) (file 'ăn' 53 món) [Bước 3] chưa?")
    except Exception as e:
        logging.error(f"LỖI 'HỒI SINH' NÃO TCVAE V13: {e}")

    logging.info(f"\n{'='*70}\n === HOÀN TẤT! ĐÃ 'VẼ' XONG TRANH (CẢ 3 NÃO)! ===\n{'='*70}")
    logging.info(f"Đại ca vào thư mục '{CHART_SAVE_DIR}' để 'thưởng thức' 3 bức ảnh PNG nhé!")