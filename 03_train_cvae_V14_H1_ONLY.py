import pandas as pd
import numpy as np
import os
import logging
import joblib
from sklearn.preprocessing import MinMaxScaler
import warnings 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import sys 
import matplotlib.pyplot as plt 
import glob 

# --- (MÓN 1) "BỊT MIỆNG" TOÀN BỘ CẢNH BÁO "RÁC" ---
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
# --- (HẾT MÓN 1) ---

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    print("CẢNH BÁO: Thiếu 'TensorBoard'. Chế độ 'Soi Hàng' (1) sẽ không 'vẽ' chart.")
    print("-> Chạy: pip install tensorboard")
    SummaryWriter = None 

psutil = None 
pynvml = None
GPU_HANDLE = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Hằng số "CÔNG XƯỞNG" V10.2 / V11 ---
MASTER_FILE_PATH = os.path.join('02_Master_Data', 'btcusdt_master_data.parquet') # (Ăn 53 món "gốc")
LOOKFORWARD = 20 
LATENT_DIM = 32
BETA_VALUE = 0.01 
BETA_ANNEAL_EPOCHS = 50 
GRAD_CLIP_VALUE = 1.0    
WEIGHTED_LOSS_MULTIPLIER = 10.0 

DEFAULT_SOI_EPOCHS = 150 # (V11 SẼ DÙNG SỐ NÀY LÀM EPOCH CỐ ĐỊNH)
DEFAULT_VALIDATION_SPLIT = 0.2 
DEFAULT_TENSORBOARD_DIR = "runs_v11_100_percent" 
CHART_SAVE_DIR = "04_Charts_V10_Final" 

DIR_MODELS = "03_Models"
DIR_PROCESSED = "01_Processed_Data"
# (SCALER "CHÍ MẠNG" MÀ TIMEGAN V4 "CẦN")
SCALER_FILENAME = os.path.join(DIR_PROCESSED, 'cvae_scaler_V23.gz') 

# (Biến "toàn cục")
CVAE_CLOSE_IDX_FOR_PLOT = -1
SCALER_MIN_CLOSE = 0.0
SCALER_SCALE_CLOSE = 1.0

# --- (Toàn bộ 8 hàm "phụ" V6.1: Giữ nguyên) ---
def worker_init_fn(worker_id):
    warnings.filterwarnings("ignore", category=FutureWarning)
    logging.getLogger().setLevel(logging.ERROR) 
class HardwareMonitor:
    def __init__(self, device_type):
        self.device_type = device_type
        self.is_gpu = (device_type == "cuda" and pynvml is not None)
    def get_stats(self):
        if self.is_gpu:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(GPU_HANDLE)
                gpu_load = f"{util.gpu}%"; mem = pynvml.nvmlDeviceGetMemoryInfo(GPU_HANDLE)
                vram_used_gb = mem.used / (1024**3); vram_total_gb = mem.total / (1024**3)
                vram_load = f"{vram_used_gb:.1f}/{vram_total_gb:.1f}GB"
                return {"GPU": gpu_load, "VRAM": vram_load}
            except Exception: self.is_gpu = False ; return self.get_stats() 
        else:
            if psutil: return {"CPU": f"{psutil.cpu_percent()}%"}
            else: return {"CPU": "N/A"}
            
# (SỬA V11: "ĐÚC" SCALER V23 TỪ 53 MÓN "GỐC")
def load_and_process_data_V23():
    logging.info(f"Đang tải file master V23 (53 món): {MASTER_FILE_PATH}...")
    try:
        df = pd.read_parquet(MASTER_FILE_PATH)
    except FileNotFoundError:
        logging.error(f"LỖI: Không tìm thấy file {MASTER_FILE_PATH}.")
        logging.error("Đại ca đã chạy 'Lò' (data_service.py) [Chế độ 2] chưa?")
        return None, None, None
        
    df.interpolate(method='time', inplace=True)
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    df.fillna(0, inplace=True) # (Vá V5: "Trám" 0 (nếu ffill/bfill vẫn sót))
    
    features_df = df
    feature_names = features_df.columns.tolist()
    num_features = len(feature_names)
    
    # (ĐÚC SCALER V23 "CHÍ MẠNG" Ở ĐÂY)
    logging.info(f"Đang 'luyện' (fit) Scaler MỚI (V23 - {num_features} món 'gốc')...")
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(features_df)
    
    # (GÁN feature_names VÀO SCALER ĐỂ "MÓC" H1_Close)
    try:
        scaler.feature_names_in_ = feature_names
    except Exception:
        pass # (Bỏ qua nếu scaler cũ)
        
    joblib.dump(scaler, SCALER_FILENAME)
    logging.info(f"Đã lưu Scaler MỚI (V23 - Dùng Chung CVAE/TimeGAN) -> {SCALER_FILENAME}")
    
    return data_scaled, num_features, feature_names # (Trả về feature_names)

def create_windows(data_scaled, lookback, lookforward):
    logging.info(f"Đang 'cắt' cửa sổ CVAE (Lookback={lookback})...")
    X_past, Y_future = [], []
    num_samples = len(data_scaled) - lookback - lookforward + 1
    if num_samples <= 0:
        logging.error(f"LỖI (Lookback={lookback}): Dữ liệu quá ít.")
        return None, None
    for i in tqdm(range(num_samples), desc=f"Cắt cửa sổ CVAE LB={lookback}"):
        X_past.append(data_scaled[i : i + lookback])
        Y_future.append(data_scaled[i + lookback : i + lookback + lookforward])
    X_past = np.array(X_past); Y_future = np.array(Y_future)
    logging.info(f"Shape X_past: {X_past.shape} | Shape Y_future: {Y_future.shape}")
    return X_past, Y_future

class Encoder(nn.Module):
    def __init__(self, lookback, lookforward, num_features, latent_dim):
        super(Encoder, self).__init__()
        self.lstm_past_1 = nn.LSTM(num_features, 64, batch_first=True, num_layers=1)
        self.lstm_past_2 = nn.LSTM(64, 32, batch_first=True, num_layers=1) 
        self.lstm_future_1 = nn.LSTM(num_features, 128, batch_first=True, num_layers=1)
        self.lstm_future_2 = nn.LSTM(128, 64, batch_first=True, num_layers=1)
        self.dense_combine = nn.Linear(32 + 64, 64) 
        self.z_mean = nn.Linear(64, latent_dim)
        self.z_log_var = nn.Linear(64, latent_dim)
        self.relu = nn.ReLU()
    def forward(self, condition_input, future_input): 
        h_past, _ = self.lstm_past_1(condition_input)
        _, (last_hidden_past, _) = self.lstm_past_2(h_past)
        h_past_features = self.relu(last_hidden_past.squeeze(0)) 
        h_future, _ = self.lstm_future_1(future_input)
        _, (last_hidden_future, _) = self.lstm_future_2(h_future)
        h_future_features = self.relu(last_hidden_future.squeeze(0)) 
        combined = torch.cat([h_past_features, h_future_features], dim=1)
        h_combined = self.relu(self.dense_combine(combined))
        mean = self.z_mean(h_combined)
        log_var = self.z_log_var(h_combined)
        return mean, log_var

def sampling(args):
    z_mean, z_log_var = args
    batch = z_mean.shape[0]; dim = z_mean.shape[1]
    epsilon = torch.randn(size=(batch, dim)).to(device)
    return z_mean + torch.exp(0.5 * z_log_var) * epsilon

def augment_batch(batch_data, jitter_strength=0.01, scale_strength=0.1):
    B, L, D = batch_data.shape
    jitter = torch.randn_like(batch_data) * jitter_strength
    scaling_factor = 1.0 + (torch.rand((B, 1, 1), device=batch_data.device) - 0.5) * 2 * scale_strength
    augmented_batch = (batch_data + jitter) * scaling_factor
    return torch.clamp(augmented_batch, 0.0, 1.0)

class Decoder(nn.Module):
    def __init__(self, lookback, lookforward, num_features, latent_dim, num_heads=4):
        super(Decoder, self).__init__()
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
# --- (Hết 8 hàm phụ) ---


# --- (HÀM "VẼ" (PLOT) V10.2) ---
# (Các hàm "Vẽ" này sẽ không được gọi ở V11 (100% train), nhưng cứ để lại)
def plot_cvae_samples(X_past_sample, Y_future_sample, encoder, decoder, writer, epoch, lookback, num_samples=5):
    """ "Vẽ" Kịch Bản (Samples) """
    try:
        encoder.eval(); decoder.eval()
        X_past_gpu = X_past_sample.unsqueeze(0).to(device) 
        Y_future_gpu = Y_future_sample.unsqueeze(0).to(device) 
        
        if CVAE_CLOSE_IDX_FOR_PLOT == -1:
             logging.warning("Không 'vẽ' PNG được vì CVAE_CLOSE_IDX_FOR_PLOT chưa được set.")
             return
             
        y_true_scaled = Y_future_gpu[0, :, CVAE_CLOSE_IDX_FOR_PLOT].cpu().numpy()
        y_true_unscaled = (y_true_scaled * SCALER_SCALE_CLOSE) + SCALER_MIN_CLOSE
        
        fig, ax = plt.subplots(figsize=(15, 7))
        for _ in range(num_samples):
            with torch.no_grad():
                z_mean, z_log_var = encoder(X_past_gpu, Y_future_gpu)
                z = sampling((z_mean, z_log_var))
                reconstruction, _ = decoder(X_past_gpu, z)
            y_pred_scaled = reconstruction[0, :, CVAE_CLOSE_IDX_FOR_PLOT].cpu().numpy()
            y_pred_unscaled = (y_pred_scaled * SCALER_SCALE_CLOSE) + SCALER_MIN_CLOSE
            ax.plot(y_pred_unscaled, color='blue', alpha=0.3)
            
        ax.plot(y_true_unscaled, color='red', linewidth=2, label='Hàng Thật (Y_future)')
        ax.set_title(f"Phòng Triển Lãm (V10.2 - LB={lookback}): 5 Kịch Bản H1_Close (Epoch {epoch} - Xịn Nhất)")
        ax.set_xlabel(f"{LOOKFORWARD} nến H1 tương lai"); ax.set_ylabel("Giá H1_Close (USDT)"); ax.legend()
        if writer: writer.add_figure(f'Sample_Plots/Kịch_Bản_Vẽ_LB{lookback}', fig, global_step=epoch)
        chart_filename = f"cvae_samples_lb{lookback}_epoch{epoch}_BEST.png"
        chart_save_path = os.path.join(CHART_SAVE_DIR, chart_filename)
        try: fig.savefig(chart_save_path, dpi=150, bbox_inches='tight') 
        except Exception as e: logging.warning(f"Lỗi lưu chart Samples PNG: {e}")
        plt.close(fig) 
    except Exception as e: logging.warning(f"Lỗi 'Vẽ' Kịch Bản (plot_cvae_samples): {e}")

def plot_attention_heatmap(attn_weights_plot_numpy, writer, epoch, lookback):
    """ "Vẽ" Mắt Thần (Attention) """
    try:
        avg_attn_heatmap = np.mean(attn_weights_plot_numpy, axis=0) # (Shape chuẩn (1, 50))
        fig, ax = plt.subplots(figsize=(15, 2))
        im = ax.imshow(avg_attn_heatmap, cmap='viridis', aspect='auto')
        ax.set_title(f"Mắt Thần (V10.2 - LB={lookback}) (Epoch {epoch} - Xịn Nhất)")
        ax.set_xlabel("Nến Quá Khứ (t-lookback ... t-1)"); ax.set_yticks([]); plt.colorbar(im, ax=ax)
        if writer: writer.add_figure(f'Attention/Heatmap_LB{lookback}', fig, global_step=epoch)
        chart_filename = f"cvae_attention_lb{lookback}_epoch{epoch}_BEST.png"
        chart_save_path = os.path.join(CHART_SAVE_DIR, chart_filename)
        try: fig.savefig(chart_save_path, dpi=150, bbox_inches='tight')
        except Exception as e: logging.warning(f"Lỗi lưu chart Attention PNG: {e}")
        plt.close(fig)
    except Exception as e:
        logging.warning(f"Lỗi 'Vẽ' Mắt Thần (plot_attention_heatmap): {e}")
        logging.warning(f"Shape 'Mắt Thần' (attn_weights) gây lỗi: {attn_weights_plot_numpy.shape}")


# --- (HÀM "LÕI" V10.2 / V11 - "God Function") ---
def train_single_cvae_pytorch(X_past, Y_future, lookback, lookforward, num_features, weights_tensor, mode_config):
    
    val_split = mode_config['val_split']
    total_epochs = mode_config['epochs'] 
    log_dir = mode_config['log_dir']
    final_model_name = mode_config['final_name'].format(lb=lookback, lf=lookforward) 
    resume_file_name = mode_config.get('resume_name') 
    use_patience = mode_config['use_patience']
    
    writer = None
    if log_dir and SummaryWriter: 
        writer = SummaryWriter(log_dir=os.path.join(log_dir, f'cvae_v11_lb_{lookback}'))
    
    pin_mem = (device.type == 'cuda')
    X_past_t = torch.tensor(X_past, dtype=torch.float32)
    Y_future_t = torch.tensor(Y_future, dtype=torch.float32)
    
    val_size = int(len(X_past_t) * val_split) # (V11: val_split=0 -> val_size=0)
    train_size = len(X_past_t) - val_size
    
    X_past_train, X_past_val = X_past_t[:train_size], X_past_t[train_size:]
    Y_future_train, Y_future_val = Y_future_t[:train_size], Y_future_t[train_size:]
    
    logging.info(f"Tách Train/Val: {train_size} (train) / {val_size} (val)")

    train_dataset = TensorDataset(X_past_train, Y_future_train)
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True,
                            pin_memory=pin_mem, num_workers=80,
                            worker_init_fn=worker_init_fn)
    
    val_loader = None
    if val_size > 0: 
        val_loader = DataLoader(TensorDataset(X_past_val, Y_future_val), 
                                batch_size=512, shuffle=False,
                                pin_memory=pin_mem, num_workers=20)
    
    encoder = Encoder(lookback, lookforward, num_features, LATENT_DIM).to(device)
    decoder = Decoder(lookback, lookforward, num_features, LATENT_DIM).to(device)
    monitor = HardwareMonitor(device.type) 
    
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params, lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)
    
    patience = 15; best_loss = float('inf'); patience_counter = 0; start_epoch = 0
    best_X_plot = None; best_Y_plot = None; best_attn_plot = None; best_epoch = 0
    
    if resume_file_name and os.path.exists(resume_file_name):
        # (V11: resume_file_name = None, sẽ không chạy vào đây)
        pass 

    try:
        for epoch in range(start_epoch, total_epochs):
            
            # --- (A) "LUYỆN" (TRAIN) ---
            encoder.train(); decoder.train()
            total_train_loss = 0
            current_beta = BETA_VALUE * min(1.0, (epoch / BETA_ANNEAL_EPOCHS))
            pbar_desc = f"LB={lookback:03d} | E {epoch+1}/{total_epochs} [{mode_config['desc']}]"
            pbar_train = tqdm(train_loader, desc=pbar_desc, leave=False)
            
            for batch_X_past, batch_Y_future in pbar_train:
                batch_X_past_gpu = batch_X_past.to(device)
                batch_Y_future_gpu = batch_Y_future.to(device)
                batch_X_past_aug = augment_batch(batch_X_past_gpu)
                batch_Y_future_aug = augment_batch(batch_Y_future_gpu)
                z_mean, z_log_var = encoder(batch_X_past_aug, batch_Y_future_aug)
                z = sampling((z_mean, z_log_var))
                reconstruction, _ = decoder(batch_X_past_aug, z) 
                recon_loss_per_feature = torch.mean(
                    torch.square(batch_Y_future_aug - reconstruction) * weights_tensor, dim=2
                )
                recon_loss_per_sample = torch.sum(recon_loss_per_feature, dim=1)
                recon_loss = torch.mean(recon_loss_per_sample)
                kl_loss_per_sample = -0.5 * (1 + z_log_var - torch.square(z_mean) - torch.exp(z_log_var))
                kl_loss = torch.mean(torch.sum(kl_loss_per_sample, dim=1))
                total_loss = recon_loss + current_beta * kl_loss 
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=GRAD_CLIP_VALUE)
                optimizer.step()
                total_train_loss += total_loss.item()
                stats = monitor.get_stats()
                pbar_train.set_postfix(loss=total_loss.item(), **stats)
            
            avg_train_loss = total_train_loss / len(train_loader)
            
            # --- (B) "SOI" (VALIDATION) ---
            avg_val_loss = -1.0; avg_val_recon_loss = -1.0
            
            # (V11: val_split = 0.0, sẽ không chạy vào đây)
            if val_split > 0.0 and val_loader:
                pass # (Logic "Soi" nằm ở đây)
            
            # --- (C) LOGGING & CHECKPOINT ---
            # (V11: val_split = 0.0, sẽ chạy vào else)
            if val_split > 0.0:
                pass # (Log "Soi" 80/20)
            else:
                logging.info(f"[LB={lookback:03d}] E{epoch+1:03d} | Beta: {current_beta:.4f} | Train Loss (100%): {avg_train_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
            
            scheduler.step()
            
            if writer:
                writer.add_scalar(f'Loss/Train_LB{lookback}', avg_train_loss, epoch)
                writer.add_scalar(f'Loss/Validation_LB{lookback}', avg_val_loss, epoch)
                writer.add_scalar(f'Loss/Val_Recon_Thuần_LB{lookback}', avg_val_recon_loss, epoch)
                writer.add_scalar(f'Params/LearningRate_LB{lookback}', optimizer.param_groups[0]['lr'], epoch)
                writer.add_scalar(f'Params/Beta_LB{lookback}', current_beta, epoch)

            # (V11: use_patience = False, sẽ không chạy vào đây)
            if use_patience:
                pass
    
    except KeyboardInterrupt:
        logging.warning(f"\n[LB={lookback}] Đã bắt được (Ctrl+C)! Đang 'lưu khẩn cấp'...")
        torch.save(decoder.state_dict(), final_model_name + ".interrupt.pth")
        if pynvml: pynvml.nvmlShutdown() 
        sys.exit(0) 
    
    # --- (D) "TRẢ HÀNG" (VẼ ẢNH) ---
    
    # (V11: use_patience = False, sẽ chạy vào đây)
    if not use_patience:
        torch.save(decoder.state_dict(), final_model_name)
        logging.info(f"-> 'Não Xịn' 100% (Epoch {total_epochs}) đã lưu -> {final_model_name}")
    else:
        # (V11: Sẽ không chạy vào đây)
        pass 
        
    logging.info(f"=== HOÀN TẤT 'DÂY CHUYỀN' (Lookback={lookback}) ===")
    if writer: writer.close() 

# --- (HÀM MAIN "CÔNG XƯỞNG" V11 - 1 CHẾ ĐỘ 100%) ---
if __name__ == "__main__":

    print("\n" + "="*70)
    print("      LÒ ĐÚC NÃO CVAE V11 (1 CHẾ ĐỘ 100%)")
    print("="*70)
    
    logging.info("--- CHẾ ĐỘ DUY NHẤT: [LUYỆN 100% DATA] ĐANG KHỞI ĐỘNG ---")
    
    mode_config = {
        'desc': "LUYỆN 100%",
        'epochs': DEFAULT_SOI_EPOCHS, 
        'val_split': 0.1,
        'log_dir': DEFAULT_TENSORBOARD_DIR, 
        'final_name': os.path.join(DIR_MODELS, "cvae_decoder_V11_100PCT_{lb}_{lf}.pth"), 
        'resume_name': None, 
        'use_patience': False,
        'log_file': "log_train_cvae_V11_100PCT.log"
    }

    # --- "LẮP" ĐỒ NGHỀ ---
    try: import psutil
    except ImportError: psutil = None
        
    log_filename = mode_config['log_file']
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - [TrainCVAE_V11] - %(message)s',
                        handlers=[
                            logging.FileHandler(log_filename, mode='w', encoding='utf-8'), # (Vá lỗi 'utf-f')
                            logging.StreamHandler(sys.stdout) 
                        ],
                        force=True) 
                        
    logging.info(f"Log sẽ được lưu vào file: {log_filename}")
    logging.info(f"Đang sử dụng thiết bị: {device}")
    
    try:
        import pynvml 
        pynvml.nvmlInit()
        GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception:
        try:
            import nvidia_ml as pynvml 
            pynvml.nvmlInit()
            GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception: pynvml = None 
    
    os.makedirs(DIR_MODELS, exist_ok=True)
    os.makedirs(DIR_PROCESSED, exist_ok=True)
    if mode_config['log_dir']: os.makedirs(mode_config['log_dir'], exist_ok=True) 
    
    logging.info(f"=== 'LÒ' V11 CHẠY CHẾ ĐỘ: {mode_config['desc']} ===")
            
    # 1. TẢI VÀ "ĐÚC" SCALER MỚI (V23 - 53 món)
    data_scaled, num_features, feature_names = load_and_process_data_V23()
    
    if data_scaled is None:
        logging.error("Dừng 'CÔNG XƯỞNG' (CVAE V11) vì không có 'thức ăn'.")
        if pynvml: pynvml.nvmlShutdown() 
        sys.exit(1)
        
    # 2. "CHẾ" LOSS "QUYỀN TRỌNG" (Weighted Loss)
    logging.info(f"Đang 'chế' Weighted Loss (x{WEIGHTED_LOSS_MULTIPLIER} cho OHLC)...")
    try:
        weights_tensor = torch.ones(num_features).to(device)
        
        vip_cols = ['H1_Close', 'H1_High', 'H1_Low', 'H1_Open']
        found_count = 0
        for col in vip_cols:
            try:
                # (Dùng feature_names (V11) trả về từ hàm load_data)
                idx = feature_names.index(col)
                weights_tensor[idx] = WEIGHTED_LOSS_MULTIPLIER
                
                if col == 'H1_Close':
                    CVAE_CLOSE_IDX_FOR_PLOT = idx
                    # "Móc" thông số unscale từ scaler V23
                    scaler_v23 = joblib.load(SCALER_FILENAME)
                    SCALER_MIN_CLOSE = scaler_v23.min_[idx]
                    SCALER_SCALE_CLOSE = scaler_v23.scale_[idx]
                    
                found_count += 1
            except (ValueError, IndexError):
                logging.warning(f"Không tìm thấy cột '{col}' trong 'thức ăn 53 món' để 'độ' Weighted Loss.")
        
        # (V11 "Đúc Xịn" 100% (không "vẽ" PNG) nên không cần "chết" (exit) nếu CVAE_CLOSE_IDX_FOR_PLOT == -1)
            
        logging.info(f"Đã 'độ' Weighted Loss (x{WEIGHTED_LOSS_MULTIPLIER}) cho {found_count} cột 'VIP'.")
    except Exception as e:
        logging.error(f"LỖI 'độ' Weighted Loss. Dùng tạm loss 'cào bằng'. Lỗi: {e}")
        weights_tensor = torch.ones(num_features).to(device)
    
    # 3. LẶP "ĐÚC" 2 NÃO (50, 168)
    ALL_LOOKBACKS = [50, 168]
    
    for lb in ALL_LOOKBACKS:
        logging.info(f"\n{'='*70}\n === BẮT ĐẦU 'DÂY CHUYỀN' (V11 53 món Lookback={lb}) ===\n{'='*70}")
        
        X_past, Y_future = create_windows(data_scaled, lb, LOOKFORWARD)
        
        if X_past is None:
            logging.warning(f"Bỏ qua Lookback={lb} do không đủ dữ liệu 'cắt'.")
            continue
            
        train_single_cvae_pytorch(
            X_past, Y_future, 
            lb, LOOKFORWARD, num_features, 
            weights_tensor,
            mode_config 
        )
        
        torch.cuda.empty_cache()
    
    logging.info(f"\n{'='*70}\n === HOÀN TẤT 'LÒ V11' - ĐÃ CHẠY XONG {len(ALL_LOOKBACKS)} NÃO! ===\n{'='*70}")
    
    logging.info(f"Đã 'đúc' não 100% data (lưu epoch cuối) vào thư mục: {DIR_MODELS}")
    logging.info(f"Đã 'đúc' SCALER 'CHÍ MẠNG' (V23) vào thư mục: {DIR_PROCESSED}")
    logging.info(f"Đại ca 'soi' (Train Loss) bằng: tensorboard --logdir={mode_config['log_dir']}")
    
    if pynvml:
        pynvml.nvmlShutdown()