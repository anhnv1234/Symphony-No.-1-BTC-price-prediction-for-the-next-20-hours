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
import math # (Cần cho "Não" Transformer)

# --- (MÓN 1) "BỊT MIỆNG" TOÀN BỘ CẢNH BÁO "RÁC" ---
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
# --- (HẾT MÓN 1) ---

# (Kiểm tra TensorBoard)
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None 

psutil = None 
pynvml = None
GPU_HANDLE = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Hằng số "CÔNG XƯỞNG" V13 (TCVAE - Transformer CVAE) ---
MASTER_FILE_PATH = os.path.join('02_Master_Data', 'btcusdt_master_data.parquet') 
SCALER_FILENAME = os.path.join('01_Processed_Data', 'cvae_scaler_V23.gz')
LOOKFORWARD = 20 
LATENT_DIM = 32
BETA_VALUE = 0.01 
BETA_ANNEAL_EPOCHS = 50 
GRAD_CLIP_VALUE = 1.0    
WEIGHTED_LOSS_MULTIPLIER = 10.0 
DEFAULT_SOI_EPOCHS = 150 
DEFAULT_TENSORBOARD_DIR = "runs_v13_transformer_cvae" 
DIR_MODELS = "03_Models"
DIR_PROCESSED = "01_Processed_Data"
# --- (SỬA V13: CẤU HÌNH "NÃO" TRANSFORMER) ---
NUM_FEATURES_GOC = 53 
D_MODEL = 64      
N_HEAD = 4        
NUM_ENC_LAYERS = 2 
NUM_DEC_LAYERS = 2 
# (Biến "toàn cục" để "vẽ")
CVAE_CLOSE_IDX_FOR_PLOT = -1
SCALER_MIN_CLOSE = 0.0
SCALER_SCALE_CLOSE = 1.0

# --- (Hàm phụ "worker_init_fn", "HardwareMonitor": Giữ nguyên) ---
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

# --- (Hàm "load_data_and_scaler_V23": Giữ nguyên) ---
def load_data_and_scaler_V23():
    logging.info(f"Đang tải file master V23 (53 món): {MASTER_FILE_PATH}...")
    try:
        df = pd.read_parquet(MASTER_FILE_PATH)
    except FileNotFoundError:
        logging.error(f"LỖI: Không tìm thấy file {MASTER_FILE_PATH}.")
        return None, None, None
        
    logging.info(f"Đang tải 'Bộ Chuẩn Hóa' (V23 - 53 món) (dùng chung CVAE): {SCALER_FILENAME}...")
    try:
        scaler = joblib.load(SCALER_FILENAME)
    except FileNotFoundError:
        logging.error(f"LỖI: Không tìm thấy file {SCALER_FILENAME}.")
        logging.error("Đại ca đã chạy 'Lò' (train_cvae_V11.py) (file 'ăn' 53 món) [Bước 1] chưa?")
        return None, None, None
    
    df.interpolate(method='time', inplace=True)
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    df.fillna(0, inplace=True) 

    features_df = df
    feature_names = features_df.columns.tolist()
    num_features = len(feature_names)
    
    if num_features != NUM_FEATURES_GOC:
        logging.error(f"LỖI 'KHỚP' NÃO: 'Lò' V13 'code' 53 món, nhưng file 'gốc' 'có' {num_features} món.")
        return None, None, None

    logging.info("Đang chuẩn hóa 'thức ăn' (dùng Scaler Lò V11)...")
    data_scaled = scaler.transform(features_df)
    
    return data_scaled, num_features, feature_names

# --- (Hàm "create_windows": Giữ nguyên) ---
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

# --- (Hàm "sampling", "augment_batch": Giữ nguyên) ---
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


# =========================================================================
# 💡 BƯỚC 1: "ĐỘ" NÃO TRANSFORMER (TCVAE V1)
# =========================================================================

class PositionalEncoding(nn.Module):
    """ "Nhồi" vị trí (thứ tự 1, 2, 3...) vào "não" """
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

# --- (SỬA V13: "ĐỘ" NÃO ENCODER DÙNG TRANSFORMER) ---
class Encoder(nn.Module):
    def __init__(self, lookback, lookforward, num_features, d_model, n_head, num_enc_layers, latent_dim):
        super(Encoder, self).__init__()
        
        self.embed_past = nn.Linear(num_features, d_model)
        self.embed_future = nn.Linear(num_features, d_model)
        self.pos_encoder_past = PositionalEncoding(d_model, max_len=lookback)
        self.pos_encoder_future = PositionalEncoding(d_model, max_len=lookforward)
        encoder_layer_past = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, batch_first=True)
        self.transformer_encoder_past = nn.TransformerEncoder(encoder_layer_past, num_layers=num_enc_layers)
        encoder_layer_future = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, batch_first=True)
        self.transformer_encoder_future = nn.TransformerEncoder(encoder_layer_future, num_layers=num_enc_layers)
        self.flatten = nn.Flatten()
        self.dense_combine = nn.Linear(d_model * (lookback + lookforward), 128)
        self.z_mean = nn.Linear(128, latent_dim)
        self.z_log_var = nn.Linear(128, latent_dim)
        self.relu = nn.ReLU()

    def forward(self, condition_input, future_input): 
        x_past = self.embed_past(condition_input) 
        x_future = self.embed_future(future_input) 
        x_past = self.pos_encoder_past(x_past)
        x_future = self.pos_encoder_future(x_future)
        past_features = self.transformer_encoder_past(x_past) 
        future_features = self.transformer_encoder_future(x_future) 
        combined = torch.cat([past_features, future_features], dim=1) 
        combined_flat = self.flatten(combined) 
        h_combined = self.relu(self.dense_combine(combined_flat)) 
        mean = self.z_mean(h_combined)
        log_var = self.z_log_var(h_combined)
        return mean, log_var

# --- (SỬA V13: "ĐỘ" NÃO DECODER DÙNG TRANSFORMER) ---
class Decoder(nn.Module):
    # === ("VÁ" 1/2) ===
    def __init__(self, lookback, lookforward, num_features, d_model, n_head, num_enc_layers, num_dec_layers, latent_dim):
        super(Decoder, self).__init__()
        self.lookforward = lookforward
        self.d_model = d_model
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
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
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
# =========================================================================

# --- (HÀM "VẼ" (PLOT) V10.2) ---
def plot_cvae_samples(X_past_sample, Y_future_sample, encoder, decoder, writer, epoch, lookback, num_scenarios=5):
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
        for _ in range(num_scenarios):
            with torch.no_grad():
                z_mean, z_log_var = encoder(X_past_gpu, Y_future_gpu)
                z = sampling((z_mean, z_log_var))
                reconstruction, _ = decoder(X_past_gpu, z)
            y_pred_scaled = reconstruction[0, :, CVAE_CLOSE_IDX_FOR_PLOT].cpu().numpy()
            y_pred_unscaled = (y_pred_scaled * SCALER_SCALE_CLOSE) + SCALER_MIN_CLOSE
            ax.plot(y_pred_unscaled, color='blue', alpha=0.3)
        ax.plot(y_true_unscaled, color='red', linewidth=2, label='Hàng Thật (Y_future)')
        ax.set_title(f"Phòng Triển Lãm (V13 TCVAE 'Siêu Cấp' - LB={lookback}): 5 Kịch Bản H1_Close (Epoch {epoch} - Xịn Nhất)")
        ax.set_xlabel(f"{LOOKFORWARD} nến H1 tương lai"); ax.set_ylabel("Giá H1_Close (USDT)"); ax.legend()
        if writer: writer.add_figure(f'Sample_Plots/Kịch_Bản_Vẽ_LB{lookback}', fig, global_step=epoch)
        chart_save_path = os.path.join(DIR_MODELS, f"cvae_samples_V13_TCVAE_lb{lookback}_epoch{epoch}_BEST.png")
        try: fig.savefig(chart_save_path, dpi=150, bbox_inches='tight') 
        except Exception as e: logging.warning(f"Lỗi lưu chart Samples PNG: {e}")
        plt.close(fig) 
    except Exception as e: logging.warning(f"Lỗi 'Vẽ' Kịch Bản (plot_cvae_samples): {e}")

def plot_attention_heatmap(attn_weights_plot_numpy, writer, epoch, lookback):
    pass # (V13 không "vẽ" Attn)

# --- (HÀM "LÕI" V13 - "God Function") ---
def train_single_cvae_pytorch(X_past, Y_future, lookback, lookforward, num_features, weights_tensor, mode_config):
    
    val_split = mode_config['val_split']
    total_epochs = mode_config['epochs'] 
    log_dir = mode_config['log_dir']
    final_model_name = mode_config['final_name'].format(lb=lookback, lf=lookforward) 
    
    # === ("VÁ" LỖI 1/2) ===
    resume_file_name = None
    if mode_config.get('resume_name'):
        resume_file_name = mode_config.get('resume_name').format(lb=lookback)
    # =====================
    
    use_patience = mode_config['use_patience']
    
    writer = None
    if log_dir and SummaryWriter: 
        writer = SummaryWriter(log_dir=os.path.join(log_dir, f'cvae_v13_lb_{lookback}'))
    
    pin_mem = (device.type == 'cuda')
    X_past_t = torch.tensor(X_past, dtype=torch.float32)
    Y_future_t = torch.tensor(Y_future, dtype=torch.float32)
    
    val_size = int(len(X_past_t) * val_split) 
    train_size = len(X_past_t) - val_size
    
    X_past_train, X_past_val = X_past_t[:train_size], X_past_t[train_size:]
    Y_future_train, Y_future_val = Y_future_t[:train_size], Y_future_t[train_size:]
    
    logging.info(f"Tách Train/Val (V13): {train_size} (train) / {val_size} (val)")

    # === ("VÁ" LỖI 3/3 - "BÓP MẺ" (OOM)) ===
    # (Nếu "não" "to" (LB=168), "ăn" "mẻ" "nhỏ" (32) để "chống" OOM)
    if lookback > 100:
        current_batch_size = 32
        logging.warning(f"Lookback 'To' ({lookback}). 'Bóp' Batch Size xuống {current_batch_size} để 'chống' OOM.")
    else:
        current_batch_size = 64
        logging.info(f"Lookback 'Nhỏ' ({lookback}). Dùng Batch Size {current_batch_size}.")
    # =======================================

    train_dataset = TensorDataset(X_past_train, Y_future_train)
    train_loader = DataLoader(train_dataset, batch_size=current_batch_size, shuffle=True, # (SỬA V13: Dùng "mẻ" "động")
                            pin_memory=pin_mem, num_workers=5,
                            worker_init_fn=worker_init_fn)
    
    val_loader = None
    if val_size > 0: 
        val_loader = DataLoader(TensorDataset(X_past_val, Y_future_val), 
                                batch_size=current_batch_size * 2, # (Dùng "mẻ" "động" x2)
                                shuffle=False,
                                pin_memory=pin_mem, num_workers=2)
    
    # === (SỬA V13: "Dựng" "Não" TRANSFORMER) ===
    encoder = Encoder(lookback, lookforward, num_features, D_MODEL, N_HEAD, NUM_ENC_LAYERS, LATENT_DIM).to(device)
    
    # === ("VÁ" 1/2) ===
    decoder = Decoder(lookback, lookforward, num_features, D_MODEL, N_HEAD, NUM_ENC_LAYERS, NUM_DEC_LAYERS, LATENT_DIM).to(device)
    # ========================================
    
    monitor = HardwareMonitor(device.type) 
    
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params, lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)
    
    # === ("VÁ" LỖI 2/2) ===
    patience = 15; # <-- "Khai báo" (Define) `patience` (chữ thường)
    # =====================
    best_loss = float('inf'); patience_counter = 0; start_epoch = 0
    best_X_plot = None; best_Y_plot = None; best_attn_plot = None; best_epoch = 0
    
    if resume_file_name and os.path.exists(resume_file_name):
        logging.info(f"Phát hiện Checkpoint 'Luyện Tiếp'! Đang tải từ: {resume_file_name}")
        try:
            checkpoint = torch.load(resume_file_name, map_location=device)
            encoder.load_state_dict(checkpoint['encoder_state_dict']) 
            decoder.load_state_dict(checkpoint['decoder_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                 scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1 
            best_loss = checkpoint['best_loss']
            patience_counter = checkpoint['patience_counter']
            logging.info(f"Tải Checkpoint thành công! Sẽ 'luyện tiếp' từ Epoch {start_epoch}.")
        except Exception as e:
            logging.error(f"Lỗi tải Checkpoint: {e}. 'Luyện' (Train) lại từ đầu (Epoch 0).")
            start_epoch = 0

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
            
            if val_split > 0.0 and val_loader:
                encoder.eval(); decoder.eval()
                total_val_loss = 0; total_val_recon_loss = 0 
                X_val_sample_plot_temp = None; Y_val_sample_plot_temp = None; attn_weights_plot_temp = None 
                
                with torch.no_grad():
                    for i_val, (batch_X_past_val, batch_Y_future_val) in enumerate(val_loader):
                        batch_X_past_val = batch_X_past_val.to(device)
                        batch_Y_future_val = batch_Y_future_val.to(device)
                        z_mean_val, z_log_var_val = encoder(batch_X_past_val, batch_Y_future_val)
                        z_val = sampling((z_mean_val, z_log_var_val))
                        reconstruction_val, attn_weights_val = decoder(batch_X_past_val, z_val)
                        recon_loss_val_feat = torch.mean(
                            torch.square(batch_Y_future_val - reconstruction_val) * weights_tensor, dim=2
                        )
                        recon_loss_val_samp = torch.sum(recon_loss_val_feat, dim=1)
                        recon_loss_val = torch.mean(recon_loss_val_samp)
                        kl_loss_val_samp = -0.5 * (1 + z_log_var_val - torch.square(z_mean_val) - torch.exp(z_log_var_val))
                        kl_loss_val = torch.mean(torch.sum(kl_loss_val_samp, dim=1))
                        total_val_loss += (recon_loss_val + current_beta * kl_loss_val).item()
                        total_val_recon_loss += recon_loss_val.item()
                        if i_val == 0:
                            X_val_sample_plot_temp = batch_X_past_val[0].cpu()
                            Y_val_sample_plot_temp = batch_Y_future_val[0].cpu()
                            # (V13 TCVAE không "nhả" Attn)
                            # attn_weights_plot_temp = attn_weights_val[0].cpu().numpy() 

                avg_val_loss = total_val_loss / len(val_loader)
                avg_val_recon_loss = total_val_recon_loss / len(val_loader)
            
            # --- (C) LOGGING & CHECKPOINT ---
            if val_split > 0.0:
                logging.info(f"[LB={lookback:03d}] E{epoch+1:03d} | Beta: {current_beta:.4f} | Train Loss: {avg_train_loss:.4f} | **Val Loss: {avg_val_loss:.4f}** | Val Recon: {avg_val_recon_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
            else:
                logging.info(f"[LB={lookback:03d}] E{epoch+1:03d} | Beta: {current_beta:.4f} | Train Loss (100%): {avg_train_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
            
            scheduler.step()
            
            if writer:
                writer.add_scalar(f'Loss/Train_LB{lookback}', avg_train_loss, epoch)
                writer.add_scalar(f'Loss/Validation_LB{lookback}', avg_val_loss, epoch)
                writer.add_scalar(f'Loss/Val_Recon_Thuần_LB{lookback}', avg_val_recon_loss, epoch)
                writer.add_scalar(f'Params/LearningRate_LB{lookback}', optimizer.param_groups[0]['lr'], epoch)
                writer.add_scalar(f'Params/Beta_LB{lookback}', current_beta, epoch)

            if use_patience:
                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss; patience_counter = 0
                    torch.save(decoder.state_dict(), final_model_name) 
                    logging.info(f"-> [LB={lookback}] Đã lưu 'Não Vẽ Transformer' (Val Loss: {best_loss:.4f}) -> {final_model_name}")
                    best_X_plot = X_val_sample_plot_temp; best_Y_plot = Y_val_sample_plot_temp
                    # best_attn_plot = attn_weights_plot_temp # (V13 không "vẽ" Attn)
                    best_epoch = epoch
                else:
                    patience_counter += 1
                if resume_file_name:
                    torch.save({ 'epoch': epoch, 'encoder_state_dict': encoder.state_dict(), 
                        'decoder_state_dict': decoder.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(), 'best_loss': best_loss,
                        'patience_counter': patience_counter }, resume_file_name)
                
                # === ("VÁ" LỖI 2/2) ===
                if patience_counter >= patience: 
                # =====================
                    logging.info(f"[LB={lookback}] Dừng sớm (EarlyStopping) tại Epoch {epoch+1}.")
                    break
    
    except KeyboardInterrupt:
        logging.warning(f"\n[LB={lookback}] Đã bắt được (Ctrl+C)! Đang 'lưu khẩn cấp'...")
        torch.save(decoder.state_dict(), final_model_name + ".interrupt.pth")
        if pynvml: pynvml.nvmlShutdown() 
        sys.exit(0) 
    
    # --- (D) "TRẢ HÀNG" (VẼ ẢNH) ---
    if not use_patience:
        torch.save(decoder.state_dict(), final_model_name)
        logging.info(f"-> 'Não Xịn' 100% (Epoch {total_epochs}) đã lưu -> {final_model_name}")
    else:
        try:
            decoder.load_state_dict(torch.load(final_model_name))
            logging.info(f"Đang 'Vẽ' 1 Chart (PNG) 'Xịn' nhất (từ Epoch {best_epoch})...")
            if best_X_plot is not None:
                plot_cvae_samples(best_X_plot, best_Y_plot, encoder, decoder, writer, best_epoch, lookback)
                # plot_attention_heatmap(best_attn_plot, writer, best_epoch, lookback) # (V13 không "vẽ" Attn)
            else:
                logging.warning("Không tìm thấy 'hàng' (data) 'xịn' nhất để 'vẽ' chart PNG.")
        except FileNotFoundError:
             logging.error(f"LỖI: Không tìm thấy file 'não xịn' {final_model_name} để 'vẽ'.")
        
    logging.info(f"=== HOÀN TẤT 'DÂY CHUYỀN' (Lookback={lookback}) ===")
    if writer: writer.close() 

    # === ("VÁ" LỖI 4 - "TRẢ HÀNG" VRAM OOM) ===
    return (encoder, decoder, X_past_t, Y_future_t, X_past_train, Y_future_train)
    # =========================================

# --- (HÀM MAIN "CÔNG XƯỞNG" V13 - "Ăn" 53 Món) ---
if __name__ == "__main__":

    print("\n" + "="*70)
    print("      LÒ ĐÚC NÃO TCVAE V1 ('Não' TRANSFORMER 'Ăn' 53 Món)")
    print("="*70)
    
    logging.info("--- CHẾ ĐỘ 'SOI HÀNG' (V13) ĐANG KHỞI ĐỘNG (Luyện 80/20) ---")
    
    mode_config = {
        'desc': "SOI V13 80/20 ('Não' Transformer)",
        'epochs': DEFAULT_SOI_EPOCHS, 
        'val_split': 0.2, 
        'log_dir': DEFAULT_TENSORBOARD_DIR,
        'final_name': os.path.join(DIR_MODELS, "transformer_cvae_decoder_V13_{lb}_{lf}_best.pth"),
        'resume_name': os.path.join(DIR_MODELS, "transformer_cvae_model_V13_{lb}_resume.pth"),
        'use_patience': True, 
        'log_file': "log_train_transformer_cvae_V13_SOI.log"
    }

    # --- "LẮP" ĐỒ NGHỀ ---
    try: import psutil
    except ImportError: psutil = None
        
    log_filename = mode_config['log_file']
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - [TrainTCVAE_V13] - %(message)s',
                        handlers=[
                            logging.FileHandler(log_filename, mode='w', encoding='utf-8'), 
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
    
    logging.info(f"=== 'LÒ' V13 (TCVAE) CHẠY CHẾ ĐỘ: {mode_config['desc']} ===")
            
    # 1. TẢI "THỨC ĂN" VÀ "ĂN KÉ" SCALER V23
    data_scaled, num_features, feature_names = load_data_and_scaler_V23()
    
    if data_scaled is None:
        logging.error("Dừng 'CÔNG XƯỞNG' (TCVAE V13) vì không có 'thức ăn' hoặc 'scaler'.")
        if pynvml: pynvml.nvmlShutdown() 
        sys.exit(1)
        
    # 2. "CHẾ" LOSS "QUYỀN TRỌNG" (Weighted Loss)
    logging.info(f"Đang 'chế' Weighted Loss V13 (x{WEIGHTED_LOSS_MULTIPLIER} cho OHLC)...")
    try:
        weights_tensor = torch.ones(num_features).to(device)
        
        vip_cols = ['H1_Close', 'H1_High', 'H1_Low', 'H1_Open']
        found_count = 0
        for col in vip_cols:
            try:
                # (Dùng feature_names (V13) trả về từ hàm load_data)
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
        
        if CVAE_CLOSE_IDX_FOR_PLOT == -1:
            logging.error("LỖI CHÍ MẠNG: Không tìm thấy 'H1_Close' trong 'thức ăn 53 món'. Không 'vẽ' được.")
            sys.exit(1)
            
        logging.info(f"Đã 'độ' Weighted Loss (x{WEIGHTED_LOSS_MULTIPLIER}) cho {found_count} cột 'VIP'.")
    except Exception as e:
        logging.error(f"LỖI 'độ' Weighted Loss. Dùng tạm loss 'cào bằng'. Lỗi: {e}")
        weights_tensor = torch.ones(num_features).to(device)
    
    # 3. LẶP "ĐÚC" 2 NÃO (50, 168)
    ALL_LOOKBACKS = [50, 168]
    
    # (Biến "toàn cục" để "Nuke" VRAM)
    encoder = None
    decoder = None
    X_past_t = None
    Y_future_t = None
    X_past_train = None
    Y_future_train = None
    
    for lb in ALL_LOOKBACKS:
        logging.info(f"\n{'='*70}\n === BẮT ĐẦU 'DÂY CHUYỀN' (TCVAE V13 'Não Mới' Lookback={lb}) ===\n{'='*70}")
        
        X_past, Y_future = create_windows(data_scaled, lb, LOOKFORWARD)
        
        if X_past is None:
            logging.warning(f"Bỏ qua Lookback={lb} do không đủ dữ liệu 'cắt'.")
            continue
            
        # (SỬA V13 - VÁ LỖI OOM)
        # "Nuke" (giết) 2 "não" và data "cũ" (nếu có)
        if 'encoder' in locals() and encoder is not None: del encoder
        if 'decoder' in locals() and decoder is not None: del decoder
        if 'X_past_t' in locals() and X_past_t is not None: del X_past_t
        if 'Y_future_t' in locals() and Y_future_t is not None: del Y_future_t
        if 'X_past_train' in locals() and X_past_train is not None: del X_past_train
        if 'Y_future_train' in locals() and Y_future_train is not None: del Y_future_train
        
        # "Dọn" (empty) VRAM
        torch.cuda.empty_cache() 
        logging.info(f"Đã 'Nuke' VRAM. Bắt đầu 'luyện' Lookback={lb}...")
        
        (encoder, decoder, X_past_t, Y_future_t, X_past_train, Y_future_train) = train_single_cvae_pytorch(
            X_past, Y_future, 
            lb, LOOKFORWARD, num_features, 
            weights_tensor,
            mode_config 
        )
    
    logging.info(f"\n{'='*70}\n === HOÀN TẤT 'LÒ V13' (TCVAE) - ĐÃ CHẠY XONG {len(ALL_LOOKBACKS)} NÃO! ===\n{'='*70}")
    
    logging.info(f"Đã 'xuất' ảnh PNG 'Xịn' nhất (TCVAE V13) vào thư mục: {DIR_MODELS}")
    logging.info(f"Đại ca 'soi' (V13) bằng: tensorboard --logdir={mode_config['log_dir']}")
    
    if pynvml:
        pynvml.nvmlShutdown()