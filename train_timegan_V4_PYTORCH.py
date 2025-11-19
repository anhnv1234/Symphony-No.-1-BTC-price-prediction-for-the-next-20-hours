import pandas as pd
import numpy as np
import os
import logging
import joblib
from sklearn.preprocessing import MinMaxScaler
import warnings 
import psutil 
import sys 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

# --- KHẮC PHỤC LỖI HIỂN THỊ TIẾNG VIỆT TRÊN WINDOWS ---
if sys.version_info.major == 3 and sys.version_info.minor >= 7:
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# --- THẦN CHÚ PYTORCH ---
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.data")

# =========================================================================
# 💡 KIẾN TRÚC ADVANCED TIME-SERIES GAN (DÙNG GRU VÀ LATENT SPACE)
# =========================================================================

# --- Mô hình GRU cơ sở ---
class BaseGRU(nn.Module):
    """Lớp GRU cơ sở cho các mạng E, R, G, D"""
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers=2):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # output: (batch_size, seq_len, hidden_dim)
        output, _ = self.rnn(x) 
        # Trả về kết quả sau khi qua lớp tuyến tính
        return self.output_layer(output)

# 1. Embedder (E): X -> H (Thực)
class AdvancedTS_Embedder(BaseGRU):
    def __init__(self, num_features, hidden_dim):
        super().__init__(num_features, hidden_dim, hidden_dim)

# 2. Recovery (R): H -> X (Thực)
class AdvancedTS_Recovery(BaseGRU):
    def __init__(self, num_features, hidden_dim):
        super().__init__(hidden_dim, num_features, hidden_dim)

# 3. Generator (G): Z -> H_tilde (Giả trong Latent Space)
class AdvancedTS_Generator(BaseGRU):
    def __init__(self, hidden_dim):
        super().__init__(hidden_dim, hidden_dim, hidden_dim)

# 4. Discriminator (D): H/H_tilde -> Score (Latent Space Discrimination)
class AdvancedTS_Discriminator(BaseGRU):
    def __init__(self, hidden_dim):
        # Đầu ra 1 scalar (score) cho mỗi bước thời gian
        super().__init__(hidden_dim, 1, hidden_dim)


class AdvancedTS_Trainer:
    """Class quản lý quá trình huấn luyện 3 bước: Reconstruction, Supervised, Adversarial"""
    def __init__(self, lookback, num_features, hidden_dim, device):
        self.device = device
        self.lookback = lookback
        self.hidden_dim = hidden_dim
        
        # Khởi tạo 4 mạng cốt lõi
        self.E = AdvancedTS_Embedder(num_features, hidden_dim).to(device)
        self.R = AdvancedTS_Recovery(num_features, hidden_dim).to(device)
        self.G = AdvancedTS_Generator(hidden_dim).to(device)
        self.D = AdvancedTS_Discriminator(hidden_dim).to(device)

        # Optimizers (Sử dụng các optimizer riêng biệt cho từng nhóm Loss)
        self.lr = 1e-3
        self.optimizer_ER = optim.Adam(list(self.E.parameters()) + list(self.R.parameters()), lr=self.lr)
        self.optimizer_G = optim.Adam(self.G.parameters(), lr=self.lr)
        self.optimizer_D = optim.Adam(self.D.parameters(), lr=self.lr)

        # Hàm mất mát
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss() # Tốt hơn cho GAN
        self.lambda_adv = 1.0 # Trọng số cho Adversarial Loss
        self.lambda_sup = 1.0 # Trọng số cho Supervised Loss
        self.lambda_rec = 10.0 # Trọng số lớn hơn cho Reconstruction (Consistency) Loss

    def _get_noise(self, batch_size):
        """Tạo vector nhiễu Z có cùng shape với dữ liệu"""
        return torch.randn(batch_size, self.lookback, self.hidden_dim).to(self.device)

    # =====================================================
    # 1. HUẤN LUYỆN TÁI TẠO (RECONSTRUCTION WARM-UP)
    # =====================================================
    def train_step_reconstruction(self, real_data_batch):
        self.optimizer_ER.zero_grad()
        
        H_real = self.E(real_data_batch) # X -> H
        X_reconstructed = self.R(H_real) # H -> X_reconstructed
        
        # Loss Tái tạo: E(R(X)) nên giống X
        loss_R = self.mse(real_data_batch, X_reconstructed)
        
        loss_R.backward()
        self.optimizer_ER.step()
        return loss_R.item()

    # =====================================================
    # 2. HUẤN LUYỆN ADVERSARIAL VÀ SUPERVISED (SUPERIOR TRAINING)
    # CÓ BỔ SUNG AUTOREGRESSIVE VÀ CONSISTENCY LOSS
    # =====================================================
    def train_step_adversarial(self, real_data_batch):
        batch_size = real_data_batch.size(0)
        
        # --- A. Cập nhật Generator (G) và Embedder/Recovery (E/R) ---
        # Mục tiêu: Đảm bảo G tạo ra dữ liệu có thể phục hồi tốt và tuân theo dynamics (S + R)
        self.optimizer_G.zero_grad()
        self.optimizer_ER.zero_grad() 
        
        # Lấy Embedded sequences thực
        H_real = self.E(real_data_batch) 
        
        # 1. Unsupervised Adversarial Loss (Loss U)
        Z = self._get_noise(batch_size)
        H_synthetic = self.G(Z)
        D_synthetic_for_G = self.D(H_synthetic)
        # G muốn D_synthetic -> 1
        loss_U = self.bce(D_synthetic_for_G, torch.ones_like(D_synthetic_for_G))
        
        # 2. Supervised Loss (Loss S - Autoregressive Proxy)
        # Khuyến khích G học ánh xạ động lực của dữ liệu thực
        H_predicted = self.G(H_real) 
        loss_S = self.mse(H_predicted, H_real)
        
        # 3. Reconstruction/Consistency Loss (Loss R)
        # Đảm bảo E và R vẫn hoạt động tốt, và G tạo ra latent code có thể recover
        X_reconstructed = self.R(H_real) # R(E(X))
        loss_R = self.mse(real_data_batch, X_reconstructed)

        # Tổng Loss G (U + S + R)
        loss_G_final = self.lambda_adv * loss_U + self.lambda_sup * loss_S + self.lambda_rec * torch.sqrt(loss_R)
        
        loss_G_final.backward(retain_graph=True) # Retain graph cần thiết vì D sẽ dùng E/G/H
        self.optimizer_G.step()
        # E, R được tối ưu cùng G (qua Loss R)
        self.optimizer_ER.step()
        
        # --- B. Cập nhật Discriminator (D) ---
        self.optimizer_D.zero_grad()
        
        # D-Loss trên dữ liệu THẬT (target=1)
        # H_real phải được tính lại mà không qua gradient của G
        H_real = self.E(real_data_batch).detach() 
        D_real = self.D(H_real)
        loss_D_real = self.bce(D_real, torch.ones_like(D_real))
        
        # D-Loss trên dữ liệu GIẢ (target=0)
        # H_synthetic phải được tính lại mà không qua gradient của G
        Z = self._get_noise(batch_size)
        H_synthetic = self.G(Z).detach()
        D_synthetic = self.D(H_synthetic)
        loss_D_fake = self.bce(D_synthetic, torch.zeros_like(D_synthetic))
        
        # D-Loss tổng
        loss_D = loss_D_real + loss_D_fake
        loss_D_final = loss_D * self.lambda_adv
        
        loss_D_final.backward()
        self.optimizer_D.step()
        
        return loss_G_final.item(), loss_D_final.item() #, loss_R.item() # Có thể trả về Loss R để theo dõi

    def get_all_states(self):
        """Lấy tất cả state dict cần thiết cho việc lưu trữ"""
        return {
            'E_state_dict': self.E.state_dict(),
            'R_state_dict': self.R.state_dict(),
            'G_state_dict': self.G.state_dict(),
            'D_state_dict': self.D.state_dict(),
            'optER_state_dict': self.optimizer_ER.state_dict(),
            'optG_state_dict': self.optimizer_G.state_dict(),
            'optD_state_dict': self.optimizer_D.state_dict(),
        }

# =========================================================================

# --- KHỞI TẠO VÀ CẤU HÌNH ---

pynvml = None
GPU_HANDLE = None

# Cố gắng import pynvml để theo dõi GPU
try:
    import pynvml
    pynvml.nvmlInit()
except (ImportError, Exception):
    pynvml = None
    if torch.cuda.is_available():
        logging.warning("Không tìm thấy 'linh kiện' pynvml. Sẽ không theo dõi được trạng thái GPU.")
    
# Cấu hình logging (Thêm FileHandler với encoding UTF-8)
log_filename = "log_train_advanced_tsgan.log" 
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - [AdvancedTSGAN_PYTORCH] - %(message)s',
                    handlers=[
                        logging.FileHandler(log_filename, mode='a', encoding='utf-8'), 
                        logging.StreamHandler()
                    ])
logging.info(f"Log sẽ được lưu vào file: {log_filename}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Đang sử dụng thiết bị: {device}")

MASTER_FILE_PATH = os.path.join('02_Master_Data', 'btcusdt_master_data.parquet')
SCALER_FILENAME = os.path.join('01_Processed_Data', 'cvae_scaler_V23.gz')

# Thay đổi số bước huấn luyện để dành cho từng giai đoạn
WARMUP_STEPS = 500 # Bước khởi động (Reconstruction)
ADVERSARIAL_STEPS = 4500 # Bước chính (Adversarial + Supervised + Consistency)
TRAIN_STEPS = WARMUP_STEPS + ADVERSARIAL_STEPS

GAN_BATCH_SIZE = 64
HIDDEN_DIM = 24 

DIR_MODELS = "03_Models"
DIR_PROCESSED = "01_Processed_Data"
os.makedirs(DIR_MODELS, exist_ok=True)
os.makedirs(DIR_PROCESSED, exist_ok=True)


class HardwareMonitor:
    # Lớp theo dõi phần cứng (giữ nguyên)
    def __init__(self, device):
        self.device = device
        self.pynvml = None
        self.gpu_handle = None
        self.nvml_imported = False
        
        if self.device.type == 'cuda':
            try:
                global pynvml
                if pynvml:
                    self.pynvml = pynvml
                    self.gpu_handle = self.pynvml.nvmlDeviceGetHandleByIndex(0)
                    self.nvml_imported = True
            except Exception:
                self.nvml_imported = False

    def log_usage(self, prefix=""):
        if self.device.type == 'cuda' and self.nvml_imported:
            try:
                info = self.pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                gpu_memory = f"VRAM: {info.used / 1024**3:.2f} GB / {info.total / 1024**3:.2f} GB | "
            except Exception:
                gpu_memory = "VRAM: N/A | "
        else:
            gpu_memory = ""

        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        ram_usage = f"RAM: {mem_info.rss / 1024**3:.2f} GB"
        
        logging.info(f"{prefix}{gpu_memory}{ram_usage}")


# --- HÀM LOAD DATA (Giữ nguyên) ---
def load_data_and_scaler_V23():
    logging.info(f"Đang tải file master V23 (53 món): {MASTER_FILE_PATH}...")
    try:
        df = pd.read_parquet(MASTER_FILE_PATH)
    except FileNotFoundError:
        logging.error(f"LỖI: Không tìm thấy file {MASTER_FILE_PATH}.")
        return None, None
        
    logging.info(f"Đang tải 'Bộ Chuẩn Hóa' (V23 - 53 món) (dùng chung CVAE): {SCALER_FILENAME}...")
    try:
        scaler = joblib.load(SCALER_FILENAME)
    except FileNotFoundError:
        logging.error(f"LỖI: Không tìm thấy file {SCALER_FILENAME}.")
        logging.error("Đại ca đã chạy 'Lò' (train_cvae_V11.py) (file 'ăn' 53 món) chưa?")
        return None, None
    
    df.interpolate(method='time', inplace=True)
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    df.fillna(0, inplace=True) # (Vá V5: "Trám" 0 (nếu ffill/bfill vẫn sót))

    features_df = df
    feature_names = features_df.columns.tolist()
    num_features = len(feature_names)
    logging.info(f"Tổng số features (món ăn V23 - 53 món) đang sử dụng: {num_features}")

    logging.info("Đang chuẩn hóa 'thức ăn' (dùng Scaler Lò V11)...")
    data_scaled = scaler.transform(features_df)
    
    return data_scaled, num_features


# --- HÀM CREATE WINDOWS (Giữ nguyên) ---
def create_windows_for_seriesgan(data_scaled, lookback):
    logging.info(f"Đang 'cắt' cửa sổ AdvancedTSGAN (Lookback={lookback})...")
    
    num_samples = len(data_scaled) - lookback + 1
    
    if num_samples <= 0:
        logging.error(f"LỖI (Lookback={lookback}): Dữ liệu quá ít, không đủ 'cắt' 1 cửa sổ.")
        return None, None

    # Dùng 'stride_tricks' để "cắt" cửa sổ (siêu nhanh)
    n_features = data_scaled.shape[1]
    shape = (num_samples, lookback, n_features)
    strides = (data_scaled.strides[0], data_scaled.strides[0], data_scaled.strides[1])
    X_tsgan_np = np.lib.stride_tricks.as_strided(data_scaled, shape=shape, strides=strides)
    
    logging.info(f"Shape 'Thức Ăn' AdvancedTSGAN (NP): {X_tsgan_np.shape}") 
    
    # Chuyển sang PyTorch Tensor
    X_tsgan_tensor = torch.tensor(X_tsgan_np, dtype=torch.float32)
    
    # Tạo Dataset và DataLoader
    dataset = TensorDataset(X_tsgan_tensor)
    dataloader = DataLoader(dataset, batch_size=GAN_BATCH_SIZE, shuffle=True, drop_last=True)
    
    return X_tsgan_tensor, dataloader


def train_single_seriesgan_pytorch(dataloader, lookback, num_features):
    """Huấn luyện Advanced Time-series GAN qua 3 giai đoạn: Reconstruction, Supervised, Adversarial"""
    
    monitor = HardwareMonitor(device)
    monitor.log_usage(prefix="[TRƯỚC TRAIN] ")

    # 1. LOGIC "LƯU/TẢI" CHECKPOINT
    checkpoint_resume_path = os.path.join(DIR_MODELS, f'advanced_tsgan_checkpoint_{lookback}_resume.pth')
    
    # 2. KHỞI TẠO MÔ HÌNH
    try:
        synthesizer = AdvancedTS_Trainer(lookback=lookback, num_features=num_features, hidden_dim=HIDDEN_DIM, device=device)
    except Exception as e:
        logging.error(f"Lỗi khởi tạo AdvancedTS_Trainer: {e}")
        return

    start_step = 0
    
    if os.path.exists(checkpoint_resume_path):
        logging.info(f"Phát hiện Checkpoint 'Luyện Tiếp'! Đang tải từ: {checkpoint_resume_path}")
        try:
            checkpoint = torch.load(checkpoint_resume_path, map_location=device)
            synthesizer.E.load_state_dict(checkpoint['E_state_dict']) 
            synthesizer.R.load_state_dict(checkpoint['R_state_dict']) 
            synthesizer.G.load_state_dict(checkpoint['G_state_dict']) 
            synthesizer.D.load_state_dict(checkpoint['D_state_dict']) 
            synthesizer.optimizer_ER.load_state_dict(checkpoint['optER_state_dict'])
            synthesizer.optimizer_G.load_state_dict(checkpoint['optG_state_dict'])
            synthesizer.optimizer_D.load_state_dict(checkpoint['optD_state_dict'])
            start_step = checkpoint['step']
            logging.info(f"Tải Checkpoint thành công! Sẽ 'luyện tiếp' từ Step: {start_step}")
        except Exception as e:
            logging.error(f"Lỗi tải Checkpoint: {e}. 'Luyện' (Train) lại từ đầu.")
            start_step = 0
            
    # 3. HUẤN LUYỆN (Vòng lặp tiêu chuẩn PyTorch)
    logging.info(f"Bắt đầu huấn luyện [AdvancedTSGAN Lookback={lookback}]...")
    logging.info(f"(Warmup: {WARMUP_STEPS} steps, Adversarial: {ADVERSARIAL_STEPS} steps) - Tổng: {TRAIN_STEPS} mẻ...")
    
    current_stage = "Reconstruction Warmup"
    
    try:
        data_iterator = iter(dataloader)
        
        for step in tqdm(range(start_step, TRAIN_STEPS), initial=start_step, total=TRAIN_STEPS, desc=f"LB={lookback}"):
            
            # Chuyển sang giai đoạn Adversarial
            if step == WARMUP_STEPS:
                current_stage = "Adversarial Training"
                logging.info(f"\n[Step {step}] === CHUYỂN SANG GIAI ĐOẠN HUẤN LUYỆN ĐỐI NGHỊCH (ADVERSARIAL + SUPERVISED + CONSISTENCY) ===")

            try:
                real_data_batch = next(data_iterator)[0].to(device)
            except StopIteration:
                data_iterator = iter(dataloader)
                real_data_batch = next(data_iterator)[0].to(device)
            
            # --- Thực hiện các bước huấn luyện theo giai đoạn ---
            
            loss_G, loss_D, loss_R_cont = None, None, None
            
            if step < WARMUP_STEPS:
                # Giai đoạn 1: Chỉ huấn luyện E và R (Tái tạo)
                loss_R_cont = synthesizer.train_step_reconstruction(real_data_batch) 
            else:
                # Giai đoạn 2: Huấn luyện G, D, và E/R liên tục
                loss_G, loss_D = synthesizer.train_step_adversarial(real_data_batch)
                # Lấy loss R riêng để log
                H_real_temp = synthesizer.E(real_data_batch).detach()
                X_reconstructed_temp = synthesizer.R(H_real_temp)
                loss_R_cont = synthesizer.mse(real_data_batch, X_reconstructed_temp).item()

            
            # Thêm log định kỳ để theo dõi
            if step > start_step and step % 500 == 0:
                 log_msg = f"[{current_stage} Step {step}]"
                 if loss_R_cont is not None: log_msg += f" Loss R: {loss_R_cont:.4f}"
                 if loss_G is not None: log_msg += f" Loss G: {loss_G:.4f}, Loss D: {loss_D:.4f}"
                 logging.info(log_msg)

        
        # 4. LƯU "Não xịn" (Cuối cùng)
        model_final_path = os.path.join(DIR_MODELS, f'advanced_tsgan_model_{lookback}_final.pth') 
        
        final_states = synthesizer.get_all_states()
        final_states['step'] = TRAIN_STEPS 
        
        torch.save(final_states, model_final_path)
        
        logging.info(f"Lưu 'Não xịn' (Final) thành công -> {model_final_path}")
        
        if os.path.exists(checkpoint_resume_path):
            os.remove(checkpoint_resume_path)
            
        logging.info(f"=== HOÀN TẤT LÒ 2.C (AdvancedTSGAN - Lookback={lookback}) ===")
    
    except KeyboardInterrupt:
        # Bắt "Ctrl+C" và Lưu Checkpoint Khẩn cấp
        logging.warning(f"\n[LB={lookback}] Đã bắt được (Ctrl+C)! Đang 'lưu khẩn cấp' Checkpoint 'Luyện Tiếp'...")
        # LƯU KHẨN CẤP
        torch.save({
            'step': step,
            **synthesizer.get_all_states()
        }, checkpoint_resume_path)
        logging.info(f"Lưu khẩn cấp thành công -> {checkpoint_resume_path}. Lần sau chạy lại sẽ 'luyện tiếp' từ đây.")
        if pynvml: pynvml.nvmlShutdown() 
        sys.exit(0) 
        
    except Exception as e:
        logging.error(f"LỖI CHÍ MẠNG khi 'luyện' AdvancedTSGAN (Lookback={lookback}): {e}")
        # Cố gắng lưu khẩn cấp lần cuối
        if 'synthesizer' in locals():
            last_states = synthesizer.get_all_states()
            last_states['step'] = step 
            torch.save(last_states, checkpoint_resume_path) 
        return 


# --- HÀM MAIN "CÔNG XƯỞNG" ---
if __name__ == "__main__":
    
    logging.info(f"=== KHỞI ĐỘNG 'CÔNG XƯỞNG' ĐÚC NÃO ADVANCED TS-GAN (PYTORCH - 53 MÓN) ===")
    
    # 1. TẢI VÀ XỬ LÝ "THỨC ĂN"
    data_scaled, num_features = load_data_and_scaler_V23()
    
    if data_scaled is None:
        logging.error("Dừng 'CÔNG XƯỞNG' (AdvancedTSGAN) vì không có 'thức ăn' hoặc 'scaler'.")
        if pynvml: 
            try: pynvml.nvmlShutdown()
            except: pass
        sys.exit(1)
        
    # 2. LẶP "ĐÚC" NÃO 
    ALL_LOOKBACKS = [50, 168]
    
    for lb in ALL_LOOKBACKS:
        logging.info(f"\n{'='*70}\n === BẮT ĐẦU 'DÂY CHUYỀN' (AdvancedTSGAN Lookback={lb}) ===\n{'='*70}")
        
        # 3. "CẮT" CỬA SỔ (Trả về DataLoader)
        X_tsgan_tensor, dataloader = create_windows_for_seriesgan(data_scaled, lb)
        
        if dataloader is None:
            logging.warning(f"Bỏ qua Lookback={lb} do không đủ dữ liệu 'cắt'.")
            continue
            
        # 4. "ĐÚC" (Train)
        train_single_seriesgan_pytorch(dataloader, lb, num_features)
        
        best_model_path_for_this_lb = os.path.join(DIR_MODELS, f'advanced_tsgan_model_{lb}_final.pth')
        logging.info(f"\n{'*' * 25} HOÀN THÀNH 'ĐÚC' NÃO (AdvancedTSGAN Lookback={lb}) {'*' * 25}")
        logging.info(f"-> 'Não xịn' nhất (final model) đã được lưu tại: {best_model_path_for_this_lb}")
        logging.info(f"Đang dọn dẹp VRAM (empty_cache) trước khi sang 'dây chuyền' Lookback tiếp theo...")
        
        torch.cuda.empty_cache()
    
    logging.info(f"\n{'='*70}\n === HOÀN TẤT 'CÔNG XƯỞNG' ADVANCED TS-GAN - ĐÃ ĐÚC XONG CẢ {len(ALL_LOOKBACKS)} NÃO! ===\n{'='*70}")
    
    if pynvml:
        try: pynvml.nvmlShutdown()
        except: pass