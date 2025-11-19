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
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight 
from tqdm import tqdm
import sys 
import math # (Cần cho "Não" Transformer)

# --- (MÓN 1) "BỊT MIỆNG" CẢNH BÁO "RÁC" ---
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- CẤU HÌNH V2 (Transformer "Vá Lỗi" T=49) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# (Ăn file ~28 món H1 "cách ly" từ "Lò" V5)
FEATURE_FILE_PATH = os.path.join('02_Master_Data', 'btcusdt_master_data_features_V5_H1_ONLY.parquet')
# ("Đúc" Scaler MỚI cho 28 món)
SCALER_FILENAME = os.path.join('01_Processed_Data', 'transformer_scaler_V1_H1_ONLY.gz')
MODEL_DIR = "03_Models"
DIR_PROCESSED = "01_Processed_Data" 
MODEL_NAME = "transformer_v2_h1_only_balanced.pth" # (Đổi tên V2)
LOG_FILE = "log_02_train_transformer_V2.log" # (Log file V2)

# --- CẤU HÌNH "NÃO" TRANSFORMER ---
LOOKBACK = 50 
LOOKFORWARD = 20 
THRESHOLD = 0.005 

# (Cấu hình "Não")
NUM_FEATURES = 28 # (Số món "dự kiến" từ "Lò" V5)
D_MODEL = 64      
N_HEAD = 4        
NUM_LAYERS = 3    
NUM_CLASSES = 3   

# (Cấu hình "Lò Luyện")
BATCH_SIZE = 64
EPOCHS = 100
PATIENCE = 10     
LEARNING_RATE = 0.0005

# --- CẤU HÌNH LOGGING (UTF-8) ---
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass
    
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - [TrainTransformer_V2] - %(message)s',
                    handlers=[
                        logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'), 
                        logging.StreamHandler(sys.stdout)
                    ])

# === (HÀM "HẬU CẦN" V2) ===

def load_data_and_prep_V2(filepath, lookback, lookforward, threshold):
    """
    "Ăn" 28 món H1, "Đúc" Scaler V1, "Chế" Đáp Án, "Cắt" Chuỗi (Vá lỗi T=49)
    """
    global NUM_FEATURES 
    
    logging.info(f"Đang tải 'thức ăn H1 cách ly' V5 (~28 món) từ: {filepath}...")
    try:
        df = pd.read_parquet(filepath)
    except Exception as e:
        logging.error(f"LỖI: Không tải được file 'thức ăn xịn' V5. Lỗi: {e}")
        logging.error("Đại ca đã chạy 'Lò Nấu Ăn' (00_build_features_V5_H1_ONLY.py) chưa?")
        return None, None
        
    logging.info(f"Tải thành công. Shape 'thức ăn' V5: {df.shape}")
    
    # --- 1. "Chế" ra "Đáp Án" V2 (0: Sideways, 1: Lên Ngon, 2: Xuống SML) ---
    logging.info(f"Đang 'chế' cột 'Đáp Án' V2 (Threshold: {threshold * 100:.2f}%) cho {lookforward} nến tới...")
    
    if 'H1_Close' not in df.columns:
        logging.error("LỖI CHÍ MẠNG: 'Thức ăn' V5 thiếu cột 'H1_Close' để 'chế' đáp án!")
        return None, None
        
    df['future_price'] = df['H1_Close'].shift(-lookforward)
    df['pct_change'] = (df['future_price'] - df['H1_Close']) / df['H1_Close']
    df.dropna(subset=['future_price', 'pct_change'], inplace=True)
    df['target'] = 0 
    df.loc[df['pct_change'] > threshold, 'target'] = 1 
    df.loc[df['pct_change'] < -threshold, 'target'] = 2 
    
    logging.info(f"Phân bổ 'Đáp Án':\n{df['target'].value_counts(normalize=True)}")
    
    # --- 2. "Tách" (X) và "Đáp Án" (Y) ---
    features_df = df.drop(columns=['future_price', 'pct_change', 'target'])
    targets_series = df['target']
    
    NUM_FEATURES = len(features_df.columns) 
    logging.info(f"Đã tách X ({NUM_FEATURES} features H1-ONLY) và Y (Target V2).")
    
    # --- 3. "Đúc" Scaler MỚI V1 ---
    # (Check "ké" xem Scaler V1 đã "đúc" chưa)
    if os.path.exists(SCALER_FILENAME):
        logging.info(f"Đã phát hiện Scaler V1 (H1-ONLY) 'đúc' sẵn. Đang 'ăn ké'...")
        scaler = joblib.load(SCALER_FILENAME)
        data_scaled = scaler.transform(features_df)
    else:
        logging.info(f"Đang 'luyện' (fit) Scaler MỚI (V1 - {NUM_FEATURES} món H1)...")
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(features_df)
        joblib.dump(scaler, SCALER_FILENAME)
        logging.info(f"Đã lưu Scaler MỚI (V1) -> {SCALER_FILENAME}")

    # --- 4. "Cắt" Chuỗi (Sequences) ---
    logging.info(f"Đang 'cắt' chuỗi (Lookback={lookback})...")
    X, y = [], []
    
    # (SỬA V2: "Lùi" 1 "nến" để "khớp" (match) X và Y)
    num_samples = len(data_scaled) - lookback - lookforward + 1
    
    if num_samples <= 0:
        logging.error(f"LỖI (Lookback={lookback}): Dữ liệu quá ít.")
        return None, None

    for i in tqdm(range(num_samples), desc=f"Cắt chuỗi Transformer LB={lookback}"):
        # X = T=0 đến T=49 (50 nến)
        X.append(data_scaled[i : i + lookback])
        # Y = "Đáp án" của 20 nến "sau" T=49 (tức là T=49 -> T=69)
        # (Lấy "đáp án" tại nến "cuối cùng" của "chuỗi" (i + lookback - 1))
        y.append(targets_series.iloc[i + lookback - 1]) 
    
    X = np.array(X)
    y = np.array(y)
    
    logging.info(f"Shape 'Thức Ăn' (X): {X.shape} | Shape 'Đáp Án' (Y): {y.shape}")
    
    return X, y

# === (NÃO TRANSFORMER V1) ===
class PositionalEncoding(nn.Module):
    """"Nhồi" vị trí (thứ tự 1, 2, 3...) vào "não" """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ x: Shape (Seq_Len, Batch, d_model) """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerModel(nn.Module):
    """ "Não" Transformer (Encoder-Only) cho "Phán Kèo" """
    def __init__(self, num_features, d_model, nhead, num_layers, num_classes, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        # 1. Lớp "Nhồi"
        self.input_embed = nn.Linear(num_features, d_model)
        # 2. Lớp "Nhớ Vị Trí"
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        # 3. "Não" Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # 4. Lớp "Phán Kèo"
        self.output_layer = nn.Linear(d_model, num_classes)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """ src: Shape (Batch, Seq_Len, num_features) """
        src = self.input_embed(src) * np.sqrt(self.d_model)
        src = src.permute(1, 0, 2)
        src = self.pos_encoder(src)
        src = src.permute(1, 0, 2)
        output = self.transformer_encoder(src)
        output = output[:, -1, :] 
        output = self.output_layer(output) 
        return output

# === (HÀM "LÕI" "LÒ LUYỆN") ===
def train_transformer_v2(X, y):
    
    # === (SỬA 90/10) ===
    logging.info("Đang 'chia sân' 90% (Train) - 10% (Test)... (KHÔNG XÁO TRỘN)")
    split_point = int(len(y) * 0.9) # <-- "VÁ" LỖI 80% THÀNH 90%
    # ===================
    
    X_train, X_test = X[:split_point], X[split_point:]
    y_train, y_test = y[:split_point], y[split_point:]

    logging.info(f"Train set: {X_train.shape[0]} samples")
    logging.info(f"Test set: {X_test.shape[0]} samples")

    # --- (SỬA V1: "CÂN BẰNG" LÒ BẰNG WEIGHTED LOSS) ---
    logging.info("Đang 'độ' 'Trọng Số Phạt' (Weighted Loss) để 'cân' 'lò'...")
    try:
        class_weights_np = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weights = torch.tensor(class_weights_np, dtype=torch.float32).to(device)
        logging.info(f"Đã 'độ' Trọng Số Phạt (Weights): {class_weights_np}")
        # (Hàm "Phạt")
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    except Exception as e:
        logging.error(f"Lỗi 'độ' Trọng Số Phạt: {e}. 'Lò' này 'luyện' sẽ 'phế' (như V4).")
        loss_fn = nn.CrossEntropyLoss() # (Chạy "chay")
    # --- (HẾT SỬA V1) ---

    # "Đút" data vào "khay" (Tensors)
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE * 2, shuffle=False)
    
    # "Dựng Lò"
    model = TransformerModel(
        num_features=NUM_FEATURES,
        d_model=D_MODEL,
        nhead=N_HEAD,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    logging.info(f"--- BẮT ĐẦU 'LUYỆN' NÃ0 TRANSFORMER (V2 - {EPOCHS} Epochs) ---")
    
    best_loss = float('inf')
    patience_counter = 0
    
    try:
        for epoch in range(EPOCHS):
            # --- (A) "LUYỆN" (TRAIN) ---
            model.train()
            total_train_loss = 0
            
            pbar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False)
            for X_batch, y_batch in pbar_train:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                optimizer.zero_grad()
                y_pred = model(X_batch) # "Phán"
                loss = loss_fn(y_pred, y_batch) # "Chấm điểm" (Cân Bằng)
                
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()
                pbar_train.set_postfix(loss=loss.item())
                
            avg_train_loss = total_train_loss / len(train_loader)
            
            # --- (B) "SOI" (VALIDATION) ---
            model.eval()
            total_val_loss = 0
            all_preds, all_labels = [], []
            
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    y_pred = model(X_batch)
                    loss = loss_fn(y_pred, y_batch)
                    total_val_loss += loss.item()
                    
                    all_preds.append(torch.argmax(y_pred, dim=1).cpu())
                    all_labels.append(y_batch.cpu())
                    
            avg_val_loss = total_val_loss / len(test_loader)
            
            # (Tính "điểm" Accuracy)
            all_preds = torch.cat(all_preds).numpy()
            all_labels = torch.cat(all_labels).numpy()
            accuracy = accuracy_score(all_labels, all_preds)
            
            logging.info(f"Epoch {epoch+1:03d} | Train Loss: {avg_train_loss:.4f} | **Val Loss: {avg_val_loss:.4f}** | Val Acc: {accuracy*100:.2f}%")
            
            # --- (C) "DỪNG SỚM" (Early Stopping) ---
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                patience_counter = 0
                # "Lưu" "não xịn" nhất
                torch.save(model.state_dict(), os.path.join(MODEL_DIR, MODEL_NAME))
                logging.info(f"-> 'Não Xịn' Mới (Val Loss: {best_loss:.4f}) -> {MODEL_NAME}")
            else:
                patience_counter += 1
                
            if patience_counter >= PATIENCE:
                logging.info(f"Dừng sớm (EarlyStopping) tại Epoch {epoch+1}.")
                break
                
    except KeyboardInterrupt:
        logging.warning("\nĐã bắt được (Ctrl+C)! Dừng 'luyện'...")

    # --- (D) "CHẤM ĐIỂM" (METRICS) ---
    logging.info("Đang 'chấm điểm' (Metrics) 'Não Xịn' nhất trên 'sân' Test...")
    
    try:
        # "Hồi sinh" "não xịn" nhất
        model.load_state_dict(torch.load(os.path.join(MODEL_DIR, MODEL_NAME)))
        model.eval()
        
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                all_preds.append(torch.argmax(y_pred, dim=1).cpu())
                all_labels.append(y_batch.cpu())
                
        y_pred = torch.cat(all_preds).numpy()
        y_test = torch.cat(all_labels).numpy()
        
        accuracy = accuracy_score(y_test, y_pred)
        target_names = ['0_Sideways', '1_Len_Ngon', '2_Xuong_SML']
        report = classification_report(y_test, y_pred, target_names=target_names)
        
        logging.info(f"\n--- KẾT QUẢ 'CHẤM ĐIỂM' (TRANSFORMER V2 - H1 'Cân Bằng' 90/10) ---")
        logging.info(f"Độ Chính Xác (Accuracy) (Tổng): {accuracy * 100:.2f}%")
        logging.info(f"Báo Cáo Chi Tiết:\n{report}")
        logging.info(f"-------------------------------------------")

        logging.info(f"Đã 'đúc' (lưu) 'Não Xịn' V2 của Thầy Transformer vào: {os.path.join(MODEL_DIR, MODEL_NAME)}")
        
    except FileNotFoundError:
        logging.error(f"LỖI: Không tìm thấy 'Não Xịn' ({MODEL_NAME}) để 'chấm điểm'.")
        logging.error("Có vẻ 'lò' đã 'dừng' trước khi 'lưu' được Epoch 'xịn' nào.")
    except Exception as e:
        logging.error(f"Lỗi 'chấm điểm': {e}")


# === (HÀM MAIN "CÔNG XƯỞNG") ===
if __name__ == "__main__":
    logging.info(f"=== BẮT ĐẦU 'LÒ ĐÚC' THẦY 3: TRANSFORMER (Não V2 'Vá Lỗi' T=49) ===")
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(DIR_PROCESSED, exist_ok=True) 
    
    # 1. "NẠP THỨC ĂN" VÀ "CẮT CHUỖI" (V2)
    X, y = load_data_and_prep_V2(FEATURE_FILE_PATH, LOOKBACK, LOOKFORWARD, THRESHOLD)
    
    if X is None:
        logging.error("Dừng 'Lò' do không có 'thức ăn'.")
        sys.exit(1)
        
    # 2. "LUYỆN"
    train_transformer_v2(X, y)
    
    logging.info(f"\n{'='*70}\n === HOÀN TẤT 'LÒ ĐÚC' THẦY 3: TRANSFORMER (V2) ===\n{'='*70}")