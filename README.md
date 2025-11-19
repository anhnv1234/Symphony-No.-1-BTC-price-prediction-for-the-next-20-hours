# 🏗️ KIẾN TRÚC HỆ THỐNG: GENERATIVE AI TRADING BOT (V53z)

## 1. TỔNG QUAN DỰ ÁN
Hệ thống là một **Cỗ máy Dự báo Chuỗi Thời gian Tạo sinh (Generative Time-Series Forecasting Engine)** dành cho Bitcoin (BTC).
Khác với các bot truyền thống chỉ dự báo xu hướng (Lên/Xuống), hệ thống này **vẽ ra kịch bản đường giá** (Price Trajectory) cho 20 giờ tiếp theo dựa trên sự đồng thuận của 3 mô hình Deep Learning tiên tiến.
<img width="1911" height="772" alt="image" src="https://github.com/user-attachments/assets/8e14f96a-ee5b-45d4-91c1-56821c69034f" />
---

## 2. 🧠 TRÁI TIM HỆ THỐNG: "TAM ĐẠI CỐT LÕI" (THE THREE BRAINS)

Hệ thống hoạt động như một **"Hội Đồng Tham Mưu"**, nơi 3 bộ não với kiến trúc khác biệt cùng phân tích dữ liệu:

### A. Não 1: CVAE-LSTM (The Stabilizer - Kẻ Ổn Định)
* **Vai trò:** "Mỏ neo" tâm lý, giữ cho dự báo bám sát xu hướng chính.
* **Công nghệ:** Kết hợp **CVAE** (Conditional Variational Autoencoder) để nén dữ liệu thành xác suất và **LSTM** (Long Short-Term Memory) để ghi nhớ chuỗi thời gian.
* **Đặc điểm:** Dự báo mượt mà, ít nhiễu, độ tin cậy cao trong thị trường đi ngang (Sideway).

### B. Não 2: TimeGAN (The Artist - Kẻ Phá Cách)
* **Vai trò:** "Cảm nhận" nhịp điệu và xung lực thị trường.
* **Công nghệ:** **GAN** (Generative Adversarial Networks - Mạng đối nghịch). Hai mạng con (Generator & Discriminator) đấu nhau để học cách tạo ra dữ liệu giả giống thật nhất.
* **Đặc điểm:** Rất nhạy với biến động mạnh (Volatility). Tuy nhiên, do hay "phóng đại" nên cần cơ chế "Giảm Xóc" (Damping) và "Làm Mượt" (Smoothing).

### C. Não 3: TCVAE (Transformer CVAE - The Visionary - Kẻ Nhìn Xa)
* **Vai trò:** Phát hiện các mối liên hệ phức tạp và dài hạn.
* **Công nghệ:** Áp dụng kiến trúc **Transformer** (cơ chế Self-Attention giống ChatGPT) kết hợp CVAE.
* **Đặc điểm:** Có khả năng nhìn toàn cảnh bức tranh thị trường (53 chỉ báo) cùng lúc, phát hiện ra các cấu trúc giá mà LSTM có thể bỏ sót.

---

## 3. 📂 CẤU TRÚC FILE & CHỨC NĂNG

Hệ thống được tối ưu hóa chỉ còn 4 file code chính cần quản lý:

| Nhóm | Tên File | Chức năng Chi tiết |
| :--- | :--- | :--- |
| **SỐNG CÒN** | **`05_live_bot_V53_ALL.py`** | **TRÙM CUỐI (Main Execution):**<br>- Điều phối toàn bộ hoạt động.<br>- Chạy vòng lặp thời gian thực (Real-time Loop).<br>- Thực hiện hậu kỳ (Post-processing) và vẽ biểu đồ. |
| **HẬU CẦN** | **`data_service.py`** | **QUẢN LÝ DỮ LIỆU:**<br>- Hút nến từ Binance (W1, D1, H1, M15).<br>- Hút dữ liệu vĩ mô (FRED) & On-chain (Bitstamp).<br>- **Hot Fix:** Cập nhật nóng 20 nến mới nhất.<br>- Tính toán 53 chỉ báo kỹ thuật. |
| **LÒ LUYỆN** | **`03_train_cvae_V14_H1_ONLY.py`** | **TẠO NÃO 1 & SCALER:**<br>- Huấn luyện CVAE-LSTM.<br>- **Quan trọng:** Tạo ra file `cvae_scaler_V23.gz` (Máy ép dữ liệu dùng chung). |
| **LÒ LUYỆN** | **`04_train_transformer_cvae_V1.py`** | **TẠO NÃO 3:**<br>- Huấn luyện mô hình TCVAE. |

---

## 4. 🔄 DÒNG CHẢY DỮ LIỆU (DATA FLOW) & QUY TRÌNH VẬN HÀNH

Để chạy hệ thống từ con số 0, thực hiện theo đúng thứ tự sau:

### GIAI ĐOẠN 1: CHUẨN BỊ DỮ LIỆU (DATA PREP)
1.  **Chạy `data_service.py` (Mode 1):** Tải lịch sử nến Binance (4 khung thời gian).
2.  **Chạy `data_service.py` (Mode 1.5):** Tải lịch sử Bitstamp (từ 2013).
3.  **Chạy `data_service.py` (Mode 2):** Gộp tất cả, tính toán chỉ báo -> Tạo ra file `02_Master_Data/btcusdt_master_data.parquet`.

### GIAI ĐOẠN 2: HUẤN LUYỆN (TRAINING)
*Bước này tạo ra "Trí Khôn" cho Bot.*
1.  **Chạy `03_train_cvae_V14...py`:**
    * Input: Master Data.
    * Output: `cvae_decoder_V11...pth` (Model) + **`cvae_scaler_V23.gz`** (Scaler).
2.  **Chạy `04_train_transformer...py`:**
    * Input: Master Data + Scaler V23.
    * Output: `transformer_cvae_decoder_V13...pth`.

### GIAI ĐOẠN 3: VẬN HÀNH LIVE (RUNTIME)
*Chạy `05_live_bot_V53_ALL.py`.*

**Quy trình xử lý mỗi giờ:**
1.  **Hot Patching (Vá Nóng):** Tải ngay 20 nến H1 mới nhất từ sàn, ghi đè vào dữ liệu cũ để triệt tiêu độ trễ.
2.  **Re-Build Master:** Tính toán lại các chỉ báo cho dữ liệu mới nhất.
3.  **Gap Filling:** Tự động phát hiện và trám các khoảng trống thời gian (nến thiếu).
4.  **Scaling (Ép Khuôn):** Dùng `cvae_scaler_V23` ép dữ liệu về khoảng [0, 1].
5.  **Generation (Mơ):** 3 Não (CVAE, TimeGAN, TCVAE) sinh ra kịch bản tương lai (dạng số nén).
6.  **Post-Processing (Hậu Kỳ - *Cực quan trọng*):**
    * **Relative Projection:** Chuyển đổi giá dự báo thành % tăng trưởng.
    * **Anchoring (Neo Giá):** Áp % tăng trưởng vào giá hiện tại (91k) để nối liền mạch.
    * **Damping (Giảm Xóc):** Giảm biên độ dao động của TimeGAN xuống 5% để bớt "ảo".
    * **Smoothing:** Làm mượt đường đi bằng EMA.
7.  **Pattern Matching:** Dùng `stumpy` quét quá khứ tìm 3 giai đoạn tương đồng nhất (tránh trùng lặp).
8.  **Visualization:** Vẽ biểu đồ TradingView (Nến + Volume Overlay) ra file ảnh.

---

## 5. 📥 ĐẦU VÀO & 📤 ĐẦU RA

### DỮ LIỆU ĐẦU VÀO (INPUT)
Hệ thống tiêu thụ **53 đặc trưng (features)** để hiểu thị trường:
* **Giá & Volume:** Open, High, Low, Close, Volume (H1, M15, D1, W1).
* **Chỉ báo kỹ thuật:** RSI, MACD, Bollinger Bands, SMA, EMA, Volatility...
* **Vĩ mô (Macro):** Lãi suất FED, CPI, Bảng cân đối kế toán (từ FRED).
* **Smart Money Concepts:** FVG (Fair Value Gaps - Vùng mất cân bằng giá).

### KẾT QUẢ ĐẦU RA (OUTPUT)
File ảnh: `live_prediction_chart_V53_ALL.png`
* **Biểu đồ Chính:**
    * Nến thực tế hiện tại.
    * 3 Đường kịch bản dự báo (Xanh Dương, Đỏ, Xanh Lá) đã được neo giá và làm mượt.
    * Volume hiển thị dạng Overlay (chồng lên nến) ở đáy biểu đồ.
* **3 Biểu đồ Phụ:**
    * Hiển thị 3 giai đoạn lịch sử có đường giá (H1 Close) giống hiện tại nhất.
    * Có kèm điểm số tương đồng (Score - càng thấp càng giống).

---

## 6. CÁC CƠ CHẾ ĐẶC BIỆT (V53z)

* **Force Align (Ép Cột):** Tự động thêm các cột thiếu (vĩ mô) vào dữ liệu nến mới để khớp với khuôn mẫu của Scaler cũ -> Chống lỗi `sklearn ValueError`.
* **No Gap Fix:** Tự động cắt bỏ phần dữ liệu thừa ở đuôi và trám các nến thiếu -> Biểu đồ liền mạch, không bị đứt đoạn giữa quá khứ và tương lai.
* **Overlay Volume:** Hiển thị Volume ngay trên biểu đồ giá bằng trục tung kép (`twinx`), ép tỉ lệ 1/4 để không che khuất nến.

## Vì một số lý do tôi kg thể gửi được các tệp đã train sẵn, nếu bạn cần bạn có thể liên hệ qua với tôi qua email nguyenvietaanh@gmail.com

##
