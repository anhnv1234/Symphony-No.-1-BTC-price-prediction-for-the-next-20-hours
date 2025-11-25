# 🚀 CỖ MÁY IN TIỀN CHẠY BẰNG CƠM & AI: TRADING BOT V53z 🚀

![Badge](https://img.shields.io/badge/Độ_Uy_Tín-Vô_Cực-red) ![Badge](https://img.shields.io/badge/Tâm_Linh-Có_Thờ_Có_Thiêng-yellow) ![Badge](https://img.shields.io/badge/Tech-Deep_Learning_Tối_Thượng-blue)

## 1. LỜI NÓI ĐẦU (INTRO)
Chào mừng các đồng nghiện đến với **Generative Time-Series Forecasting Engine** (nghe tên Tây cho nó sang mồm thôi, chứ nó là con Bot soi cầu Bitcoin).

Khác với mấy con bot "lùa gà" ngoài kia chỉ biết phán Xanh/Đỏ (Tài/Xỉu), con hàng V53z này là một **Họa Sĩ Thực Thụ**. Nó không đoán mò, nó **vẽ ra đường chạy của giá (Trajectory)** trong 20 giờ tới.
Tại sao phải đoán giá đóng nến khi bạn có thể vẽ cả cái chart? 😎

### 📸 Ảnh minh họa cho anh em trầm trồ
*(Nhìn cái đường nó vẽ kìa, uy tín chưa?)*
<img width="1911" height="772" alt="image" src="https://github.com/user-attachments/assets/8e14f96a-ee5b-45d4-91c1-56821c69034f" />

### 📉 Thực chiến (Real-time)
*(Chạy mượt như Sunsilk)*
<img width="1273" height="617" alt="image" src="https://github.com/user-attachments/assets/c8177350-2f4e-417a-8bf7-2411d7e3e4dc" />

---

## 2. 🧠 BỘ NÃO QUÁI VẬT: "TAM ĐẠI DANH BỔ"

Hệ thống này không dùng 1 não (vì sợ cô đơn), mà dùng **3 bộ não** đấm nhau liên tục để tìm ra chân lý:

### A. Não 1: CVAE-LSTM (Thanh Niên Nghiêm Túc)
* **Biệt danh:** *The Stabilizer (Kẻ Ổn Định)*.
* **Tính cách:** Ăn chắc mặc bền, sợ rủi ro. Chuyên trị những lúc thị trường đi ngang (Sideway) buồn ngủ.
* **Vũ khí:** Lai tạo giữa **CVAE** (nén dữ liệu) và **LSTM** (trí nhớ dai như người yêu cũ).
* **Tác dụng:** Giữ cho con bot không bị "ngáo" giá.

### B. Não 2: TimeGAN (Nghệ Sĩ Nhân Dân)
* **Biệt danh:** *The Artist (Kẻ Phá Cách)*.
* **Tính cách:** Bay bổng, thích cảm giác mạnh. Chuyên trị những cú "Kill Long Diệt Short" biến động mạnh.
* **Vũ khí:** **GAN** (Mạng đối nghịch). Hai thằng AI tự đấm nhau để học cách lừa người dùng bằng dữ liệu giả giống y như thật.
* **Tác dụng:** Bắt sóng cực nhanh. Nhưng vì hay "bay" quá nên phải gắn thêm cái "Giảm xóc" (Damping) cho nó bớt ảo.

### C. Não 3: TCVAE (Giáo Sư Biết Tuốt)
* **Biệt danh:** *The Visionary (Kẻ Nhìn Xa)*.
* **Tính cách:** Thâm sâu khó lường.
* **Vũ khí:** **Transformer** (Công nghệ lõi của ChatGPT) kết hợp CVAE.
* **Tác dụng:** Soi cùng lúc 53 chỉ báo, nhìn thấy những thứ mà mắt thường (và mấy con bot ghẻ) không thấy được.

---

## 3. 📂 ĐỒ NGHỀ & ĐỆ TỬ (FILE STRUCTURE)

Code gọn nhẹ, chỉ giữ lại những thằng làm được việc:

| Chức vụ | Tên File | Mô tả công việc |
| :--- | :--- | :--- |
| **TRÙM CUỐI** | **`05_live_bot_V53_ALL.py`** | **TỔNG QUẢN:** Điều phối đàn em, chạy real-time, vẽ vời, hậu kỳ. Nói chung là thằng to đầu nhất. |
| **OSIN CAO CẤP** | **`data_service.py`** | **CULI DỮ LIỆU:** Chuyên đi bốc vác nến từ Binance, số liệu vĩ mô FRED, On-chain... Vá víu dữ liệu hỏng, tính toán 53 chỉ báo. |
| **LÒ LUYỆN ĐAN** | **`03_train_cvae_V14...py`** | **HUẤN LUYỆN NÃO 1:** Nơi tu luyện CVAE-LSTM. Đặc biệt sản xuất ra cái `Scaler` (máy ép dữ liệu) dùng chung cho cả hội. |
| **LÒ BÁT QUÁI** | **`04_train_transformer...py`** | **HUẤN LUYỆN NÃO 3:** Nơi tu luyện con quái vật TCVAE. |

---

## 4. 🔄 QUY TRÌNH "LUYỆN TỊCH TÀ KIẾM PHỔ"

Muốn bot chạy ngon thì phải làm đúng quy trình, sai một ly đi một dặm (ra đảo):

### GIAI ĐOẠN 1: ĐI CHỢ (DATA PREP)
1.  Sai thằng `data_service` đi tải nến Binance (W1, D1, H1, M15).
2.  Tải tiếp dữ liệu Bitstamp từ thời đồ đá (2013).
3.  Gộp hết lại, nêm nếm gia vị (Indicators) -> Ra nồi lẩu thập cẩm `btcusdt_master_data.parquet`.

### GIAI ĐOẠN 2: TU LUYỆN (TRAINING)
*Bước này tốn điện + tốn GPU.*
1.  Chạy file `03` để luyện Não 1. Nhớ giữ kỹ cái **`cvae_scaler_V23.gz`** (mất cái này là ăn cám).
2.  Chạy file `04` để luyện Não 3.

### GIAI ĐOẠN 3: RA TRẬN (LIVE RUNTIME)
*Bật file `05_live_bot_V53_ALL.py` lên và khấn.*

**Vòng lặp mỗi giờ của Bot:**
1.  **Vá Nóng (Hot Patching):** Tải ngay 20 nến mới nhất đắp vào dữ liệu cũ (chống lag).
2.  **Trám Lỗ (Gap Filling):** Chỗ nào thiếu nến thì tự bịa... à nhầm, tự tính toán điền vào cho đẹp.
3.  **Ép Khuôn (Scaling):** Nén hết dữ liệu về dạng [0, 1] cho AI nó dễ nuốt.
4.  **Mơ (Generation):** 3 thằng Não chụm đầu vào "mơ" về tương lai.
5.  **Hậu Kỳ (Make-up):**
    * **Neo Giá:** Lấy % dự báo ốp vào giá hiện tại (91k).
    * **Giảm Xóc:** Tát cho thằng TimeGAN tỉnh lại (giảm biên độ 5%) kẻo nó hưng phấn quá.
    * **Làm Mượt:** Vuốt lại đường giá cho nuột nà (EMA).
6.  **Vẽ Tranh:** Xuất ra cái ảnh `live_prediction_chart` đẹp như mơ.

---

## 5. 📥 ĐẦU VÀO & 📤 ĐẦU RA

### ĂN GÌ? (INPUT)
Nó ăn tạp lắm, nuốt **53 loại dữ liệu** khác nhau:
* Giá nến OHLCV (từ nến Giờ đến nến Tuần).
* RSI, MACD, Bollinger Bands... (đủ món ăn chơi).
* Lãi suất FED, CPI (mấy cái tin vĩ mô làm sập thị trường).
* Smart Money Concept (Vết chân cá mập).

### Ị RA GÌ? (OUTPUT)
Một file ảnh `live_prediction_chart_V53_ALL.png` chứa đựng tinh hoa vũ trụ:
* **Đường Chính:** 3 kịch bản giá (Xanh, Đỏ, Tím Vàng gì đó) cho 20h tới.
* **Volume Overlay:** Volume đè lên nến, nhìn rất chuyên nghiệp.
* **Quá Khứ Tương Đồng:** Nó lôi lại 3 đoạn lịch sử giống hệt hiện tại để anh em tham khảo (History repeats itself mà lị).

---

## 6. CÔNG NGHỆ ĐỘC QUYỀN (V53z Features)

* **Force Align (Ép Cột):** Dữ liệu thiếu cột? Kệ, bố mày tự thêm vào cho đủ, miễn là chạy được. Chống crash app cực mạnh.
* **No Gap Fix:** Cắt đuôi thừa, đắp đầu thiếu. Đảm bảo chart liền mạch không bị gãy khúc như răng bà lão.
* **Twinx Volume:** Vẽ Volume chồng lên giá nhưng ép xuống tỉ lệ 1/4 đáy màn hình. Đỉnh cao hiển thị (TradingView gọi bằng cụ).

---

> **⛔ CẢNH BÁO QUAN TRỌNG:**
>
> 1.  Hàng này tôi tự train, tốn bao nhiêu tiền điện nên **KHÔNG SHARE MODEL (Weights)** đâu, đừng xin mất công.
> 2.  Ai muốn hợp tác làm giàu, hoặc donate tiền cà phê thì liên hệ qua mail bên dưới.
>
> 📧 Email chính chủ: **nguyenvietaanh@gmail.com**
>
> *"Dùng Bot thì phải tin Bot, còn không tin thì... tự đi mà đánh tay!"*
