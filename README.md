# Symphony-No.-1-BTC-price-prediction-for-the-next-20-hours
Hệ thống này không phải là bot "dự đoán giá lên/xuống" đơn thuần (Classification), mà là bot "Vẽ Tranh Tương Lai" (Generative Time-Series Forecasting). Nó cố gắng hình dung ra đường đi của giá trong 20 giờ tới sẽ trông như thế nào.
**1.  Đầu Vào (Thức Ăn - Input)**
    Hệ thống "ăn" một lượng dữ liệu khổng lồ để hiểu bối cảnh thị trường:
    Nguồn: Binance (W1, D1, H1, M15), Bitstamp (lịch sử xa xưa), FRED (Lịch kinh tế vĩ mô Mỹ).
    Số lượng món: 53 đặc trưng (features) bao gồm:
    Giá (OHLCV).
    Chỉ báo kỹ thuật (RSI, MACD, Bollinger Bands, SMA, EMA...).
    Dữ liệu vĩ mô (Lãi suất FED, Lạm phát CPI, Bảng cân đối kế toán...).
    Hành vi thông minh (Smart Money Concepts - FVG).
    Khẩu phần: Nó nhìn lại quá khứ 50 nến (ngắn hạn) hoặc 168 nến (dài hạn/1 tuần) để lấy đà dự đoán.
**2. Bộ Não Trung Tâm (The Core Models)**
    Thay vì tin vào một ông thầy bói, Bot dùng cơ chế "Hội Đồng Tham Mưu" gồm 3 mô hình AI khác nhau cùng đưa ra ý kiến:
    CVAE-LSTM (Màu Xanh Dương):
    Công nghệ: Kết hợp Autoencoder (nén dữ liệu) và LSTM (nhớ chuỗi dài).
    Tính cách: Thận trọng, ổn định. Thường dự báo theo xu hướng chính, ít khi "phán bừa".
    TimeGAN (Màu Xanh Lá):
    Công nghệ: Generative Adversarial Networks (Mạng đối nghịch - Hai thằng AI tự đấu với nhau để học cách tạo dữ liệu giả giống thật nhất).
    Tính cách: Nghệ sĩ, bay bổng. Thường nắm bắt tốt các biến động mạnh (volatility) nhưng hay bị "ngáo" (quá đà) nên cần phải lắp "giảm xóc" (Damping) và "làm mượt" (Smoothing).
    TCVAE - Transformer CVAE (Màu Đỏ):
    Công nghệ: Dùng kiến trúc Transformer (giống ChatGPT) kết hợp CVAE. Dùng cơ chế "Sự chú ý" (Attention) để lọc nhiễu.
    Tính cách: Thông thái, nhìn xa trông rộng. Giỏi phát hiện các mối liên hệ phức tạp mà LSTM có thể bỏ sót.
**3. Cơ Chế Vận Hành (The Workflow)**
    Quy trình từ lúc bật Bot đến lúc ra ảnh:
    Hút & Nấu (Data Pipeline): Tải nến mới nhất -> Vá nóng (Hot Fix) để lấp Gap -> Tính toán 53 chỉ báo -> Ép khuôn (Scale) về dạng số từ 0 đến 1.
    Mơ (Generation):
    Dữ liệu được ném vào 3 cái não.
    Mỗi não sẽ sinh ra một véc-tơ ngẫu nhiên (Z-space) rồi giải mã nó thành 20 cây nến tương lai.
    Hậu Kỳ (Post-Processing - Rất quan trọng):
    Relative Projection: Không quan tâm não đoán giá bao nhiêu (20k hay 100k), chỉ quan tâm nó đoán tăng/giảm bao nhiêu %.
    Neo Giá (Anchoring): Áp tỷ lệ % đó vào giá hiện tại (ví dụ 91k) để đường dự báo nối liền mạch với nến hiện tại, không bị gãy khúc.
    Giảm Xóc (Damping): Riêng TimeGAN bị ép giảm biên độ dao động để bớt "ảo".
    Soi Gương (Pattern Matching - Stumpy):
    Bot dùng thư viện stumpy để quét toàn bộ lịch sử BTC từ năm 2019.
    Nó tìm ra 3 giai đoạn trong quá khứ có đường đi của giá giống hệt hiện tại nhất (Top 3 Similar Patterns).
    Mục đích: "Lấy sử làm gương". Nếu quá khứ nó sập, thì hiện tại cũng nên cẩn thận.
**4. Đầu Ra (Output)**
    Đại ca nhận được một tấm ảnh "Siêu Cấp" gồm:
    Biểu đồ Chính: Nến hiện tại + 3 đường kịch bản dự báo (Xanh/Đỏ/Lá) + Volume chồng lên (Overlay).
**3 Biểu đồ Phụ: 3 giai đoạn lịch sử giống nhất để tham khảo.**
    Tóm lại: Đây là một cỗ máy tổng hợp trí tuệ nhân tạo và xác suất thống kê để vẽ ra các kịch bản tương lai khả thi nhất dựa trên đa góc nhìn.
