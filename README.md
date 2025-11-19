```markdown
# Symphony No.1 — Dự đoán giá BTC cho 20 giờ tiếp theo (Generative Time-Series)

Hệ thống này không phải là bot "dự đoán lên/xuống" đơn thuần (classification). Đây là một "Máy Vẽ Tương Lai" — một hệ thống generative time-series forecasting, cố gắng sinh ra các kịch bản giá khả thi cho 20 cây nến tiếp theo dựa trên nhiều góc nhìn.

---

## 1. Đầu vào (Input)
Hệ thống "ăn" một lượng dữ liệu lớn để hiểu bối cảnh thị trường:

- Nguồn dữ liệu:
  - Binance (W1, D1, H1, M15)
  - Bitstamp (lịch sử xa)
  - FRED (dữ liệu kinh tế vĩ mô của Mỹ)
- Tổng quan đặc trưng:
  - 53 features bao gồm:
    - Giá (OHLCV)
    - Chỉ báo kỹ thuật: RSI, MACD, Bollinger Bands, SMA, EMA, ...
    - Dữ liệu vĩ mô: lãi suất FED, CPI, bảng cân đối, ...
    - Hành vi thông minh: Smart Money Concepts (FVG), ...
- Khung nhìn lịch sử (lookback):
  - Ngắn hạn: 50 nến
  - Dài hạn / 1 tuần: 168 nến

---

## 2. Bộ não trung tâm (Core Models — "Hội đồng tham mưu")
Ba mô hình AI vận hành đồng thời, mỗi mô hình có phong cách và điểm mạnh riêng:

1. CVAE-LSTM (Màu Xanh Dương)
   - Công nghệ: CVAE kết hợp LSTM
   - Tính cách: Thận trọng, ổn định — thường dự báo theo xu hướng chính, ít nhiễu.

2. TimeGAN (Màu Xanh Lá)
   - Công nghệ: Generative Adversarial Networks cho chuỗi thời gian
   - Tính cách: Nghệ sĩ, nắm bắt tốt biến động mạnh (volatility) nhưng có xu hướng quá đà => cần cơ chế giảm xóc (damping).

3. TCVAE — Transformer + CVAE (Màu Đỏ)
   - Công nghệ: Kiến trúc Transformer kết hợp CVAE, dùng cơ chế Attention
   - Tính cách: Nhìn xa trông rộng, phát hiện mối liên hệ phức tạp mà LSTM có thể bỏ sót.

---

## 3. Cơ chế vận hành (Workflow)

1. Tiền xử lý (Data pipeline)
   - Tải nến mới nhất
   - Vá gap (hot fix)
   - Tính 53 chỉ báo
   - Scale dữ liệu (chuẩn hoá về 0–1)

2. Sinh kịch bản (Generation)
   - Dữ liệu đầu vào được đưa vào 3 mô hình.
   - Mỗi mô hình sinh một vector ngẫu nhiên (z-space) và giải mã thành 20 cây nến tương lai.

3. Hậu kỳ (Post-processing)
   - Relative projection: Chỉ quan tâm đến tỷ lệ thay đổi (%) mà mô hình dự đoán, không quan tâm mức giá tuyệt đối.
   - Anchoring (Neo giá): Áp tỷ lệ phần trăm dự báo vào giá hiện tại để đường dự báo liền mạch với nến hiện tại.
   - Damping (Giảm xóc): Giảm biên độ dự báo của TimeGAN để tránh dự báo quá cực đoan.
   - Pattern matching (Soi gương): Dùng thư viện stumpy để quét lịch sử BTC từ 2019, tìm 3 giai đoạn có đường giá tương tự (Top-3 similar patterns) để tham khảo — "lấy sử làm gương".

---

## 4. Đầu ra (Output)
Người dùng nhận được một tấm ảnh/biểu đồ tổng hợp gồm:

- Biểu đồ chính:
  - Nến hiện tại
  - 3 đường kịch bản dự báo (Xanh dương — CVAE-LSTM, Xanh lá — TimeGAN, Đỏ — TCVAE)
  - Volume hiển thị overlay
- 3 biểu đồ phụ:
  - 3 giai đoạn lịch sử giống nhất để tham khảo

Tóm lại: Hệ thống là sự tổng hợp giữa mô hình generative AI và thống kê để vẽ ra các kịch bản tương lai khả thi từ nhiều góc nhìn khác nhau.

---

## Ghi chú
- Mục tiêu của hệ thống là tạo kịch bản khả thi, không phải đưa ra lời khuyên giao dịch chắc chắn.
- Kết quả là probabilistic — luôn có rủi ro; hãy sử dụng kết hợp quản trị rủi ro và phán đoán của con người.
```
