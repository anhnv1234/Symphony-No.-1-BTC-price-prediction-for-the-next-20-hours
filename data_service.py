import requests
import pandas as pd
import numpy as np
import time
import os
import logging
from tqdm import tqdm
import datetime
import pyarrow # Cần thiết cho Parquet
import sys

# (BẮT BUỘC) Kiểm tra thư viện "Hút" Vĩ Mô
try:
    import pandas_datareader.data as web
except ImportError:
    logging.error("LỖI: Thiếu thư viện 'pandas-datareader' để hút dữ liệu FRED.")
    logging.error("Đại ca vui lòng chạy lệnh: pip install pandas-datareader")
    sys.exit(1)


# --- Cấu hình logging ---
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - [MasterData_V23_idxmaxFix] - %(message)s')

class MasterDataServiceV23: # Đổi tên V22 -> V23
    
    # --- Dữ liệu Halving từ FEDhistory.py ---
    HALVING_DATES = [
        pd.to_datetime('2020-05-11'),
        pd.to_datetime('2024-04-19'),
        pd.to_datetime('2028-04-15'), # Ước tính
        pd.to_datetime('2032-04-10')  # Ước tính
    ]
    
    def __init__(self, symbol='BTCUSDT'):
        # (SỬA V16) Chuẩn hóa symbol (BTCUSDT -> btcusd cho Bitstamp)
        self.symbol_binance = symbol
        self.symbol_bitstamp = "btcusd" # API Bitstamp dùng "btcusd"
        
        self.BINANCE_BASE_URL = "https://fapi.binance.com/fapi/v1/klines"
        # (MỚI V16) API Bitstamp
        self.BITSTAMP_BASE_URL = f"https://www.bitstamp.net/api/v2/ohlc/{self.symbol_bitstamp}/"
        
        self.API_LIMIT = 1500
        
        # MỐC "KHỞI THỦY" (DÙNG LÀM MỐC "CẮT")
        self.START_DATE_STR = '2019-11-01T00:00:00Z'
        
        # --- (ĐỘ) Tổ chức lại thư mục ---
        self.DIR_RAW = "00_Raw_Data"
        self.DIR_PROCESSED = "01_Processed_Data"
        self.DIR_MASTER = "02_Master_Data"
        
        # Tạo thư mục nếu chưa có
        os.makedirs(self.DIR_RAW, exist_ok=True)
        os.makedirs(self.DIR_PROCESSED, exist_ok=True)
        os.makedirs(self.DIR_MASTER, exist_ok=True)
        
        # --- Tên các file dữ liệu Nến ---
        self.file_m15 = os.path.join(self.DIR_RAW, f"{self.symbol_binance.lower()}_m15_raw.parquet")
        self.file_h1 = os.path.join(self.DIR_RAW, f"{self.symbol_binance.lower()}_h1_raw.parquet")
        self.file_d1 = os.path.join(self.DIR_RAW, f"{self.symbol_binance.lower()}_d1_raw.parquet")
        self.file_w1 = os.path.join(self.DIR_RAW, f"{self.symbol_binance.lower()}_w1_raw.parquet")
        
        # --- (MỚI V16) File "Mồi" Lịch Sử Dài Hạn (từ Bitstamp) ---
        self.file_bitstamp_daily = os.path.join(self.DIR_RAW, f"{self.symbol_bitstamp.lower()}_bitstamp_daily_raw.parquet")

        # --- (MỚI) File "đã nấu" (processed) từ Lịch Kinh Tế FRED ---
        self.file_fred_features = os.path.join(self.DIR_PROCESSED, "fred_economic_features.parquet")

        # File "Thành Phẩm"
        self.file_master = os.path.join(self.DIR_MASTER, f"{self.symbol_binance.lower()}_master_data.parquet")

    # --- CHỨC NĂNG HÚT DỮ LIỆU (NẾN) ---

    def _fetch_historical_forward(self, timeframe, since_timestamp):
        """Hút XUÔI (Nến) từ 'since_timestamp' cho đến hiện tại."""
        all_data = []
        pbar = tqdm(desc=f"Hấp Tinh (Nến - {timeframe})", unit=" nến")
        
        while True:
            try:
                params = {
                    'symbol': self.symbol_binance, # Dùng symbol Binance
                    'interval': timeframe, 
                    'limit': self.API_LIMIT,
                    'startTime': since_timestamp
                }
                
                response = requests.get(self.BINANCE_BASE_URL, params=params, timeout=10)
                response.raise_for_status() 
                data = response.json()

                if not data:
                    logging.info(f"Đã hút đến nến mới nhất ({timeframe}). Dừng lại.")
                    break
                
                all_data.extend(data)
                last_timestamp = data[-1][0]
                since_timestamp = last_timestamp + 1
                
                pbar.update(len(data))
                pbar.set_description(f"Hấp Tinh ({timeframe}) - Đã đến {pd.to_datetime(last_timestamp, unit='ms')}")
                time.sleep(0.1) 

            except Exception as e:
                logging.error(f"Lỗi khi hút nến ({timeframe}): {e}. Dừng lại.")
                pbar.close()
                return None
        
        pbar.close()
        logging.info(f"Hút thành công {len(all_data)} nến mới ({timeframe}).")
        return all_data

    def _process_and_save_parquet(self, raw_data, filename, is_resumable=False):
        """Xử lý và LƯU (hoặc NỐI) (Nến) vào file Parquet."""
        if not raw_data:
            logging.info(f"Không có dữ liệu (nến) mới để xử lý ({filename}).")
            return
        
        columns = [
            'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ]
        df_new = pd.DataFrame(raw_data, columns=columns)
        df_new = df_new[['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
        # (VÁ LỖI V10) Khởi tạo index Nến là UTC
        df_new['datetime'] = pd.to_datetime(df_new['timestamp'], unit='ms', utc=True) 
        
        df_new.set_index('datetime', inplace=True)
        df_new.drop('timestamp', axis=1, inplace=True)
        for col in df_new.columns:
            df_new[col] = pd.to_numeric(df_new[col])
        
        if is_resumable and os.path.exists(filename):
            logging.info(f"File {filename} đã tồn tại. Đang đọc và nối dữ liệu...")
            try:
                df_old = pd.read_parquet(filename)
                df_final = pd.concat([df_old, df_new])
                df_final = df_final[~df_final.index.duplicated(keep='last')]
            except Exception as e:
                logging.warning(f"Lỗi đọc file {filename}: {e}. Tạo file mới.")
                df_final = df_new
        else:
            df_final = df_new
            
        df_final.sort_index(inplace=True)
        df_final.to_parquet(filename, engine='pyarrow')
        logging.info(f"Đã lưu thành công {len(df_final)} dòng (nến) vào: {filename}")


    # --- CHẾ ĐỘ 1: "HẤP TINH NẾN" ---

    def run_download_klines(self):
        """
        Chế độ 1: "Hút Xuôi" 4 Khung Nến (W1, D1, H1, M15).
        """
        logging.info(f"--- BẮT ĐẦU TẢI DỮ LIỆU NẾN (4 KHUNG) ---")
        
        start_date_ms = int(pd.to_datetime(self.START_DATE_STR).timestamp() * 1000)

        # Hút Nến (4 file)
        for (timeframe, filename) in [('1w', self.file_w1),
                                     ('1d', self.file_d1), 
                                     ('1h', self.file_h1), 
                                     ('15m',self.file_m15)]:
            logging.info(f"Đang kiểm tra file nến {filename}...")
            since_ms = start_date_ms
            if os.path.exists(filename):
                try:
                    df_old = pd.read_parquet(filename)
                    if not df_old.empty:
                        last_saved_time = df_old.index.max()
                        since_ms = int(last_saved_time.timestamp() * 1000) + 1 
                        logging.info(f"File đã có dữ liệu. Hút tiếp nến từ: {last_saved_time}")
                except Exception as e:
                    logging.warning(f"Không đọc được file {filename}. Sẽ hút lại từ đầu. Lỗi: {e}")
            
            raw_data = self._fetch_historical_forward(timeframe, since_ms)
            self._process_and_save_parquet(raw_data, filename, is_resumable=True)

        logging.info("--- TẢI DỮ LIỆU NẾN 4 KHUNG HOÀN TẤT ---")
        logging.info(f"Dữ liệu đã được lưu vào thư mục: {self.DIR_RAW}")

    # --- (MỚI V16) CHẾ ĐỘ 1.5: "MỒI" DỮ LIỆU LỊCH SỬ TỪ BITSTAMP ---

    def run_fetch_bitstamp_backfill(self):
        """
        (RESUMABLE) Hút data pre-2019 từ Bitstamp.
        """
        logging.info(f"--- BẮT ĐẦU HÚT LỊCH SỬ DÀI HẠN (BITSTAMP) ---")
        
        # Mốc "Khởi Thủy" của Binance (sẽ không hút data sau mốc này)
        cutoff_timestamp = int(pd.to_datetime(self.START_DATE_STR, utc=True).timestamp())
        
        # Logic "Hút Tiếp" (Resumable)
        df_old = pd.DataFrame()
        if os.path.exists(self.file_bitstamp_daily):
            logging.info(f"Đã có file 'mồi' {self.file_bitstamp_daily}. Kiểm tra data...")
            df_old = pd.read_parquet(self.file_bitstamp_daily)
            if not df_old.empty:
                last_saved_time = df_old.index.max()
                start_timestamp = int(last_saved_time.timestamp()) + 86400 # +1 ngày (tính bằng giây)
                logging.info(f"Sẽ hút tiếp data Bitstamp từ: {last_saved_time + pd.Timedelta(days=1)}")
            else:
                start_timestamp = int(pd.to_datetime('2013-01-01', utc=True).timestamp())
        else:
            logging.info(f"Chưa có file 'mồi' Bitstamp. Bắt đầu hút từ 2013...")
            start_timestamp = int(pd.to_datetime('2013-01-01', utc=True).timestamp())

        # Kiểm tra xem đã "hút" đủ chưa
        if start_timestamp >= cutoff_timestamp:
            logging.info("File 'mồi' Bitstamp đã đầy đủ (đã hút đến mốc 2019). Không cần hút thêm.")
            return

        all_data = []
        pbar = tqdm(desc="Hút Bitstamp (pre-2019)", unit=" 1000 ngày")
        
        while True:
            try:
                # Bitstamp dùng step=86400 (1 ngày) và limit=1000
                params = {'step': 86400, 'start': start_timestamp, 'limit': 1000}
                response = requests.get(self.BITSTAMP_BASE_URL, params=params, timeout=20)
                response.raise_for_status()
                data_list = response.json()['data']['ohlc']
                
                # Nếu API trả về list rỗng
                if not data_list:
                    logging.info("Bitstamp không trả về data mới. Dừng lại.")
                    pbar.close()
                    break
                    
                # Lọc data: Chỉ lấy data TRƯỚC mốc "Khởi Thủy"
                new_data_batch = []
                batch_ended = False
                for item in data_list:
                    ts = int(item['timestamp'])
                    if ts < cutoff_timestamp:
                        new_data_batch.append(item)
                    else:
                        batch_ended = True # Đã chạm mốc 2019
                        break
                
                if new_data_batch:
                    all_data.extend(new_data_batch)
                    # Lấy timestamp cuối cùng đã "hút"
                    last_timestamp_in_batch = int(new_data_batch[-1]['timestamp'])
                    start_timestamp = last_timestamp_in_batch + 86400
                
                if batch_ended or not new_data_batch:
                    logging.info("Đã hút chạm mốc 'Khởi Thủy' (2019). Dừng hút Bitstamp.")
                    pbar.close()
                    break # Dừng vòng lặp
                
                pbar.update(1)
                time.sleep(1) # Chờ 1s cho đỡ "căng" API

            except Exception as e:
                logging.error(f"Lỗi khi hút Bitstamp: {e}")
                pbar.close()
                break

        if not all_data:
            logging.info("Không có data Bitstamp mới để xử lý.")
            return

        # Xử lý data mới
        logging.info(f"Đang xử lý {len(all_data)} ngày data Bitstamp mới...")
        df_new = pd.DataFrame(all_data)
        df_new = df_new.astype(float) # API trả về string
        df_new['datetime'] = pd.to_datetime(df_new['timestamp'], unit='s', utc=True)
        df_new = df_new.set_index('datetime')
        
        # Đổi tên cột để "nhận diện"
        df_new = df_new[['open', 'high', 'low', 'close', 'volume']]
        df_new.columns = ['Bitstamp_Open', 'Bitstamp_High', 'Bitstamp_Low', 'Bitstamp_Close', 'Bitstamp_Volume']
        
        # Concat và Lưu
        df_final = pd.concat([df_old, df_new])
        df_final = df_final[~df_final.index.duplicated(keep='last')]
        df_final.sort_index(inplace=True)
        
        df_final.to_parquet(self.file_bitstamp_daily, engine='pyarrow')
        logging.info(f"Đã cập nhật/lưu {len(df_final)} ngày lịch sử Bitstamp -> {self.file_bitstamp_daily}")

    # --- (MỚI) HÀM "NẤU" LỊCH KINH TẾ TỪ FRED (LOGIC TỪ FEDhistory.py) ---
    
    def _fetch_and_process_fred_data(self):
        """
        Hàm này "bê" toàn bộ logic từ FEDhistory.py vào
        để "hút" và "nấu" dữ liệu vĩ mô từ FRED.
        """
        logging.info("--- BẮT ĐẦU HÚT DỮ LIỆU VĨ MÔ (FRED) ---")
        
        # 1. CÀI ĐẶT THÔNG SỐ
        start_date = datetime.date(2019, 6, 1) # Lùi 3 tháng để tính .diff(3)
        end_date = datetime.date.today()

        monthly_codes = {
            'Core_CPI': 'CPILFESL',
            'CPI': 'CPIAUCSL',
            'Nonfarm': 'PAYEMS',
            'Unemployment': 'UNRATE',
            'Fed_Funds_Rate': 'FEDFUNDS'
        }
        weekly_codes = {
            'Fed_Balance_Sheet': 'WALCL' 
        }

        try:
            logging.info("Đang kéo dữ liệu HÀNG THÁNG (CPI, Nonfarm, Lãi suất...) từ FRED...")
            df_monthly = web.DataReader(list(monthly_codes.values()), 'fred', start_date, end_date)
            df_monthly.columns = list(monthly_codes.keys())

            logging.info("Đang kéo dữ liệu HÀNG TUẦN (QE/QT - Bảng cân đối FED) từ FRED...")
            df_weekly = web.DataReader(list(weekly_codes.values()), 'fred', start_date, end_date)
            df_weekly.columns = list(weekly_codes.keys())
        except Exception as e:
            logging.error(f"LỖI CHÍ MẠNG khi hút dữ liệu FRED: {e}")
            logging.error("Kiểm tra kết nối mạng hoặc API của FRED.")
            return pd.DataFrame() # Trả về DF rỗng

        # 2. XỬ LÝ VÀ GỘP DỮ LIỆU VỀ HÀNG THÁNG
        logging.info("Đang chuẩn hóa index về CUỐI THÁNG...")
        df_monthly_resampled = df_monthly.resample('M').last() 
        df_weekly_resampled = df_weekly.resample('M').last() 
        df_raw_monthly = df_monthly_resampled.join(df_weekly_resampled, how='outer')
        df_raw_monthly = df_raw_monthly.ffill() 

        # 3. TÍNH TOÁN CÁC CỘT (VALUE) VÀ CỘT (CHANGE)
        logging.info("Đang tính toán XU HƯỚNG (trend 3 tháng) cho tất cả chỉ số...")
        df_monthly_calcs = pd.DataFrame(index=df_raw_monthly.index)

        # 3A. LOGIC TREND 3 THÁNG
        for indicator_name in monthly_codes.keys():
            df_monthly_calcs[indicator_name] = df_raw_monthly[indicator_name]
            monthly_diff = df_raw_monthly[indicator_name].diff(3) 
            change_col_name = f"{indicator_name}_Change"
            df_monthly_calcs[change_col_name] = np.sign(monthly_diff).fillna(0).astype(int)

        # 3B. LOGIC QE/QT
        df_monthly_calcs['Fed_Balance_Sheet'] = df_raw_monthly['Fed_Balance_Sheet']
        qe_qt_trend = df_raw_monthly['Fed_Balance_Sheet'].diff(3) 
        df_monthly_calcs['QE_QT_Status'] = np.sign(qe_qt_trend).fillna(0).astype(int) * -1

        # 4. TẠO BẢNG DAILY (HÀNG NGÀY) VÀ FILL DỮ LIỆU
        logging.info("Đang tạo bảng DAILY (hàng ngày) và 'fill' dữ liệu xuyên suốt...")
        df_monthly_calcs = df_monthly_calcs.loc[df_monthly_calcs.index >= '2019-10-31']
        
        # (VÁ LỖI V10) Khởi tạo daily_index là UTC
        daily_index = pd.date_range(start='2019-11-01', end=end_date, freq='D', tz='UTC')
        
        # (VÁ LỖI V10) Chuyển index "mồi" (monthly) sang UTC
        df_monthly_calcs.index = df_monthly_calcs.index.tz_localize('UTC')
        
        # Dùng logic "vá lỗi" V7 (reindex FFILL)
        df_daily_final = df_monthly_calcs.reindex(daily_index, method='ffill')

        # 5. HOÀN THIỆN BẢNG (THÊM COUNTDOWN HALVING)
        df_daily_final = df_daily_final.reset_index()
        df_daily_final = df_daily_final.rename(columns={'index': 'DateTime'})

        logging.info("Đang tính toán đếm ngược ngày BTC Halving...")
        def calculate_halving_countdown(current_date):
            # (VÁ LỖI V10) Đảm bảo so sánh (naive vs naive)
            current_date_naive = current_date.tz_localize(None) 
            for halving_date in self.HALVING_DATES:
                if current_date_naive < halving_date:
                    return (halving_date - current_date_naive).days
            return 0 

        df_daily_final['BTC_Halving_Countdown'] = df_daily_final['DateTime'].apply(calculate_halving_countdown)

        # 6. Sắp xếp cột và Lưu Cache
        final_columns_order = [
            'DateTime', 'BTC_Halving_Countdown', 'QE_QT_Status', 
            'Fed_Funds_Rate', 'Fed_Funds_Rate_Change',
            'Core_CPI', 'Core_CPI_Change', 'CPI', 'CPI_Change',
            'Nonfarm', 'Nonfarm_Change', 'Unemployment', 'Unemployment_Change',
            'Fed_Balance_Sheet' 
        ]
        existing_cols = [col for col in final_columns_order if col in df_daily_final.columns]
        df_daily_final = df_daily_final[existing_cols]

        # 7. LƯU CACHE & CHUẨN BỊ ĐỂ JOIN
        logging.info(f"Đã 'nấu' xong Lịch Kinh Tế FRED, lưu cache vào {self.file_fred_features}")
        df_daily_final.to_parquet(self.file_fred_features, index=False, engine='pyarrow')
        
        # 8. CHUẨN BỊ ĐỂ JOIN (Set index DateTime)
        df_daily_final['DateTime'] = pd.to_datetime(df_daily_final['DateTime'])
        df_daily_final.set_index('DateTime', inplace=True)
        # (Index bây giờ đã là UTC, "khớp" với Nến)
        return df_daily_final

    # --- (MỚI V19) HÀM "ĐỘ" HÌNH NẾN ---
    def _get_candle_shape(self, df_in, open_col='Open', high_col='High', low_col='Low', close_col='Close'):
        """
        "Nấu" feature H1_Shape hoặc D1_Shape
        0 = Nến Thường
        1 = Doji / Spinning Top (Thân < 10%)
        2 = Marubozu / Lực Mạnh (Thân > 90%)
        """
        body = (df_in[close_col] - df_in[open_col]).abs()
        range_ = df_in[high_col] - df_in[low_col]
        
        # Thêm 1e-9 (epsilon) để tránh lỗi "Chia cho 0"
        body_ratio = body / (range_ + 1e-9) 
        
        shape = pd.Series(0, index=df_in.index) # 0 = Nến Thường
        shape.loc[body_ratio < 0.1] = 1 # 1 = Doji
        shape.loc[body_ratio > 0.9] = 2 # 2 = Marubozu
        return shape
            
    # --- CHẾ ĐỘ 2: "TẠO FILE MASTER" (ĐÃ ĐẠI PHẪU) ---
    
    def run_create_master_file(self):
        """
        Chế độ 2: Hợp nhất TẤT CẢ (Nến + Lịch FRED Tự Động).
        """
        logging.info("--- BẮT ĐẦU TẠO FILE MASTER (V23 - Sửa lỗi FVG Scanner) ---")
        try:
            df_m15 = pd.read_parquet(self.file_m15)
            df_h1 = pd.read_parquet(self.file_h1)
            df_d1 = pd.read_parquet(self.file_d1)
            df_w1 = pd.read_parquet(self.file_w1) # Data nến W1 (Binance)
            # (MỚI) Load file "Mồi"
            df_bitstamp_daily = pd.read_parquet(self.file_bitstamp_daily)
        except FileNotFoundError as e:
            logging.error(f"LỖI: Thiếu file dữ liệu gốc: {e}.")
            logging.error("Đại ca đã chạy 'Chế độ 1' và 'Chế độ 1.5' (ít nhất 1 lần) chưa?")
            return

        # 1. "NẤU" GÓI 4 (LỊCH KINH TẾ TỰ ĐỘNG TỪ FRED)
        df_features_fred = self._fetch_and_process_fred_data()
        if df_features_fred.empty:
            logging.error("Không thể tiếp tục vì không có dữ liệu FRED.")
            return
        
        # 2. (ĐÃ XÓA) "NẤU" GÓI 4 (CHẾ ĐỘ VĨ MÔ TỪ XLSX)

        # 3. "NẤU" GÓI 1 (KHUNG W1) - (ĐÃ "ĐỘ" MỒI DỮ LIỆU)
        logging.info("Đang 'độ' Gói 1 (Khung W1 - 'Mồi' Bitstamp để tính SMA)...")
        
        # A. Resample Bitstamp (Daily) -> Weekly (Index là UTC)
        df_w1_bitstamp = df_bitstamp_daily['Bitstamp_Close'].resample('W').last().to_frame()
        df_w1_bitstamp.columns = ['Close'] # Đổi tên cột để "nối"
        
        # B. "Mồi" dữ liệu Bitstamp (pre-2019) vào dữ liệu Binance W1 (post-2019)
        # (df_w1 là từ Binance, index đã là UTC)
        df_w1_combined = pd.concat([df_w1_bitstamp, df_w1[['Close']]])
        df_w1_combined = df_w1_combined[~df_w1_combined.index.duplicated(keep='last')] # (Giữ data Binance nếu trùng)
        df_w1_combined.sort_index(inplace=True)
        
        # C. Tính SMA trên "toàn bộ" lịch sử
        logging.info(f"Tổng lịch sử W1 (đã mồi) để tính SMA: {len(df_w1_combined)} nến (tuần).")
        df_w1_combined['W1_EMA21'] = df_w1_combined['Close'].ewm(span=21, adjust=False).mean()
        df_w1_combined['W1_SMA20'] = df_w1_combined['Close'].rolling(window=20).mean()
        df_w1_combined['W1_SMA50'] = df_w1_combined['Close'].rolling(window=50).mean()
        df_w1_combined['W1_SMA200'] = df_w1_combined['Close'].rolling(window=200).mean() # Sẽ CÓ DATA
        df_w1_combined['W1_Above_BMSB'] = (df_w1_combined['Close'] > df_w1_combined['W1_EMA21']).astype(int)
        
        # D. Lấy feature, resample H1, và join (như cũ)
        # (SỬA V19) Giữ lại các cột "thô" (W1_SMA200...)
        df_features_w1 = df_w1_combined[['W1_EMA21', 'W1_SMA20', 'W1_SMA50', 'W1_SMA200', 'W1_Above_BMSB']]
        df_features_w1 = df_features_w1.resample('1H').ffill()
        
        # 4. "NẤU" GÓI 1 (KHUNG D1)
        logging.info("Đang 'độ' Gói 1 (Khung D1 - Cross Distance & Hình Nến)...")
        df_d1['D1_RSI'] = df_d1['Close'].rolling(window=14).apply(
            lambda x: pd.Series(x).pct_change().fillna(0).pipe(
                lambda s: 100 - (100 / (1 + s.where(s > 0).mean() / s.where(s < 0).abs().mean()))
            ), raw=False
        )
        ema_long_d1 = df_d1['Close'].ewm(span=50, adjust=False).mean()
        ema_short_d1 = df_d1['Close'].ewm(span=20, adjust=False).mean()
        df_d1['D1_Regime'] = np.where(ema_short_d1 > ema_long_d1, 1, -1)
        df_d1['D1_SMA50'] = df_d1['Close'].rolling(window=50).mean()
        df_d1['D1_SMA200'] = df_d1['Close'].rolling(window=200).mean()
        
        # (SỬA V19) Thêm "Hình Dáng Nến" D1
        df_d1['D1_Shape'] = self._get_candle_shape(df_d1, 'Open', 'High', 'Low', 'Close')
        
        d1_cols_to_keep = ['D1_RSI', 'D1_Regime', 'D1_SMA50', 'D1_SMA200', 'D1_Shape']
        df_features_d1 = df_d1[d1_cols_to_keep].resample('1H').ffill()

        # 5. "NẤU" M15 (Bối cảnh vi mô)
        logging.info("Đang xử lý M15 (Bối cảnh vi mô)...")
        
        # (SỬA LỖI V9) 
        m15_close = df_m15['Close'].resample('1H').last()
        m15_volume = df_m15['Volume'].resample('1H').sum()
        m15_volatility = df_m15['Close'].resample('1H').std()
        
        df_features_m15 = pd.DataFrame({
            'M15_Close': m15_close,
            'M15_Volume': m15_volume,
            'M15_Volatility': m15_volatility
        })

        # 6. HỢP NHẤT (JOIN TẤT CẢ)
        logging.info("Đang HỢP NHẤT (Join) tất cả dữ liệu...")

        # (SỬA V19) "Nấu" Hình Nến H1
        logging.info("Đang 'độ' Gói 1 (Khung H1 - Hình Nến)...")
        df_h1['H1_Shape'] = self._get_candle_shape(df_h1, 'Open', 'High', 'Low', 'Close')
        
        df_h1.columns = [f"H1_{col}" for col in df_h1.columns]
        
        # (SỬA V19) GIỮ LẠI H1_Open, H1_High, H1_Low
        
        # (VÁ LỖI V10) Resample FRED (UTC) sang H1 (UTC)
        df_features_fred_resampled = df_features_fred.resample('1H').ffill()
        
        df_master = df_h1
        df_master = df_master.join(df_features_w1, how='left')
        df_master = df_master.join(df_features_d1, how='left')
        df_master = df_master.join(df_features_m15, how='left')
        df_master = df_master.join(df_features_fred_resampled, how='left') # (MỚI) Join FRED data

        # 7. "ĐỘ" MÓN CUỐI (Vol Spike, Giờ, FVG)
        
        # Món 1: Volume Spike
        logging.info("Đang 'độ' Món 1 (Volume Spike)...")
        vol_ma_50 = df_master['H1_Volume'].rolling(window=50, min_periods=10).mean()
        df_master['H1_Vol_Spike_50'] = (df_master['H1_Volume'] - vol_ma_50) / (vol_ma_50 + 1e-9)

        # Món 2: Phiên + Giờ (Sin/Cos) - (SỬA V21)
        logging.info("Đang 'độ' Món 2 (Gom Session & Giờ Sin/Cos)...")
        hour_series = df_master.index.hour
        df_master['Hour_Sin'] = np.sin(2 * np.pi * hour_series / 24.0)
        df_master['Hour_Cos'] = np.cos(2 * np.pi * hour_series / 24.0)
        df_master['Day_of_Week'] = df_master.index.dayofweek # (0=T2, 6=CN)
        
        # (SỬA V21) Logic "Gom Session" (Killzone)
        # 0=Nghỉ (21-23), 1=Á (0-6), 2=Âu (7-11), 3=Mỹ (12-20)
        bins = [-1, 7, 12, 21, 24]
        labels = [1, 2, 3, 0]
        # (right=False: [0, 7) là 1, [7, 12) là 2, ...)
        df_master['H1_Session'] = pd.cut(hour_series, bins=bins, labels=labels, right=False).astype(int)
        
                # --- (SỬA V23 FIXED) Món 3: FVG SCANNER (1 Strongest + 1 Nearest) ---
        logging.info("Đang 'độ' Món 3 (FVG Scanner V23 - Bản vá tương thích mọi pandas)...")

        # A. Tìm "nguyên liệu" FVG
        high_prev_2 = df_master['H1_High'].shift(2)
        low_prev_2 = df_master['H1_Low'].shift(2)
        vol_prev_1 = df_master['H1_Volume'].shift(1)
        h1_close_col = df_master['H1_Close']

        # B. "Nấu" FVG Tăng (Bull FVG)
        bull_fvg_size_raw = df_master['H1_Low'] - high_prev_2
        is_bull_fvg = bull_fvg_size_raw > 0
        df_master['temp_bull_fvg_high'] = high_prev_2
        df_master['temp_bull_fvg_vol'] = vol_prev_1.where(is_bull_fvg, np.nan)
        df_master['temp_bull_fvg_size_pct'] = (bull_fvg_size_raw.clip(lower=0) / h1_close_col).where(is_bull_fvg, np.nan)
        df_master['temp_bull_fvg_dist_pct'] = ((h1_close_col - df_master['temp_bull_fvg_high']) / h1_close_col).where(is_bull_fvg, np.nan)

        # C. "Nấu" FVG Giảm (Bear FVG)
        bear_fvg_size_raw = low_prev_2 - df_master['H1_High']
        is_bear_fvg = bear_fvg_size_raw > 0
        df_master['temp_bear_fvg_low'] = low_prev_2
        df_master['temp_bear_fvg_vol'] = vol_prev_1.where(is_bear_fvg, np.nan)
        df_master['temp_bear_fvg_size_pct'] = (bear_fvg_size_raw.clip(lower=0) / h1_close_col).where(is_bear_fvg, np.nan)
        df_master['temp_bear_fvg_dist_pct'] = ((h1_close_col - df_master['temp_bear_fvg_low']) / h1_close_col).where(is_bear_fvg, np.nan)

        # D. "Quét" (Scan) – Bản vá tương thích
        SCAN_WINDOW = 100
        PRICE_ZONE_PCT = 0.10

        def rolling_idx_extreme(series, window, mode='max'):
            """Tìm index (timestamp) của giá trị lớn nhất/nhỏ nhất trong rolling window."""
            series = pd.to_numeric(series, errors='coerce')
            idx_list = series.index.to_list()
            result = []
            for i in range(len(series)):
                start = max(0, i - window + 1)
                win = series.iloc[start:i+1]
                if win.isna().all():
                    result.append(np.nan)
                    continue
                idx = win.idxmax() if mode == 'max' else win.idxmin()
                result.append(idx)
            return pd.Series(result, index=series.index)

        logging.info("Đang quét (rolling) FVG 'Strongest' (Bản vá)...")
        idx_strongest_bull = rolling_idx_extreme(df_master['temp_bull_fvg_vol'], SCAN_WINDOW, mode='max')
        idx_strongest_bear = rolling_idx_extreme(df_master['temp_bear_fvg_vol'], SCAN_WINDOW, mode='max')

        strongest_bull_dist = df_master['temp_bull_fvg_dist_pct'].reindex(idx_strongest_bull).values
        strongest_bull_size = df_master['temp_bull_fvg_size_pct'].reindex(idx_strongest_bull).values
        strongest_bear_dist = df_master['temp_bear_fvg_dist_pct'].reindex(idx_strongest_bear).values
        strongest_bear_size = df_master['temp_bear_fvg_size_pct'].reindex(idx_strongest_bear).values

        # E. "Nearest" – tìm FVG gần nhất (ZOI)
        logging.info("Đang quét (rolling) FVG 'Nearest' (Bản vá)...")
        bull_dist_zoi = df_master['temp_bull_fvg_dist_pct'].where(
            (df_master['temp_bull_fvg_dist_pct'] > 0) &
            (df_master['temp_bull_fvg_dist_pct'] < PRICE_ZONE_PCT)
        )
        bear_dist_zoi = df_master['temp_bear_fvg_dist_pct'].where(
            (df_master['temp_bear_fvg_dist_pct'] < 0) &
            (df_master['temp_bear_fvg_dist_pct'] > -PRICE_ZONE_PCT)
        )

        idx_nearest_bull = rolling_idx_extreme(bull_dist_zoi, SCAN_WINDOW, mode='min')
        idx_nearest_bear = rolling_idx_extreme(bear_dist_zoi, SCAN_WINDOW, mode='max')

        for col in [
            'temp_bull_fvg_dist_pct', 'temp_bull_fvg_size_pct',
            'temp_bear_fvg_dist_pct', 'temp_bear_fvg_size_pct'
        ]:
            if col not in df_master.columns:
                df_master[col] = np.nan
        
        # Bảo vệ khi reindex (tránh KeyError)
        nearest_bull_dist = df_master.get('temp_bull_fvg_dist_pct', pd.Series(index=df_master.index)).reindex(idx_nearest_bull).values
        nearest_bull_size = df_master.get('temp_bull_fvg_size_pct', pd.Series(index=df_master.index)).reindex(idx_nearest_bull).values
        nearest_bear_dist = df_master.get('temp_bear_fvg_dist_pct', pd.Series(index=df_master.index)).reindex(idx_nearest_bear).values
        nearest_bear_size = df_master.get('temp_bear_fvg_size_pct', pd.Series(index=df_master.index)).reindex(idx_nearest_bear).values

        # F. Gán kết quả vào bảng chính
        df_master['H1_Bull_FVG_Strong_Dist_Pct'] = strongest_bull_dist
        df_master['H1_Bull_FVG_Strong_Size_Pct'] = strongest_bull_size
        df_master['H1_Bear_FVG_Strong_Dist_Pct'] = strongest_bear_dist
        df_master['H1_Bear_FVG_Strong_Size_Pct'] = strongest_bear_size

        df_master['H1_Bull_FVG_Nearest_Dist_Pct'] = nearest_bull_dist
        df_master['H1_Bull_FVG_Nearest_Size_Pct'] = nearest_bull_size
        df_master['H1_Bear_FVG_Nearest_Dist_Pct'] = nearest_bear_dist
        df_master['H1_Bear_FVG_Nearest_Size_Pct'] = nearest_bear_size

        # G. Lọc kết quả theo Zone of Interest (10%)
        df_master['H1_Bull_FVG_Strong_Dist_Pct'] = df_master['H1_Bull_FVG_Strong_Dist_Pct'].where(
            (df_master['H1_Bull_FVG_Strong_Dist_Pct'] > 0) &
            (df_master['H1_Bull_FVG_Strong_Dist_Pct'] < PRICE_ZONE_PCT), 0
        )
        df_master['H1_Bull_FVG_Strong_Size_Pct'] = df_master['H1_Bull_FVG_Strong_Size_Pct'].where(
            df_master['H1_Bull_FVG_Strong_Dist_Pct'] != 0, 0
        )
        df_master['H1_Bear_FVG_Strong_Dist_Pct'] = df_master['H1_Bear_FVG_Strong_Dist_Pct'].where(
            (df_master['H1_Bear_FVG_Strong_Dist_Pct'] < 0) &
            (df_master['H1_Bear_FVG_Strong_Dist_Pct'] > -PRICE_ZONE_PCT), 0
        )
        df_master['H1_Bear_FVG_Strong_Size_Pct'] = df_master['H1_Bear_FVG_Strong_Size_Pct'].where(
            df_master['H1_Bear_FVG_Strong_Dist_Pct'] != 0, 0
        )

        # "Trám" NaN còn sót
        df_master.fillna({
            'H1_Bull_FVG_Nearest_Dist_Pct': 0, 'H1_Bull_FVG_Nearest_Size_Pct': 0,
            'H1_Bear_FVG_Nearest_Dist_Pct': 0, 'H1_Bear_FVG_Nearest_Size_Pct': 0
        }, inplace=True)

        # Dọn cột tạm
        df_master.drop(columns=[
            'temp_bull_fvg_high', 'temp_bull_fvg_vol', 'temp_bull_fvg_size_pct', 'temp_bull_fvg_dist_pct',
            'temp_bear_fvg_low', 'temp_bear_fvg_vol', 'temp_bear_fvg_size_pct', 'temp_bear_fvg_dist_pct'
        ], inplace=True, errors='ignore')


        # --- (KẾT THÚC MÓN 3 - BẢN VÁ TOÀN DIỆN) ---

        
        

        # E. "Gán" kết quả (8 cột)
        df_master['H1_Bull_FVG_Strong_Dist_Pct'] = strongest_bull_dist
        df_master['H1_Bull_FVG_Strong_Size_Pct'] = strongest_bull_size
        df_master['H1_Bear_FVG_Strong_Dist_Pct'] = strongest_bear_dist
        df_master['H1_Bear_FVG_Strong_Size_Pct'] = strongest_bear_size
        
        df_master['H1_Bull_FVG_Nearest_Dist_Pct'] = nearest_bull_dist
        df_master['H1_Bull_FVG_Nearest_Size_Pct'] = nearest_bull_size
        df_master['H1_Bear_FVG_Nearest_Dist_Pct'] = nearest_bear_dist
        df_master['H1_Bear_FVG_Nearest_Size_Pct'] = nearest_bear_size

        # F. "Lọc" (Filter) FVG "Strongest" theo 10% ZOI
        # (Nếu FVG "Mạnh nhất" nằm ngoài ZOI 10%, "trám" nó bằng 0)
        df_master['H1_Bull_FVG_Strong_Dist_Pct'] = df_master['H1_Bull_FVG_Strong_Dist_Pct'].where(
            (df_master['H1_Bull_FVG_Strong_Dist_Pct'] > 0) & 
            (df_master['H1_Bull_FVG_Strong_Dist_Pct'] < PRICE_ZONE_PCT), 0
        )
        df_master['H1_Bull_FVG_Strong_Size_Pct'] = df_master['H1_Bull_FVG_Strong_Size_Pct'].where(
            df_master['H1_Bull_FVG_Strong_Dist_Pct'] != 0, 0 # Nếu Dist=0, Size=0
        )
        df_master['H1_Bear_FVG_Strong_Dist_Pct'] = df_master['H1_Bear_FVG_Strong_Dist_Pct'].where(
            (df_master['H1_Bear_FVG_Strong_Dist_Pct'] < 0) & 
            (df_master['H1_Bear_FVG_Strong_Dist_Pct'] > -PRICE_ZONE_PCT), 0
        )
        df_master['H1_Bear_FVG_Strong_Size_Pct'] = df_master['H1_Bear_FVG_Strong_Size_Pct'].where(
            df_master['H1_Bear_FVG_Strong_Dist_Pct'] != 0, 0 # Nếu Dist=0, Size=0
        )
        
        # (FVG "Nearest" đã tự "lọc" ZOI ở bước D2, chỉ cần "trám" NaN)
        df_master.fillna({
            'H1_Bull_FVG_Nearest_Dist_Pct': 0, 'H1_Bull_FVG_Nearest_Size_Pct': 0,
            'H1_Bear_FVG_Nearest_Dist_Pct': 0, 'H1_Bear_FVG_Nearest_Size_Pct': 0
        }, inplace=True)

        # G. "Dọn dẹp" (Drop) các cột "temp"
        df_master.drop(columns=[
            'temp_bull_fvg_high', 'temp_bull_fvg_vol', 'temp_bull_fvg_size_pct', 'temp_bull_fvg_dist_pct',
            'temp_bear_fvg_low', 'temp_bear_fvg_vol', 'temp_bear_fvg_size_pct', 'temp_bear_fvg_dist_pct'
        ], inplace=True,errors='ignore')
        # --- (KẾT THÚC MÓN 3) ---

        # --- (MỚI) BƯỚC 7.6: "ĐỘ" FEATURE KHOẢNG CÁCH (THEO YÊU CẦU V18) ---
        logging.info("Đang 'độ' Món 4 (Feature Khoảng Cách %)...")
        
        # (Lấy H1_Close làm "chuẩn")
        h1_close_col = df_master['H1_Close']
        
        # Danh sách các cột "giá trị" (SMA/EMA) cần "độ"
        cols_to_distance = [
            'W1_EMA21', 'W1_SMA20', 'W1_SMA50', 'W1_SMA200',
            'D1_SMA50', 'D1_SMA200',
            'M15_Close' # (Bonus) "Độ" luôn khoảng cách M15
        ]
        
        for col in cols_to_distance:
            if col in df_master.columns:
                # 1. Tính toán
                new_col_name = f"{col}_Dist_Pct"
                df_master[new_col_name] = (h1_close_col - df_master[col]) / h1_close_col
                
        # (Bonus) Tính D1_Cross_Distance
        if 'D1_SMA50' in df_master.columns and 'D1_SMA200' in df_master.columns:
             df_master['D1_Cross_Distance_Pct'] = (df_master['D1_SMA50'] - df_master['D1_SMA200']) / h1_close_col
             
        # (SỬA V19) KHÔNG "dọn dẹp" (drop) các cột "thô" (W1_SMA200, D1_SMA50...)
        logging.info("Đã 'độ' xong Feature Khoảng Cách (Giữ lại cột 'thô').")
        # --- (KẾT THÚC BƯỚC 7.6) ---

        # 8. "Vá" lần cuối và Lưu (SỬA LỖI V17 - Quay lại logic "Trám 0")
        logging.info("Đang 'vá' các lỗ hổng dữ liệu (ffill)...")
        df_master.fillna(method='ffill', inplace=True)
        
        logging.info("Đang 'vá ngược' (bfill) dữ liệu vĩ mô...")
        df_master.fillna(method='bfill', inplace=True)

        logging.info("Đang 'trám' 0 vào các chỉ báo chưa 'khởi động'...")
        df_master.fillna(0, inplace=True)
        
        # (ĐÃ XÓA) df_master.dropna(inplace=True) 
        
        if df_master.empty:
            logging.error("LỖI: File master bị rỗng (Lỗi V23 - Logic Bất Thường). Kiểm tra lại.")
            return

        df_master.to_parquet(self.file_master, engine='pyarrow')
        
        logging.info(f"--- HOÀN TẤT --- Đã tạo file 'Thức ăn AI' (V23) tại: {self.file_master}")
        logging.info(f"Tổng số 'món' (features) trong 'thực đơn': {len(df_master.columns)}")


# --- MENU CHẠY FILE (ĐÃ SỬA) ---
if __name__ == "__main__":
    
    service = MasterDataServiceV23(symbol='BTCUSDT') # Đổi tên V22 -> V23

    print("\n" + "="*70)
    print("      CHỌN CHẾ ĐỘ CHẠY (V23 - SỬA LỖI FVG BẰNG idxmax)")
    print("="*70)
    print(f"  1. [HẤP TINH NẾN] - Tải/Cập nhật 4 file Binance (W1, D1, H1, M15).")
    print(f"  1.5.[MỒI LỊCH SỬ] - (RESUMABLE) Hút data pre-2019 từ Bitstamp.")
    print(f"  2. [TẠO FILE MASTER (V23)] - Gộp (Nến + Lịch FRED + Mồi Bitstamp).")
    print("="*70)
    print("LƯU Ý 1: Đại ca phải chạy 1 VÀ 1.5 (ít nhất 1 lần) trước khi chạy 2.")
    print("LƯU Ý 2: Cần chạy 'pip install pandas-datareader'.")
    print("LƯU Ý 3: File sẽ được lưu vào 3 thư mục: 00_Raw_Data, 01_Processed_Data, 02_Master_Data.")
    print("="*70) 
    
    choice = input("Đại ca muốn chạy chế độ nào? (1, 1.5, hoặc 2): ")

    if choice == '1':
        service.run_download_klines()

    elif choice == '1.5':
        service.run_fetch_bitstamp_backfill()
        
    elif choice == '2':
        service.run_create_master_file()
            
    else:
        print("Lựa chọn không hợp lệ. Vui lòng nhập 1, 1.5 hoặc 2.")