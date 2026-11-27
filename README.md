# 🎾 Dự đoán kết quả trận đấu quần vợt nam bằng Gradient Boosting — ATP Tour & ITF Men's Circuit

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Tổng quan

Dự án **Nghiên cứu Khoa học (NCKH)**: ứng dụng và **đánh giá nghiêm ngặt,
có kiểm định thống kê** khả năng dự đoán kết quả trận đấu quần vợt nam
của 4 thuật toán Gradient Boosting/Ensemble (**CatBoost**, **LightGBM**,
**XGBoost**, **Random Forest**) trên dữ liệu ATP Tour & ITF Men's Circuit
2020–2025, kèm ứng dụng demo Streamlit trực quan.

> 📄 Đề bài nghiên cứu đầy đủ (tên đề tài, câu hỏi nghiên cứu, giả thuyết,
> mục tiêu, đóng góp mới, giới hạn) tại
> [docs/DE_TAI_NGHIEN_CUU.md](docs/DE_TAI_NGHIEN_CUU.md). Phần dưới đây
> chỉ tóm tắt mục tiêu cụ thể.

### Mục tiêu nghiên cứu
*(chi tiết + câu hỏi nghiên cứu tương ứng: xem
[docs/DE_TAI_NGHIEN_CUU.md](docs/DE_TAI_NGHIEN_CUU.md#5-mục-tiêu-nghiên-cứu))*

1. Xây dựng pipeline ML hoàn chỉnh, có kiểm soát chất lượng, không rò rỉ
   thông tin tương lai (data leakage), từ dữ liệu thô đến mô hình dự đoán.
2. So sánh 4 thuật toán, xác minh model tốt nhất vượt trội **có ý nghĩa
   thống kê** so với các chiến lược đơn giản (luôn chọn hạng cao hơn, luôn
   theo tỷ lệ cược thấp nhất), và đối chiếu với kết quả trong y văn đã
   công bố (literature benchmark).
3. Đánh giá độ ổn định theo thời gian (walk-forward), mức độ overfitting,
   và độ tin cậy của xác suất dự đoán (calibration/ECE) — không chỉ báo
   cáo 1 con số accuracy.
4. Phân tích các yếu tố ảnh hưởng lớn nhất đến kết quả trận đấu (SHAP
   explainability).
5. *(Khám phá, phụ — không phải mục tiêu chính)* Kiểm định giả thuyết
   không H0: model KHÔNG có lợi thế thống kê khai thác được so với thị
   trường tỷ lệ cược (backtest ROI + CI 95%). **Kết quả thật: H0 KHÔNG bị
   bác bỏ** — xem
   [LIMITATIONS_AND_INTERPRETATION.md](LIMITATIONS_AND_INTERPRETATION.md).

### Phạm vi dữ liệu

| Thuộc tính | Chi tiết |
|-----------|---------|
| **Nguồn** | ATP match data 2020–2025 kết hợp tỷ lệ cược (odds) nhà cái |
| **Giới tính** | Chỉ nam (ATP) |
| **Cấp giải** | ITF Futures/Challenger (phần lớn), ATP 250/500/1000, Grand Slam |
| **Số trận** | Train ≈ 91,913 · Test ≈ 32,168 (theo log chạy thật; xem `data/features/*.parquet` để có số chính xác của cả 3 tập, bao gồm Val) |
| **Khoảng thời gian** | Train < 2024, Val = 2024, Test ≥ 2025 (chia theo thời gian, không random) |

> ⚠️ **Giới hạn nghiên cứu**: phần lớn dữ liệu là giải nhỏ (ITF Futures/Challenger), không dùng dữ liệu trong trận/chấn thương/thời tiết, độ phủ odds thấp (~6%). Xem chi tiết đầy đủ tại [docs/methodology.md](docs/methodology.md) (mục 4 — Limitations & Future Work) hoặc trực tiếp trong app ở tab **Analytics → ⚠️ Giới hạn nghiên cứu**.

## Kiến trúc Pipeline

Pipeline gồm **21 bước module hoá**, đánh số theo thứ tự chạy (một số bước phân tích 14–21 độc lập với nhau, chạy sau khi đã có model):

```
src/pipelines/
├── 01_load_data.py              ← Load CSV → Parquet
├── 02_data_validation.py        ← Kiểm tra schema, duplicate
├── 03_clean_schema.py           ← Chuẩn hoá tên cột, kiểu dữ liệu
├── 04_handle_missing.py         ← Xử lý NaN (median, fillna có chủ đích)
├── 05_remove_data_leakage.py    ← Loại bỏ thông tin trong/sau trận (w_ace, score...)
├── 06_eda.py                    ← EDA toàn diện (10+ biểu đồ)
├── 07_feature_engineering.py    ← Elo, surface Elo, H2H, rolling form 5/10, biến đổi đối xứng
├── 08_feature_selection.py      ← Loại cột ID, xử lý categorical
├── 09_split_dataset.py          ← Time-based split (Train <2024, Val 2024, Test ≥2025)
├── 10_train_models.py           ← Train 4 model baseline
├── 11_hyperparameter_tuning.py  ← Optuna + TimeSeriesSplit CV
├── 12_evaluate_models.py        ← So sánh toàn diện + baseline strategies + calibration curve
├── 13_explain_model.py          ← SHAP values (model chính: CatBoost_tuned)
├── 14_save_model.py             ← Đóng gói model production
├── 15_statistical_significance.py ← DeLong's test + bootstrap CI so sánh các model
├── 16_data_leakage_analysis.py  ← Minh hoạ định lượng tác động của data leakage (AUC có/không leakage)
├── 17_backtest_roi.py           ← Backtest ROI (de-vig đúng chuẩn) + CI 95% + ổn định theo quý
├── 18_literature_benchmark.py   ← Đối chiếu kết quả với các nghiên cứu đã công bố
├── 19_overfitting_analysis.py   ← Gap Train/Val/Test, learning curve, validation curve, CV stability
├── 20_odds_ablation.py          ← Ablation có/không odds
├── 21_research_enhancements.py  ← Rolling-aware research: calibration, ablation, walk-forward
├── 22_research_audit.py         ← Audit tự động bằng chứng nghiên cứu
└── 20_odds_ablation.py          ← Đo đóng góp của odds
└── 21_research_enhancements.py  ← Feature Ablation + Calibration/ECE + framework Walk-forward
```

Tiện ích dùng chung: `src/utils/categorical.py` — đồng nhất categorical dtype/encoding giữa train/val/test cho mọi script (tránh lỗi model gặp category chưa từng thấy lúc train, và tránh mã hoá lệch âm thầm giữa các lần chạy độc lập).

## Kết quả chính

Số liệu dưới đây phản ánh lần chạy gần nhất được ghi vào `experiments/`; xem trực tiếp các file JSON hoặc mở app (trang **Analytics**) để có số liệu mới nhất sau khi bạn tự chạy lại pipeline.

| Nguồn số liệu | File |
|---|---|
| Metrics baseline (val set) | `experiments/metrics.json` |
| Đánh giá đầy đủ trên test set (Accuracy/AUC/F1/LogLoss/Brier + baseline heuristics) | `experiments/test_evaluation.json`, `.csv` |
| Kiểm định thống kê (DeLong's test, bootstrap CI) | `experiments/statistical_significance.json` |
| Phân tích overfitting (gap, learning/validation curve, CV stability) | `experiments/overfitting_analysis.json` |
| Backtest ROI (de-vig, CI 95%, ổn định theo quý) | `experiments/backtest_roi.json` |
| Đối chiếu y văn | `experiments/literature_benchmark.json` |

**Model chính được chọn để triển khai: `CatBoost_tuned`** — chọn dựa trên AUC và gap Train/Val/Test (overfitting analysis), không chọn dựa trên ROI (tránh data snooping — xem `17_backtest_roi.py`).

### Kiểm soát chất lượng dữ liệu & phương pháp

Trong quá trình phát triển, dự án đã chủ động phát hiện và sửa nhiều vấn đề phương pháp luận, ghi chép đầy đủ tại [CHANGELOG_FIXES.md](CHANGELOG_FIXES.md), trong đó đáng chú ý:

- **2 lỗi data leakage nghiêm trọng** (cột thống kê trong trận bị sót lại, biến diff tính trước khi swap Player A/B) — đã sửa và có test tự động xác nhận (`tests/test_leakage_detection.py`, `src/pipelines/16_data_leakage_analysis.py` minh hoạ định lượng chênh lệch AUC có/không leakage).
- **Category encoding không nhất quán giữa train/val/test** (gây lỗi "category not in training set" ở XGBoost, và mã hoá lệch âm thầm ở RandomForest) — sửa tận gốc bằng `src/utils/categorical.py`, áp dụng thống nhất cho toàn bộ 6 script liên quan.
- **De-vig sai công thức** trong tính implied probability cho backtest ROI (nay khớp đúng công thức de-vig chuẩn dùng trong app).
- **Data snooping khi chọn "best model"** theo ROI đo trên chính test set — nay cố định model đầu tàu theo AUC/overfitting-gap, không theo ROI.

## Ứng dụng Demo (Streamlit)

```
project/
├── main.py                 ← Entry point, router giữa các trang, inject theme
├── pages/
│   ├── Home.py              ← Trang chủ — bento grid hero, số liệu thật
│   ├── Prediction.py        ← Dự đoán trận đấu (nhập 2 tay vợt) + SHAP + phân tích kèo cược
│   ├── Bracket.py           ← Mô phỏng nhánh đấu giải
│   ├── PlayerProfile.py     ← Hồ sơ chi tiết từng tay vợt
│   ├── Analytics.py         ← 5 tab: EDA / Đánh giá mô hình / SHAP / Thống kê mô tả / Giới hạn nghiên cứu
│   └── Dataset.py           ← Xem trực tiếp bộ dữ liệu
├── components/              ← Card, gauge, biểu đồ SHAP/radar dùng chung
├── services/                ← Load dữ liệu, inference, hồ sơ tay vợt, sinh báo cáo PDF
└── utils/                   ← Theme "Night Court" (bảng màu, typography), formatters, validators
```

Tab **Analytics → Đánh giá mô hình** trình bày đầy đủ: so sánh 4 model, ma trận nhầm lẫn, DeLong's test, calibration curve, phân tích overfitting, và backtest ROI (bảng CI 95% + ổn định theo quý) — mỗi biểu đồ đều kèm câu hỏi phân tích và nhận định, không chỉ trưng hình.

## Cài đặt & Chạy

### Yêu cầu
- Python 3.10+
- pip

### Cài đặt
```bash
cd "path/to/ATP_Prediction_Project_FIXED/proj"

# Tạo virtual environment
python -m venv .venv
.venv\Scripts\activate     # Windows
# source .venv/bin/activate   # Linux/Mac

# Cài dependencies
pip install -r requirements.txt
```

### Chạy Pipeline
```bash
# Chạy tuần tự từ đầu (khuyến nghị lần đầu)
python src/pipelines/01_load_data.py
python src/pipelines/02_data_validation.py
python src/pipelines/03_clean_schema.py
python src/pipelines/04_handle_missing.py
python src/pipelines/05_remove_data_leakage.py
python src/pipelines/06_eda.py
python src/pipelines/07_feature_engineering.py
python src/pipelines/08_feature_selection.py
python src/pipelines/09_split_dataset.py
python src/pipelines/10_train_models.py
python src/pipelines/11_hyperparameter_tuning.py    # tốn nhiều thời gian nhất (Optuna)
python src/pipelines/12_evaluate_models.py
python src/pipelines/13_explain_model.py
python src/pipelines/14_save_model.py

# Các phân tích bổ sung (chạy sau khi đã có model ở bước 10/11)
python src/pipelines/15_statistical_significance.py
python src/pipelines/16_data_leakage_analysis.py
python src/pipelines/17_backtest_roi.py
python src/pipelines/18_literature_benchmark.py
python src/pipelines/19_overfitting_analysis.py
python src/pipelines/20_odds_ablation.py
python src/pipelines/21_research_enhancements.py --calibration
python src/pipelines/21_research_enhancements.py --ablation
```

### Chạy Tests
```bash
python -m pytest tests/ -v --tb=short
```

### Chạy App Demo
```bash
python runapp.py
# hoặc
streamlit run project/main.py
```

## Cấu trúc thư mục

```
├── project/                # Ứng dụng Streamlit demo (xem chi tiết ở mục "Ứng dụng Demo" trên)
├── configs/                # config.yaml — đường dẫn data/model/reports dùng chung cho pipeline
├── data/
│   ├── raw/                # Dữ liệu gốc CSV
│   ├── interim/             # Dữ liệu trung gian (parquet)
│   ├── processed/           # Dữ liệu đã clean
│   └── features/            # Features + train/val/test splits
├── docs/
│   └── methodology.md       # Phương pháp luận chi tiết, bao gồm Limitations & Future Work
├── experiments/             # Metrics/kết quả phân tích dạng JSON/CSV (xem bảng "Kết quả chính")
├── models/
│   ├── *.joblib             # Model baseline
│   ├── tuned/                # Model đã tune (Optuna)
│   └── final/                 # Model production
├── reports/
│   ├── figures/              # Biểu đồ EDA + evaluation + overfitting + backtest
│   └── shap_plots/           # SHAP explainability
├── src/
│   ├── pipelines/            # 22 bước pipeline (xem mục "Kiến trúc Pipeline")
│   └── utils/                 # categorical.py (encoding dùng chung), logger.py, config loader
├── tests/                    # Unit test (pytest) — bao gồm test phát hiện leakage
├── runapp.py                 # Script khởi chạy app (gọi streamlit run project/main.py)
├── CHANGELOG_FIXES.md         # Nhật ký đầy đủ các lỗi đã phát hiện & sửa trong quá trình phát triển
├── requirements.txt
└── README.md
```

## Tài liệu

### Nâng cấp nghiên cứu (Research Enhancements)

Bản MASTERFIXED bổ sung ba hướng có giá trị học thuật, nhưng **không dùng test set để chọn model**:

1. **Rolling Form (5/10 trận)** — phong độ trước trận, tính theo snapshot từng ngày; kết quả của các trận cùng ngày không được dùng cho nhau.
2. **Feature Ablation** — so sánh tuần tự Ranking → +Elo → +H2H → +Form → +Odds để đo đóng góp biên của từng nhóm feature.
3. **Probability Calibration + ECE** — hiệu chuẩn xác suất trên validation rồi đánh giá trên test; báo cáo AUC, LogLoss, Brier và ECE.
4. **Walk-forward evaluation** — khung đánh giá cuốn chiếu theo thời gian được thiết kế để mô phỏng đúng quy trình triển khai thực tế.

> Lưu ý: các thí nghiệm mới là **research experiments (thí nghiệm nghiên cứu)**, không được dùng để cherry-pick (chọn kết quả đẹp) hoặc đổi primary model dựa trên test AUC.

- [Phương pháp luận chi tiết](docs/methodology.md) — Mô tả đầy đủ pipeline, features, chiến lược chống leakage, và giới hạn nghiên cứu.
- [CHANGELOG_FIXES.md](CHANGELOG_FIXES.md) — Nhật ký toàn bộ lỗi/thiếu sót đã phát hiện và cách sửa, theo từng PHẦN đánh số.
- [Biểu đồ EDA & evaluation](reports/figures/) — Toàn bộ biểu đồ phân tích dữ liệu và đánh giá model.
- [SHAP Analysis](reports/shap_plots/) — Giải thích quyết định của model chính (CatBoost_tuned).

## Nhóm nghiên cứu

Dự án NCKH Sinh viên — Ứng dụng Machine Learning trong dự đoán kết quả thể thao.

## Research Upgrade (21–22)

`21_research_enhancements.py` bổ sung Feature Ablation, Calibration/ECE và Walk-forward evaluation. `22_research_audit.py` kiểm tra tự động xem artifact thực nghiệm đã đủ bằng chứng hay chưa; **BLOCKED/WARN không được tự động biến thành PASS**.

Chạy toàn bộ nghiên cứu trên Windows bằng `run_research_upgrade.bat`.

> Lưu ý: các metrics mới chỉ hợp lệ sau khi chạy lại pipeline trên dữ liệu sau khi thêm Rolling Form. Không dùng metrics cũ để tuyên bố kết quả của phiên bản mới.
