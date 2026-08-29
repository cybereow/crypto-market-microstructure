# Trading Bot — v2.0 (Machine Learning & Statistical Arbitrage)

این پروژه از یک ربات آربیتراژ مثلثی ناموفق، به یک فریم‌ورک پیشرفته‌تر برای استراتژی‌های کمی (Quantitative) مبتنی بر ماشین‌لرنینگ و آربیتراژ آماری تغییر کاربری داده است.

## استراتژی‌های جدید

۱. **آربیتراژ آماری (Pairs Trading)**: پیدا کردن جفت‌ارزهایی که از نظر آماری هم‌بستگی بالایی دارند (Cointegrated) و معامله بر اساس Z-Score اسپرد آن‌ها.
۲. **یادگیری ماشین (Machine Learning)**: استفاده از Random Forest برای پیش‌بینی جهت کندل بعدی بر اساس اندیکاتورهای تکنیکال (مثل SMA, Volatility) به عنوان ویژگی‌ها (Features).

## نصب و راه‌اندازی

ابتدا کتابخانه‌های مورد نیاز را نصب کنید:

```bash
pip install -r requirements.txt
```

## نحوه استفاده

### ۱. دریافت داده‌های تاریخی (Data Collection)
از `ccxt` برای دانلود داده‌های OHLCV استفاده می‌شود.

```bash
# دانلود داده روزانه بیت‌کوین از صرافی کراکن
python scripts/download_data.py --exchange kraken --symbol BTC/USDT --timeframe 1d --limit 1000

# دانلود داده اتریوم برای جفت‌گیری
python scripts/download_data.py --exchange kraken --symbol ETH/USDT --timeframe 1d --limit 1000
```
*توجه: صرافی بایننس ممکن است در برخی مناطق مسدود باشد، به همین دلیل از کراکن به عنوان پیش‌فرض تست استفاده شده است.*

### ۲. استراتژی آربیتراژ آماری (Pairs Trading)

ابتدا جفت‌هایی که هم‌بستگی آماری دارند (Cointegration) را پیدا کنید:
```bash
python scripts/find_pairs.py
```

سپس استراتژی را روی دو جفت بک‌تست بگیرید:
```bash
python scripts/backtest_pairs.py --asset1 kraken_BTC_USDT_1d.csv --asset2 kraken_ETH_USDT_1d.csv
```

### ۳. استراتژی ماشین‌لرنینگ (ML Strategy)

ابتدا مدل را روی داده‌های یک ارز آموزش دهید. این اسکریپت اندیکاتورها را می‌سازد و یک مدل `RandomForestClassifier` را آموزش داده و ذخیره می‌کند:
```bash
python scripts/train_ml.py --data kraken_BTC_USDT_1d.csv
```

سپس بک‌تست مدل را روی داده‌های Out-of-Sample (تست نشده) اجرا کنید تا سود استراتژی در مقایسه با روش هولد کردن (Buy & Hold) بررسی شود:
```bash
python scripts/backtest_ml.py --data kraken_BTC_USDT_1d.csv
```

## ساختار پروژه

```
trading-bot/
├── scripts/
│   ├── download_data.py       # دانلود داده‌های OHLCV از طریق CCXT
│   ├── find_pairs.py          # کشف جفت‌ارزهای هم‌بسته (Cointegration Test)
│   ├── backtest_pairs.py      # بک‌تست آربیتراژ آماری
│   ├── train_ml.py            # استخراج فیچرها و آموزش مدل ML
│   └── backtest_ml.py         # بک‌تست استراتژی ML
├── src/
│   ├── config.py              # تنظیمات پروژه و مسیرها
│   └── strategies/
│       ├── pairs_trading.py   # منطق سیگنال‌های Pairs Trading (Z-Score)
│       └── ml_strategy.py     # منطق سیگنال‌دهی مدل ML
├── data/                      # داده‌های دانلود شده و مدل‌های ذخیره شده
├── config.yaml                # فایل تنظیمات
└── requirements.txt
```
