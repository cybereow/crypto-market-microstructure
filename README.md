# Trading Bot — v2.1 (Machine Learning & Quant Strategies)

این پروژه شامل یک فریم‌ورک پیشرفته برای استراتژی‌های کمی (Quantitative) و الگوریتم‌های مبتنی بر ماشین‌لرنینگ جهت معامله در بازارهای مالی است.

## استراتژی‌های پیاده‌سازی شده

۱. **یادگیری ماشین (Machine Learning - XGBoost)**: استفاده از اندیکاتورهای پیشرفته تکنیکال (مثل MACD, RSI, ATR, Bollinger Bands از طریق کتابخانه `pandas-ta`) برای آموزش یک مدل قدرتمند **XGBoost** جهت پیش‌بینی جهت کندل بعدی. این استراتژی سود بسیار بالاتری نسبت به روش Buy & Hold تولید می‌کند.
۲. **معاملات شبکه‌ای (Grid Trading)**: یکی از امن‌ترین الگوریتم‌ها برای بازارهای خنثی (Range). این ربات محدوده‌ای از قیمت‌ها را شبکه‌بندی کرده، در افت قیمت‌ها به صورت پله‌ای خرید می‌کند و در رشد قیمت می‌فروشد تا از نوسانات کوچک سود مستمر بگیرد.
۳. **آربیتراژ آماری (Pairs Trading)**: پیدا کردن جفت‌ارزهایی که از نظر آماری هم‌بستگی بالایی دارند (Cointegrated) و معامله بر اساس انحراف از معیار (Z-Score) اسپرد آن‌ها.

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

### ۲. استراتژی ماشین‌لرنینگ (XGBoost)

ابتدا مدل را روی داده‌های یک ارز آموزش دهید. این اسکریپت ده‌ها فیچر حرفه‌ای می‌سازد و مدل را آموزش داده و ذخیره می‌کند:
```bash
python scripts/train_ml.py --data kraken_BTC_USDT_1d.csv
```

سپس بک‌تست مدل را روی داده‌های Out-of-Sample اجرا کنید:
```bash
python scripts/backtest_ml.py --data kraken_BTC_USDT_1d.csv
```

### ۳. استراتژی معاملات شبکه‌ای (Grid Trading)

برای بررسی عملکرد ربات نوسان‌گیر روی بازارهای رنج، اسکریپت زیر را اجرا کنید. تنظیماتی مثل تعداد گرید و دامنه شبکه قابل تنظیم است:
```bash
python scripts/backtest_grid.py --data kraken_BTC_USDT_1d.csv --grids 10 --range-pct 0.2
```

### ۴. استراتژی آربیتراژ آماری (Pairs Trading)

ابتدا جفت‌هایی که هم‌بستگی آماری دارند (Cointegration) را پیدا کنید:
```bash
python scripts/find_pairs.py
```

سپس استراتژی را روی دو جفت بک‌تست بگیرید:
```bash
python scripts/backtest_pairs.py --asset1 kraken_BTC_USDT_1d.csv --asset2 kraken_ETH_USDT_1d.csv
```

## ساختار پروژه

```
trading-bot/
├── scripts/
│   ├── download_data.py       # دانلود داده‌های OHLCV از طریق CCXT
│   ├── find_pairs.py          # کشف جفت‌ارزهای هم‌بسته (Cointegration Test)
│   ├── backtest_pairs.py      # بک‌تست آربیتراژ آماری
│   ├── train_ml.py            # استخراج فیچرهای پیشرفته با pandas-ta و آموزش XGBoost
│   ├── backtest_ml.py         # بک‌تست استراتژی XGBoost
│   └── backtest_grid.py       # بک‌تست ربات Grid Trading
├── src/
│   ├── config.py              # تنظیمات پروژه و مسیرها
│   └── strategies/
│       ├── pairs_trading.py   # منطق سیگنال‌های Pairs Trading (Z-Score)
│       ├── ml_strategy.py     # منطق سیگنال‌دهی مدل ML
│       └── grid_trading.py    # منطق شبکه‌بندی و معاملات پله‌ای Grid
├── data/                      # داده‌های دانلود شده و مدل‌های ذخیره شده
├── config.yaml                # فایل تنظیمات
└── requirements.txt
```