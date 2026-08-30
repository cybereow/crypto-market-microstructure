# Trading Bot — v2.1 (Machine Learning & Quant Strategies)

این پروژه شامل یک فریم‌ورک پیشرفته برای استراتژی‌های کمی (Quantitative) و الگوریتم‌های مبتنی بر ماشین‌لرنینگ جهت معامله در بازارهای مالی است.

## استراتژی‌های پیاده‌سازی شده

۱. **یادگیری ماشین (Machine Learning - XGBoost)**: استفاده از اندیکاتورهای پیشرفته تکنیکال (مثل MACD, RSI, ATR, Bollinger Bands از طریق کتابخانه `pandas-ta`) برای آموزش یک مدل قدرتمند **XGBoost** جهت پیش‌بینی جهت کندل بعدی. توجه: پیش‌بینی مستقیم جهت کندل بعدی روی داده‌های واقعی معمولاً دقتی نزدیک به ۵۰٪ (تصادفی) دارد — برای یک روش معتبرتر به بخش Meta-Labeling زیر مراجعه کنید.
۲. **معاملات شبکه‌ای (Grid Trading)**: یکی از امن‌ترین الگوریتم‌ها برای بازارهای خنثی (Range). این ربات محدوده‌ای از قیمت‌ها را شبکه‌بندی کرده، در افت قیمت‌ها به صورت پله‌ای خرید می‌کند و در رشد قیمت می‌فروشد تا از نوسانات کوچک سود مستمر بگیرد.
۳. **آربیتراژ آماری (Pairs Trading)**: پیدا کردن جفت‌ارزهایی که از نظر آماری هم‌بستگی بالایی دارند (Cointegrated) و معامله بر اساس انحراف از معیار (Z-Score) اسپرد آن‌ها.
۴. **Meta-Labeling (پیشنهادی)**: به‌جای پیش‌بینی مستقیم جهت قیمت، یک سیگنال ساده‌ی rule-based (شکست Donchian یا بازگشت RSI) معامله‌های کاندید تولید می‌کند و مدل XGBoost فقط پیش‌بینی می‌کند که آیا این معامله‌ی مشخص به هدف سود می‌رسد یا استاپ می‌خورد (روش Triple-Barrier). داده‌ی هر ۶ ارز با هم pool می‌شود و اعتبارسنجی به‌صورت walk-forward (چند بار retrain روی بازه‌های زمانی متوالی) انجام می‌شود.

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

### ۵. Meta-Labeling (پیشنهادی برای win rate بالاتر)

**قدم صفر — داده‌ی عمیق (مهم‌ترین قدم):** صرافی کراکن برای تایم‌فریم ۴ ساعته فقط حدود ۷۲۰ کندل برمی‌گرداند (~۴ ماه) که تنها ~۶۰ معامله‌ی کاندید تولید می‌کند — برای آموزش مدل به‌شکل ناامیدکننده‌ای کم است. آرشیو عمومی بایننس (`data.binance.vision`) تاریخچه‌ی کامل می‌دهد:

```bash
for s in BTCUSDT ETHUSDT SOLUSDT LINKUSDT AVAXUSDT DOTUSDT ADAUSDT XRPUSDT DOGEUSDT LTCUSDT ATOMUSDT; do
  python scripts/download_klines_vision.py --symbol $s --timeframe 4h --start-date 2020-01-01
done
```
نتیجه: ~۱۴٫۴۰۰ کندل به‌جای ۷۲۱ (۲۰ برابر) و ~۱۲٫۵۰۰ معامله‌ی کاندید به‌جای چند صد.

آموزش مدل (BTC به‌عنوان کانتکست رژیم بازار پاس داده می‌شود، نه به‌عنوان یک دارایی معامله‌شده):
```bash
python scripts/train_meta_ml.py \
  --data binance_ETH_USDT_4h.csv binance_SOL_USDT_4h.csv binance_LINK_USDT_4h.csv \
         binance_AVAX_USDT_4h.csv binance_DOT_USDT_4h.csv binance_ADA_USDT_4h.csv \
         binance_XRP_USDT_4h.csv binance_DOGE_USDT_4h.csv binance_LTC_USDT_4h.csv \
         binance_ATOM_USDT_4h.csv \
  --btc-regime-file binance_BTC_USDT_4h.csv \
  --signal reversion --pt-mult 2.0 --sl-mult 2.0 --target-precision 0.60
```

بک‌تست سریع روی یک ارز (نمونه‌ی کوچک، فقط برای بررسی سریع — آستانه‌ی کالیبره‌شده به‌طور خودکار خوانده می‌شود):
```bash
python scripts/backtest_meta_ml.py --data binance_ETH_USDT_4h.csv --btc-regime-file binance_BTC_USDT_4h.csv
```

اعتبارسنجی واقعی و قابل‌اتکا (walk-forward — این خروجی را ملاک قضاوت قرار دهید، نه بک‌تست تک‌ارزی بالا). جدول ablation سهم واقعی هر کامپوننت را نشان می‌دهد:
```bash
python scripts/backtest_meta_ml_walkforward.py \
  --data binance_ETH_USDT_4h.csv binance_SOL_USDT_4h.csv binance_LINK_USDT_4h.csv \
         binance_AVAX_USDT_4h.csv binance_DOT_USDT_4h.csv binance_ADA_USDT_4h.csv \
         binance_XRP_USDT_4h.csv binance_DOGE_USDT_4h.csv binance_LTC_USDT_4h.csv \
         binance_ATOM_USDT_4h.csv \
  --btc-regime-file binance_BTC_USDT_4h.csv \
  --signal reversion --pt-mult 2.0 --sl-mult 2.0 --target-precision 0.60
```

### ۶. 🎯 هدف «۹۰٪ win rate» — نتیجه‌ی اندازه‌گیری واقعی

این مهم‌ترین یافته‌ی پروژه است و مستقیماً به سؤال شما پاسخ می‌دهد.

**۹۰٪ win rate قابل دستیابی است، ولی تقریباً بی‌ارزش.** دلیلش یک اتحاد ریاضی است، نه کیفیت مدل:

```
breakeven_win_rate = 1 / (1 + pt/sl)
```

یعنی win rate اساساً توسط **هندسه‌ی حد سود/ضرر** تعیین می‌شود، نه توسط مدل. اگر حد سود را نزدیک و حد ضرر را دور بگذارید، win rate بالا «می‌خرید» — ولی هر ضرر چند برابر هر سود می‌شود.

خروجی واقعی `scripts/sweep_barrier_geometry.py` (۱۰ ارز، ۱۲٫۵۰۰ معامله، walk-forward با purge، بعد از کسر کارمزد ۰٫۴٪):

| سیگنال | pt/sl | breakeven WR | win rate (۱۰٪ برتر) | Profit Factor | نتیجه |
|---|---|---|---|---|---|
| breakout | 0.25/3.0 | **92.3%** | **91.6%** | **0.56** | ❌ win rate عالی، **پول از دست می‌دهد** |
| breakout | 0.33/3.0 | 90.1% | **91.0%** | 0.82 | ❌ ۹۱٪ برد، ولی زیان‌ده |
| breakout | 0.5/3.0 | 85.7% | 89.0% | 1.10 | ⚠️ مرزی |
| **reversion** | **2.0/2.0** | **50.0%** | **58.8%** | **1.28** | ✅ **بهترین سوددهی** |

**۹۱٪ win rate با profit factor 0.56 یعنی نابودی حساب.** در مقابل، ۵۸٫۸٪ win rate با payoff متعادل واقعاً سود می‌دهد.

برای دیدن این جدول روی داده‌ی خودتان:
```bash
python scripts/sweep_barrier_geometry.py \
  --data binance_ETH_USDT_4h.csv binance_SOL_USDT_4h.csv ... \
  --btc-regime-file binance_BTC_USDT_4h.csv
```

**پس هدف درست چیست؟** به‌جای «۹۰٪ win rate»، هدف را مشترکاً بیان کنید:
> win rate ایکس، با **profit factor > 1.3** بعد از کسر هزینه‌ها، روی **حداقل ۲۰۰ معامله‌ی walk-forward**.

### ۷. چهار کامپوننت بهبود win rate و سهم واقعی هرکدام

هر چهار ایده پیاده‌سازی و به‌صورت مستقل قابل خاموش‌کردن هستند تا سهمشان **اندازه‌گیری** شود، نه فرض:

| کامپوننت | فایل | کاری که می‌کند |
|---|---|---|
| ۱. رگ رژیم بازار (BTC alignment) | `src/regime.py` | فیچر `btc_alignment = sign(side) == sign(BTC_trend)` را **صریحاً** می‌سازد، به‌جای امید به کشف این تعامل توسط درخت |
| ۲. آستانه‌ی confidence دینامیک | `src/gating.py` | خلاف جهت BTC → آستانه بالاتر؛ هم‌جهت → پایین‌تر |
| ۳. کالیبراسیون آستانه بر مبنای precision | `src/calibration.py` | جایگزین `scoring='f1'`؛ بدون ریسک custom gradient |
| ۴. فیلتر OOD با leaf-frequency | `src/novelty.py` | از `pred_leaf` همان مدل XGBoost — بدون مدل دوم |

**جدول ablation واقعی** (reversion، pt/sl=2.0/2.0، ۱۰ ارز، ۳٫۳۶۲ معامله‌ی walk-forward، بعد از هزینه):

| پیکربندی | تعداد | win rate | PF | بازده |
|---|---|---|---|---|
| فقط سیگنال اولیه (بدون مدل) | 2734 | 48.9% | 1.09 | +13.9% |
| + آستانه‌ی ثابت 0.55 (روش قبلی) | 609 | 53.0% | 1.21 | +36.2% |
| + آستانه‌ی کالیبره‌شده (ایده ۳) | 389 | 57.6% | 1.38 | +47.0% |
| + آستانه‌ی دینامیک BTC (ایده ۱+۲) | 133 | **60.2%** | 1.69 | +37.2% |
| + فیلتر OOD (ایده ۴) | 383 | 57.4% | 1.39 | +47.4% |
| **+ همه (گیت کامل)** | **130** | **60.0%** | **1.74** | +38.9% |

بهبود واقعی: **48.9% → 60.0% win rate** و **PF 1.09 → 1.74**.

شاهد اینکه ایده‌ها واقعاً کار می‌کنند (از خروجی همین اجراها):
- **۵ فیچر از ۱۰ فیچر مهم مدل، فیچرهای رژیم BTC هستند** (`btc_above_sma`, `btc_ret_5_aligned`, `btc_vol`, `btc_alignment`, `btc_alignment_strength`) — ایده ۱ تأیید شد.
- معامله‌های هم‌جهت با BTC: win rate **55.0%** / خلاف جهت: **48.8%** — ایده ۲ تأیید شد.
- مسیرهای leaf آشنا: **49.6%** / مسیرهای نادر: **41.7%** — ایده ۴ تأیید شد؛ مدل در شرایط ناشناخته واقعاً بدتر عمل می‌کند.

⚠️ **هشدار صداقت علمی:** `corr(p_win, label)` حدود **۰٫۰۸** است. یعنی قدرت رتبه‌بندی مدل واقعی ولی **ضعیف** است. با تعداد فولد کم، بخشی از بهبود بالا می‌تواند شانس باشد — به‌همین دلیل ستون «تعداد» همیشه کنار win rate چاپ می‌شود. عدد ۱۳۰ معامله برای نتیجه‌گیری قطعی کم است.

استراتژی cross-sectional (رتبه‌بندی ارزها نسبت به هم با استفاده از امتیاز مدل، long-short بین بهترین و بدترین):
```bash
python scripts/backtest_cross_sectional.py --data kraken_BTC_USDT_4h.csv kraken_ETH_USDT_4h.csv kraken_SOL_USDT_4h.csv --long-short
```

برای افزودن فیچر نرخ فاندینگ (داده‌ی جایگزین، مستقل از قیمت/حجم) از آرشیو عمومی بایننس:
```bash
python scripts/download_funding_vision.py --symbol BTCUSDT --start-date 2022-01-01 --end-date 2026-01-01 --out kraken_BTC_USDT_4h_funding.csv
python scripts/train_meta_ml.py --use-funding --data kraken_BTC_USDT_4h.csv ...
```

## ساختار پروژه

```
trading-bot/
├── scripts/
│   ├── download_data.py               # دانلود داده‌های OHLCV از طریق CCXT
│   ├── download_klines_vision.py      # دانلود تاریخچه‌ی عمیق OHLCV از آرشیو بایننس (۲۰ برابر کراکن)
│   ├── sweep_barrier_geometry.py      # جاروب هندسه‌ی حد سود/ضرر: کجا win rate بالا واقعاً سودده است
│   ├── download_funding_vision.py     # دانلود تاریخچه‌ی نرخ فاندینگ از آرشیو عمومی بایننس
│   ├── find_pairs.py                  # کشف جفت‌ارزهای هم‌بسته (Cointegration Test)
│   ├── backtest_pairs.py              # بک‌تست آربیتراژ آماری
│   ├── train_ml.py                    # استخراج فیچرهای پیشرفته با pandas-ta و آموزش XGBoost
│   ├── backtest_ml.py                 # بک‌تست استراتژی XGBoost (پیش‌بینی مستقیم جهت)
│   ├── train_meta_ml.py               # آموزش pool‌شده‌ی مدل Meta-Labeling (Triple-Barrier)
│   ├── backtest_meta_ml.py            # بک‌تست تک‌ارزی مدل Meta-Labeling
│   ├── backtest_meta_ml_walkforward.py# اعتبارسنجی walk-forward مدل Meta-Labeling
│   ├── backtest_cross_sectional.py    # استراتژی رتبه‌بندی cross-sectional بین ارزها
│   └── backtest_grid.py               # بک‌تست ربات Grid Trading
├── src/
│   ├── config.py               # تنظیمات پروژه و مسیرها
│   ├── metrics.py              # محاسبه‌ی win rate استاندارد (بر اساس معامله‌ی بسته‌شده)
│   ├── labeling.py             # سیگنال‌های اولیه‌ی rule-based و برچسب‌گذاری Triple-Barrier
│   ├── regime.py               # ایده ۱: کانتکست رژیم بازار و فیچر btc_alignment
│   ├── calibration.py          # ایده ۳: کالیبراسیون آستانه بر مبنای precision (جای F1)
│   ├── novelty.py              # ایده ۴: تشخیص OOD با leaf-frequency (بدون مدل دوم)
│   ├── gating.py               # ایده ۲: لایه‌ی تصمیم با آستانه‌ی دینامیک
│   └── strategies/
│       ├── pairs_trading.py   # منطق سیگنال‌های Pairs Trading (Z-Score)
│       ├── ml_strategy.py     # منطق سیگنال‌دهی مدل ML
│       └── grid_trading.py    # منطق شبکه‌بندی و معاملات پله‌ای Grid
├── data/                      # داده‌های دانلود شده و مدل‌های ذخیره شده
├── config.yaml                # فایل تنظیمات
└── requirements.txt
```