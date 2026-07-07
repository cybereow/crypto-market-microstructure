# Trading Bot — v0.1

ساده، مرحله‌ای، با research framework. این نسخه ادغام دو تلاش قبلیه:
یکی معماری کلی (research/ per idea, Strategy interface مشترک، فازبندی
backtest → forward-test → paper → live)، یکی هم منطق واقعی بک‌تست
funding arbitrage که قبلاً با داده‌ی واقعی تست و تأیید شده بود.

## چرا این ساختار

تلاش قبلی روی همین ایده به این دلیل شکست خورد که برای گرفتن تاریخچه‌ی
funding rate از API زنده‌ی `fapi.binance.com` (از طریق ccxt) استفاده کرده
بود، که از خیلی از محیط‌ها geo-block می‌شه. راه‌حل: تاریخچه‌ی funding و
کندل‌های روزانه رو از آرشیو عمومی و بدون‌نیاز-به-کلید
`data.binance.vision` می‌گیریم (`src/binance_funding.py`,
`src/binance_klines.py`). ccxt (`src/collector.py`) فقط برای داده‌ی
زنده‌ی فازهای بعدی (paper/live) استفاده می‌شه، نه برای بک‌تست.

## ساختار پوشه

```
trading-bot/
├── research/
│   └── funding/
│       ├── hypothesis.md
│       └── results.md
├── src/
│   ├── config.py             # از config.yaml می‌خونه
│   ├── binance_funding.py     # تاریخچه‌ی واقعی funding rate (data.binance.vision)
│   ├── binance_klines.py      # کندل روزانه‌ی واقعی، برای چک لیکوئیدیشن
│   ├── storage.py             # SQLite برای داده‌ی زنده (v0.3+)
│   ├── collector.py           # جمع‌آوری زنده via ccxt (v0.3+)
│   ├── strategy.py            # Strategy interface مشترک
│   ├── backtester.py          # بک‌تست عمومی trade-signal (برای basis/grid بعدی)
│   └── strategies/
│       └── funding.py         # اقتصاد واقعی funding arb: فی به تفکیک پا، leverage، rolling window، pre-entry calculator
├── scripts/
│   ├── run_backtest.py
│   └── plan_entry.py
├── data/                       # cache و db (gitignored)
└── config.yaml
```

## اجرا

```bash
pip install -r requirements.txt

# بک‌تست کامل (BTCUSDT + ETHUSDT، 2020-الان)
python scripts/run_backtest.py

# سریع‌تر: فقط BTC، فقط ۲ سال اخیر
python scripts/run_backtest.py --symbols BTCUSDT --start 2024-01 --end 2026-06

# با leverage روی پای فیوچرز (اسپات همیشه ۱x می‌مونه)
python scripts/run_backtest.py --symbols BTCUSDT --leverage 2

# قبل از باز کردن پوزیشن واقعی: مبلغ دقیق هر پا، قیمت لیکوئیدیشن، چک‌لیست
python scripts/plan_entry.py --capital 7000 --leverage 2 --symbol BTCUSDT
```

نتیجه توی `research/funding/output/funding_summary.json` و
`research/funding/output/{SYMBOL}_funding_history.csv`.

## چطور نتیجه رو بخونی

- `naive_annualized_pct`: میانگین کل تاریخچه، سالانه‌شده — فرض اینکه این
  میانگین برای همیشه ادامه پیدا کنه.
- `apy.net_apy_pct` (taker/maker): بازده خالص واقعی روی سرمایه، بعد از
  کارمزد واقعی هر دو پا (نیمی اسپات، نیمی مارجین شورت فیوچرز ۱x).
- `rolling` (پنجره‌ی ۹۰ روزه): مهم‌ترین بخش — نشون می‌ده اگه دقیقاً الان
  وارد بشی بازده واقعی چقدر نوسان داره (`min_pct` تا `max_pct`) و
  `pct_windows_negative` یعنی چند درصد دوره‌ها اصلاً ضرر بوده.

## Leverage

فقط روی پای شورت فیوچرز اعمال می‌شه؛ پای اسپات همیشه بدون‌اهرمه. سقف
ریاضی بازده حدود ۲ برابر حالت L=1 هست (نه بیشتر)، چون نصف سرمایه همیشه
توی اسپات گیره. با leverage بالای ۱، بک‌تست خودکار کندل‌های روزانه‌ی
واقعی رو می‌کشه بیرون و بدترین حرکت صعودی تاریخی (۱، ۷، ۳۰ روزه) رو با
حاشیه‌ی لیکوئیدیشن مقایسه می‌کنه.

## Research Framework

```
Idea
 ↓
research/{name}/hypothesis.md   ← فرضیه بنویس
 ↓
run_backtest.py                 ← با داده‌ی واقعی
 ↓
research/{name}/results.md      ← نتیجه بنویس
 ↓
Forward Test                    ← همون backtest روی داده‌ی جدید (out-of-sample)
 ↓
Paper Trade                     ← با پول مجازی (v0.3 — هنوز ساخته نشده)
 ↓
Small Capital ($100)            ← live (v1 — هنوز ساخته نشده)
 ↓
Scale                           ← (v2)
```

## چک‌لیست

### v0.1 ← الان اینجایی
```
[x] binance_funding.py / binance_klines.py داده‌ی واقعی می‌گیرن
[x] strategies/funding.py اقتصاد واقعی (فی، leverage، rolling) رو حساب می‌کنه
[x] plan_entry.py قبل از ورود واقعی محاسبه می‌کنه
[ ] research/funding/hypothesis.md نهایی شده
[ ] research/funding/results.md با نتیجه‌ی واقعی پر شده
```

### v0.2 — Signal Generator (out-of-sample)
```
[ ] forward_test روی بازه‌ی جدید اجرا شده
[ ] نتیجه با v0.1 مقایسه شده (drift دیده می‌شه؟)
```

### v0.3 — Paper Trading
```
[ ] یه PaperTrader مخصوص cash-flow (نه tick-buy/sell) ساخته بشه — چون
    funding arb یه پوزیشن باز-نگه‌دار-ببند هست، نه دنباله‌ای از معاملات
[ ] ۳۰ روز اجرا شده با collector.py زنده
[ ] Slippage واقعی تخمین زده شده
```

### v1 — Live ($100)
```
[ ] Paper trading موفق بوده
[ ] ۷ روز بدون مشکل اجرا شده
[ ] Kill switch و alert روی قیمت لیکوئیدیشن تست شده
```

### v2
```
[ ] استراتژی‌های جدید (basis, grid) اضافه شدن — از Strategy/Backtester عمومی استفاده می‌کنن
```
