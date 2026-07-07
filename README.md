# Trading Bot — v0.1 (Tabdeal, single-exchange)

استراتژی: **آربیتراژ مثلثی درون خودِ تبدیل** (X/USDT × USDT/IRT در مقابل X/IRT واقعی).

## مسیری که تا اینجا طی شد (برای این‌که دوباره تکرار نشه)

۱. اول فرض شد تبدیل مثل بایننس funding rate داره → غلط بود (مستندات رسمی
تأیید کرد: `fapi` نه `fundingRate` endpoint داره نه `FUNDING_FEE` توی
`incomeType`ها؛ مکانیزم واقعی بهره‌ی وام مارجینه).
۲. بعد آربیتراژ بین‌صرافی‌ای تبدیل↔نوبیتکس پیشنهاد شد → کاربر رد کرد
(نوبیتکس نمی‌خواد).
۳. الان: **آربیتراژ مثلثی تک‌صرافی** — فقط تبدیل، بدون فرض روی یه دارایی
خاص مثل PAXG (که اصلاً توی مستندات رسمی تبدیل نیومده و وجودش تأیید نشده).
دارایی‌های واجد شرایط از خودِ `exchangeInfo` زنده کشف می‌شن.

## هدف و محدودیت واقع‌بینانه

هدف کاربر: حداقل **$۲۰/ماه** با سرمایه‌ی زیر $۵۰۰ (یعنی ~۴-۶٪ ماهانه).
این عدد برای یه استراتژی آربیتراژ کم‌ریسک هدف بالاییه. راهکار اینجا برای
بالا بردن شانس رسیدن به این عدد: اسکن هم‌زمان ده‌ها دارایی (نه فقط یه
جفت‌ارز ثابت) تا فرصت بیشتر پیدا بشه — ولی این فقط بعد از جمع‌آوری
داده‌ی واقعی (نه قبلش) قابل تأییده.

## ⚠️ محدودیت مهم این build

محیطی که این کد توش نوشته شده نتونست به `api1.tabdeal.org` وصل بشه (هر
تلاش HTTP 503 داد). **این کد از این محیط تست نشده.** قبل از هر تصمیمی:
```bash
python scripts/scan_triangular.py --once
```
رو از جایی که واقعاً قراره بات اجرا بشه بزن و مطمئن شو داده‌ی واقعی
(نه خطا) برمی‌گرده.

## ساختار پوشه

```
trading-bot/
├── research/
│   └── triangular_arbitrage/
│       ├── hypothesis.md
│       └── results.md
├── src/
│   ├── config.py                       # از config.yaml می‌خونه
│   ├── storage.py                       # SQLite برای snapshot ها (تنها منبع تاریخچه)
│   ├── strategy.py                      # Strategy interface مشترک
│   ├── backtester.py                    # بک‌تست عمومی trade-signal (برای استراتژی‌های بعدی)
│   ├── exchanges/
│   │   └── tabdeal.py                    # کلاینت واقعی depth + exchangeInfo
│   └── strategies/
│       └── triangular_arbitrage.py       # کشف دارایی‌های واجد شرایط + محاسبه‌ی edge
├── scripts/
│   ├── scan_triangular.py                # مانیتور زنده + ذخیره‌ی snapshot
│   └── analyze_triangular.py             # تحلیل آماری snapshot های جمع‌شده
├── data/                                  # db (gitignored)
└── config.yaml
```

## اجرا

```bash
pip install -r requirements.txt

# مانیتور زنده (هر ۳۰ ثانیه، طبق config.yaml) — بدون ثبت سفارش
python scripts/scan_triangular.py

# فقط یه دور اسکن و خروج (برای تست اتصال)
python scripts/scan_triangular.py --once

# بعد از چند روز جمع‌آوری، تحلیل آماری
python scripts/analyze_triangular.py
python scripts/analyze_triangular.py --base-asset BTC
```

## منطق محاسبه

برای هر دارایی X که هم بازار `X/USDT` هم `X/IRT` داره:

- **مسیر A** (IRT→X→USDT→IRT): بخر X با IRT، بفروش X به USDT، بفروش USDT به IRT
  `gross = X_USDT_bid * USDT_IRT_bid / X_IRT_ask`
- **مسیر B** (IRT→USDT→X→IRT): بخر USDT با IRT، بخر X با USDT، بفروش X به IRT
  `gross = X_IRT_bid / (USDT_IRT_ask * X_USDT_ask)`

هرکدوم gross بیشتری داشت انتخاب می‌شه؛ `net_edge_bps` بعد از کسر کارمزد
هر ۳ پا (taker) حساب می‌شه.

## چطور نتیجه رو بخونی

- `net_edge_bps`: سود خالص هر چرخه‌ی کامل، بعد از ۳ پا کارمزد. اسلیپیج و
  محدودیت عمق واقعی بازار لحاظ نشده — عدد روی ۵ ردیف اول orderbook حسابه.
- `pct_time_viable` (از `analyze_triangular.py`): چند درصد از زمان
  جمع‌آوری‌شده، حداقل یه دارایی فرصت واقعی داشته.
- `top_assets_by_max_edge`: اگه یه دارایی خاص مدام بالای لیست بود، ارزش
  بررسی داره که آیا دلیل ساختاری داره (نه فقط نویز).

## Research Framework

```
Idea
 ↓
research/{name}/hypothesis.md   ← فرضیه بنویس
 ↓
scan_triangular.py               ← چون آرشیو عمومی نیست، تاریخچه رو خودت می‌سازی
 ↓
analyze_triangular.py            ← بعد از چند روز جمع‌آوری
 ↓
research/{name}/results.md       ← نتیجه بنویس، تصمیم بگیر VIABLE/NOT
 ↓
Paper Trade                      ← با پول مجازی (v0.3 — هنوز ساخته نشده)
 ↓
Small Capital                    ← live (v1 — هنوز ساخته نشده)
 ↓
Scale                            ← (v2)
```

## چک‌لیست

### v0.1 ← الان اینجایی
```
[x] tabdeal.py با exchangeInfo + depth واقعی کار می‌کنه
[x] triangular_arbitrage.py دارایی‌های واجد شرایط رو کشف و edge رو حساب می‌کنه
[x] scan_triangular.py snapshot ها رو ذخیره می‌کنه
[ ] از محیط واقعی deployment تست شده (این build نتونست — 503 همه‌جا)
[ ] rate limit واقعی API تبدیل چک شده (max_symbols/request_delay فعلاً حدسیه)
[ ] چند روز داده‌ی واقعی جمع شده و research/triangular_arbitrage/results.md پر شده
```

### v0.2
```
[ ] بررسی عمق واقعی بازار در اندازه‌ی معامله‌ی واقعی (نه فقط ۵ ردیف اول)
[ ] بررسی طول عمر فرصت‌ها — واسه دستی گرفتنشون به‌اندازه‌ی کافی طولانی هستن؟
```

### v0.3 — Paper Trading
```
[ ] PaperTrader که ۳ پای معامله رو شبیه‌سازی کنه
[ ] حساب واقعی روی تبدیل — بدون سفارش واقعی
```

### v1 — Live
```
[ ] سرمایه‌ی کوچیک (زیر $۵۰۰)، اجرای واقعی
[ ] ردیابی این‌که واقعاً به $۲۰/ماه می‌رسه یا نه
[ ] Kill switch اگه edge برعکس شد وسط اجرای ۳ پا
```

### v2
```
[ ] max_symbols رو بعد از تأیید rate limit افزایش بده (اسکن کل ~۵۰۰+ بازار)
[ ] اگه edge کافی نبود، سراغ market making روی جفت‌ارزهای پراسپرد IRT برو
```
