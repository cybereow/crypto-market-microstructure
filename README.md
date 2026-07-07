# Trading Bot — v0.1 (Tabdeal)

استراتژی: **آربیتراژ بین‌صرافی‌ای اسپات (Tabdeal vs Nobitex)**.

## چرا این استراتژی، نه funding-rate arbitrage

نسخه‌ی اول این پروژه (که موجوده توی history) فرض می‌کرد تبدیل مثل بایننس
یه مکانیزم funding rate واقعی داره (Long Spot + Short Perp → جمع‌آوری
funding). این فرض غلط بود — از مستندات رسمی `docs.tabdeal.org` تأیید شد:

- محصول «اهرم حرفه‌ای» (`fapi`) هیچ `fundingRate`/`premiumIndex` endpoint
  ای نداره.
- `incomeType` های واقعی `/fapi/v1/income` این‌ها هستن:
  `Transfer, TakerCommission, MakerCommission, TradePNL, AdlPNL, Liquidation, InsuranceFund`
  — هیچ `FUNDING_FEE` توش نیست.
- مکانیزم واقعی هزینه‌ی نگه‌داشتن پوزیشن، بهره‌ی وام مارجینه
  (`GET /api/v1/margin/interestHistory`، انواع `ON_BORROW`/`PERIODIC`)، نه
  پرداخت periodic بین لانگ و شورت. یعنی چیزی برای «جمع‌آوری» وجود نداره.
- ضمناً `ccxt` اصلاً کلاس تبدیل رو نداره (`ccxt.exchanges` رو چک کردم — از
  ۱۰۵ صرافی، هیچ‌کدوم تبدیل نیست)، پس `collector.py` مبتنی بر ccxt هم از
  اول کار نمی‌کرد.

آربیتراژ بین‌صرافی‌ای اسپات به هیچ‌کدوم این‌ها وابسته نیست — فقط به
order book عمومی (بدون auth) هر دو صرافی نیاز داره، که واقعی و
تأییدشده‌ست:

```
GET https://api1.tabdeal.org/r/api/v1/depth?symbol=USDTIRT&limit=N   [تأییدشده از docs.tabdeal.org]
GET https://apiv2.nobitex.ir/v3/orderbook/USDTIRT                    [تأییدشده از github.com/nobitex/docs-api]
```

## ⚠️ محدودیت مهم این build

محیطی که این کد توش نوشته شده نتونست به هیچ‌کدوم از این دو API متصل بشه —
هر تلاش (هم برای تبدیل، هم برای نوبیتکس، هم برای دامنه‌های ایرانی دیگه)
با HTTP 503 مواجه شد. **این کد از این محیط تست نشده.** قبل از هر تصمیمی،
خودت `scripts/scan_irt_arb.py` رو از جایی که واقعاً قراره بات اجرا بشه
اجرا کن و مطمئن شو داده‌ی واقعی (نه خطا) برمی‌گرده.

همچنین کارمزد Nobitex توی `config.yaml` («۲۵ bps») **فقط placeholder**
هست — قبل از اعتماد به `net_edge_bps` باید با نرخ واقعی جایگزینش کنی.

## ساختار پوشه

```
trading-bot/
├── research/
│   └── irt_arbitrage/
│       ├── hypothesis.md
│       └── results.md
├── src/
│   ├── config.py                    # از config.yaml می‌خونه
│   ├── storage.py                    # SQLite برای snapshot های اسپرد (تنها منبع تاریخچه — آرشیو عمومی وجود نداره)
│   ├── strategy.py                   # Strategy interface مشترک
│   ├── backtester.py                 # بک‌تست عمومی trade-signal (برای استراتژی‌های بعدی)
│   ├── exchanges/
│   │   ├── tabdeal.py                 # کلاینت واقعی depth تبدیل
│   │   └── nobitex.py                 # کلاینت واقعی orderbook نوبیتکس
│   └── strategies/
│       └── irt_arbitrage.py           # محاسبه‌ی اسپرد، جهت معامله، net edge بعد از کارمزد
├── scripts/
│   ├── scan_irt_arb.py                # مانیتور زنده + ذخیره‌ی snapshot
│   └── analyze_irt_arb.py             # تحلیل آماری snapshot های جمع‌شده
├── data/                               # db (gitignored)
└── config.yaml
```

## اجرا

```bash
pip install -r requirements.txt

# مانیتور زنده (هر ۱۵ ثانیه، طبق config.yaml) — بدون ثبت سفارش
python scripts/scan_irt_arb.py

# فقط یه snapshot و خروج (برای تست اتصال)
python scripts/scan_irt_arb.py --once

# بعد از چند روز جمع‌آوری، تحلیل آماری
python scripts/analyze_irt_arb.py
```

## چطور نتیجه رو بخونی

- `net_edge_bps`: اسپرد بین دو صرافی بعد از کسر کارمزد هر دو (taker).
  هزینه‌ی انتقال بین صرافی (کارمزد + زمان + ریسک نوسان قیمت دارایی انتقال،
  مثلاً TRX) توش لحاظ نشده — باید جدا اضافه بشه.
- `pct_time_viable` (از `analyze_irt_arb.py`): چند درصد از زمانِ جمع‌آوری‌شده
  اسپرد بالای threshold بوده. اگه فقط چندتا spike نادر باشه، این استراتژی
  عملاً قابل‌اجرا نیست (فرصت رد می‌شه قبل از این‌که بتونی دو تا سفارش رو
  دستی/حتی برنامه‌ای اجرا کنی).

## Research Framework

```
Idea
 ↓
research/{name}/hypothesis.md   ← فرضیه بنویس
 ↓
scan_irt_arb.py                  ← چون آرشیو عمومی نیست، تاریخچه رو خودت می‌سازی
 ↓
analyze_irt_arb.py               ← بعد از چند روز جمع‌آوری
 ↓
research/{name}/results.md       ← نتیجه بنویس
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
[x] tabdeal.py / nobitex.py با endpoint های واقعی و تأییدشده کار می‌کنن
[x] irt_arbitrage.py اسپرد و net edge رو حساب می‌کنه
[x] scan_irt_arb.py snapshot ها رو ذخیره می‌کنه
[ ] از محیط واقعی deployment تست شده (این build نتونست — 503 همه‌جا)
[ ] کارمزد واقعی Nobitex جایگزین placeholder شده
[ ] چند روز داده‌ی واقعی جمع شده و research/irt_arbitrage/results.md پر شده
```

### v0.2
```
[ ] هزینه‌ی انتقال (کارمزد TRX + زمان + slippage) به مدل net edge اضافه شده
[ ] بررسی شده چند تا از فرصت‌ها واقعاً long enough بودن که یه ترید دستی/API بگیره
```

### v0.3 — Paper Trading
```
[ ] یه PaperTrader که واقعاً دو تا سفارش شبیه‌سازی‌شده (یکی هر صرافی) رو track کنه
[ ] حساب واقعی (یا تست) روی هر دو صرافی — بدون سفارش واقعی
```

### v1 — Live
```
[ ] سرمایه‌ی کوچیک، اجرای واقعی روی هر دو صرافی
[ ] Kill switch اگه اسپرد برعکس شد وسط ترید
```

### v2
```
[ ] صرافی سوم اضافه بشه (اگه edge بیشتری داشت)
[ ] استراتژی‌های دیگه‌ی لیست (PAXG triangular, market making روی IRT) با همین Strategy/Backtester
```
