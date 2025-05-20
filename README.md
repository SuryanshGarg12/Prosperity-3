# IMC Prosperity 3 Challenge 2025 – Team Quantists

**Team Quantists** — [*Suryansh Garg*](https://github.com/SuryanshGarg12), [*Tanishq Godha*](https://github.com/Tanishq-Godha), and [*Garv Jain*](https://github.com/zengarv) — placed **87th** globally 🌐(Top 0.7%) out of 12,620+ teams (20,000+ participants) in the IMC Trading Prosperity 3 Challenge 2025, and ranked 16th in India 🇮🇳.

This repository contains the codebase we developed throughout the competition. We hope it serves as a useful resource for anyone interested in quantitative finance, algorithmic trading, or competitive trading challenges.

![Archipelago Screenshot](https://github.com/SuryanshGarg12/Prosperity-3/blob/main/images/archipelago.png "Rank 87, Quantists")

---

## Results

| Round   | Algo Score | Manual Score | **Total Score** | Rank |
|---------|------------|---------------|------------------|------|
| Round 1 | 0.000      | 44.340        | 44.340           | 2344 |
| Round 2 | 37.126     | 26.015        | 63.141           | 1180 |
| Round 3 | 234.523    | 51.500        | 286.023          | 194  |
| Round 4 | 137.892    | 61.002        | 198.894          | 82   |
| Round 5 | 12.084     | 154.505       | 166.589          | 87   |
| **Overall** | **421.625** | **337.362** | **758.987**       |  **87**    |

---

## Round 1

### RAINFOREST_RESIN – Mean Reversion with Adaptive Quoting

Rainforest Resin was the simplest asset to trade since its price reliably oscillated around 10,000. We set initial thresholds to buy at ≤ 9,999 and sell at ≥ 10,001, then introduced a dynamic “edge” that, when orders at (9,999 buy, 10,001 sell) filled consistently, widened the band to buy at ≤ 9,998 and sell at ≥ 10,002, and, if fills became infrequent, narrowed it back to the original levels.


---

### KELP – Filtered Reversion Model

KELP followed a similar reversion idea, but with added filtering to avoid large (potentially toxic) orders. We calculated a fair price using filtered mid-prices and applied a basic reversion adjustment. The strategy had three parts: take good offers, clear risky positions, and place passive quotes near the fair value. This helped avoid bad fills and manage risk better.

---

### SQUID_INK – Oscillator-Based Entries with Trailing Exit

For SQUID_INK, we used a stochastic oscillator to find possible trend reversals. Entry signals were based on %K and %D crossovers with basic price confirmation. Once in a trade, we tracked the highest/lowest price and used a 5% trailing stop to exit. This kept trades simple and avoided holding through big reversals.

---

## ⚙️ Strategy Overview – Round 2

In Round 2, we continued using the same base logic for **RAINFOREST_RESIN**, **KELP**, and **SQUID_INK**, with slight improvements. The main focus was on extending the strategy to support newly introduced assets: **PICNIC_BASKET1**, **PICNIC_BASKET2**, **CROISSANTS**, **JAMS**, and **DJEMBES**.

---

### PICNIC_BASKET1 & PICNIC_BASKET2 – Basket Arbitrage with Component Hedging

We introduced a **basket arbitrage strategy** for PICNIC_BASKET1 and PICNIC_BASKET2. The idea was to compute a theoretical price for each basket based on its components (e.g., CROISSANTS, JAMS, DJEMBES) and compare it with the live market price.

- If the basket was overpriced, we sold it and bought the components.
- If it was underpriced, we bought the basket and sold the components.
- All trades were **hedged** using component legs to keep exposure balanced.
- A **dynamic threshold** was used to account for volatility and avoid false signals.

---

### CROISSANTS, JAMS, DJEMBES – Used Only for Hedging

These products were **not traded independently**. They were used exclusively for hedging the basket positions during arbitrage trades. Volumes were calculated based on the basket composition to neutralize exposure after a PICNIC_BASKET1 or 2 trade.

---

### Minor Adjustments

- **RAINFOREST_RESIN** and **KELP**: Slight tuning to edge widths and fair value smoothing.
- **SQUID_INK**: Refined reversion model and filtered price levels to reduce overtrading during noise.

---

## 🔥 Round 3 – Volatility and Vouchers

Round 3 added new dimensions to the trading landscape, notably with **VOLCANIC_ROCK** and its **VOUCHERS**. We expanded our system while preserving all previous strategies.

---

### VOLCANIC_ROCK – Z-Score Mean Reversion

We implemented a **mean reversion strategy** using Z-score logic over a rolling 50-tick window:

- Computed rolling mean and standard deviation of mid-prices.
- If Z-score < -1.9, we entered a **long** position.
- If Z-score > +1.9, we entered a **short** position.
- Position sizing was controlled using current exposure vs. soft limits.

This approach profited from temporary overreactions in price while managing risk via thresholds.

---

### VOUCHERS – Paired Mean Reversion by Strike

Each voucher (e.g., **VOUCHER_9500**, **VOUCHER_9750**, etc.) was handled using a **strike-specific threshold** for Z-score entry:

- When price deviated significantly (Z > threshold), a trade was executed in the **opposite direction** of the deviation.
- This strategy allowed for **directional bets** when mispricings occurred in relation to VOLCANIC_ROCK.

---

### Minor Enhancements

- Expanded inventory control and Z-score tracking.
- Added volatility-aware execution and smoothing on key indicators.
- Integrated minimal persistent state with `jsonpickle` to retain history during backtesting.

---
