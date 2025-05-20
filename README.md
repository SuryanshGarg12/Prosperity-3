# IMC Prosperity 3 Challenge 2025 – Team Quantists

**Team Quantists** — [*Suryansh Garg*](https://github.com/SuryanshGarg12), [*Tanishq Godha*](https://github.com/Tanishq-Godha), and [*Garv Jain*](https://github.com/zengarv) — placed **87th** globally 🌐(Top 0.7%) out of 12,620+ teams (20,000+ participants) in the IMC Trading Prosperity 3 Challenge 2025, and ranked 16th in India 🇮🇳.

This repository contains the codebase we developed throughout the competition. We hope it serves as a useful resource for anyone interested in quantitative finance, algorithmic trading, or competitive trading challenges.

![Archipelago Screenshot](https://github.com/SuryanshGarg12/Prosperity-3/blob/main/images/archipelago.png "Rank 87, Quantists")

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

## Round 2

In Round 2, we continued using the same base logic for **RAINFOREST_RESIN**, **KELP**, and **SQUID_INK**, with slight improvements and added support for new products introduced in this round.


### PICNIC_BASKET1 & PICNIC_BASKET2 – Basket Arbitrage with Hedging

New to this round were two basket products made of underlying components like CROISSANTS, JAMS, and DJEMBES. We implemented a **arbitrage strategy**: calculating the fair value of the basket using its components, comparing it with the market price, and trading when the mispricing crossed a dynamic threshold. Basket orders were always **hedged** using component legs to reduce directional risk.

---




