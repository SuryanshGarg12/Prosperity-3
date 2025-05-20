# IMC Prosperity 3 Challenge 2025 – Team Quantists

**Team Quantists** — [*Suryansh Garg*](https://github.com/SuryanshGarg12), [*Tanishq Godha*](https://github.com/Tanishq-Godha), and [*Garv Jain*](https://github.com/zengarv) — placed **87th** globally 🌐(Top 0.7%) out of 12,620+ teams (20,000+ participants) in the IMC Trading Prosperity 3 Challenge 2025, and ranked 16th in India 🇮🇳.

This repository contains the codebase we developed throughout the competition. We hope it serves as a useful resource for anyone interested in quantitative finance, algorithmic trading, or competitive trading challenges.

![Archipelago Screenshot](https://github.com/SuryanshGarg12/Prosperity-3/blob/main/images/archipelago.png "Rank 87, Quantists")

---

## Round 1

### RAINFOREST_RESIN – Mean Reversion with Adaptive Quoting

For RAINFOREST_RESIN, we used a simple mean-reversion strategy. A rolling average of the mid-price (over the last 10 ticks) was used as a fair value estimate. If the market offered prices significantly away from this fair value, we took them using market orders. Otherwise, we placed passive limit orders slightly outside the fair value. Position size was monitored, and quotes were adjusted if the position became too large.

---

### KELP – Filtered Reversion Model

KELP followed a similar reversion idea, but with added filtering to avoid large (potentially toxic) orders. We calculated a fair price using filtered mid-prices and applied a basic reversion adjustment. The strategy had three parts: take good offers, clear risky positions, and place passive quotes near the fair value. This helped avoid bad fills and manage risk better.

---

### SQUID_INK – Oscillator-Based Entries with Trailing Exit

For SQUID_INK, we used a stochastic oscillator to find possible trend reversals. Entry signals were based on %K and %D crossovers with basic price confirmation. Once in a trade, we tracked the highest/lowest price and used a 5% trailing stop to exit. This kept trades simple and avoided holding through big reversals.

---

