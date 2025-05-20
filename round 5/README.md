## Round 5 Updates

Round 5 brings targeted enhancements to options strategies, robust delta-neutral hedging, and a novel implied volatility surface model to detect mispricing more accurately.

### Implied Vol Surface–Driven Trading (IV Curve Arbitrage)

**Simple Overview:**

* Uses a quadratic model of implied volatility as a function of **moneyness** to identify over- and underpriced vouchers.
* Compares each voucher’s **observed implied volatility** (via inverse Black-Scholes) to a fitted surface (`IV = a + b·m + c·m²`) where *m* = log(strike/underlying) / sqrt(T).
* When observed IV diverges significantly from modeled IV, triggers directional trades in the option **and** delta hedge in the underlying.

**Key Additions:**

* `trade_9500`, `trade_10000`, `trade_10500`: voucher-specific trading logic with surface coefficients and thresholds.
* Tracks a short moving average of recent IV deviations to avoid overfitting to short-term noise.
* **Delta Hedge**: Option trades are always accompanied by the appropriate **hedge in Volcanic Rock** to maintain delta-neutral exposure.

**Threshold Example:**

* If observed IV − fair IV > threshold → short the voucher (sell option), buy underlying.
* If observed IV − fair IV < −threshold → long the voucher, sell underlying.

---

### Expanded MarketData for Coordinated Execution

**Purpose:** Enables coordinated, multi-asset execution across voucher and underlying markets.

**Key Structures:**

* `end_pos`, `buy_sum`, `sell_sum`: Net position and depth-aware volume tracking across assets.
* `ask_prices`, `ask_volumes`, `bid_prices`, `bid_volumes`: Full book snapshots to simulate realistic fills for basket-style execution.

**Use Case:**

* Voucher trading routines (e.g., `trade_10000`) simulate fills through book iteration using this depth data for tighter execution control.

---

### General Enhancements

* **Dynamic Thresholding**: Each voucher has its own threshold tuned based on past simulations (e.g., `threshold_10000 = 0.0035`, `threshold_10500 = 0.001`).
* **Regression-Based IV Modeling**: Coefficients for the quadratic volatility surface (base, linear, squared) are product-specific and allow for richer surface modeling.
* **Trade Book and PnL Logging**: Trade execution routines integrate with the existing `trade_book` and `calculate_profit` for real-time profit tracking.
* **Error Handling in Deltas**: Added try-catch logic to gracefully handle edge cases in delta calculation when moneyness or volatility may cause instability.

---