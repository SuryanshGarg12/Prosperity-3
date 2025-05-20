## Round 3 Updates

In Round 3, the bot expands to new products (Volcanic Rock and its vouchers) and refines configuration and mean-reversion logic across strategies.

### New Products: Volcanic Rock & Vouchers

**Simple Overview:**

* Introduces `VOLCANIC_ROCK` and a series of strike-specific vouchers (`VOLCANIC_ROCK_VOUCHER_9500`, etc.).  The bot tracks a 50‑period mid‑price history for the rock, computes a z‑score relative to its recent mean and standard deviation, and executes trades when the z‑score breaches predefined thresholds.

**Key Steps:**

1. **History Buffer:** Maintain last 50 mid-prices in a deque for `VOLCANIC_ROCK`.
2. **Z‑Score Calculation:** Once ≥50 samples, compute `(current_mid - mean) / std`.
3. **Threshold Trading:** If z < −threshold, buy rock; if z > threshold, sell rock—quantities limited by `position_limits`.
4. **Voucher Mean Reversion:** For each voucher product, apply a generic mean-reversion stub that uses the same z‑score: buy when z < −threshold\[voucher], sell when z > threshold\[voucher].

**Key Parameters:**

* Z‑score thresholds in `threshold` dict for rock and each voucher.
* History length fixed at 50.
* Position limits updated to include these new instruments (`position_limits`).

### Configuration & Generalization

**Changes from Round 2:**

* **Product Configuration:** Introduced `product_config` mapping for Rainforest Resin and Kelp, specifying `position_limit`, `target_position`, `max_trade_size`, and `min_spread`, enabling per‑instrument spread and size control.
* **History Tracking:** Centralized a `history` dict for each product to record past mid‑prices or spreads, supporting future analytics or dynamic parameter tuning.
* **Unified Utility Methods:** General methods `fair_value`, `trade_mean_reversion`, and `trade_jams` serve as stubs for future expansion, decoupling strategy logic from run loops.

**Simple Overview:**

* The bot’s architecture now supports easy on‑boarding of new instruments: define position limits, thresholds, and history buffers, then invoke generic mean‑reversion or specialized update methods in the `run()` flow.

### Other Refinements

* Persist only minimal state (same as before) but now also maintain recent `history` for potential analytics.
* Basket strategy remains unchanged from Round 2, and unused voucher strategies are stubbed for future enhancement.

---
