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
