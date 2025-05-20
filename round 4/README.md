## Round 4 Updates

In Round 4, the bot introduces options workflow enhancements, sunlight-triggered macro logic for `MAGNIFICENT_MACARONS`, and refines volatility and IV/arb strategies.

### New Sunlight-Based Macro on MAGNIFICENT\_MACARONS

**Simple Overview:**

* Monitors a `SUNLIGHT` observation.  If sunlight remains below a critical level (`sunlight_critical = 45`) for a sustained period (`sunlight_duration` ticks), triggers an **emergency sell-off** of all `MAGNIFICENT_MACARONS` inventory at market.  Once sunlight rebounds or after execution, resets the macro.

**Key Steps:**

1. Track consecutive ticks `below_count` when sunlight < critical.
2. If `below_count ≥ sunlight_duration`, set `critical = True` and sell all at best bid.
3. Reset `below_count` and `critical` post-sell or when sunlight recovers.

**Parameters:** `sunlight_critical`, `sunlight_duration`, `conversion_limit` for conversions during clearing.

### Volatility & Options Refinements

**Changes from Round 3:**

* **EWMA Vol Update:** Uses `lambda_ewma` to smooth realized volatility in `update_ewma_vol`, updating `self.volatility` each price tick.
* **IV & Arb Size Limits:** Now computes maximum notional size from Vega-based risk (`target_daily_pnl_vol`, `max_loss_per_contract`), capping order volume per strike.
* **Stop-Loss & Delta Hedge:** After IV/arbitrage, runs `generate_stop_loss_orders` to liquidate positions exceeding per-contract loss thresholds, and `generate_delta_hedge` to neutralize net option delta by trading `VOLCANIC_ROCK`.
* **Strategy Selection Persistence:** Retains `chosen_strategy` across sessions until `strategy_reset_window` is reached, balancing IV vs. arb performance dynamically.

### General Workflow Adjustments

* **Enhanced Persistence:** Saves extended `trader_obj` state including `prev_price`, `iv_history`, `entry_prices`, `unrealized_pnl`, and `price_history` for new strategies.
* **Arbitrage Clearing Limit:** Restricts conversion actions via `conversion_limit` when clearing excess positions.
* **Helper Consolidation:** Introduced common `_take_basic` and `_take_with_adverse` in `SQUID_INK`, reused by both basic and adverse-taking strategies.