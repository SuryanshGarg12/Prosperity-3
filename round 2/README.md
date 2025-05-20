## Simple Overview

This trading bot runs three independent strategies, each tailored to a different product. In everyday language:

* **Rainforest Resin** acts like a friendly market maker: it watches the average of recent prices and tries to buy when the price is a bit low or sell when it’s a bit high. It also places limit orders to provide steady liquidity, gently nudging prices back toward fair levels.

* **Kelp** is a cautious market maker. It focuses on trades with enough volume to avoid being picked off by large players. It estimates a fair price, leans toward mean reversion, and cleanly rebalances its inventory if it strays too far from neutral.

* **Squid Ink** looks for short-term price momentum. It uses a simple oscillator to spot when prices are gaining or losing strength, enters momentum trades, and then uses trailing stops to lock in gains or cut losses. After exiting a trade, it waits a bit before jumping back in.

---

## Strategy Details

This bot employs three specialized market-making and trading strategies, each dedicated to a distinct product.

---

### Rainforest Resin

**Objective:** Provide liquidity around a fair value derived from a rolling mid-price average, executing both aggressive (market) and passive (limit) orders.

1. **Fair Value Calculation**

   * Compute the mid-price: `(best_ask + best_bid) / 2`.
   * Maintain a rolling window of the last *N* mid-prices (default `N = 10`).
   * Once the window is full, average the stored mid-prices to obtain the *fair value*; otherwise, use the instantaneous mid-price.

2. **Market-Taking (`take_width`)**

   * If `best_ask ≤ fair_value − take_width`, send a market **buy** up to position limit.
   * If `best_bid ≥ fair_value + take_width`, send a market **sell** down to negative position limit.

3. **Limit-Order Quoting**

   * Identify candidate prices outside a `disregard_edge` from fair value.
   * **Joining**: If an existing book order is within a `join_edge`, quote at that price; else, quote at `default_edge` offset from fair value.
   * **Edge Adjustment**: If current position exceeds a `soft_position_limit`, widen quotes to lean toward reducing exposure.

**Key Parameters:** `take_width`, `default_edge`, `disregard_edge`, `join_edge`, `soft_position_limit`, `rolling_window`.

---

### Kelp

**Objective:** Implement adverse-volume–filtered market making with mean-reversion bias, inspired by Jane Street's KELP.

1. **Filtered Mid-Price**

   * Compute `best_ask`/`best_bid` as usual.
   * Filter to orders whose size ≥ `adverse_volume`; compute mid-price `mmmid_price` from this subset, falling back to unfiltered mid-price if insufficient volume.

2. **Reversion Adjustment**

   * Calculate last returns: `(mmmid_price - last_price) / last_price`.
   * Apply `reversion_beta` to predict short-term reversion.
   * Adjust fair value: `fair = mmmid_price + (mmmid_price * predicted_returns)`.

3. **Market-Taking**

   * Similar logic to Rainforest Resin but can optionally **prevent adverse** trades when large opposing orders exist.

4. **Position Clearing (`clear_width`)**

   * After taking trades, if inventory deviates, send limit orders inside the book (inside `clear_width`) to rebalance toward zero.

5. **Passive Quoting**

   * Post new quotes outside a `disregard_edge`; join or place at a `default_edge` offset.

**Persistent State:** Stores `kelp_last_price` between runs in `traderData`.

**Key Parameters:** `adverse_volume`, `reversion_beta`, `prevent_adverse`, `clear_width`, `default_edge`, `disregard_edge`, `join_edge`.

---

### Squid Ink

**Objective:** Capture momentum using a stochastic oscillator (%K, %D) combined with trailing stops and cooldowns.

1. **Price History & Oscillator**

   * Track mid-prices over a `lookback` window (default 22 ticks).
   * Compute `%K = (current_price - lowest_low) / (highest_high - lowest_low) * 100`.
   * Compute `%D` as 3-period SMA of recent %K values.

2. **Entry Signals**

   * **Long**: %K crosses above %D, %K < `buy_k_val`, and short-term price momentum confirms (price rising).
   * **Short**: %K crosses below %D, %K > `sell_k_val`.
   * Enforce a `cooldown_period` after each exit before entering again.

3. **Trailing Stops & Exits**

   * Maintain `long_trailing_price` / `short_trailing_price` as peak/trough since entry.
   * **Exit** when:

     * Price retreats beyond a `trailing_pct` threshold from the peak/trough, or
     * (Optional) reaches a take-profit multiple of entry (e.g., `take_profit_pct`).

4. **State Reset**

   * On exit, reset `entry_price`, trailing markers, and record the tick (`last_exit_tick`).

**Key Parameters:** `lookback`, `buy_k_val`, `sell_k_val`, `trailing_pct`, `cooldown_period`, `take_profit_pct` (not actively used).

---

---

## Round 2 Updates

Building on the original three strategies, Round 2 introduces **Basket/Hedge** trading for composite products and refines **SQUID\_INK** for standalone mean-reversion market-making.

### New Products: Basket/Hedge Strategy

**Simple Overview:**

* We now treat `PICNIC_BASKET1` and `PICNIC_BASKET2` as composite instruments composed of other items (e.g., Croissants, Jams, Djembes).  The bot watches for when the basket’s market price deviates from its theoretical value and then trades the basket plus hedges its risk by trading the component products in proportion.

**Key Steps:**

1. **Theoretical Value:** Calculate basket value from underlying prices using predefined composition ratios.
2. **Mispricing Detection:** Track percent difference over a rolling window (`vol_window = 20`) and compute a dynamic threshold based on volatility (`sigma_min`, `sigma_max`).
3. **Soft Liquidation:** If the basket is overpriced, **sell** the basket and simultaneously **sell** proportional amounts of Jams, Croissants (and optionally Djembes).  If underpriced, do the reverse.  Volumes scale with how far mispricing exceeds threshold, with staged multipliers up to 100%.
4. **Depth History:** Maintain book-depth buffers to optionally weight order volume by liquidity.

**Key Parameters:**

* Basket compositions (`basket1_composition`, `basket2_composition`)
* Mispricing thresholds (`mispricing_threshold_percent_basket_1/2`)
* Volatility weighting (`volatality_weightage`, `momemtum_weighage`)
* Soft liquidation percentages (`soft_liquidate_vol_percentage_1/2`)

### SQUID\_INK Refinements

**Changes from Round 1:**

* Renamed and generalized market-making code into reusable `take_best_orders`, `market_make`, and `clear_position_order` methods.
* Introduced **adverse-volume filtering** exactly as in Kelp, controlled by `prevent_adverse` and `adverse_volume`.
* Added a **minimum quoting edge** (`SQUID_INK_min_edge`) to ensure the bot only posts passive quotes at least this far from fair value.
* Consolidated state persistence to store only the latest price (`SQUID_INK_last_price`) alongside other minimal strategy data.

**Simple Overview:**

* SQUID\_INK now behaves like a small-scale, adverse-aware market maker: it takes liquidity when the price deviates far enough from its fair value, then quotes both bid/ask passively outside a fixed minimum edge, always respecting position limits and avoiding large opposing orders.

**Key Parameters:**

* `take_width`, `clear_width`, `prevent_adverse`, `adverse_volume`, `SQUID_INK_min_edge`
