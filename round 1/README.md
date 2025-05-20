## Round - 1

There were a total of 3 assets we had to trade in this round. Here are the strategies that we built for them.

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
