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
