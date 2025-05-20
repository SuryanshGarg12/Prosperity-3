from typing import Dict, List, Tuple
from datamodel import Order, OrderDepth, TradingState
import jsonpickle

class Trader:
    def __init__(self):
        # ----- Shared Position Limits -----
        self.position_limits = {
            "RAINFOREST_RESIN": 50,
            "KELP": 50,
            "SQUID_INK": 50
        }

        # ----- RAINFOREST_RESIN Strategy Parameters and Data -----
        self.resin_params = {
            "take_width": 1,        # Execute market orders if price is this far from fair value.
            "clear_width": 0,       
            "disregard_edge": 1,    # Disregard orders for joining/pennying within this distance from fair value.
            "join_edge": 2,         # Join orders within this edge.
            "default_edge": 4,      # Default edge when no orders to penny/join.
            "soft_position_limit": 30,  # Start leaning against the position when this large.
        }
        self.resin_mid_prices = []
        self.resin_rolling_window = 10
        self.resin_fair_value = None

        # ----- KELP Strategy Parameters and Data -----
        self.kelp_params = {
            "take_width": 1,
            "clear_width": 0,
            "prevent_adverse": True,
            "adverse_volume": 20,
            "reversion_beta": -0.229,
            "disregard_edge": 1,
            "join_edge": 0,
            "default_edge": 1,
        }
        self.kelp_last_price = None

        # ----- SQUID_INK Strategy (Stochastic Oscillator + Trailing Stops) Parameters and Data -----
        self.prices: List[float] = []
        self.lookback = 22  # Stochastic oscillator window
        self.trailing_pct = 0.05
        self.take_profit_pct = 100  # Currently not used but kept for reference
        self.cooldown_period = 0
        self.buy_k_val = 27
        self.sell_k_val = 90

        # State variables for SQUID_INK
        self.long_trailing_price = None
        self.short_trailing_price = None
        self.entry_price = None
        self.prev_k = None
        self.prev_d = None
        self.last_exit_tick = -1000  # initialize far in the past
        self.tick = 0

    # ================= RAINFOREST_RESIN Methods =================
    def resin_calculate_fair_value(self, order_depth: OrderDepth) -> float:
        if order_depth.sell_orders and order_depth.buy_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            mid_price = (best_ask + best_bid) / 2

            self.resin_mid_prices.append(mid_price)
            if len(self.resin_mid_prices) > self.resin_rolling_window:
                self.resin_mid_prices.pop(0)
            if len(self.resin_mid_prices) == self.resin_rolling_window:
                return sum(self.resin_mid_prices) / len(self.resin_mid_prices)
            else:
                return mid_price
        return None

    def resin_take_best_orders(self, fair_value: float, take_width: float,
                               orders: List[Order], order_depth: OrderDepth, position: int) -> Tuple[int, int]:
        position_limit = self.position_limits["RAINFOREST_RESIN"]
        buy_order_volume = 0
        sell_order_volume = 0

        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -order_depth.sell_orders[best_ask]
            if best_ask <= fair_value - take_width:
                quantity = min(best_ask_amount, position_limit - position)
                if quantity > 0:
                    orders.append(Order("RAINFOREST_RESIN", best_ask, quantity))
                    buy_order_volume += quantity

        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if best_bid >= fair_value + take_width:
                quantity = min(best_bid_amount, position_limit + position)
                if quantity > 0:
                    orders.append(Order("RAINFOREST_RESIN", best_bid, -quantity))
                    sell_order_volume += quantity

        return buy_order_volume, sell_order_volume

    def resin_make_orders(self, order_depth: OrderDepth, fair_value: float,
                          position: int, buy_order_volume: int, sell_order_volume: int) -> Tuple[List[Order], int, int]:
        orders = []
        # Find candidate order prices away from fair value.
        asks_above_fair = [price for price in order_depth.sell_orders.keys()
                           if price > fair_value + self.resin_params["disregard_edge"]]
        bids_below_fair = [price for price in order_depth.buy_orders.keys()
                           if price < fair_value - self.resin_params["disregard_edge"]]

        best_ask_above_fair = min(asks_above_fair) if asks_above_fair else None
        best_bid_below_fair = max(bids_below_fair) if bids_below_fair else None

        ask = round(fair_value + self.resin_params["default_edge"])
        if best_ask_above_fair is not None:
            if abs(best_ask_above_fair - fair_value) <= self.resin_params["join_edge"]:
                ask = best_ask_above_fair
            else:
                ask = best_ask_above_fair - 1

        bid = round(fair_value - self.resin_params["default_edge"])
        if best_bid_below_fair is not None:
            if abs(fair_value - best_bid_below_fair) <= self.resin_params["join_edge"]:
                bid = best_bid_below_fair
            else:
                bid = best_bid_below_fair + 1

        # Adjust orders if positions are large.
        if position > self.resin_params["soft_position_limit"]:
            ask -= 1
        elif position < -self.resin_params["soft_position_limit"]:
            bid += 1

        position_limit = self.position_limits["RAINFOREST_RESIN"]
        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order("RAINFOREST_RESIN", bid, buy_quantity))

        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order("RAINFOREST_RESIN", ask, -sell_quantity))

        return orders, buy_order_volume, sell_order_volume

    # ================= KELP Methods =================
    def kelp_fair_value(self, order_depth: OrderDepth) -> float:
        if order_depth.sell_orders and order_depth.buy_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [price for price in order_depth.sell_orders.keys()
                            if abs(order_depth.sell_orders[price]) >= self.kelp_params["adverse_volume"]]
            filtered_bid = [price for price in order_depth.buy_orders.keys()
                            if abs(order_depth.buy_orders[price]) >= self.kelp_params["adverse_volume"]]
            mm_ask = min(filtered_ask) if filtered_ask else None
            mm_bid = max(filtered_bid) if filtered_bid else None

            if mm_ask is None or mm_bid is None:
                mmmid_price = (best_ask + best_bid) / 2 if self.kelp_last_price is None else self.kelp_last_price
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            if self.kelp_last_price is not None:
                last_price = self.kelp_last_price
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = last_returns * self.kelp_params["reversion_beta"]
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price

            self.kelp_last_price = mmmid_price
            return fair
        return None

    def kelp_take_orders(self, order_depth: OrderDepth, fair_value: float,
                         take_width: float, position: int) -> Tuple[List[Order], int, int]:
        orders = []
        buy_order_volume = 0
        sell_order_volume = 0
        position_limit = self.position_limits["KELP"]

        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -order_depth.sell_orders[best_ask]
            if (not self.kelp_params["prevent_adverse"] or abs(best_ask_amount) <= self.kelp_params["adverse_volume"]) and \
               best_ask <= fair_value - take_width:
                quantity = min(best_ask_amount, position_limit - position)
                if quantity > 0:
                    orders.append(Order("KELP", best_ask, quantity))
                    buy_order_volume += quantity

        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if (not self.kelp_params["prevent_adverse"] or abs(best_bid_amount) <= self.kelp_params["adverse_volume"]) and \
               best_bid >= fair_value + take_width:
                quantity = min(best_bid_amount, position_limit + position)
                if quantity > 0:
                    orders.append(Order("KELP", best_bid, -quantity))
                    sell_order_volume += quantity

        return orders, buy_order_volume, sell_order_volume

    def kelp_clear_orders(self, order_depth: OrderDepth, fair_value: float,
                          clear_width: int, position: int, buy_order_volume: int, sell_order_volume: int) -> Tuple[List[Order], int, int]:
        orders = []
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - clear_width)
        fair_for_ask = round(fair_value + clear_width)
        position_limit = self.position_limits["KELP"]
        buy_quantity = position_limit - (position + buy_order_volume)
        sell_quantity = position_limit + (position - sell_order_volume)

        if position_after_take > 0:
            clear_quantity = sum(volume for price, volume in order_depth.buy_orders.items()
                                 if price >= fair_for_ask)
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order("KELP", fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            clear_quantity = sum(abs(volume) for price, volume in order_depth.sell_orders.items()
                                 if price <= fair_for_bid)
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order("KELP", fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return orders, buy_order_volume, sell_order_volume

    def kelp_make_orders(self, order_depth: OrderDepth, fair_value: float,
                         position: int, buy_order_volume: int, sell_order_volume: int) -> Tuple[List[Order], int, int]:
        orders = []
        asks_above_fair = [price for price in order_depth.sell_orders.keys()
                           if price > fair_value + self.kelp_params["disregard_edge"]]
        bids_below_fair = [price for price in order_depth.buy_orders.keys()
                           if price < fair_value - self.kelp_params["disregard_edge"]]
        best_ask_above_fair = min(asks_above_fair) if asks_above_fair else None
        best_bid_below_fair = max(bids_below_fair) if bids_below_fair else None

        ask = round(fair_value + self.kelp_params["default_edge"])
        if best_ask_above_fair is not None:
            if abs(best_ask_above_fair - fair_value) <= self.kelp_params["join_edge"]:
                ask = best_ask_above_fair
            else:
                ask = best_ask_above_fair - 1

        bid = round(fair_value - self.kelp_params["default_edge"])
        if best_bid_below_fair is not None:
            if abs(fair_value - best_bid_below_fair) <= self.kelp_params["join_edge"]:
                bid = best_bid_below_fair
            else:
                bid = best_bid_below_fair + 1

        buy_qty = self.position_limits["KELP"] - (position + buy_order_volume)
        if buy_qty > 0:
            orders.append(Order("KELP", bid, buy_qty))

        sell_qty = self.position_limits["KELP"] + (position - sell_order_volume)
        if sell_qty > 0:
            orders.append(Order("KELP", ask, -sell_qty))

        return orders, buy_order_volume, sell_order_volume

    # ================= SQUID_INK Methods =================
    def process_squid_ink(self, order_depth: OrderDepth, current_position: int) -> List[Order]:
        orders = []
        # SQUID_INK strategy only runs if both sides exist.
        if not order_depth.sell_orders or not order_depth.buy_orders:
            return orders

        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        mid_price = (best_ask + best_bid) / 2

        # Update price history.
        self.prices.append(mid_price)
        if len(self.prices) < self.lookback:
            return orders

        # Cooldown check.
        if self.tick - self.last_exit_tick < self.cooldown_period:
            return orders

        recent_prices = self.prices[-self.lookback:]
        lowest_low = min(recent_prices)
        highest_high = max(recent_prices)
        if highest_high == lowest_low:
            return orders

        percent_k = (mid_price - lowest_low) / (highest_high - lowest_low) * 100
        # Calculate %D as the moving average of the last 3 %K values.
        recent_k = []
        for i in range(max(0, len(self.prices)-3), len(self.prices)):
            window = self.prices[max(0, i - self.lookback + 1): i + 1]
            low = min(window)
            high = max(window)
            if high == low:
                continue
            recent_k.append((self.prices[i] - low) / (high - low) * 100)
        if len(recent_k) < 3:
            return orders
        percent_d = sum(recent_k[-3:]) / 3

        # --- Buy condition: when %K crosses above %D and price is rising ---
        if (self.prev_k is not None and self.prev_d is not None and
            self.prev_k < self.prev_d and percent_k > percent_d and
            percent_k < self.buy_k_val and
            len(self.prices) >= 4 and
            mid_price > self.prices[-2] and mid_price > self.prices[-3]):
            
            volume = min(self.position_limits["SQUID_INK"] - current_position, -order_depth.sell_orders[best_ask])
            if volume > 0:
                self.entry_price = best_ask
                self.long_trailing_price = mid_price
                self.short_trailing_price = None
                orders.append(Order("SQUID_INK", best_ask, volume))
        
        # --- Sell condition: when %K crosses below %D and %K is high ---
        elif (self.prev_k is not None and self.prev_d is not None and
              self.prev_k > self.prev_d and percent_k < percent_d and
              percent_k > self.sell_k_val):
            
            volume = min(current_position + self.position_limits["SQUID_INK"], order_depth.buy_orders[best_bid])
            if volume > 0:
                self.entry_price = best_bid
                self.short_trailing_price = mid_price
                self.long_trailing_price = None
                orders.append(Order("SQUID_INK", best_bid, -volume))
        
        # --- Manage Trailing Stop Loss & Take Profit for LONG positions ---
        if current_position > 0:
            self.long_trailing_price = max(self.long_trailing_price or mid_price, mid_price)
            if mid_price >= self.entry_price * (1 + self.take_profit_pct):
                volume = min(current_position, order_depth.buy_orders[best_bid])
                if volume > 0:
                    orders.append(Order("SQUID_INK", best_bid, -volume))
                    self._reset_trade_state()
            elif mid_price < self.long_trailing_price * (1 - self.trailing_pct):
                volume = min(current_position, order_depth.buy_orders[best_bid])
                if volume > 0:
                    orders.append(Order("SQUID_INK", best_bid, -volume))
                    self._reset_trade_state()
        
        # --- Manage Trailing Stop Loss & Take Profit for SHORT positions ---
        elif current_position < 0:
            self.short_trailing_price = min(self.short_trailing_price or mid_price, mid_price)
            if mid_price <= self.entry_price * (1 - self.take_profit_pct):
                volume = min(-current_position, -order_depth.sell_orders[best_ask])
                if volume > 0:
                    orders.append(Order("SQUID_INK", best_ask, volume))
                    self._reset_trade_state()
            elif mid_price > self.short_trailing_price * (1 + self.trailing_pct):
                volume = min(-current_position, -order_depth.sell_orders[best_ask])
                if volume > 0:
                    orders.append(Order("SQUID_INK", best_ask, volume))
                    self._reset_trade_state()

        self.prev_k = percent_k
        self.prev_d = percent_d

        return orders

    def _reset_trade_state(self):
        self.entry_price = None
        self.long_trailing_price = None
        self.short_trailing_price = None
        self.last_exit_tick = self.tick

    # ================= Combined run Method =================
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        result: Dict[str, List[Order]] = {}
        
        # Optionally, load persistent KELP data.
        traderData = {}
        if state.traderData not in [None, ""]:
            traderData = jsonpickle.decode(state.traderData)
            if "kelp_last_price" in traderData:
                self.kelp_last_price = traderData["kelp_last_price"]

        # Process each product according to its own strategy.
        for product, order_depth in state.order_depths.items():
            orders: List[Order] = []
            current_position = state.position.get(product, 0)
            
            if product == "RAINFOREST_RESIN":
                # --- RAINFOREST_RESIN Strategy ---
                fair_value = self.resin_calculate_fair_value(order_depth)
                if fair_value is None:
                    result[product] = []
                    continue
                # Take immediate liquidity if advantageous.
                take_buy_volume, take_sell_volume = self.resin_take_best_orders(
                    fair_value, self.resin_params["take_width"],
                    orders, order_depth, current_position
                )
                # Place limit orders with position management.
                make_orders, _, _ = self.resin_make_orders(
                    order_depth, fair_value, current_position, take_buy_volume, take_sell_volume
                )
                orders.extend(make_orders)
                result[product] = orders

            elif product == "KELP":
                # --- KELP Strategy ---
                kelp_fv = self.kelp_fair_value(order_depth)
                if kelp_fv is None:
                    result[product] = []
                    continue
                take_orders, buy_volume, sell_volume = self.kelp_take_orders(
                    order_depth, kelp_fv, self.kelp_params["take_width"], current_position
                )
                orders.extend(take_orders)
                clear_orders, buy_volume, sell_volume = self.kelp_clear_orders(
                    order_depth, kelp_fv, self.kelp_params["clear_width"], current_position, buy_volume, sell_volume
                )
                orders.extend(clear_orders)
                make_orders, buy_volume, sell_volume = self.kelp_make_orders(
                    order_depth, kelp_fv, current_position, buy_volume, sell_volume
                )
                orders.extend(make_orders)
                result[product] = orders

            elif product == "SQUID_INK":
                # --- SQUID_INK Strategy ---
                orders = self.process_squid_ink(order_depth, current_position)
                result[product] = orders

            else:
                result[product] = []

        # Save persistent KELP data.
        traderData["kelp_last_price"] = self.kelp_last_price
        traderDataEncoded = jsonpickle.encode(traderData)

        # Increase tick for SQUID_INK cooldown management.
        self.tick += 1
        # Set conversions to 1 if any SQUID_INK trade occurred, otherwise it can be 0.
        conversions = 1 if "SQUID_INK" in result and len(result["SQUID_INK"]) > 0 else 0

        return result, conversions, traderDataEncoded
