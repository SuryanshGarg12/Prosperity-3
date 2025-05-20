from typing import Dict, List, Tuple, Any
from datamodel import Order, OrderDepth, TradingState
import jsonpickle
import statistics

# --- SQUID_INK Setup ---
class Product:
    SQUID_INK = "SQUID_INK"

DEFAULT_PARAMS = {
    Product.SQUID_INK: {
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 15,
        "reversion_beta": -0.229,
        "SQUID_INK_min_edge": 2,
    }
}

class Trader:
    def __init__(self, squid_params: Dict[str, Any] = None):
        # ---------------------------- Shared Configuration ----------------------------
        self.position_limits = {
            "RAINFOREST_RESIN": 50,
            "KELP": 50,
            "CROISSANTS": 250,
            "JAMS": 350,
            "DJEMBES": 60,
            "PICNIC_BASKET1": 60,
            "PICNIC_BASKET2": 100,
            Product.SQUID_INK: 20,
        }
        # ------------------- RAINFOREST_RESIN Strategy Setup -------------------
        self.resin_params = {
            "take_width": 1,
            "clear_width": 0,
            "disregard_edge": 1,
            "join_edge": 2,
            "default_edge": 4,
            "soft_position_limit": 30,
        }
        # Save only the minimal rolling mid-price history.
        self.resin_mid_prices: List[float] = []
        self.resin_rolling_window = 10

        # ------------------- KELP Strategy Setup -------------------
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

        # ------------------- Basket/Hedge Strategy Setup -------------------
        # Only the histories needed for threshold calculations are stored.
        self.mispricing_history = {"BASKET1": [], "BASKET2": []}
        self.depth_history = {"BASKET1": [], "BASKET2": []}
        self.vol_window = 20
        self.sigma_min = 0.002
        self.sigma_max = 0.02
        # Basket compositions for hedging.
        self.basket1_composition = {"CROISSANTS": 6, "JAMS": 3, "DJEMBES": 1}
        self.basket2_composition = {"CROISSANTS": 4, "JAMS": 2}
        self.mispricing_threshold_percent_basket_1 = 0.001
        self.mispricing_threshold_percent_basket_2 = 0.002
        self.volatality_weightage = 0
        self.momemtum_weighage = 0
        self.soft_liquidate_vol_percentage_1 = 1
        self.soft_liquidate_vol_percentage_2 = 1

        # ------------------- SQUID_INK Strategy Setup -------------------
        self.squid_params = squid_params if squid_params is not None else DEFAULT_PARAMS
        # The last price for SQUID_INK is stored externally in persistent state.

    # ============================= RAINFOREST_RESIN Methods =============================
    def resin_calculate_fair_value(self, product: str, od: OrderDepth) -> float:
        if od.sell_orders and od.buy_orders:
            sell_orders = od.sell_orders
            buy_orders = od.buy_orders
            best_ask = min(sell_orders)
            best_bid = max(buy_orders)
            mid_price = (best_ask + best_bid) / 2
            self.resin_mid_prices.append(mid_price)
            if len(self.resin_mid_prices) > self.resin_rolling_window:
                self.resin_mid_prices.pop(0)
            if len(self.resin_mid_prices) == self.resin_rolling_window:
                return sum(self.resin_mid_prices) / self.resin_rolling_window
            return mid_price
        return None

    def resin_take_best_orders(self, product: str, fair_value: float, take_width: float,
                               orders: List[Order], od: OrderDepth, position: int) -> Tuple[int, int]:
        pos_limit = self.position_limits[product]
        buy_vol = 0
        sell_vol = 0
        sell_orders = od.sell_orders
        buy_orders = od.buy_orders
        if sell_orders:
            best_ask = min(sell_orders)
            best_ask_amt = -sell_orders[best_ask]
            if best_ask <= fair_value - take_width:
                qty = min(best_ask_amt, pos_limit - position)
                if qty > 0:
                    orders.append(Order(product, best_ask, qty))
                    buy_vol += qty
        if buy_orders:
            best_bid = max(buy_orders)
            best_bid_amt = buy_orders[best_bid]
            if best_bid >= fair_value + take_width:
                qty = min(best_bid_amt, pos_limit + position)
                if qty > 0:
                    orders.append(Order(product, best_bid, -qty))
                    sell_vol += qty
        return buy_vol, sell_vol

    def resin_make_orders(self, product: str, od: OrderDepth, fair_value: float,
                          position: int, buy_vol: int, sell_vol: int) -> Tuple[List[Order], int, int]:
        orders: List[Order] = []
        sell_orders = od.sell_orders
        buy_orders = od.buy_orders
        asks_above = [p for p in sell_orders if p > fair_value + self.resin_params["disregard_edge"]]
        bids_below = [p for p in buy_orders if p < fair_value - self.resin_params["disregard_edge"]]
        best_ask_above = min(asks_above) if asks_above else None
        best_bid_below = max(bids_below) if bids_below else None
        ask = round(fair_value + self.resin_params["default_edge"])
        if best_ask_above is not None:
            ask = best_ask_above if abs(best_ask_above - fair_value) <= self.resin_params["join_edge"] else best_ask_above - 1
        bid = round(fair_value - self.resin_params["default_edge"])
        if best_bid_below is not None:
            bid = best_bid_below if abs(fair_value - best_bid_below) <= self.resin_params["join_edge"] else best_bid_below + 1
        if position > self.resin_params["soft_position_limit"]:
            ask -= 1
        elif position < -self.resin_params["soft_position_limit"]:
            bid += 1
        pos_limit = self.position_limits[product]
        buy_qty = pos_limit - (position + buy_vol)
        if buy_qty > 0:
            orders.append(Order(product, bid, buy_qty))
        sell_qty = pos_limit + (position - sell_vol)
        if sell_qty > 0:
            orders.append(Order(product, ask, -sell_qty))
        return orders, buy_vol, sell_vol

    # ============================= KELP Methods =============================
    def kelp_fair_value(self, od: OrderDepth) -> float:
        sell_orders = od.sell_orders
        buy_orders = od.buy_orders
        if sell_orders and buy_orders:
            best_ask = min(sell_orders)
            best_bid = max(buy_orders)
            filtered_ask = (p for p in sell_orders if abs(sell_orders[p]) >= self.kelp_params["adverse_volume"])
            filtered_bid = (p for p in buy_orders if abs(buy_orders[p]) >= self.kelp_params["adverse_volume"])
            try:
                mm_ask = min(filtered_ask)
            except ValueError:
                mm_ask = None
            try:
                mm_bid = max(filtered_bid)
            except ValueError:
                mm_bid = None
            if mm_ask is None or mm_bid is None:
                mmmid_price = (best_ask + best_bid) / 2 if self.kelp_last_price is None else self.kelp_last_price
            else:
                mmmid_price = (mm_ask + mm_bid) / 2
            if self.kelp_last_price is not None:
                last_price = self.kelp_last_price
                ret = (mmmid_price - last_price) / last_price
                fair = mmmid_price + (mmmid_price * ret * self.kelp_params["reversion_beta"])
            else:
                fair = mmmid_price
            self.kelp_last_price = mmmid_price
            return fair
        return None

    def kelp_take_orders(self, product: str, od: OrderDepth, fair_value: float,
                         take_width: float, position: int) -> Tuple[List[Order], int, int]:
        orders: List[Order] = []
        buy_vol = 0
        sell_vol = 0
        pos_limit = self.position_limits[product]
        sell_orders = od.sell_orders
        buy_orders = od.buy_orders
        if sell_orders:
            best_ask = min(sell_orders)
            best_ask_amt = -sell_orders[best_ask]
            if ((not self.kelp_params["prevent_adverse"]) or (abs(best_ask_amt) <= self.kelp_params["adverse_volume"])) and \
               best_ask <= fair_value - take_width:
                qty = min(best_ask_amt, pos_limit - position)
                if qty > 0:
                    orders.append(Order(product, best_ask, qty))
                    buy_vol += qty
        if buy_orders:
            best_bid = max(buy_orders)
            best_bid_amt = buy_orders[best_bid]
            if ((not self.kelp_params["prevent_adverse"]) or (abs(best_bid_amt) <= self.kelp_params["adverse_volume"])) and \
               best_bid >= fair_value + take_width:
                qty = min(best_bid_amt, pos_limit + position)
                if qty > 0:
                    orders.append(Order(product, best_bid, -qty))
                    sell_vol += qty
        return orders, buy_vol, sell_vol

    def kelp_clear_orders(self, product: str, od: OrderDepth, fair_value: float,
                          clear_width: int, position: int, buy_vol: int, sell_vol: int) -> Tuple[List[Order], int, int]:
        orders: List[Order] = []
        pos_after = position + buy_vol - sell_vol
        fair_bid = round(fair_value - clear_width)
        fair_ask = round(fair_value + clear_width)
        pos_limit = self.position_limits[product]
        buy_qty = pos_limit - (position + buy_vol)
        sell_qty = pos_limit + (position - sell_vol)
        buy_orders = od.buy_orders
        sell_orders = od.sell_orders
        if pos_after > 0 and buy_orders:
            clear_qty = sum(v for p, v in buy_orders.items() if p >= fair_ask)
            clear_qty = min(clear_qty, pos_after)
            sent = min(sell_qty, clear_qty)
            if sent > 0:
                orders.append(Order(product, fair_ask, -abs(sent)))
                sell_vol += abs(sent)
        if pos_after < 0 and sell_orders:
            clear_qty = sum(abs(v) for p, v in sell_orders.items() if p <= fair_bid)
            clear_qty = min(clear_qty, abs(pos_after))
            sent = min(buy_qty, clear_qty)
            if sent > 0:
                orders.append(Order(product, fair_bid, abs(sent)))
                buy_vol += abs(sent)
        return orders, buy_vol, sell_vol

    def kelp_make_orders(self, product: str, od: OrderDepth, fair_value: float,
                         position: int, buy_vol: int, sell_vol: int) -> Tuple[List[Order], int, int]:
        orders: List[Order] = []
        sell_orders = od.sell_orders
        buy_orders = od.buy_orders
        asks_above = [p for p in sell_orders if p > fair_value + self.kelp_params["disregard_edge"]]
        bids_below = [p for p in buy_orders if p < fair_value - self.kelp_params["disregard_edge"]]
        best_ask = min(asks_above) if asks_above else None
        best_bid = max(bids_below) if bids_below else None
        ask = round(fair_value + self.kelp_params["default_edge"])
        if best_ask is not None:
            ask = best_ask if abs(best_ask - fair_value) <= self.kelp_params["join_edge"] else best_ask - 1
        bid = round(fair_value - self.kelp_params["default_edge"])
        if best_bid is not None:
            bid = best_bid if abs(fair_value - best_bid) <= self.kelp_params["join_edge"] else best_bid + 1
        if (self.position_limits[product] - (position + buy_vol)) > 0:
            orders.append(Order(product, bid, self.position_limits[product] - (position + buy_vol)))
        if (self.position_limits[product] + (position - sell_vol)) > 0:
            orders.append(Order(product, ask, -(self.position_limits[product] + (position - sell_vol))))
        return orders, buy_vol, sell_vol

    # ============================= Basket/Hedge Methods =============================
    def get_mid_price(self, od: OrderDepth) -> float:
        sell_orders = od.sell_orders
        buy_orders = od.buy_orders
        if sell_orders and buy_orders:
            return (min(sell_orders) + max(buy_orders)) / 2
        if sell_orders:
            return min(sell_orders)
        if buy_orders:
            return max(buy_orders)
        return None

    def calculate_theoretical_value(self, prices: Dict[str, float], basket: int) -> float:
        comp = self.basket1_composition if basket == 1 else self.basket2_composition
        if all(p in prices for p in comp):
            return sum(comp[p] * prices[p] for p in comp)
        return None

    def calculate_mispricing(self, prices: Dict[str, float]) -> Dict[str, dict]:
        mis = {}
        for idx, prod in enumerate(["PICNIC_BASKET1", "PICNIC_BASKET2"], start=1):
            if prod not in prices:
                continue
            tv = self.calculate_theoretical_value(prices, idx)
            if tv is None:
                continue
            diff = prices[prod] - tv
            pct = diff / tv
            key = f"BASKET{idx}"
            mis[key] = {"market_price": prices[prod], "theoretical_value": tv, "diff": diff, "percent_diff": pct}
            hist = self.mispricing_history[key]
            hist.append(pct)
            if len(hist) > self.vol_window:
                hist.pop(0)
        return mis

    def base_threshold(self, key: str) -> float:
        return self.mispricing_threshold_percent_basket_1 if key == "BASKET1" else self.mispricing_threshold_percent_basket_2

    def dynamic_threshold(self, key: str) -> float:
        hist = self.mispricing_history[key]
        base = self.base_threshold(key)
        if len(hist) < 2:
            return base
        sigma = statistics.pstdev(hist)
        v = (sigma - self.sigma_min) / (self.sigma_max - self.sigma_min)
        v = max(0.0, min(1.0, v))
        return base * ((1 - self.volatality_weightage) + 2 * self.volatality_weightage * v)

    def volume_multiplier(self, key: str) -> float:
        buf = self.depth_history[key]
        if not buf:
            return 1.0
        avg_depth = sum(buf) / len(buf)
        u = (avg_depth - self.depth_min) / (self.depth_max - self.depth_min)
        u = max(0.0, min(1.0, u))
        return (1 - self.momemtum_weighage) + 2 * self.momemtum_weighage * u

    def calculate_order_volume(self, avail: int, limit: int, pos: int, is_buy: bool) -> int:
        return min(avail, limit - pos) if is_buy else min(avail, limit + pos)

    def generate_basket_orders(self, mis: Dict[str, dict], state: TradingState) -> Dict[str, List[Order]]:
        orders: Dict[str, List[Order]] = {}
        for idx, prod in enumerate(["PICNIC_BASKET1", "PICNIC_BASKET2"], start=1):
            key = f"BASKET{idx}"
            if key not in mis:
                continue
            pct = mis[key]["percent_diff"]
            thr = self.dynamic_threshold(key)
            # Begin with basket order itself.
            book = state.order_depths.get(prod)
            pos = state.position.get(prod, 0)
            if pct > thr and book and book.buy_orders:
                vol_mult = self.soft_liquidate_vol_percentage_1
                if pct > 1.5 * thr:
                    vol_mult = self.soft_liquidate_vol_percentage_2
                if pct > 2 * thr:
                    vol_mult = 1
                price = max(book.buy_orders)
                avail = book.buy_orders[price]
                vol = int(vol_mult * self.calculate_order_volume(abs(avail), self.position_limits[prod], pos, is_buy=False))
                if vol > 0:
                    orders.setdefault(prod, []).append(Order(prod, price, -vol))
                    # Hedge for SELL side:
                    # Hedge JAMS.
                    jam_vol = vol 
                    jam_book = state.order_depths.get("JAMS")
                    if jam_book and jam_book.buy_orders:
                        jam_price = max(jam_book.buy_orders)
                        jam_pos = state.position.get("JAMS", 0)
                        max_sell_jam = min(jam_vol, self.position_limits["JAMS"] + jam_pos)  # Adjusted for sell side.
                        if max_sell_jam > 0:
                            orders.setdefault("JAMS", []).append(Order("JAMS", jam_price, -max_sell_jam))
                    # Hedge CROISSANTS.
                    cro_vol = vol * self.basket1_composition["CROISSANTS"]
                    cro_book = state.order_depths.get("CROISSANTS")
                    if cro_book and cro_book.buy_orders:
                        cro_price = max(cro_book.buy_orders)
                        cro_pos = state.position.get("CROISSANTS", 0)
                        max_sell_cro = min(cro_vol, self.position_limits["CROISSANTS"] - cro_pos + cro_pos)
                        if max_sell_cro > 0:
                            orders.setdefault("CROISSANTS", []).append(Order("CROISSANTS", cro_price, -max_sell_cro))
                    # Hedge DJEMBES only for Basket1.
                    # if idx == 1:
                    #     djem_vol = vol * self.basket1_composition["DJEMBES"]
                    #     djem_book = state.order_depths.get("DJEMBES")
                    #     if djem_book and djem_book.buy_orders:
                    #         djem_price = max(djem_book.buy_orders)
                    #         djem_pos = state.position.get("DJEMBES", 0)
                    #         max_sell_djem = min(djem_vol, self.position_limits["DJEMBES"] - djem_pos + djem_pos)
                    #         if max_sell_djem > 0:
                    #             orders.setdefault("DJEMBES", []).append(Order("DJEMBES", djem_price, -max_sell_djem))
            elif pct < -thr and book and book.sell_orders:
                vol_mult = self.soft_liquidate_vol_percentage_1
                if pct < -1.5 * thr:
                    vol_mult = self.soft_liquidate_vol_percentage_2
                if pct < -2 * thr:
                    vol_mult = 1
                price = min(book.sell_orders)
                avail = book.sell_orders[price]
                vol = int(vol_mult * self.calculate_order_volume(abs(avail), self.position_limits[prod], pos, is_buy=True))
                if vol > 0:
                    orders.setdefault(prod, []).append(Order(prod, price, vol))
                    # Hedge for BUY side:
                    # Hedge JAMS.
                    jam_vol = vol * self.basket1_composition["JAMS"]
                    jam_book = state.order_depths.get("JAMS")
                    if jam_book and jam_book.sell_orders:
                        jam_price = min(jam_book.sell_orders)
                        jam_pos = state.position.get("JAMS", 0)
                        max_buy_jam = min(jam_vol, self.position_limits["JAMS"] - jam_pos)
                        if max_buy_jam > 0:
                            orders.setdefault("JAMS", []).append(Order("JAMS", jam_price, max_buy_jam))
                    # Hedge CROISSANTS.
                    cro_vol = vol * self.basket1_composition["CROISSANTS"]
                    cro_book = state.order_depths.get("CROISSANTS")
                    if cro_book and cro_book.sell_orders:
                        cro_price = min(cro_book.sell_orders)
                        cro_pos = state.position.get("CROISSANTS", 0)
                        max_buy_cro = min(cro_vol, self.position_limits["CROISSANTS"] - cro_pos)
                        if max_buy_cro > 0:
                            orders.setdefault("CROISSANTS", []).append(Order("CROISSANTS", cro_price, max_buy_cro))
                    # Hedge DJEMBES for Basket1.
                    # if idx == 1:
                    #     djem_vol = vol * self.basket1_composition["DJEMBES"]
                    #     djem_book = state.order_depths.get("DJEMBES")
                    #     if djem_book and djem_book.sell_orders:
                    #         djem_price = min(djem_book.sell_orders)
                    #         djem_pos = state.position.get("DJEMBES", 0)
                    #         max_buy_djem = min(djem_vol, self.position_limits["DJEMBES"] - djem_pos)
                    #         if max_buy_djem > 0:
                    #             orders.setdefault("DJEMBES", []).append(Order("DJEMBES", djem_price, max_buy_djem))
        return orders

    # ============================= SQUID_INK Methods =============================
    def take_best_orders(self, product: str, fair_value: int, take_width: float,
                         orders: List[Order], od: OrderDepth, position: int,
                         buy_vol: int, sell_vol: int, prevent_adverse: bool = False,
                         adverse_volume: int = 0) -> Tuple[int, int]:
        pos_limit = self.position_limits[product]
        if od.sell_orders:
            best_ask = min(od.sell_orders)
            best_ask_amt = -od.sell_orders[best_ask]
            if best_ask <= fair_value - take_width:
                qty = min(best_ask_amt, pos_limit - position)
                if qty > 0:
                    orders.append(Order(product, best_ask, qty))
                    buy_vol += qty
                    od.sell_orders[best_ask] += qty
                    if od.sell_orders[best_ask] == 0:
                        del od.sell_orders[best_ask]
        if od.buy_orders:
            best_bid = max(od.buy_orders)
            best_bid_amt = od.buy_orders[best_bid]
            if best_bid >= fair_value + take_width:
                qty = min(best_bid_amt, pos_limit + position)
                if qty > 0:
                    orders.append(Order(product, best_bid, -qty))
                    sell_vol += qty
                    od.buy_orders[best_bid] -= qty
                    if od.buy_orders[best_bid] == 0:
                        del od.buy_orders[best_bid]
        return buy_vol, sell_vol

    def take_best_orders_with_adverse(self, product: str, fair_value: int, take_width: float,
                                      orders: List[Order], od: OrderDepth, position: int,
                                      buy_vol: int, sell_vol: int, adverse_volume: int) -> Tuple[int, int]:
        pos_limit = self.position_limits[product]
        if od.sell_orders:
            best_ask = min(od.sell_orders)
            best_ask_amt = -od.sell_orders[best_ask]
            if abs(best_ask_amt) <= adverse_volume:
                if best_ask <= fair_value - take_width:
                    qty = min(best_ask_amt, pos_limit - position)
                    if qty > 0:
                        orders.append(Order(product, best_ask, qty))
                        buy_vol += qty
                        od.sell_orders[best_ask] += qty
                        if od.sell_orders[best_ask] == 0:
                            del od.sell_orders[best_ask]
        if od.buy_orders:
            best_bid = max(od.buy_orders)
            best_bid_amt = od.buy_orders[best_bid]
            if abs(best_bid_amt) <= adverse_volume:
                if best_bid >= fair_value + take_width:
                    qty = min(best_bid_amt, pos_limit + position)
                    if qty > 0:
                        orders.append(Order(product, best_bid, -qty))
                        sell_vol += qty
                        od.buy_orders[best_bid] -= qty
                        if od.buy_orders[best_bid] == 0:
                            del od.buy_orders[best_bid]
        return buy_vol, sell_vol

    def market_make(self, product: str, orders: List[Order], bid: int, ask: int,
                    position: int, buy_vol: int, sell_vol: int) -> Tuple[int, int]:
        buy_qty = self.position_limits[product] - (position + buy_vol)
        if buy_qty > 0:
            orders.append(Order(product, round(bid), buy_qty))
        sell_qty = self.position_limits[product] + (position - sell_vol)
        if sell_qty > 0:
            orders.append(Order(product, round(ask), -sell_qty))
        return buy_vol, sell_vol

    def clear_position_order(self, product: str, fair_value: float, width: int,
                             orders: List[Order], od: OrderDepth, position: int,
                             buy_vol: int, sell_vol: int) -> Tuple[int, int]:
        pos_after = position + buy_vol - sell_vol
        fair_bid = round(fair_value - width)
        fair_ask = round(fair_value + width)
        buy_qty = self.position_limits[product] - (position + buy_vol)
        sell_qty = self.position_limits[product] + (position - sell_vol)
        if pos_after > 0:
            clear_qty = sum(v for p, v in od.buy_orders.items() if p >= fair_ask)
            clear_qty = min(clear_qty, pos_after)
            sent = min(sell_qty, clear_qty)
            if sent > 0:
                orders.append(Order(product, fair_ask, -abs(sent)))
                sell_vol += abs(sent)
        if pos_after < 0:
            clear_qty = sum(abs(v) for p, v in od.sell_orders.items() if p <= fair_bid)
            clear_qty = min(clear_qty, abs(pos_after))
            sent = min(buy_qty, clear_qty)
            if sent > 0:
                orders.append(Order(product, fair_bid, abs(sent)))
                buy_vol += abs(sent)
        return buy_vol, sell_vol

    def SQUID_INK_fair_value(self, od: OrderDepth, trader_obj: Dict[str, Any]) -> float:
        if od.sell_orders and od.buy_orders:
            sell_orders = od.sell_orders
            buy_orders = od.buy_orders
            best_ask = min(sell_orders)
            best_bid = max(buy_orders)
            filtered_ask = [p for p in sell_orders if abs(sell_orders[p]) >= self.squid_params[Product.SQUID_INK]["adverse_volume"]]
            filtered_bid = [p for p in buy_orders if abs(buy_orders[p]) >= self.squid_params[Product.SQUID_INK]["adverse_volume"]]
            mm_ask = min(filtered_ask) if filtered_ask else None
            mm_bid = max(filtered_bid) if filtered_bid else None
            if mm_ask is None or mm_bid is None:
                mmmid = (best_ask + best_bid) / 2 if trader_obj.get("SQUID_INK_last_price") is None else trader_obj["SQUID_INK_last_price"]
            else:
                mmmid = (mm_ask + mm_bid) / 2
            if trader_obj.get("SQUID_INK_last_price") is not None:
                last_price = trader_obj["SQUID_INK_last_price"]
                ret = (mmmid - last_price) / last_price
                fair = mmmid + (mmmid * ret * self.squid_params[Product.SQUID_INK]["reversion_beta"])
            else:
                fair = mmmid
            trader_obj["SQUID_INK_last_price"] = mmmid
            return fair
        return None

    def take_orders_SQUID_INK(self, od: OrderDepth, fair_value: float, take_width: float, position: int) -> Tuple[List[Order], int, int]:
        orders: List[Order] = []
        buy_vol = 0
        sell_vol = 0
        params = self.squid_params[Product.SQUID_INK]
        if params["prevent_adverse"]:
            buy_vol, sell_vol = self.take_best_orders_with_adverse(Product.SQUID_INK, fair_value, take_width,
                                                                    orders, od, position, buy_vol, sell_vol,
                                                                    params["adverse_volume"])
        else:
            buy_vol, sell_vol = self.take_best_orders(Product.SQUID_INK, fair_value, take_width,
                                                       orders, od, position, buy_vol, sell_vol)
        return orders, buy_vol, sell_vol

    def clear_orders_SQUID_INK(self, od: OrderDepth, fair_value: float, clear_width: int, position: int,
                               buy_vol: int, sell_vol: int) -> Tuple[List[Order], int, int]:
        orders: List[Order] = []
        buy_vol, sell_vol = self.clear_position_order(Product.SQUID_INK, fair_value, clear_width,
                                                       orders, od, position, buy_vol, sell_vol)
        return orders, buy_vol, sell_vol

    def make_SQUID_INK_orders(self, od: OrderDepth, fair_value: float, min_edge: float, position: int,
                                buy_vol: int, sell_vol: int) -> Tuple[List[Order], int, int]:
        orders: List[Order] = []
        aaf = [p for p in od.sell_orders if p >= round(fair_value + min_edge)]
        bbf = [p for p in od.buy_orders if p <= round(fair_value - min_edge)]
        ask = min(aaf) if aaf else round(fair_value + min_edge)
        bid = max(bbf) if bbf else round(fair_value - min_edge)
        buy_vol, sell_vol = self.market_make(Product.SQUID_INK, orders, bid + 1, ask - 1, position, buy_vol, sell_vol)
        return orders, buy_vol, sell_vol

    # ============================= Combined Run Method =============================
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        result: Dict[str, List[Order]] = {}
        trader_obj: Dict[str, Any] = {}
        if state.traderData not in [None, ""]:
            trader_obj = jsonpickle.decode(state.traderData)

        # Restore essential state.
        if "kelp_last_price" in trader_obj:
            self.kelp_last_price = trader_obj["kelp_last_price"]
        if "resin_mid_prices" in trader_obj:
            self.resin_mid_prices = trader_obj["resin_mid_prices"]
        if "mispricing_history" in trader_obj:
            self.mispricing_history = trader_obj["mispricing_history"]
        if "depth_history" in trader_obj:
            self.depth_history = trader_obj["depth_history"]
        # SQUID_INK_last_price is stored in trader_obj by SQUID_INK_fair_value.

        # ---------------- RAINFOREST_RESIN Strategy ----------------
        if "RAINFOREST_RESIN" in state.order_depths:
            resin_pos = state.position.get("RAINFOREST_RESIN", 0)
            resin_orders: List[Order] = []
            od = state.order_depths["RAINFOREST_RESIN"]
            fair_value = self.resin_calculate_fair_value("RAINFOREST_RESIN", od)
            if fair_value is not None:
                take_buy, take_sell = self.resin_take_best_orders("RAINFOREST_RESIN", fair_value,
                                                                   self.resin_params["take_width"],
                                                                   resin_orders, od, resin_pos)
                make_orders, _, _ = self.resin_make_orders("RAINFOREST_RESIN", od, fair_value, resin_pos, take_buy, take_sell)
                resin_orders.extend(make_orders)
            result["RAINFOREST_RESIN"] = resin_orders

        # ---------------- KELP Strategy ----------------
        if "KELP" in state.order_depths:
            kelp_pos = state.position.get("KELP", 0)
            kelp_orders: List[Order] = []
            od = state.order_depths["KELP"]
            kelp_fv = self.kelp_fair_value(od)
            if kelp_fv is not None:
                take_orders, buy_vol, sell_vol = self.kelp_take_orders("KELP", od, kelp_fv, self.kelp_params["take_width"], kelp_pos)
                kelp_orders.extend(take_orders)
                clear_orders, buy_vol, sell_vol = self.kelp_clear_orders("KELP", od, kelp_fv, self.kelp_params["clear_width"], kelp_pos, buy_vol, sell_vol)
                kelp_orders.extend(clear_orders)
                make_orders, buy_vol, sell_vol = self.kelp_make_orders("KELP", od, kelp_fv, kelp_pos, buy_vol, sell_vol)
                kelp_orders.extend(make_orders)
            result["KELP"] = kelp_orders
        trader_obj["kelp_last_price"] = self.kelp_last_price

        # ---------------- Basket/Hedge Strategy ----------------
        current_prices: Dict[str, float] = {}
        for prod, od in state.order_depths.items():
            mid = self.get_mid_price(od)
            if mid is not None:
                current_prices[prod] = mid
        # Update basket depth histories.
        for idx, prod in enumerate(["PICNIC_BASKET1", "PICNIC_BASKET2"], start=1):
            od = state.order_depths.get(prod)
            if od and od.buy_orders and od.sell_orders:
                sell_orders = od.sell_orders
                buy_orders = od.buy_orders
                bid_sz = buy_orders[min(buy_orders)]
                ask_sz = sell_orders[min(sell_orders)]
                avg_depth = (bid_sz + ask_sz) / 2
                buf = self.depth_history[f"BASKET{idx}"]
                buf.append(avg_depth)
                if len(buf) > self.vol_window:
                    buf.pop(0)
        mis = self.calculate_mispricing(current_prices)
        if mis:
            basket_orders = self.generate_basket_orders(mis, state)
            for key, orders_list in basket_orders.items():
                result.setdefault(key, []).extend(orders_list)

        # ---------------- SQUID_INK Strategy ----------------
        if Product.SQUID_INK in self.squid_params and Product.SQUID_INK in state.order_depths:
            squid_pos = state.position.get(Product.SQUID_INK, 0)
            od = state.order_depths[Product.SQUID_INK]
            squid_fv = self.SQUID_INK_fair_value(od, trader_obj)
            squid_take, buy_vol, sell_vol = self.take_orders_SQUID_INK(od, squid_fv,
                                                                         self.squid_params[Product.SQUID_INK]["take_width"],
                                                                         squid_pos)
            squid_clear, buy_vol, sell_vol = self.clear_orders_SQUID_INK(od, squid_fv,
                                                                         self.squid_params[Product.SQUID_INK]["clear_width"],
                                                                         squid_pos, buy_vol, sell_vol)
            squid_make, _, _ = self.make_SQUID_INK_orders(od, squid_fv,
                                                          self.squid_params[Product.SQUID_INK]["SQUID_INK_min_edge"],
                                                          squid_pos, buy_vol, sell_vol)
            result[Product.SQUID_INK] = squid_take + squid_clear + squid_make

        # ---------------- Preserve Minimal State ----------------
        minimal_state = {
            "kelp_last_price": self.kelp_last_price,
            "resin_mid_prices": self.resin_mid_prices,
            "mispricing_history": self.mispricing_history,
            "depth_history": self.depth_history,
            "SQUID_INK_last_price": trader_obj.get("SQUID_INK_last_price", None)
        }
        traderDataEncoded = jsonpickle.encode(minimal_state)
        conversions = 0
        return result, conversions, traderDataEncoded
