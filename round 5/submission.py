from typing import Dict, List, Tuple, Any, Optional
from datamodel import Order, OrderDepth, TradingState, ConversionObservation
import jsonpickle
import math
import statistics
import numpy as np

DAYS_LEFT = 3
class MarketData:
    end_pos: Dict[str, int] = {}
    buy_sum: Dict[str, int] = {}
    sell_sum: Dict[str, int] = {}
    bid_prices: Dict[str, List[float]] = {}
    bid_volumes: Dict[str, List[int]] = {}
    ask_prices: Dict[str, List[float]] = {}
    ask_volumes: Dict[str, List[int]] = {}
    fair: Dict[str, float] = {}

# --- Shared Product Constants ---
class Product:
    SQUID_INK = "SQUID_INK"
    VOLCANIC_ROCK = "VOLCANIC_ROCK"

# --- Default Parameters ---
DEFAULT_SQUID_PARAMS = {
    Product.SQUID_INK: {
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 15,
        "reversion_beta": -0.229,
        "min_edge": 2,
    }
}

STRIKE_PRICES = {
    "VOLCANIC_ROCK_VOUCHER_9500": 9500,
    "VOLCANIC_ROCK_VOUCHER_9750": 9750,
    "VOLCANIC_ROCK_VOUCHER_10000": 10000,
    "VOLCANIC_ROCK_VOUCHER_10250": 10250,
    "VOLCANIC_ROCK_VOUCHER_10500": 10500,
}

def round_price(price: float) -> int:
    """
    Helper to round float prices to nearest integer.
    """
    return int(round(price))

POSITION_LIMITS = {"JAMS": 35}


class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    DJEMBES = "DJEMBES"
    CROISSANT = "CROISSANTS"
    JAMS = "JAMS"
    PICNIC_BASKET2 = "PICNIC_BASKET2"
    ARTIFICAL1 = "ARTIFICAL1"
    ARTIFICAL2 = "ARTIFICAL2"
    SPREAD1 = "SPREAD1"
    SPREAD2 = "SPREAD2"
    VOLCANIC_ROCK = "VOLCANIC_ROCK"
    VOLCANIC_ROCK_VOUCHER_9500 = "VOLCANIC_ROCK_VOUCHER_9500"
    VOLCANIC_ROCK_VOUCHER_9750 = "VOLCANIC_ROCK_VOUCHER_9750"
    VOLCANIC_ROCK_VOUCHER_10000 = "VOLCANIC_ROCK_VOUCHER_10000"
    VOLCANIC_ROCK_VOUCHER_10250 = "VOLCANIC_ROCK_VOUCHER_10250"
    VOLCANIC_ROCK_VOUCHER_10500 = "VOLCANIC_ROCK_VOUCHER_10500"
    MAGNIFICENT_MACARONS = "MAGNIFICENT_MACARONS"


PARAMS = {
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,
        "take_width": 1,
        "clear_width": 0,
        "disregard_edge": 1,
        "join_edge": 2,
        "default_edge": 1,
        "soft_position_limit": 50,  # 30
    },
    Product.KELP: {
        "take_width": 2,
        "clear_width": 0,
        "prevent_adverse": False,
        "adverse_volume": 15,  # 20 - doesn't work as great
        "reversion_beta": -0.18,  # -0.2184
        "disregard_edge": 2,
        "join_edge": 0,
        "default_edge": 1,
        "ink_adjustment_factor": 0.05,
    },
    Product.SQUID_INK: {
        "take_width": 2,
        "clear_width": 1,
        "prevent_adverse": False,
        "adverse_volume": 15,
        "reversion_beta": -0.228,
        "disregard_edge": 2,
        "join_edge": 0,
        "default_edge": 1,
        "spike_lb": 3,
        "spike_ub": 5.6,
        "offset": 2,
        "reversion_window": 55,  # altered
        "reversion_weight": 0.12,
    },
    Product.SPREAD1: {
        "default_spread_mean": 48.777856,
        "default_spread_std": 85.119723,
        "spread_window": 55,
        "zscore_threshold": 4,
        "target_position": 100,
    },
    Product.SPREAD2: {
        "default_spread_mean": 30.2336,
        "default_spread_std": 59.8536,
        "spread_window": 59,
        "zscore_threshold": 6,
        "target_position": 100,
    },
    Product.PICNIC_BASKET1: {
        "b2_adjustment_factor": 0.05
    },
    Product.VOLCANIC_ROCK_VOUCHER_9500: {
        "mean_volatility": 0.119077,
        # "threshold": 0.1,
        "strike": 10000,  # this might be silly but profits are much higher with this strike
        # I do not know why that is the case for all of them
        # You can try if you want the actual strike but it makes a loss
        # to note I've uploaded this exactly after I have managed to get the code to run
        # I will look into this now along with mean-reverting deltas for selected vouchers
        "starting_time_to_expiry": 7 / 365,
        "std_window": 6,
        "z_score_threshold": 21,
    },
    Product.VOLCANIC_ROCK_VOUCHER_9750: {
        "mean_volatility": 0.147417,
        # "threshold": 0,
        "strike": 10000,
        "starting_time_to_expiry": 7 / 365,
        "std_window": 6,
        "z_score_threshold": 21,
    },
    Product.VOLCANIC_ROCK_VOUCHER_10000: {
        "mean_volatility": 0.140554,
        # "threshold": 0.01,
        "strike": 10000,
        "starting_time_to_expiry": 7 / 365,
        "std_window": 6,
        "z_score_threshold": 21,
    },
    Product.VOLCANIC_ROCK_VOUCHER_10250: {
        "mean_volatility": 0.128666,
        # "threshold": 0.728011,
        "strike": 10000,
        "starting_time_to_expiry": 7 / 365,
        "std_window": 6,
        "z_score_threshold": 25,
    },
    Product.VOLCANIC_ROCK_VOUCHER_10500: {
        "mean_volatility": 0.127146,
        # "threshold": 0.0552,
        "strike": 10000,
        "starting_time_to_expiry": 7 / 365,
        "std_window": 6,
        "z_score_threshold": 25,
    }

}

PICNIC1_WEIGHTS = {
    Product.DJEMBES: 1,
    Product.CROISSANT: 6,
    Product.JAMS: 3,
}
PICNIC2_WEIGHTS = {
    Product.CROISSANT: 4,
    Product.JAMS: 2}


class Trader:
    def __init__(self, traderData: str = None):
        # ---------------------------- Shared Configuration ----------------------------
        self.rsi_period = 14
        self.atr_period = 14
        
        self.PRODUCT_LIMIT = {Product.RAINFOREST_RESIN: 50,
                              Product.KELP: 50,
                              Product.SQUID_INK: 50,
                              Product.CROISSANT: 250,
                              Product.JAMS: 350,
                              Product.DJEMBES: 60,
                              Product.PICNIC_BASKET1: 60,
                              Product.PICNIC_BASKET2: 100,
                              Product.VOLCANIC_ROCK: 400,
                              Product.VOLCANIC_ROCK_VOUCHER_9500: 200,
                              Product.VOLCANIC_ROCK_VOUCHER_9750: 200,
                              Product.VOLCANIC_ROCK_VOUCHER_10000: 200,
                              Product.VOLCANIC_ROCK_VOUCHER_10250: 200,
                              Product.VOLCANIC_ROCK_VOUCHER_10500: 200,
                              Product.MAGNIFICENT_MACARONS:75}
        self.position_limits = {
            "RAINFOREST_RESIN": 50,
            "KELP": 50,
            "CROISSANTS": 250,
            "JAMS": 350,
            "DJEMBES": 60,
            "PICNIC_BASKET1": 60,
            "PICNIC_BASKET2": 100,
            Product.SQUID_INK: 20,
            **{p: 200 for p in STRIKE_PRICES},
            Product.VOLCANIC_ROCK: 400,
        }

        # ------------------- RAINFOREST_RESIN Setup -------------------
        self.resin_mid_prices: List[float] = []
        self.resin_rolling_window = 10
        self.resin_params = {
            "take_width": 1,
            "clear_width": 0,
            "disregard_edge": 1,
            "join_edge": 2,
            "default_edge": 4,
            "soft_position_limit": 30,
        }

        # ------------------- KELP Setup -------------------
        self.kelp_last_price = None
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
        
                # After defining self.position_limits:
        self.strike_prices = STRIKE_PRICES.copy()


        # ------------------- Basket/Hedge Setup -------------------
        self.basket1_composition = {"CROISSANTS": 6, "JAMS": 3, "DJEMBES": 1}
        self.basket2_composition = {"CROISSANTS": 4, "JAMS": 2}
        self.mispricing_history = {"BASKET1": [], "BASKET2": []}
        self.depth_history = {"BASKET1": [], "BASKET2": []}
        self.vol_window = 20
        self.sigma_min = 0.002
        self.sigma_max = 0.02
        self.mispricing_threshold_percent = {"BASKET1": 0.001, "BASKET2": 0.002}
        self.soft_liquidate_vol_percentage = {"BASKET1": 1, "BASKET2": 1}

        # ------------------- SQUID_INK Setup -------------------
        self.squid_params = DEFAULT_SQUID_PARAMS

        # ------------------- Volatility / Options Setup -------------------
        self.lambda_ewma = 0.97
        self.volatility = 0.16
        self.prev_price = None
        self.iv_history: List[float] = []

        self.total_days_to_expiry = 2
        self.days_left = self.total_days_to_expiry

        self.target_daily_pnl_vol = 1000
        self.max_loss_per_contract = 10
        self.max_daily_drawdown = 2000

        self.entry_prices: Dict[str, float] = {}
        self.unrealized_pnl: Dict[str, float] = {}
        self.high_water_mark = 0.0
        self.equity = 0.0

        self.last_profit_iv: List[float] = []
        self.last_profit_arb: List[float] = []
        self.profit_window = 10
        self.strategy_reset_window = 100

        self.chosen_strategy = None
        self.trade_book: Dict[str, List[Tuple[int, float]]] = {}
        self.realized_profit = 0.0

        # Restore from saved state if provided
        if traderData:
            restored = jsonpickle.decode(traderData)
            self.__dict__.update(restored)
        
        self.LIMIT: Dict[str, int] = {"MAGNIFICENT_MACARONS": 75}
        self.params: Dict[str, Dict[str, float]] = {
            "MACARONS": {
                "init_make_edge": 2.0,
                "min_edge": 0.5,
                "volume_avg_timestamp": 5,
                "volume_bar": 20,
                "dec_edge_discount": 0.8,
                "step_size": 0.5,
                "make_probability": 0.5,
                # number of consecutive ticks below threshold
                "sunlight_duration": 10
            }
        }
        # limit on maximum conversions per time-step
        self.conversion_limit = 10
        # critical sunlight threshold
        self.sunlight_critical = 45
        
        self.price_history: Dict[str, List[float]] = {product: [] for product in self.position_limits}

    def get_mid_price(self, order_depth: OrderDepth) -> float | None:
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        return (best_bid + best_ask) / 2

    def update_price_history(self, product: str, mid_price: float):
        history = self.price_history[product]
        history.append(mid_price)
        # Keep only the most recent 50 entries
        if len(history) > 50:
            history.pop(0)

    def compute_stochastic(self, prices: List[float], period: int = 14) -> float:
        if len(prices) < period:
            return 50.0  # Neutral
        high = max(prices[-period:])
        low = min(prices[-period:])
        if high == low:
            return 50.0
        return (prices[-1] - low) / (high - low) * 100

    
    def implied_bid_ask(
        self,
        obs: ConversionObservation
    ) -> Tuple[float, float]:
        ibid = obs.bidPrice - obs.exportTariff - obs.transportFees - 0.1
        iask = obs.askPrice + obs.importTariff + obs.transportFees
        return ibid, iask

    def adapt_edge(
        self,
        timestamp: int,
        curr_edge: float,
        position: int,
        state: Dict
    ) -> float:
        p = self.params["MACARONS"]

        if timestamp == 0:
            state["curr_edge"] = p["init_make_edge"]
            state.setdefault("volume_history", [])
            state.setdefault("optimized", False)
            return state["curr_edge"]

        history = state.setdefault("volume_history", [])
        history.append(abs(position))
        if len(history) > p["volume_avg_timestamp"]:
            history.pop(0)

        if len(history) == p["volume_avg_timestamp"] and not state.get("optimized", False):
            avg_vol = np.mean(history)
            if avg_vol >= p["volume_bar"]:
                history.clear()
                state["curr_edge"] = curr_edge + p["step_size"]
                return state["curr_edge"]
            threshold = p["dec_edge_discount"] * p["volume_bar"] * (curr_edge - p["step_size"])
            if threshold > avg_vol * curr_edge:
                if curr_edge - p["step_size"] > p["min_edge"]:
                    history.clear()
                    state["curr_edge"] = curr_edge - p["step_size"]
                    state["optimized"] = True
                    return state["curr_edge"]
                state["curr_edge"] = p["min_edge"]
                return state["curr_edge"]

        return curr_edge

    def arb_clear(self, position: int) -> int:
        # convert only up to conversion_limit when clearing long
        if position > 0:
            return -min(position, self.conversion_limit)
        return 0

    def arb_take(
        self,
        depth: OrderDepth,
        obs: ConversionObservation,
        edge: float,
        position: int
    ) -> Tuple[List[Order], int, int]:
        orders: List[Order] = []
        buy_vol = 0
        sell_vol = 0
        limit = self.LIMIT["MAGNIFICENT_MACARONS"]

        # implied prices
        ibid, iask = self.implied_bid_ask(obs)

        # remaining capacities
        buy_capacity = limit - position
        sell_capacity = limit + position

        # dynamic ask adjustment
        ask_ref = iask + edge
        foreign_mid = (obs.askPrice + obs.bidPrice) / 2
        aggressive_ask = foreign_mid - 1.6
        if aggressive_ask > iask:
            ask_ref = aggressive_ask

        # take-side edge scaling
        take_edge = (ask_ref - iask) * self.params["MACARONS"]["make_probability"]

        # BUY: lift asks below implied bid minus edge
        for price in sorted(depth.sell_orders.keys()):
            if price > ibid - take_edge:
                break
            qty = min(abs(depth.sell_orders[price]), buy_capacity - buy_vol)
            if qty > 0:
                orders.append(Order("MAGNIFICENT_MACARONS", round_price(price), qty))
                buy_vol += qty

        # SELL: hit bids above implied ask plus edge
        for price in sorted(depth.buy_orders.keys(), reverse=True):
            if price < iask + take_edge:
                break
            qty = min(abs(depth.buy_orders[price]), sell_capacity - sell_vol)
            if qty > 0:
                orders.append(Order("MAGNIFICENT_MACARONS", round_price(price), -qty))
                sell_vol += qty

        return orders, buy_vol, sell_vol

    def arb_make(
        self,
        depth: OrderDepth,
        obs: ConversionObservation,
        position: int,
        edge: float,
        buy_vol: int,
        sell_vol: int
    ) -> Tuple[List[Order], int, int]:
        orders: List[Order] = []
        limit = self.LIMIT["MAGNIFICENT_MACARONS"]

        ibid, iask = self.implied_bid_ask(obs)
        min_edge = self.params["MACARONS"]["min_edge"]
        foreign_mid = (obs.askPrice + obs.bidPrice) / 2
        aggressive_ask = foreign_mid - 1.6

        bid_price = ibid - edge
        ask_price = iask + edge
        if aggressive_ask >= iask + min_edge:
            ask_price = aggressive_ask

        # filter levels for competitiveness
        filtered_asks = sorted([p for p, v in depth.sell_orders.items() if abs(v) >= 40])
        filtered_bids = sorted([p for p, v in depth.buy_orders.items() if abs(v) >= 25], reverse=True)

        if filtered_asks and ask_price > filtered_asks[0]:
            ask_price = filtered_asks[0] - 1 if filtered_asks[0] - 1 > iask else iask + edge
        if filtered_bids and bid_price < filtered_bids[0]:
            bid_price = filtered_bids[0] + 1 if filtered_bids[0] + 1 < ibid else ibid - edge

        # size calculations
        buy_qty = limit - (position + buy_vol)
        if buy_qty > 0:
            orders.append(Order("MAGNIFICENT_MACARONS", round_price(bid_price), buy_qty))

        sell_qty = limit + (position - sell_vol)
        if sell_qty > 0:
            orders.append(Order("MAGNIFICENT_MACARONS", round_price(ask_price), -sell_qty))

        return orders, buy_vol, sell_vol

    # ============================= Shared Helpers =============================
    def get_mid_price(self, od: OrderDepth) -> float:
        if od.sell_orders and od.buy_orders:
            return (min(od.sell_orders) + max(od.buy_orders)) / 2
        if od.sell_orders:
            return min(od.sell_orders)
        if od.buy_orders:
            return max(od.buy_orders)
        return None

    # ============================= RAINFOREST_RESIN Methods =============================
    def resin_calculate_fair_value(self, od: OrderDepth) -> float:
        if od.sell_orders and od.buy_orders:
            best_ask = min(od.sell_orders)
            best_bid = max(od.buy_orders)
            mid_price = (best_ask + best_bid) / 2
            self.resin_mid_prices.append(mid_price)
            if len(self.resin_mid_prices) > self.resin_rolling_window:
                self.resin_mid_prices.pop(0)
            if len(self.resin_mid_prices) == self.resin_rolling_window:
                return sum(self.resin_mid_prices) / self.resin_rolling_window
            return mid_price
        return None

    def resin_take_best_orders(self, product: str, fair_value: float,
                               orders: List[Order], od: OrderDepth,
                               position: int) -> Tuple[int, int]:
        buy_vol = sell_vol = 0
        pos_limit = self.position_limits[product]

        if od.sell_orders:
            best_ask = min(od.sell_orders)
            amt = -od.sell_orders[best_ask]
            if best_ask <= fair_value - self.resin_params["take_width"]:
                qty = min(amt, pos_limit - position)
                if qty > 0:
                    orders.append(Order(product, best_ask, qty))
                    buy_vol += qty

        if od.buy_orders:
            best_bid = max(od.buy_orders)
            amt = od.buy_orders[best_bid]
            if best_bid >= fair_value + self.resin_params["take_width"]:
                qty = min(amt, pos_limit + position)
                if qty > 0:
                    orders.append(Order(product, best_bid, -qty))
                    sell_vol += qty

        return buy_vol, sell_vol
    
    
    def resin_make_orders(self, product: str, order_depth: OrderDepth, fair_value: float,
                          position: int, buy_order_volume: int, sell_order_volume: int) -> List[Order]:
        """Place limit orders on both sides of the book with position management."""
        orders = []

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

        # Adjust order prices if positions become too large.
        if position > self.resin_params["soft_position_limit"]:
            ask -= 1
        elif position < -self.resin_params["soft_position_limit"]:
            bid += 1

        position_limit = self.position_limits[product]
        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, bid, buy_quantity))

        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, ask, -sell_quantity))

        return orders

    # def resin_make_orders(self, product: str, od: OrderDepth, fair_value: float,
    #                       position: int, buy_vol: int, sell_vol: int) -> List[Order]:
        
    #     orders: List[Order] = []
    #     p = self.resin_params
    #     asks_above = [pr for pr in od.sell_orders if pr > fair_value + p["disregard_edge"]]
    #     bids_below = [pr for pr in od.buy_orders if pr < fair_value - p["disregard_edge"]]

    #     ask = round(fair_value + p["default_edge"])
    #     if asks_above:
    #         best = min(asks_above)
    #         ask = best if abs(best - fair_value) <= p["join_edge"] else best - 1

    #     bid = round(fair_value - p["default_edge"])
    #     if bids_below:
    #         best = max(bids_below)
    #         bid = best if abs(fair_value - best) <= p["join_edge"] else best + 1

    #     # soft limit skew
    #     if position > p["soft_position_limit"]:
    #         ask -= 1
    #     elif position < -p["soft_position_limit"]:
    #         bid += 1

    #     pos_limit = self.position_limits[product]
    #     buy_qty = pos_limit - (position + buy_vol)
    #     if buy_qty > 0:
    #         orders.append(Order(product, bid, buy_qty))

    #     sell_qty = pos_limit + (position - sell_vol)
    #     if sell_qty > 0:
    #         orders.append(Order(product, ask, -sell_qty))

    #     return orders

    # ============================= KELP Methods =============================
    def kelp_fair_value(self, od: OrderDepth) -> float:
        if od.sell_orders and od.buy_orders:
            best_ask = min(od.sell_orders)
            best_bid = max(od.buy_orders)

            filt_ask = [pr for pr, sz in od.sell_orders.items() if abs(sz) >= self.kelp_params["adverse_volume"]]
            filt_bid = [pr for pr, sz in od.buy_orders.items() if abs(sz) >= self.kelp_params["adverse_volume"]]

            mm_ask = min(filt_ask) if filt_ask else None
            mm_bid = max(filt_bid) if filt_bid else None

            mid = ((best_ask + best_bid) / 2) if (mm_ask is None or mm_bid is None) else ((mm_ask + mm_bid) / 2)

            if self.kelp_last_price is not None and mm_ask is not None and mm_bid is not None:
                ret = (mid - self.kelp_last_price) / self.kelp_last_price
                fair = mid + mid * ret * self.kelp_params["reversion_beta"]
            else:
                fair = mid

            self.kelp_last_price = mid
            return fair
        return None

    def kelp_take_orders(self, product: str, od: OrderDepth, fair_value: float,
                         position: int) -> Tuple[List[Order], int, int]:
        orders: List[Order] = []
        buy_vol = sell_vol = 0
        p = self.kelp_params
        pos_limit = self.position_limits[product]

        if od.sell_orders:
            best_ask = min(od.sell_orders)
            amt = -od.sell_orders[best_ask]
            cond = (not p["prevent_adverse"] or amt <= p["adverse_volume"])
            if cond and best_ask <= fair_value - p["take_width"]:
                qty = min(amt, pos_limit - position)
                if qty > 0:
                    orders.append(Order(product, best_ask, qty))
                    buy_vol += qty

        if od.buy_orders:
            best_bid = max(od.buy_orders)
            amt = od.buy_orders[best_bid]
            cond = (not p["prevent_adverse"] or amt <= p["adverse_volume"])
            if cond and best_bid >= fair_value + p["take_width"]:
                qty = min(amt, pos_limit + position)
                if qty > 0:
                    orders.append(Order(product, best_bid, -qty))
                    sell_vol += qty

        return orders, buy_vol, sell_vol

    def kelp_clear_orders(self, product: str, od: OrderDepth, fair_value: float,
                          position: int, buy_vol: int, sell_vol: int) -> Tuple[List[Order], int, int]:
        orders: List[Order] = []
        p = self.kelp_params
        pos_after = position + buy_vol - sell_vol
        fair_bid = round(fair_value - p["clear_width"])
        fair_ask = round(fair_value + p["clear_width"])
        pos_limit = self.position_limits[product]

        # clear profitable
        if pos_after > 0 and od.buy_orders:
            clear_qty = sum(sz for pr, sz in od.buy_orders.items() if pr >= fair_ask)
            sent = min(pos_after, pos_limit + (position - sell_vol), clear_qty)
            if sent > 0:
                orders.append(Order(product, fair_ask, -sent))
                sell_vol += sent

        if pos_after < 0 and od.sell_orders:
            clear_qty = sum(abs(sz) for pr, sz in od.sell_orders.items() if pr <= fair_bid)
            sent = min(abs(pos_after), pos_limit - (position + buy_vol), clear_qty)
            if sent > 0:
                orders.append(Order(product, fair_bid, sent))
                buy_vol += sent

        return orders, buy_vol, sell_vol

    def kelp_make_orders(self, product: str, od: OrderDepth, fair_value: float,
                         position: int, buy_vol: int, sell_vol: int) -> Tuple[List[Order], int, int]:
        orders: List[Order] = []
        p = self.kelp_params
        asks_above = [pr for pr in od.sell_orders if pr > fair_value + p["disregard_edge"]]
        bids_below = [pr for pr in od.buy_orders if pr < fair_value - p["disregard_edge"]]

        ask = round(fair_value + p["default_edge"])
        if asks_above:
            best = min(asks_above)
            ask = best if abs(best - fair_value) <= p["join_edge"] else best - 1

        bid = round(fair_value - p["default_edge"])
        if bids_below:
            best = max(bids_below)
            bid = best if abs(fair_value - best) <= p["join_edge"] else best + 1

        pos_limit = self.position_limits[product]
        if pos_limit - (position + buy_vol) > 0:
            orders.append(Order(product, bid, pos_limit - (position + buy_vol)))
        if pos_limit + (position - sell_vol) > 0:
            orders.append(Order(product, ask, -(pos_limit + (position - sell_vol))))

        return orders, buy_vol, sell_vol

    # ============================= Basket/Hedge Methods =============================
    def calculate_theoretical_value(self, prices: Dict[str, float], basket: int) -> float:
        comp = self.basket1_composition if basket == 1 else self.basket2_composition
        if all(p in prices for p in comp):
            return sum(comp[p] * prices[p] for p in comp)
        return None

    def calculate_mispricing(self, prices: Dict[str, float]) -> Dict[str, dict]:
        out = {}
        for idx, prod in enumerate(["PICNIC_BASKET1", "PICNIC_BASKET2"], start=1):
            if prod not in prices:
                continue
            tv = self.calculate_theoretical_value(prices, idx)
            if tv is None:
                continue
            diff = prices[prod] - tv
            pct = diff / tv
            key = f"BASKET{idx}"
            hist = self.mispricing_history[key]
            hist.append(pct)
            if len(hist) > self.vol_window:
                hist.pop(0)
            out[key] = {
                "market_price": prices[prod],
                "theoretical_value": tv,
                "diff": diff,
                "percent_diff": pct,
            }
        return out

    def dynamic_threshold(self, key: str) -> float:
        hist = self.mispricing_history[key]
        base = self.mispricing_threshold_percent[key]
        if len(hist) < 2:
            return base
        sigma = statistics.pstdev(hist)
        v = (sigma - self.sigma_min) / (self.sigma_max - self.sigma_min)
        v = max(0.0, min(1.0, v))
        return base * (1 + v)  # adjust as needed

    def generate_basket_orders(self, mis: Dict[str, dict], state: TradingState) -> Dict[str, List[Order]]:
        orders: Dict[str, List[Order]] = {}
        for idx, prod in enumerate(["PICNIC_BASKET1"], start=1):
            key = f"BASKET{idx}"
            if key not in mis:
                continue
            pct = mis[key]["percent_diff"]
            thr = self.dynamic_threshold(key)
            book = state.order_depths.get(prod)
            pos = state.position.get(prod, 0)
            # decide side
            if pct > thr and book and book.buy_orders:
                price = max(book.buy_orders)
                avail = book.buy_orders[price]
                vol = int(self.soft_liquidate_vol_percentage[key] * min(avail, self.position_limits[prod] + pos))
                if vol > 0:
                    orders.setdefault(prod, []).append(Order(prod, price, -vol))
                    # hedge JAMS & CROISSANTS similarly...
            elif pct < -thr and book and book.sell_orders:
                price = min(book.sell_orders)
                avail = book.sell_orders[price]
                vol = int(self.soft_liquidate_vol_percentage[key] * min(abs(avail), self.position_limits[prod] - pos))
                if vol > 0:
                    orders.setdefault(prod, []).append(Order(prod, price, vol))
                    # hedge JAMS & CROISSANTS similarly...
        return orders

    # ============================= SQUID_INK Methods =============================
    def SQUID_INK_fair_value(self, od: OrderDepth, state_obj: Dict[str, Any]) -> float:
        params = self.squid_params[Product.SQUID_INK]
        if od.sell_orders and od.buy_orders:
            best_ask = min(od.sell_orders)
            best_bid = max(od.buy_orders)
            filt_ask = [pr for pr, sz in od.sell_orders.items() if abs(sz) >= params["adverse_volume"]]
            filt_bid = [pr for pr, sz in od.buy_orders.items() if abs(sz) >= params["adverse_volume"]]
            mm_ask = min(filt_ask) if filt_ask else None
            mm_bid = max(filt_bid) if filt_bid else None
            if mm_ask is None or mm_bid is None:
                mid = (best_ask + best_bid) / 2 if state_obj.get("SQUID_INK_last_price") is None else state_obj["SQUID_INK_last_price"]
            else:
                mid = (mm_ask + mm_bid) / 2
            if state_obj.get("SQUID_INK_last_price") is not None:
                last = state_obj["SQUID_INK_last_price"]
                ret = (mid - last) / last
                fair = mid + mid * ret * params["reversion_beta"]
            else:
                fair = mid
            state_obj["SQUID_INK_last_price"] = mid
            return fair
        return None

    def take_orders_SQUID_INK(self, od: OrderDepth, fair_value: float, position: int) -> Tuple[List[Order], int, int]:
        orders: List[Order] = []
        buy_vol = sell_vol = 0
        params = self.squid_params[Product.SQUID_INK]
        if params["prevent_adverse"]:
            buy_vol, sell_vol = self._take_with_adverse(Product.SQUID_INK, fair_value, orders, od, position, buy_vol, sell_vol, params["adverse_volume"])
        else:
            buy_vol, sell_vol = self._take_basic(Product.SQUID_INK, fair_value, orders, od, position, buy_vol, sell_vol)
        return orders, buy_vol, sell_vol

    def _take_basic(self, product, fair, orders, od, pos, buy_vol, sell_vol):
        limit = self.position_limits[product]
        if od.sell_orders:
            ask = min(od.sell_orders)
            amt = -od.sell_orders[ask]
            if ask <= fair - self.squid_params[product]["take_width"]:
                qty = min(amt, limit - pos)
                if qty>0:
                    orders.append(Order(product, ask, qty))
                    buy_vol+=qty
                    od.sell_orders[ask]+=qty
                    if od.sell_orders[ask]==0: del od.sell_orders[ask]
        if od.buy_orders:
            bid = max(od.buy_orders)
            amt = od.buy_orders[bid]
            if bid>= fair + self.squid_params[product]["take_width"]:
                qty = min(amt, limit + pos)
                if qty>0:
                    orders.append(Order(product, bid, -qty))
                    sell_vol+=qty
                    od.buy_orders[bid]-=qty
                    if od.buy_orders[bid]==0: del od.buy_orders[bid]
        return buy_vol, sell_vol

    def _take_with_adverse(self, product, fair, orders, od, pos, buy_vol, sell_vol, adv):
        limit = self.position_limits[product]
        if od.sell_orders:
            ask = min(od.sell_orders)
            amt = -od.sell_orders[ask]
            if amt<=adv and ask<=fair-self.squid_params[product]["take_width"]:
                qty = min(amt, limit-pos)
                if qty>0:
                    orders.append(Order(product, ask, qty))
                    buy_vol+=qty
                    od.sell_orders[ask]+=qty
                    if od.sell_orders[ask]==0: del od.sell_orders[ask]
        if od.buy_orders:
            bid = max(od.buy_orders)
            amt = od.buy_orders[bid]
            if amt<=adv and bid>=fair+self.squid_params[product]["take_width"]:
                qty = min(amt, limit+pos)
                if qty>0:
                    orders.append(Order(product, bid, -qty))
                    sell_vol+=qty
                    od.buy_orders[bid]-=qty
                    if od.buy_orders[bid]==0: del od.buy_orders[bid]
        return buy_vol, sell_vol

    def clear_orders_SQUID_INK(self, od: OrderDepth, fair_value: float, position: int, buy_vol: int, sell_vol: int) -> Tuple[List[Order], int, int]:
        orders: List[Order] = []
        p = self.squid_params[Product.SQUID_INK]
        return self.clear_position_order(Product.SQUID_INK, fair_value, p["clear_width"], orders, od, position, buy_vol, sell_vol)

    def make_SQUID_INK_orders(self,
                              od: OrderDepth,
                              fair_value: float,
                              position: int,
                              buy_vol: int,
                              sell_vol: int
                              ) -> Tuple[List[Order], int, int]:
        """
        Place residual market‐making orders for SQUID_INK after
        taking and clearing. Returns (orders, buy_vol, sell_vol).
        """
        orders: List[Order] = []
        params = self.squid_params[Product.SQUID_INK]
        # find edge‐qualified quotes
        asks = [p for p in od.sell_orders if p >= round(fair_value + params["min_edge"])]
        bids = [p for p in od.buy_orders if p <= round(fair_value - params["min_edge"])]
        ask = min(asks) if asks else round(fair_value + params["min_edge"])
        bid = max(bids) if bids else round(fair_value - params["min_edge"])

        # now fill out to position limits
        # use market_make helper which returns updated buy_vol, sell_vol
        new_buy_vol, new_sell_vol = self.market_make(
            Product.SQUID_INK,
            orders,
            bid + 1,
            ask - 1,
            position,
            buy_vol,
            sell_vol
        )
        return orders, new_buy_vol, new_sell_vol


    def market_make(self, product, orders, bid, ask, pos, buy_vol, sell_vol):
        limit = self.position_limits[product]
        buy_qty = limit - (pos + buy_vol)
        if buy_qty>0:
            orders.append(Order(product, round(bid), buy_qty))
        sell_qty = limit + (pos - sell_vol)
        if sell_qty>0:
            orders.append(Order(product, round(ask), -sell_qty))
        return buy_vol, sell_vol

    def clear_position_order(self, product, fair, width, orders, od, pos, buy_vol, sell_vol):
        pos_after = pos + buy_vol - sell_vol
        fair_bid = round(fair - width)
        fair_ask = round(fair + width)
        limit = self.position_limits[product]
        if pos_after>0:
            clear_qty = sum(sz for pr, sz in od.buy_orders.items() if pr>=fair_ask)
            sent = min(clear_qty, pos_after, limit+(pos-sell_vol))
            if sent>0:
                orders.append(Order(product, fair_ask, -sent))
                sell_vol+=sent
        elif pos_after<0:
            clear_qty = sum(abs(sz) for pr, sz in od.sell_orders.items() if pr<=fair_bid)
            sent = min(clear_qty, -pos_after, limit-(pos+buy_vol))
            if sent>0:
                orders.append(Order(product, fair_bid, sent))
                buy_vol+=sent
        return orders, buy_vol, sell_vol

    # ============================= Volatility / Options Methods =============================
    def update_ewma_vol(self, S: float):
        if self.prev_price and self.prev_price>0:
            ret = math.log(S/self.prev_price)
            one_day_var = self.volatility**2/252
            var = (1-self.lambda_ewma)*(ret**2) + self.lambda_ewma*one_day_var
            self.volatility = math.sqrt(var*252)
        self.prev_price = S
        self.iv_history.append(self.volatility)

    def time_to_expiry(self) -> float:
        self.days_left = max(self.days_left-1, 0)
        T = self.days_left/252
        return max(T, 1/252/6)

    def black_scholes_call(self, S, K, T, r, sigma):
        if T<=0:
            return max(S-K,0)
        d1 = (math.log(S/K)+(r+0.5*sigma**2)*T)/(sigma*math.sqrt(T))
        d2 = d1 - sigma*math.sqrt(T)
        cdf = lambda x: 0.5*(1+math.erf(x/math.sqrt(2)))
        return S*cdf(d1)-K*math.exp(-r*T)*cdf(d2)

    import math

    def calculate_vega(self, S, K, T, r, sigma):
    # Safety checks to prevent math domain errors
        if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
            return 0.0

        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        pdf = math.exp(-0.5 * d1 ** 2) / math.sqrt(2 * math.pi)
        return S * math.sqrt(T) * pdf


    def theoretical(self, S, T) -> Tuple[Dict[str,float],Dict[str,float]]:
        theo, delta = {}, {}
        for v,K in self.strike_prices.items():
            price = self.black_scholes_call(S, K, T, 0.0, self.volatility)
            d1 = (math.log(S/K)+(0.5*self.volatility**2)*T)/(self.volatility*math.sqrt(T))
            delta[v] = 0.5*(1+math.erf(d1/math.sqrt(2)))
            theo[v] = price
        return theo, delta

    def pnl_and_drawdown_check(self, state: TradingState) -> float:
        total_unreal = 0.0
        for v in self.strike_prices:
            pos = state.position.get(v,0)
            if pos!=0 and v in state.order_depths and v in self.entry_prices:
                mid = self.get_mid_price(state.order_depths[v])
                if mid is not None:
                    pnl = (mid - self.entry_prices[v]) * pos
                    self.unrealized_pnl[v] = pnl
                    total_unreal += pnl
        self.equity = self.high_water_mark + total_unreal
        if self.equity>self.high_water_mark:
            self.high_water_mark = self.equity
        return self.high_water_mark - self.equity

    def calculate_profit(self, orders: Dict[str,List[Order]], state: TradingState, strategy: str) -> float:
        profit = 0.0
        for product, lst in orders.items():
            if product not in self.trade_book:
                self.trade_book[product] = []
            for order in lst:
                if order.quantity>0:
                    self.trade_book[product].append((order.quantity, order.price))
                elif order.quantity<0:
                    sell_qty = -order.quantity
                    sell_price = order.price
                    book = self.trade_book[product]
                    while sell_qty>0 and book:
                        buy_qty, buy_price = book[0]
                        if buy_qty<=sell_qty:
                            matched = buy_qty
                            profit += matched*(sell_price-buy_price)
                            sell_qty -= matched
                            book.pop(0)
                        else:
                            matched = sell_qty
                            profit += matched*(sell_price-buy_price)
                            book[0] = (buy_qty-matched, buy_price)
                            sell_qty = 0
        self.realized_profit += profit
        return profit

    def generate_iv_orders(self, state: TradingState) -> Dict[str,List[Order]]:
        orders: Dict[str,List[Order]] = {}
        T = self.time_to_expiry()
        if not self.iv_history:
            sma_short = sma_long = self.volatility
        else:
            N_short = min(50, len(self.iv_history))
            sma_short = sum(self.iv_history[-N_short:])/N_short
            N_long = min(200, len(self.iv_history))
            sma_long = sum(self.iv_history[-N_long:])/N_long

        signal = sma_short - sma_long
        threshold = 0.0

        for v in self.strike_prices:
            # if v=="VOLCANIC_ROCK_VOUCHER_10000": continue
            od = state.order_depths.get(v)
            if not od:
                continue
            mid = self.get_mid_price(od)
            if mid is None:
                continue

            vega = self.calculate_vega(mid, self.strike_prices[v], T, 0.0, self.volatility)
            max_notional = (self.target_daily_pnl_vol / (vega*self.volatility*math.sqrt(T))) if vega>0 else self.position_limits[v]
            size_limit = min(abs(int(max_notional)), self.position_limits[v])

            if signal>threshold and od.sell_orders:
                price = min(od.sell_orders)
                vol = min(abs(od.sell_orders[price]), size_limit)
                if vol>0:
                    orders.setdefault(v,[]).append(Order(v, price, vol))
                    self.entry_prices.setdefault(v, price)
            elif signal< -threshold and od.buy_orders:
                price = max(od.buy_orders)
                vol = min(abs(od.buy_orders[price]), size_limit)
                if vol>0:
                    orders.setdefault(v,[]).append(Order(v, price, -vol))
                    self.entry_prices.setdefault(v, price)

        profit = self.calculate_profit(orders, state, "iv")
        self.last_profit_iv.append(profit)
        return orders

    def generate_arbitrage_orders(self, theo: Dict[str,float], state: TradingState) -> Dict[str,List[Order]]:
        orders: Dict[str,List[Order]] = {}
        T = self.time_to_expiry()

        for v in self.strike_prices:
            # if v=="VOLCANIC_ROCK_VOUCHER_10000": continue
            od = state.order_depths.get(v)
            if not od:
                continue
            mid = self.get_mid_price(od)
            if mid is None:
                continue

            vega = self.calculate_vega(mid, self.strike_prices[v], T, 0.0, self.volatility)
            max_notional = (self.target_daily_pnl_vol / (vega*self.volatility*math.sqrt(T))) if vega>0 else self.position_limits[v]
            size_limit = min(abs(int(max_notional)), self.position_limits[v])

            buffer = 0.1 * self.volatility * math.sqrt(T) * mid
            buy_thr = theo[v] - buffer
            sell_thr = theo[v] + buffer

            if mid < buy_thr and od.sell_orders:
                price = min(od.sell_orders)
                vol = min(abs(od.sell_orders[price]), size_limit)
                if vol>0:
                    orders.setdefault(v,[]).append(Order(v, price, vol))
                    self.entry_prices.setdefault(v, price)

            if mid > sell_thr and od.buy_orders:
                price = max(od.buy_orders)
                vol = min(abs(od.buy_orders[price]), size_limit)
                if vol>0:
                    orders.setdefault(v,[]).append(Order(v, price, -vol))
                    self.entry_prices.setdefault(v, price)

        profit = self.calculate_profit(orders, state, "arb")
        self.last_profit_arb.append(profit)
        return orders

    def generate_stop_loss_orders(self, state: TradingState) -> Dict[str,List[Order]]:
        orders: Dict[str,List[Order]] = {}
        for v, pnl in self.unrealized_pnl.items():
            pos = state.position.get(v,0)
            if pos!=0 and pnl/abs(pos) < -self.max_loss_per_contract:
                od = state.order_depths.get(v)
                if od:
                    if pos>0 and od.buy_orders:
                        price = max(od.buy_orders)
                        orders[v] = [Order(v, price, -pos)]
                    elif pos<0 and od.sell_orders:
                        price = min(od.sell_orders)
                        orders[v] = [Order(v, price, -pos)]
        return orders

    def generate_delta_hedge(self, delta: Dict[str,float], state: TradingState) -> Dict[str,List[Order]]:
        orders: Dict[str,List[Order]] = {}
        total_delta = sum(delta[v] * state.position.get(v,0) for v in delta)
        target = round(total_delta)
        current = state.position.get(Product.VOLCANIC_ROCK, 0)
        adjust = target - current
        if abs(adjust) < 1:
            return orders
        od = state.order_depths.get(Product.VOLCANIC_ROCK)
        if not od:
            return orders

        limit = self.position_limits[Product.VOLCANIC_ROCK]
        if adjust>0 and od.sell_orders:
            price = min(od.sell_orders)
            vol = min(adjust, limit-current)
            if vol>0:
                orders[Product.VOLCANIC_ROCK] = [Order(Product.VOLCANIC_ROCK, price, vol)]
        elif adjust<0 and od.buy_orders:
            price = max(od.buy_orders)
            vol = min(-adjust, limit+current)
            if vol>0:
                orders[Product.VOLCANIC_ROCK] = [Order(Product.VOLCANIC_ROCK, price, -vol)]
        return orders

    def select_strategy(self, iv_orders: Dict[str,List[Order]], arb_orders: Dict[str,List[Order]]) -> Dict[str,List[Order]]:
        # Reset strategy if too much history
        total = len(self.last_profit_iv) + len(self.last_profit_arb)
        if total >= self.strategy_reset_window:
            self.chosen_strategy = None
            self.last_profit_iv = self.last_profit_iv[-5:]
            self.last_profit_arb = self.last_profit_arb[-5:]

        # Lock in once we have enough data
        if self.chosen_strategy is None:
            if len(self.last_profit_iv) >= self.profit_window and len(self.last_profit_arb) >= self.profit_window:
                avg_iv = sum(self.last_profit_iv[-self.profit_window:]) / self.profit_window
                avg_arb = sum(self.last_profit_arb[-self.profit_window:]) / self.profit_window
                self.chosen_strategy = "iv" if avg_iv >= avg_arb else "arb"

        if self.chosen_strategy == "iv":
            return iv_orders
        else:
            return arb_orders
        
    def norm_cdf(self, x: float) -> float:
        """Standard normal cumulative distribution function."""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def black_scholes_call(self, S: float, K: float, T_days: float, r: float, sigma: float) -> float:
        """Black-Scholes price of a European call option."""
        T = T_days / 365.0
        if T <= 0 or sigma <= 0:
            return max(S - K, 0.0)

        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return S * self.norm_cdf(d1) - K * math.exp(-r * T) * self.norm_cdf(d2)

    def implied_vol_call(self, market_price, S, K, T_days, r, tol=0.00000000000001, max_iter=250):
        """
        Calculate implied volatility from market call option price using bisection.

        Parameters:
        - market_price: observed market price of the option
        - S: spot price
        - K: strike price
        - T_days: time to maturity in days
        - r: risk-free interest rate
        - tol: convergence tolerance
        - max_iter: maximum number of iterations

        Returns:
        - Implied volatility (sigma)
        """
        # Set reasonable initial bounds
        sigma_low = 0.01
        sigma_high = 0.35

        for _ in range(max_iter):
            sigma_mid = (sigma_low + sigma_high) / 2
            price = self.black_scholes_call(S, K, T_days, r, sigma_mid)

            if abs(price - market_price) < tol:
                return sigma_mid

            if price > market_price:
                sigma_high = sigma_mid
            else:
                sigma_low = sigma_mid

        return (sigma_low + sigma_high) / 2  # Final estimate
    
    def call_delta(self, S: float, K: float, T: float, sigma: float) -> float:
        """
        Calculate the Black-Scholes delta of a European call option.

        Parameters:
        - S: Current stock price
        - K: Strike price
        - T: Time to maturity (in days)
        - r: Risk-free interest rate
        - sigma: Volatility (annual)

        Returns:
        - delta: Call option delta
        """
        r = 0
        T = T / 365
        if T == 0 or sigma == 0:
            return 1.0 if S > K else 0.0

        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        return 0.5 * (1 + math.erf(d1 / math.sqrt(2)))
        
    def trade_10000(self, state, market_data, traderObject):
        product = "VOLCANIC_ROCK_VOUCHER_10000"
        delta_sum = 0
        orders = {}
        for p in ["VOLCANIC_ROCK", "VOLCANIC_ROCK_VOUCHER_10000"]:
            orders[p] = []
        fair = market_data.fair[product]
        underlying_fair = market_data.fair["VOLCANIC_ROCK"]
        dte = DAYS_LEFT - state.timestamp / 1_000_000
        v_t = self.implied_vol_call(fair, underlying_fair, 10000, dte, 0)
        delta = self.call_delta(fair, underlying_fair, dte, v_t)
        m_t = np.log(10000 / underlying_fair) / np.sqrt(dte / 365)
        # print(f"m_t = {m_t}")

        # , ,
        base_coef = 0.14786181  # 0.13571776890273662 + 0.00229274*dte
        linear_coef = 0.00099561  # -0.03685812200491957 + 0.0072571*dte
        squared_coef = 0.23544086  # 0.16277746617792221 + 0.01096456*dte
        fair_iv = base_coef + linear_coef * m_t + squared_coef * (m_t ** 2)
        # print(fair_iv)

        diff = v_t - fair_iv
        if "prices_10000" not in traderObject:
            traderObject["prices_10000"] = [diff]
        else:
            traderObject["prices_10000"].append(diff)
        threshold = 0.0035
        # print(diff)
        if len(traderObject["prices_10000"]) > 20:
            diff -= np.mean(traderObject["prices_10000"])
            traderObject["prices_10000"].pop(0)
            if diff > threshold:  # short vol so sell option, buy und
                amount = market_data.sell_sum["VOLCANIC_ROCK_VOUCHER_10000"]
                amount = min(amount, sum(market_data.bid_volumes["VOLCANIC_ROCK_VOUCHER_10000"]))
                option_amount = amount
                rock_amount = amount

                # print(rock_amount)
                for i in range(0, len(market_data.ask_prices["VOLCANIC_ROCK"])):
                    fill = min(-market_data.ask_volumes["VOLCANIC_ROCK"][i], rock_amount)
                    #print(fill)
                    if fill != 0:
                        orders["VOLCANIC_ROCK"].append(Order("VOLCANIC_ROCK", market_data.ask_prices["VOLCANIC_ROCK"][i], fill))
                        market_data.buy_sum["VOLCANIC_ROCK"] -= fill
                        market_data.end_pos["VOLCANIC_ROCK"] += fill
                        rock_amount -= fill
                        #print(fill)

                for i in range(0, len(market_data.bid_prices["VOLCANIC_ROCK_VOUCHER_10000"])):
                    fill = min(market_data.bid_volumes["VOLCANIC_ROCK_VOUCHER_10000"][i], option_amount)
                    delta_sum -= delta * fill
                    if fill != 0:
                        orders["VOLCANIC_ROCK_VOUCHER_10000"].append(Order("VOLCANIC_ROCK_VOUCHER_10000",
                                                                           market_data.bid_prices[
                                                                               "VOLCANIC_ROCK_VOUCHER_10000"][i],
                                                                           -fill))
                        market_data.sell_sum["VOLCANIC_ROCK_VOUCHER_10000"] -= fill
                        market_data.end_pos["VOLCANIC_ROCK_VOUCHER_10000"] -= fill
                        option_amount -= fill

            elif diff < -threshold:  # long vol
                # print("LONG")
                # print("----")
                amount = market_data.buy_sum["VOLCANIC_ROCK_VOUCHER_10000"]
                # print(amount)
                amount = min(amount, -sum(market_data.ask_volumes["VOLCANIC_ROCK_VOUCHER_10000"]))
                # print(amount)
                option_amount = amount
                rock_amount = amount
                # print(f"{rock_amount} rocks")
                for i in range(0, len(market_data.ask_prices["VOLCANIC_ROCK_VOUCHER_10000"])):
                    fill = min(-market_data.ask_volumes["VOLCANIC_ROCK_VOUCHER_10000"][i], option_amount)
                    delta_sum += delta * fill
                    if fill != 0:
                        orders["VOLCANIC_ROCK_VOUCHER_10000"].append(Order("VOLCANIC_ROCK_VOUCHER_10000",
                                                                           market_data.ask_prices[
                                                                               "VOLCANIC_ROCK_VOUCHER_10000"][i], fill))
                        market_data.buy_sum["VOLCANIC_ROCK_VOUCHER_10000"] -= fill
                        market_data.end_pos["VOLCANIC_ROCK_VOUCHER_10000"] += fill
                        option_amount -= fill

                for i in range(0, len(market_data.bid_prices["VOLCANIC_ROCK"])):
                        fill = min(market_data.bid_volumes["VOLCANIC_ROCK"][i], rock_amount)
                        #print(fill)
                        if fill != 0:
                            orders["VOLCANIC_ROCK"].append(Order("VOLCANIC_ROCK", market_data.bid_prices["VOLCANIC_ROCK"][i], -fill))
                            market_data.sell_sum["VOLCANIC_ROCK"] -= fill
                            market_data.end_pos["VOLCANIC_ROCK"] -= fill
                            rock_amount -= fill

        return orders
    
    def trade_10500(self, state, market_data, traderObject):
        product = "VOLCANIC_ROCK_VOUCHER_10500"
        delta_sum = 0
        orders = {}
        for p in ["VOLCANIC_ROCK", "VOLCANIC_ROCK_VOUCHER_10500"]:
            orders[p] = []
        fair = market_data.fair[product]
        underlying_fair = market_data.fair["VOLCANIC_ROCK"]
        dte = DAYS_LEFT - state.timestamp / 1_000_000
        v_t = self.implied_vol_call(fair, underlying_fair, 10500, dte, 0)
        try:
            delta = self.call_delta(fair, underlying_fair, dte, v_t)
        except:
            return [], []
        m_t = np.log(10500 / underlying_fair) / np.sqrt(dte / 365)
        # print(f"m_t = {m_t}")

        # , ,
        base_coef = 0.264416  # 0.13571776890273662 + 0.00229274*dte
        linear_coef = 0.010031  # -0.03685812200491957 + 0.0072571*dte
        squared_coef = 0.147604  # 0.16277746617792221 + 0.01096456*dte
        fair_iv = base_coef + linear_coef * m_t + squared_coef * (m_t ** 2)
        # print(fair_iv)

        diff = v_t - fair_iv
        if "prices_10500" not in traderObject:
            traderObject["prices_10500"] = [diff]
        else:
            traderObject["prices_10500"].append(diff)
        # print(diff)
        if len(traderObject["prices_10500"]) > 13:
            diff -= np.mean(traderObject["prices_10500"])
            traderObject["prices_10500"].pop(0)
        threshold = 0.001
        if diff > threshold:  # short vol so sell option, buy und
            amount = min(market_data.buy_sum["VOLCANIC_ROCK"], market_data.sell_sum["VOLCANIC_ROCK_VOUCHER_10500"])
            amount = min(amount, -sum(market_data.ask_volumes["VOLCANIC_ROCK"]),
                         sum(market_data.bid_volumes["VOLCANIC_ROCK_VOUCHER_10500"]))
            option_amount = amount
            if np.mean(traderObject["prices_10500"]) > 0:
                rock_amount = amount
            else:
                rock_amount = amount // 2

            # print(rock_amount)
            for i in range(0, len(market_data.ask_prices["VOLCANIC_ROCK"])):
                fill = min(-market_data.ask_volumes["VOLCANIC_ROCK"][i], rock_amount)
                # print(fill)
                if fill != 0:
                    orders["VOLCANIC_ROCK"].append(
                        Order("VOLCANIC_ROCK", market_data.ask_prices["VOLCANIC_ROCK"][i], fill))
                    market_data.buy_sum["VOLCANIC_ROCK"] -= fill
                    market_data.end_pos["VOLCANIC_ROCK"] += fill
                    rock_amount -= fill
                    # print(fill)

            for i in range(0, len(market_data.bid_prices["VOLCANIC_ROCK_VOUCHER_10500"])):
                fill = min(market_data.bid_volumes["VOLCANIC_ROCK_VOUCHER_10500"][i], option_amount)
                delta_sum -= delta * fill
                if fill != 0:
                    orders["VOLCANIC_ROCK_VOUCHER_10500"].append(Order("VOLCANIC_ROCK_VOUCHER_10500",
                                                                       market_data.bid_prices[
                                                                           "VOLCANIC_ROCK_VOUCHER_10500"][i],
                                                                       -fill))
                    market_data.sell_sum["VOLCANIC_ROCK_VOUCHER_10500"] -= fill
                    market_data.end_pos["VOLCANIC_ROCK_VOUCHER_10500"] -= fill
                    option_amount -= fill

        elif diff < -threshold:  # long vol
            # print("LONG")
            # print("----")
            amount = min(market_data.buy_sum["VOLCANIC_ROCK_VOUCHER_10500"], market_data.sell_sum["VOLCANIC_ROCK"])
            # print(amount)
            amount = min(amount, -sum(market_data.ask_volumes["VOLCANIC_ROCK_VOUCHER_10500"]),
                         sum(market_data.bid_volumes["VOLCANIC_ROCK"]))
            # print(amount)
            option_amount = amount
            if np.mean(traderObject["prices_10500"]) < 0:
                rock_amount = amount
            else:
                rock_amount = amount // 2
                raise Exception(state.timestamp, option_amount)
            # print(f"{rock_amount} rocks")
            for i in range(0, len(market_data.ask_prices["VOLCANIC_ROCK_VOUCHER_10500"])):
                fill = min(-market_data.ask_volumes["VOLCANIC_ROCK_VOUCHER_10500"][i], option_amount)
                delta_sum += delta * fill
                if fill != 0:
                    orders["VOLCANIC_ROCK_VOUCHER_10500"].append(Order("VOLCANIC_ROCK_VOUCHER_10500",
                                                                       market_data.ask_prices[
                                                                           "VOLCANIC_ROCK_VOUCHER_10500"][i], fill))
                    market_data.buy_sum["VOLCANIC_ROCK_VOUCHER_10500"] -= fill
                    market_data.end_pos["VOLCANIC_ROCK_VOUCHER_10500"] += fill
                    option_amount -= fill

            for i in range(0, len(market_data.bid_prices["VOLCANIC_ROCK"])):
                fill = min(market_data.bid_volumes["VOLCANIC_ROCK"][i], rock_amount)
                # print(fill)
                if fill != 0:
                    orders["VOLCANIC_ROCK"].append(
                        Order("VOLCANIC_ROCK", market_data.bid_prices["VOLCANIC_ROCK"][i], -fill))
                    market_data.sell_sum["VOLCANIC_ROCK"] -= fill
                    market_data.end_pos["VOLCANIC_ROCK"] -= fill
                    rock_amount -= fill

        return orders["VOLCANIC_ROCK_VOUCHER_10500"]

    def trade_9500(self, state, market_data, traderObject):
        product = "VOLCANIC_ROCK_VOUCHER_9500"
        delta_sum = 0
        orders = {}
        for p in ["VOLCANIC_ROCK", "VOLCANIC_ROCK_VOUCHER_9500"]:
            orders[p] = []
        fair = market_data.fair[product]
        underlying_fair = market_data.fair["VOLCANIC_ROCK"]
        dte = DAYS_LEFT - state.timestamp / 1_000_000
        v_t = self.implied_vol_call(fair, underlying_fair, 9500, dte, 0)
        try:
            delta = self.call_delta(fair, underlying_fair, dte, v_t)
        except:
            return [], []
        m_t = np.log(9500 / underlying_fair) / np.sqrt(dte / 365)
        # print(f"m_t = {m_t}")

        # , ,
        base_coef = 0.264416  # 0.13571776890273662 + 0.00229274*dte
        linear_coef = 0.010031  # -0.03685812200491957 + 0.0072571*dte
        squared_coef = 0.147604  # 0.16277746617792221 + 0.01096456*dte
        fair_iv = base_coef + linear_coef * m_t + squared_coef * (m_t ** 2)
        # print(fair_iv)

        diff = v_t - fair_iv
        if "prices_9500" not in traderObject:
            traderObject["prices_9500"] = [diff]
        else:
            traderObject["prices_9500"].append(diff)
        # print(diff)
        if len(traderObject["prices_9500"]) > 13:
            diff -= np.mean(traderObject["prices_9500"])
            traderObject["prices_9500"].pop(0)
        threshold = 0.0005
        if diff > threshold:  # short vol so sell option, buy und
            amount = min(market_data.buy_sum["VOLCANIC_ROCK"], market_data.sell_sum["VOLCANIC_ROCK_VOUCHER_9500"])
            amount = min(amount, -sum(market_data.ask_volumes["VOLCANIC_ROCK"]),
                         sum(market_data.bid_volumes["VOLCANIC_ROCK_VOUCHER_9500"]))
            option_amount = amount
            rock_amount = amount
            # print(rock_amount)
            for i in range(0, len(market_data.ask_prices["VOLCANIC_ROCK"])):
                fill = min(-market_data.ask_volumes["VOLCANIC_ROCK"][i], rock_amount)
                # print(fill)
                if fill != 0:
                    orders["VOLCANIC_ROCK"].append(
                        Order("VOLCANIC_ROCK", market_data.ask_prices["VOLCANIC_ROCK"][i], fill))
                    market_data.buy_sum["VOLCANIC_ROCK"] -= fill
                    market_data.end_pos["VOLCANIC_ROCK"] += fill
                    rock_amount -= fill
                    # print(fill)

            for i in range(0, len(market_data.bid_prices["VOLCANIC_ROCK_VOUCHER_9500"])):
                fill = min(market_data.bid_volumes["VOLCANIC_ROCK_VOUCHER_9500"][i], option_amount)
                delta_sum -= delta * fill
                if fill != 0:
                    orders["VOLCANIC_ROCK_VOUCHER_9500"].append(Order("VOLCANIC_ROCK_VOUCHER_9500",
                                                                      market_data.bid_prices[
                                                                          "VOLCANIC_ROCK_VOUCHER_9500"][i],
                                                                      -fill))
                    market_data.sell_sum["VOLCANIC_ROCK_VOUCHER_9500"] -= fill
                    market_data.end_pos["VOLCANIC_ROCK_VOUCHER_9500"] -= fill
                    option_amount -= fill

        elif diff < -threshold:  # long vol
            # print("LONG")
            # print("----")
            amount = min(market_data.buy_sum["VOLCANIC_ROCK_VOUCHER_9500"], market_data.sell_sum["VOLCANIC_ROCK"])
            # print(amount)
            amount = min(amount, -sum(market_data.ask_volumes["VOLCANIC_ROCK_VOUCHER_9500"]),
                         sum(market_data.bid_volumes["VOLCANIC_ROCK"]))
            # print(amount)
            option_amount = amount
            rock_amount = amount

            # print(f"{rock_amount} rocks")
            for i in range(0, len(market_data.ask_prices["VOLCANIC_ROCK_VOUCHER_9500"])):
                fill = min(-market_data.ask_volumes["VOLCANIC_ROCK_VOUCHER_9500"][i], option_amount)
                delta_sum += delta * fill
                if fill != 0:
                    orders["VOLCANIC_ROCK_VOUCHER_9500"].append(Order("VOLCANIC_ROCK_VOUCHER_9500",
                                                                      market_data.ask_prices[
                                                                          "VOLCANIC_ROCK_VOUCHER_9500"][i], fill))
                    market_data.buy_sum["VOLCANIC_ROCK_VOUCHER_9500"] -= fill
                    market_data.end_pos["VOLCANIC_ROCK_VOUCHER_9500"] += fill
                    option_amount -= fill

            for i in range(0, len(market_data.bid_prices["VOLCANIC_ROCK"])):
                fill = min(market_data.bid_volumes["VOLCANIC_ROCK"][i], rock_amount)
                # print(fill)
                if fill != 0:
                    orders["VOLCANIC_ROCK"].append(
                        Order("VOLCANIC_ROCK", market_data.bid_prices["VOLCANIC_ROCK"][i], -fill))
                    market_data.sell_sum["VOLCANIC_ROCK"] -= fill
                    market_data.end_pos["VOLCANIC_ROCK"] -= fill
                    rock_amount -= fill

        return orders["VOLCANIC_ROCK_VOUCHER_9500"]

    def trade_9750(self, state, market_data, traderObject):
        product = "VOLCANIC_ROCK_VOUCHER_9750"
        orders = {}
        for p in ["VOLCANIC_ROCK", "VOLCANIC_ROCK_VOUCHER_9750"]:
            orders[p] = []
        fair = market_data.fair[product]
        underlying_fair = market_data.fair["VOLCANIC_ROCK"]
        dte = DAYS_LEFT - state.timestamp / 1_000_000

        v_t = self.implied_vol_call(fair, underlying_fair, 9750, dte, 0)
        m_t = np.log(9750 / underlying_fair) / np.sqrt(dte / 365)
        # print(f"m_t = {m_t}")

        # , ,
        base_coef = 0.264416  # 0.13571776890273662 + 0.00229274*dte
        linear_coef = 0.010031  # -0.03685812200491957 + 0.0072571*dte
        squared_coef = 0.147604  # 0.16277746617792221 + 0.01096456*dte
        fair_iv = base_coef + linear_coef * m_t + squared_coef * (m_t ** 2)
        # print(fair_iv)
        diff = v_t - fair_iv
        if "prices_9750" not in traderObject:
            traderObject["prices_9750"] = [diff]
        else:

            traderObject["prices_9750"].append(diff)
        threshold = 0.0055
        # print(diff)
        if len(traderObject["prices_9750"]) > 13:
            diff -= np.mean(traderObject["prices_9750"])
            traderObject["prices_9750"].pop(0)
        if diff > threshold:  # short vol so sell option, buy und
            amount = market_data.sell_sum["VOLCANIC_ROCK_VOUCHER_9750"]
            amount = min(amount, sum(market_data.bid_volumes["VOLCANIC_ROCK_VOUCHER_9750"]))
            option_amount = amount

            rock_amount = amount

            for i in range(0, len(market_data.bid_prices["VOLCANIC_ROCK_VOUCHER_9750"])):
                fill = min(market_data.bid_volumes["VOLCANIC_ROCK_VOUCHER_9750"][i], option_amount)
                if fill != 0:
                    orders["VOLCANIC_ROCK_VOUCHER_9750"].append(Order("VOLCANIC_ROCK_VOUCHER_9750",
                                                                      market_data.bid_prices[
                                                                          "VOLCANIC_ROCK_VOUCHER_9750"][i],
                                                                      -fill))
                    market_data.sell_sum["VOLCANIC_ROCK_VOUCHER_9750"] -= fill
                    market_data.end_pos["VOLCANIC_ROCK_VOUCHER_9750"] -= fill
                    option_amount -= fill

        elif diff < -threshold:  # long vol
            # print("LONG")
            # print("----")
            amount = market_data.buy_sum["VOLCANIC_ROCK_VOUCHER_9750"]
            # print(amount)
            amount = min(amount, -sum(market_data.ask_volumes["VOLCANIC_ROCK_VOUCHER_9750"]))
            # print(amount)
            option_amount = amount
            rock_amount = amount
            # print(f"{rock_amount} rocks")
            for i in range(0, len(market_data.ask_prices["VOLCANIC_ROCK_VOUCHER_9750"])):
                fill = min(-market_data.ask_volumes["VOLCANIC_ROCK_VOUCHER_9750"][i], option_amount)
                if fill != 0:
                    orders["VOLCANIC_ROCK_VOUCHER_9750"].append(Order("VOLCANIC_ROCK_VOUCHER_9750",
                                                                      market_data.ask_prices[
                                                                          "VOLCANIC_ROCK_VOUCHER_9750"][i], fill))
                    market_data.buy_sum["VOLCANIC_ROCK_VOUCHER_9750"] -= fill
                    market_data.end_pos["VOLCANIC_ROCK_VOUCHER_9750"] += fill
                    option_amount -= fill

            """for i in range(0, len(market_data.bid_prices["VOLCANIC_ROCK"])):
                    fill = min(market_data.bid_volumes["VOLCANIC_ROCK"][i], rock_amount)
                    #print(fill)
                    if fill != 0:
                        orders["VOLCANIC_ROCK"].append(Order("VOLCANIC_ROCK", market_data.bid_prices["VOLCANIC_ROCK"][i], -fill))
                        market_data.sell_sum["VOLCANIC_ROCK"] -= fill
                        market_data.end_pos["VOLCANIC_ROCK"] -= fill
                        rock_amount -= fill"""
        return orders["VOLCANIC_ROCK_VOUCHER_9750"]
    
    def get_djembe_mid_price(self, order_depth: OrderDepth) -> Optional[float]:
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        best_bid = max(order_depth.buy_orders)
        best_ask = min(order_depth.sell_orders)
        return 0.5 * (best_bid + best_ask)

    def update_djembe_price_history(self, mid_price: float):
        self.price_history["DJEMBES"].append(mid_price)
        max_len = max(self.rsi_period, self.atr_period) + 1
        if len(self.price_history["DJEMBES"]) > max_len:
            self.price_history["DJEMBES"].pop(0)

    def compute_djembe_RSI(self, prices: List[float]) -> float:
        if len(prices) < self.rsi_period + 1:
            return 50.0
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = gains[-self.rsi_period :].mean()
        avg_loss = losses[-self.rsi_period :].mean()
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - 100 / (1 + rs)

    def compute_djembe_ATR(self, prices: List[float]) -> float:
        if len(prices) < self.atr_period + 1:
            return 0.0
        tr = np.abs(np.diff(prices))
        return float(tr[-self.atr_period :].mean())

    def _handle_djembe_mirror_paris(self, od: OrderDepth, state: TradingState) -> Optional[Dict[str, List[Order]]]:
        net_qty = 0
        for t in state.market_trades.get("DJEMBES", []):
            if t.timestamp != state.timestamp - 100:
                continue
            if getattr(t, "buyer", None) == "Paris":
                net_qty -= t.quantity
            elif getattr(t, "seller", None) == "Paris":
                net_qty += t.quantity

        if net_qty == 0:
            return None

        if not od.sell_orders or not od.buy_orders:
            return {}
        best_ask = min(od.sell_orders)
        best_bid = max(od.buy_orders)

        pos = state.position.get("DJEMBES", 0)
        # Mirror position limit is 60
        if net_qty > 0:
            volume = min(net_qty, 60 - pos)
        else:
            volume = max(net_qty, -60 - pos)

        if volume == 0:
            return {}

        price = best_ask if volume > 0 else best_bid
        return {"DJEMBES": [Order("DJEMBES", price, volume)]}

    def _handle_djembe_rsi_quoting(self, od: OrderDepth, state: TradingState) -> Dict[str, List[Order]]:
        mid = self.get_djembe_mid_price(od)
        if mid is None:
            return { "DJEMBES": []}

        self.update_djembe_price_history(mid)
        prices = self.price_history["DJEMBES"]
        rsi = self.compute_djembe_RSI(prices)
        atr = self.compute_djembe_ATR(prices)
        atr_med = float(
            np.median(prices[-self.atr_period :]) if len(prices) >= self.atr_period else atr
        )

        signal = 0
        if rsi < 28 and atr > atr_med:
            signal = 1
        elif rsi > 72 and atr > atr_med:
            signal = -1

        if signal == 1:
            bid_price, ask_price = int(mid - 1), int(mid + 2)
        elif signal == -1:
            bid_price, ask_price = int(mid - 2), int(mid + 1)
        else:
            bid_price, ask_price = int(mid - 1), int(mid + 1)

        pos = state.position.get("DJEMBES", 0)
        # Tech position limit is 60
        bid_vol = 60 - pos
        ask_vol = -60 - pos

        return {
            "DJEMBES": [
                Order("DJEMBES", bid_price, bid_vol),
                Order("DJEMBES", ask_price, ask_vol),
            ]
        }

    def _generate_djembe_orders(self, od: OrderDepth, state: TradingState) -> Dict[str, List[Order]]:
        mirror = self._handle_djembe_mirror_paris(od, state)
        if mirror is not None:
            return mirror
        return self._handle_djembe_rsi_quoting(od, state)
    
    def get_croissants_mid_price(self, order_depth: OrderDepth) -> Optional[float]:
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        best_bid = max(order_depth.buy_orders)
        best_ask = min(order_depth.sell_orders)
        return 0.5 * (best_bid + best_ask)

    def update_croissants_price_history(self, mid_price: float):
        self.price_history["CROISSANTS"].append(mid_price)
        max_len = max(self.rsi_period, self.atr_period) + 1
        if len(self.price_history["CROISSANTS"]) > max_len:
            self.price_history["CROISSANTS"].pop(0)

    def compute_croissants_RSI(self, prices: List[float]) -> float:
        if len(prices) < self.rsi_period + 1:
            return 50.0
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = gains[-self.rsi_period :].mean()
        avg_loss = losses[-self.rsi_period :].mean()
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - 100 / (1 + rs)

    def compute_croissants_ATR(self, prices: List[float]) -> float:
        if len(prices) < self.atr_period + 1:
            return 0.0
        tr = np.abs(np.diff(prices))
        return float(tr[-self.atr_period :].mean())

    def _handle_croissants_mirror_paris(self, od: OrderDepth, state: TradingState) -> Optional[Dict[str, List[Order]]]:
        net_qty = 0
        hard_limit = 250
        for t in state.market_trades.get("CROISSANTS", []):
            if t.timestamp >= state.timestamp - 300:
                continue
            if getattr(t, "buyer", None) == "Olivia":
                net_qty += hard_limit
            elif getattr(t, "seller", None) == "Olivia":
                net_qty -= hard_limit

        if net_qty == 0:
            return None

        if not od.sell_orders or not od.buy_orders:
            return {}
        best_ask = min(od.sell_orders)
        best_bid = max(od.buy_orders)

        pos = state.position.get("CROISSANTS", 0)
        # Mirror position limit is 60
        
        if net_qty > 0:
            volume = min(net_qty, hard_limit - pos)
        else:
            volume = max(net_qty, -hard_limit - pos)

        if volume == 0:
            return {}

        price = best_ask if volume > 0 else best_bid
        return {"CROISSANTS": [Order("CROISSANTS", price, volume)]}

    def _handle_croissants_rsi_quoting(self, od: OrderDepth, state: TradingState) -> Dict[str, List[Order]]:
        mid = self.get_croissants_mid_price(od)
        if mid is None:
            return { "CROISSANTS": []}

        self.update_croissants_price_history(mid)
        prices = self.price_history["CROISSANTS"]
        rsi = self.compute_croissants_RSI(prices)
        atr = self.compute_croissants_ATR(prices)
        atr_med = float(
            np.median(prices[-self.atr_period :]) if len(prices) >= self.atr_period else atr
        )

        signal = 0
        if rsi < 10:
            signal = 1
        elif rsi > 90 :
            signal = -1

        if signal == 1:
            bid_price, ask_price = int(mid - 1), int(mid + 2)
        elif signal == -1:
            bid_price, ask_price = int(mid - 2), int(mid + 1)
        else:
            bid_price, ask_price = int(mid - 1), int(mid + 1)

        pos = state.position.get("CROISSANTS", 0)
        # Tech position limit is 60
        soft_limit = 50
        bid_vol = soft_limit - pos
        ask_vol = -soft_limit - pos

        return {
            "CROISSANTS": [
                Order("CROISSANTS", bid_price, bid_vol),
                Order("CROISSANTS", ask_price, ask_vol),
            ]
        }

    def _generate_croissants_orders(self, od: OrderDepth, state: TradingState) -> Dict[str, List[Order]]:
        mirror = self._handle_croissants_mirror_paris(od, state)
        if mirror is not None:
            return mirror
        return self._handle_croissants_rsi_quoting(od, state)
    
    def get_mid_price_Basket_2(self, od: OrderDepth) -> Optional[float]:
        if not od.buy_orders or not od.sell_orders:
            return None
        best_bid = max(od.buy_orders.keys())
        best_ask = min(od.sell_orders.keys())
        return 0.5 * (best_bid + best_ask)

    def update_price_history_Basket_2(self, product: str, mid: float):
        history = self.price_history[product]
        history.append(mid)
        max_len = max(self.rsi_period, self.atr_period) + 1
        if len(history) > max_len:
            history.pop(0)

    def compute_RSI_Basket_2(self, prices: List[float], period: int) -> float:
        if len(prices) < period + 1:
            return 50.0
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def compute_ATR_Basket_2(self, prices: List[float], period: int) -> float:
        if len(prices) < period + 1:
            return 0.0
        tr = np.abs(np.diff(prices))
        return float(np.mean(tr[-period:]))

    def compute_orders_Basket_2(
        self,
        state: TradingState,
        product: str = "PICNIC_BASKET2"
    ) -> Dict[str, List[Order]]:
        orders: Dict[str, List[Order]] = {}
        od: OrderDepth = state.order_depths.get(product)

        if od is None:
            return orders

        # 1. Compute mid-price and update history
        mid = self.get_mid_price_Basket_2(od)
        if mid is not None:
            self.update_price_history_Basket_2(product, mid)

        # 2. Mirror "Charlie" trades if found
        net_qty = 0
        BOT_NAME = "Charlie"
        for t in state.market_trades.get(product, []):
            # only consider the last matching timestamp
            if t.timestamp != state.timestamp - 100:
                continue
            if getattr(t, 'buyer', None) == BOT_NAME:
                net_qty += int(t.quantity)
            elif getattr(t, 'seller', None) == BOT_NAME:
                net_qty -= int(t.quantity)

        if net_qty != 0 and od.sell_orders and od.buy_orders:
            best_ask = min(od.sell_orders.keys())
            best_bid = max(od.buy_orders.keys())
            pos = state.position.get(product, 0)
            limit = 100
            if net_qty > 0:
                volume = min(net_qty, limit - pos)
            else:
                volume = max(net_qty, -limit - pos)
            if volume != 0:
                price = best_ask if volume > 0 else best_bid
                orders[product] = [Order(product, price, volume)]
            return orders

        # 3. Otherwise normal RSI/ATR market trading
        prices = self.price_history[product]
        rsi = self.compute_RSI_Basket_2(prices, self.rsi_period)
        atr = self.compute_ATR_Basket_2(prices, self.atr_period)
        atr_med = (
            np.median(prices[-self.atr_period:])
            if len(prices) >= self.atr_period
            else atr
        )

        # Determine signal
        signal = 0
        if rsi < 28:
            signal = 1
        elif rsi > 72:
            signal = -1

        # Price levels based on signal
        if mid is not None:
            if signal in (1, -1):
                bid_price = int(mid - 3)
                ask_price = int(mid + 3)
            else:
                bid_price = int(mid - 2)
                ask_price = int(mid + 2)

            pos = state.position.get(product, 0)
            limit = 100
            bid_vol = limit - pos
            ask_vol = -limit - pos
            orders[product] = [
                Order(product, bid_price, bid_vol),
                Order(product, ask_price, ask_vol)
            ]

        return orders

    # ============================= Combined Run Method =============================
    def run(self, state: TradingState) -> Tuple[Dict[str,List[Order]], int, str]:
        result: Dict[str,List[Order]] = {}
        trader_obj: Dict[str,Any] = {}
        if state.traderData not in (None, ""):
            trader_obj = jsonpickle.decode(state.traderData)
            
        market_data = MarketData()
        products = ["VOLCANIC_ROCK_VOUCHER_10000", "VOLCANIC_ROCK","VOLCANIC_ROCK_VOUCHER_9500",
                    "VOLCANIC_ROCK_VOUCHER_9750","VOLCANIC_ROCK_VOUCHER_10250",
                    "VOLCANIC_ROCK_VOUCHER_10500"]
        for product in products:
            position = state.position.get(product, 0)
            order_depth = state.order_depths[product]
            bids, asks = order_depth.buy_orders, order_depth.sell_orders
            if order_depth.buy_orders:
                mm_bid = max(bids.items(), key=lambda tup: tup[1])[0]
            if order_depth.sell_orders:
                mm_ask = min(asks.items(), key=lambda tup: tup[1])[0]
            if order_depth.sell_orders and order_depth.buy_orders:
                fair_price = (mm_ask + mm_bid) / 2
            elif order_depth.sell_orders:
                fair_price = mm_ask
            elif order_depth.buy_orders:
                fair_price = mm_bid
            else:
                fair_price = trader_obj[f"prev_fair_{product}"]
            trader_obj[f"prev_fair_{product}"] = fair_price
            
            market_data.end_pos[product] = position
            market_data.buy_sum[product] = self.PRODUCT_LIMIT[product] - position
            market_data.sell_sum[product] = self.PRODUCT_LIMIT[product] + position
            market_data.bid_prices[product] = list(bids.keys())
            market_data.bid_volumes[product] = list(bids.values())
            market_data.ask_prices[product] = list(asks.keys())
            market_data.ask_volumes[product] = list(asks.values())
            market_data.fair[product] = fair_price
            
            
            
        mac = trader_obj.setdefault("MACARONS", {
        "curr_edge": self.params["MACARONS"]["init_make_edge"],
        "volume_history": [],
        "optimized": False,
        "sunlight": {"below_count": 0, "critical": False}
        })

        # Restore partial state
        if "kelp_last_price" in trader_obj:
            self.kelp_last_price = trader_obj["kelp_last_price"]
        if "resin_mid_prices" in trader_obj:
            self.resin_mid_prices = trader_obj["resin_mid_prices"]
        if "mispricing_history" in trader_obj:
            self.mispricing_history = trader_obj["mispricing_history"]
        if "depth_history" in trader_obj:
            self.depth_history = trader_obj["depth_history"]
        if "SQUID_INK_last_price" in trader_obj:
            self.SQUID_INK_last_price = trader_obj["SQUID_INK_last_price"]
        if "volatility" in trader_obj:
            self.volatility = trader_obj["volatility"]
        if "prev_price" in trader_obj:
            self.prev_price = trader_obj["prev_price"]
        if "iv_history" in trader_obj:
            self.iv_history = trader_obj["iv_history"]
        if "days_left" in trader_obj:
            self.days_left = trader_obj["days_left"]
        if "entry_prices" in trader_obj:
            self.entry_prices = trader_obj["entry_prices"]
        if "unrealized_pnl" in trader_obj:
            self.unrealized_pnl = trader_obj["unrealized_pnl"]
        if "high_water_mark" in trader_obj:
            self.high_water_mark = trader_obj["high_water_mark"]
        if "equity" in trader_obj:
            self.equity = trader_obj["equity"]
        if "last_profit_iv" in trader_obj:
            self.last_profit_iv = trader_obj["last_profit_iv"]
        if "last_profit_arb" in trader_obj:
            self.last_profit_arb = trader_obj["last_profit_arb"]
        if "chosen_strategy" in trader_obj:
            self.chosen_strategy = trader_obj["chosen_strategy"]
        if "trade_book" in trader_obj:
            self.trade_book = trader_obj["trade_book"]
        if "realized_profit" in trader_obj:
            self.realized_profit = trader_obj["realized_profit"]
        if "price_history" in trader_obj:
            self.price_history = trader_obj['price_history']
            
        od_dj = state.order_depths.get("DJEMBES")
        if od_dj is not None:
            ord_dj = self._generate_djembe_orders(od_dj, state)
            result_1 = ord_dj
            if "DJEMBES" in result_1:
                result["DJEMBES"]=result_1["DJEMBES"]
            
        result_2 = {}
        od_cr = state.order_depths.get("CROISSANTS")
        if od_cr is not None:
            ord_cr = self._generate_croissants_orders(od_cr, state)
            result_2 = ord_cr
            if "CROISSANTS" in result_2:
                result["CROISSANTS"] = result_2["CROISSANTS"]
        
        # result_3 = {}
        # od_bk2 = state.order_depths.get("CROISSANTS")
        # if od_bk2 is not None:
        #     ord_bk2 = self.compute_orders_Basket_2(state)
        #     result_3 = ord_bk2
        #     if "PICNIC_BASKET2" in result_3:
        #         result["PICNIC_BASKET2"] = result_3["PICNIC_BASKET2"]
    
        
        # ---- RAINFOREST_RESIN ----
        if "RAINFOREST_RESIN" in state.order_depths:
            od = state.order_depths["RAINFOREST_RESIN"]
            pos = state.position.get("RAINFOREST_RESIN", 0)
            fair = self.resin_calculate_fair_value(od)
            resin_orders = []
            if fair is not None:
                buy_v, sell_v = self.resin_take_best_orders("RAINFOREST_RESIN", fair, resin_orders, od, pos)
                make = self.resin_make_orders("RAINFOREST_RESIN", od, fair, pos, buy_v, sell_v)
                resin_orders.extend(make)
                result["RAINFOREST_RESIN"]=resin_orders

        # ---- KELP ----
        if "KELP" in state.order_depths:
            od = state.order_depths["KELP"]
            pos = state.position.get("KELP", 0)
            fair = self.kelp_fair_value(od)
            if fair is not None:
                take, b_v, s_v = self.kelp_take_orders("KELP", od, fair, pos)
                clear, b_v, s_v = self.kelp_clear_orders("KELP", od, fair, pos, b_v, s_v)
                make, _, _ = self.kelp_make_orders("KELP", od, fair, pos, b_v, s_v)
                result["KELP"] = take + clear + make

        # ---- Basket / Hedge ----
        current_prices: Dict[str,float] = {}
        for prod, od in state.order_depths.items():
            mid = self.get_mid_price(od)
            if mid is not None:
                current_prices[prod] = mid

        # update depth history
        for idx, prod in enumerate(["PICNIC_BASKET1","PICNIC_BASKET2"], start=1):
            od = state.order_depths.get(prod)
            if od and od.buy_orders and od.sell_orders:
                bid_sz = od.buy_orders[max(od.buy_orders)]
                ask_sz = od.sell_orders[min(od.sell_orders)]
                avg = (bid_sz + abs(ask_sz))/2
                buf = self.depth_history[f"BASKET{idx}"]
                buf.append(avg)
                if len(buf)>self.vol_window:
                    buf.pop(0)

        mis = self.calculate_mispricing(current_prices)
        if mis:
            basket_orders = self.generate_basket_orders(mis, state)
            for p, lst in basket_orders.items():
                result.setdefault(p, []).extend(lst)

        # ---- SQUID_INK ----
        if Product.SQUID_INK in state.order_depths:
            od = state.order_depths[Product.SQUID_INK]
            pos = state.position.get(Product.SQUID_INK, 0)
            fair = self.SQUID_INK_fair_value(od, trader_obj)
            take, b_v, s_v = self.take_orders_SQUID_INK(od, fair, pos)
            clear, b_v, s_v = self.clear_orders_SQUID_INK(od, fair, pos, b_v, s_v)
            make, _, _ = self.make_SQUID_INK_orders(od, fair, pos, b_v, s_v)
            result[Product.SQUID_INK] = take + clear + make

        # ---- Volatility / Options on VOLCANIC_ROCK ----
        od_rock = state.order_depths.get(Product.VOLCANIC_ROCK)
        if od_rock:
            S = self.get_mid_price(od_rock)
            if S is not None:
                self.update_ewma_vol(S)
                T = self.time_to_expiry()
                theo, delta = self.theoretical(S, T)
                drawdown = self.pnl_and_drawdown_check(state)
                if drawdown <= self.max_daily_drawdown:
                    iv_orders = self.generate_iv_orders(state)
                    arb_orders = self.generate_arbitrage_orders(theo, state)
                    chosen = self.select_strategy(iv_orders, arb_orders)
                    sl = self.generate_stop_loss_orders(state)
                    hedge = self.generate_delta_hedge(delta, state)
                    for grp in (chosen, sl, hedge):
                        for p, lst in grp.items():
                            result.setdefault(p, []).extend(lst)
                            
        vc_orders = self.trade_10000(state,market_data,trader_obj)
        result["VOLCANIC_ROCK_VOUCHER_10000"] = vc_orders["VOLCANIC_ROCK_VOUCHER_10000"]
        result["VOLCANIC_ROCK_VOUCHER_10500"] = []
        result["VOLCANIC_ROCK_VOUCHER_9750"] = self.trade_9750(state,market_data,trader_obj)
        result["VOLCANIC_ROCK_VOUCHER_9500"] = []
        result["VOLCANIC_ROCK_VOUCHER_10250"] = []
        result["VOLCANIC_ROCK"] = []
        
        
        # get current sunlight index
        sunlight = state.observations.plainValueObservations.get("SUNLIGHT", None)
        
        regular_order_check = 1
        
        if sunlight is not None:
            below = mac["sunlight"]["below_count"]
            crit = mac["sunlight"]["critical"]
            duration = self.params["MACARONS"]["sunlight_duration"]

            if sunlight < self.sunlight_critical:
                mac["sunlight"]["below_count"] = below + 1
                if mac["sunlight"]["below_count"] >= duration:
                    mac["sunlight"]["critical"] = True
            else:
                # bounced above threshold
                if crit:
                    regular_order_check = 0
                    # aggressive sell all at market
                    depth = state.order_depths.get("MAGNIFICENT_MACARONS")
                    position = state.position.get("MAGNIFICENT_MACARONS", 0)
                    if depth and position > 0:
                        sell_orders: List[Order] = []
                        cap = self.LIMIT["MAGNIFICENT_MACARONS"] + position
                        sold = 0
                        for price in sorted(depth.buy_orders.keys(), reverse=True):
                            if sold >= cap:
                                break
                            qty = min(abs(depth.buy_orders[price]), cap - sold)
                            if qty > 0:
                                sell_orders.append(Order("MAGNIFICENT_MACARONS", round_price(price), -qty))
                                sold += qty
                        result["MAGNIFICENT_MACARONS"] = sell_orders
                    # reset sunlight
                    mac["sunlight"] = {"below_count": 0, "critical": False}
                # reset if above without critical
                mac["sunlight"] = {"below_count": 0, "critical": False}

        # regular arbitrage flow
        if "MAGNIFICENT_MACARONS" in state.order_depths and regular_order_check == 1:
            depth = state.order_depths["MAGNIFICENT_MACARONS"]
            position = state.position.get("MAGNIFICENT_MACARONS", 0)
            obs_conv = state.observations.conversionObservations.get("MAGNIFICENT_MACARONS")

            # 1) Clear excess long
            conversions = self.arb_clear(position)

            # 2) Adapt edge
            edge = self.adapt_edge(
                state.timestamp,
                mac["curr_edge"],
                position,
                mac
            )
            mac["curr_edge"] = edge

            # 3) Take opportunities
            take_orders, bv, sv = self.arb_take(depth, obs_conv, edge, position)
            # 4) Make new quotes
            make_orders, _, _ = self.arb_make(depth, obs_conv, position, edge, bv, sv)

            result["MAGNIFICENT_MACARONS"] = take_orders + make_orders
            result["MAGNIFICENT_MACARONS"] = []
            
            
        product = "JAMS"
        od = state.order_depths.get(product)
        jam_order = 1
        if od is None:
            jam_order = 0

        mid = self.get_mid_price(od)
        if mid is None:
            jam_order = 0

        # Update price history and compute stochastic oscillator
        if jam_order == 1:
            self.update_price_history(product, mid)
            prices = self.price_history[product]
            stoch_val = self.compute_stochastic(prices, period=14)

            # Determine trading signal
            signal = 0
            if stoch_val < 10:
                signal = 1  # Buy
            elif stoch_val > 90:
                signal = -1  # Sell

            # Set bid/ask offsets based on signal
            if signal == 1:
                bid_price = int(mid - 1)
                ask_price = int(mid + 2)
            elif signal == -1:
                bid_price = int(mid - 2)
                ask_price = int(mid + 1)
            else:
                bid_price = int(mid - 1)
                ask_price = int(mid + 1)

            # Position limits enforcement
            pos = state.position.get(product, 0)
            limit = 35
            bid_volume = limit - pos
            ask_volume = -limit - pos

            result[product] = [
                Order(product, bid_price, bid_volume),
                Order(product, ask_price, ask_volume),
            ]        
            
        
        minimal_state = {
            "kelp_last_price": self.kelp_last_price,
            "resin_mid_prices": self.resin_mid_prices,
            "mispricing_history": self.mispricing_history,
            "depth_history": self.depth_history,
            "SQUID_INK_last_price": trader_obj.get("SQUID_INK_last_price"),
            "volatility": self.volatility,
            "prev_price": self.prev_price,
            "iv_history": self.iv_history,
            "days_left": self.days_left,
            "entry_prices": self.entry_prices,
            "unrealized_pnl": self.unrealized_pnl,
            "high_water_mark": self.high_water_mark,
            "equity": self.equity,
            "last_profit_iv": self.last_profit_iv,
            "last_profit_arb": self.last_profit_arb,
            "chosen_strategy": self.chosen_strategy,
            "trade_book": self.trade_book,
            "realized_profit": self.realized_profit,
            "price_history": self.price_history
        }
        trader_obj.update(minimal_state)
        traderDataEncoded = jsonpickle.encode(trader_obj)

        return result, 0, traderDataEncoded
