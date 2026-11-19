"""Real Kalshi orders for the trader's `--live` mode. Production account, real money.

Uses the V2 order endpoints (`/portfolio/events/orders`), which quote every
order from the YES side: buying YES at p is a `bid` at p, buying NO at p is an
`ask` at 1-p. Only buys are ever sent; positions are held to settlement.
"""
import uuid

import httpx

import config
from integration.kalshi_client import KalshiClient


class BrokerError(RuntimeError):
    pass


class Broker:
    def __init__(self, contracts, max_exposure, max_loss, rules=("tennis_dip", "cbb_dip", "wcbb_dip", "tennis_dip_jit", "cbb_dip_jit")):
        self.contracts, self.max_exposure, self.max_loss, self.rules = contracts, max_exposure, max_loss, tuple(rules)
        if not config.KALSHI_API_KEY or not config.KALSHI_PRIVATE_KEY_PATH:
            raise BrokerError("Set KALSHI_API_KEY and KALSHI_PRIVATE_KEY_PATH in .env for live orders.")
        self.signer = KalshiClient(config.KALSHI_API_KEY, config.KALSHI_PRIVATE_KEY_PATH, demo=False)
        self.client = httpx.Client(timeout=15)

    def call(self, method, endpoint, params=None, body=None):
        headers = self.signer._auth_headers(method, "/trade-api/v2"+endpoint)
        r = self.client.request(method, config.KALSHI_PROD_BASE+endpoint, params=params, json=body, headers=headers)
        if r.status_code >= 400:
            raise BrokerError(f"{method} {endpoint}: HTTP {r.status_code} {r.text[:200]}")
        return r.json()

    def balance(self):
        return float(self.call("GET", "/portfolio/balance")["balance_dollars"])

    def buy(self, ticker, side, price, count, rest, post_only=False):
        """Buy `count` contracts of `side` at up to `price` cents. Resting orders stay until the market closes."""
        body = dict(ticker=ticker, client_order_id=f"rules-{uuid.uuid4()}",
                    side="bid" if side == "yes" else "ask",
                    price=f"{(price if side == 'yes' else 100-price)/100:.4f}", count=f"{count:.2f}",
                    time_in_force="good_till_canceled" if rest else "immediate_or_cancel",
                    self_trade_prevention_type="taker_at_cross", post_only=post_only)
        try:
            return self.call("POST", "/portfolio/events/orders", body=body)
        except httpx.RequestError as exc:
            # The request may or may not have reached the exchange. Never resend: look for it instead.
            for o in self.call("GET", "/portfolio/orders", params=dict(ticker=ticker, limit=200)).get("orders", []):
                if o.get("client_order_id") == body["client_order_id"]:
                    return dict(order_id=o["order_id"], fill_count=o["fill_count_fp"], remaining_count=o["remaining_count_fp"])
            raise BrokerError(f"order for {ticker} was not accepted ({type(exc).__name__})") from exc

    def order(self, order_id):
        """Filled contracts, dollars paid and fees for one order."""
        o = self.call("GET", f"/portfolio/orders/{order_id}")["order"]
        money = lambda *keys: sum(float(o.get(k) or 0) for k in keys)
        return dict(status=o["status"], filled=float(o["fill_count_fp"]), remaining=float(o["remaining_count_fp"]),
                    cost=money("taker_fill_cost_dollars", "maker_fill_cost_dollars"),
                    fees=money("taker_fees_dollars", "maker_fees_dollars"))

    def queue(self, order_id):
        """Contracts resting ahead of this order at its price."""
        return float(self.call("GET", f"/portfolio/orders/{order_id}/queue_position")["queue_position_fp"])

    def cancel(self, order_id, ticker):
        return self.call("DELETE", f"/portfolio/events/orders/{order_id}", params=dict(market_ticker=ticker))
