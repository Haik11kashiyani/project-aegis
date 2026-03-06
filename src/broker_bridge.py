"""
====================================================
PROJECT AEGIS — Broker Bridge v1.0
====================================================
Paper-to-Live bridge supporting 3 Indian brokers:
  1. Zerodha (Kite Connect API)
  2. Angel One (SmartAPI)
  3. Groww (unofficial REST)

Modes:
  - PAPER   : No real orders, only logs (default)
  - DRY_RUN : Simulates order flow, shows confirmations
  - LIVE    : Sends real orders to broker API

All modes log to data/broker_orders.json for audit.
====================================================
"""

import os
import json
import time
from datetime import datetime
from abc import ABC, abstractmethod
import pytz

IST = pytz.timezone("Asia/Kolkata")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORDERS_FILE = os.path.join(BASE_DIR, "data", "broker_orders.json")
BROKER_CONFIG_FILE = os.path.join(BASE_DIR, "data", "broker_config.json")


# ══════════════════════════════════════════════════
#  ORDER DATA
# ══════════════════════════════════════════════════
class Order:
    """Standardized order object across all brokers."""
    def __init__(self, symbol: str, side: str, qty: int, price: float,
                 order_type: str = "MARKET", product: str = "MIS",
                 stop_loss: float = 0, target: float = 0):
        self.id = f"AEGIS-{datetime.now(IST).strftime('%H%M%S')}-{symbol.replace('.NS','')}"
        self.symbol = symbol
        self.side = side.upper()       # BUY or SELL
        self.qty = qty
        self.price = price
        self.order_type = order_type   # MARKET, LIMIT, SL
        self.product = product         # MIS (intraday), CNC (delivery)
        self.stop_loss = stop_loss
        self.target = target
        self.status = "PENDING"
        self.broker_order_id = None
        self.broker_response = None
        self.timestamp = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")

    def to_dict(self) -> dict:
        return {
            "id": self.id, "symbol": self.symbol, "side": self.side,
            "qty": self.qty, "price": self.price, "order_type": self.order_type,
            "product": self.product, "stop_loss": self.stop_loss,
            "target": self.target, "status": self.status,
            "broker_order_id": self.broker_order_id,
            "broker_response": self.broker_response,
            "timestamp": self.timestamp,
        }


# ══════════════════════════════════════════════════
#  ABSTRACT BROKER
# ══════════════════════════════════════════════════
class BrokerBase(ABC):
    """Interface that all broker implementations must follow."""
    name = "BASE"

    @abstractmethod
    def connect(self, credentials: dict) -> bool:
        """Authenticate with broker. Returns True on success."""
        ...

    @abstractmethod
    def place_order(self, order: Order) -> dict:
        """Place order. Returns {"success": bool, "order_id": str, "message": str}."""
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> dict:
        """Cancel pending order."""
        ...

    @abstractmethod
    def get_positions(self) -> list:
        """Get current open positions."""
        ...

    @abstractmethod
    def get_balance(self) -> dict:
        """Get account balance/margin info."""
        ...

    @abstractmethod
    def get_order_status(self, order_id: str) -> dict:
        """Get status of a placed order."""
        ...

    def is_connected(self) -> bool:
        return False


# ══════════════════════════════════════════════════
#  ZERODHA (Kite Connect)
# ══════════════════════════════════════════════════
class ZerodhaBroker(BrokerBase):
    """
    Zerodha Kite Connect API integration.
    Requires: pip install kiteconnect
    Credentials: api_key, api_secret, request_token (or access_token)
    """
    name = "ZERODHA"

    def __init__(self):
        self.kite = None
        self._connected = False

    def connect(self, credentials: dict) -> bool:
        try:
            from kiteconnect import KiteConnect
            api_key = credentials.get("api_key", os.getenv("ZERODHA_API_KEY", ""))
            api_secret = credentials.get("api_secret", os.getenv("ZERODHA_API_SECRET", ""))
            access_token = credentials.get("access_token", os.getenv("ZERODHA_ACCESS_TOKEN", ""))
            request_token = credentials.get("request_token", os.getenv("ZERODHA_REQUEST_TOKEN", ""))

            self.kite = KiteConnect(api_key=api_key)

            if access_token:
                self.kite.set_access_token(access_token)
            elif request_token:
                data = self.kite.generate_session(request_token, api_secret=api_secret)
                self.kite.set_access_token(data["access_token"])
            else:
                return False

            # Test connection
            profile = self.kite.profile()
            self._connected = True
            print(f"[ZERODHA] Connected as: {profile.get('user_name', 'N/A')}")
            return True
        except ImportError:
            print("[ZERODHA] kiteconnect not installed. Run: pip install kiteconnect")
            return False
        except Exception as e:
            print(f"[ZERODHA] Connection failed: {e}")
            return False

    def is_connected(self) -> bool:
        return self._connected

    def _symbol_to_exchange(self, symbol: str) -> tuple:
        """Convert RELIANCE.NS → (NSE, RELIANCE)."""
        clean = symbol.replace(".NS", "").replace(".BO", "")
        exchange = "BSE" if ".BO" in symbol else "NSE"
        return exchange, clean

    def place_order(self, order: Order) -> dict:
        if not self._connected:
            return {"success": False, "order_id": None, "message": "Not connected"}
        try:
            exchange, tradingsymbol = self._symbol_to_exchange(order.symbol)
            kite_order = {
                "exchange": exchange,
                "tradingsymbol": tradingsymbol,
                "transaction_type": self.kite.TRANSACTION_TYPE_BUY if order.side == "BUY" else self.kite.TRANSACTION_TYPE_SELL,
                "quantity": order.qty,
                "product": self.kite.PRODUCT_MIS if order.product == "MIS" else self.kite.PRODUCT_CNC,
                "order_type": self.kite.ORDER_TYPE_MARKET if order.order_type == "MARKET" else self.kite.ORDER_TYPE_LIMIT,
                "variety": self.kite.VARIETY_REGULAR,
            }
            if order.order_type == "LIMIT":
                kite_order["price"] = order.price

            order_id = self.kite.place_order(**kite_order)
            return {"success": True, "order_id": str(order_id), "message": "Order placed"}
        except Exception as e:
            return {"success": False, "order_id": None, "message": str(e)}

    def cancel_order(self, order_id: str) -> dict:
        try:
            self.kite.cancel_order(variety=self.kite.VARIETY_REGULAR, order_id=order_id)
            return {"success": True, "message": "Cancelled"}
        except Exception as e:
            return {"success": False, "message": str(e)}

    def get_positions(self) -> list:
        try:
            positions = self.kite.positions()
            return positions.get("net", [])
        except Exception:
            return []

    def get_balance(self) -> dict:
        try:
            margins = self.kite.margins(segment="equity")
            return {
                "available": margins.get("available", {}).get("live_balance", 0),
                "used": margins.get("utilised", {}).get("debits", 0),
                "total": margins.get("net", 0),
            }
        except Exception:
            return {"available": 0, "used": 0, "total": 0}

    def get_order_status(self, order_id: str) -> dict:
        try:
            orders = self.kite.orders()
            for o in orders:
                if str(o.get("order_id")) == str(order_id):
                    return {
                        "status": o.get("status", "UNKNOWN"),
                        "filled_qty": o.get("filled_quantity", 0),
                        "avg_price": o.get("average_price", 0),
                    }
            return {"status": "NOT_FOUND"}
        except Exception as e:
            return {"status": "ERROR", "message": str(e)}


# ══════════════════════════════════════════════════
#  ANGEL ONE (SmartAPI)
# ══════════════════════════════════════════════════
class AngelOneBroker(BrokerBase):
    """
    Angel One SmartAPI integration.
    Requires: pip install smartapi-python
    Credentials: api_key, client_id, password, totp_secret
    """
    name = "ANGEL_ONE"

    def __init__(self):
        self.smart_api = None
        self._connected = False

    def connect(self, credentials: dict) -> bool:
        try:
            from SmartApi import SmartConnect
            api_key = credentials.get("api_key", os.getenv("ANGEL_API_KEY", ""))
            client_id = credentials.get("client_id", os.getenv("ANGEL_CLIENT_ID", ""))
            password = credentials.get("password", os.getenv("ANGEL_PASSWORD", ""))
            totp_secret = credentials.get("totp_secret", os.getenv("ANGEL_TOTP_SECRET", ""))

            self.smart_api = SmartConnect(api_key=api_key)

            totp = ""
            if totp_secret:
                try:
                    import pyotp
                    totp = pyotp.TOTP(totp_secret).now()
                except ImportError:
                    print("[ANGEL] pyotp not installed for TOTP generation.")
                    return False

            data = self.smart_api.generateSession(client_id, password, totp)
            if data.get("status"):
                self._connected = True
                print(f"[ANGEL ONE] Connected as: {client_id}")
                return True
            return False
        except ImportError:
            print("[ANGEL ONE] smartapi-python not installed. Run: pip install smartapi-python")
            return False
        except Exception as e:
            print(f"[ANGEL ONE] Connection failed: {e}")
            return False

    def is_connected(self) -> bool:
        return self._connected

    def _get_token(self, symbol: str) -> str:
        """Look up Angel One symbol token. Simplified mapping."""
        # In production, load from Angel's instrument master file
        token_map = {
            "RELIANCE": "2885", "HDFCBANK": "1333", "ICICIBANK": "4963",
            "SBIN": "3045", "TCS": "11536", "INFY": "1594",
            "TATASTEEL": "3499", "NTPC": "11630", "POWERGRID": "14977",
            "COALINDIA": "20374",
        }
        clean = symbol.replace(".NS", "").replace(".BO", "")
        return token_map.get(clean, "0")

    def place_order(self, order: Order) -> dict:
        if not self._connected:
            return {"success": False, "order_id": None, "message": "Not connected"}
        try:
            clean = order.symbol.replace(".NS", "").replace(".BO", "")
            token = self._get_token(order.symbol)
            params = {
                "variety": "NORMAL",
                "tradingsymbol": clean,
                "symboltoken": token,
                "transactiontype": order.side,
                "exchange": "NSE",
                "ordertype": "MARKET" if order.order_type == "MARKET" else "LIMIT",
                "producttype": "INTRADAY" if order.product == "MIS" else "DELIVERY",
                "duration": "DAY",
                "quantity": str(order.qty),
            }
            if order.order_type == "LIMIT":
                params["price"] = str(order.price)

            resp = self.smart_api.placeOrder(params)
            if resp:
                return {"success": True, "order_id": str(resp), "message": "Order placed"}
            return {"success": False, "order_id": None, "message": "No response from Angel"}
        except Exception as e:
            return {"success": False, "order_id": None, "message": str(e)}

    def cancel_order(self, order_id: str) -> dict:
        try:
            self.smart_api.cancelOrder(order_id, "NORMAL")
            return {"success": True, "message": "Cancelled"}
        except Exception as e:
            return {"success": False, "message": str(e)}

    def get_positions(self) -> list:
        try:
            pos = self.smart_api.position()
            return pos.get("data", []) if pos else []
        except Exception:
            return []

    def get_balance(self) -> dict:
        try:
            funds = self.smart_api.rmsLimit()
            data = funds.get("data", {}) if funds else {}
            return {
                "available": float(data.get("availablecash", 0)),
                "used": float(data.get("utiliseddebits", 0)),
                "total": float(data.get("net", 0)),
            }
        except Exception:
            return {"available": 0, "used": 0, "total": 0}

    def get_order_status(self, order_id: str) -> dict:
        try:
            orders = self.smart_api.orderBook()
            if orders and orders.get("data"):
                for o in orders["data"]:
                    if str(o.get("orderid")) == str(order_id):
                        return {
                            "status": o.get("orderstatus", "UNKNOWN"),
                            "filled_qty": int(o.get("filledshares", 0)),
                            "avg_price": float(o.get("averageprice", 0)),
                        }
            return {"status": "NOT_FOUND"}
        except Exception as e:
            return {"status": "ERROR", "message": str(e)}


# ══════════════════════════════════════════════════
#  GROWW (Unofficial REST)
# ══════════════════════════════════════════════════
class GrowwBroker(BrokerBase):
    """
    Groww broker integration (unofficial REST API).
    
    NOTE: Groww does not have an official public trading API as of 2025.
    This uses their internal REST endpoints which may change.
    Use at your own risk — not recommended for large capital.
    
    Credentials: email, password (or auth_token)
    """
    name = "GROWW"

    def __init__(self):
        self._session = None
        self._connected = False
        self._auth_token = None
        self._base_url = "https://groww.in/v1/api"

    def connect(self, credentials: dict) -> bool:
        try:
            import urllib.request
            import urllib.parse

            auth_token = credentials.get("auth_token", os.getenv("GROWW_AUTH_TOKEN", ""))

            if auth_token:
                self._auth_token = auth_token
                # Verify token works
                req = urllib.request.Request(
                    f"{self._base_url}/user/v1/user/profile",
                    headers={
                        "Authorization": f"Bearer {auth_token}",
                        "Content-Type": "application/json",
                    }
                )
                try:
                    resp = urllib.request.urlopen(req, timeout=10)
                    data = json.loads(resp.read().decode())
                    self._connected = True
                    print(f"[GROWW] Connected as: {data.get('userName', 'User')}")
                    return True
                except Exception as e:
                    print(f"[GROWW] Token validation failed: {e}")
                    return False

            # Login with email/password
            email = credentials.get("email", os.getenv("GROWW_EMAIL", ""))
            password = credentials.get("password", os.getenv("GROWW_PASSWORD", ""))

            if not email or not password:
                print("[GROWW] No credentials provided")
                return False

            login_data = json.dumps({"email": email, "password": password}).encode()
            req = urllib.request.Request(
                f"{self._base_url}/user/v1/user/login",
                data=login_data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                resp = urllib.request.urlopen(req, timeout=10)
                data = json.loads(resp.read().decode())
                self._auth_token = data.get("authToken", "")
                if self._auth_token:
                    self._connected = True
                    print("[GROWW] Login successful")
                    return True
            except Exception as e:
                print(f"[GROWW] Login failed: {e}")
            return False
        except Exception as e:
            print(f"[GROWW] Connection error: {e}")
            return False

    def is_connected(self) -> bool:
        return self._connected

    def _api_call(self, endpoint: str, method: str = "GET", data: dict = None) -> dict:
        """Make authenticated API call to Groww."""
        import urllib.request
        url = f"{self._base_url}/{endpoint}"
        headers = {
            "Authorization": f"Bearer {self._auth_token}",
            "Content-Type": "application/json",
        }
        body = json.dumps(data).encode() if data else None
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            resp = urllib.request.urlopen(req, timeout=15)
            return json.loads(resp.read().decode())
        except Exception as e:
            return {"error": str(e)}

    def place_order(self, order: Order) -> dict:
        if not self._connected:
            return {"success": False, "order_id": None, "message": "Not connected"}
        try:
            clean = order.symbol.replace(".NS", "").replace(".BO", "")
            order_data = {
                "stockSymbol": clean,
                "transactionType": order.side,
                "quantity": order.qty,
                "orderType": order.order_type,
                "productType": "INTRADAY" if order.product == "MIS" else "DELIVERY",
                "exchange": "NSE",
            }
            if order.order_type == "LIMIT":
                order_data["limitPrice"] = order.price

            resp = self._api_call("stocks/v1/order/place", method="POST", data=order_data)
            if "error" not in resp:
                oid = resp.get("orderId", resp.get("order_id", ""))
                return {"success": True, "order_id": str(oid), "message": "Order placed on Groww"}
            return {"success": False, "order_id": None, "message": resp.get("error", "Unknown")}
        except Exception as e:
            return {"success": False, "order_id": None, "message": str(e)}

    def cancel_order(self, order_id: str) -> dict:
        try:
            resp = self._api_call(f"stocks/v1/order/cancel/{order_id}", method="POST")
            return {"success": "error" not in resp, "message": resp.get("message", "Done")}
        except Exception as e:
            return {"success": False, "message": str(e)}

    def get_positions(self) -> list:
        try:
            resp = self._api_call("stocks/v1/holdings")
            return resp.get("holdings", []) if isinstance(resp, dict) else []
        except Exception:
            return []

    def get_balance(self) -> dict:
        try:
            resp = self._api_call("stocks/v1/funds")
            return {
                "available": float(resp.get("availableFunds", 0)),
                "used": float(resp.get("usedFunds", 0)),
                "total": float(resp.get("totalFunds", 0)),
            }
        except Exception:
            return {"available": 0, "used": 0, "total": 0}

    def get_order_status(self, order_id: str) -> dict:
        try:
            resp = self._api_call(f"stocks/v1/order/{order_id}")
            return {
                "status": resp.get("orderStatus", "UNKNOWN"),
                "filled_qty": int(resp.get("filledQuantity", 0)),
                "avg_price": float(resp.get("averagePrice", 0)),
            }
        except Exception as e:
            return {"status": "ERROR", "message": str(e)}


# ══════════════════════════════════════════════════
#  PAPER BROKER (default — no real orders)
# ══════════════════════════════════════════════════
class PaperBroker(BrokerBase):
    """Simulated broker for paper trading. Always succeeds, never touches real money."""
    name = "PAPER"

    def __init__(self):
        self._connected = True
        self._orders = {}
        self._balance = 1000.0

    def connect(self, credentials: dict) -> bool:
        self._balance = credentials.get("capital", 1000.0)
        self._connected = True
        print(f"[PAPER] Paper broker active. Capital: ₹{self._balance:,.2f}")
        return True

    def is_connected(self) -> bool:
        return True

    def place_order(self, order: Order) -> dict:
        order.status = "FILLED"
        order.broker_order_id = f"PAPER-{int(time.time())}"
        self._orders[order.broker_order_id] = order
        return {"success": True, "order_id": order.broker_order_id, "message": "Paper order filled"}

    def cancel_order(self, order_id: str) -> dict:
        if order_id in self._orders:
            self._orders[order_id].status = "CANCELLED"
        return {"success": True, "message": "Paper order cancelled"}

    def get_positions(self) -> list:
        return [o.to_dict() for o in self._orders.values() if o.status == "FILLED"]

    def get_balance(self) -> dict:
        return {"available": self._balance, "used": 0, "total": self._balance}

    def get_order_status(self, order_id: str) -> dict:
        if order_id in self._orders:
            return {"status": "FILLED", "filled_qty": self._orders[order_id].qty,
                    "avg_price": self._orders[order_id].price}
        return {"status": "NOT_FOUND"}


# ══════════════════════════════════════════════════
#  BROKER BRIDGE — The Master Controller
# ══════════════════════════════════════════════════
BROKER_REGISTRY = {
    "PAPER": PaperBroker,
    "ZERODHA": ZerodhaBroker,
    "ANGEL_ONE": AngelOneBroker,
    "GROWW": GrowwBroker,
}


class BrokerBridge:
    """
    Master controller for all broker interactions.
    
    Modes:
      PAPER   — Default. No real orders.
      DRY_RUN — Goes through the full flow but pauses for confirmation.
      LIVE    — Real money. Sends actual orders.
    """

    def __init__(self, mode: str = "PAPER", broker_name: str = "PAPER"):
        self.mode = mode.upper()              # PAPER, DRY_RUN, LIVE
        self.broker_name = broker_name.upper()
        self.broker: BrokerBase = None
        self.order_history = []
        self._confirmations_pending = []

        # Instantiate the right broker
        broker_cls = BROKER_REGISTRY.get(self.broker_name, PaperBroker)
        self.broker = broker_cls()

        self._load_order_history()

    def connect(self, credentials: dict = None) -> bool:
        """Connect to broker. In PAPER mode, always succeeds."""
        if self.mode == "PAPER":
            return self.broker.connect(credentials or {"capital": 1000})

        if credentials is None:
            credentials = self._load_credentials()

        success = self.broker.connect(credentials)
        if success:
            print(f"[BRIDGE] Connected to {self.broker_name} in {self.mode} mode")
        else:
            print(f"[BRIDGE] Failed to connect to {self.broker_name}. Falling back to PAPER.")
            self.mode = "PAPER"
            self.broker = PaperBroker()
            self.broker.connect({"capital": 1000})
        return success

    def execute_buy(self, symbol: str, qty: int, price: float,
                    stop_loss: float = 0, target: float = 0,
                    confidence: float = 0, neuro_score: float = 0) -> dict:
        """Execute a BUY order through the broker."""
        order = Order(symbol, "BUY", qty, price, "MARKET", "MIS", stop_loss, target)

        if self.mode == "PAPER":
            result = self.broker.place_order(order)
            order.status = "FILLED"
            order.broker_response = result
            self._log_order(order, confidence, neuro_score)
            return result

        if self.mode == "DRY_RUN":
            # Log as dry run — don't actually place
            print(f"\n{'='*50}")
            print(f"🔶 DRY RUN — Order Preview")
            print(f"  Symbol : {symbol}")
            print(f"  Side   : BUY")
            print(f"  Qty    : {qty}")
            print(f"  Price  : ₹{price:,.2f}")
            print(f"  SL     : ₹{stop_loss:,.2f}")
            print(f"  Target : ₹{target:,.2f}")
            print(f"  Risk   : ₹{(price - stop_loss) * qty:,.2f}")
            print(f"  AI Conf: {confidence:.2%} | NeuroScore: {neuro_score:+.3f}")
            print(f"{'='*50}")
            order.status = "DRY_RUN"
            self._log_order(order, confidence, neuro_score)
            return {"success": True, "order_id": order.id, "message": "DRY RUN — not placed"}

        # LIVE mode
        print(f"[LIVE] Placing BUY: {qty} × {symbol} @ ₹{price:,.2f}")
        result = self.broker.place_order(order)
        order.status = "SENT" if result["success"] else "FAILED"
        order.broker_order_id = result.get("order_id")
        order.broker_response = result
        self._log_order(order, confidence, neuro_score)
        return result

    def execute_sell(self, symbol: str, qty: int, price: float,
                     exit_type: str = "TARGET_HIT") -> dict:
        """Execute a SELL order through the broker."""
        order = Order(symbol, "SELL", qty, price, "MARKET", "MIS")

        if self.mode == "PAPER":
            result = self.broker.place_order(order)
            order.status = "FILLED"
            self._log_order(order, exit_type=exit_type)
            return result

        if self.mode == "DRY_RUN":
            print(f"\n🔶 DRY RUN SELL: {qty} × {symbol} @ ₹{price:,.2f} ({exit_type})")
            order.status = "DRY_RUN"
            self._log_order(order, exit_type=exit_type)
            return {"success": True, "order_id": order.id, "message": "DRY RUN"}

        # LIVE
        print(f"[LIVE] Placing SELL: {qty} × {symbol} @ ₹{price:,.2f} ({exit_type})")
        result = self.broker.place_order(order)
        order.status = "SENT" if result["success"] else "FAILED"
        order.broker_order_id = result.get("order_id")
        self._log_order(order, exit_type=exit_type)
        return result

    def get_positions(self) -> list:
        return self.broker.get_positions()

    def get_balance(self) -> dict:
        return self.broker.get_balance()

    def get_status(self) -> dict:
        """Get full bridge status for dashboard."""
        return {
            "mode": self.mode,
            "broker": self.broker_name,
            "connected": self.broker.is_connected(),
            "total_orders": len(self.order_history),
            "recent_orders": self.order_history[-10:] if self.order_history else [],
        }

    # ── Internal helpers ──

    def _log_order(self, order: Order, confidence: float = 0,
                   neuro_score: float = 0, exit_type: str = ""):
        """Log order to history file."""
        entry = order.to_dict()
        entry["mode"] = self.mode
        entry["broker"] = self.broker_name
        entry["ai_confidence"] = confidence
        entry["neuro_score"] = neuro_score
        if exit_type:
            entry["exit_type"] = exit_type

        self.order_history.append(entry)
        self._save_order_history()

    def _save_order_history(self):
        try:
            os.makedirs(os.path.dirname(ORDERS_FILE), exist_ok=True)
            # Keep last 500
            to_save = self.order_history[-500:]
            with open(ORDERS_FILE, "w") as f:
                json.dump(to_save, f, indent=2, default=str)
        except Exception:
            pass

    def _load_order_history(self):
        try:
            if os.path.exists(ORDERS_FILE):
                with open(ORDERS_FILE, "r") as f:
                    self.order_history = json.load(f)
        except Exception:
            self.order_history = []

    def _load_credentials(self) -> dict:
        """Load credentials from broker_config.json or env vars."""
        creds = {}
        try:
            if os.path.exists(BROKER_CONFIG_FILE):
                with open(BROKER_CONFIG_FILE, "r") as f:
                    all_config = json.load(f)
                creds = all_config.get(self.broker_name, {})
        except Exception:
            pass
        return creds


# ══════════════════════════════════════════════════
#  CONVENIENCE FACTORY
# ══════════════════════════════════════════════════
def create_bridge(mode: str = None, broker: str = None) -> BrokerBridge:
    """
    Create a BrokerBridge from environment variables or defaults.
    
    Env vars:
      AEGIS_TRADE_MODE  = PAPER | DRY_RUN | LIVE
      AEGIS_BROKER      = PAPER | ZERODHA | ANGEL_ONE | GROWW
    """
    mode = mode or os.getenv("AEGIS_TRADE_MODE", "PAPER").upper()
    broker = broker or os.getenv("AEGIS_BROKER", "PAPER").upper()

    bridge = BrokerBridge(mode=mode, broker_name=broker)
    bridge.connect()
    return bridge
