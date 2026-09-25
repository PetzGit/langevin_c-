# Pulls SPX option chain data via yfinance and writes a "T K price" CSV that
# cldiff's MarketData::loadFromCsv can read directly. Also estimates the
# risk-free rate r and dividend yield q (via put-call parity), since pricing
# an index option with only r biases the forward -- SPX pays a real, material
# dividend yield.

import yfinance as yf
import statistics
from datetime import datetime, timezone

TICKER = "^SPX"
RATE_PROXY_TICKER = "^IRX"  # 13-week T-bill discount rate, used as an r proxy
# Pushed out further still (to 60 days) and moneyness narrowed: isolates
# "does calibration work on data the model can actually fit" from Heston's
# known structural inability to match steep short-dated/wing skew.
TARGET_DAYS_OUT = [60, 90, 120, 180, 270, 365, 540]
MONEYNESS_BAND = (0.90, 1.10)  # only keep strikes within this band of spot
MIN_OPEN_INTEREST = 50  # liquidity floor
MAX_REL_SPREAD = 0.20  # (ask-bid)/mid must be below this to count as liquid
OUT_CSV = "dat.csv"
OUT_META = "meta.txt"


def fetch_risk_free_rate():
    try:
        irx = yf.Ticker(RATE_PROXY_TICKER).history(period="5d")
        if len(irx) == 0:
            raise ValueError("no ^IRX data")
        # ^IRX quotes as a percentage discount rate, e.g. 4.5 means 4.5%.
        return float(irx["Close"].iloc[-1]) / 100.0
    except Exception as e:
        print(f"warning: could not fetch risk-free rate proxy ({e}); "
              f"falling back to r=0.045")
        return 0.045


def year_frac(expiry_str):
    expiry = datetime.strptime(expiry_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    return max((expiry - now).days, 1) / 365.0


def pick_target_expiries(all_expiries):
    """Nearest available expiry to each of TARGET_DAYS_OUT, deduplicated."""
    now = datetime.now(timezone.utc)
    parsed = [(e, (datetime.strptime(e, "%Y-%m-%d").replace(tzinfo=timezone.utc) - now).days)
              for e in all_expiries]
    chosen = []
    for target in TARGET_DAYS_OUT:
        best = min(parsed, key=lambda p: abs(p[1] - target))
        if best[0] not in chosen:
            chosen.append(best[0])
    return chosen


def mid_price(row):
    bid, ask = row["bid"], row["ask"]
    if bid is None or ask is None or bid <= 0 or ask <= 0 or ask < bid:
        return None
    return 0.5 * (bid + ask)


def is_liquid(row):
    """Open-interest floor + tight relative bid-ask spread -- the two
    standard, directly-observable liquidity proxies in an option chain."""
    price = mid_price({"bid": row.bid, "ask": row.ask})
    if price is None:
        return False
    if getattr(row, "openInterest", 0) is None or row.openInterest < MIN_OPEN_INTEREST:
        return False
    rel_spread = (row.ask - row.bid) / price
    return rel_spread <= MAX_REL_SPREAD


def estimate_q(calls, puts, S0, r, T):
    """Put-call parity per matched strike: C - P = S0*e^{-qT} - K*e^{-rT}
    => q = -ln((C - P + K*e^{-rT}) / S0) / T. Only uses strikes near the
    money, where both legs are liquid and parity holds cleanly."""
    calls_by_k = {row.strike: row for row in calls.itertuples()}
    puts_by_k = {row.strike: row for row in puts.itertuples()}
    estimates = []
    for K, c in calls_by_k.items():
        if K not in puts_by_k:
            continue
        if not (MONEYNESS_BAND[0] <= K / S0 <= MONEYNESS_BAND[1]):
            continue
        p = puts_by_k[K]
        C = mid_price({"bid": c.bid, "ask": c.ask})
        P = mid_price({"bid": p.bid, "ask": p.ask})
        if C is None or P is None:
            continue
        parity_term = C - P + K * pow(2.718281828459045, -r * T)
        if parity_term <= 0:
            continue
        import math
        q_est = -math.log(parity_term / S0) / T
        estimates.append(q_est)
    if not estimates:
        return None
    return statistics.median(estimates)


def main():
    ticker = yf.Ticker(TICKER)
    S0 = ticker.fast_info["lastPrice"]
    r = fetch_risk_free_rate()
    print(f"S0={S0:.2f}  r={r:.4f}")

    expiries = pick_target_expiries(ticker.options)
    print(f"target expiries: {expiries}")

    rows = []  # (T, K, price)
    q_estimates = []
    for expiry in expiries:
        T = year_frac(expiry)
        chain = ticker.option_chain(expiry)
        calls, puts = chain.calls, chain.puts

        # Parity-implied q is dominated by bid/ask noise at very short
        # maturities (dividing by tiny T amplifies it) -- skip those for the
        # q estimate specifically, though their quotes still go into dat.csv.
        q_t = estimate_q(calls, puts, S0, r, T) if T >= 0.05 else None
        if q_t is not None:
            q_estimates.append(q_t)
            print(f"  {expiry} (T={T:.3f}): q_est={q_t:.4f}")
        else:
            print(f"  {expiry} (T={T:.3f}): q_est unavailable (no clean parity pairs)")

        # Thin to a handful of strikes spanning the moneyness band per
        # maturity (like the 5x5 synthetic grid used earlier) -- keeps the
        # calibration cheap; HestonObjective re-evaluates every quote on
        # every objective call, so a full multi-hundred-quote chain would
        # make even a "quick" Optimiser run take hours, not minutes.
        # Restricted to strikes passing the liquidity filter (open interest
        # + tight relative spread) so the nearest-to-target search only picks
        # among genuinely liquid quotes, not just whatever's closest.
        target_moneyness = [0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20]
        liquid_rows = [row for row in calls.itertuples() if is_liquid(row)]
        available = {row.strike: row for row in liquid_rows}
        n_added = 0
        n_illiquid = calls.shape[0] - len(liquid_rows)
        seen_strikes = set()
        for m in target_moneyness:
            if not available:
                break
            target_k = m * S0
            nearest = min(available.keys(), key=lambda k: abs(k - target_k))
            if nearest in seen_strikes:
                continue
            row = available[nearest]
            price = mid_price({"bid": row.bid, "ask": row.ask})
            if price is None:
                continue
            rows.append((T, nearest, price))
            seen_strikes.add(nearest)
            n_added += 1
        print(f"    added {n_added} liquid call quotes (thinned; {n_illiquid} filtered out as illiquid)")

    if not q_estimates:
        print("warning: no q estimates from any maturity; defaulting q=0.0")
        q = 0.0
    else:
        q = statistics.median(q_estimates)
    print(f"\nfinal q estimate (median across maturities): {q:.4f}")

    with open(OUT_CSV, "w", newline="") as f:
        for T, K, price in rows:
            f.write(f"{T:.6f} {K:.2f} {price:.4f}\n")
    print(f"wrote {len(rows)} quotes to {OUT_CSV}")

    with open(OUT_META, "w") as f:
        f.write(f"S0 {S0:.4f}\nr {r:.6f}\nq {q:.6f}\n")
    print(f"wrote S0/r/q to {OUT_META}")


if __name__ == "__main__":
    main()
