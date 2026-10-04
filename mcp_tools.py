"""
mcp_tools.py Simulates core banking utilities exposed to agents via deterministic function calls.
"""
import math

def tool_transaction_fetcher(user_id: str) -> dict:
    """Retrieves baseline account metrics and historical spending limits."""
    # Simulated database lookup
    return {
        "user_id": user_id,
        "avg_transaction_amount": 75.00,
        "monthly_credit_limit": 5000.00,
        "current_balance": 4200.00,
        "home_location_coords": (48.8566, 2.3522),  # Paris, France
        "account_status": "ACTIVE"
    }


def tool_location_validator(coords_a: tuple[float, float], coords_b: tuple[float, float],
                            time_delta_hours: float) -> dict:
    """Calculates physical travel feasibility between two geographic points."""
    lat1, lon1 = coords_a
    lat2, lon2 = coords_b
    R = 6371  # Earth radius in km

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance_km = R * c

    required_speed_kmh = distance_km / max(time_delta_hours, 0.01)
    is_impossible = required_speed_kmh > 900.0  # Speed exceeding commercial flight threshold

    return {
        "distance_km": round(distance_km, 2),
        "required_speed_kmh": round(required_speed_kmh, 2),
        "impossible_travel": is_impossible
    }


def tool_fraud_scoring_model(amount: float, baseline_avg: float, impossible_travel: bool) -> float:
    """Calculates ML-based risk probability score."""
    ratio = amount / max(baseline_avg, 1.0)
    score = min(1.0, (ratio * 0.15) + (0.60 if impossible_travel else 0.0))
    return round(score, 4)