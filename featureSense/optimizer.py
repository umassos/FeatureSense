import numpy as np
from scipy.optimize import linprog

from .metrics import get_metrics


def optimize_features_with_puc_tradeoff(
    latency=5.0,
    alpha=0.5,
    candidate_features=None,
):
    """
    Select features by maximizing utility and minimizing privacy leakage
    under a latency constraint.

    Objective:
        maximize sum_i (alpha * utility_i - (1 - alpha) * privacy_i) * x_i

    subject to:
        sum_i cost_i * x_i <= latency
    """

    metrics = get_metrics()

    if candidate_features is not None:
        metrics = metrics[metrics["feature"].isin(candidate_features)]

    if metrics.empty:
        return []

    features = metrics["feature"].tolist()
    utility = metrics["utility"].to_numpy()
    privacy = metrics["privacy"].to_numpy()
    cost = metrics["cost"].to_numpy()

    score = alpha * utility - (1 - alpha) * privacy

    # linprog minimizes, so negate score.
    c = -score
    A_ub = [cost]
    b_ub = [latency]
    bounds = [(0, 1) for _ in features]

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    if not result.success:
        raise RuntimeError(result.message)

    selected = [
        feature for feature, value in zip(features, result.x)
        if value >= 0.5
    ]

    return selected
