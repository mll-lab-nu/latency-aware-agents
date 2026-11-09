"""Session-aware log-AR(1) and log-ARMA(1,1) timing fits.

Each feature has shared coefficients across independent sessions. Likelihoods
restart at session boundaries; ordinary episode boundaries preserve state.
Runtime sampling is NumPy-only, with correlated Gaussian innovations.
"""
from __future__ import annotations

import numpy as np


def fit_autoregressive(sessions, *, method):
    from scipy.optimize import minimize
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    q = {"ar1": 0, "arma": 1}[method]
    arrays = [np.asarray(s["pairs"], dtype=float) for s in sessions]
    if any(len(x) < 32 or np.any(x[:, 1] <= 0) or np.any(x[:, 0] <= x[:, 1]) for x in arrays):
        raise ValueError("Log-AR fits require >=32 admitted requests per session and action > busy > 0")
    features = [np.log(np.column_stack((x[:, 1], x[:, 0] - x[:, 1]))) for x in arrays]
    pooled = np.concatenate(features)
    center, scale = pooled.mean(axis=0), pooled.std(axis=0)
    if np.any(scale < 1e-12):
        raise ValueError("Log-AR fits require nonconstant busy time and overhead")
    normalized = [(x - center) / scale for x in features]
    coefficients, residuals, fits = [], [], []
    for column in range(2):
        models = [SARIMAX(x[:, column], order=(1, 0, q), trend="c",
                          enforce_stationarity=True, enforce_invertibility=True,
                          initialization="stationary") for x in normalized]
        count = sum(len(x) for x in normalized)

        def objective(params):
            return -sum(model.loglike(params, transformed=False) for model in models) / count

        trials = []
        for phi, theta in ((0., 0.), (.5, -.3), (.9, .3)):
            constrained = np.array([0., phi, theta, 1.] if q else [0., phi, 1.])
            result = minimize(objective, models[0].untransform_params(constrained),
                              method="L-BFGS-B", options={"maxiter": 500, "ftol": 1e-10})
            trials.append(result)
        valid = [i for i, trial in enumerate(trials) if trial.success and np.isfinite(trial.fun)]
        if not valid:
            raise RuntimeError(f"{method} feature {column}: no converged likelihood fit; "
                               + "; ".join(str(trial.message) for trial in trials))
        best = min(valid, key=lambda i: trials[i].fun)
        params = models[0].transform_params(trials[best].x)
        coefficients.append([params[0], params[1], params[2] if q else 0., params[-1]])
        # Discard the first 16 filtering residuals of each independent session
        # only for cross-feature innovation correlation, never for likelihood.
        residuals.append(np.concatenate([model.filter(params).resid[16:] for model in models]))
        fits.append({"feature": column, "selected_restart": best,
                     "log_likelihood": float(-trials[best].fun * count),
                     "trials": [{"converged": bool(t.success), "iterations": int(t.nit),
                                 "message": str(t.message),
                                 "log_likelihood": float(-t.fun * count) if np.isfinite(t.fun) else None}
                                for t in trials]})
    intercept, ar, ma, variance = np.asarray(coefficients).T
    correlation = np.corrcoef(residuals)
    covariance = correlation * np.sqrt(np.outer(variance, variance))
    return {"order": [1, 0, q], "features": ["log(worker_busy_ms)", "log(action_latency_ms - worker_busy_ms)"],
            "center": center.tolist(), "scale": scale.tolist(),
            "intercept": intercept.tolist(), "ar": ar.tolist(), "ma": ma.tolist(),
            "innovation_covariance": covariance.tolist(), "fits": fits,
            "fitting": "Sum of independent-session stationary Gaussian marginal log-likelihoods, "
                       "three starts per feature; paired filtering residuals estimate innovation correlation",
            "initialization": "Joint stationary Gaussian state; no clipping or injected burst states",
            "emission_support": "unbounded_positive"}
