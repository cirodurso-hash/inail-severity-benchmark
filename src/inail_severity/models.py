"""Model specifications: the configurations used in the paper and the search spaces of the
comparable tuning procedure (Supplementary Table S8)."""
import numpy as np
import xgboost as xgb
from catboost import CatBoostClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.linear_model import LogisticRegression

# --- configurations used in the main analysis (Tables 3-4) -------------------------------
DEFAULTS = {
    "Logistic Regression": dict(C=1.0),
    "XGBoost": dict(n_estimators=100, max_depth=6, learning_rate=0.3),      # library defaults
    "Balanced Random Forest": dict(n_estimators=500, max_depth=12, min_samples_split=5,
                                   min_samples_leaf=2, max_features="sqrt"),
    "CatBoost (binary)": dict(iterations=1000, learning_rate=0.01, depth=8, l2_leaf_reg=5),
}

# --- search spaces of the comparable tuning procedure (same budget for every model) --------
SPACES = {
    "Logistic Regression": dict(C=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0]),
    "XGBoost": dict(n_estimators=[1000], max_depth=[2, 3, 4, 6, 8], learning_rate=[0.03, 0.1, 0.3],
                    min_child_weight=[1, 5, 20], subsample=[0.7, 1.0], colsample_bytree=[0.7, 1.0],
                    reg_lambda=[1, 5]),
    "Balanced Random Forest": dict(n_estimators=[300], max_depth=[6, 8, 12, 16, None],
                                   min_samples_leaf=[1, 2, 5, 20], min_samples_split=[2, 5, 10],
                                   max_features=["sqrt", 0.5]),
    "CatBoost (binary)": dict(iterations=[2000], learning_rate=[0.01, 0.03, 0.1], depth=[4, 6, 8],
                              l2_leaf_reg=[1, 3, 5, 10]),
}


def sample_configs(name, n, rng):
    """n distinct random configurations from the search space (all of them if fewer exist)."""
    space = SPACES[name]; keys = list(space)
    total = int(np.prod([len(space[k]) for k in keys]))
    if total <= n:
        import itertools
        return [dict(zip(keys, v)) for v in itertools.product(*[space[k] for k in keys])]
    seen = set(); out = []
    while len(out) < n:
        c = tuple(rng.integers(0, len(space[k])) for k in keys)
        if c not in seen:
            seen.add(c); out.append({k: space[k][i] for k, i in zip(keys, c)})
    return out


def make(name, params, seed, n_jobs, weighted=True):
    """Instantiate a model. weighted=False gives the unweighted probability models (Panel C)."""
    p = dict(params)
    if name == "Logistic Regression":
        return LogisticRegression(solver="lbfgs", class_weight="balanced" if weighted else None,
                                  max_iter=2000 if weighted else 3000, **p)
    if name == "XGBoost":
        return xgb.XGBClassifier(objective="multi:softprob", num_class=4, eval_metric="mlogloss",
                                 random_state=seed, n_jobs=n_jobs, **p)
    if name == "Balanced Random Forest":
        return BalancedRandomForestClassifier(random_state=seed, sampling_strategy="not minority",
                                              bootstrap=True, replacement=False, n_jobs=n_jobs, **p)
    if name == "CatBoost (binary)":
        kw = dict(loss_function="Logloss", early_stopping_rounds=200, verbose=0, random_seed=seed,
                  thread_count=n_jobs, allow_writing_files=False)
        if weighted:
            kw.update(auto_class_weights="Balanced", eval_metric="AUC")
        else:
            kw.update(eval_metric="Logloss")
        return CatBoostClassifier(**kw, **p)
    raise ValueError(name)
