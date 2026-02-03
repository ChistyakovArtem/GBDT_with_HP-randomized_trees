import os
import tempfile
import numpy as np
import copy
from catboost import CatBoostRegressor, Pool


# ---------------------------------------------------------------------------
# Bootstrap samplers
# ---------------------------------------------------------------------------


class BayesianBootstrap:
    """
    Default CatBoost bootstrap for regression on CPU when boosting_type != Plain+MVS path.
    Weights ~ Exp(1/temperature). temperature=1 -> standard exponential weights.
    temperature=0 -> all weights = 1 (no bagging).
    """

    def __init__(self, bagging_temperature: float = 1.0):
        self.bagging_temperature = bagging_temperature

    def sample_weights(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if self.bagging_temperature == 0:
            return np.ones(n)
        # Exp(1) scaled by temperature
        return rng.exponential(scale=self.bagging_temperature, size=n)


class MVSBootstrap:
    """
    Minimal Variance Sampling (default for CPU regression in modern CatBoost).
    Subsample fraction applied, objects selected via importance-weighted sampling.
    For RMSE with uniform sample weights this simplifies to: each object is included
    independently with probability `subsample`, weight = 1/subsample if included.
    (Full MVS uses gradient-adaptive weights; we implement the simplified version
    that matches CatBoost's behaviour on uniform-weight regression.)
    """

    def __init__(self, subsample: float = 0.8):
        self.subsample = subsample

    def sample_weights(self, n: int, rng: np.random.Generator) -> np.ndarray:
        mask = rng.random(n) < self.subsample
        weights = np.zeros(n)
        weights[mask] = 1.0 / self.subsample
        return weights


class NoBootstrap:
    def sample_weights(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return np.ones(n)


# ---------------------------------------------------------------------------
# Ordered boosting helpers
# ---------------------------------------------------------------------------


def _ordered_n_models(n: int) -> int:
    """
    CatBoost maintains ceil(log2(n)) + 1 supporting models instead of n.
    Model j covers the first 2^j examples in the permutation.
    """
    if n <= 1:
        return 1
    return int(np.ceil(np.log2(n))) + 1


def _ordered_model_index(position: int, n_models: int) -> int:
    """
    For an example at `position` in the permutation, which supporting model
    should supply the prediction used to compute its residual?
    CatBoost uses the largest power-of-two boundary that is <= position.
    Model 0 covers prefix of size 1 (2^0), model 1 covers prefix of size 2 (2^1), etc.
    For position 0, there is no history -> use init_value (model index -1 conceptually).
    """
    if position == 0:
        return -1  # no history available
    # largest j such that 2^j <= position
    j = int(np.floor(np.log2(position)))
    return min(j, n_models - 1)


# ---------------------------------------------------------------------------
# Main boosting class
# ---------------------------------------------------------------------------


class MyCatBoost:
    """
    CatBoost-style gradient boosting with:
      - Plain or Ordered boosting
      - MVS or Bayesian or No bootstrap
      - random_strength with NormalWithModelSizeDecrease (default CatBoost behaviour)
      - l2_leaf_reg applied during leaf-value estimation
      - boost_from_average initialisation
      - early stopping on val RMSE

    base_model_fn: callable () -> unfitted sklearn-compatible regressor.
        Must support .fit(X, y, sample_weight=...) and .predict(X).
        For true CatBoost replication use CatBoostRegressor(iterations=1, learning_rate=1.0,
        loss_function="RMSE", bootstrap_type="No", random_strength=0).
    """

    def __init__(
        self,
        base_model_fn,
        n_estimators: int = 1000,
        learning_rate: float = 0.03,
        boosting_type: str = "Plain",          # "Plain" | "Ordered"
        bootstrap_type: str = "MVS",           # "MVS" | "Bayesian" | "No"
        subsample: float = 0.8,                # used when bootstrap_type == "MVS"
        bagging_temperature: float = 1.0,      # used when bootstrap_type == "Bayesian"
        random_strength: float = 1.0,          # multiplier for split-score noise
        random_seed: int = 42,
        verbose: int = 0,
    ):
        self.base_model_fn = base_model_fn
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.boosting_type = boosting_type
        self.random_strength = random_strength
        self.verbose = verbose

        # Bootstrap
        if bootstrap_type == "MVS":
            self.bootstrap = MVSBootstrap(subsample)
        elif bootstrap_type == "Bayesian":
            self.bootstrap = BayesianBootstrap(bagging_temperature)
        else:
            self.bootstrap = NoBootstrap()

        self.rng = np.random.default_rng(random_seed)

        # State set during fit
        self.models = []
        self.init_value = 0.0

    # ------------------------------------------------------------------
    # random_strength noise: NormalWithModelSizeDecrease
    # ------------------------------------------------------------------
    # CatBoost source (catboost/core/tree/score_function.cpp) uses:
    #   noise_std = random_strength * max(1e-12, sum_of_squared_gradients)^0.5
    # But the "ModelSizeDecrease" part decays as 1/sqrt(iteration+1).
    # Net effect per iteration t:
    #   noise ~ N(0, random_strength / sqrt(t + 1))
    # (the gradient-magnitude scaling cancels out when comparing splits within
    #  the same iteration, so for structure-selection the effective noise is
    #  just the decay factor).  We implement exactly this decay.
    # ------------------------------------------------------------------

    def _noise_std(self, iteration: int) -> float:
        return self.random_strength / np.sqrt(iteration + 1)

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, X, y, eval_set, early_stopping_rounds=None):
        X_val, y_val = eval_set
        n = len(y)

        self.init_value = float(y.mean())  # boost_from_average
        pred_val = np.full(len(y_val), self.init_value)

        best_score = np.inf
        best_iter = -1
        patience = 0

        if self.boosting_type == "Plain":
            self._fit_plain(X, y, pred_val, X_val, y_val,
                            early_stopping_rounds, best_score, best_iter, patience)
        elif self.boosting_type == "Ordered":
            self._fit_ordered(X, y, pred_val, X_val, y_val,
                              early_stopping_rounds, best_score, best_iter, patience)
        else:
            raise ValueError(f"Unknown boosting_type: {self.boosting_type}")

        return self

    # ------------------------------------------------------------------
    # Plain boosting
    # ------------------------------------------------------------------

    def _fit_plain(self, X, y, pred_val, X_val, y_val,
                   early_stopping_rounds, best_score, best_iter, patience, fix_borders=True):
        n = len(y)
        pred = np.full(n, self.init_value)
        train_pool = Pool(X, y)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.borders', delete=False) as f:
            self.borders_file = f.name
        print(self.borders_file)
        
        # Create initial pool and extract borders
        if fix_borders:
            train_pool_initial = Pool(X, y)
            train_pool_initial.quantize()
            train_pool_initial.save_quantization_borders(self.borders_file)
        
        # Now use these borders for all iterations
        for it in range(self.n_estimators):
            residual = y - pred
            weights = self.bootstrap.sample_weights(n, self.rng)
            train_pool = Pool(X, residual, weight=weights)
            if fix_borders:
                train_pool.quantize(input_borders=self.borders_file)

            model = self.base_model_fn(noise_std=self._noise_std(it))

            model.fit(train_pool)
            self.models.append(model)

            # update predictions
            update = model.predict(X)
            pred += self.learning_rate * update

            pred_val += self.learning_rate * model.predict(X_val)
            rmse_val = np.sqrt(np.mean((y_val - pred_val) ** 2))

            # early stopping logic
            best_score, best_iter, patience = self._check_early_stop(
                it, rmse_val, best_score, best_iter, patience, early_stopping_rounds, y, pred
            )
            if best_score is None:  # signal to break
                break
        
        # Cleanup borders file
        if self.borders_file and os.path.exists(self.borders_file):
            os.unlink(self.borders_file)
            
        return self

    # ------------------------------------------------------------------
    # Ordered boosting  (log2(n) supporting models approximation)
    # ------------------------------------------------------------------

    def _fit_ordered(self, X, y, pred_val, X_val, y_val,
                     early_stopping_rounds, best_score, best_iter, patience):
        raise NotImplementedError("Ordered boosting is not implemented in this version.")
        # n = len(y)
        # n_supporting = _ordered_n_models(n)

        # # Generate one permutation per boosting iteration (CatBoost resamples each iter)
        # # Supporting predictions: shape (n_supporting, n)
        # # supporting_pred[j][i] = prediction for object i using the model trained on
        # # the first 2^j objects in the current permutation.
        # # We maintain these across iterations; each iteration adds one tree to each
        # # supporting model (same tree structure, different leaf values).

        # # Init all supporting predictions to init_value
        # supporting_pred = np.full((n_supporting, n), self.init_value)

        # for it in range(self.n_estimators):
        #     # Fresh permutation each iteration (matches CatBoost behaviour)
        #     perm = self.rng.permutation(n)
        #     inv_perm = np.empty(n, dtype=int)
        #     inv_perm[perm] = np.arange(n)

        #     # --- compute per-object residuals using ordered predictions ---
        #     # For object at position p in perm, use supporting model
        #     # _ordered_model_index(p) to get its prediction.
        #     residual = np.empty(n)
        #     for p in range(n):
        #         obj_idx = perm[p]
        #         model_j = _ordered_model_index(p, n_supporting)
        #         if model_j < 0:
        #             pred_for_obj = self.init_value
        #         else:
        #             pred_for_obj = supporting_pred[model_j, obj_idx]
        #         residual[obj_idx] = y[obj_idx] - pred_for_obj

        #     # --- bootstrap weights ---
        #     weights = self.bootstrap.sample_weights(n, self.rng)

        #     # --- build ONE tree on the full dataset with these residuals & weights ---
        #     # (CatBoost builds one shared tree structure per iteration)
        #     model = self.base_model_fn()
        #     model.fit(X, residual, sample_weight=weights)
        #     self.models.append(model)

        #     # --- update supporting predictions ---
        #     # Each supporting model j covers prefix [0, 2^j) of perm.
        #     # The leaf values differ per supporting model because they are computed
        #     # only on the objects in that prefix.  We approximate this by:
        #     # retraining leaf values for each supporting model on its prefix.
        #     # In practice CatBoost does this efficiently; we do it explicitly.
        #     tree_preds_full = model.predict(X)  # full-data leaf values

        #     for j in range(n_supporting):
        #         prefix_size = min(2 ** j, n)
        #         prefix_indices = perm[:prefix_size]

        #         # Refit leaf values on prefix only (same structure, different values)
        #         # We approximate by using the ratio of prefix-mean residual to
        #         # full-mean residual per leaf — but that requires leaf assignments.
        #         # Simpler correct approach: refit a new model on the prefix.
        #         if prefix_size >= 2:  # need at least 2 samples
        #             sub_model = self.base_model_fn()
        #             sub_residual = residual[prefix_indices]
        #             sub_weights = weights[prefix_indices]
        #             # only refit if we have non-zero weight
        #             if sub_weights.sum() > 0:
        #                 sub_model.fit(X[prefix_indices], sub_residual,
        #                               sample_weight=sub_weights)
        #                 supporting_pred[j] += self.learning_rate * sub_model.predict(X)
        #             else:
        #                 supporting_pred[j] += self.learning_rate * tree_preds_full
        #         else:
        #             # prefix too small, use full prediction as fallback
        #             supporting_pred[j] += self.learning_rate * tree_preds_full

        #     # --- val prediction uses the LAST (largest) supporting model ---
        #     # At test time CatBoost uses all training data -> equivalent to the
        #     # full-data tree.
        #     pred_val += self.learning_rate * model.predict(X_val)
        #     rmse_val = np.sqrt(np.mean((y_val - pred_val) ** 2))

        #     # For early stopping we need train pred; use the full-data tree pred
        #     # (this is what gets stored in the final model anyway)
        #     # We track a "plain-style" train pred just for logging/early-stop
        #     if it == 0:
        #         _train_pred = np.full(n, self.init_value)
        #     _train_pred += self.learning_rate * tree_preds_full

        #     best_score, best_iter, patience = self._check_early_stop(
        #         it, rmse_val, best_score, best_iter, patience,
        #         early_stopping_rounds, y, _train_pred
        #     )
        #     if best_score is None:
        #         break

        # return self

    # ------------------------------------------------------------------
    # Early-stop helper  (returns updated state or None to signal break)
    # ------------------------------------------------------------------

    def _check_early_stop(self, it, rmse_val, best_score, best_iter, patience,
                          early_stopping_rounds, y, pred):
        if rmse_val < best_score:
            best_score = rmse_val
            best_iter = it
            patience = 0
        else:
            patience += 1

        if self.verbose and it % self.verbose == 0:
            train_rmse = np.sqrt(np.mean((y - pred) ** 2))
            print(f"[{it}] train RMSE={train_rmse:.6f}, val RMSE={rmse_val:.6f}")

        if early_stopping_rounds is not None and patience >= early_stopping_rounds:
            if self.verbose:
                print(
                    f"Early stopping at iter {it}, "
                    f"best iter = {best_iter}, "
                    f"best RMSE = {best_score:.6f}"
                )
            self.models = self.models[: best_iter + 1]
            return None, None, None  # signal break

        return best_score, best_iter, patience

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(self, X):
        pred = np.full(len(X), self.init_value)
        for model in self.models:
            pred += self.learning_rate * model.predict(X)
        return pred
