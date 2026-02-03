import numpy as np


class MyCatBoost:
    """
    CatBoost-like wrapper with per-iteration hyperparameters and early stopping
    (uniform weights, eval_set always provided)
    """

    def __init__(
        self,
        model_generator,
        n_estimators=100,
        learning_rate=0.1,
        random_state=42,
        verbose=0,
    ):
        self.model_generator = model_generator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.rng = np.random.default_rng(random_state)

        self.models = []
        self.lrs = []
        self.init_value = 0.0

    def fit(self, X, y, hp_set, eval_set, early_stopping_rounds=None):
        X_hp, y_hp = hp_set
        X_val, y_val = eval_set

        self.init_value = y.mean()
        pred = np.full(len(y), self.init_value)
        pred_hp = np.full(len(y_hp), self.init_value)
        pred_val = np.full(len(y_val), self.init_value)

        best_score = np.inf
        best_iter = -1
        patience = 0

        for it in range(self.n_estimators):
            residual = y - pred

            res = self.model_generator.generate(
                iteration=it,
                train_set=(X, residual),
                hp_set=hp_set,
                pred_hp=pred_hp,
                learning_rate=self.learning_rate,
            )
            if len(res) == 1:
                model = res
                lr = self.learning_rate
            else:
                model, lr = res

            self.models.append(model)
            self.lrs.append(lr)

            pred += lr * model.predict(X)
            pred_hp += lr * model.predict(X_hp)
            pred_val += lr * model.predict(X_val)
            
            rmse_val = np.sqrt(np.mean((y_val - pred_val) ** 2))
            if rmse_val < best_score:
                best_score = rmse_val
                best_iter = it
                patience = 0
            else:
                patience += 1

            self.model_generator.update_on_iteration_end(
                iteration=it, metric_val=rmse_val, patience=patience
            )

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
                self.lrs = self.lrs[: best_iter + 1]
                break

        return self

    def predict(self, X):
        pred = np.full(len(X), self.init_value)
        for model, lr in zip(self.models, self.lrs):
            pred += lr * model.predict(X)
        return pred
