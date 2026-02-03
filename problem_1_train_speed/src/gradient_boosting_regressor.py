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
        self.init_value = 0.0

    def fit(self, X, y, eval_set, early_stopping_rounds=None):
        X_val, y_val = eval_set

        self.init_value = y.mean()
        pred = np.full(len(y), self.init_value)
        pred_val = np.full(len(y_val), self.init_value)

        best_score = np.inf
        best_iter = -1
        patience = 0

        for it in range(self.n_estimators):
            residual = y - pred

            model = self.model_generator.generate(iteration=it)
            model.fit(X, residual)

            update = model.predict(X)
            pred += self.learning_rate * update
            self.models.append(model)

            pred_val += self.learning_rate * model.predict(X_val)
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
                break

        return self

    def predict(self, X):
        pred = np.full(len(X), self.init_value)
        for model in self.models:
            pred += self.learning_rate * model.predict(X)
        return pred
