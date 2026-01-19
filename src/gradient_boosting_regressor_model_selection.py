import numpy as np

class MyCatBoost:
    """
    CatBoost-like wrapper with per-iteration hyperparameters and early stopping
    """

    def __init__(
        self,
        model_generator,
        n_estimators=100,
        n_tries_per_estimator=50,
        learning_rate=0.1,
        random_state=42,
        verbose=0
    ):
        self.model_generator = model_generator
        self.n_estimators = n_estimators
        self.n_tries_per_estimator = n_tries_per_estimator
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.rng = np.random.default_rng(random_state)

        self.models = []
        self.init_value = 0.0

    # =========================
    # Fit
    # =========================
    def fit(self, X, y, sample_weight=None, eval_set=None, early_stopping_rounds=None, hp_search_set=None):
        n_samples = X.shape[0]
        if sample_weight is None:
            sample_weight = np.ones(n_samples)

        self.init_value = np.average(y, weights=sample_weight)
        pred = np.full(n_samples, self.init_value)

        if eval_set is not None:
            X_val, y_val, w_val = eval_set if len(eval_set) == 3 else (eval_set[0], eval_set[1], None)
            if w_val is None:
                w_val = np.ones(X_val.shape[0])
            pred_val = np.full(X_val.shape[0], self.init_value)
            best_score = np.inf
            best_iter = -1
            patience = 0

        for it in range(self.n_estimators):
            # 1) вычисляем остатки на тренировке
            residual = y - pred

            # 2) вычисляем остатки на hp_search_set, если он задан
            if hp_search_set is not None:
                X_hp, y_hp = hp_search_set
                w_hp = None
                residual_hp = y_hp - self.predict(X_hp)  # актуальные остатки
                dynamic_hp_search_set = (X_hp, residual_hp, w_hp)
                model = self.model_generator.generate(iteration=it, train_set = (X, residual), hp_search_set=dynamic_hp_search_set)
            else:
                model = self.model_generator.generate(iteration=it)

            # 3) обучаем модель на остатках
            model.fit(X, residual, sample_weight=sample_weight)
            update = model.predict(X)
            pred += self.learning_rate * update
            self.models.append(model)

            # --- ранняя остановка и логирование ---
            if eval_set is not None:
                val_update = model.predict(X_val)
                pred_val += self.learning_rate * val_update
                rmse_val = np.sqrt(np.average((y_val - pred_val) ** 2, weights=w_val))

                if rmse_val < best_score:
                    best_score = rmse_val
                    best_iter = it
                    patience = 0
                else:
                    self.models.pop()
                    patience += 1

                self.model_generator.update_on_iteration_end(iteration=it, metric_val=rmse_val, patience=patience)

                if early_stopping_rounds is not None and patience >= early_stopping_rounds:
                    if self.verbose:
                        print(
                            f"Early stopping at iter {it}, "
                            f"best iter = {best_iter}, "
                            f"best RMSE = {best_score:.6f}"
                        )
                    self.models = self.models[: best_iter + 1]
                    break
            else:
                self.model_generator.update_on_iteration_end(iteration=it, metric_val=0, patience=0)

            if self.verbose and it % self.verbose == 0:
                train_rmse = np.sqrt(np.average((y - pred) ** 2, weights=sample_weight))
                if eval_set is not None:
                    print(f"[{it}] train RMSE={train_rmse:.6f}, val RMSE={rmse_val:.6f}")
                else:
                    print(f"[{it}] train RMSE={train_rmse:.6f}")

    # =========================
    # Predict
    # =========================
    def predict(self, X):
        pred = np.full(X.shape[0], self.init_value)
        for model in self.models:
            pred += self.learning_rate * model.predict(X)
        return pred
