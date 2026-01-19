import copy
import numpy as np
import copy
import optuna


class BaseModelGenerator:
    def generate(self):
        raise NotImplementedError


class CopyModelGenerator(BaseModelGenerator):
    
    def __init__(self, base_model):
        self.base_model = base_model

    def generate(self, iteration):
        return copy.deepcopy(self.base_model)
    
    def update_on_iteration_end(self, iteration, metric_val, patience):
        pass


class OptunaModelGenerator(BaseModelGenerator):

    def __init__(self, base_model, study):
        self.base_model = base_model
        self.starting_study = copy.deepcopy(study)
        self.iteration_to_trial = {}

    def refresh_study(self):
        self.study = self.starting_study

    def generate(self, iteration, train_set, hp_search_set=None):
        """
        hp_search_set: (X_hp, residual_hp, w_hp) для HPO на текущей итерации
        """
        X_train, res_train = train_set
        if hp_search_set is not None:
            X_hp, residual_hp, w_hp = hp_search_set
        else:
            X_hp, residual_hp, w_hp = None, None, None

        if X_hp is None:
            # fallback: просто копируем базовую модель
            return copy.deepcopy(self.base_model)

        def objective(trial):
            
            grow_policy = trial.suggest_categorical(
                "grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"]
            )

            params = {
                "iterations": 1,
                "learning_rate": 1.0,
                "loss_function": "RMSE",
                "grow_policy": grow_policy,

                # Splits
                "border_count": trial.suggest_int("border_count", 32, 255),
                "feature_border_type": trial.suggest_categorical(
                    "feature_border_type",
                    ["GreedyLogSum", "Median", "Uniform"]
                ),

                # Regularization
                "l2_leaf_reg": trial.suggest_float(
                    "l2_leaf_reg", 1e-6, 100.0, log=True
                ),
                "min_data_in_leaf": trial.suggest_int(
                    "min_data_in_leaf", 1, 64
                ),

                # Stochasticity
                "random_strength": trial.suggest_float(
                    "random_strength", 1e-3, 10.0, log=True
                ),
                "rsm": trial.suggest_float("rsm", 0.3, 1.0),

                # Score noise
                "score_function": trial.suggest_categorical(
                    "score_function", ["Cosine", "L2"]
                ),

                "verbose": False,
            }

            # Depth vs leaves
            if grow_policy in ["SymmetricTree", "Depthwise"]:
                params["depth"] = trial.suggest_int("depth", 2, 12)
            else:
                params["max_leaves"] = trial.suggest_int("max_leaves", 8, 64)

            # Bootstrap
            bootstrap_type = trial.suggest_categorical(
                "bootstrap_type", ["Bayesian", "Bernoulli", "No"]
            )
            params["bootstrap_type"] = bootstrap_type

            if bootstrap_type == "Bernoulli":
                params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)

            elif bootstrap_type == "Bayesian":
                params["bagging_temperature"] = trial.suggest_float(
                    "bagging_temperature", 0.0, 10.0
                )

            model = copy.deepcopy(self.base_model)
            model.set_params(**params)
            model.fit(X_train, res_train)
            pred = model.predict(X_hp)
            return np.sqrt(np.average((residual_hp - pred) ** 2))

        optuna.logging.set_verbosity(optuna.logging.WARNING)  # или ERROR, чтобы совсем ничего

        study = copy.deepcopy(self.starting_study)
        study.optimize(objective, n_trials=50)
        best_model = copy.deepcopy(self.base_model)
        best_model.set_params(**study.best_params)
        return best_model


    def update_on_iteration_end(self, iteration, metric_val, patience):
        pass
