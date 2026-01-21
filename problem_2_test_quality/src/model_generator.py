import copy
import numpy as np


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

    def __init__(self, base_model, study, n_trials_per_iter=5):
        self.base_model = base_model
        self.study = study
        self.n_trials_per_iter = n_trials_per_iter

    def _sample_params(self, trial):
        grow_policy = trial.suggest_categorical(
            "grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"]
        )

        params = {
            "iterations": 1,
            "learning_rate": 1.0,
            "loss_function": "RMSE",
            "grow_policy": grow_policy,

            "border_count": trial.suggest_int("border_count", 32, 255),
            "feature_border_type": trial.suggest_categorical(
                "feature_border_type",
                ["GreedyLogSum", "Median", "Uniform"]
            ),
            "l2_leaf_reg": trial.suggest_float(
                "l2_leaf_reg", 1e-6, 100.0, log=True
            ),
            "min_data_in_leaf": trial.suggest_int(
                "min_data_in_leaf", 1, 64
            ),
            "random_strength": trial.suggest_float(
                "random_strength", 1e-3, 10.0, log=True
            ),
            "rsm": trial.suggest_float("rsm", 0.3, 1.0),
            "score_function": trial.suggest_categorical(
                "score_function", ["Cosine", "L2"]
            ),
            "verbose": False,
        }

        if grow_policy in ["SymmetricTree", "Depthwise"]:
            params["depth"] = trial.suggest_int("depth", 2, 12)
        else:
            params["max_leaves"] = trial.suggest_int("max_leaves", 8, 64)

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

        return params

    def generate(
        self,
        iteration,
        train_set,
        hp_set,
        pred_hp,
        learning_rate,
    ):
        X_train, y_train = train_set
        X_hp, y_hp = hp_set

        best_model = None
        best_score = np.inf
        study = copy.deepcopy(self.study)

        for _ in range(self.n_trials_per_iter):
            trial = study.ask()
            params = self._sample_params(trial)

            model = copy.deepcopy(self.base_model)
            model.set_params(**params)
            model.fit(X_train, y_train)

            pred_hp_new = pred_hp + learning_rate * model.predict(X_hp)
            rmse_hp = np.sqrt(np.mean((y_hp - pred_hp_new) ** 2))

            study.tell(trial, rmse_hp)

            if rmse_hp < best_score:
                best_score = rmse_hp
                best_model = model

        return best_model

    def update_on_iteration_end(self, iteration, metric_val, patience):
        pass