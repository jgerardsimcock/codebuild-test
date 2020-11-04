from sklearn.ensemble import IsolationForest
from Dataset import Dataset
from utils import MCC_at_threshhold
import lightgbm as lgb
import numpy as np
import pandas as pd
from collections import defaultdict
import pickle
import logging
import s3fs

from sklearn.metrics import (
    recall_score,
    f1_score,
    accuracy_score,
    precision_score,
    confusion_matrix,
)


class Outlier_Model:
    """
    An isolation forest model for identifying numerical outliers.

    Class Vars:
        model: (sklean model) the isolation forest model
        features: (list) a list of feature names
        trained: (bool) a Boolean representing whether or not the model has been trained

    Class Methods:
        init: initialize the model object
        fit: fit the model object on a Dataset object
        predict: identify the outliers
        fit_predict: fit the model and identify outliers
        explain_precictions: generate an outlier classifier to determine feature importances
    """

    def __init__(self, contamination=0.05):
        """
        Initialize a model instance.

        Args:
            contamintation: (float) the percentage of the data which is contaminated, 5% is recommended

        Returns
            None

        """
        self.model = IsolationForest(
            n_estimators=105, contamination=0.05, max_samples=256, max_features=0.9
        )
        self.features = []
        self.trained = False

    def fit(self, dataset):
        """
        Fit the model on a Dataset

        Args:
            dataset: (Dataset) the training data, assumed to be in the form:
                USUBJID| Site | Time | Feature1 | Feature2 | Feature3 | ... | FeatureN
        Returns:
            None
        """
        if not isinstance(dataset, Dataset):
            logging.error("Data must be in Dataset format")
            raise ValueError
        if hasattr(dataset, "outlier_ready"):
            if dataset.outlier_ready:
                raw_data = dataset.dataset
                train_data = raw_data.drop(columns=["USUBJID", "Site", "Time"])
                
                self.features = list(train_data.columns)
                self.model = self.model.fit(train_data)
                self.trained = True
            else:
                logging.error(
                    "Data must be preprocessed with align_numerical_table method"
                )
        else:
            logging.error("Data must be preprocessed with align_numerical_table method")

    def predict(self, dataset):
        """
        Identify outliers using the trained model.

        Args:
            dataset: (Dataset) the numerical data, assumed to be in the form:
                USUBJID| Site | Time | Feature1 | Feature2 | Feature3 | ... | FeatureN
        Returns:
            a Dataset of the form:
                USUBJID| Site | Time | Feature1 | Feature2 | Feature3 | ... | FeatureN | Outlier
            where Outlier is True if the row is an outlier and False otherwise
        """
        if not isinstance(dataset, Dataset):
            logging.error("Data must be in Dataset format")
            raise ValueError

        if self.trained:
            raw_data = dataset.dataset
            pred_data = raw_data[self.features]
            raw_pred = self.model.predict(pred_data)
            raw_bool = raw_pred == -1
            out_data = raw_data.copy()
            out_data["Outlier"] = raw_bool
            return Dataset(dataset=out_data, params={"outlier_scored": True})

        else:
            logging.error("Model must be trained before prediction")
            raise ValueError

    def fit_predict(self, dataset):
        """
        Fit the model and identify outliers using the trained model.

        Args:
            dataset: (Dataset) the training data, assumed to be in the form:
                USUBJID| Site | Time | Feature1 | Feature2 | Feature3 | ... | FeatureN
        Returns:
            a Dataset of the form:
                USUBJID| Site | Time | Feature1 | Feature2 | Feature3 | ... | FeatureN | Outlier
            where Outlier is True if the row is an outlier and False otherwise
        """
        if "Outlier" in dataset.dataset.columns:
            logging.error("Dataset has outlier scores")
            raise Warning

        self.fit(dataset)
        return self.predict(dataset)

    def explain_predictions(self, dataset, num_tests=3):
        """
        An engine for explaining the outlier predictions.

        Args:
            dataset: (Dataset) the training data, assumed to be in the form:
                USUBJID| Site | Time | Feature1 | Feature2 | Feature3 | ... | FeatureN
            num_tests: (int) the number of features to use in the explanations

        Returns
            a pandas DataFrame of outlier scores and explanations
        """
        labeled_dataset = self.fit_predict(dataset)
        labeled_data = labeled_dataset.dataset
        num_rows = labeled_data.shape[0]
        split_bool = np.random.rand(num_rows) < 0.5

        X_data = labeled_dataset.dataset[self.features]
        X_train = X_data.iloc[split_bool]
        X_test = X_data.iloc[~split_bool]

        Y_data = labeled_dataset.dataset["Outlier"]
        Y_train = Y_data.iloc[split_bool]
        Y_test = Y_data.iloc[~split_bool]

        explan_model = lgb.LGBMRegressor(
            boosting_type="gbdt",
            num_leaves=12,
            learning_rate=0.05,
            n_estimators=125,
            subsample_for_bin=120,
            reg_alpha=0.05,
            reg_lambda=0.06,
        )
        explan_model = explan_model.fit(
            X_train,
            Y_train,
            eval_set=(X_test, Y_test),
            early_stopping_rounds=10,
            verbose=False,
        )

        SHAP_scores = explan_model.predict(X_data, pred_contrib=True)

        out = []
        for idx, row in enumerate(SHAP_scores):
            features_w_scores = list(zip(self.features, row))
            sorted_features_w_scores = sorted(features_w_scores, key=lambda x: x[1])

            top_safe_with_scores = sorted_features_w_scores[:num_tests]
            top_risky_with_scores = sorted_features_w_scores[-num_tests:]

            top_safe = list(map(lambda x: x[0], top_safe_with_scores))
            top_risky = list(map(lambda x: x[0], top_risky_with_scores))

            top_safe_dict = {
                f"#{idx+1} Test decreaing Outlierness": test
                for idx, test in enumerate(top_safe)
            }
            top_risky_dict = {
                f"#{idx+1} Test increasing Outlierness": test
                for idx, test in enumerate(top_risky)
            }

            joined_dict = top_safe_dict
            joined_dict.update(top_risky_dict)
            joined_dict["USUBJID"] = dataset.dataset.iloc[idx]["USUBJID"]
            joined_dict["Site"] = dataset.dataset.iloc[idx]["Site"]
            joined_dict["Time"] = dataset.dataset.iloc[idx]["Time"]
            out.append(joined_dict)

        out = pd.DataFrame(out)
        return out


class ORM:
    """
    An Operational Risk Model (ORM) uses current KRI scores to predict whether or not active sites will remain active.

    Class Vars:
        trained: (bool) whether the model has been trained
        model: (LightGBM Reggressor) the model
        features: (list) the list of model features
        MCC_thesh: (float) the binary classification theshhold
        metrics: (dict) the metrics by which to judge model performance, keys are test names, values are functions for computing the metric
    """

    def __init__(self, KRIs, params={"timestep": "30d"}):
        self.params = params
        self.trained = False
        self.model = lgb.LGBMRegressor(
            boosting_type="gbdt",
            num_leaves=25,
            learning_rate=0.1,
            n_estimators=75,
            subsample_for_bin=50,
            reg_alpha=0.01,
            reg_lambda=0.02,
            random_state=42,
        )

        self.features = KRIs
        self.MCC_thresh = None
        self.metrics = {
            "Precision": precision_score,
            "Recall": recall_score,
            "F1": f1_score,
            "Accuracy": accuracy_score,
        }

    def check_fit(self):
        """
        Check to see if the model had been trained.

        Args:
            None
        Returns:
            None
        """
        if not self.trained:
            logging.error("Model must be trained first")
            raise ValueError

    def check_tune(self):
        """
        Check to see if the model had been tuned.

        Args:
            None
        Returns:
            None
        """
        if not self.MCC_thresh:
            logging.error("Model must be tuned first")
            raise ValueError

    def fit(self, KRI_train, target_train, KRI_test, target_test, params={}):
        """
        Fit the model

        Args:
            KRI_train (pd.DataFrame) the training KRI data
            target_train (pd.DataFrame) the training target data
            KRI_test (pd.DataFrame) the testing KRI data
            target_test (pd.DataFrame) the testing target data
        Returns:
            None
        """
        self.model = self.model.fit(
            KRI_train[self.features],
            target_train,
            eval_set=(KRI_test[self.features], target_test),
            early_stopping_rounds=10,
            verbose=False,
        )
        self.trained = True

    def predict(self, KRI_data, params={}):
        """
        Use the fitted model to predict.

        Args:
            KRI_data (pd.DataFrame) the input KRI data
        Returns:
            a pandas DataFrame of model predictions
        """
        self.check_fit()

        if "KRIs" in params:
            KRIs = params["KRIs"]
        else:
            KRIs = self.features

        return self.model.predict(KRI_data[KRIs])

    def explain_predictions(self, KRI_data, num_KRIs=5, params={}):
        """
        Use SHAP scores to explain the model outputs:

        Args:
            KRI_data: (pd.DataFrame) the input KRI data
            num_KRIs: (int) the number of KRIs to use in the model explanation, default is 5

        Returns:
            a pandas DataFrame explaining model predictions
        """
        self.check_fit()
        SHAP_scores = self.model.predict(KRI_data[self.features], pred_contrib=True)

        out = []
        for idx, row in enumerate(SHAP_scores):
            scores = -row
            scores = scores / np.linalg.norm(scores)

            features_w_scores = list(zip(self.features, scores))
            sorted_features_w_scores = sorted(features_w_scores, key=lambda x: x[1])

            top_safe_with_scores = sorted_features_w_scores[:num_KRIs]
            top_risky_with_scores = sorted_features_w_scores[-num_KRIs:]
            top_risky_with_scores = top_risky_with_scores[::-1]

            top_safe_KRI_dict = {
                f"#{idx+1} KRI decreasing Risk": KRI
                for idx, (KRI, score) in enumerate(top_safe_with_scores)
            }
            top_safe_scores_dict = {
                f"#{idx+1} KRI decreasing Risk Score": score
                for idx, (KRI, score) in enumerate(top_safe_with_scores)
            }

            top_risky_KRI_dict = {
                f"#{idx+1} KRI increasing Risk": KRI
                for idx, (KRI, score) in enumerate(top_risky_with_scores)
            }
            top_risky_scores_dict = {
                f"#{idx+1} KRI increasing Risk Score": score
                for idx, (KRI, score) in enumerate(top_risky_with_scores)
            }

            joined_dict = top_safe_KRI_dict
            joined_dict.update(top_safe_scores_dict)
            joined_dict.update(top_risky_KRI_dict)
            joined_dict.update(top_risky_scores_dict)

            joined_dict["Site"] = KRI_data.iloc[idx]["Site"]
            joined_dict["Start_Time"] = KRI_data.iloc[idx]["Start_Time"]
            out.append(joined_dict)

        out = pd.DataFrame(out)
        return out

    def validate(self, KRI_data, target_data, params={}):
        """
        Validate the performance of the model on test data.

        Args:
            KRI_data: (pd.DataFrame) the input KRI data
            target_data: (pd.Series) the target ground truth data

        Returns:
            3 pandas dataframes, the first is the aggregate perfromance statistics drawn from self.metrics, the last two returns are the confusion matrices for the ORM and the ZKM, respectively
        """
        self.check_fit()
        self.check_tune()
        pred = self.predict(KRI_data)
        pred_bool = 1 * (pred > self.MCC_thresh)

        valid_scores = {}
        confusion_matrices = {}

        valid_scores["ORM"] = defaultdict(float)
        for score, function in self.metrics.items():
            valid_scores["ORM"][score] = function(pred_bool, target_data)

        raw_ORM_conf_mat = confusion_matrix(1 - pred_bool, 1 - target_data)
        ORM_conf = pd.DataFrame(raw_ORM_conf_mat)
        ORM_conf = ORM_conf.rename(columns={0: "Low Risk", 1: "High Risk"})
        ORM_conf = ORM_conf.rename(
            {0: "ORM Predicts Low Risk", 1: "ORM Predicts High Risk"}
        )
        confusion_matrices["ORM"] = ORM_conf

        # Compute the expected performance of the optimal binomial model
        mean_survival_rate = target_data.values.mean()
        valid_scores["ZKM"] = defaultdict(float)
        raw_ZKM_conf_mat = np.zeros((2, 2))

        if "rand_iter" in params:
            rand_iter = params["rand_iter"]
        else:
            rand_iter = 100

        for _ in range(rand_iter):
            rand_nums = np.random.rand(target_data.shape[0])
            rand_bool = rand_nums <= mean_survival_rate
            for score, function in self.metrics.items():
                valid_scores["ZKM"][score] += function(rand_bool, target_data)
            raw_ZKM_conf_mat += confusion_matrix(1 - rand_bool, 1 - target_data)

        for score in self.metrics:
            valid_scores["ZKM"][score] = valid_scores["ZKM"][score] / rand_iter

        ZKM_conf = pd.DataFrame(raw_ZKM_conf_mat // rand_iter).astype("int32")
        ZKM_conf = ZKM_conf.rename(columns={0: "Low Risk", 1: "High Risk"})
        ZKM_conf = ZKM_conf.rename(
            {0: "ZKM Predicts Low Risk", 1: "ZKM Predicts High Risk"}
        )
        confusion_matrices["ZKM"] = ZKM_conf

        valid_scores = pd.DataFrame(valid_scores)

        return valid_scores, confusion_matrices

    def KRI_ranking(self, num_KRIs=10):
        """
        Return the most important KRIs.

        Args:
            num_KRIs: (int) the number of top features to return
        Returns:
            a list of most important model features
        """
        self.check_fit()
        scored_features = list(zip(self.model.feature_importances_, self.features))
        ranked_features = sorted(scored_features, reverse=True)
        top_features_w_scores = ranked_features[:num_KRIs]
        top_features = list(map(lambda x: x[1], top_features_w_scores))
        return top_features

    def tune_threshholds(self, KRI_data, target_data):
        """
        Tune the MCC_theshhold variable based on best performance.

        Args:
            KRI_data: (pd.DataFrame) the input KRI data
            target_data: (pd.Series) the target ground truth data

        Returns:
            None
        """
        self.check_fit()
        y_pred = self.predict(KRI_data)
        MCC_scores = [
            (MCC_at_threshhold(target_data, y_pred, thresh), thresh,)
            for thresh in np.arange(0.5, 1, 0.01)
        ]
        best_MCC, best_MCC_thesh = max(MCC_scores)
        self.MCC_thresh = best_MCC_thesh

    def score_risk(self, KRI_data, params={"explain": False}):
        """
        Use a fitted model to score risk.

        Args:
            KRI_data: (pd.DataFrame) the input KRI data

        Returns:
            a pandas DataFrame of risk scores and risk classes, sorted in decreasing order by risk score
        """
        self.check_fit()
        self.check_tune()

        out = KRI_data["Site"].to_frame()
        out["Risk_Score"] = 1 - self.predict(KRI_data, params=params)
        out["Risk_Class"] = out["Risk_Score"] < 1 - self.MCC_thresh
        out["Risk_Class"] = out["Risk_Class"].replace(
            {True: "Low Risk", False: "High Risk"}
        )

        if params["explain"]:
            explanations = self.explain_predictions(KRI_data)
            explanations = explanations.drop(columns=["Start_Time"])
            out = out.merge(explanations, on="Site")

        out = out.sort_values(by=["Risk_Score"], ascending=False)
        return out

    def save_model(self, path="", name=""):
        """
        Save the model to a pkl file.

        Args:
            path: (str) the path to the saved models directory
            name: (str) the name of the model

        Returns:
            None
        """
        if path and name:
            with open(f"{path}/{name}.pkl", "wb") as f:
                pickle.dump(self, f)
        else:
            logging.error("Must provide path and name of model")
            raise FileNotFoundError

    def load_model(self, path=""):
        """
        Load a model from a pkl file.

        Args:
            path: (str) the path to the saved model

        Returns:
            None
        """
        if path:
            with open(path, "rb") as f:
                loaded_model = pickle.load(f)
            for key, value in loaded_model.__dict__.items():
                setattr(self, key, value)
        else:
            logging.error("Must provide path to model")
            raise FileNotFoundError

    #called by Infer function
    def load_model_s3(self, bucket="", model_name=""):
        '''
        Loads pkl file from s3 bucket into model instance
        
        Args:
            bucket: (str) bucket name
            model_name: (str) model file name
        
        Returns:
            None
        '''
        
        s3 = s3fs.S3FileSystem(anon=True)
        try:
            
            f = s3.open(f"{bucket}/{model_name}")
            logging.info(f'Loading {model_name} from {bucket}')
            loaded_model = pickle.load(f)
            for key, value in loaded_model.__dict__.items():
                setattr(self, key, value)
            
        except:
            logging.error(f'Unable to load model from {bucket}')
            raise FileNotFoundError
       

    #called by inder 
    def save_model_s3(self, bucket="", model_name=""):
        '''
        Saved model as pkl file to s3 bucket
        
        Args:
            model: (pkl) instance of model pickle file
            bucket: (str) bucket name
            model_name: (str) model name
        
        Returns:
            None
        '''
        s3 = s3fs.S3FileSystem(anon=True)
        try:
            with s3.open(f"{bucket}/{model_name}.pkl",'wb') as f:
                pickle.dump(self, f)
            logging.info(f'Saving {model_name}.pkl to {bucket}')
        except:
            logging.error(f'Model not saved to {bucket}/{model_name}')
            