import os
import tarfile
from urllib import request
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
import joblib


class HousingETL(object):

    def __init__(self, housing_url, housing_path):
        self.housing_url = housing_url
        self.housing_path = housing_path

    def fetch_housing_data(self):
        """
        Downloads data from Github and saves it to local
        :return: None
        """

        if not os.path.isdir(self.housing_path):
            os.makedirs(self.housing_path)

        if not os.path.exists(os.path.join(self.housing_path, "housing.csv")):
            tgz_path = os.path.join(self.housing_path, "housing.tgz")
            request.urlretrieve(self.housing_url, tgz_path)
            housing_tgz = tarfile.open(tgz_path)
            housing_tgz.extractall(path=self.housing_path)
            housing_tgz.close()

    def load_housing_data(self):
        """
        A function to load Housing Pandas DataFrame
        :return: a Pandas DataFrame
        """
        csv_path = os.path.join(self.housing_path, "housing.csv")
        return pd.read_csv(csv_path)

    def stratified_sampling(self):
        """
        Loads the Pandas DataFrame and creates a stratified sampling policy
        :return: the Train and Test set, stratified
        """

        global strat_test_set
        global strat_train_set

        housing = self.load_housing_data()

        housing["income_cat"] = pd.cut(housing["median_income"],
                                       bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                       labels=[1, 2, 3, 4, 5])


        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

        for train_index, test_index in split.split(housing, housing["income_cat"]):
            strat_train_set = housing.loc[train_index]
            strat_test_set = housing.loc[test_index]

        return strat_train_set, strat_test_set

    def X_y_split(self):
        """
        Splits the stratified dataset into train and test (taking labels out)

        :return: housing_X_train, housing_y_train, housing_X_test, housing_y_test
        """
        train, test = self.stratified_sampling()

        housing_X_train = train.drop("median_house_value", axis=1)  # drop labels for training set
        housing_y_train = train["median_house_value"].copy()

        housing_X_test = test.drop("median_house_value", axis=1)
        housing_y_test = test["median_house_value"].copy()

        return housing_X_train, housing_y_train, housing_X_test, housing_y_test

    def train_numeric_categorical_split(self):
        """
        Splits numerical and categorical attributes to pass to the Pipeline
        :return: housing_num, housing_cat, housing_X_train
        """
        housing_X_train, housing_y_train, housing_X_test, housing_y_test = self.X_y_split()

        housing_num = housing_X_train.drop("ocean_proximity", axis=1)

        housing_cat = housing_X_train[["ocean_proximity"]]

        return housing_num, housing_cat, housing_X_train, housing_y_train

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True, rooms_ix=3, bedrooms_ix=4, population_ix=5, households_ix=6):
        self.add_bedrooms_per_room = add_bedrooms_per_room
        self.rooms_ix = rooms_ix
        self.bedrooms_ix = bedrooms_ix
        self.population_ix = population_ix
        self.households_ix = households_ix

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X):
        rooms_per_household = X[:, self.rooms_ix] / X[:, self.households_ix]
        population_per_household = X[:, self.population_ix] / X[:, self.households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, self.bedrooms_ix] / X[:, self.rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

class HousingPipeline(object):
    def __init__(self, housing_numeric, housing_categoric, X, y, my_etl_name, my_model_name):
        """
        The full ETL and algorithm pipeline
        :param housing_numeric: the numeric part of the dataset
        :param housing_categoric: the categoric part of the dataset
        :param X: the full X values
        :param y: the full y values
        :param my_etl_name: the name of the etl pipeline to save
        :param my_model_name: the name we want for our model file
        """
        self.housing_numeric = housing_numeric
        self.housing_categoric = housing_categoric
        self.X = X
        self.y = y
        self.my_etl_name = my_etl_name
        self.my_model_name = my_model_name

    def etl_pipeline(self):
        """
        Returns a dataset to be ingested into an ML algorithm
        :return:
        """

        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('attribs_adder', CombinedAttributesAdder()),
            ('std_scaler', StandardScaler()),
        ])

        num_attribs = list(self.housing_numeric)
        cat_attribs = ["ocean_proximity"]

        full_etl_pipeline = ColumnTransformer([
            ("num", num_pipeline, num_attribs),
            ("cat", OneHotEncoder(), cat_attribs),
        ])

        housing_prepared = full_etl_pipeline.fit_transform(self.X)

        my_etl_model = joblib.dump(full_etl_pipeline, self.my_etl_name, 'wb')

        return housing_prepared, full_etl_pipeline, my_etl_model

    def random_forest_pipeline(self):
        housing_prepared, etl_pipeline, etl_pkl = self.etl_pipeline()

        random_forest_pipe = Pipeline([
            ("ETL", etl_pipeline),
            ("RF_regressor", RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto',
                                                   max_leaf_nodes=None, min_impurity_decrease=0.0,
                                                   min_impurity_split=None, min_samples_leaf=1,
                                                   min_samples_split=2, min_weight_fraction_leaf=0.0,
                                                   n_estimators=100, n_jobs=None, oob_score=False,
                                                   random_state=42, verbose=0, warm_start=False))
        ])

        random_forest_pipe.fit(self.X, self.y)

        my_model = joblib.dump(random_forest_pipe, self.my_model_name)

        return my_model, random_forest_pipe

class HousingInference(object):

    def __init__(self, number_of_samples, etl_model, housing_X_train, housing_y_train):
        self.number_of_samples = number_of_samples
        self.etl_model = etl_model
        self.housing_X_train = housing_X_train
        self.housing_y_train = housing_y_train

    def create_inference_dataset(self):
        etl_pipe = joblib.load(self.etl_model)

        created_x = self.housing_X_train.iloc[:self.number_of_samples]
        created_y = self.housing_y_train.iloc[:self.number_of_samples]

        created_prepared = etl_pipe.transform(created_x)

        return created_prepared, created_y









