import os
import sys

# Putting Talaria as Package Parent
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from E2E_extended.dataset_creator import HousingETL
from E2E_extended.dataset_creator import HousingPipeline

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/vfp1/bts-mbds-data-science-foundations-2019/master/"
HOUSING_PATH = os.path.join("/home/victorsvarmi/Desktop/ADA_2020/bts-advanced-data-analysis-2020/E2E_extended", "housing_data")
HOUSING_URL = DOWNLOAD_ROOT + "sessions/data/housing.tgz"

my_model = "/home/victorsvarmi/Desktop/ADA_2020/bts-advanced-data-analysis-2020/E2E_extended/random_forest.pkl"
my_etl_model = "/home/victorsvarmi/Desktop/ADA_2020/bts-advanced-data-analysis-2020/E2E_extended/housing_etl.pkl"

if __name__ == '__main__':
    houseETL = HousingETL(housing_url=HOUSING_URL, housing_path=HOUSING_PATH)
    numeric, categorical, housing_X_train, housing_y_train = houseETL.train_numeric_categorical_split()

    pipe = HousingPipeline(housing_numeric=numeric, housing_categoric=categorical, X=housing_X_train,
                           y=housing_y_train, my_etl_name=my_etl_model, my_model_name=my_model)
    pkl_model, rf_pipeline = pipe.random_forest_pipeline()

    print(pkl_model)


