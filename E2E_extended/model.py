import os

from E2E_extended.housing_dataset import HousingETL
from E2E_extended.housing_dataset import HousingPipeline

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/vfp1/bts-mbds-data-science-foundations-2019/master/"
HOUSING_PATH = os.path.join("/home/victorsvarmi/Desktop/ADA_2020/bts-advanced-data-analysis-2020/E2E_extended", "housing_data")
HOUSING_URL = DOWNLOAD_ROOT + "sessions/data/housing.tgz"

if __name__ == '__main__':
    houseETL = HousingETL(housing_url=HOUSING_URL, housing_path=HOUSING_PATH)
    numeric, categorical, housing_X_train = houseETL.train_numeric_categorical_split()

    pipe = HousingPipeline(housing_numeric=numeric, housing_categoric=categorical, housing_full=housing_X_train)
    ready_dataset = pipe.full_pipeline()
    print(ready_dataset)

