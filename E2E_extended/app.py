import pandas as pd
from flask import Flask, jsonify, request
import joblib

import os
import sys

# Putting Talaria as Package Parent
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from E2E_extended.dataset_creator import DirectoryFinder

# Setting paths
directory_finder = DirectoryFinder()
gitroot = directory_finder.get_root_path()

pickle_file = os.path.join(gitroot, "E2E_extended/random_forest.pkl")

# Load our model
model = joblib.load(pickle_file)

# Our app
app = Flask(__name__)

# Our app route
@app.route('/', methods=['POST'])

def predict():
    # get data
    data = request.get_json(force=True)

    # convert data into dataframe
    data.update((x, [y]) for x, y in data.items())
    data_df = pd.DataFrame.from_dict(data)

    # predictions
    result = model.predict(data_df)

    # send back to browser
    output = {'results': int(result[0])}

    # return data
    return jsonify(results=output)

if __name__ == '__main__':
    app.run(port = 5000, debug=True)
