import pandas as pd
from flask import Flask, jsonify, request, render_template
from flask_caching import Cache
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
cache = Cache(config={'CACHE_TYPE': 'simple'})
app = Flask(__name__, template_folder=os.path.join(gitroot, "E2E_extended"))

cache.init_app(app)
# Our app route
@app.route('/predict', methods=['GET', 'POST'])
@cache.cached(timeout=50)

def predict(result=None):
    if request.method == 'POST':
        df = pd.read_csv(request.files.get('file'))

        results = model.predict(df)
        print(results)

        return render_template('predict.html', dataset=df, result=results)

    return render_template('predict.html')

if __name__ == '__main__':
    app.run(port = 5000, debug=True)
