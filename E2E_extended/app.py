import pandas as pd
from flask import Flask, jsonify, request
import pickle

# Load our model
model = pickle.load(open('model.pkl', 'rb'))