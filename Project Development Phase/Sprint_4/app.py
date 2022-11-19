#importing required libraries

from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn import metrics 
import warnings
import requests
import pickle
from feature import FeatureExtraction
import json

warnings.filterwarnings('ignore')

API_KEY = "Mj2K6a-kTKcO-cpBW96DqfXayq46Sn945WG00YKiJE7W"
token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey":
 API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
mltoken = token_response.json()["access_token"]

header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":

        url = request.form["url"]
        obj = FeatureExtraction(url)
        x = np.array(obj.getFeaturesList()).reshape(1,30) 

        # Feature list from input URL
        payload_scoring = {"input_data": [{"fields": [obj.getFeaturesList()], "values": x.tolist()}]}

        response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/5efc83c1-aad3-41be-b570-ca2c04fee541/predictions?version=2022-11-18', json=payload_scoring,
        headers={'Authorization': 'Bearer ' + mltoken})

        _pred=response_scoring.json()
        output=_pred['predictions'][0]['values'][0][1][0]

        # Response from IBM Cloud Model rendered to UI from Flask
        return render_template('index.html',xx =round(int(output),2),url=url )

    return render_template("index.html", xx =-1)


if __name__ == "__main__":
    app.run(debug=True)