import pandas as pd
import numpy as np
import pickle
from flask import Flask,request,app,jsonify,url_for, render_template
import json 

app = Flask(__name__)
#load the model
regmodel = pickle.load(open('regmodel1.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/ct', methods = ['Posts'])
def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values()).reshape(1,-1)))
    new_data = scaler.transform(np.array(list(data.values()).reshape(1,-1)))
    
    output = regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

if __name__ == '__main__':
    app.run(debug=True)