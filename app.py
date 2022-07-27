from unicodedata import digit
from flask import Flask, render_template,request
import pandas as pd
import numpy as np
import pickle

pipe=pickle.load(open('CarModelSave.pkl','rb'))
car=pd.read_csv('Car.csv')

app=Flask(__name__)

@app.route('/')
def index():
    companies= sorted(car['company'].unique())
    car_models=sorted(car['name'].unique())
    year=sorted(car['year'].unique())
    fuel_type=car['fuel_type'].unique()
    return render_template('index.html', companies=companies, car_models=car_models, years=year, fuel_type=fuel_type)

@app.route('/predict',methods=['POST'])
def predict():
    company=request.form.get('company')
    model= request.form.get('car_model')
    years=request.form.get('year')
    year=int(years)
    fuel=request.form.get('fuel_type')
    drivens=request.form.get('km_driven')
    driven=int(drivens)

    result=pipe.predict(pd.DataFrame([[model, company, year,driven, fuel]],columns=['name','company','year','kms_driven',"fuel_type"]))
    print(result)
    if result:
        return render_template('index.html',label=int(result),cars_model=model)




if __name__=="__main__":
    app.run(debug=True)




