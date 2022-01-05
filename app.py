from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

model = pickle.load(open("drug.pkl", "rb"))
app = Flask(__name__)


@app.route('/')
def man():
    return render_template('home.html')


@app.route('/predict/', methods=['POST'])
def home():
    Age = request.form['Age']
    Gender = request.form['Gender']
    Education = request.form['Education']
    Country = request.form['Country']
    Ethnicity = request.form['Ethnicity']
    Nscore = request.form['Nscore']
    Escore = request.form['Escore']
    Oscore = request.form['Oscore']
    Ascore = request.form['Ascore']
    Cscore = request.form['Cscore']
    Impulsive = request.form['Impulsive']
    SS = request.form['SS']
    Alcohol = request.form['Alcohol']
    Amphet = request.form['Amphet']
    Amyl = request.form['Amyl']
    Benzos = request.form['Benzos']
    Caff = request.form['Caff']
    Choc = request.form['Choc']
    Coke = request.form['Coke']
    Crack = request.form['Crack']
    Ecstasy = request.form['Ecstasy']
    Heroin = request.form['Heroin']
    Ketamine = request.form['Ketamine']
    Legalh = request.form['Legalh']
    LSD = request.form['LSD']
    Meth = request.form['Meth']
    Mushrooms = request.form['Mushrooms']
    Nicotine = request.form['Nicotine']
    Semer = request.form['Semer']
    VSA = request.form['VSA']
    liste =np.array([[Age, Gender, Education, Country, Ethnicity, Nscore, Escore, Oscore, Ascore, Cscore, Impulsive, SS, Alcohol,
             Amphet, Amyl, Benzos, Caff, Choc, Coke, Crack, Ecstasy, Heroin, Ketamine, Legalh, LSD, Meth, Mushrooms,
             Nicotine, Semer, VSA]])
    #newdf = pd.DataFrame(
        #columns=['Age', 'Gender', 'Education', 'Country', 'Ethnicity', 'Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore',
                 #'Impulsive', 'SS', 'Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caff', 'Choc', 'Coke', 'Crack', 'Ecstasy',
                 #'Heroin', 'Ketamine', 'Legalh', 'LSD', 'Meth', 'Mushrooms', 'Nicotine', 'Semer', 'VSA'], data=liste)
    pred = model.predict(liste)
    return render_template('next.html', data=pred)


if __name__ == "__main__":
    app.run(debug=True, port=3000)
