from flask import Flask, request, render_template
import json
import numpy as np
from model import VelocityModel

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            # Extrahieren der Daten aus dem Formular
            param1 = float(request.form['param1'])
            param2 = float(request.form['param2'])
            param3 = float(request.form['param3'])
            param4 = float(request.form['param4'])
            
            # Erzeugen einer Instanz des Modells
            model = VelocityModel(config="config_model.json", check_results=True)
            prediction = model.predict([[param1, param2, param3, param4]])
            print(param1, param2, param3, param4, prediction)
            # Rückgabe der Vorhersage
            return render_template('index.html', prediction=prediction)
        except Exception as e:
            return f"An error occurred: {e}"
    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)

