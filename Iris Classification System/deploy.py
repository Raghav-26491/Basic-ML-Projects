from flask import Flask,render_template,request
import pickle

app = Flask(__name__)
# load the model

model = pickle.load(open('saved_model.sav','rb'))

# Mapping class index to label
class_index_to_label = {
    0: 'Setosa',
    1: 'Versicolor',
    2: 'Virginica'
}

@app.route('/')
def home():
    result = ""
    return render_template('index.html',**locals())

@app.route('/predict', methods =['POST','GET'])
def predict():
    Sepal_Length = float(request.form['Sepal_Length'])
    Sepal_Width = float(request.form['Sepal_Width'])
    Petal_Width = float(request.form['Petal_Width'])
    Petal_Length = float(request.form['Petal_Length'])
    prediction_index = model.predict([[Sepal_Width, Sepal_Length, Petal_Length, Petal_Width]])[0]
    predicted_label = class_index_to_label[prediction_index]

    return render_template('index.html', result=predicted_label)

if __name__== '__main__':
    app.run(debug=True)