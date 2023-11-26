from flask import Flask,render_template,request
import pickle


app=Flask(__name__)
#loading the model
model=pickle.load(open('my_model.pkl','rb'))


@app.route('/')
def home():
    result=''
    return render_template('index.html', **locals())


@app.route('/predict', methods=['POST','GET'])
def predict():
    sepal_length=float(request.form['SL'])
    sepal_width=float(request.form['SW'])
    petal_length=float(request.form['PL'])
    petal_width=float(request.form['PW'])

    result=model.predict([[sepal_length,sepal_width,petal_length,petal_width]])[0]
    return render_template('result.html', **locals())

if __name__=='__main__':
    app.run(debug=True)
