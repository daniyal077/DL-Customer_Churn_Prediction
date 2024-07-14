from flask import Flask,request,render_template
import numpy as np
import pickle
from tensorflow.keras.models import load_model


app=Flask(__name__)

model = load_model('dl_churn_prediction.keras')
scaler = pickle.load(open('sc.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    data=request.form
    data = [float(data[i]) for i in data]
    scale=scaler.transform(np.array(data).reshape(1,-1))
    prediction=model.predict(scale)
    result=prediction[0]
    
    if result <= 0.5:
        result_text = "Customer will not exit"
    else:
        result_text = "Customer will exit"
    
    return render_template('home.html', result=result_text)

if __name__=='__main__':
    app.run(debug=True)