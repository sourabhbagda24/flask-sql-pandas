from flask import Flask, request, jsonify
import joblib
import numpy as np 


app = Flask(__name__)

# load the model 
model = joblib.load('lr_model.pkl')

@app.route("/predict", methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_data = np.array(data['input']).reshape(1, -1)
        precdiction = model.predict(input_data)
        return jsonify({'precdiction':int(precdiction[0])})

    except Exception as e:
     return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(debug=True)