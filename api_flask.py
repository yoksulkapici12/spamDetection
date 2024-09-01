import pandas as pd
from flask import Flask, request, jsonify
import pickle

api = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
cv = pickle.load(open("vectorizer.pkl", "rb")) 

@api.route("/prediction", methods=["POST"])
def prediction():
    json_ = request.json
    req_df = pd.DataFrame(json_)
    
    transformed_text = cv.transform(req_df['v2'])
    
    pred = model.predict(transformed_text)
    pred = [int(p) for p in pred]
  


    return jsonify({"Prediction": pred})

if __name__ == "__main__":
    api.run(debug=True)
