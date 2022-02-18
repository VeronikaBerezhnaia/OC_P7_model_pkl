from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import pickle
import sklearn
import pandas

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

model = pickle.load(open('saved_model.pkl','rb'))

@app.route('/predict', methods = ['POST','GET'])
@cross_origin()
def predict():
    try:
        data = request.get_json()
       df=pandas.DataFrame(data["data"], columns=data["columns"])
 #       print("print: "+str(df))
        prediction = model.predict_proba(df)
 #       prediction = model.predict_proba(pandas.DataFrame(data["data"], columns=data["columns"]))
        print(prediction)
        output = {'predictions': prediction.tolist()[0][0]}
 #       output = {'predictions': prediction[0]}
        return jsonify( output )
    except NameError:
        print(NameError)
        return 'something went wrong'

@app.errorhandler(404)
def resource_not_found(e):
    return jsonify(error='The requested URL was not found on the server'), 404

if __name__ == "__main__":
    app.run(host ='0.0.0.0', port = 5000, debug = True)