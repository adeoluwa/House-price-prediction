from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import os

# Get the port number from the PORT environment variable
port = int(os.environ.get("PORT", 5000))


app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))


# @app.route('/')
# def index():
#     return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    content_type = request.headers.get('Content-Type')
    if content_type == 'application/json':
        data = request.get_json()
    else:
        data = request.form
    
    bedrooms = data['bedrooms']
    bathrooms = data['bathrooms']
    floors = data['floors']
    yr_built = data['yr_built']
    arr = np.array([bedrooms, bathrooms,floors, yr_built])
    arr = arr.astype(np.float64)
    pred = model.predict([arr])

    # return render_template('index.html', data=int(pred))
    return jsonify({'data':int(pred)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
