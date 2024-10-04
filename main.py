from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open('spam_mail.pkl', 'rb'))
vec=pickle.load(open('vector.pkl','rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    msg = request.form['message']
    feature_extract=vec.transform([msg])
    prediction = model.predict(feature_extract)
    if prediction[0].item()==0:
       response_message = f'The Email "{msg}" is not spam.'
       # return jsonify({'prediction':'The Email  is not Spam'})
    else:
        response_message = f'The Email "{msg}" is not spam.'
       # return jsonify({'prediction':'The Given Email is Spam'})
    return jsonify({'prediction':response_message})
   


if __name__ == '__main__':
    app.run(debug=True)