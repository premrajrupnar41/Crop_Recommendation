from flask import Flask, render_template, request , session, url_for , redirect,jsonify, flash
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import requests
import sqlite3
import numpy as np
import pandas as pd
import sklearn as sk
import pickle 


model = pickle.load(open('model.pkl','rb'))
sc = pickle.load(open('standscaler.pkl','rb'))
ms = pickle.load(open('minmaxscaler.pkl','rb'))


app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for session handling
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
CORS(app)


class User(db.Model):
    __tablename__ = 'user'  # Explicitly set the table name to 'user'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)

# Create tables
with app.app_context():
    db.create_all()


@app.route('/')
def main():
    return render_template('login.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check the user's credentials
        user = get_user(username, password)

        if user:
            return redirect(url_for('index'))  # Redirect to index if login is successful
        else:
            flash('Invalid credentials!')  # Flash message if login failed
            return redirect(url_for('login'))  # Redirect back to the login page if failed

    return render_template('login.html')  # Show login page on GET

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        username = request.form['username']
        password = request.form['password']

        new_user = User(name=name, username=username, password=password)
        db.session.add(new_user)
        db.session.commit()
        flash('Registered successfully! Please log in.')  # Show flash message on successful registration

        return redirect(url_for('login'))  # Redirect to login page after successful registration

    return render_template('register.html')  # Show register page on GET

@app.route('/')
def register1():
    return render_template('login.html')
        
    
def get_user(username, password):
    # Use SQLAlchemy ORM to query the User table
    user = User.query.filter_by(username=username, password=password).first()
    return user

@app.route('/index.html')
def index():
    return render_template('index.html')


def fetch_info(query):
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query}"
    response = requests.get(url)  # <-- FIXED LINE

    if response and response.status_code == 200:
        data = response.json()
        title = data.get("title", "No title found")
        summary = data.get("extract", "No information available.")
        return title, summary
    else:
        return None, "Error retrieving information."


@app.route("/search.html", methods=["GET", "POST"])
def index1():
    if request.method == "POST":
        query = request.form["query"]
        print("User query:", query)
        if query:
            title, summary = fetch_info(query)
            print("Title:", title)
            print("Summary:", summary)
            return render_template("search.html", title=title, summary=summary, query=query)
        else:
            return render_template("search.html", error="Please enter a search term.")
    return render_template("search.html")



@app.route('/')
def home():
    return render_template('login.html')




@app.route('/about.html')
def about():
    return render_template('about.html')

@app.route('/login.html')
def logout():
    return render_template('login.html')

@app.route('/register.html')
def regs():
    return render_template('register.html')





@app.route('/predict.html')
def prediction():
    return render_template('predict.html')

@app.route('/search.html')
def search():
    return render_template('search.html')


@app.route("/predict", methods=["POST"])
def predict():
    try:
        N = float(request.form['Nitrogen'])
        P = float(request.form['Phosporus'])
        K = float(request.form['Potassium'])
        temp = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        ph = float(request.form['Ph'])
        rainfall = float(request.form['Rainfall'])

        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)

        scaled_features = ms.transform(single_pred)
        final_features = sc.transform(scaled_features)
        prediction = model.predict(final_features)

        crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                     8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                     14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                     19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

        if prediction[0] in crop_dict:
            crop = crop_dict[prediction[0]]
            result = crop
        else:
            result = "Unknown crop"

        return render_template("predict.html", result=result)

    except Exception as e:
        print("Error:", e)
        return render_template("predict.html", result="Error occurred during prediction.")




if __name__=="__main__":
    app.run(debug=True)
