from flask import Flask, redirect, url_for, render_template

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/todaysprediction/")
def TodaysPrediction():
    return render_template('todaypred.html')

@app.route("/pastpredictions/")
def PastPredictions():
    return render_template('pastpred.html')

@app.route("/themethod/")
def TheMethod():
    return render_template('themethod.html')

@app.route("/about/")
def About():
    return render_template('about.html')

@ app.route("/contacts/")
def Contact():
    return render_template('contact.html')


if __name__ == '__main__':
    app.run()