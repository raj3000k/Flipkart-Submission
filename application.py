from flask import Flask, flash, redirect, render_template, request, session
#from flask_session import Session
from functions import apology, login_required
from tempfile import mkdtemp
from werkzeug.exceptions import default_exceptions, HTTPException, InternalServerError
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename
from Code import calc
import os

app = Flask(__name__)

'''@app.route("/")
@login_required
def index():
    db.execute("SELECT * FROM pictures WHERE userID=?",(session["user_id"],))
    record = db.fetchall()
    return render_template("index.html",length=len(record),record=record,flag=0)'''


@app.route("/")
def index():
    #Starting page
    return render_template("index.html")


@app.route("/generate", methods=["GET", "POST"])
def generate():
    age = request.form.get("age")
    dept = request.form.get("dept")
    dist = request.form.get("dist")
    education = request.form.get("education")
    edufield = request.form.get("edufield")
    years = request.form.get("years")
    predict = calc(age,dept,dist,education,edufield,years)
    no = predict[0][0]
    yes = predict[0][1]
    return render_template("result.html",no=no,yes=yes,age=int(age), dist=int(dist), education=int(education), years=int(years))

'''@app.route("/signin", methods=["GET", "POST"])
def signin():
    session.clear()
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        db.execute("SELECT * FROM users WHERE username = '"+username+"';")
        temp.commit()
        users = db.fetchall()
        if len(users) != 1 or not check_password_hash(users[0][4],password):
            return render_template("signin.html",flag=2)
        session["user_id"] = users[0][0]
        #db.execute("SELECT * FROM pictures WHERE userID=?",(session["user_id"],))
        record = db.fetchall()
        return render_template("index.html",length=len(record),record=record,flag=1)
    else:
        return render_template("signin.html")
    
@app.route("/logout")
def logout():
    session.clear()
    return render_template("welcome.html",flag=2)'''

if __name__ == "__main__":
	app.run()