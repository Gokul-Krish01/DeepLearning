from flask import Flask,redirect,url_for,render_template,request
app=Flask(__name__)

# @app.route("/")
# def home():
#     return "hello world"
# @app.route("/students")
# def table():
#     data=[{"name":"krish" ,"place":"meow heart"},
#           {"name":"poora","place":"buttu heart"}]
#     return render_template("index.html",stud=data)

# @app.route("/user/<name>")
# def user(name):
#     if(name=="krish"):
#         return redirect(url_for("user1",id=21))
#     return f"hi{name}"

# @app.route("/userq/<int:id>")
# def user1(id):
#     return render_template("index.html",marks=id)

@app.route("/")
def home():
   return render_template("index.html")

@app.route("/calculate",methods=["POST"])
def add():
    n1=int(request.form["num1"])
    n2=int(request.form["num2"])
    return f"the sum is {n1+n2}"

if (__name__=='__main__'):
    app.run( debug=True)
    