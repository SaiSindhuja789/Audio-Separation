from flask import Flask,render_template,url_for
app=Flask(__name__)



@app.route("/")
@app.route("/voicesep")
def voicesep():
	return render_template('voicesep.html')



if __name__=='__main__':
	app.run(debug=True)