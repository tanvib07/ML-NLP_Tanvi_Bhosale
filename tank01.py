from flask import Flask,url_for,redirect,render_template,request
import pickle 
#from pip._vendor.urllib3 import request
import numpy as np

#model = pickle.load(open('task1.pkl', 'rb'))
app = Flask(__name__, template_folder='Templates')
model = pickle.load(open('task1.pkl', 'rb'))

@app.route('/task1')
#@app.route('/index')
def  task1():
	return render_template('index.html')


@app.route('/success', methods=['POST','GET'])
def success():
    int_features = [x for x in request.form.values()]
    print(int_features)
    type(int_features)
    #final_features = [np.array(int_features)]
    hours = model.predict(np.array(int(int_features[1])).reshape(-1,1))
    name = int_features[0]
    return render_template('success.html', pred='{} can be active for {} hours'.format(name, hours[0]))
  


if __name__ == '__main__':
	app.run(debug=True)