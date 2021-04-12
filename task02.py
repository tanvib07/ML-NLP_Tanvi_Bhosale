from flask import Flask,url_for,redirect,render_template,request
import pickle 
#from pip._vendor.urllib3 import request
import numpy as np
import pandas as pd
import spacy 
from IPython.display import HTML
from IPython.core.display import display
nlp = spacy.load("en_core_web_sm")

#model = pickle.load(open('task1.pkl', 'rb'))
app = Flask(__name__, template_folder='Templates')
model = pickle.load(open('task2.pkl', 'rb'))

@app.route('/task2')
#@app.route('/index1')
def  task2():
	return render_template('index1.html')


@app.route('/success1', methods=['POST','GET'])
def success():
    int_features = [x for x in request.form.values()]
    name = int_features[0]
    para = nlp(str(int_features[1]))
    text =[]
    data =[]
    for sent in para.sents:
        text.append(str(sent))
        analyses = model[0].polarity_scores(str(sent))
        data.append({'Negative':analyses['neg'],'Neutral':analyses['neu'],'Positive':analyses['pos'],'Compound':analyses['compound'],'Text':str(sent)})
    df = pd.DataFrame(data) #pred
    Text = pd.Series(text)
    Para = model[0].polarity_scores(str(para))
    para_sentiment = "\nThe sentiments analysis of full paragraph are --> "+ str(Para) #pred
    dtm = model[1].fit_transform(Text) 
    model[2].fit(dtm) 
    topic_list=[]
    topics = ""
    for index,topic in enumerate(model[2].components_):
        topic_list.append([model[1].get_feature_names()[i] for i in topic.argsort()[-1:]])
        topics = topics + str(topic_list[-1]) +" "    #pred
    return render_template('success2.html',  tables=[df.to_html(classes='data', header="true")], pred='{}          &    The TOP 4 Topics for the paragraph will be : {}'.format( para_sentiment,topics))
    
if __name__ == '__main__':
    app.run(debug=True)
  