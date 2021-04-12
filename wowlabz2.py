import spacy
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer 

#load core english library
nlp = spacy.load("en_core_web_sm")
  
#take unicode string  
#here u stands for unicode
doc = nlp(u"The Politics of India works within the framework of the country's Constitution. India is a parliamentary Democratic Republic in which the President of India is the head of state and the Prime Minister of India is the head of government. It is based on the federal structure of government, although the word is not used in the Constitution itself. India follows the dual polity system, i.e. a double government (federal in nature) that consists of the central authority at the centre and states at the periphery. The Constitution defines the organisational powers and limitations of both central and state governments; it is well recognised, fluid (Preamble of the Constitution being rigid and to dictate further amendments to the Constitution) and considered supreme, i.e., the laws of the nation must conform to it.There is a provision for a bicameral legislature consisting of an upper house, the Rajya Sabha (Council of States), which represents the states of the Indian federation, and a lower house, the Lok Sabha (House of the People), which represents the people of India as a whole. The Constitution provides for an independent judiciary, which is headed by the Supreme Court. The court's mandate is to protect the Constitution, to settle disputes between the central government and the states, to settle inter-state disputes, to nullify any central or state laws that go against the Constitution and to protect the fundamental rights of citizens, issuing writs for their enforcement in cases of violation.[1] There are 543 members in the Lok Sabha, who are elected from the 543 constituencies. There are 245 members in the Rajya Sabha, out of which 233 are elected through indirect elections by single transferable vote by the members of the state legislative assemblies; 12 other members are elected/nominated by the President of India. Governments are formed through elections held every five years (unless otherwise specified), by parties that secure a majority of members in their respective lower houses (Lok Sabha in the central government and Vidhan Sabha in states). India had its first general election in 1951, which was won by the Indian National Congress, a political party that went on to dominate subsequent elections until 1977, when a non-Congress government was formed for the first time in independent India. The 1990s saw the end of single-party domination and the rise of coalition governments. The elections for the 16th Lok Sabha, held from April 2014 to May 2014, once again brought back single-party rule in the country, with the Bharatiya Janata Party being able to claim a majority in the Lok Sabha.[2].In recent decades, Indian politics has become a dynastic affair.[3] Possible reasons for this could be the party stability, absence of party organisations, independent civil society associations that mobilise support for the parties and centralised financing of elections.[4] The Economist Intelligence Unit generally rates India as a \"flawed democracy\", and continues to do so as of 2020.[5]")
#to print sentences

text = []
#for sentiments
sid = SentimentIntensityAnalyzer()
print("**************")
print("Negative  Neutral   Positive  Compound  Text"+"\n")
for sent in doc.sents:
    text.append(str(sent))
    analyses = sid.polarity_scores(str(sent))
    print(f"{analyses['neg']:{8}}  {analyses['neu']:{8}}  {analyses['pos']:{8}}  {analyses['compound']:{8}}  {str(sent)} ")

Text = pd.Series(text) #array to series
#Sentiments of complete paragraph
para = sid.polarity_scores(str(doc))
print("\nThe sentiments analysis of full paragraph are --> "+ str(para))

print(type(Text))

#Topic modelling

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
dtm = tfidf.fit_transform(Text) #

#NMF-Non neg Matrix Factorization
from sklearn.decomposition import NMF
nmf_model = NMF(n_components=4,random_state=42)
nmf_model.fit(dtm) #

import random
feat_len = len(tfidf.get_feature_names())
for i in range(10):
    random_word_id = random.randint(0,feat_len)
    print(tfidf.get_feature_names()[random_word_id])

#final topics
print('THE TOP 4 WORDS FOR TOPIC ARE')
for index,topic in enumerate(nmf_model.components_):
    print([tfidf.get_feature_names()[i] for i in topic.argsort()[-1:]])
    
import pickle
pickle.dump([sid,tfidf,nmf_model],open('task2.pkl','wb'))
model = pickle.load(open('task2.pkl', 'rb'))