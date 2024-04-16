from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')
	
@app.route('/homepage', methods =['GET', 'POST'])
def homepage():
    return render_template('index.html')
	

@app.route('/preprocessing', methods =['GET', 'POST'])
def preprocessing():
    msg="Before Preprocessing"
    msg+="\n\nMissing values = true"
    msg+="\n\nAfter Preprocessing"
    msg+="\n\nMissing values = false"
    return render_template('classification.html',msg=msg)

@app.route('/dataset', methods =['GET', 'POST'])
def dataset():
    return render_template('dataset.html')

@app.route('/prediction', methods =['GET', 'POST'])
def prediction():
    return render_template('prediction.html')

@app.route('/predict', methods =['GET', 'POST'])
def predict():
    msg = ''
    if request.method == 'POST' and 's1' and 's2' and 's3' and 's4' and 's5' and 's6' and 's7' and 's8' and 's9' and 's10' and 's11' and 's12' and 's13' and 's14' and 's15' and 's16' and 's17' and 's18' and 's19' and 's20' in request.form :
        s1 = request.form['s1']
        s2 = request.form['s2']
        s3 = request.form['s3']
        s4 = request.form['s4']
        s5 = request.form['s5']
        s6 = request.form['s6']
        s7 = request.form['s7']
        s8 = request.form['s8']
        s9 = request.form['s9']
        s10 = request.form['s10']
        s11 = request.form['s11']
        s12 = request.form['s12']
        s13 = request.form['s13']
        s14 = request.form['s14']
        s15 = request.form['s15']
        s16 = request.form['s16']
        s17 = request.form['s17']
        s18 = request.form['s18']
        s19 = request.form['s19']
        s20 = request.form['s20']
        dataset = pd.read_csv('data1.csv')
        dataset=dataset.replace(to_replace='?',value='0')
        labelencoder_X = LabelEncoder()
        dataset['Class']= labelencoder_X.fit_transform(dataset['Class'])
        y = dataset['Class']
        X = dataset.drop(['Class'], axis = 1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05, random_state = 0)
        gnb = BaggingClassifier(n_estimators = 40, random_state = 22)
        gnb.fit(X_train, y_train)
        Y_pred = gnb.predict([[float(s1),float(s2),float(s3),float(s4),float(s5),float(s6),float(s7),float(s8),float(s9),float(s10),float(s11),float(s12),float(s13),float(s14),float(s15),float(s16),float(s17),float(s18),float(s19),float(s20)]])
        msg1=str(Y_pred)
        msg='Result - Anamoly'
        if msg1 == '[0]':
            msg='Result - Normal'
    return render_template('resultpage.html', msg = msg)

@app.route('/naivebayes', methods =['GET', 'POST'])
def naivebayes():
    dataset = pd.read_csv('data1.csv')
    dataset=dataset.replace(to_replace='?',value='0')
    labelencoder_X = LabelEncoder()
    dataset['Class']= labelencoder_X.fit_transform(dataset['Class'])
    y = dataset['Class']
    X = dataset.drop(['Class'], axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05, random_state = 0)
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    msg=""
    for i in range(0, len(y_pred)):
        i2=str(i)
        res=''
        if y_pred[i]==0:
            res='normal'
        elif y_pred[i]==1:
            res='Anamoly'
        msg1 = '\n Test data '+i2 + ' is '+res
        msg += msg1
    return render_template('classification.html',msg=msg)

@app.route('/knn', methods =['GET', 'POST'])
def knn():
    dataset = pd.read_csv('data1.csv')
    dataset=dataset.replace(to_replace='?',value='0')
    labelencoder_X = LabelEncoder()
    dataset['Class']= labelencoder_X.fit_transform(dataset['Class'])
    y = dataset['Class']
    X = dataset.drop(['Class'], axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05, random_state = 0)
    gnb = KNeighborsClassifier(n_neighbors=7)
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    msg=""
    for i in range(0, len(y_pred)):
        i2=str(i)
        res=''
        if y_pred[i]==0:
            res='normal'
        elif y_pred[i]==1:
            res='Anamoly'
        msg1 = '\n Test data '+i2 + ' is '+res
        msg += msg1
    return render_template('classification.html',msg=msg)

@app.route('/baggingtree', methods =['GET', 'POST'])
def baggingtree():
    dataset = pd.read_csv('data1.csv')
    dataset=dataset.replace(to_replace='?',value='0')
    labelencoder_X = LabelEncoder()
    dataset['Class']= labelencoder_X.fit_transform(dataset['Class'])
    y = dataset['Class']
    X = dataset.drop(['Class'], axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05, random_state = 0)
    gnb = BaggingClassifier(n_estimators = 40, random_state = 22)
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    msg=""
    for i in range(0, len(y_pred)):
        i2=str(i)
        res=''
        if y_pred[i]==0:
            res='normal'
        elif y_pred[i]==1:
            res='Anamoly'
        msg1 = '\n Test data '+i2 + ' is '+res
        msg += msg1
    return render_template('classification.html',msg=msg)

@app.route('/graph', methods =['GET', 'POST'])
def graph():
    return render_template('graph.html')
    
def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)
    return df[indices_to_keep].astype(np.float64)

if __name__ == "__main__":
    app.run(debug=True)
