
import pandas as pd
import numpy as np
from pandas.tseries.offsets import BDay

from sklearn.linear_model import LogisticRegression,LinearRegression,Lasso,Ridge,ElasticNet
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics

import matplotlib.pyplot as plt
import seaborn as sns

def regress(data):
    
    X = data[['Pos','Neg','Neu']]
    y = data['response']
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
    
    rfRegr = RandomForestRegressor(n_estimators = 1000, random_state = 42)
    svmCl = svm.SVR(kernel='linear')
    MLPR = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=0)
    
    linReg = LinearRegression()
    linReg.fit(X_train, y_train)
    
    ## RIDGE REGRESSION (L2 regularization)
    
    # alters the cost function by adding a penalty equivalent to the 
    # square of the magnitude of the coefficients
    
    rr = Ridge(alpha=0.01)
    rr.fit(X_train, y_train)
    rr100 = Ridge(alpha=100)
    rr100.fit(X_train, y_train)
    
    ## LASSO REGRESSION (L1 regularization)
    
    # alters the cost function by adding a penalty equivalent
    # to the the absolute magnitude of the coefficients

    lasso = Lasso()
    lasso.fit(X_train,y_train)
    lasso001 = Lasso(alpha=0.01, max_iter=10e5)
    lasso001.fit(X_train,y_train)
    lasso00001 = Lasso(alpha=0.0001, max_iter=10e5)
    lasso00001.fit(X_train,y_train)

    ## ELASTIC NET REGRESSION (L1 + L2 regularization)
    
#    train_score = lr.score(X_train, y_train)
#    test_score = lr.score(X_test, y_test)
#    Ltrain_score=lasso.score(X_train,y_train)
#    Ltest_score=lasso.score(X_test,y_test)
#    coeff_used = np.sum(lasso.coef_!=0)
#    print ("training score:", Ltrain_score)
#    print ("test score: ", Ltest_score)
#    print ("number of features used: ", coeff_used)
#    Lscore001=lasso001.score(X_train,y_train)
#    Lscore001=lasso001.score(X_test,y_test)
#    coeff_used001 = np.sum(lasso001.coef_!=0)
#    print ("training score for alpha=0.01:", Ætrain_score001)
#    print ("test score for alpha =0.01: ", Ltest_score001)
#    print ("number of features used: for alpha =0.01:", coeff_used001)
#    Ltrain_score00001=lasso00001.score(X_train,y_train)
#    Ltest_score00001=lasso00001.score(X_test,y_test)
#    coeff_used00001 = np.sum(lasso00001.coef_!=0)    
#    print ("training score for alpha=0.0001:", Ltrain_score00001)
#    print ("test score for alpha =0.0001: ", Ltest_score00001)
#    print ("number of features used: for alpha =0.0001:", coeff_used00001)
#    Ridge_train_score = rr.score(X_train,y_train)
#    Ridge_test_score = rr.score(X_test, y_test)
#    Ridge_train_score100 = rr100.score(X_train,y_train)
#    Ridge_test_score100 = rr100.score(X_test, y_test)
#    print "linear regression train score:", train_score
#    print "linear regression test score:", test_score
#    print "ridge regression train score low alpha:", Ridge_train_score
#    print "ridge regression test score low alpha:", Ridge_test_score
#    print "ridge regression train score high alpha:", Ridge_train_score100
#    print "ridge regression test score high alpha:", Ridge_test_score100
#    plt.plot(rr.coef_,alpha=0.7,linestyle='none',marker='*',markersize=5,color='red',label=r'Ridge; $\alpha = 0.01$',zorder=7) # zorder for ordering the markers
#    plt.plot(rr100.coef_,alpha=0.5,linestyle='none',marker='d',markersize=6,color='blue',label=r'Ridge; $\alpha = 100$') # alpha here is for transparency
#    plt.plot(lr.coef_,alpha=0.4,linestyle='none',marker='o',markersize=7,color='green',label='Linear Regression')
#    plt.xlabel('Coefficient Index',fontsize=16)
#    plt.ylabel('Coefficient Magnitude',fontsize=16)
#    plt.legend(fontsize=13,loc=4)
#    plt.show()

def classify(data):
    
    logReg = LogisticRegression(solver='lbfgs')
    svmCl = svm.SVC(kernel='linear', probability=True)
    rfClass = RandomForestClassifier(n_estimators = 1000)
    MLPC = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=0)
    
    X = data[['Pos', 'Neg', 'Neu']]
    y = data['response']
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
    
    models = [logReg, svmCl, rfClass]

    for model in models:
    
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        
        cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
        class_names=[0,1]
        
        fig, ax = plt.subplots()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names)
        plt.yticks(tick_marks, class_names)
        
        sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
        ax.xaxis.set_label_position("top")
        plt.tight_layout()
        plt.title('Confusion matrix', y=1.1)
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')

def simulateData():
    date = pd.date_range('2018-01-02', '2019-05-20', freq=BDay())
    n = len(date)
    Date=[]
    for i in range(len(date)):
        Date.append(str(date[i].year)+"-"+str(date[i].month)+"-"+str(date[i].day))

    ticker = ['AAPL']
    tickers = sorted(ticker*n)
    ran = np.random.rand(n,3)
    for i in range(len(ran)):
        Sum = sum(ran[i])
        ran[i][0] = ran[i][0]/Sum
        ran[i][1] = ran[i][1]/Sum
        ran[i][2] = ran[i][2]/Sum
    data = {'Date':Date,'Ticker':tickers,'Pos':ran[:,0],'Neg':ran[:,1],'Neu':ran[:,2]}
    DF_A = pd.DataFrame(data)
    
    ticker=["GOOG"]
    Ticker=sorted(ticker*n)
    ran=np.random.rand(n,3)
    for i in range(len(ran)):
        Sum=sum(ran[i])
        ran[i][0]=ran[i][0]/Sum
        ran[i][1]=ran[i][1]/Sum
        ran[i][2]=ran[i][2]/Sum
    data = {'Date':Date,'Ticker':Ticker,'Pos':ran[:,0],'Neg':ran[:,1],'Neu':ran[:,2]}
    DF_G = pd.DataFrame(data)
    
    DF = pd.concat([DF_A, DF_G])
    
    return DF