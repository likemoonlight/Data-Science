# -*- coding: utf-8 -*-
"""
Created on Mon Jan 06 04:36:01 2020

@author: jason
"""
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import tree
import warnings
warnings.filterwarnings('ignore')
import pickle

df = pd.read_csv('2012-18_teamBoxScore.csv')
df = df.dropna()

df['teamLoc'] = df['teamLoc'].map({'Home':1, 'Away':-1})
df['teamRslt'] = df['teamRslt'].map({'Win':1, 'Loss':2})
df['opptLoc'] = df['opptLoc'].map({'Home':1, 'Away':-1})
df['opptRslt'] = df['opptRslt'].map({'Win':1, 'Loss':2})

corrmat = df.corr()
f, ax = plt.subplots(figsize=(20,18))
sns.heatmap(corrmat, vmax=.8, square=True)
f.savefig('a.jpg')
#team result

k = 9
cols = corrmat.nlargest(k, 'teamRslt')['teamRslt'].index
f, ax = plt.subplots(figsize=(10,6))
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
f.savefig('b.jpg')

#team points

k = 12
cols = corrmat.nlargest(k, 'teamPTS')['teamPTS'].index
f, ax = plt.subplots(figsize=(10,6))
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
f.savefig('c.jpg')

#opponent points

k = 12
cols = corrmat.nlargest(k, 'opptPTS')['opptPTS'].index
f, ax = plt.subplots(figsize=(10,6))
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
f.savefig('d.jpg')

#location; h/a

k = 12
cols = corrmat.nlargest(k, 'teamLoc')['teamLoc'].index
f, ax = plt.subplots(figsize=(10,6))
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
f.savefig('e.jpg')

# team days off

k = 12
cols = corrmat.nlargest(k, 'teamDayOff')['teamDayOff'].index
f, ax = plt.subplots(figsize=(10,6))
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
f.savefig('f.jpg')

# team turnovers

k = 12
cols = corrmat.nlargest(k, 'teamTO')['teamTO'].index
f, ax = plt.subplots(figsize=(10,6))
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
f.savefig('g.jpg')

#scatter plots

cols1 = ['opptPTS', 'teamDrtg', 'teamPF', 'teamTO', 'teamORB', 'teamFGA']
sns.pairplot(df[cols1], size=2.5).savefig('h.jpg')
plt.show()

#prepare x and y

feature_cols = ['opptPTS', 'teamDrtg', 'teamPF', 'teamTO', 'teamORB', 'teamFGA']
x = df[feature_cols]
y = df['teamRslt']
x.head()

#train test split, standardize data

x_train, x_test, y_train, y_test = train_test_split(x, y , test_size=0.4, random_state=2)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

#knn 

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
pred = knn.predict(x_test)
print('KNN accuracy:%f'%(metrics.accuracy_score(y_test, pred)))
#print(knn.predict_proba(x_test))

#linear svm

clf = LinearSVC(random_state=2)
clf.fit(x_train, y_train)
#print(clf.coef_)
#print(clf.intercept_)
pred = (clf.predict(x_test))
#print(pred)
print('LinearSVM accuracy:%f'%(metrics.accuracy_score(y_test, pred)))

#random forrest classifier

clf = RandomForestClassifier()
clf.fit(x_train, y_train)

#print(clf.feature_importances_)

pred = clf.predict(x_test)
#print(pred)
#print(clf.predict_proba(x_test))
print('RandomForestClassifier accuarcy:%f'%(metrics.accuracy_score(y_test, pred)))

# Gradient Treee Boosting

clfgtb = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(x_train, y_train)
print('GradientBoostingClassifier accuracy:%f'%(clfgtb.score(x_test, y_test)))

df2 = pd.read_csv('2012-18_teamBoxScore.csv')
df2.head()

#prepare x and y

new_feature_cols = ['opptPTS', 'teamDrtg', 'teamPF', 'teamTO', 'teamORB', 'teamFGA']
x_new = df2[new_feature_cols]
y_new = df2['teamRslt']
x.head()

# gradient tree boosting


clfgtb = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(x, y)
#clfgtb.score(x_new, y_new)

filename = 'nba_pred_modelv2.sav'
pickle.dump(clfgtb, open(filename, 'wb'))




def versus(team1='BOS',team2='LAL'):
    Ace=df[['teamAbbr','opptAbbr','teamLoc','opptPTS','teamDrtg','teamPF','teamTO','teamORB','teamFGA']]
    Acea=np.array(Ace)
    Aceb=Acea.tolist()
    keypoint=[]
    gg=0
    for Acec in Aceb:
        if Aceb[gg][0]==team1 and Aceb[gg][1]==team2:
                keypoint.append(Acec)
        gg+=1
    sum_opptPTS=0
    sum_team_Drtg=0
    sum_teamPF=0
    sum_teamTO=0
    sum_teamORB=0
    sum_teamFGA=0
    for kp in range(len(keypoint)):
        sum_opptPTS+=keypoint[kp][3]
        sum_team_Drtg+=keypoint[kp][4]
        sum_teamPF+=keypoint[kp][5]
        sum_teamTO+=keypoint[kp][6]
        sum_teamORB+=keypoint[kp][7]
        sum_teamFGA+=keypoint[kp][8]
    NN=len(keypoint)    
    avg_opptPTS=sum_opptPTS/NN
    avg_team_Drtg=sum_team_Drtg/NN
    avg_teamPF=sum_teamPF/NN
    avg_teamTO=sum_teamTO/NN
    avg_teamORB=sum_teamORB/NN
    avg_teamFGA=sum_teamFGA/NN
    games = '%s vs %s'%(team1,team2)
    g11 = [[avg_opptPTS, avg_team_Drtg, avg_teamPF, avg_teamTO, avg_teamORB, avg_teamFGA]]
    nba_pred_modelv2 = pickle.load(open(filename, 'rb'))   
    pred11 = nba_pred_modelv2.predict(g11)
    prob11 = nba_pred_modelv2.predict_proba(g11)
    if pred11==1:
        winnerteam=team1
    else:
        winnerteam=team2
    print('比賽 %s'%(games))
    print('兩隊相對獲勝機率 %s'%(prob11))
    return prob11
#team1預測贏的隊伍 g預測人的準確率
def fixed(team1='HOU',team2='NO',g=(metrics.accuracy_score(y_test, pred))):        
    NBAScorePredict=open("NBAScorePredict  %s.txt"%('%s vs %s'%(team1,team2)),"w",encoding = 'utf8')
    p=versus(team1,team2)
    aa=('比賽 %s'%('%s vs %s'%(team1,team2)))
    bb=str(p)
    cc=('兩隊相對獲勝機率:'+bb) 
    NBAScorePredict.write(aa)
    NBAScorePredict.write('\n'+'\n'+cc)    
    if p[0][0]>0.7:
        m=0.65
    else:
        m=0.35   
    Probability=p[0][0]*m+g*(1-m)
    print('調整後預測%s的獲勝機率為'%(team1),Probability)
    dd=str(Probability)
    ee=('調整後預測%s的獲勝機率為:'%(team1)+dd)
    NBAScorePredict.write('\n'+'\n'+ee)
    if Probability>0.5:
        print('預測獲勝隊伍是%s'%(team1))
        ff=('預測獲勝隊伍是%s'%(team1))
        NBAScorePredict.write('\n'+'\n'+ff)
    else:
        print('預測獲勝隊伍是%s'%(team2))
        ff=('預測獲勝隊伍是%s'%(team2))
        NBAScorePredict.write('\n'+'\n'+ff)
    NBAScorePredict.close()       
        
fixed(sys.argv[1],sys.argv[2])