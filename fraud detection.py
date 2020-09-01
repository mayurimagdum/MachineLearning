
"""
Fraud detection using machine learning techniques
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import LabelEncoder,StandardScaler,LabelBinarizer
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score
from imblearn.over_sampling import RandomOverSampler,SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
#!pip install imblearn


# Importing data

df = pd.read_csv('C:\\Users\\rohan\\Datasets\\transactions_obf.csv')
df1= pd.read_csv('C:\\Users\\rohan\\Datasets\\labels_obf.csv')
#print(df.head())
#print(df.dtypes)

# Preparing dataset

df['transactionTime'] = pd.to_datetime(df['transactionTime'])
df['hour'] = df['transactionTime'].dt.hour
df['Date'] = df['transactionTime'].dt.date
df['Day']  = df['transactionTime'].dt.day
df['Month']= df['transactionTime'].dt.month
df['weekday']= df['transactionTime'].dt.weekday

# Merding labels with the data

df3 =       pd.merge(df,  
                     df1,  
                     on ='eventId',  
                     how ='left')

#print(df3.head())

df3['reportedTime'].notnull().sum()
df3['reportedTime'] = df3['reportedTime'].fillna(0)
df3['reportedTime'] = df3['reportedTime'].where(df3['reportedTime'] == 0, 1)
#print(df3['reportedTime'].value_counts())
df3 = df3.rename(columns={'reportedTime': 'fraud status'}) 
#print(df3.head())


# Outiler handling 

#print(df3['transactionAmount'].describe())
print(df3.columns)
#print(np.percentile(df3['transactionAmount'],90))
df_fraud = df3[df3['fraud status']==1] 
#print(np.percentile(df_fraud['transactionAmount'],95))
# We will consider data with transactionAmount < 500 of the transactionAmount
df3 = df3[df3['transactionAmount'] < 500]
print(df3.shape)
plt.hist(df3['transactionAmount'])
#print(df3['transactionAmount'].describe())
df3 = df3[df3['transactionAmount']>0] 
#print(df3['availableCash'].describe())
#print(np.percentile(df3['availableCash'],95))
#print(np.percentile(df_fraud['availableCash'],95))
# We will consider data with transactionAmount < 500 of the transactionAmount
df3 = df3[df3['availableCash'] < 10600]
#plt.hist(df3['availableCash'])
print(df3['fraud status'].value_counts())


# Missing Values

print(df3.isnull().sum())
df3= df3.apply(lambda x: x.fillna('Other'))
print(df3.isnull().sum())


#Exploratory data analysis

fraud_df = df3[df3['fraud status'] == 1]
no_fraud_df = df3[df3['fraud status'] == 0]

a = fraud_df['transactionAmount']
b = no_fraud_df['transactionAmount']

plt.figure(figsize=(10, 6))
plt.hist(a, bins = 50, alpha=0.5, label='fraud')
plt.hist(b, bins = 50, alpha=0.5, label='normal')
plt.legend(loc='upper right')
plt.xlabel('transactionAmount')
plt.ylabel('Count')
plt.show();


# Scatter plot of transactionAmount Vs availableCash for fraud data
plt.scatter(x = fraud_df.availableCash , y = fraud_df.transactionAmount)
#plt.scatter(x = no_fraud_df.availableCash , y = no_fraud_df.transactionAmount)
plt.title('availableCash Vs transactionAmount scatterplot for fraudualent activities')
plt.xlabel('available Cash')
plt.ylabel('transaction Amount')
plt.legend(['fraudualent transaction'], loc = 'upper right')
plt.show();

# For fraudulent activities most of the transactions are less than 200

plt.scatter(x = no_fraud_df.availableCash , y = no_fraud_df.transactionAmount)
plt.title('availableCash Vs transactionAmount scatterplot for fraudualent activities')
plt.xlabel('available Cash')
plt.ylabel('transaction Amount')
plt.legend(['Normal transaction'], loc = 'upper right')
plt.show();

# Fraudulent activities per posEntryMode
fraud_df_pem = (fraud_df.groupby('posEntryMode')['fraud status'].count())
fraud_df_pem = pd.DataFrame(fraud_df_pem).reset_index()
fraud_df_pem['ratio'] = (fraud_df_pem['fraud status']/ len(df3))
sns.barplot(x= fraud_df_pem['posEntryMode'],
            y= fraud_df_pem['ratio'], 
            order= fraud_df_pem.sort_values('ratio',ascending = False).posEntryMode
           )

plt.title('count of fraudulent activites per posEntryMode')
plt.show()
# from the above barchart we can say that transactions with '81 : POS Entry E-Commerce' and '1: POS Entry Mode Manual' type of Point Of Sale entry mode have maximum occurace of fraudulent activities

############################################################################ 
#Locationwise analysis of dataset

#  top 10 merchantCountries  w.r.t Fraud rate  
merchcountry_pos_fraud =(fraud_df.groupby(['merchantCountry'])['eventId'].count())
merchcountry_pos_all   =(df3.groupby(['merchantCountry'])['eventId'].count())
merchcountry_pos = pd.DataFrame(merchcountry_pos_fraud/merchcountry_pos_all)
merchcountry_pos= merchcountry_pos.rename(columns= {'eventId':'fraud ratio'})
merchcountry_pos = merchcountry_pos.reset_index()
print(merchcountry_pos.sort_values(by = 'fraud ratio',ascending = False)[:10])


# top 5  merchantzip with the fraudelent activities
top_ten_merzip = fraud_df['merchantZip'].value_counts()[:5]
sns.barplot(x = top_ten_merzip.index, y=top_ten_merzip.values)
plt.xlabel('No of fradulent activities')
plt.ylabel('count')
plt.title('top5 with fradulent activities')
plt.show()


##############################################################################
# Merchantid, accoutnumber analysis of dataset
# top 5 merchantid with the fraudelent activities

top_ten_merid = fraud_df['merchantId'].value_counts()[:5]
sns.barplot(x = top_ten_merid.index, y=top_ten_merid.values)
plt.xlabel('No of fradulent activities')
plt.ylabel('merchantId')
plt.title('Top5 merchantId with fradulent activities')
plt.show()

# top 10 accounts with the fraudelent activities
#print(fraud_df.shape)
top_ten_acc_fraud = fraud_df['accountNumber'].value_counts()
#print(top_ten_acc_fraud)
top_ten_acc_fraud_all = df3['accountNumber'].value_counts()
#print(top_ten_acc_fraud_all)
ratio = pd.DataFrame(top_ten_acc_fraud/top_ten_acc_fraud_all).reset_index()
#print(ratio.head())
accountNumber_ratio_top10= ratio.sort_values(by='accountNumber',ascending = False)[:10]
print(accountNumber_ratio_top10)
# Plotting most likelyhood of the 
sns.barplot(y = accountNumber_ratio_top10['index'], x =accountNumber_ratio_top10['accountNumber'])
plt.xlabel('ratio of fraudulent actities to total activities')
plt.ylabel('AccountNumber')
plt.title('Top10 AccountNumber with fradulent activities')
plt.show()


# Number of events per account per posEntryMode and merchantZip with fraud status
account_zip = fraud_df.groupby(['accountNumber','merchantZip','posEntryMode','fraud status'])['eventId'].count()
account_zip = pd.DataFrame(account_zip.reset_index())
print(account_zip)

###################################################################
# timewise analysis of data
# Fraudulent activities per Hour
fraud_df_hour = (fraud_df.groupby('hour')['fraud status'].count())

plt.figure(figsize=(12,5))
fraud_df_hour = pd.DataFrame(fraud_df_hour).reset_index()
sns.barplot(x= fraud_df_hour['hour'],
            y= fraud_df_hour['fraud status'], 
            order= fraud_df_hour.sort_values('fraud status',ascending = False).hour
           )

plt.title('count of fraudulent activites per hour')
plt.xlabel('Hour')
plt.ylabel('Number of Fraudulent activities')
plt.show()


# Fraudulent activities per Month
fraud_df_date = (fraud_df.groupby('Month')['fraud status'].count())
plt.figure(figsize=(12,5))
fraud_df_date = pd.DataFrame(fraud_df_date).reset_index()
sns.barplot(x= fraud_df_date['Month'],
            y= fraud_df_date['fraud status'], 
            order= fraud_df_date.sort_values('fraud status',ascending = False).Month
           )

plt.title('count of fraudulent activites per Month')
plt.xlabel('Month')
plt.ylabel('Number of Fraudulent activities')
plt.show()

# Fraudulent activities per Month
fraud_df_date = (fraud_df.groupby('weekday')['fraud status'].count())
plt.figure(figsize=(12,5))
fraud_df_date = pd.DataFrame(fraud_df_date).reset_index()
sns.barplot(x= fraud_df_date['weekday'],
            y= fraud_df_date['fraud status'], 
            order= fraud_df_date.sort_values('fraud status',ascending = False).weekday
           )

plt.title('count of fraudulent activites per Month')
plt.xlabel('weekday')
plt.ylabel('Number of Fraudulent activities')
plt.show() 
# Here 0 = Monday and 6 = Sunday

#############################################################################
# lets check for the correlation between merchantcountry and posentrymode for 
#the frauduluent activities

merchcountry_pos_fraud =(fraud_df.groupby(['merchantCountry','posEntryMode'])['eventId'].count())
merchcountry_pos_all   =(df3.groupby(['merchantCountry','posEntryMode'])['eventId'].count())
merchcountry_pos = pd.DataFrame(merchcountry_pos_fraud/merchcountry_pos_all)
merchcountry_pos= merchcountry_pos.rename(columns= {'eventId':'fraud ratio'})
merchcountry_pos = merchcountry_pos.reset_index()
print(merchcountry_pos.sort_values(by = 'fraud ratio',ascending = False)[:10])


# 1. All the transactions done in the country are reported as fraudulent activities 
# 2. POSentryMode : 81: Ecommerce, 1: Mannuel Mode are more prone
# for fraudulent actities 

##########################################################################
#Feature Engineering 
# Lets consider ratio of the transactionAmount and availableCash as 
# one of the features
df3['ratio'] = df3['transactionAmount']/df3['availableCash']

# lets consider top countries and posentrymode combinations for the fraudulent  
# activities as one of the features

df3['f1'] = np.where((df3['merchantCountry'].isin(['659','36','616'])) | 
                      df3['posEntryMode'].isin(['81','1'])
                     ,1,0)

#Binery feature for top 10 accounts with fraudulent activities
df3['f2'] = np.where((df3['accountNumber'].isin(accountNumber_ratio_top10['index']))
                     ,1,0)

df3.dtypes

##########################################################################

# Normalization of the numerical values 
scaler = StandardScaler()
df3['transactionAmount'] = scaler.fit_transform(df3['transactionAmount'].values.reshape(-1,1))
df3['availableCash']     = scaler.fit_transform(df3['availableCash'].values.reshape(-1,1))


#Label Encoding categorical variables
encoder = LabelEncoder()
df3['fraud status'] = encoder.fit_transform(df3['fraud status'])

################################################################

# Checking for correlation between the variables
correlation = df3.corr()
plt.figure(figsize=(10,8))
sns.heatmap(correlation,cmap = 'RdBu')
plt.show()
#print(correlation.f1)

#########################################################################
# Predicting frauduelent activities using anomaly detection using isolation forest
# Feature selection od done based on the correlation pattern


data = df3[['transactionAmount','availableCash','merchantCountry','posEntryMode','weekday','ratio','f1','f2']]
#data1 = df3[['transactionAmount','availableCash']]
print(data.shape)
   
# Buliding model and training it 
model =  IsolationForest(contamination=0.01,
                         max_samples = len(data),
                         n_estimators = 100,
                         random_state = 0
                        )
model.fit(data)

# Predicting anomalies
pred = model.predict(data)
df3['anomalies'] =pred
df3['anomalies'].unique()
df3['anomalies'] = df3['anomalies'].map( {1: 0, -1: 1})
df3['anomalies'].unique()
df3['score'] = pd.Series(model.decision_function(data))
df3['fraud status'].value_counts()
# Model Performance

target = df3['fraud status']
prediction = df3['anomalies']

print('Recall rate of the model is:',round(recall_score(target,prediction),2))
print('accuracy score of the model is:',round(accuracy_score(target,prediction),2))

###############################################################################

#Fraud prediction using supervised techinque

from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.metrics import f1_score,classification_report
from sklearn.metrics import precision_score,recall_score
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import plot_confusion_matrix

#Fraud prediction using supervised techinque: Random Forest algorithm

## Dataset 

df_supervised = df3[['merchantCountry','posEntryMode','transactionAmount',
                     'availableCash','fraud status','weekday','ratio','f1','f2']]
df_supervised.columns
df_supervised.head()
print(df_supervised['fraud status'].value_counts())
df_supervised.dtypes


# Seperating data and target variable
X = df_supervised.drop('fraud status', axis = 1)
y = df_supervised['fraud status']
print(X.shape,y.shape)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,
                                                 random_state = 42)

print(X_train.shape,y_train.shape)
print(X_test.shape,y_test.shape)

# Defining classifier
clf = RandomForestClassifier(class_weight='balanced', n_jobs = -1)

# Hyperparameters tuning using RandomizedSearchCV
# Function for parameter tuning
def hyper_tuning(clf, p_destr, n_iter,X,y):
    randsearch = RandomizedSearchCV(clf, param_distributions= p_destr, n_jobs= -1, n_iter=n_iter,cv= 5)
    randsearch.fit(X,y)
    ht_params = randsearch.best_params_
    ht_score = randsearch.best_score_
    return ht_params,ht_score

random_prams ={
    'max_depth': [3,5,1,4,'None'],
    'n_estimators': [50,10,200,100],
    'criterion'   : ['entropy','gini'],
    'bootstrap'   : ['True','False'],
    'min_samples_leaf' : [1,2,3,4,5],
   }



rf_parameters, rf_score = hyper_tuning(clf,random_prams,5,X_train,y_train)
print(rf_parameters, rf_score)


classifier = RandomForestClassifier(bootstrap= 'True',
                                    criterion= 'entropy',
                                    max_depth= 5,
                                    min_samples_leaf= 3,
                                    n_estimators= 200,
                                    class_weight = 'balanced',
                                    n_jobs = -1,
                                    random_state = 0)


classifier.fit(X_train,y_train)
y_predict  = classifier.predict(X_test)
accuracy  = accuracy_score(y_test,y_predict)
print('Model accuracy is:',round(accuracy,2))
recall = recall_score(y_test,y_predict)
print('Model recall rate is:',round(recall,2))
precision = precision_score(y_test,y_predict)
print('Model precision rate is:',round(precision,2))
probabilities = classifier.predict_proba(X_test)


# ROC_AUC_SCORE

roc_auc = roc_auc_score(y_test,y_predict)
print ("Area under curve : ",roc_auc,"\n")
fpr,tpr,thresholds = roc_curve(y_test,probabilities[:,1])

# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(y_test))]
ns_auc = roc_auc_score(y_test, ns_probs)
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)


# plot the roc curve for the model
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(fpr, tpr, marker='.', label='RandomForest')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()

# Confusion Matrix
print(classification_report(y_test,y_predict,target_names = ['Normal','Fraud']))
print('AUC:','{:.1%}'.format(roc_auc_score(y_test,y_predict)))
cm = confusion_matrix(y_test,y_predict)


LABELS = ['Normal', 'Fraud'] 
conf_matrix = confusion_matrix(y_test, y_predict) 
print(conf_matrix)
plt.figure(figsize =(12, 12)) 
sns.heatmap(conf_matrix, xticklabels = LABELS,  
            yticklabels = LABELS, annot = True, fmt ="d",cmap = 'Greens',
            annot_kws = {'size':16} ); 
plt.title("Confusion matrix") 
plt.ylabel('True class') 
plt.xlabel('Predicted class') 
plt.show() 
###############################################################################
# Important features by RF model
feature_score = pd.DataFrame({'features':X_train.columns,'score':classifier.feature_importances_})
print(feature_score.sort_values(by ='score', ascending = False))
sns.barplot(y= feature_score.features, x = feature_score.score,
            order= feature_score.sort_values('score',ascending = False).features )
plt.title('Feature Importance')

# feature f1 along with posEntryMode and merchantCountry are top predictors of 
# the target variable

###############################################################################


