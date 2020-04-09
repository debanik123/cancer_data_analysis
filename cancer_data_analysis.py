#!/usr/bin/env python
# coding: utf-8

# # Library Importing

# In[62]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
#pip install catboost
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import GridSearchCV
from sklearn.svm import NuSVC, SVR


# In[2]:


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()


# # Data Loading 

# In[3]:


cancer= pd.read_csv("data.csv")


# In[4]:


cancer


# In[5]:


cancer.head()


# # Data preprocessing and visualization

# In[6]:


sns.pairplot(cancer,hue='diagnosis',vars=['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean'])


# In[7]:


plt.figure(figsize=(20,12))
sns.heatmap(cancer.corr(),annot=True)


# In[8]:


X=cancer.drop(['diagnosis','Unnamed: 32'],axis=1)
X.head()


# In[9]:


y= cancer['diagnosis']
y.head()


# In[10]:



label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(y)
print(integer_encoded)
y  = integer_encoded
'''
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)'''


# In[11]:


x_train,x_test,y_train,y_test =train_test_split(X,y,test_size=0.25,random_state=20)


# In[12]:


print("Size of the training set 'X' (input features) is:",x_train.shape)
print('\n')
print("Size of the testing set 'X' (input features) is:",x_test.shape)
print('\n')
print("Size of the training set 'y' (output features) is:",y_train.shape)
print('\n')
print("Size of the testing set 'y' (output features) is:",y_test.shape)


# # Satistical Analysis

# In[13]:


def gen_features(X):
    s = []
    s.append(X.mean())
    s.append(X.std())
    s.append(X.min())
    s.append(X.kurtosis())
    s.append(X.skew())
    s.append(np.quantile(X,0.01))
    s.append(np.quantile(X,0.05))
    s.append(np.quantile(X,0.95))
    s.append(np.quantile(X,0.99))
    s.append(np.abs(X).std())
    s.append(np.abs(X).max())
    s.append(np.abs(X).mean())
    return pd.Series(s)
X_train_stat = pd.DataFrame()
stat = []
for df in x_train:
    #print(cancer[df].head())
    ch = gen_features(cancer[df])
    print(ch)
    #stat.append(ch)
    X_train_stat[df] = ch
    #X_train_stat.append(ch, ignore_index=True)


# In[14]:


X_train_stat.describe()


# # SVM withSVC

# In[15]:


sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[16]:


svc_model =SVC(kernel = 'linear', random_state = 0)


# In[17]:


svc_model.fit(x_train,y_train)


# In[18]:


y_predict = svc_model.predict(x_test)


# In[19]:


from sklearn.metrics import classification_report, confusion_matrix
cm = np.array(confusion_matrix(y_test, y_predict, labels=[1,0]))
confusion = pd.DataFrame(cm, index=['is_cancer', 'is_healthy'],
                         columns=['predicted_cancer','predicted_healthy'])
confusion


# In[20]:


sns.heatmap(confusion, annot=True)


# In[21]:


print("classification Repot")
all_labels = ['M','B']
print(classification_report(y_test, y_predict,target_names=all_labels))


# In[22]:


auc = roc_auc_score(y_test, y_predict)
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y_test, y_predict)
plot_roc_curve(fpr, tpr)


# # SVM with RBF

# In[23]:


svc_model = SVC(kernel = 'rbf', random_state = 0)
svc_model.fit(x_train, y_train)
y_predict = svc_model.predict(x_test)

cm = np.array(confusion_matrix(y_test, y_predict, labels=[1,0]))
confusion = pd.DataFrame(cm, index=['is_cancer', 'is_healthy'],
                         columns=['predicted_cancer','predicted_healthy'])
confusion


# In[24]:


sns.heatmap(confusion, annot=True)


# In[25]:


auc = roc_auc_score(y_test, y_predict)
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y_test, y_predict)
plot_roc_curve(fpr, tpr)


# # Result compare with other machine learning algorithm

# # RandomForestClassifier

# In[26]:


model_r = RandomForestClassifier()
model_r.fit(x_train, y_train)
y_predict_r = model_r.predict_proba(x_test)


# In[27]:


cm = np.array(confusion_matrix(y_test, np.argmax(y_predict_r,axis=1), labels=[1,0]))
confusion = pd.DataFrame(cm, index=['is_cancer', 'is_healthy'],
                         columns=['predicted_cancer','predicted_healthy'])
confusion


# In[28]:


sns.heatmap(confusion, annot=True)


# In[29]:


print("classification Repot")
all_labels = ['M','B']
print(classification_report(y_test, np.argmax(y_predict_r,axis=1),target_names=all_labels))


# In[30]:


auc = roc_auc_score(y_test, np.argmax(y_predict_r,axis=1))
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y_test, np.argmax(y_predict_r,axis=1))
plot_roc_curve(fpr, tpr)


# # KNeighborsClassifier

# In[37]:


model_k = KNeighborsClassifier()
model_k.fit(x_train, y_train)


# In[38]:


y_predict_k = model_r.predict_proba(x_test)


# In[39]:


cm = np.array(confusion_matrix(y_test, np.argmax(y_predict_k,axis=1), labels=[1,0]))
confusion = pd.DataFrame(cm, index=['is_cancer', 'is_healthy'],
                         columns=['predicted_cancer','predicted_healthy'])
confusion


# In[40]:


sns.heatmap(confusion, annot=True)


# In[41]:


print("classification Repot")
all_labels = ['M','B']
print(classification_report(y_test, np.argmax(y_predict_r,axis=1),target_names=all_labels))


# In[42]:


auc = roc_auc_score(y_test, np.argmax(y_predict_k,axis=1))
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y_test, np.argmax(y_predict_k,axis=1))
plot_roc_curve(fpr, tpr)


# # LogisticRegression

# In[43]:


classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train, y_train)


# In[44]:


y_predict_l = classifier.predict_proba(x_test)
cm = np.array(confusion_matrix(y_test, np.argmax(y_predict_l,axis=1), labels=[1,0]))
confusion = pd.DataFrame(cm, index=['is_cancer', 'is_healthy'],
                         columns=['predicted_cancer','predicted_healthy'])
confusion


# In[45]:


sns.heatmap(confusion, annot=True)


# In[46]:


print("classification Repot")
all_labels = ['M','B']
print(classification_report(y_test, np.argmax(y_predict_l,axis=1),target_names=all_labels))


# In[47]:


auc = roc_auc_score(y_test, np.argmax(y_predict_l,axis=1))
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y_test, np.argmax(y_predict_l,axis=1))
plot_roc_curve(fpr, tpr)


# # GaussianNB (Naïve Bayes)

# In[48]:


classifier = GaussianNB()
classifier.fit(x_train, y_train)


# In[49]:


y_predict_G = classifier.predict_proba(x_test)
cm = np.array(confusion_matrix(y_test, np.argmax(y_predict_G,axis=1), labels=[1,0]))
confusion = pd.DataFrame(cm, index=['is_cancer', 'is_healthy'],
                         columns=['predicted_cancer','predicted_healthy'])
confusion


# In[50]:


sns.heatmap(confusion, annot=True)


# In[51]:


print("classification Repot")
all_labels = ['M','B']
print(classification_report(y_test, np.argmax(y_predict_G,axis=1),target_names=all_labels))


# In[52]:


auc = roc_auc_score(y_test, np.argmax(y_predict_G,axis=1))
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y_test, np.argmax(y_predict_G,axis=1))
plot_roc_curve(fpr, tpr)


# # Decision Tree Algorithm

# In[53]:


classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(x_train, y_train)


# In[54]:


y_predict_D = classifier.predict_proba(x_test)
cm = np.array(confusion_matrix(y_test, np.argmax(y_predict_D,axis=1), labels=[1,0]))
confusion = pd.DataFrame(cm, index=['is_cancer', 'is_healthy'],
                         columns=['predicted_cancer','predicted_healthy'])
confusion


# In[55]:


sns.heatmap(confusion, annot=True)


# In[56]:


print("classification Repot")
all_labels = ['M','B']
print(classification_report(y_test, np.argmax(y_predict_D,axis=1),target_names=all_labels))


# In[57]:


auc = roc_auc_score(y_test, np.argmax(y_predict_D,axis=1))
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y_test, np.argmax(y_predict_D,axis=1))
plot_roc_curve(fpr, tpr)


# # catboost

# In[58]:


train_pool = Pool(x_train,y_train)
m = CatBoostRegressor(iterations=1000, loss_function="MAE", boosting_type="Ordered")
m.fit(x_train,y_train, silent=True)
m.best_score_


# In[59]:


y_pred_c = m.predict(np.argmax(x_test,axis=1))


# In[60]:


y_pred_c


# # GridSearchCV

# In[63]:


parameters = [{'gamma': [0.001, 0.005, 0.01, 0.02, 0.05, 0.1],
               'C': [0.1, 0.2, 0.25, 0.5, 1, 1.5, 2]}]
               #'nu': [0.75, 0.8, 0.85, 0.9, 0.95, 0.97]}]
reg1 = GridSearchCV(SVR(kernel='rbf', tol=0.01), parameters, cv=5, scoring='neg_mean_absolute_error')
reg1.fit(x_train, y_train.flatten())
y_pred1 = reg1.predict(x_train)

print("Best CV score: {:.4f}".format(reg1.best_score_))
print(reg1.best_params_)
#print(y_pred1)


# In[ ]:





# In[ ]:




