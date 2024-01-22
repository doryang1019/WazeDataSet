#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 1. Import Libraries and Data
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
from sklearn.pipeline import Pipeline, FeatureUnion
import shap
import xgboost
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score


# In[2]:


waze1=pd.read_csv("./waze_dataset.csv")


# In[ ]:


waze1


# In[4]:


# 2. Converting & cleaning data
waze1['label'] = waze1['label'].replace({'churned': 0, 'retained': 1})
waze1['device'] = waze1['device'].replace({'Android': 0, 'iPhone': 1})


# In[5]:


waze1.dropna(inplace=True)


# In[6]:


# 3. Oversampling
waze_tmp =  waze1[waze1['label'] == 0]
waze_cleansed = pd.concat([waze1, waze_tmp,waze_tmp, waze_tmp], ignore_index=True, sort=False)


# In[7]:


# 4. Correlation Heatmap
cor2=waze_cleansed.corr()
sb.heatmap(cor2)
plt.show()


# In[8]:


# 6. Split into x and y
x = waze_cleansed[['sessions', 'drives', 'total_sessions', 'n_days_after_onboarding', 'total_navigations_fav1', 
           'total_navigations_fav2', 'driven_km_drives', 'duration_minutes_drives', 'activity_days', 'driving_days','device']]
y = waze_cleansed['label']


# In[ ]:


# 7. Using feature importance (Shap)
rf = RandomForestRegressor(n_estimators=100)
rf.fit(x,y)
explainer = shap.TreeExplainer(rf, feature_names= x.columns)
shap_values = explainer.shap_values(x)
shap.summary_plot(shap_values, x, plot_type="bar")


# In[ ]:


xgb1=XGBClassifier(n_estimator=100, learning_rate=0.05)
xgb1.fit(x,y)
xgb1.get_booster().get_score(importance_type='gain')


# In[ ]:


# XGBoost Feature importance Top 5 features
# 'activity_days': 116.81880187988281,
# 'n_days_after_onboarding': 18.31635093688965,
# 'driving_days': 15.739612579345703,
# 'drives': 12.531010627746582,
# 'total_navigations_fav1': 11.800623893737793,


# In[ ]:


# Top 3 common features (Shap vs XGBoost)
# 'activity_days': 116.81880187988281,
# 'n_days_after_onboarding': 18.31635093688965,
# 'total_navigations_fav1': 11.800623893737793,


# In[ ]:


# 8. Define the x again based on the top 3 features
x = waze_cleansed[['n_days_after_onboarding', 'total_navigations_fav1', 'activity_days']]
y = waze_cleansed['label']


# In[ ]:


# 9. Splitting into test and train
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42,stratify=y)


# In[ ]:


# 10. Modeling
# 10.1 DecisionTreeClassifier
tree1=DecisionTreeClassifier()

tree1.fit(x_train,y_train)
y_pre_dt=tree1.predict(x_test)
print(tree1.score(x_test, y_test))

cm_dt=confusion_matrix(y_test,y_pre_dt)
print(cm_dt)


# In[15]:


# 10.2 KNN
knn1=KNeighborsClassifier(n_neighbors=8, metric='minkowski', p=2)

knn1.fit(x_train,y_train)
y_pre_knn=knn1.predict(x_test)
print(knn1.score(x_test, y_test))

cm_knn=confusion_matrix(y_test,y_pre_knn)
print(cm_knn)


# In[16]:


# 10.3 Naive Bayes
gnb1=GaussianNB()

y_pre_nb=gnb1.fit(x_train,y_train).predict(x_test)
print(gnb1.score(x_test, y_test))

cm_nb=confusion_matrix(y_test,y_pre_nb)
print(cm_nb)


# In[17]:


# 10.4 Random Forest Regressor
RF1 = RandomForestRegressor(n_estimators=50,random_state=0)

RF1.fit(x_train,y_train)
y_pre_rfr=RF1.predict(x_test)
print(RF1.score(x_test,y_test))


# In[18]:


# 10.5 Random Forest Classifier
RF2 = RandomForestClassifier(max_depth=4, oob_score=True)

RF2.fit(x_train,y_train)
y_pre_rfc = RF2.predict(x_test)
print(RF2.score(x_test,y_test))

cm_rfc=confusion_matrix(y_test,y_pre_rfc)
print(cm_rfc)


# In[19]:


# 10.6 Bagging
Bag1=BaggingClassifier(n_estimators=10,random_state=22)

Bag1.fit(x_train,y_train)
y_pre_bag = Bag1.predict(x_test)
print(Bag1.score(x_test,y_test))

cm_bag=confusion_matrix(y_test,y_pre_bag)
print(cm_bag)


# In[20]:


# 10.7 ADA Boost
Ada1 = AdaBoostClassifier(n_estimators=50,learning_rate=0.2).fit(x_train,y_train)
y_pre_boost = Ada1.predict(x_test)
print(Ada1.score(x_test,y_test))

cm_boost = confusion_matrix(y_test, y_pre_boost)
print(cm_boost)


# In[21]:


# 10.8 XGBoost
xg1 = XGBClassifier(n_estimators = 1000, learning_rate = 0.05)

xg1.fit(x_train, y_train, early_stopping_rounds = 5, eval_set = [(x_test, y_test)],verbose = False)
y_pre_xg = xg1.predict(x_test)
print(xg1.score(x_test, y_test))

cm_xgb = confusion_matrix(y_test, y_pre_xg)
print(cm_xgb)


# In[22]:


# 10.9 Logistic Regression
reg2=LogisticRegression()

reg2.fit(x_train,y_train)
y_pre_logit=reg2.predict(x_test)
print(reg2.score(x_test,y_test))

cm_logit=confusion_matrix(y_test,y_pre_logit)
print(cm_logit)


# In[23]:


# 10.10.1 K-Fold with logistic regression (3)
klog=LogisticRegression() 
cvl_score=cross_val_score(klog,x,y,cv=3)
print(np.mean(cvl_score))


# In[24]:


# 10.10.2 K-Fold with logistic regression (4)
klog=LogisticRegression() 
cvl_score=cross_val_score(klog,x,y,cv=4)
print(np.mean(cvl_score))


# In[25]:


# 10.10.3 K-Fold with logistic regression (5)
klog=LogisticRegression() 
cvl_score=cross_val_score(klog,x,y,cv=5)
print(np.mean(cvl_score))


# In[26]:


# 10.10.4 K-Fold with logistic regression (6)
klog=LogisticRegression() 
cvl_score=cross_val_score(klog,x,y,cv=6)
print(np.mean(cvl_score))


# In[27]:


# 10.11.1 K-Fold with decision tree (3)
ktree = DecisionTreeClassifier()
cvk_score=cross_val_score(ktree,x,y,cv=3)
print(np.mean(cvk_score))


# In[28]:


# 10.11.2 K-Fold with decision tree (4)
ktree = DecisionTreeClassifier()
cvk_score=cross_val_score(ktree,x,y,cv=4)
print(np.mean(cvk_score))


# In[29]:


# 10.11.3 K-Fold with decision tree (5)
ktree = DecisionTreeClassifier()
cvk_score=cross_val_score(ktree,x,y,cv=5)
print(np.mean(cvk_score))


# In[30]:


# 10.11.4 K-Fold with decision tree (6)
ktree = DecisionTreeClassifier()
cvk_score=cross_val_score(ktree,x,y,cv=6)
print(np.mean(cvk_score))


# In[31]:


# 10.12.1 K-Fold with Naive Bayes (3)
kgnb=GaussianNB()
cvn_score=cross_val_score(kgnb,x,y,cv=3)
print(np.mean(cvn_score))


# In[32]:


# 10.12.2 K-Fold with Naive Bayes (4)
kgnb=GaussianNB()
cvn_score=cross_val_score(kgnb,x,y,cv=4)
print(np.mean(cvn_score))


# In[33]:


# 10.12.3 K-Fold with Naive Bayes (5)
kgnb=GaussianNB()
cvn_score=cross_val_score(kgnb,x,y,cv=5)
print(np.mean(cvn_score))


# In[34]:


# 10.12.4 K-Fold with Naive Bayes (6)
kgnb=GaussianNB()
cvn_score=cross_val_score(kgnb,x,y,cv=6)
print(np.mean(cvn_score))


# In[35]:


# 10.13.1 kfold with Random Forest Regressor (3)
RF2 = RandomForestRegressor(n_estimators=50,random_state=0)
cvf_score = cross_val_score(RF2, x,y,cv=3)
print(np.mean(cvf_score))


# In[36]:


# 10.13.2 kfold with Random Forest Regressor (4)
RF2 = RandomForestRegressor(n_estimators=50,random_state=0)
cvf_score = cross_val_score(RF2, x,y,cv=4)
print(np.mean(cvf_score))


# In[37]:


# 10.13.3 kfold with Random Forest Regressor (5)
RF2 = RandomForestRegressor(n_estimators=50,random_state=0)
cvf_score = cross_val_score(RF2, x,y,cv=5)
print(np.mean(cvf_score))


# In[38]:


# 10.13.4 kfold with Random Forest Regressor (6)
RF2 = RandomForestRegressor(n_estimators=50,random_state=0)
cvf_score = cross_val_score(RF2, x,y,cv=6)
print(np.mean(cvf_score))


# In[39]:


# 10.14.1 kfold with Random Forest Classifier (3)
RF3 = RandomForestClassifier(max_depth=4, oob_score=True)
cvf2_score = cross_val_score(RF3, x,y,cv=3)
print(np.mean(cvf2_score))


# In[40]:


# 10.14.2 kfold with Random Forest Classifier (4)
RF3 = RandomForestClassifier(max_depth=4, oob_score=True)
cvf2_score = cross_val_score(RF3, x,y,cv=4)
print(np.mean(cvf2_score))


# In[41]:


# 10.14.3 kfold with Random Forest Classifier (5)
RF3 = RandomForestClassifier(max_depth=4, oob_score=True)
cvf2_score = cross_val_score(RF3, x,y,cv=5)
print(np.mean(cvf2_score))


# In[42]:


# 10.14.4 kfold with Random Forest Classifier (6)
RF3 = RandomForestClassifier(max_depth=4, oob_score=True)
cvf2_score = cross_val_score(RF3, x,y,cv=6)
print(np.mean(cvf2_score))


# In[43]:


# 10.15.1 kfold with Bagging (3)
Bag2=BaggingClassifier(n_estimators=10,random_state=22)
cvb_score = cross_val_score(Bag2, x,y,cv=3)
print(np.mean(cvb_score))


# In[44]:


# 10.15.2 kfold with Bagging (4)
Bag2=BaggingClassifier(n_estimators=10,random_state=22)
cvb_score = cross_val_score(Bag2, x,y,cv=4)
print(np.mean(cvb_score))


# In[45]:


# 10.15.3 kfold with Bagging (5)
Bag2=BaggingClassifier(n_estimators=10,random_state=22)
cvb_score = cross_val_score(Bag2, x,y,cv=5)
print(np.mean(cvb_score))


# In[46]:


# 10.15.4 kfold with Bagging (6)
Bag2=BaggingClassifier(n_estimators=10,random_state=22)
cvb_score = cross_val_score(Bag2, x,y,cv=6)
print(np.mean(cvb_score))


# In[47]:


# 10.16.1 kfold with Ada Boost (3)
Ada2 = AdaBoostClassifier(n_estimators=50,learning_rate=0.2)
cva_score = cross_val_score(Ada2, x,y,cv=3)
print(np.mean(cva_score))


# In[48]:


# 10.16.2 kfold with Ada Boost (4)
Ada2 = AdaBoostClassifier(n_estimators=50,learning_rate=0.2)
cva_score = cross_val_score(Ada2, x,y,cv=4)
print(np.mean(cva_score))


# In[49]:


# 10.16.3 kfold with Ada Boost (5)
Ada2 = AdaBoostClassifier(n_estimators=50,learning_rate=0.2)
cva_score = cross_val_score(Ada2, x,y,cv=5)
print(np.mean(cva_score))


# In[50]:


# 10.16.4 kfold with Ada Boost (6)
Ada2 = AdaBoostClassifier(n_estimators=50,learning_rate=0.2)
cva_score = cross_val_score(Ada2, x,y,cv=6)
print(np.mean(cva_score))


# In[51]:


# 10.17.1 kfold with XGBoost (3)
xg2 = XGBClassifier(n_estimators = 1000, learning_rate = 0.05)
cvx_score = cross_val_score(xg2, x,y,cv=3)
print(np.mean(cvx_score))


# In[52]:


# 10.17.2 kfold with XGBoost (4)
xg2 = XGBClassifier(n_estimators = 1000, learning_rate = 0.05)
cvx_score = cross_val_score(xg2, x,y,cv=4)
print(np.mean(cvx_score))


# In[53]:


# 10.17.3 kfold with XGBoost (5)
xg2 = XGBClassifier(n_estimators = 1000, learning_rate = 0.05)
cvx_score = cross_val_score(xg2, x,y,cv=5)
print(np.mean(cvx_score))


# In[54]:


# 10.17.4 kfold with XGBoost (6)
xg2 = XGBClassifier(n_estimators = 1000, learning_rate = 0.05)
cvx_score = cross_val_score(xg2, x,y,cv=6)
print(np.mean(cvx_score))


# In[55]:


# 10.18.1 Neural Network - Keras model with 3 variables
model1 = Sequential()
model1.add(Dense(12, input_shape=(3,), activation='relu'))
model1.add(Dense(8, activation='relu'))
model1.add(Dense(1, activation='sigmoid'))
model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model1.fit(x, y, epochs=500, batch_size=10)


# In[56]:


_, accuracy = model1.evaluate(x,y)
print('Accuracy: %.2f' % (accuracy*100))


# In[58]:


from keras.layers import LeakyReLU
# 7.18.1 Neural Network - Keras model with 3 variables
model1 = Sequential()
model1.add(Dense(20, input_shape=(3,), activation=LeakyReLU(alpha=0.01)))
model1.add(Dense(12, activation='elu'))
model1.add(Dense(8, activation='tanh'))
model1.add(Dense(1, activation='sigmoid'))
model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model1.fit(x, y, epochs=500, batch_size=10)


# In[59]:


_, accuracy = model1.evaluate(x,y)
print('Accuracy: %.2f' % (accuracy*100))


# In[65]:


# Defining my x and y again
x = waze_cleansed[['sessions', 'drives', 'total_sessions', 'n_days_after_onboarding', 'total_navigations_fav1', 
           'total_navigations_fav2', 'driven_km_drives', 'duration_minutes_drives', 'activity_days', 'driving_days','device']]
y = waze_cleansed['label']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42,stratify=y)


# In[66]:


# Create a pipeline with feature selection and classification
# Feature Selection: RandomForestClassifier
# Classification: RandomForestClassifier
feature_selection_pipeline = Pipeline([
    ('feature_union', FeatureUnion([
        ('model_based_selection', SelectFromModel(RandomForestClassifier())),
        ('univariate_selection', SelectKBest(score_func=f_classif, k=5))
    ])),
    ('classification', RandomForestClassifier())
])

# Fit the pipeline
feature_selection_pipeline.fit(x_train, y_train)

print(accuracy_score(y_test, feature_selection_pipeline.predict(x_test)))
y_pre_pipeline=feature_selection_pipeline.predict(x_test)

cm_pipeline=confusion_matrix(y_test,y_pre_pipeline)
print(cm_pipeline)


# In[67]:


# Create a pipeline with feature selection and classification
# Feature Selection: XGBClassifier
# Classification: GaussianNB
feature_selection_pipeline = Pipeline([
    ('feature_union', FeatureUnion([
        ('model_based_selection', SelectFromModel(XGBClassifier(random_state=42))),
        ('univariate_selection', SelectKBest(score_func=f_classif, k=5))
    ])),
    ('classification', GaussianNB())
])

# Fit the pipeline
feature_selection_pipeline.fit(x_train, y_train)

print(accuracy_score(y_test, feature_selection_pipeline.predict(x_test)))
y_pre_pipeline=feature_selection_pipeline.predict(x_test)

cm_pipeline=confusion_matrix(y_test,y_pre_pipeline)
print(cm_pipeline)


# In[68]:


# Create a pipeline with feature selection and classification
# Feature Selection: XGBClassifier
# Classification: BaggingClassifier
feature_selection_pipeline = Pipeline([
    ('feature_union', FeatureUnion([
        ('model_based_selection', SelectFromModel(XGBClassifier(random_state=42))),
        ('univariate_selection', SelectKBest(score_func=f_classif, k=5))
    ])),
    ('classification', BaggingClassifier())
])

# Fit the pipeline
feature_selection_pipeline.fit(x_train, y_train)

print(accuracy_score(y_test, feature_selection_pipeline.predict(x_test)))
y_pre_pipeline=feature_selection_pipeline.predict(x_test)

cm_pipeline=confusion_matrix(y_test,y_pre_pipeline)
print(cm_pipeline)


# In[69]:


# Create a pipeline with feature selection and classification
# Feature Selection: XGBClassifier
# Classification: ADA Classifier
feature_selection_pipeline = Pipeline([
    ('feature_union', FeatureUnion([
        ('model_based_selection', SelectFromModel(XGBClassifier(random_state=42))),
        ('univariate_selection', SelectKBest(score_func=f_classif, k=5))
    ])),
    ('classification', AdaBoostClassifier())
])

# Fit the pipeline
feature_selection_pipeline.fit(x_train, y_train)

print(accuracy_score(y_test, feature_selection_pipeline.predict(x_test)))
y_pre_pipeline=feature_selection_pipeline.predict(x_test)

cm_pipeline=confusion_matrix(y_test,y_pre_pipeline)
print(cm_pipeline)


# In[70]:


# Create a pipeline with feature selection and classification
# Feature Selection: XGBClassifier
# Classification: DecisionTreeClassifier
feature_selection_pipeline = Pipeline([
    ('feature_union', FeatureUnion([
        ('model_based_selection', SelectFromModel(XGBClassifier(random_state=42))),
        ('univariate_selection', SelectKBest(score_func=f_classif, k=5))
    ])),
    ('classification', DecisionTreeClassifier())
])

# Fit the pipeline
feature_selection_pipeline.fit(x_train, y_train)

print(accuracy_score(y_test, feature_selection_pipeline.predict(x_test)))
y_pre_pipeline=feature_selection_pipeline.predict(x_test)

cm_pipeline=confusion_matrix(y_test,y_pre_pipeline)
print(cm_pipeline)


# In[71]:


# Create a pipeline with feature selection and classification
# Feature Selection: XGBClassifier
# Classification: KNeighborsClassifier
feature_selection_pipeline = Pipeline([
    ('feature_union', FeatureUnion([
        ('model_based_selection', SelectFromModel(XGBClassifier(random_state=42))),
        ('univariate_selection', SelectKBest(score_func=f_classif, k=5))
    ])),
    ('classification', KNeighborsClassifier())
])

# Fit the pipeline
feature_selection_pipeline.fit(x_train, y_train)

print(accuracy_score(y_test, feature_selection_pipeline.predict(x_test)))
y_pre_pipeline=feature_selection_pipeline.predict(x_test)

cm_pipeline=confusion_matrix(y_test,y_pre_pipeline)
print(cm_pipeline)


# In[72]:


# Create a pipeline with feature selection and classification
# Feature Selection: XGBClassifier
# Classification: XGBClassifier
feature_selection_pipeline = Pipeline([
    ('feature_union', FeatureUnion([
        ('model_based_selection', SelectFromModel(XGBClassifier(random_state=42))),
        ('univariate_selection', SelectKBest(score_func=f_classif, k=5))
    ])),
    ('classification', XGBClassifier())
])

# Fit the pipeline
feature_selection_pipeline.fit(x_train, y_train)

print(accuracy_score(y_test, feature_selection_pipeline.predict(x_test)))
y_pre_pipeline=feature_selection_pipeline.predict(x_test)

cm_pipeline=confusion_matrix(y_test,y_pre_pipeline)
print(cm_pipeline)


# In[73]:


# Create a pipeline with feature selection and classification
# Feature Selection: XGBClassifier
# Classification: LogisticRegression
feature_selection_pipeline = Pipeline([
    ('feature_union', FeatureUnion([
        ('model_based_selection', SelectFromModel(XGBClassifier(random_state=42))),
        ('univariate_selection', SelectKBest(score_func=f_classif, k=5))
    ])),
    ('classification', LogisticRegression())
])

# Fit the pipeline
feature_selection_pipeline.fit(x_train, y_train)

print(accuracy_score(y_test, feature_selection_pipeline.predict(x_test)))
y_pre_pipeline=feature_selection_pipeline.predict(x_test)

cm_pipeline=confusion_matrix(y_test,y_pre_pipeline)
print(cm_pipeline)


# In[ ]:




