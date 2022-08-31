import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn import datasets
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, GridSearchCV, cross_val_score

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from PIL import Image

###########################################################################################

st.title('Macine Learning Classification Web App')

###########################################################################################
st.subheader('The Feature Data')
dataset_name = st.sidebar.selectbox('Select Datasets', ('Brast Cancer', 'Iris Flower', 'Wine'))
classifier_name = st.sidebar.selectbox('Select Algorithm', ('SVM','KNN'))

###########################################################################################

def get_datasets(name):
    data=None
    if name=='Iris Flower':
        data=datasets.load_iris()
    elif name=='Wine':
        data=datasets.load_wine()
    else:
        data=datasets.load_breast_cancer()
    x = data.data
    y = data.target
    return x,y,data

x,y,data = get_datasets(dataset_name)
x = pd.DataFrame(x, columns=data['feature_names'])
st.dataframe(x)
st.write('Shape of your dataset is:', x.shape)

st.subheader('The Target Data')
st.write('The number of unique Target values are:', len(np.unique(y)))
st.write('The unique Target values are: ', np.unique(y))
st.write('The Target value names are: ', data['target_names'])

###########################################################################################

st.subheader('Simple Boxplot')
fig = plt.figure(figsize=(10,8))
sns.boxplot(data=x, orient='h')
plt.tight_layout()
st.pyplot(fig)

###########################################################################################

def add_parameter(name_of_clf):
    params=dict()
    rand_state = st.sidebar.slider('Random State',1,100,1)
    if name_of_clf == 'SVM':
        c= st.sidebar.slider('C', 0.01,15.0,1.14)
        gamma = st.sidebar.number_input('gamma',0.0001)
        st.sidebar.write('The gamma you selected is ',gamma)
        params['C']= c
        params['gamma'] = gamma
    else:
        name_of_clf == 'KNN'
        k = st.sidebar.slider('k',1,20,16)
        params['n_neighbors'] = k
    return params, rand_state

params, rand_state = add_parameter(classifier_name)

###########################################################################################

def get_classifier(name_of_clf,params):
    clf = None
    if name_of_clf =='SVM':
        clf = SVC(C=params['C'], gamma=params['gamma'])
    else:
        clf = KNeighborsClassifier(n_neighbors=params['n_neighbors'])
    return clf
        
clf = get_classifier(classifier_name,params)

###########################################################################################

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state= rand_state)

clf.fit(X_train,y_train)

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=rand_state)

score = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=cv, n_jobs=-1, scoring='accuracy')

st.subheader('Training the model based on the classifier: {} and cross validation'.format(clf))

st.write('The average training score is: {}%'.format(round(score.mean()*100,2)))

###########################################################################################

st.subheader('Testing the model on unseen data')

y_pred = clf.predict(X_test)
score1 = round(accuracy_score(y_test,y_pred)*100,2)

st.write('The accuracy score is {}%'.format(score1))

cm = pd.DataFrame(confusion_matrix(y_test,y_pred), columns=data['target_names'], index=data['target_names'])

st.write('Confusion Matrix')
st.dataframe(cm)

