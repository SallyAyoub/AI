import nltk
import csv
nltk.download('punkt')
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn import svm
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score  
from sklearn.externals import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix


df = pd.read_csv('/content/drive/MyDrive/Colab_Notebooks/Output.csv', engine='python')

df = df.dropna()
df = df.reset_index(drop=True)

df_ham = df[df['Filtered'] == 0];
df_ham = df_ham.sample(n=50000)
df_spam = df[df['Filtered'] == 1];
df_spam = df_spam.sample(n=10000)

df = pd.concat([df_spam, df_ham], axis=0)
df = df.sample(frac=1.0)

feature=df['reviewContent']
result=df['Filtered']
X_train, X_test, y_train, y_test = train_test_split(feature, result, test_size=0.30, random_state=42)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X_train_final = vectorizer.fit_transform(X_train)

from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
text_clf = Pipeline([('tfidf', TfidfVectorizer()),('clf', LinearSVC()),])
text_clf.fit(X_train, y_train)
predictions = text_clf.predict(X_test)


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
print("accuracy: ",accuracy_score(y_test, predictions))
cm=confusion_matrix(y_test, predictions)
precision = (cm[0][0])/(cm[0][0]+cm[1][0])
recoil = (cm[0][0])/(cm[0][0]+cm[0][1])
print("Precision: ",precision)
print("Recoil: ",recoil)
print(classification_report(y_test, predictions))
joblib.dump(nb, '/content/drive/MyDrive/Colab_Notebooks/filename.pkl')