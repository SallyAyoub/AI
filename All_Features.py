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
df_ham = df_ham.sample(n=10)
df_spam = df[df['Filtered'] == 1];
df_spam = df_spam.sample(n=10)

df = pd.concat([df_spam, df_ham], axis=0)
df = df.sample(frac=1.0)


reviews = []

sims        = []     #Feature number 1
Lengths     = []     #Feature number 2
revID       = []     #Feature Number 3
firstCount  = []     #Feature number 4
usefulCount = []     #Feature number 5
result      = []
 


porter = PorterStemmer()
for i in range(len(df.axes[0])):
  if(df.iloc[i][1] == "#NAME?" ):
    continue
  text=df.iloc[i][2]
  revID.append(df.iloc[i][1])
  Lengths.append(len(text))
  firstCount.append(df.iloc[i][7])
  usefulCount.append(df.iloc[i][4])
  result.append(df.iloc[i][9])


  text_tokens = word_tokenize(text.lower())
  tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
  for j in range(len(tokens_without_sw)):
    porter.stem(tokens_without_sw[j])
    filtered_sentence = (" ").join(tokens_without_sw)
  reviews.append(filtered_sentence)

vectorizer = TfidfVectorizer(max_features=500)

vec1 = np.array(reviews)
x = vectorizer.fit_transform(vec1)
z=cosine_similarity(x)


for i in range(len(z)):
  sim=0
  for j in range(len(z[i])):
    if(i == j):
      continue
    sim = sim + z[i][j]
  sims.append(sim)


le = preprocessing.LabelEncoder()
revID1=le.fit_transform(revID)


Features = np.array(list(zip(sims,Lengths,firstCount))).astype(np.float)
y = np.array(result).astype(np.float)



X_train, X_test, y_train, y_test = train_test_split(Features, y, test_size=0.30, random_state=42)
nb = MultinomialNB()      #naiv bayes
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)
print("Accuray: ",accuracy_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)

precision = (cm[0][0])/(cm[0][0]+cm[1][0])
recoil = (cm[0][0])/(cm[0][0]+cm[0][1])
print("Precision: ",precision)
print("Recoil: ",recoil)
print(cm)




joblib.dump(nb, '/content/drive/MyDrive/Colab_Notebooks/trash.pkl')



