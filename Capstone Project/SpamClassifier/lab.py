import pandas as pd
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


df=pd.read_csv('C:/Users/Subir/Learn Bay/Learn Bay Local Copy/Dataset_Backup/SMSSpamCollection', sep='\t', names=['Labels','Messages'])

corpus=[]
for i in range(0,len(df)):
    doc=re.sub('[^a-zA-Z]',' ',df['Messages'][i])
    doc=doc.lower()
    doc=doc.split()    
    doc=[ps.stem(j) for j in doc if not j in stopwords.words('english')]
    doc=' '.join(doc)
    corpus.append(doc)
        
# Bag of words
    
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000)
X=cv.fit_transform(corpus).toarray()
y=pd.get_dummies(df['Labels'],drop_first=True)

# Train test split

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)

# Model

from sklearn.naive_bayes import MultinomialNB
spam=MultinomialNB().fit(X_train,y_train)

y_pred=spam.predict(X_test)

from sklearn.metrics import confusion_matrix
con=confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test, y_pred)




 






    
    