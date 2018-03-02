from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import LogisticRegression


#first of all we have to add dataset for the spam that we have to put into text then it will work an save the model and deploy it

text_train, text_test, y_train, y_test = train_test_split(text, y, random_state=42, test_size=0.5, stratify=y)

vectorizer = CountVectorizer()
vectorizer.fir(text_train)

x_train = vectorizer.transform(text_train)
x_test = vectorizer.transform(text_test)

vectorizer.get_feature_names()

reg = LogisticRegression()

reg.fit(x_train, y_train)

#find accuracy and save the model and make the api for it
