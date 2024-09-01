import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Load dataset and preprocess
df = pd.read_csv("example_path", encoding='ISO-8859-1')
df.dropna(inplace=True)

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(df.email, df.label, test_size=0.25, random_state=11)

# Initialize CountVectorizer
cv = CountVectorizer()
X_train_count = cv.fit_transform(X_train.values)

# Train the model
model = MultinomialNB()
model.fit(X_train_count, Y_train)

# Save the model and vectorizer using pickle
pickle.dump(model, open("example_path/model.pkl", "wb"))
pickle.dump(cv, open("example_path/vectorizer.pkl", "wb"))
