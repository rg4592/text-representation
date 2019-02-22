# Prepare Text Data for Machine Learning with scikit-learn
# Word Counts with CountVectorizer
# from sklearn.feature_extraction.text import CountVectorizer
# text = ["The quick brown fox jumped over the lazy dog."]
# vectorizer = CountVectorizer()
# vectorizer.fit(text)
# print(vectorizer.vocabulary_)
# vector = vectorizer.transform(text)
# print(vector.shape)
# print(type(vector))
# print(vector.toarray())
# Word Frequencies with TfidfVectorizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# text = ["The quick brown fox jumped over the lazy dog.",
#         "The dog.",
#         "The fox"]
# vectorizer = TfidfVectorizer()
# vectorizer.fit(text)
# print(vectorizer.vocabulary_)
# print(vectorizer.idf_)
# vector = vectorizer.transform([text[0]])
# print(vector.shape)
# print(vector.toarray())

from sklearn.feature_extraction.text import HashingVectorizer
text = ["The quick brown fox jumped over the lazy dog."]
vectorizer = HashingVectorizer(n_features=20)
vector = vectorizer.transform(text)
print(vector.shape)
print(vector.toarray())