import re
import pickle
# from emotions import create_feature
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
import joblib


def ngram(token, n): 
    output = []
    for i in range(n-1, len(token)): 
        ngram = ' '.join(token[i-n+1:i+1])
        output.append(ngram) 
    return output

def create_feature(text, nrange=(1, 1)):
    text_features = [] 
    text = text.lower() 
    text_alphanum = re.sub('[^a-z0-9#]', ' ', text)
    for n in range(nrange[0], nrange[1]+1): 
        text_features += ngram(text_alphanum.split(), n)    
    text_punc = re.sub('[a-z0-9]', ' ', text)
    text_features += ngram(text_punc.split(), 1)
    return Counter(text_features)




print("\n\n***** Welcome to Emotion Predictor! *****\n\n")

dummy_sentences = [
    "This looks so impressive",
    "I have a fear of dogs",
    "My dog died yesterday",
    "I don't love you anymore..!",
    "Go kill yourself you piece of shit!"
]
num_dummy_sentences = len(dummy_sentences)

for x in range(num_dummy_sentences):
  print(x, dummy_sentences[x])

num_sentences = input("How many additional sentences would you like to predict the emotion for? ")
if num_sentences.isdigit():
    num_sentences = int(num_sentences)
    if (num_sentences > max_sentences):
        print("Error!", num_sentences, "is more than the maximum number of sentences:", max_sentences)
        quit()
else:
    print("Error:", num_sentences, "is not an integer.")
    quit()

all_sentences = dummy_sentences.copy()
for x in range(num_sentences):
    sentence_num = num_dummy_sentences + x
    sent = str(input("Enter sentence # " + str(sentence_num) + " : "))
    all_sentences.append(sent)

num_all_sentences = len(all_sentences)
print("\nWe will be predicting emotion for each of the following", num_all_sentences, "sentences...\n") 
for x in range(num_all_sentences):
    print(x, all_sentences[x])


print("\nPredicting emotion for each of the", num_all_sentences, "sentences...\n") 


with open('saved_model', 'rb') as f:
    clf = pickle.load(f)

emoji_dict = {"joy":"😂", "fear":"😱", "anger":"😠", "sadness":"😢", "disgust":"😒", "shame":"😳", "guilt":"😳"}
print("Input four sentences to predict the emotion of each:") 

vectorizer = DictVectorizer(sparse = True)

def predict_emotion(txt):
    features = create_feature(txt, nrange=(1, 4))
    features = vectorizer.transform(features)
    prediction = clf.predict(features)[0]
    print(txt,emoji_dict[prediction])

for text in all_sentences:
    predict_emotion(text)

print("Done\a")
# quit()
