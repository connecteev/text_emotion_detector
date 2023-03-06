import re 
import pickle
from collections import Counter
from emotions import create_feature
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


with open('saved_model', 'rb') as f:
    clf = pickle.load(f)

l = ["joy", 'fear', "anger", "sadness", "disgust", "shame", "guilt"]
l.sort()
label_freq = {}
for label, _ in data: 
    label_freq[label] = label_freq.get(label, 0) + 1


emoji_dict = {"joy":"😂", "fear":"😱", "anger":"😠", "sadness":"😢", "disgust":"😒", "shame":"😳", "guilt":"😳"} 

print("Follow the instructions and input sentences to predict the emotion of each..") 

var = 1
texts = []
while var == 1:
    t = str(input("Enter sentence: "))
    texts.append(t)
    var = int(input("Enter 0 to check emotions, 1 to add another sentence"))
    if var == 0:
        break

for text in texts: 
   features = create_feature(text, nrange=(1, len(texts)))
   features = vectorizer.transform(features)
   prediction = clf.predict(features)[0]
   print( text,emoji_dict[prediction])
