# Start with testing code from https://thecleverprogrammer.com/2021/02/19/text-emotions-detection-with-machine-learning/

import re 
import pickle
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib

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

max_sentences = 5
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

# print("\ndummy_sentences\n") 
# for x in range(num_dummy_sentences):
#   print(x, dummy_sentences[x])

num_all_sentences = len(all_sentences)

print("\nWe will be predicting emotion for each of the following", num_all_sentences, "sentences...\n") 
for x in range(num_all_sentences):
    print(x, all_sentences[x])






def read_data(file):
    data = []
    with open(file, 'r')as f:
        for line in f:
            line = line.strip()
            label = ' '.join(line[1:line.find("]")].strip().split())
            text = line[line.find("]")+1:].strip()
            data.append([label, text])
    return data

print("\n\n***** Reading Training Data *****\n\n")
file = 'text.txt'
data = read_data(file)
print("Number of instances: {}".format(len(data)))




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




def convert_label(item, name): 
    items = list(map(float, item.split()))
    label = ""
    for idx in range(len(items)): 
        if items[idx] == 1: 
            label += name[idx] + " "
    
    return label.strip()

emotions = ["joy", 'fear', "anger", "sadness", "disgust", "shame", "guilt"]

X_all = []
y_all = []
for label, text in data:
    y_all.append(convert_label(label, emotions))
    X_all.append(create_feature(text, nrange=(1, 4)))





X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size = 0.2, random_state = 123)

def train_test(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test))
    return train_acc, test_acc

from sklearn.feature_extraction import DictVectorizer
vectorizer = DictVectorizer(sparse = True)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)







svc = SVC()
lsvc = LinearSVC(random_state=123)
rforest = RandomForestClassifier(random_state=123)
dtree = DecisionTreeClassifier()

clifs = [svc, lsvc, rforest, dtree]

# train and test them 
print("| {:25} | {} | {} |".format("Classifier", "Training Accuracy", "Test Accuracy"))
print("| {} | {} | {} |".format("-"*25, "-"*17, "-"*13))
for clf in clifs: 
    clf_name = clf.__class__.__name__
    train_acc, test_acc = train_test(clf, X_train, X_test, y_train, y_test)
    print("| {:25} | {:17.7f} | {:13.7f} |".format(clf_name, train_acc, test_acc))


with open('saved_model', 'wb') as f:
    pickle.dump(clf, f)
with open('saved_model', 'rb') as f:
    clf = pickle.load(f)
# Can also use joblib above as:
# joblib.dump(clf, 'saved_model')
# clf = joblib.load('saved_model')
    

l = ["joy", 'fear', "anger", "sadness", "disgust", "shame", "guilt"]
l.sort()
label_freq = {}
for label, _ in data: 
    label_freq[label] = label_freq.get(label, 0) + 1

# print the labels and their counts in sorted order 
for l in sorted(label_freq, key=label_freq.get, reverse=True):
    print("{:10}({})  {}".format(convert_label(l, emotions), l, label_freq[l]))








emoji_dict = {"joy":"😂", "fear":"😱", "anger":"😠", "sadness":"😢", "disgust":"😒", "shame":"😳", "guilt":"😳"}
print("Input four sentences to predict the emotion of each:") 

def predict_emotion(txt):
    features = create_feature(txt, nrange=(1, 4))
    features = vectorizer.transform(features)
    prediction = clf.predict(features)[0]
    print(txt,emoji_dict[prediction])

print("\nPredicting emotion for each of the", num_all_sentences, "sentences...\n") 
for text in all_sentences:
    predict_emotion(text)

print('\a')
# quit()
