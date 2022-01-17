import nltk
import sklearn_crfsuite
import eli5
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from collections import Counter
from nltk.corpus import stopwords
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_recall_fscore_support as score
from tqdm import tqdm

def extract_data(raw_text,mode='train'):
  if mode=='train':
    entries = {'index':[],
              'tokenized_sents':[],
              'label':[]}
  else:
    entries = {'index':[],
              'tokenized_sents':[]}

  current_index = []
  current_sentence = []
  current_label = []
  for i, line in enumerate(raw_text) : 

    if line=="\n":
        entries['index'].append(current_index)
        entries['tokenized_sents'].append(current_sentence)
        if mode=='train':
          entries['label'].append(current_label)
        
        current_index = []
        current_sentence = []
        current_label = []

    else : 
        if mode=='train':
          index, word, label = line.split("\t")
          current_index.append(index)
          current_sentence.append(word)
          current_label.append(label.strip())
        else:
          index, word = line.split("\t")
          current_index.append(index)
          current_sentence.append(word.strip())
  if mode=='train':
          entries['label'].append(current_label)
  entries['index'].append(current_index)
  entries['tokenized_sents'].append(current_sentence)
  return entries

raw_text = open('S21-gene-train.txt').readlines()  
entries=extract_data(raw_text)
df = pd.DataFrame(entries)

def format_data(dataset):
    data = []
    for l1,l2 in zip(dataset['tokenized_sents'].tolist(), dataset['label'].tolist()):
        temp_list = []
        for ele1, ele2 in zip(l1, l2):
            temp_list.append((ele1, ele2))
        data.append(temp_list)
    return data

np.random.seed(10)
print("Formatting data according to model specifications...")
train_df, test_df = train_test_split(df, test_size=0.2)
train_data = format_data(train_df)
test_data = format_data(test_df)

def get_words(entity,df):
  words=[]
  all_words = []
  all_labels = []
  for i, row in df.iterrows():
      for k, ele in enumerate(row['label']):
          all_words.append(row['tokenized_sents'][k])
          all_labels.append(ele)
          if ele == entity:
              words.append(row['tokenized_sents'][k])  
  return words

b_words = get_words('B',df)
i_words = get_words('I',df)
o_words = get_words('O',df)

""" Storing the 20 most occurring words in each tag."""
i_most_common = set([ele[0] for ele in Counter(i_words).most_common(20)])
b_most_common = set([ele[0] for ele in Counter(b_words).most_common(20)])
o_most_common = set([ele[0] for ele in Counter(o_words).most_common(20)])

"""Checking for the unique words in each tag set which does not occur in other tag sets."""
def find_diff(i, ii, iii):
    diff = i.difference(ii)
    diff = diff.difference(iii)
    return diff

class buildFeatures():

  def create_features(self,sent,i,test=False):
    self.word = sent[i][0]
    if test:
      self.word = sent[i]
    features = {
        'bias': 1.0,
        'word.lower()': self.word.lower(),
        'word.upper()': self.word.upper(),
        'word[-3:]': self.word[-3:],
        'word.isupper()': self.word.isupper(),
        'word.istitle()': self.word.istitle(),
        'word.isdigit()': self.word.isdigit(),
        'word.islower()': self.word.islower(),
        'word.is_dash()' : self.check_word(self.word,'-'),
        'word.i_most_common()' : self.word in find_diff(i_most_common, b_most_common, o_most_common),
        'word.b_most_common()' : self.word in find_diff(b_most_common, i_most_common, o_most_common),
        'word.o_most_common()' : self.word in find_diff(o_most_common, i_most_common, b_most_common),
        'word.ene()' : self.check_word(self.word,'ene'),
        'word.ein()' : self.check_word(self.word,'ein'),
        'word.ase()': self.check_word(self.word,'ase')
    }
    if i > 0:
        word1 = sent[i-1][0]
        if test == True:
            word1 = sent[i-1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:word.isdigit()': self.word.isdigit(),
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        if test == True:
            word1 = sent[i+1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:word.isdigit()': self.word.isdigit(),
        })
    else:
        features['EOS'] = True

    return features
  def check_word(self,word,string):
    if word[-3:].lower() == string.lower():
          return True
    return False

  def sentence_to_features(self,sent):
      return [self.create_features(sent, i) for i in range(len(sent))]

  def sentence_to_test_features(self,sent):
      return [self.create_features(sent, i, test=True) for i in range(len(sent))]

  def sentence_to_labels(self,sent):
      return [label for token, label in sent]

  def sentence_to_tokens(self,sent):
      return [token for token, label in sent]

print("Building feature vectors for training...")
build_features= buildFeatures()
X_train = [build_features.sentence_to_features(s) for s in train_data]
y_train = [build_features.sentence_to_labels(s) for s in train_data]

X_test = [build_features.sentence_to_features(s) for s in test_data]
y_test = [build_features.sentence_to_labels(s) for s in test_data]


print("Training CRF model...")
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=200,
    all_possible_transitions=False,
)
crf.fit(X_train, y_train)
eli5.show_weights(crf, top=30)

# Evaluation of individual labels

mlb = MultiLabelBinarizer()
preds = crf.predict(X_test)

bin_preds = mlb.fit_transform(preds)
bin_test = mlb.fit_transform(y_test)

precision, recall, fscore, support = score(bin_test, bin_preds)

scores = pd.DataFrame()
scores['labels'] = ['B', 'I', 'O']
scores['precision'] = precision
scores['recall'] = recall 
scores['fscore'] = fscore 
scores['support'] = support

scores.style.format({
    'precision': '{:,.2%}'.format,
    'recall': '{:,.2%}'.format,
    'fscore': '{:,.2f}'.format,
})

print(scores)

def write_to_file(filename, df, label_col):
    sentences = df.tokenized_sents.tolist()
    labels = df[label_col].tolist()
    
    with open(filename, 'w') as f:
        for k,ele in enumerate(sentences):
            for i,val in enumerate(zip(ele, labels[k])):
                f.write("\t".join([str(i+1),val[0],val[1]]) + "\n")
            f.write("\n")

test_df['preds'] = preds

write_to_file('goldstandardfile.txt', test_df, 'label')
write_to_file('yoursystemoutput.txt', test_df, 'preds')

### Starting Test Data Evaluataion
data = open('F21-gene-test.txt').readlines()
entries= extract_data(data,mode='test')
test_df = pd.DataFrame(entries)

def format_data(dataset):
    data = []
    for l1 in dataset['tokenized_sents'].tolist():
        temp_list = []
        for ele1 in l1:
            temp_list.append(ele1)
        data.append(temp_list)
    return data


test_final = [build_features.sentence_to_test_features(s) for s in format_data(test_df)]

fin_preds = crf.predict(test_final)

test_df['preds'] = fin_preds

print("Writing outputs to file..")
print("Completed successfully..")
write_to_file('test_data_predictions.txt', test_df, 'preds')
