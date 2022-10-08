import string
import os
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import brown, stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer,TfidfTransformer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#Step 1 Install Python and NLTK
print(brown.words())
print()

#Step2 tokenize, stop word removal, stemming
data_path = '/Users/maksy/Desktop/demodata'
token_dict = {}
rawDoc = {}
all_stemmed_words = []
ps = PorterStemmer()
stop_words = set(stopwords.words("english"))
punc = set(string.punctuation)
#output_file = open('/Users/maksy/Desktop/demoDataOutput.txt','w')
text_lines = "\n"

for subir, dirs, files in os.walk(data_path):
    for file in files:
        file_path = subir + os.path.sep + file
        file_contents = open(file_path,'r')
        if '.txt' in file_path:
            text = file_contents.read()
            lowered = text.lower()
            token_dict[file] = lowered
            rawDoc[file] = lowered
            file_contents.close()
num_docs = len(token_dict)

doc_names = []
for file_name in token_dict.keys():
    doc_names.append(file_name)

#tokenizing
for x,file in enumerate(token_dict.keys()):
    words = word_tokenize(token_dict[file])
    print("Tokenized file %s: %s" % (doc_names[x],words))
    stemmed_words = []
    for w in words:
        #removing stop words and punctuation
        if w not in stop_words | punc:
            #stemming
            stemmed_words.append(ps.stem(word=w))
    print("Stemming and stop word removed %s: %s \n" % (doc_names[x], stemmed_words))
    all_stemmed_words.append(stemmed_words)

#Step 3 tf-idf
tfidf = TfidfVectorizer(input=all_stemmed_words,stop_words='english')
tfs = tfidf.fit_transform(token_dict.values())
doc_matrix = tfs.toarray()
set_vocab = tfidf.get_feature_names()
df = pd.DataFrame(doc_matrix, columns=set_vocab)
df.index = doc_names
df.to_csv('/Users/maksy/Desktop/out.csv')

#Step 4 pairwise cosine similarity
for i in range(0,len(doc_names)):
    for j in range(i,len(doc_names)):
        print("Cosine similarity between %s to %s is %s. \n" % (
              doc_names[i], doc_names[j], cosine_similarity(tfs[i,],tfs[j,])))