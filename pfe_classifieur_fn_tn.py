#!/usr/bin/env python
# coding: utf-8

# In[251]:


#https://machinelearningmastery.com/develop-word-embedding-model-predicting-movie-review-sentiment/
#fichiers fake news : fn001, fn002, fn003...
#fichiers true news : tn001, tn002, tn003...
#fichiers démystification : dm001, dm002, dm003...

from string import punctuation
from os import listdir
from collections import Counter
from nltk.corpus import stopwords

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r', encoding="utf-8")
	#print(filename)
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# turn a doc into clean tokens
def clean_doc(doc):
	# split into tokens by white space / split par espace
	tokens = doc.split()
	# remove punctuation from each token / enlever la ponctuation pour chaque token
	table = str.maketrans('', '', punctuation)
	tokens = [w.translate(table) for w in tokens]
	# remove remaining tokens that are not alphabetic #enlever tokens non alphabétiques (nombres, caractères spéciaux...)
	tokens = [word for word in tokens if word.isalpha()]
	# filter out stop words / enlever les mots vides (type "de", "un",...)
	stop_words = set(stopwords.words('french')) #mots vides français
	custom_sw= {"les","Les","le","Le","la","La"} # on ajoute des mots vides qu'on veut enlever qui ne sont pas déjà dans stop_words
	stop_words |= custom_sw
	tokens = [w for w in tokens if not w in stop_words] #si le mot est dans tokens et n'est pas dans stop_words, on le garde
	# filter out short tokens
	tokens = [word for word in tokens if len(word) > 1] #on enlève les tokens courts, càd d'une lettre, q'il reste peut-être
	return tokens

# load doc and add to vocab
def add_doc_to_vocab(filename, vocab):
	# load doc / charge texte du document
	doc = load_doc(filename) 
	# clean doc / nettoie le document pour avoir les tokens
	tokens = clean_doc(doc)
	# update counts / ajoute les tokens récupérés au vocabulaire total
	vocab.update(tokens)

# load all docs in a directory
def process_docs(directory, vocab, is_trian): #is_trian = ??????????? laissé car on ne sait pas exactement ce que c'est donc on n'y touche pas
	# walk through all files in the folder / on parcourt tous les fichiers dans le dossier au chemin désigné par directory
	for filename in listdir(directory):
		# skip any reviews in the test set
        # filtrage des documents : on sélectionne les documents qui ne sont PAS assignés au set sur lequel on testera
        # les documents sur lesquels on va tester sont les documents de numéro 50 ou plus
        # on sélectionne donc les documents dont le numéro est inférieur à 50
		if is_trian and filename.startswith('04',2,4):
			continue #L’instruction continue, également empruntée au C, fait passer la boucle à son itération suivante
		if not is_trian and not filename.startswith('04',2,4):
			continue
		# create the full path of the file to open
		path = directory + '/' + filename
		# add doc to vocab
		add_doc_to_vocab(path, vocab)

# define vocab
vocab = Counter() #counter = classe de dictionnaire de comptage d'occurrences (https://pymotw.com/2/collections/counter.html / https://docs.python.org/2/library/collections.html)
# add all docs to vocab / on ajoute les tokens nettoyés de tous les documents au vocabulaire
#(intégration des éléments linguistiques : possibilité de leur donner plus de poids ici ???)
process_docs('D:/m2_s1/PFE/corpus_PFE_2/fn', vocab, True)
process_docs('D:/m2_s1/PFE/corpus_PFE_2/tn', vocab, True)
#process_docs('D:/m2_s1/PFE/corpus_PFE_2/dm', vocab, True)
# print the size of the vocab
print(len(vocab))
# print the top words in the vocab
print(vocab.most_common(50))


# In[252]:


# keep tokens with a min occurrence / on enlève les tokens qui n'apparaissent qu'une fois (hapax)
min_occurane = 2
tokens = [k for k,c in vocab.items() if c >= min_occurane]
print(len(tokens))


# In[253]:


# save list to file / sauvegarde liste finales des tokens dans un fichier
def save_list(lines, filename):
	# convert lines to a single blob of text
	data = '\n'.join(lines)
	# open file
	file = open(filename, 'w', encoding="utf-8")
	# write text
	file.write(data)
	# close file
	file.close()

# save tokens to a vocabulary file
save_list(tokens, 'D:/m2_s1/PFE/vocab_fn.txt')


# In[254]:


# load the vocabulary / charge vocab dans une variable, puis split pour avoir des mots séparés, puis mis en set
# utile si on veut charger un set de vocab prédéfini
vocab_filename = 'D:/m2_s1/PFE/vocab_fn.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)


# In[255]:


# turn a doc into clean tokens / on convertit les documents en tokens propres
# on ne garde du document que les tokens dans le vocabulaire
# cela donne une liste de tokens par document qui est ensuite convertie
# en une seule grande string spécifique à ce document
def clean_doc2(doc, vocab):
	# split into tokens by white space
	tokens = doc.split()
	# remove punctuation from each token
	table = str.maketrans('', '', punctuation)
	tokens = [w.translate(table) for w in tokens]
	# filter out tokens not in vocab
	tokens = [w for w in tokens if w in vocab]
	tokens = ' '.join(tokens)
	return tokens


# In[256]:


# load all docs in a directory
def process_docs2(directory, vocab, is_trian):
	documents = list()
	# walk through all files in the folder / parcourt tous les fichiers dans le dossier
	for filename in listdir(directory):
		# skip any doc in the test set
		if is_trian and filename.startswith('04',2,4):
			continue 
		if not is_trian and not filename.startswith('04',2,4):
			continue
        # create the full path of the file to open
		path = directory + '/' + filename
		# load the doc
		doc = load_doc(path)
		# clean doc / on récupère la string de token du document
		tokens = clean_doc2(doc, vocab)
		# add to list / on ajoute à la liste qui récapitule toutes ces string
		documents.append(tokens)
	return documents

# load all training reviews
fake_news = process_docs2('D:/m2_s1/PFE/corpus_PFE_2/fn', vocab, True)
true_news = process_docs2('D:/m2_s1/PFE/corpus_PFE_2/tn', vocab, True)
#demys = process_docs2('D:/m2_s1/PFE/corpus_PFE_2/dm', vocab, True)
train_docs = fake_news + true_news # + demys


# In[257]:


# on crée le "keras embedding layer" qui va permettre au classifieur de fonctionner.
# il requiert des inputs d'entiers où chaque entier renvvoie à un seul token qui a une représentation vectorielle spécifique dans l'imbrication (embedding)
# c'est accompli grâce à la classe Tokenizer qui sert à vectoriser les texteS/tokens
import keras
from keras.preprocessing.text import Tokenizer
# create the tokenizer
tokenizer = Tokenizer()
# fit the tokenizer on the documents
# met à jour un vocabulaire interne basé sur les textes mis en argument, qui sont déjà nettoyés
tokenizer.fit_on_texts(train_docs)


# In[258]:


# sequence encode
# examine les mots de chaque texte et les transforme en entiers
# on a donc des textes transformés en séquences d'entiers
encoded_docs = tokenizer.texts_to_sequences(train_docs)


# In[259]:


# pad sequences
# pour que keras soit efficace il faut que tous les documents (donc en fait toutes les séquences d'entiers) aient la même longueur
# ici, on va "rembourrer" les séquences pour qu'elles soient toutes aussi longue que la séquence la plus longue
# (il y a d'autres méthodes)
from keras.preprocessing.sequence import pad_sequences
max_length = max([len(s.split()) for s in train_docs]) #on cherche la séquence la plus longue
xtrain = pad_sequences(encoded_docs, maxlen=max_length, padding='post')


# In[260]:


# define training labels
from numpy import array
# à adapter en fonction du nombres de documents entrés comme données
ytrain = array([0 for _ in range(40)] + [1 for _ in range(40)]) # + [2 for _ in range(900)])


# In[261]:


# maintenant on s'occupe du dataset de test : on fait le même traitement, mais pour ce dataset
# donc nettoyage des docs, récupération des strings, conversion en séquences d'entiers
# load all test reviews
fake_news = process_docs2('D:/m2_s1/PFE/corpus_PFE_2/fn', vocab, False)
true_news = process_docs2('D:/m2_s1/PFE/corpus_PFE_2/tn', vocab, False)
#demys = process_docs2('D:/m2_s1/PFE/textes_demystification', vocab, False)
test_docs = fake_news+true_news #+demys
print(len(fake_news))
print(len(true_news))
# sequence encode
encoded_docs = tokenizer.texts_to_sequences(test_docs)
# pad sequences
xtest = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# define test labels
ytest = array([0 for _ in range(10)] + [1 for _ in range(10)]) # + [2 for _ in range(900)])


# In[262]:


# define vocabulary size (largest integer value) / on cherche l'entier le plus haut
# vu qu'un entier différent est attribué à chaque mot distinct, l'entier le plus haut est égal à la longueur du vocabulaire
# càd le nombre total de mots différents, + 1 (pour être sur d'avoir le plus haut possible et des bornes min/max)
vocab_size = len(tokenizer.word_index) + 1


# In[263]:


# define model
# ref pour bidouillage : https://www.actuia.com/keras/debuter-avec-le-modele-sequentiel-de-keras/
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=max_length)) # on utilise un espace vectoriel à 100 dimensions mais on peut essayer d'autres valeurs (50 ou 150 dimensions par ex)
model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
print(model.summary())


# In[264]:


print(xtest.shape)
print(ytest.shape)


# In[265]:


# compile network
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
model.fit(xtrain, ytrain, epochs=10, verbose=2)


# In[266]:


# evaluate
loss, acc = model.evaluate(xtest, ytest, verbose=0)

print('Test Accuracy: %f' % (acc*100))

