# Reference code
####################
# nltk - Natural Language Tool Kit, do dtuff with words
# numpy - Array management
# tflearn - Tensorflow deep learning lib
# tensorflow - Bruh AI stuff
import json  # To read json data from intents.
import random
import tensorflow
import tflearn
import numpy
import nltk
import pickle
from nltk.stem.lancaster import LancasterStemmer  # stem words
from tensorflow.python.framework import ops # Fix the default graph error
stemmer = LancasterStemmer()


with open("intents.json") as file:  # open json file and put it in data
    data = json.load(file)

try: # rb = read bytes
	# x
	with open("data.pickle", "rb") as f:
		words, labels, training, output = pickle.load(f)

except:
	# Data pre-processing
	#####################
	words = []
	labels = []
	docs_x = []
	docs_y = []

	for intent in data["intents"]:  # Loop through all the dictionaries
	    for pattern in intent["patterns"]:  # Start stemming
	        # Stemming takes each word in a pattern and bring it down to the root word to get the main meaning of the word
	        # Tokenize - Get all words in our pattern
	        # return a list with different words
	        wrds = nltk.word_tokenize(pattern)
	        words.extend(wrds)  # Extend the list
	        docs_x.append(wrds)
	        docs_y.append(intent["tag"])

	        if intent["tag"] not in labels:  # greeting, goodbye etc
	            labels.append(intent["tag"])  # get all tags

	words = [stemmer.stem(w.lower()) for w in words if w != "?"]
	words = sorted(list(set(words)))  # Remove dupes

	labels = sorted(labels)

	# This piece of code uses one hot encoding

	training = []
	output = []

	out_empty = [0 for _ in range(len(labels))]
	for x, doc in enumerate(docs_x):
	    bag = []

	    wrds = [stemmer.stem(w) for w in doc]

	    for w in words:
	        if w in wrds:
	            bag.append(1)
	        else:
	            bag.append(0)

	    output_row = out_empty[:]
	    output_row[labels.index(docs_y[x])] = 1

	    training.append(bag)
	    output.append(output_row)

	training = numpy.array(training)
	output = numpy.array(output)

	training = numpy.array(training)
	output = numpy.array(output)

	with open("data.pickle", "wb") as f: # WB = write bytes
		pickle.dump((words, labels, training, output), f)	
# Model
########
ops.reset_default_graph()

# MAIN AI FUNCTIONALITY
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
	# Load the model
	model.load("model.tflearn")
except:
	# Train the model :D
	model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
	model.save("model.tflearn")


def bag_of_words(s, words): # Turn sentence to bag of words
	bag = [0 for _ in range(len(words))] # Creat blank bag of words

	s_words = nltk.word_tokenize(s) # Get list of tokenized words
	s_words = [stemmer.stem(word.lower()) for word in s_words] # Stem all words

	for se in s_words:
		for i, w in enumerate(words):
			if w == se:
				bag[i] = 1

	return numpy.array(bag)

def chat(): # Chat with the bot :D
	print("The bot's ready to go! (Type quit to exit)")
	while True:
		inp = input("You: ")
		if inp.lower() == "quit":
			break

		results = model.predict([bag_of_words(inp, words)]) #predict a reply
		results_index = numpy.argmax(results) # get indes of greatest value in our list
		tag = labels[results_index] # give the label

		for tg in data["intents"]: # Get the response text
			if tg['tag'] == tag:
				responses = tg['responses']

		print(random.choice(responses)) # Choose any response from the appropriate tag
chat()