from __future__ import print_function
import numpy as np
import random
import collections
import nltk
import gensim
import pickle
from sklearn.model_selection import train_test_split
import csv
from string import punctuation
from dateutil.parser import parse
import logging

def is_url(word):
	if 'http' in word or 't.co' in word or ':' in word or (len(word) > 1 and word[0] == '/') or '=' in word or 'www' in word:
		return True
	return False
def is_number(word):
	if word.isdigit() and len(word) == 4 and word[0] == '2' and word[1] == '0':
		return '2017'
	if word.isdigit():
		return 'randomnumber'
	if is_date(word):
		return 'randomdate'
	elif '#' in word:
		return 'twitterhashtag'
	elif '@' in word or (len(word) > 4 and word[-4:].isdigit()):
		return 'twitterusername'
	elif '%' in word:
		return 'percent'
	elif word[0] == '.' and word != '...':
		return word[1:]
	elif len(word) > 1 and (word[-1] == '.' or word[-1] == '-'):
		return word[:len(word)-1]
	return word

def is_date(string):
    try: 
        parse(string)
        return True
    except ValueError:
        return False

def get_tweets():
	fileNames = ["./data/Donald2.csv"]
	tweets = list()
	for name in fileNames:
		with open(name, "r") as f:
			reader = csv.reader(f, delimiter=',')
			for line in reader:
				if len(line) > 1:
					tweet = nltk.word_tokenize(line[1])
					tweet[:] = [w.lower() for w in tweet if not is_url(w)]
					for i in range(0,len(tweet)):
						tweet[i]  = is_number(tweet[i])
					tweets.append(tweet)
				else:
					print('error at' + str(len(tweets)))
					print(line)
					print(" ")
		print('file finished')
	no_punc = [[word.lower() for word in sent if word not in punctuation and "'" not in word and " " not in word and '``' not in word and '--' not in word] for sent in tweets]
	return no_punc
def get_tweets_some(number_of_tweets):
	fileNames = ["./data/Donald2.csv"]
	tweets = list()
	for name in fileNames:
		with open(name, "r") as f:
			counter = 0
			reader = csv.reader(f, delimiter=',')
			for line in reader:
				if len(line) > 1:
					if counter >= number_of_tweets:
						break
					tweet = nltk.word_tokenize(line[1])
					tweet[:] = [w.lower() for w in tweet if not is_url(w)]
					for i in range(0,len(tweet)):
						tweet[i]  = is_number(tweet[i])
					tweets.append(tweet)
					counter = counter + 1
				else:
					print('error at' + str(len(tweets)))
					print(line)
					print(" ")
		print('file finished')
	no_punc = [[word.lower() for word in sent if word not in punctuation and "'" not in word and " " not in word and '``' not in word and '--' not in word] for sent in tweets]
	return no_punc
def train_word_2_vec(tweets):
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	model = gensim.models.Word2Vec(tweets, iter=50,min_count=1,size=300,workers=4)
	model.save('word_model_all_300_50_donald_larger_dataset_better')
	return model
def create_word_embed():
	tweets = get_tweets()
	print(len(tweets))
	train_word_2_vec(tweets)
def create_data_set(tweets):
	model  = gensim.models.Word2Vec.load('word_model_all_300_50_donald_larger_dataset_better')
	dataset = []
	for i in range(0,len(tweets)):
		tweet_in_vector_format = convert_words_to_vectors(tweets[i],model.wv)
		dataset.append(tweet_in_vector_format)
		if i %1000 == 0:
			print('at ' + str(i))
	return dataset

def create_training_set(tweets,n_input=4,vocab_size=300,flat = True):
	ds = []
	for tweet in tweets:
		i = 0
		if flat == True and len(tweet) > n_input:
			while i <= len(tweet)-n_input:
				sample = tweet[i:i+n_input]
				ds.append(sample)
				i = i + 1
			if len(tweet) % n_input != 0:
				sample = tweet[len(tweet)-n_input:len(tweet)]
				ds.append(sample)
	return ds

def convertSamplesToVectors(tweets,model,reverseOrder = False):
	vectorSample = np.ndarray(shape=(len(tweets),len(tweets[0]),300))
	if reverseOrder == True:
		vectorSample = list()
	# for words in tweet
	for tweet_index in range(0,len(tweets)):
		tweet = tweets[tweet_index]
		vectors_for_tweet = np.ndarray(shape=(len(tweet),300))
		if reverseOrder == True:
			vectors_for_tweet = list()
		for word_index in range(0,len(tweet)):
			word = tweet[word_index]
			if reverseOrder == False:
				vectors_for_tweet[word_index] = np.asarray(model.wv[word])
			else:

				vectors_for_tweet.append(model.similar_by_vector(word, topn=1)[0][0])
		if reverseOrder == True:
			vectorSample.append(vectors_for_tweet)
		else:
			vectorSample[tweet_index] = vectors_for_tweet
	return vectorSample


def createTraining(vectors,x_size,y_size,vocab_size):
	xData = np.ndarray(shape=(len(vectors),x_size,vocab_size))
	yData = np.ndarray(shape=(len(vectors),y_size,vocab_size))
	for  i in range(0,len(vectors)):
		for j in range(0,x_size):
			xData[i][j] = vectors[i][j]
		for k in range(0,y_size):
			yData[i][k] = vectors[i][x_size+k]
	return xData, yData

#takes in list of words and returns list of vectors
def convert_words_to_vectors(words,wv):
	return np.asarray([convert_data_to_vector(w,wv) for w in words])

def convert_data_to_vector(string_data,wv):
	return np.asarray(wv[string_data])

#converts entier dataset into vectors and then reconstructs
def testConversionAndReconstruction():
	model  = gensim.models.Word2Vec.load('word_model_all_300_50_donald_larger_dataset_better')
	tweets = get_tweets()[1:]
	full_set = create_training_set(tweets)
	vectors = convertSamplesToVectors(full_set,model)
	print(vectors.shape)
	rev = convertSamplesToVectors(vectors,model,True)
	print(len(full_set[0]))
	print(len(rev[0]))
	for i in range(0,len(rev)):
		if len(rev[i]) != len(full_set[i]):
			print(i)
			print(rev[i])
			print(full_set[i])
			print('+1')
			print(rev[i+1])
			print(full_set[i+1])
			print('wrong length')
		for j in range(0,len(rev[i])):
			if rev[i][j] != full_set[i][j]:
				print('error')
				print(rev[i][j])
				print(full_set[i][j])
		if i%100== 0:
			print('working')
			print(str(i))
if __name__ == "__main__":
	tweets = get_tweets2()[1:]
	train_word_2_vec(tweets)