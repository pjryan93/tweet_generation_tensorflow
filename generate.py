import gensim
import numpy as np
import tensorflow as tf
from app import *
from preproc import *

def generate_tweet(tweets,ninput,word_2_vec_model,words_to_generate=None,model_file_name=None,checkpoint_dir = './trained_models/'):
    tf.reset_default_graph()
    run_var  = '2_input_d_3_dense_d'
    batch_size = 64

    #load config and hyper params
    config = config1()

    data = tf.placeholder("float",[None,config['n_timesteps'],config['n_inputs']])
    labels = tf.placeholder("float",[None,config['n_classes']])
    drop_prob = tf.placeholder(tf.float32 ,name = 'drop_prob')
    drop_prob2 = tf.placeholder(tf.float32, name = 'drop_prob2')

    #create model
    m1 = Model(data, labels,drop_prob,drop_prob2,config)
    #load word embedding
    generated_tweets = []
    with tf.Session() as sess:
        if model_file_name is None:
            previous_variables = [var_name for var_name, _ in tf.contrib.framework.list_variables(tf.train.latest_checkpoint(checkpoint_dir))]
            restore_map = {variable.op.name:variable for variable in tf.global_variables() if variable.op.name in previous_variables}
            tf.contrib.framework.init_from_checkpoint(tf.train.latest_checkpoint(checkpoint_dir), restore_map)
        else:
            previous_variables = [var_name for var_name, _ in tf.contrib.framework.list_variables(model_file_name)]
            restore_map = {variable.op.name:variable for variable in tf.global_variables() if variable.op.name in previous_variables}
            tf.contrib.framework.init_from_checkpoint(model_file_name,restore_map)
        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver()
        sess.run(init_op)
        for i, tweet in enumerate(tweets):
            if len(tweet) >= ninput-1:
                full_set = [tweets[i][0:ninput-1]]
                tweet_start = " ".join(full_set[0][:len(full_set[0])])
                sample_correct = full_set[0][0]
                last_predicted_word = full_set[0][-1]
                s = full_set[0][0:len(full_set[0])-1]
                if words_to_generate is None:
                    number_of_words == len(tweets[i])
                else:
                    number_of_words = words_to_generate
                for k in range(0,number_of_words):
                    s.append(last_predicted_word)
                    x_set = convertSamplesToVectors([s],word_2_vec_model)
                    pred1 = sess.run([m1.prediction], feed_dict={data: x_set, drop_prob: 1.0, drop_prob2:1.0})
                    word_predicted = convertSamplesToVectors([pred1[0]],word_2_vec_model,True)
                    tweet_start = tweet_start + " " + str(word_predicted[0][0])
                    last_predicted_word = str(word_predicted[0][0])
                    s = s[1:]
                generated_tweets.append(tweet_start)
                print('generated tweet')
                print(tweet_start)
                print('base_tweet')
                print(" ".join(tweets[i]))
                print('---------------------')
        return generated_tweets

if __name__ == '__main__':
    tensorflow_model_name = ''
    tweets = get_tweets_some(5)[1:]
    ninput = 5
    word_2_vec_model = gensim.models.Word2Vec.load('word_model_all_300_50_donald_larger_dataset_better')
    generated = generate_tweet(tweets=tweets,ninput=ninput,word_2_vec_model=word_2_vec_model)
    print('generated tweets')
    for tweet in generated:
        print(tweet)
        print("-------")




