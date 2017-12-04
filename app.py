import tensorflow as tf
from model import Model
import numpy as np
from preproc import *
import gensim
from sklearn.model_selection import train_test_split

def get_data(vocabsize,ninput,model):
  tweets = get_tweets()[1:]
  full_set = create_training_set(tweets,n_input=ninput)
  vectors = convertSamplesToVectors(full_set,model)
  x_set, y_set = createTraining(vectors,ninput-1,1,vocabsize)
  data_train, data_test, labels_train, labels_test = train_test_split(x_set, y_set, test_size=0.2, random_state=42)
  labels_train = np.reshape(labels_train,(len(labels_train),vocabsize))
  labels_test = np.reshape(labels_test,(len(labels_test),vocabsize))
  return data_train, labels_train,data_test, labels_test

def shuffle(a, b, rand_state):
   rand_state.shuffle(a)
   rand_state.shuffle(b)

def config1():
  conf = dict()
  conf['n_inputs'] = 300
  conf['n_classes'] = 300
  conf['n_timesteps'] = 4
  conf['hidden_size'] = 512
  conf['hidden_size_small'] = 300
  return conf

def get_batch(data_x,data_y,current,batch_size):
  batchX = data_x[current:current+batch_size]
  batchY = data_y[current:current+batch_size]
  return batchX, batchY


def get_var(all_vars,name):
    for i in range(len(all_vars)):
        if all_vars[i].name.startswith(name):
            return all_vars[i]
    return None
def main():
    run_var  = '2_input_d_3_dense_d_50_drop_40_data_b32'
    batch_size = 32

    #load config and hyper params
    config = config1()

    data = tf.placeholder("float",[None,config['n_timesteps'],config['n_inputs']])
    labels = tf.placeholder("float",[None,config['n_classes']])
    drop_prob = tf.placeholder(tf.float32 ,name = 'drop_prob')
    drop_prob2 = tf.placeholder(tf.float32, name = 'drop_prob2')

    #create model
    model = Model(data, labels,drop_prob,drop_prob2,config)

    #load word embedding
    word_2_vec_model = gensim.models.Word2Vec.load('word_model_all_300_50_donald_larger_dataset_better')
    trainX, trainY, testX, testY = get_data(config['n_inputs'], config['n_timesteps'] + 1, word_2_vec_model)

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())

      writer = tf.summary.FileWriter('%s/%s' % ("./real", run_var), graph=sess.graph)
      tf.summary.scalar("loss",model.loss)


      #tensorboard
      all_vars= tf.global_variables()

      lstm_0b = get_var(all_vars,'prediction/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias')
      lstm_1b = get_var(all_vars,'prediction/rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias')

      lstm_0k = get_var(all_vars,'prediction/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel')
      lstm_1k = get_var(all_vars,'prediction/rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel')


      tf.summary.histogram("prediction/cell_0/basic_lstm_cell/kernel", lstm_0k)
      tf.summary.histogram("prediction/cell_1/basic_lstm_cell/kernel", lstm_1k)

      tf.summary.histogram("prediction/cell_0/basic_lstm_cell/bias", lstm_0b)
      tf.summary.histogram("prediction/cell_1/basic_lstm_cell/bias", lstm_1b)

      fc0_w = get_var(all_vars,'prediction/fc0/weights')
      fc1_w = get_var(all_vars,'prediction/fc1/weights')
      fc2_w = get_var(all_vars,'prediction/fc2/weights')

      fc0_b = get_var(all_vars,'prediction/fc0/biases')
      fc1_b = get_var(all_vars,'prediction/fc1/biases')
      fc2_b = get_var(all_vars,'prediction/fc2/biases')

      tf.summary.histogram("prediction/fc0/weights", fc0_w)
      tf.summary.histogram("prediction/fc1/weights", fc1_w)
      tf.summary.histogram("prediction/fc2/weights", fc2_w)

      tf.summary.histogram("prediction/fc0/biases", fc0_b)
      tf.summary.histogram("prediction/fc1/biases", fc1_b)
      tf.summary.histogram("prediction/fc2/biases", fc2_b)

      summary_op = tf.summary.merge_all()

      validation_summary = tf.summary.scalar("validation_loss",model.loss)

      saver = tf.train.Saver()
      print('starting')
      for epoch in range(50):
        iter=0
        while iter < len(trainX) - batch_size:
          batch_x,batch_y = get_batch(data_x= trainX, data_y = trainY, current=iter,batch_size=batch_size)
          batch_x = batch_x.reshape((batch_size, config['n_timesteps'], config['n_inputs']))
          summary, _ = sess.run([summary_op,model.optimize], feed_dict={data: batch_x, labels: batch_y , drop_prob: 6.0,drop_prob2:5.0})
          writer.add_summary(summary,(epoch*len(trainX))+iter)
          iter=iter + batch_size
        
        los_test, validation_summ = sess.run([model.loss, validation_summary], feed_dict={data: testX, labels: testY, drop_prob: 1.0, drop_prob2:1.0})
        writer.add_summary(validation_summ,(epoch*len(trainX)))
        print("epoch" + str(epoch))
        print("Testing Loss:", los_test)
        #print out a few predicted words so we can see improvements
        predction_number = 15
        for i in range(0,predction_number):
          pred, prediction_loss =  sess.run([model.prediction, model.loss], feed_dict={data: trainX[i:i+1], labels: trainY[predction_number:predction_number+1], drop_prob: 1.0, drop_prob2:1.0})

          x_in = np.reshape(trainX[i],(config['n_timesteps'],300))
          pred1 = np.reshape(pred[0],(1,300))
          c_word = np.reshape(trainY[i],(1,300))

          sentance = convertSamplesToVectors([x_in],word_2_vec_model,True)
          word_predicted = convertSamplesToVectors([pred1],word_2_vec_model,True)
          correct_word = convertSamplesToVectors([c_word],word_2_vec_model,True)
          print(" ".join(sentance[0]) + " " + word_predicted[0][0])
          print(" ".join(sentance[0]) + " " + correct_word[0][0])
          print("similarity score :" + str(word_2_vec_model.similarity(word_predicted[0][0], correct_word[0][0])))
          print('prediction_loss :' + str(prediction_loss))
          print(" ")
          print("__________________")
        print(" ")
        if epoch%2 == 0:
          file_name = './trained_models2/_saved_'+str(epoch) +'_' + str(run_var)
          saver.save(sess,file_name,global_step=epoch*len(trainX))
          print('saved')
          print(file_name)
          print(" ")
          shuffle(trainX,train)
      file_name = './trained_models2/_saved_final' + str(run_var)
      saver.save(sess,file_name)

if __name__ == '__main__':
  main()