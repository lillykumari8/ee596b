import numpy as np
import pickle

folder_path = '/home/lkumari/data/cifar-10-batches-py/'


def unpickle_org(file_path):
   with open(file_path, 'rb') as f:
      data = pickle.load(f, encoding='latin1')
   return data


def unpickle(file_path):
   with open(file_path, 'rb') as f:
      data = pickle.load(f)
   return data

def save_pickle(data, file_path):
   with open(file_path, 'wb') as f:
      pickle.dump(data, f)


#Step 1: define a function to load traing batch data from directory
def load_training_batch(folder_path,batch_id):
   """
   Args:
      folder_path: the directory contains data files
      batch_id: training batch id (1,2,3,4,5)
   Return:
      features: numpy array that has shape (10000,3072)
      labels: a list that has length 10000
   """

   ###load batch using pickle###
   # with open(folder_path + 'data_batch_' + batch_id, 'rb') as f:
   batch_data = unpickle_org(folder_path + 'data_batch_' + str(batch_id))
   ###fetch features using the key ['data']###
   features = batch_data['data']
   ###fetch labels using the key ['labels']###
   labels = batch_data['labels']
   return features,labels

#Step 2: define a function to load testing data from directory
def load_testing_batch(folder_path):
   """
   Args:
      folder_path: the directory contains data files
   Return:
      features: numpy array that has shape (10000,3072)
      labels: a list that has length 10000
   """

   ###load batch using pickle###
   # with open(folder_path + 'test_batch', 'rb') as f:
   test_data = unpickle_org(folder_path + 'test_batch')
   ###fetch features using the key ['data']###
   features = test_data['data']
   ###fetch labels using the key ['labels']###
   labels = test_data['labels']

   return features,labels

#Step 3: define a function that returns a list that contains label names (order is matter)
"""
   airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
"""
def load_label_names():
   
   meta_data = unpickle_org(folder_path + 'batches.meta')
   labels = meta_data['label_names']

   return labels

#Step 4: define a function that reshapes the features to have shape (10000, 32, 32, 3)
def features_reshape(features):
   """
   Args:
      features: a numpy array with shape (10000, 3072)
   Return:
      features: a numpy array with shape (10000,32,32,3)
   """
   features = features.reshape((features.shape[0], 32, 32, 3))
   return features

#Step 5 (Optional): A function to display the stats of specific batch data.
# def display_data_stat(folder_path,batch_id,data_id):
#    """
#    Args:
#       folder_path: directory that contains data files
#       batch_id: the specific number of batch you want to explore.
#       data_id: the specific number of data example you want to visualize
#    Return:
#       None

#    Descrption: 
#       1)You can print out the number of images for every class. 
#       2)Visualize the image
#       3)Print out the minimum and maximum values of pixel 
#    """
#    pass

#Step 6: define a function that does min-max normalization on input
def normalize(x):
   """
   Args:
      x: features, a numpy array
   Return:
      x: normalized features
   """
   ## x - numpy array in shape (# of images, 3072)
   min_x = np.min(x, axis=1, keepdims=True)
   max_x = np.max(x, axis=1, keepdims=True)
   x = (x - min_x) / (max_x - min_x)

   return x

#Step 7: define a function that does one hot encoding on input
def one_hot_encoding(x):
   """
   Args:
      x: a list of labels
   Return:
      a numpy array that has shape (len(x), # of classes)
   """
   label_names = load_label_names()
   num_classes = len(label_names)

   # num_classes = 10
   ohe_labels = np.zeros((len(x), num_classes))
   
   flat_indx = np.arange(len(x)) * num_classes
   ohe_labels.flat[flat_indx + np.array(x).ravel()] = 1

   return ohe_labels

#Step 8: define a function that perform normalization, one-hot encoding and save data using pickle
def preprocess_and_save(features,labels,filename):
   """
   Args:
      features: numpy array
      labels: a list of labels
      filename: the file you want to save the preprocessed data
   """
   ohe_labels = one_hot_encoding(labels)
   features = normalize(features)
   # features = features_reshape(features)
   data = {}
   data['data'] = features
   data['labels'] = ohe_labels
   save_pickle(data, filename)


   # pass

#Step 9:define a function that preprocesss all training batch data and test data. 
#Use 10% of your total training data as your validation set
#In the end you should have 5 preprocessed training data, 1 preprocessed validation data and 1 preprocessed test data
def preprocess_data(folder_path):
   """
   Args:
      folder_path: the directory contains your data files
   """
   val_features, val_labels = None, []
   for i in range(1, 6):
      features, labels = load_training_batch(folder_path, i)
      if val_features is None:
         val_features = features[:1000]
      else:
         val_features = np.concatenate((val_features, features[:1000]), axis=0)
      val_labels.extend(labels[:1000])

      preprocess_and_save(features[1000:], labels[1000:], folder_path + 'preprcoessed_train_batch_' + str(i))

   features, labels = load_testing_batch(folder_path)
   preprocess_and_save(features, labels, folder_path + 'preprcoessed_test')

   assert len(val_labels) == 5000
   preprocess_and_save(val_features, val_labels, folder_path + 'preprcoessed_val')
   # pass

#Step 10: define a function to yield mini_batch
def mini_batch(features,labels,mini_batch_size):
   """
   Args:
      features: features for one batch
      labels: labels for one batch
      mini_batch_size: the mini-batch size you want to use.
   Hint: Use "yield" to generate mini-batch features and labels
   """
 #  while True:
 #     idx = np.random.choice(features.shape[0], mini_batch_size)
 #     yield features[idx], labels[idx]
   for i in range(0, len(labels), mini_batch_size):
       end = min(i + mini_batch_size, len(labels))
       yield features[i:end], labels[i:end]

#Step 11: define a function to load preprocessed training batch, the function will call the mini_batch() function
def load_preprocessed_training_batch(batch_id,mini_batch_size):
   """
   Args:
      batch_id: the specific training batch you want to load
      mini_batch_size: the number of examples you want to process for one update
   Return:
      mini_batch(features,labels, mini_batch_size)
   """
   file_name = folder_path + 'preprcoessed_train_batch_' + str(batch_id)
   data = unpickle(file_name)

   features, labels = data['data'], data['labels']
   return mini_batch(features,labels,mini_batch_size)


#Step 12: load preprocessed validation batch
def load_preprocessed_validation_batch():
   file_name = folder_path + 'preprcoessed_val'
   data = unpickle(file_name)
   features,labels = data['data'], data['labels']
   return features,labels

#Step 13: load preprocessed test batch
def load_preprocessed_test_batch(test_mini_batch_size):
   file_name = folder_path + 'preprcoessed_test'
   data = unpickle(file_name)
   features, labels = data['data'], data['labels']
   return mini_batch(features,labels,test_mini_batch_size)


#Step 14: load preprocessed test data (full)
def load_preprocessed_test():
    file_name = folder_path + 'preprcoessed_test'
    data = unpickle(file_name)
    features,labels = data['data'], data['labels']
    return features,labels
