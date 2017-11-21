import pandas as pd
import numpy as np
from collections import Counter
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
import time
import pickle
from textfeatures import TextFeatureExtractor
from sparsehelpers import dict_to_csr_matrix
from sklearn.model_selection import train_test_split
import json


def timer(func):
	def wrapper(*args, **kwargs):
		t_start = time.time()
		res = func(*args, **kwargs)
		print("f: {} # elapsed time: {:.0f} m {:.0f}s".format(
			func.__name__.upper(), *divmod(time.time() - t_start, 60)))
		return res
	return wrapper


class ChineseNameDetector(object):

	def __init__(self, resample=10000):
		"""
		read up data into a data frame that look like below             

								 full_name  is_chinese
		0      dianne  van eck           0
		1         chen zaichun           1

		"""
		self.data = pd.read_csv('~/Data/names/chinesenames-data.csv')
		print("name dataset contains {} names ({} chinese)".format(len(self.data), Counter(self.data.is_chinese)[1]))

		assert set(self.data.columns) == set({"full_name", "is_chinese"}), print("wrong column names in data csv...")
		assert sum(list(Counter(self.data.is_chinese).values())) == len(self.data), print(
			"seems like not all names in data are labelled...")

		# resample here
		if resample:
			self.data = (pd.concat([self.data[self.data.is_chinese == 0]
							.sample(resample), self.data[self.data.is_chinese == 1]
								.sample(resample)]).sample(frac=1))
			print("resampled name dataset contains {} names ({} chinese)".format(len(self.data), Counter(self.data.is_chinese)[1]))

		self.tfe = TextFeatureExtractor()

		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data.drop('is_chinese', axis=1),
																				self.data.is_chinese, test_size=0.2, random_state=42, stratify=self.data.is_chinese)
		#print("X_train=", self.X_train.shape)
		#print("index lenght=", len(set(self.X_train.index)))
		#X1 = self.X_train.drop_duplicates('full_name')
		#print('X1=',X1.shape)

		#print("y_train=", self.y_train.head())
			

	@timer
	def create_features(self, train_or_test='Train'):
		"""
		creates features for the training and testing sets
		"""
		assert set(self.X_train.columns) == {"full_name"}, print(
			"[create_features]: wrong column names in supplied data frame!")

		self.features = dict()

		if train_or_test == 'Train':

			for row in self.X_train.iterrows():
				self.features.update({row[0]: self.tfe.get_features(row[1].full_name)})
	  		
	  		# save features as a json
			json.dump(self.features, open('model_fs.json', 'w'))
			print(f"created feature dictionary for {len(self.features)} full names")

			self.features_train = pd.DataFrame.from_dict(self.features, orient='index').fillna(0)

			missing_index = set(self.X_train.index) - set(self.features_train.index)

			if missing_index:
				self.y_train = self.y_train[~self.y_train.index.isin(missing_index)]
		
		elif train_or_test == 'Test':

			for row in self.X_test.iterrows():
				self.features.update({row[0]: self.tfe.get_features(row[1].full_name)})

			self.features_test = pd.DataFrame.from_dict(self.features, orient='index').fillna(0)



		# create a sparce feature matrix
		# self.feature_names, self.features_as_csr = dict_to_csr_matrix(self.features)
		# print('sparse matrix is {}x{}'.format(*self.features_as_csr.shape))

		return self
	"""
	training_generator = DataGenerator(**params).generate(labels, partition['train'])
	validation_generator = DataGenerator(**params).generate(labels, partition['validation'])
	"""
	# def __get_exploration_order(self, list_IDs):
		
	# 	indexes = np.arange(len(list_IDs))    #  evenly spaced values within [0, len(list_IDs) - 
	# 	if self.shuffle == True:
	# 		np.random.shuffle(indexes)
	# 	return indexes

	# def __data_generation(self, labels, list_IDs_temp):
	# 	# IN: list of IDs included in batches as well as their corresponding labels
  
	# 	X = np.empty((self.batch_size, self.dim_x, self.dim_y, 1))
	# 	y = np.empty((self.batch_size), dtype = int)

	# 	# Generate data
	# 	for i, ID in enumerate(list_IDs_temp):
	# 		# Store volume
	# 		X[i, :, :, 0] = self.features_as_csr[i*self.batch_size:(i+1)*self.batch_size].toarray()
	# 		# Store class
	# 		y[i] = labels[ID]

	# 	return X, y

	
	def generate(self):
		
		while 1:
			# Generate order of exploration of dataset
			indexes = self.X_train.index
		
			# number of batches
			imax = int(len(indexes)/32)

			for i in range(imax):

				#batch insedex
				batch_indexes = [k for k in indexes[i*32:(i+1)*32]]
				#batch data
				X = np.empty((32, self.X_train.shape[1]))
				y = np.empty((32, 1), dtype=int)

				# Generate data
				X[i] = self.features_as_csr[i*32:(i+1)*32].toarray()
				# Store class
				y[i] = labels[i]
		
				yield X, y


	@timer
	def create_model(self):

		model = Sequential()
		#print('training data: ', self.features_train.shape)
		#print('lablels:', self.y_train.shape)
		model.add(Dense(512, input_shape=(self.features_train.shape[1],)))
		model.add(Activation('relu'))
		model.add(Dense(2))
		model.add(Activation('softmax'))

		# model.add(Dense(units=64, activation='relu', input_shape=(self.X_train.shape[1],)))
		# #model.add(Dense(units=16, activation='relu'))
		# #model.add(Flatten())
		# model.add(Dense(self.X_train.shape[1], activation='relu'))
		model.add(Dense(units=1, activation='sigmoid'))

		model.compile(loss='binary_crossentropy', 
			  optimizer='rmsprop', 
			  metrics=['accuracy'])

		return model

# class DataGenerator:

# 	def __init__(self, dim_x = 32, dim_y = 32, batch_size = 32, shuffle = True):

# 		self.dim_x = dim_x
# 		self.dim_y = dim_y
# 		self.batch_size = batch_size
# 		self.shuffle = shuffle

	


if __name__ == '__main__':

	cd = ChineseNameDetector()
	#cd.train_test()

	cd.create_features()

	# partition = {'train': cd.X_train.index.tolist(), 'validation': cd.X_test.index.tolist()}  # training and validation ids
	# labels = {t[0]: t[1] for t in zip(cd.X_train.index.tolist(), cd.y_train)}.update({t[0]: t[1] for t in zip(cd.X_test.index.tolist(), cd.y_test)})

	# # Generators
	# training_generator = cd.generate()
	# #validation_generator = DataGenerator(dim_x = 1, dim_y = cd.X_train.shape[1], batch_size = 32, shuffle = True).generate(labels, partition['validation'])

	# print('created generator')

	
	model = cd.create_model()

	print('created model')

	#print('X_train=', cd.X_train.shape)
	#print(cd.y_train.values.shape, cd.X_train.values.shape)

	model.fit(cd.fe.values, cd.y_train.values, 
					batch_size=32, 
					epochs=2, 
					verbose=1)
	score = model.evaluate(x_test, y_test, batch_size=128)

	# model.fit_generator(generator = training_generator,
	# 				steps_per_epoch = len(partition['train'])//batch_size,
	# 				# validation_data = validation_generator,
	# 				validation_steps = len(partition['validation'])//batch_size)
