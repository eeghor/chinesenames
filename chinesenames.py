import pandas as pd
from collections import Counter
from keras.models import Sequential
from keras.layers import Dense, Activation
import time
import pickle
from textfeatures import TextFeatureExtractor
from sparsehelpers import dict_to_csr_matrix
from sklearn.model_selection import train_test_split


def timer(func):
    def wrapper(*args, **kwargs):
        t_start = time.time()
        res = func(*args, **kwargs)
        print("f: {} # elapsed time: {:.0f} m {:.0f}s".format(func.__name__.upper(), *divmod(time.time() - t_start, 60)))
        return res
    return wrapper

class ChineseNameDetector(object):

	def __init__(self):

		"""
		read up data into a data frame that look like below             
					
					 full_name  is_chinese
		0      dianne  van eck           0
		1         chen zaichun           1

		"""
		self.data = pd.read_csv('~/Data/names/chinesenames-data.csv')
		print("name dataset contains {} names".format(len(self.data)))

		assert set(self.data.columns) == set({"full_name", "is_chinese"}), print("#@#@# error! wrong column names in data csv...")
		assert sum(list(Counter(self.data.is_chinese).values())) == len(self.data), print("#@#@# error! seems like not all names in data are labelled...")

		self.tfe = TextFeatureExtractor()
		self.features = dict()

	def __repr__(self):
	 	pass

	def train_test(self):
		# split into the training and testing dats
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data.drop('is_chinese', axis=1), 
											self.data.is_chinese, test_size=0.33, random_state=42, stratify=self.data.is_chinese)

		return self

	# generator that supplies chunks of data
	def chunk_generator(self, sdf, ch=1000):

		i = 0

		while i*ch <= sdf.shape[0]:
			yield sdf[i*ch:(i+1)*ch].toarray()
			i += 1

	@timer
	def create_features(self, df, train_or_test = 'training'):
		
		print(df.head())
		assert set(df.columns) == {"full_name"}, print("[create_features] error: wrong column names in supplied data frame!")
		
		for row in df.iterrows():
			self.features.update({row[0]: self.tfe.get_features(row[1].full_name)})
		
		print(f"created {len(self.features)} features, {train_or_test} mode")

		# create a sparce feature matrix
		self.feature_names, self.features_as_csr = dict_to_csr_matrix(self.features)

		return self

	@timer
	def train(self):
		print('ativating a model..')
		model = Sequential()



if __name__ == '__main__':

	cd = ChineseNameDetector()
	print(cd.data.head())
	cd.train_test()
	cd.create_features(cd.X_train)
	cd.train()