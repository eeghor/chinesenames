import pandas as pd
from collections import Counter
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
import time
import pickle
from textfeatures import TextFeatureExtractor
from sparsehelpers import dict_to_csr_matrix
from sklearn.model_selection import train_test_split


def timer(func):
    def wrapper(*args, **kwargs):
        t_start = time.time()
        res = func(*args, **kwargs)
        print("f: {} # elapsed time: {:.0f} m {:.0f}s".format(
            func.__name__.upper(), *divmod(time.time() - t_start, 60)))
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
        print("name dataset contains {} names ({} chinese)".format(len(self.data), Counter(self.data.is_chinese)[1]))

        assert set(self.data.columns) == set({"full_name", "is_chinese"}), print("wrong column names in data csv...")
        assert sum(list(Counter(self.data.is_chinese).values())) == len(self.data), print(
            "seems like not all names in data are labelled...")

        self.tfe = TextFeatureExtractor()
        self.features = dict()

    def train_test(self):
        # split into the training and testing dats
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data.drop('is_chinese', axis=1),
                                                                                self.data.is_chinese, test_size=0.2, random_state=42, stratify=self.data.is_chinese)

        return self

    # generator that supplies chunks of data
    def data_generator(self, batchsize=256):
    	"""
		this generator needs to be running indefinitely
    	"""
    	already_read = 0
    	while 1:
    		while already_read < self.features_as_csr.shape[0] - batchsize:
    			already_read += batchsize
    			yield (self.features_as_csr[already_read:already_read + batchsize].toarray(),
            		self.y_train[already_read:already_read + batchsize])
            

    @timer
    def create_features(self, df):

        assert set(df.columns) == {"full_name"}, print(
            "[create_features]: wrong column names in supplied data frame!")

        for row in df.iterrows():
        	self.features.update({row[0]: self.tfe.get_features(row[1].full_name)})
      

        print(f"created feature dictionary for {len(self.features)} names")

        # create a sparce feature matrix
        self.feature_names, self.features_as_csr = dict_to_csr_matrix(self.features)
        print('sparse matrix is {}x{}'.format(*self.features_as_csr.shape))

        return self

    @timer
    def create_model(self):

        model = Sequential()
        model.add(Dense(units=64, activation='relu', input_shape=self.features_as_csr.shape[1:]))
        #model.add(Dense(units=16, activation='relu'))
        model.add(Flatten())
        model.add(Dense(units=1, activation='sigmoid'))

        model.compile(optimizer='rmsprop',
                      loss='binary_crossentropy', metrics=['accuracy'])

        return model


if __name__ == '__main__':

    cd = ChineseNameDetector()
    cd.train_test()
    cd.create_features(cd.data.drop('is_chinese', axis=1))
    model = cd.create_model()
    model.fit_generator(
            cd.data_generator(), steps_per_epoch=cd.X_train.shape[0]//256, epochs=10)
