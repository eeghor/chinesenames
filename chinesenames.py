import pandas as pd
from collections import Counter
from keras.models import Sequential
from keras.layers import Dense, Activation
import time
from namefeatures import NameFeatureExtractor

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

		self.nfe = NameFeatureExtractor()

	@timer
	def create_features(self):

		fs = dict()

		for row in self.data.sample(frac=1).iloc[:120000].iterrows():
			#print(self.nfe.get_features(row[1].full_name)
			fs.update({row[0]: self.nfe.get_features(row[1].full_name)})
		
		fs_df = pd.DataFrame.from_dict(fs, orient='index')
		print("created a {}x{} feature data frame".format(fs_df.shape))


if __name__ == '__main__':

	cd = ChineseNameDetector()
	print(cd.data.head())
	cd.create_features()