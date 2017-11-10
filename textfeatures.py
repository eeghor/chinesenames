import itertools as it
import string
from collections import defaultdict, Counter
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS as stopwords
import enchant

"""
Normalise text and extract various features

input:   a string (sentence)
returns: features will be a dictionary like {feature name: feature value}

"""

class TextFeatureExtractor:
	
	d = enchant.Dict('en_US')

	def __init__(self, ignore_punc=True, ignore_numb=True, ignore_capit=True, ignore_stopwords=True):

		self.ignore_punc = ignore_punc
		self.ignore_numb = ignore_numb
		self.ignore_capit = ignore_capit
		self.ignore_stopwords = ignore_stopwords	

	@staticmethod
	def _check_validity(st):

		if isinstance(st, str):
			if bool(st.strip()):
				return True
		return False

	def _normalise(self, st):

		st = st.strip()

		if self.ignore_capit:
			st = st.lower()

		if self.ignore_punc:
			st = ''.join([c if c not in string.punctuation else ' ' for c in st])

		words = st.split()

		if self.ignore_numb:
			words = [word for word in words if word.isalpha()]

		if self.ignore_stopwords:
			words = [word for word in words if word.lower() not in stopwords]

		if words:
			return ' '.join(words)

	@staticmethod
	def collect_features(st):

		features = dict()

		fpref = 'f_'

		"""
		count words (longer than 1 letter) separated by white space
		"""
		words = st.split()

		for w, cnt in Counter(words).items():
			if len(w) > 1:
				features.update({fpref + w: cnt})

		"""
		bi-grams
		"""
		for i, w in enumerate(words):
			if len(words) > i + 1:
				features.update({fpref + '-'.join([w, words[i+1]]): 1})

		"""
		count the number of words separated by white space
		"""
		features.update({fpref + 'n_words': len(words)})

		"""
		count the first letters of each word
		"""
		for l, cnt in Counter([w[0] for w in words]).items():
			features.update({fpref + 'fst_letter_' + l: cnt})

		"""
		count ocurrences of all letters in full name
		"""
		for l, cnt in Counter(st.replace(' ', '')).items():
			features.update({fpref + 'cnt_' + l: cnt})

		"""
		count pairs of letters (for each word in full name)
		"""
		for p, cnt in Counter([w[i: i+2] for w in words for i in range(len(w)) if len(w[i: i+2]) == 2]).items():
			features.update({fpref + 'cnt_' + p: cnt})

		"""
		word endings, last n letters; binary
		"""
		for w in words:
			if len(w) > 2:
				features.update({fpref + 'end_' + w[-2:]: 1})
			if len(w) > 3:
				features.update({fpref + 'end_' + w[-3:]: 1})
			if len(w) > 4:
				features.update({fpref + 'end_' + w[-4:]: 1})

		"""
		fraction of all letters used in both first and last word
		"""
		ok_words = [w for w  in words if len(w) > 1]

		if len(ok_words) > 1:
			inboth = set(ok_words[0]) & set(ok_words[-1])
			if inboth:
				features.update({fpref + 'frac_let_both_': round(len(inboth)/len(set(''.join(ok_words))), 2)})

		# count dictionary words
		features.update({fpref + 'cnt_dict_wrds': sum(TextFeatureExtractor.d.check(w) for w in words)})

		# ratio of the shortest to the longest word length (ignore 1-letter words)
		features.update({fpref + 'shtst_to_lngst': round(len(min(ok_words, key=len))/len(max(ok_words, key=len)), 2) if len(ok_words) > 1 else None})


		return features

	def get_features(self, st):
		"""
		extract all features
		"""
		if TextFeatureExtractor._check_validity(st):
			# normalize first; note than st can become None after normalisation
			st = self._normalise(st)
			if st:
				fs = TextFeatureExtractor.collect_features(st)
			else:
				fs = {}
		else:
			fs = {}
			
		return fs


if __name__ == '__main__':

	 tfe = TextFeatureExtractor()
	 print(tfe.get_features('90sw zhu choi chain 021344 556'))
