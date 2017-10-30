import itertools as it
from collections import defaultdict, Counter

class NameFeatureExtractor(object):

	def __init__(self):

		# features will be a set of tuples (feature name, feature value)
		self.features = set()

	def _normalise(self, st):
		"""
		normalise the full name so that it look like 'john smith', i.e. 
			* words separated by a single white space
			* all lower case
			* only letters allowed
		"""
		pass 

	def _get_word_counts(self, st):
		"""
		count words (longer than 1 letter) separated by white space
		"""
		for w, cnt in Counter(st.split()).items():
			if len(w) > 1:
				self.features.add((w, cnt))

	def _get_word_pairs(self, st):
		"""
		words (longer than 1 letter) separated by white space
		"""
		ok_words = [w for w in st.split() if len(w) > 1]

		for i, w in enumerate(ok_words):
			if len(ok_words) > i + 1:
				self.features.add(('-'.join([w, ok_words[i+1]]), 1))

	def _get_number_words(self, st):
		"""
		count the number of words separated by white space
		"""
		self.features.add(('n_words', len(st.split())))

	def _get_first_letters(self, st):
		"""
		count the first letters of each word in full name; 
		example: 'john j smith' shoud result in ('fst_let_j', 2), ('fst_let_s', 1)
		"""
		for l, cnt in Counter([w[0] for w in st.split()]).items():
			self.features.add(('fst_let_' + l, cnt))

	def _get_letter_counts(self, st):
		"""
		count ocurrences of all letters in full name
		"""
		for l, cnt in Counter(st.replace(' ', '')).items():
			self.features.add(('cnt_' + l, cnt))

	def _get_ngram_counts(self, st, n=2):
		"""
		count pairs of letters (for each word in full name)
		"""
		for p, cnt in Counter([w[i: i+n] for w in st.split() for i in range(len(w)) if len(w[i: i+n]) == n]).items():
			self.features.add(('cnt_' + p, cnt))

	def _get_endings(self, st, n=2):
		"""
		word endings, last n letters; binary
		"""
		for w in st.split():
			if len(w) > n:
				self.features.add(('end_' + w[-n:], 1))

	def _get_word_lengths(self, st):
		
		"""
		sorted word lengths for words longer than 1 letter
		"""
		ok_words = sorted([str(len(w)) for w in st.split() if len(w) > 1], reverse=True)
		if ok_words:
			self.features.add(('len_' + '_'.join(ok_words), 1))

	def _get_frac_letters_all_words(self, st):
		"""
		fraction of all letters used in name and surname used in both name and surname
		"""
		ok_words = [w for w  in st.split() if len(w) > 1]

		if len(ok_words) > 1:
			inboth = set(ok_words[0]) & set(ok_words[-1])
			if inboth:
				self.features.add(('frac_let_both_', round(len(inboth)/len(set(''.join(ok_words))), 2)))

	def get_features(self, st):
		"""
		extract all features for the given full name
		"""
		self._get_word_counts(st)
		self._get_number_words(st)
		self._get_first_letters(st)
		self._get_letter_counts(st)
		self._get_ngram_counts(st)
		self._get_ngram_counts(st, n=3)
		self._get_endings(st)
		self._get_endings(st, n=3)
		self._get_word_pairs(st)
		self._get_word_lengths(st)
		self._get_frac_letters_all_words(st)

		return self


if __name__ == '__main__':
	 ne = NameFeatureExtractor()
	 print(ne.get_features('britney bob spears').features)
