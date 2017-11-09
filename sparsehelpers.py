import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd

def dict_to_csr_matrix(dic):

	"""
	giver a dictionary of the form 
		{id: {'feature': value,
  			  'feature': value,
  			  ...},
  		 id: {...},
  		 ...
  	create a sparce csr matrix
	"""
	
	sorted_feature_names = sorted({k for _ in dic for k in dic[_]})  # list

	# create a dictionary that maps feature name to its index, like enumerate 
	name_to_index = {k: i for i, k in enumerate(sorted_feature_names)}


	row = []
	col = []
	dat = []
	
	for c in dic:  		# c is a full name index, presumably starting from 0

	    if not dic[c]:  # if no features for some id, ignore this id
	        continue
	    
	    for d in dic[c]:

	        col.append(name_to_index[d])
	        dat.append(dic[c][d])

	    row.extend([c]*len(dic[c]))
	
	rown = np.array(row, dtype=np.int32)
	coln = np.array(col, dtype=np.int32)
	datn = np.array(dat, dtype=np.float32)
	
	return (sorted_feature_names, csr_matrix((datn, (rown, coln)), shape=(max(row)+1, max(col)+1)))

if __name__ == '__main__':

	d = {0: {'fa': 1, "re": 2.1, "p": 3},
			1: { "re": 3, "p": -11, "ee": 2},
			2: { "p": 1}}
	print(pd.DataFrame.from_dict(d, orient='index'))
	nms, m = dict_to_csr_matrix(d)
	print(pd.SparseDataFrame(m))
	print(nms)