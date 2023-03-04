

import itertools
from collections import ChainMap



def subset(input_list, start_index, end_index):
	n= end_index-start_index
	newlist = [None]*n
	i=0
	while start_index != end_index:
		newlist[i]= input_list[start_index]
		i=i+1
		start_index=start_index+1

	return newlist

#print(subset([1,2,3,4,5,6,7,8,9],0,6))

def every_nth(input_list,step_size):
	n=len(input_list)
	newlist = []
	
	for inde in range(0,n,step_size):
		newlist.append(input_list[inde])
	

	return newlist

#print(every_nth([1,2,3,4,5,6,7,8,9,10],2))

def unique(input_list):
	uniquelist= []
	for x in input_list:
		if x not in uniquelist:
			uniquelist.append(x)
	return False if not uniquelist else True

#print(unique([1,23,45,23,2,3]))

def flatten(input_list):
	newlist = [num for sublist in input_list for num in sublist]
	return newlist

#print(flatten([[1,2],[3],[4,5,6],[7,8,9,10]]))

def merge_lists(*args):
	merge=list(itertools.chain(*args))
	return merge

#print(merge_lists([2,4,5],[1,3,6],[112,456,8132]))

def reverse_tuples(input_list):
	newlist=[]
	#tmptuple = ()
	i=len(input_list)-1
	
	for x in input_list:
		j = len(x)-1
		tmptuple=[]
		while j > -1:
			tmptuple.append(x[j])
			j=j-1
		tuple_ = tuple(tmptuple) 
		newlist.append(tuple_)

	return newlist

#print(reverse_tuples([(1,2),(3,4,5)]))




def remove_duplicates(input_list):
	newlist=[]
	for c in input_list:
		if c not in newlist:
			newlist.append(c)

	return newlist

#print(remove_tuplicates([22,15,489,22,67,465741,5486,25,15,69,84,67,465741]))

def transpose(input_list):

	transposed = []
	for i in range(len(input_list[0])):
		row = []
		for j in input_list:
			row.append(j[i])
		transposed.append(row)

	return transposed

#print(transpose([[1,2,3,8],[4,5,6,10],[7,8,9,15]]))

def split_into_chunks( input_list, chunk_size) :

	output = [input_list[i: i+ chunk_size] for i in range(0, len(input_list), chunk_size)]
	return output
	

#print(split_into_chunks([1,2,3,4,5,6,7,8],2))

def merge_dicts(*dict):
	merged = ChainMap(*dict)
	
	return merged

#print(merge_dicts({'k':2, 'j':3}, {'d':4,'z':8, 'f':9}))






