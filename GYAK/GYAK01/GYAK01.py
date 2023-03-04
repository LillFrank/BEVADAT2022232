
  

from typing import List


def contains_odd(input_list: List) -> bool:
    ind = 0
    while ind < len(input_list) and input_list[ind] % 2 == 0:
        ind = ind+1

    return True if ind < len(input_list) else False

def is_odd(input_list):
    
    newlist = [None]*len(input_list)
    i=0
    while i<len(newlist):
    
        if input_list[i] % 2 == 0:
            newlist[i] = False
        else:
            newlist[i] = True
        i=i+1


    return newlist

def element_wise_sum(input_list_1, input_list_2):

    sumlist = [None]*len(input_list_1)
    i=0
    while i< len(input_list_1):
        sumlist[i] = input_list_1[i] + input_list_2[i]
        i=i+1
    

    return sumlist


def dict_to_list(input_dict):

    newlist = [None]*len(input_dict)
    i = 0
    for key, value in input_dict.items():
        newlist[i] = f"{key}:{value}"
        i=i+1

    return tuple(newlist)


#print(contains_odd([4,2,8,4,10,6,8]))
#print(is_odd([1,1,2,4,5,5]))
#print(element_wise_sum([1,2,3],[4,5,6]))
#print(dict_to_list({'age':25, 'name':'Jhon','student':True}))


