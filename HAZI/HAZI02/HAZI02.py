# %%
import sys
print(sys.path)

# %%
import sys 
sys.path.extend('C:\\Users\\Admin\\AppData\\Roaming\\Python\\Python311\\site-packages')
import datetime as dt
from datetime import date, timedelta
import calendar
import time

import numpy as np

# %%
# Írj egy olyan fügvényt, ami megfordítja egy 2d array oszlopait
# Be: [[1,2],[3,4]]
# Ki: [[2,1],[4,3]]
# column_swap()

# %%
def column_swap(input: np.array):
    return np.roll(input,1,1)



# %%
#Készíts egy olyan függvényt ami összehasonlít két array-t és adjon vissza egy array-ben, hogy hol egyenlőek 
# Pl Be: [7,8,9], [9,8,7] 
# Ki: [1]
# compare_two_array()
# egyenlő elemszámúakra kell csak hogy működjön

# %%
def compare_two_array(arr1: np.array, arr2: np.array):
    x = np.where(np.equal(arr1,arr2))
    return x[0]




# %%
#Készíts egy olyan függvényt, ami vissza adja a megadott array dimenzióit:
# Be: [[1,2,3], [4,5,6]]
# Ki: "sor: 2, oszlop: 3, melyseg: 1"
# get_array_shape()
# 3D-vel még műküdnie kell!

# %%
def get_array_shape(arr: np.array):
    dim = np.shape(arr)
    d = arr.ndim
    return f"sor: {dim[0]}, oszlop: {dim[1] if d >1 else 1}, melyseg: {dim[2] if d > 2 else 1}"

#ari = np.array([[1,2,3], [4,5,6]])
#print(get_array_shape(ari))

# %%
# Készíts egy olyan függvényt, aminek segítségével elő tudod állítani egy neurális hálózat tanításához szükséges Y-okat egy numpy array-ből. 
#Bementként add meg az array-t, illetve hogy mennyi class-od van. Kimenetként pedig adjon vissza egy 2d array-t, ahol a sorok az egyes elemek. Minden nullákkal teli legyen és csak ott álljon egyes, ahol a bementi tömb megjelöli
# Be: [1, 2, 0, 3], 4
# Ki: [[0,1,0,0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]
# encode_Y()

# %%
def encode_Y(arr: np.array, class_):
    arr1= np.array([arr],class_)
    
    return None

# %%
# A fenti feladatnak valósítsd meg a kiértékelését. Adj meg a 2d array-t és adj vissza a decodolt változatát
# Be:  [[0,1,0,0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]
# Ki:  [1, 2, 0, 3]
# decode_Y()

# %%
def decode_Y(arr: np.array):
    a = arr.nonzero()
    return a[1]

# %%
# Készíts egy olyan függvényt, ami képes kiértékelni egy neurális háló eredményét! Bemenetként egy listát és egy array-t és adja vissza a legvalószínübb element a listából.
# Be: ['alma', 'körte', 'szilva'], [0.2, 0.2, 0.6]. 
# Ki: 'szilva'
# eval_classification()

# %%
def eval_classification(arr1: list, arr2: np.array):
    i = np.argmax(arr2)
    out = arr1[i]
    return out

#print(eval_classification(['alma', 'körte', 'szilva'], [0.2, 0.2, 0.6]))

# %%
# Készíts egy olyan függvényt, ahol az 1D array-ben a páratlan számokat -1-re cseréli
# Be: [1,2,3,4,5,6]
# Ki: [-1,2,-1,4,-1,6]
# repalce_odd_numbers()

# %%
def replace_odd_numbers(arr:np.array):
    arr[arr%2 > 0] = -1
    return arr

#ari = np.array([1,2,3,4,5,6])
#print(replace_odd_numbers(ari))

# %%
# Készíts egy olyan függvényt, ami egy array értékeit -1 és 1-re változtatja, attól függően, hogy az adott elem nagyobb vagy kisebb a paraméterként megadott számnál.
# Ha a szám kisebb mint a megadott érték, akkor -1, ha nagyobb vagy egyenlő, akkor pedig 1.
# Be: [1, 2, 5, 0], 2
# Ki: [-1, 1, 1, -1]
# replace_by_value()

# %%
def replace_by_value(arr:np.array, n):
    arr[arr >= n]= 1
    arr[arr < n]= -1  
    return arr

#ari = np.array([1,2,5,0])
#print(replace_by_value(ari,2))

# %%
# Készítsd egy olyan függvényt, ami az array értékeit összeszorozza és az eredmény visszaadja
# Be: [1,2,3,4]
# Ki: 24
# array_multi()
# Ha több dimenziós a tömb, akkor az egész tömb elemeinek szorzatával térjen vissza

# %%
def array_multi(arr:np.array):
    out = np.prod(arr)
    return out

#ari = np.array([1,2,3,4])
#print(array_multi(ari))

# %%
# Készítsd egy olyan függvényt, ami a 2D array értékeit összeszorozza és egy olyan array-el tér vissza, aminek az elemei a soroknak a szorzata
# Be: [[1, 2], [3, 4]]
# Ki: [2, 12]
# array_multi_2d()

# %%
def array_multi_2d(arr:np.array):
    s = np.prod(arr, axis=1)
    out2 = np.array(s)
    return out2

#ari = np.array([[1, 2], [3, 4]])
#print(array_multi_2d(ari))

# %%
# Készíts egy olyan függvényt, amit egy meglévő numpy array-hez készít egy bordert nullásokkal. Bementként egy array-t várjon és kimenetként egy array jelenjen meg aminek van border-je
# Be: [[1,2],[3,4]]
# Ki: [[0,0,0,0],[0,1,2,0],[0,3,4,0],[0,0,0,0]]
# add_border()


# %%
def add_border(arr:np.array):
    arr = np.pad(arr,pad_width=1,mode='constant', constant_values=0)
    return arr

#ari = np.array([[1,2],[3,4]])
#print(add_border(ari))

# %%
# Készíts egy olyan függvényt ami két dátum között felsorolja az összes napot.
# Be: '2023-03', '2023-04'
# Ki: ['2023-03-01', '2023-03-02', .. , '2023-03-31',]
# list_days()

# %%
def list_days(start: dt.datetime, end: dt.datetime):
  return np.arange(start,end,dtype = 'datetime64[D]')

# %%
# Írj egy fügvényt ami vissza adja az aktuális dátumot az alábbi formában: YYYY-MM-DD
# Be:
# Ki: 2017-03-24 

# %%
def get_act_date():
    return date.today()



# %%
# Írj egy olyan függvényt ami visszadja, hogy mennyi másodperc telt el 1970 január 01. 00:00:00 óta.
# Be: 
# Ki: másodpercben az idó, int-é kasztolva
# sec_from_1970()


# %%
def sec_from_1970():
     return int(dt.datetime(1970,1,1,2).timestamp())
  

#sec_from_1970()


