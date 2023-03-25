# %%
import sys 
sys.path.extend('C:\\Users\\Admin\\AppData\\Roaming\\Python\\Python311\\site-packages')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import pylab as pl
# %%
'''
FONTOS: Az első feladatáltal visszaadott DataFrame-et kell használni a további feladatokhoz. 
A függvényeken belül mindig készíts egy másolatot a bemenő df-ről, (new_df = df.copy() és ezzel dolgozz tovább.)
'''

# %%
'''
Készíts egy függvényt, ami egy string útvonalat vár paraméterként, és egy DataFrame ad visszatérési értékként.

Egy példa a bemenetre: 'test_data.csv'
Egy példa a kimenetre: df_data
return type: pandas.core.frame.DataFrame
függvény neve: csv_to_df
'''

# %%
#path_= 'C:\\Users\\Admin\\source\\repos\\BEVADAT2022232\\StudentsPerformance.csv'

def csv_to_df(path:str)-> pd.core.frame.DataFrame:
    p_df = pd.read_csv(path)
    return p_df

#df=csv_to_df(path_)
#new_df = df.copy() 

# %%
'''
Készíts egy függvényt, ami egy DataFrame-et vár paraméterként, 
és átalakítja azoknak az oszlopoknak a nevét nagybetűsre amelyiknek neve nem tartalmaz 'e' betüt.

Egy példa a bemenetre: df_data
Egy példa a kimenetre: df_data_capitalized
return type: pandas.core.frame.DataFrame
függvény neve: capitalize_columns
'''

# %%
def  capitalize_columns(df:pd.DataFrame) -> pd.core.frame.DataFrame: 
   df.columns = df.columns.map(lambda x: str.upper(x) if 'e' not in x else x)
   return df
    

#new_df = df.copy()
#print(capitalize_columns(new_df))

# %%
'''
Készíts egy függvényt, ahol egy szám formájában vissza adjuk, hogy hány darab diáknak sikerült teljesíteni a matek vizsgát.
(legyen az átmenő ponthatár 50).

Egy példa a bemenetre: df_data
Egy példa a kimenetre: 5
return type: int
függvény neve: math_passed_count
'''

# %%
def math_passed_count(df:pd.DataFrame)-> int:
    return df['math score'].astype(int).apply(lambda x : x >= 50).value_counts()[True]
   

#new_df = df.copy()
#print(math_passed_count(new_df))

# %%
'''
Készíts egy függvényt, ahol Dataframe ként vissza adjuk azoknak a diákoknak az adatait (sorokat), akik végeztek előzetes gyakorló kurzust.

Egy példa a bemenetre: df_data
Egy példa a kimenetre: df_did_pre_course
return type: pandas.core.frame.DataFrame
függvény neve: did_pre_course
'''

# %%
def did_pre_course(df:pd.DataFrame): 
       return df.loc[df["test preparation course"] == "completed"]
#new_df = df.copy()
#print( did_pre_course(new_df))

# %%
'''
Készíts egy függvényt, ahol a bemeneti Dataframet a diákok szülei végzettségi szintjei alapján csoportosításra kerül,
majd aggregációként vegyük, hogy átlagosan milyen pontszámot értek el a diákok a vizsgákon.

Egy példa a bemenetre: df_data
Egy példa a kimenetre: df_average_scores
return type: pandas.core.frame.DataFrame
függvény neve: average_scores
'''

# %%
def average_scores(df:pd.DataFrame)-> pd.core.frame.DataFrame:
    return df.groupby('parental level of education')['math score', 'reading score', 'writing score'].mean()


# %%
'''
Készíts egy függvényt, ami a bementeti Dataframet kiegészíti egy 'age' oszloppal, töltsük fel random 18-66 év közötti értékekkel.
A random.randint() függvényt használd, a random sorsolás legyen seedleve, ennek értéke legyen 42.

Egy példa a bemenetre: df_data
Egy példa a kimenetre: df_data_with_age
return type: pandas.core.frame.DataFrame
függvény neve: add_age
'''

# %%
def add_age(df:pd.DataFrame)-> pd.core.frame.DataFrame:
    random.seed(42)
    df["age"] = np.random.randint(16,67,len(df))
    return df

# %%
'''
Készíts egy függvényt, ami vissza adja a legjobb teljesítményt elérő női diák pontszámait.

Egy példa a bemenetre: df_data
Egy példa a kimenetre: (99,99,99) #math score, reading score, writing score
return type: tuple
függvény neve: female_top_score
'''

# %%
def female_top_score(df:pd.DataFrame) :
     ndf= df.loc[df["gender"] == "female"].sort_values(["math score", "writing score", "reading score"],ascending=[False,False,False]).iloc[0]
     return (ndf["math score"], ndf["writing score"], ndf["reading score"])

# %%
'''
Készíts egy függvényt, ami a bementeti Dataframet kiegészíti egy 'grade' oszloppal. 
Számoljuk ki hogy a diákok hány százalékot ((math+reading+writing)/300) értek el a vizsgán, és osztályozzuk őket az alábbi szempontok szerint:

90-100%: A
80-90%: B
70-80%: C
60-70%: D
<60%: F

Egy példa a bemenetre: df_data
Egy példa a kimenetre: df_data_with_grade
return type: pandas.core.frame.DataFrame
függvény neve: add_grade
'''

# %%
def add_grade (df:pd.DataFrame) -> pd.core.frame.DataFrame:
    map_dict = { pd.Interval(90,100,closed='right'): "A", pd.Interval(80,90,closed='right'):"B",pd.Interval(70,80,closed='right'):"C",pd.Interval(60,70,closed='right'):"D" ,pd.Interval(0,60,closed='right'):"F"}
    df['grade'] = ((df['math score']+df['reading score'] + df['writing score'])/300*100).map(map_dict)
    return df

# %%
'''
Készíts egy függvényt, ami a bemeneti Dataframe adatai alapján elkészít egy olyan oszlop diagrammot,
ami vizualizálja a nemek által elért átlagos matek pontszámot.

Oszlopdiagram címe legyen: 'Average Math Score by Gender'
Az x tengely címe legyen: 'Gender'
Az y tengely címe legyen: 'Math Score'

Egy példa a bemenetre: df_data
Egy példa a kimenetre: fig
return type: matplotlib.figure.Figure
függvény neve: math_bar_plot
'''

# %%

def math_bar_plot(df:pd.DataFrame) -> plt.Figure:
    fig,ax = plt.subplots(1,1)
    ax = df.groupby(["gender"])["math score"].mean().plot.bar(x="gender",y="math score")
    ax.set_title("Average Math Score by Gender")
    ax.set_xlabel("Gender")
    ax.set_ylabel("Math Score")
    return fig
# %%
''' 
Készíts egy függvényt, ami a bemeneti Dataframe adatai alapján elkészít egy olyan histogramot,
ami vizualizálja az elért írásbeli pontszámokat.

A histogram címe legyen: 'Distribution of Writing Scores'
Az x tengely címe legyen: 'Writing Score'
Az y tengely címe legyen: 'Number of Students'

Egy példa a bemenetre: df_data
Egy példa a kimenetre: fig
return type: matplotlib.figure.Figure
függvény neve: writing_hist
'''

# %%
def writing_hist(df:pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(1,1)
    ax.hist(df["writing score"])
    pl.title("Distribution of Writing Scores")
    pl.xlabel("Writing Score")
    pl.ylabel("Number of Students")
    return fig

# %%
''' 
Készíts egy függvényt, ami a bemeneti Dataframe adatai alapján elkészít egy olyan kördiagramot,
ami vizualizálja a diákok etnikum csoportok szerinti eloszlását százalékosan.

Érdemes megszámolni a diákok számát, etnikum csoportonként,majd a százalékos kirajzolást az autopct='%1.1f%%' paraméterrel megadható.
Mindegyik kör szelethez tartozzon egy címke, ami a csoport nevét tartalmazza.
A diagram címe legyen: 'Proportion of Students by Race/Ethnicity'

Egy példa a bemenetre: df_data
Egy példa a kimenetre: fig
return type: matplotlib.figure.Figure
függvény neve: ethnicity_pie_chart
'''

# %%
def ethnicity_pie_chart(df:pd.DataFrame)-> plt.Figure:
    fig, ax = plt.subplots()
    ax.pie(df.groupby(["race/ethnicity"])["race/ethnicity"].count(),None,df.groupby(["race/ethnicity"])["race/ethnicity"].first(),autopct='%1.1f%%')
    ax.axis('equal')
    ax.set_title("Proportion of Students by Race/Ethnicity")
    return fig


