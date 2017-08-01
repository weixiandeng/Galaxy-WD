# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
unames=['names','gender','frequency']
df = pd.read_table('/Users/dengweixian/Desktop/Bioinformatics/names/yob1881.txt', sep=',', header = None, names=unames)
a=df.groupby('gender').frequency.sum()

years=range(1880,2011)
pieces=[]
columns=['name','sex','births']

for year in years:
    path='/Users/dengweixian/Desktop/Bioinformatics/names/yob%d.txt'%year
    frame=pd.read_csv(path,names=columns)
    frame['year']=year
    pieces.append(frame)
    names=pd.concat(pieces,ignore_index=True)


total_births=names.pivot_table('births', index='year',columns='sex',aggfunc=sum)

def add_prop(group):
    group['prop']=group.births/group.births.sum()
    return group

names=names.groupby(['year','sex']).apply(add_prop)

import numpy as np
k=np.allclose(names.groupby(['year','sex']).prop.sum(),1)

from pandas import Series, DataFrame
obj = Series([4, 7, -5, 3])
obj2=Series([4,7,-5,3],index=['a','b','c','d'])

sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
obj3 = Series(sdata)

states = ['California', 'Ohio', 'Oregon', 'Texas']
obj4=Series(sdata,index=states)

print(obj4.isnull())
obj3+obj4

data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'], 'year': [2000, 2001, 2002, 2001, 2002],
'pop': [1.5, 1.7, 3.6, 2.4, 2.9]}
frame = DataFrame(data)

a=DataFrame(frame, columns=['pop','year','state'])

frame2=DataFrame(data, columns=['year', 'state', 'pop', 'debt'], index=['one','two','three','four','five'])

pop = {'Nevada': {2001: 2.4, 2002: 2.9},'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}
b=DataFrame(pop)
c=b.T

index = pd.Index(np.arange(3))

obj2 = Series([1.5, -2.5, 0], index=index)

mwl=pd.read_table('/Users/dengweixian/Desktop/96.csv',sep=',',header=None)
num=range(1,13)
import string

letter=string.ascii_uppercase[:8]
final_columns=[]
for l in letter:
    for i in num:
        final_columns.append('%s%d'%(l,i))
        
mwl=DataFrame(mwl,columns=final_columns)

obj3 = Series(['blue', 'purple', 'yellow'], index=[0, 2, 4])
obj3.reindex(range(6), method='ffill')
obj3.reindex(range(6), method='pad')

A1=[]
for j in mwl.index:
    if j%10==1:
        A1.append(mwl.ix[j,0])
        
data = DataFrame(np.arange(16).reshape((4, 4)),index=['Ohio', 'Colorado', 'Utah', 'New York'],columns=['one', 'two', 'three', 'four'])

df1 = DataFrame(np.arange(12.).reshape((3, 4)), columns=list('abcd'))
df2 = DataFrame(np.arange(20.).reshape((4, 5)), columns=list('abcde'))

frame = DataFrame(np.arange(12.).reshape((4, 3)), columns=list('bde'), index=['Utah', 'Ohio', 'Texas', 'Oregon'])
series = frame.ix[0]

obj = Series([7, -5, 7, 4, 2, 0, 4])
mwl3=mwl.reindex(['0','A','B','C','D','E','F','G','H','1']*614+['0','A','B','C','D','E','F','G','H'])
'''mwl4=DataFrame(mwl3,index=mwl3.index+[], columns=mwl3.columns)'''
'''How to add new rows and new columns to dataframe?'''

from pandas_datareader import data, wb


all_data = {}
for ticker in ['AAPL', 'IBM', 'MSFT', 'GOOG','NVDA','TSLA']:
    all_data[ticker] = data.get_data_yahoo(ticker, '1/1/2012', '1/1/2017')

price = DataFrame({tic: data['Adj Close'] for tic, data in all_data.items()})
volume = DataFrame({tic: data['Volume']for tic, data in all_data.items()})

data = Series(np.random.randn(10), index=[['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'd', 'd'],[1, 2, 3, 1, 2, 3, 1, 2, 2, 3]])

ser = Series(np.arange(3.))

ser2 = Series(np.arange(3.), index=['a', 'b', 'c'])

ser3 = Series(range(3), index=[-5, 1, 3])

frame = DataFrame(np.arange(6).reshape(3, 2), index=[2, 0, 1])

Chapter6=pd.read_table('/Users/dengweixian/Desktop/MSFragger/psm.tsv')

from lxml.html import parse
from urllib.request import urlopen

parsed = parse(urlopen('http://finance.yahoo.com/q/op?s=AAPL+Options'))
doc = parsed.getroot()

links=doc.findall('.//a')
urls = [lnk.get('href') for lnk in doc.findall('.//a')]
tables = doc.findall('.//table')
calls = tables[0]
puts = tables[1]
rows = calls.findall('.//tr')

def _unpack(row, kind='td'):
    elts = row.findall('.//%s' % kind)
    return [val.text_content() for val in elts]

_unpack(rows[0], kind='th')

obj = """ {"name": "Wes",
"places_lived": ["United States", "Spain", "Germany"], "pet": null,
"siblings": [{"name": "Scott", "age": 25, "pet": "Zuko"},{"name": "Katie", "age": 33, "pet": "Cisco"}]
} """
    
result = json.loads(obj)

asjson = json.dumps(result)

siblings = DataFrame(result['siblings'], columns=['name', 'age'])


##### folowing lines are practicing for Chapter 7 #######

df1 = DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'a', 'b'], 'data1': range(7)})
df2 = DataFrame({'key': ['a', 'b', 'd'], 'Data2': range(3)})

df1 = DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'b'], 'data1': range(6)})
df2 = DataFrame({'key': ['a', 'b', 'a', 'b', 'd'], 'data2': range(5)})
pd.merge(df1,df2)

left = DataFrame({'key1': ['foo', 'foo', 'bar'], 'key2': ['one', 'two', 'one'],
'lval': [1, 2, 3]})

right = DataFrame({'key1': ['foo', 'foo', 'bar', 'bar'], 'key2': ['one', 'one', 'one', 'two'],
'rval': [4, 5, 6, 7]})

pd.merge(left, right,on=['key1','key2'])

 s1 = Series([0, 1], index=['a', 'b'])
 s2 = Series([2, 3, 4], index=['c', 'd', 'e'])
  s3 = Series([5, 6], index=['f', 'g'])
  
s4 = pd.concat([s1 * 5, s3])

df1 = DataFrame(np.random.randn(3, 4), columns=['a', 'b', 'c', 'd'])
df2 = DataFrame(np.random.randn(2, 3), columns=['b', 'd', 'a'])
pd.concat([df1, df2],axis=1)