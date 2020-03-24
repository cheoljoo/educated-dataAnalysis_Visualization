import numpy as np
import pandas as pd

# 1번 풀이
dfn = pd.read_csv('yob1880.txt')
print(dfn)
print('\nsolve 1 : yob11880\n',pd.read_csv('yob1880.txt',names=['name','sex','births']))
df2 = pd.read_csv('yob1880.txt',names=['name','sex','births'])

# 2번 풀이
print(df2[:10])
print(df2[-10:])

# 3번 풀이
df2.groupby(['sex']).sum()
print(df2.groupby(['sex']).sum())



# 4번 풀이
import glob
import re

filenames = glob.glob("./*.txt")
dfs = []
ccnt = []
for filename in filenames:
    df=pd.read_csv(filename,names=['name','sex','births'])
    # 5번 풀이
    df['year'] = filename[5:9]
    dfs.append(df)
    #dfs['year'] = filename[5:9]
#    ccnt.append(filename[5:9])
    #print(filename[5:9],'  ')
#print(dfs)


big_frame = pd.concat(dfs)
print('\nsolve:\n' , big_frame.groupby(['year','sex']).sum())
#big_frame['year'] = ccnt
print('\nbig_frame:\n' , big_frame)

# 6 번 풀이
print('\nsolve 6 : big_frame.pivot_table\n' , big_frame.pivot_table(index=['year','sex'],aggfunc=sum))

# 8번 풀이
tsum = big_frame['births'].sum()
names = big_frame.groupby(['name']).sum()
names['prop'] = names['births'] *100 / tsum
print(tsum,names)

# 9 번 풀이
namesort = names.sort_values(by='births')
print ('\nsolve 9-1:\n',namesort[-1000:] )


yssort = big_frame.groupby(['year','sex','name']).sum().sort_values(by='births')
print('\nsolve 9-2:\n' , yssort[-1000:])

# 10번 풀이
tt = big_frame.pivot_table(index=['sex','year','name'],aggfunc=sum)
print('\nsolve 10:\n' , tt)
