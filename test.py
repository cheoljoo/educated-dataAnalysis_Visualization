import numpy as np
import pandas as pd
from pandas import DataFrame

print(np.__version__)
print(pd.__version__)

data = [[1, 2, 3, 4], [5, 6, 7, 8]]
arr2 = np.array(data)
print('arr2 :\n', arr2)
print('ndim:', arr2.ndim)
print('shape:', arr2.shape)
print('dtype:', arr2.dtype)

arr3 = np.array([1, 2, 3, 4, 5], dtype=np.float32)
print('arr3 dtype:', arr3.dtype)

arr4 = np.arange(10)
print('arr4:', arr4)

# arr5 = np.zeros(3)
# print('arr5:\n',arr5)

print('arr2 * arr2:\n', arr2 * arr2)
print('arr2 - arr2:\n', arr2 - arr2)

print('arr3 * arr3:\n', arr3 * arr3)
print('arr3 - arr3:\n', arr3 - arr3)

a01 = np.empty((8, 4))
for i in range(8):
    a01[i] = i
print('\n', a01[[4, 3, 0, 6]])  # 4,3,0,6번 행의 값을 가져와라.
print('\n', a01[[-3, -5, -7]])

# 배열 재형성 : reshape()
a02 = a01.reshape((4, 8))  # 위의 생성된 8*4 = 32개의 data와 같은 숫자여야 한다.
print(a02)
print('\n', a02.T)  # 행과 열 바꿈

# 3-1 배열을 위한 데이터 처리
# ufunc : 유니버셜 함수 - 데이터 원소 별로 연산을 수행 (고속화가 잘되어져있음)
# 사칙연산 , 삼각함수 , 비트 (bitwise_and , left_shift)  , 관계형 , 논리 , maximum , minimum , modf , 부동소수점에 적용할수 있는 함수
a03 = np.arange(10)
print('sqrt:', np.sqrt(a03))

# 3-2. 조건부 함수 : where 함수
# where(c,a,b) : a if c else b
# result = np.where(c,a,b)
# result = [(x if c else y) for x,y,c in zip(xarr,yarr,cond)]


# 3-3 수학 메서드와 통계 메서드
# sum , mean , cumsum : 누적합 , cumprod , std , var , min , max , argmin , argmax
a04 = np.random.randn(5, 4)
print('\n', a04)
print('mean1:', a04.mean())
print('mean2:', np.mean(a04))  # mean 구하는 2가지 방식
print('mean1:', a04.mean(axis=1))  # x축 : 각행의 평균   , 0일때는 y축으로 열의 평균

# boolean 배열을 위한 베서드
a05 = np.random.randn(100)
print('\n bool sum:', (a05 > 0).sum())
# data[ (data>3).any(axis=1)]   : 3을 초과하는 값이 하나라도 존해하면 그 행을 가져와라.

# 3-4 정렬 및 집합 함수
# sort(axis=1)
a06 = np.random.randn(3, 3)
print('\n', a06)
a06.sort(axis=1)
print('\n', a06)
# unique , intersect1d , union1d , in1d(x,y)x가 y를 포함하는지?  ,  setdiff1d : 차집합 , setxor1d : 대칭차집합


# 4-1 선형대수
# dot method를 이용하여 행렬 곱을 수행
a07 = np.random.randn(2, 3)
a08 = np.random.randn(3, 2)
print('\nmatrix prod:\n', a07.dot(a08))
# diag , dot , trace (대각선 원소의 합) , linalg.inv : 역행렬 계산 , linalg.solve : A가 정사각 행력일때 Ax=b를 만족하는 x를 구한다.
# identity(6)
a09 = np.identity(6)
print('\nidentity:\n', a09)

# 4-2 난수 생성
# samples = np.random.normal(size=(4,4)) : 표준분포
# seed , permutation , shuffle , rand 균등분포 , randn 표준편차1 평균0 정규분포 , binominal , chisquare , uniform
# randint (3)  정수값 1개


# pandas 소개 및 Series 소개
# 고수준 자료구조 제공
# index 와 data
o = pd.Series([4, 7, -5, 3])  # list형식이나 ndimension 넣으면 됨.
print('obj:\n', o)  # index + value 형식
print('obj values:\n', o.values)
print('obj index:\n', o.index)
print('o.v>0:', o[o > 0])

# 5-2 data frame
# index + column 과 data  (spredsheet와 같음)
# data는 dictionary 안에 list들을 가진다.
# frame['state']  or frame.state
# 값음 frame.loc['three]

o01 = {'state': ['a', 'b'],
       'year': [1000, 1500],
       'pop': [1.5, 2]}
#f01: DataFrame = pd.DataFrame(o01, columns=['year', 'state', 'pop', 'debt'], index=['one', 'two', 'three', 'four', 'five'])
#print('dataFrame:', f01)
# f01['two'] 로는 안됨
# f01은 2차원 array가 됨.

# pandas 핵심 기술 : 재색인
# .reindex 새롭게 색인 객체 사용 : ffill , bfill option

o02 = pd.Series([4.5,7.2,-5.3,3.6] , index=['d','b','a','c'])
print('\nseries:\n',o02)

o03 = o02.reindex(['a','b','c','d','e'])
print('\nreindex:\n',o03)

o04= pd.Series(['blue','purple','yellow'], index = [0,2,4])
print('\nnew series:\n',o04)
print('\nreindex:\n', o04.reindex(range(6),method='ffill'))

# 6-2 색인 선택 및 삭제
#  .drop   => 원본에서 그 부분만 제외하고 가져온다는 의미   => new_obj = obj.drop('c')
o05 = pd.DataFrame(np.arange(16).reshape((4,4)), index=['O','C','U','N'] , columns=['one','two','three','four'])
print ('\no05\n',o05)
print ('\no05.drop index C U\n',o05.drop(['C','U']))
print ('\no05.drop columns one three\n',o05.drop(['one','three'], axis='columns'))

# 6-3 산술연산
# add , sub , div , mul
# 객체를 연산할때 짝이 맞지 않는 색인이 있다면 결과에 두 색인이 통합된다.  df1.add(df2,fill_value=0)
# numpy 배열의 연산처럼 DataFrame과 Series 간의 연산도 잘 정의되어있다.


# 7-1 함수 적용과 매핑
lf = lambda xx:xx.max() - xx.min()
print ('\nlamda function\n',o05.apply(lf),'\n')
print ('\nlamda function row(axis=1)\n',o05.apply(lf,axis=1),'\n')

def f1(x):
    return pd.Series([x.min(), x.max()], index = ['min','max'])
print('\nf1 function\n', o05.apply(f1), '\n')


# 7-2 정렬 순위 중복 색인
# 정렬 : obj.sort_index()  -> index기준으로 sort   sort_index(axis=1)column 기준 소트   / sort_values()
#  frame에서는 sort_values(by=['a','b'])  여러개의 key의 순서를 가지고 sort

# 순위 : 몇번째 인지?
# obj.rank()

# 중복되는게 있는지? : obj.index.is_unique
# boj['a'] 중복된 것이 있으면 중복된 것을 모두 뿌려줌



# 8-1 상관관계와 공분산
# pandas는 numpy와 겹치는데 누락된 데이터를 제외하도록 설계되었다.

df01 = pd.DataFrame([[1.4,np.nan],[7.1,-4.5]], index=['a','b'], columns=['one','two'])
print('\n8-1 df\n',df01,'\n')
print('\n8-1 df describe\n',df01.describe())

# 공분산 : 상승하는 경향의 상관관계가 있다면 공분산은 양수가 될 것이다.
#price = pd.read_pickle('yahoo_price.pk1') <- 수행안됨
#volume = pd.read_pickle('yahoo_volume.pk1')
#print('\n8-1 price\n',price,'\n')
#print('\n8-1 volume\n',volume,'\n')

# 누락된 데이터 처리를 할수 있어야 한다.
# NaN (not a number) : dropna , fillna , isnull , notnull


# 9-1 텍스트 파일 로딩
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

#import mathplotlib.pyplot as plt
#plt.title('Births Sum')
#plt.plot(big_frame.groupby(['year','sex']).sum())
#plt.show()



# 18-2 선거 데이터

fec = pd.read_csv()
print(fec.info())
print(fec.loc[123456])  # 일부분을 찍어보는 것이다.

#기부자와 선거 자금에서 찾을수 있는 패텅에 대한 통계를 추출하기 위해 이 데이터를 적당한 크기로 쪼개서  나누는 다양한 방법을 찾을수 있다.
# 정당 가입 여부에 대한 데이터가 없으므로 추가해 준다. (이름을 기준으로)
# unique 메서드를 이용해서 모든 정당 후보 목록을 얻어온다.

unique_cands = fec.cand_nm.unique()
print(unique_cands)   # 후보자의 이름들만 뽑아온다.

parties = {'Bachmann, Michelle':'Republican' ,
           'Cain, Herman' : 'Republican'
           }

# 새로운 칼럼을 추가한다.
fec['party'] = fec.cand_nm.map(parties)
print(fec['party'].value_counts() )  # 정당별로 기부한 숫자 확인 가능
print(fec.loc(3000))   # 3000번째를 찍어본다.   party가 포함된 것을 확인 가능

# receipt_amt : 기부한 금액
# 분석을 단순화 하기 위해 기부금이 양수인 데이터만 골라낸다.
fec = fec[fec.contb_receipt_amt > 0]

# 바락 오바마와 미트 롭니가 양대 후보이므로 두 후보의 기부금 정보만 추려낸다.
fec_mrbo = fec[fec.cand_nm.isin(['Obama, Barack', 'Romney, Mitt'])]
print(fec_mrbo)

# 직업별 기부 숫자
print(fec.contbr_occupation.value_counts()[:10] )

# 작업 유형은 같지만 이름이 다른 결과가 있으므로 하나의 직업을 다른 직업으로 매핑한다.
occ_mapping = {
    'INFORMATION REQUESTED PER BEST EFFORTS' : 'NOT PROVIDED',
    'C.E.O':'CEO'
}

# 매핑 정보가 없는 직업은 키를 그대로 가져온다.
f = lambda x:occ_mapping.get(x,x)    # x,x는 처음 x가 없으면 원래 값인 x
fec.contbr_occupation = fec.contbr_occupation.map(f)

# 고용주도 직업과 같이 변경 가능

# 피벗테이블을 사용해서 정당과 직업별로 테이터를 집계한 다음 최소한 2백만불 이상 기분한 직업군만 골라낸다.
print('pivot table')
by_occupation = fec.pivot_table('contb_receipt_amt', index = 'contbr_occupation' , columns='party',aggfunc='sum')
#'contb_receipt_amt' 만 추스려서 index로 group화 한다. columns는 party 별로 (민주당과 공화당으로 나누어짐)  , 적용할 function은 정당별,직업별로 sum을 한다.
print(by_occupation)
# Row 직업별 ,   열은 정당 2가지
# party     민주 , 공화
# 직업      sum , sum

over_2mm = by_occupation[by_occupation.sum(axis=1) > 2000000]
# axis=1 이면 row 단위로 더함.
print(over_2mm)

import matplotlib.pyplot as plt

over_2mm.plot(kind='barh')
plt.show()

# 오바마 후보와 몸니 후보별로 가장 많은 금액ㅇ을 기부한 직군을 찾는다.
# 이를 위해 후보 이름으로 그룹을 묶고 top 메소드를 사용한다.
def get_top_amounts(group,key,n=5):
    totals = group.goupby(key)['contb_receipt_amt'].sum()  # key로 group화해서 그 안의 amount의 합을 구한다.
    return totals.sort_values(ascending=False)[:n]  # 기부 많이한 n개만 뽑는다.

grouped = fec_mrbo.groupby('cand_nm')  # 후보자로 group화 한후에...  직업군으로 그룹해서 많이 기부한 7개의 직업군들을 뽑는다.
print(grouped.apply(get_top_amounts, 'contbr_occupation',n=7))

# 기부금액: cut 함수를 이용해서 기부 규모별 버킷을 만들어 기부자 수를 분할 한다.
bins = np.array([0,1,10,100,1000,10000,100000,1000000,10000000])
labels = pd.cut(fec_mrbo.contb_receipt_amt,bins)
print(labels)  # 각기 기분한 몇번째 사람들이 어떤 구간에 기부를 했는지를 보여줌.

# 위의 데이터를 기부자 이름과 버킷 이름으로 그룹을 묶어 기부 규모의 금액에 따라 히스토그램을 그릴수 있다.
# 후보자 별로 group ->  bins로 group화
grouped = fec_mrbo.groupby(['cand_nm',labels])
print(grouped.size())  # can_nm 다음에 amt로 grouping
print(grouped.size().unstack(0))  # 피벗을 하면 더 쉽게 볼수 있다   구간별 \ 후보자 이름별

# 결과를 확인해보면 오바마는 롬니보다 적은 금액의 기부를 훨씬 많이 받았다.
# 기부금액을 모두 더한 후 버킷별로 정교화해서후보별 전체 기부금액 대비 비율을 시각화 한다.
bucket_sums = grouped.contb_receipt_amt.sum().unstack(0) # 기부 총 금액
print(bucket_sums)

normed_sums = bucket_sums.div(bucket_sums.sum(axis=1),axis=0)  # 비율로 바꿔주는 div
    # row로 더한 값sum 을 가지고 각기 나누어준다.
print(normed_sums)
normed_sums[:-2].plot(kind='barh',stacked==True)  # 기부금액 순에서 가장 높은 2개[:-2]는 오바마에게만 있으니 제외
plt.show()



# 19-1 날짜와 시간 처리
from dateutil.parser import parse
print(parse('2017-01-03'))

# datetime 객체로 index로 구성
# TimeSeries 인덱싱

index = pd.date_range('4/1/2007','1/1/2020')
print(index)



