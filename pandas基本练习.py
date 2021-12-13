import numpy as np
from numpy.core.defchararray import title
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
##数据分组----酒类消费数据
path3 ='./数据挖掘/practice/home/aistudio/exercise_data/drinks.csv'
drinks = pd.read_csv(path3)
drinks.head()

#哪个大陆(continent)平均消耗的啤酒(beer)更多
drinks.groupby('continent').beer_servings.mean()

#打印出每个大陆(continent)的红酒消耗(wine_servings)的描述性统计值
drinks.groupby('continent').wine_servings.describe()

#打印出每个大陆每种酒类别的消耗平均值
drinks.groupby('continent').mean()

# 打印出每个大陆每种酒类别的消耗中位数
drinks.groupby('continent').median()

#打印出每个大陆对spirit饮品消耗的平均值，最大值和最小值
drinks.groupby('continent').spirit_servings.agg(['mean','min','max'])

#---------------------------------------------------------------------------------------------
##Apply函数----1960 - 2014 美国犯罪数据
path4 = './数据挖掘/practice/home/aistudio/exercise_data/US_Crime_Rates_1960_2014.csv'
crime = pd.read_csv(path4)
crime.head()

#每一列(column)的数据类型是什么样的
crime.info()

#将Year的数据类型转换为 datetime64
crime.Year = pd.to_datetime(crime.Year,format='%Y')
crime.info()

#将列Year设置为数据框的索引
crime = crime.set_index('Year',drop=True)
crime.head()

#删除名为Total的列
del crime['Total']

#按照Year对数据框进行分组并求和
crime.resample('10AS').sum() #?不明白啥意思

#何时是美国历史上生存最危险的年代
crime.idxmax(0)

#-------------------------------------------------------------------------------------
##合并----探索虚拟姓名数据
raw_data_1 = {
        'subject_id': ['1', '2', '3', '4', '5'],
        'first_name': ['Alex', 'Amy', 'Allen', 'Alice', 'Ayoung'], 
        'last_name': ['Anderson', 'Ackerman', 'Ali', 'Aoni', 'Atiches']}

raw_data_2 = {
        'subject_id': ['4', '5', '6', '7', '8'],
        'first_name': ['Billy', 'Brian', 'Bran', 'Bryce', 'Betty'], 
        'last_name': ['Bonder', 'Black', 'Balwner', 'Brice', 'Btisan']}

raw_data_3 = {
        'subject_id': ['1', '2', '3', '4', '5', '7', '8', '9', '10', '11'],
        'test_id': [51, 15, 15, 61, 16, 14, 15, 1, 61, 16]}

data1 = pd.DataFrame(raw_data_1, columns = ['subject_id', 'first_name', 'last_name'])
data2 = pd.DataFrame(raw_data_2, columns = ['subject_id', 'first_name', 'last_name'])
data3 = pd.DataFrame(raw_data_3, columns = ['subject_id','test_id'])

#data1和data2两个数据框按照行的维度进行合并，命名为all_data
all_data = pd.concat([data1, data2])
all_data

#data1和data2两个数据框按照列的维度进行合并，命名为all_data_col
all_data_col = pd.concat([data1,data2],axis=1)
all_data_col

#subject_id的值对all_data和data3作合并
pd.merge(all_data,data3,on='subject_id')

#对data1和data2按照subject_id作连接
pd.merge(data1,data2,on='subject_id',how='inner')

#找到 data1 和 data2 合并之后的所有匹配结果
pd.merge(data1,data2,on='subject_id',how='outer')

#----------------------------------------------------------------------------------
##统计----风速数据
path6 = './数据挖掘/practice/home/aistudio/exercise_data/wind.data'
data = pd.read_table(path6, sep = "\s+", parse_dates = [[0,1,2]]) 
data.head()

#2061年？我们真的有这一年的数据？创建一个函数并用它去修复这个bug
def fix_century(x):
    year = x.year - 100 if x.year > 1989 else x.year
    return datetime.date(year, x.month, x.day)

# apply the function fix_century on the column and replace the values to the right ones
data['Yr_Mo_Dy'] = data['Yr_Mo_Dy'].apply(fix_century)

# data.info()
data.head()

#将日期设为索引，注意数据类型，应该是datetime64[ns]
data['Yr_Mo_Dy'] = pd.to_datetime(data['Yr_Mo_Dy'])
#设置索引
data = data.set_index('Yr_Mo_Dy')
data.head()
#对应每一个location，一共有多少数据值缺失
data.isnull().sum()
#对应每一个location，一共有多少完整的数据值
#data.shape
data.shape[0] - data.isnull().sum()
#对于全体数据，计算风速的平均值
data.mean().mean()
#创建一个名为loc_stats的数据框去计算并存储每个location的风速最小值，最大值，平均值和标准差
loc_stats = pd.DataFrame()
loc_stats['min'] = data.min()
loc_stats['max'] = data.max()
loc_stats['mean'] = data.mean()
loc_stats['std'] = data.std()
loc_stats
#创建一个名为day_stats的数据框去计算并存储所有location的风速最小值，最大值，平均值和标准差
day_stats = pd.DataFrame()
day_stats['min'] = data.min(axis=1)
day_stats['max'] = data.max(axis=1)
day_stats['mean'] = data.mean(axis=1)
day_stats['std'] = data.std(axis=1)
day_stats
#对于每一个location，计算一月份的平均风速.注意，1961年的1月和1962年的1月应该区别对待
data['date'] = data.index
data['month'] = data['date'].apply(lambda data: data.month)
data['year'] = data['date'].apply(lambda data: data.year)
data['day'] = data['date'].apply(lambda data: data.day)
#get all value from month 1 and assign to january_winds
january_winds = data.query('month == 1')
january_winds
# gets the mean from january_winds, using .loc to not print the mean of month, year and day
january_winds.loc[:,'RPT':'MAL'].mean()
#对于数据记录按照年为频率取样
data.query('month == 1 and day == 1')
#对于数据记录按照月为频率取样
data.query('day == 1')

#--------------------------------------------------------------------------------------------
##可视化-----探索泰坦尼克灾难数据
path7 = './数据挖掘/practice/home/aistudio/exercise_data/train.csv'
titantic = pd.read_csv(path7)
titantic.head()
#将PassengerId设置为索引
titantic.set_index('PassengerId').head()
#绘制一个展示男女乘客比例的扇形图
males = (titantic['Sex'] == 'male').sum()
females = (titantic['Sex'] == 'female').sum()
proportions = [males,females]
plt.pie(proportions,labels=['Males','Females'],shadow=False,
        colors=['blue','red'],explode=(0.15,0),startangle=90,autopct='%1.1f%%')
plt.axis('equal')
plt.title('Sex Proportion')
plt.tight_layout()
plt.show()
#绘制一个展示船票Fare, 与乘客年龄和性别的散点图
lm = sns.lmplot(x = 'Age', y = 'Fare', data = titantic, hue = 'Sex', fit_reg=False)
# set title
lm.set(title = 'Fare x Age')

# get the axes object and tweak it
axes = lm.axes
axes[0,0].set_ylim(-5,)
axes[0,0].set_xlim(-5,85)
plt.show()
#有多少人生还？
titantic.Survived.sum()
#绘制一个展示船票价格的直方图
df = titantic.Fare.sort_values(ascending=False)
binsVal = np.arange(0,600,10)
binsVal
plt.hist(df,bins=binsVal)
plt.xlabel('Fare')
plt.ylabel('Frequency')
plt.title('Fare Payed Histrogram')
plt.show()

#-------------------------------------------------------------------------
##创建数据框----Pokemon数据
raw_data = {"name": ['Bulbasaur', 'Charmander','Squirtle','Caterpie'],
            "evolution": ['Ivysaur','Charmeleon','Wartortle','Metapod'],
            "type": ['grass', 'fire', 'water', 'bug'],
            "hp": [45, 39, 44, 45],
            "pokedex": ['yes', 'no','yes','no']                        
            }
pokemon = pd.DataFrame(raw_data)
pokemon.head()
#数据框的列排序是字母顺序，请重新修改为name, type, hp, evolution, pokedex这个顺序
pokemon = pokemon[['name','type','hp','evolution','pokedex']]
pokemon
#添加一个列place
pokemon['place'] = ['park','street','lake','forest']
pokemon
#查看每个列的数据类型
pokemon.dtypes

#-----------------------------------------------------------------
##时间序列----apple公司股价数据
path9 = './数据挖掘/practice/home/aistudio/exercise_data/Apple_stock.csv'
apple = pd.read_csv(path9)
apple.head()
#查看每一列的数据类型
apple.dtypes
#将Date这个列转换为datetime类型
apple.Date = pd.to_datetime(apple.Date)
apple['Date'].head()
#将Date设置为索引
apple = apple.set_index('Date')
apple.head()
#有重复的日期吗
apple.index.is_unique
#将index设置为升序
apple.sort_index(ascending=True).head(40)
#找到每个月的最后一个交易日(business day)
apple.resample('BM').last()
#数据集中最早的日期和最晚的日期相差多少天？
(apple.index.max()-apple.index.min()).days
#在数据中一共有多少个月？
apple_months = apple.resample('BM').mean()
len(apple_months.index)
#按照时间顺序可视化Adj Close值
appl_open = apple['Adj Close'].plot(title='Apple Stock')
fig = appl_open.get_figure()
fig.set_size_inches(13.5,9)
plt.show()

#--------------------------------------------------------------------
##删除数据----鸢尾花数据
path10 ='./数据挖掘/practice/home/aistudio/exercise_data/iris.csv'
iris = pd.read_csv(path10)
iris.head()
#创建数据框的列名称
iris = pd.read_csv(path10,names=['sepal_length','sepal_width','petal_length','petal_width','class'])
iris.head()
#数据框中有缺失值吗？
pd.isnull(iris).sum()
#将列petal_length的第10到19行设置为缺失值
iris.iloc[10:20,2:3] = np.nan
iris.head(20)
#将缺失值全部替换为1.0
iris.petal_length.fillna(1,inplace=True)
iris.head(20)
#删除列class
del iris['class']
iris.head()
#将数据框前三行设置为缺失值
iris.iloc[0:3 ,:] = np.nan
iris.head()
# 删除有缺失值的行
iris = iris.dropna(how='any')
iris.head()
#重新设置索引
iris =iris.reset_index(drop=True)
iris.head()