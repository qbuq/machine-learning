##One Hot编码
import importlib
import matplotlib
from matplotlib.pyplot import clim, figure, plot
from mlxtend import feature_selection
from numpy.core.records import array
from numpy.lib.shape_base import _array_split_dispatcher
import pandas as pd
from pandas.core.indexes.base import InvalidIndexError
from scipy.sparse.linalg.interface import aslinearoperator
from seaborn import regression
from seaborn.distributions import ecdfplot
g = pd.DataFrame({'gender':['man','woman','woman','man','woman']})
g
pd.get_dummies(g)
df = pd.DataFrame({'gene_seg':['A','B','B','A','A'],
                    'dis':['gall','hyp','gall','hyp','hyp']})
df
pd.get_dummies(df)
persons = pd.DataFrame({'name':['Newton','Andrew Ng','Jodan','Bill Gates'],
                        'color':['white','yellow','black','white']})
persons
df_dum = pd.get_dummies(persons['color'],drop_first=True)
df_dum
persons.merge(df_dum,left_index=True,right_index=True)

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
features = ohe.fit_transform(persons[['color']])
features.toarray()
features.toarray()[:,1:]

#项目案例
df = pd.DataFrame({
    'color':['green','red','blue','red'],
    'size':['M','L','XL','L'],
    'price':[29.9,69.9,99.9,59.9],
    'classlabel':['class1','class2','class1','class1']
})
df
size_mapping = {'XL':3,'L':2,'M':1}
df['size'] = df['size'].map(size_mapping)
df
ohe = OneHotEncoder()
fs = ohe.fit_transform(df[['color']])
fs_ohe = pd.DataFrame(fs.toarray()[:,1:],columns=['color_green','color_red'])
df = pd.concat([df,fs_ohe],axis=1)
df

ohe = OneHotEncoder()
c1 = ohe.fit_transform(df[['classlabel']])
c1.toarray()

df = pd.read_csv("./数据挖掘/practice/home/aistudio/data20513/breast-cancer.data",header=None).iloc[:,1:]
df.shape
dataset = df.values
X = dataset[:,0:8]
X = X.astype(str)
Y = dataset[:,8]

from sklearn.preprocessing import LabelEncoder
import numpy as np
encoded_x = None
for i in range(0,X.shape[1]):
    label_encoder = LabelEncoder()  #数值化
    feature = label_encoder.fit_transform(X[:,i])
    feature = feature.reshape(X.shape[0],1)
    onehot_encoder = OneHotEncoder(sparse=False)  #OneHot编码
    feature = onehot_encoder.fit_transform(feature)
    if encoded_x is None:
        encoded_x = feature
    else:
        encoded_x = np.concatenate((encoded_x,feature),axis=1)
print('X shape: :',encoded_x.shape)

label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(Y)
label_encoder_y = label_encoder.transform(Y)
label_encoder_y

##数据变换
data = pd.read_csv('./数据挖掘/practice/home/aistudio/data20514/freefall.csv',index_col=0)
data.describe()

import seaborn as sns
ax = sns.scatterplot(x='time',y='location',data=data)
import matplotlib.pyplot as plt
plt.show()

import numpy as np
data
data.drop([0],inplace=True) #去掉0，不计算log0
data['logtime'] = np.log10(data['time'])
data['logloc'] = np.log10(data['location'])
data.head()
ax2 = sns.scatterplot(x='logtime',y='logloc',data=data)
plt.show()

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(data['logtime'].values.reshape(-1,1),data['logloc'].values.reshape(-1,1))
(reg.coef_,reg.intercept_)

import numpy as np
X = np.arange(6).reshape(3,2)
X
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(2)
poly.fit_transform(X)

#项目案例
dc_data = pd.read_csv('./数据挖掘/practice/home/aistudio/data20514/sample_data.csv')
dc_data.head()

h = plt.hist(dc_data['AIR_TIME'],bins=100)
plt.show()

from scipy import stats
transform = (dc_data[['AIR_TIME']].values).flatten()
transform
dft = stats.boxcox(transform)[0]
hbc = plt.hist(dft,bins=100)
plt.show()

from sklearn.preprocessing import power_transform
dft2 = power_transform(dc_data[['AIR_TIME']],method='box-cox')
hbcs = plt.hist(dft2,bins=100)
plt.show()

#动手练习
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import _final_estimator_has, make_pipeline

df = pd.read_csv('./数据挖掘/practice/home/aistudio/data20514/xsin.csv')
colors = ['teal','yellowgreen','gold']
plt.scatter(df['x'],df['y'],color='navy',s=30,marker='o',label='training points')

for count,degree in enumerate([3,4,5]):
    model = make_pipeline(PolynomialFeatures(degree),Ridge())
    model.fit(df[['x']],df[['y']])
    y_pre = model.predict(df[['x']])
    plt.plot(df['x'],y_pre,color=colors[count],linewidth=2,label="degree %d" %degree)
plt.legend()
plt.show()

##3.5特征离散化
#3.5.1无监督离散化
import pandas as pd
ages = pd.DataFrame({'years':[10,14,30,53,67,32,45],'name':['A','B','C','D','E','F','G']})
ages
pd.cut(ages['years'],3)
pd.qcut(ages['years'],3)

klass = pd.cut(ages['years'],3,labels=[0,1,2])
ages['label'] = klass
ages

ages2 = pd.DataFrame({'years':[10,14,30,53,300,32,45],'name':['A','B','C','D','E','F','G']})
klass2 = pd.cut(ages['years'],3,labels=['Young','Middle','Senior'])
ages2['label'] = klass2
ages2

ages2 = pd.DataFrame({'years':[10,14,30,53,300,32,45],'name':['A','B','C','D','E','F','G']})
klass2 = pd.cut(ages['years'],bins=[9,30,50,300],labels=['Young','Middle','Senior'])
ages2['label'] = klass2
ages2

from sklearn.preprocessing import KBinsDiscretizer
kbd = KBinsDiscretizer(n_bins=3,encode='ordinal',strategy='uniform')
trans = kbd.fit_transform(ages[['years']])
trans #二维数组，(n,1)的
ages['kbd'] = trans[:,0]
ages

#项目案例
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
iris = load_iris()

iris.feature_names

X = iris.data
y = iris.target
X = X[:,[2,3]]
import matplotlib.pyplot as plt
plt.scatter(X[:,0],X[:,1],c=y,alpha=0.3,cmap=plt.cm.RdYlBu,edgecolors='black')
plt.show()

Xd = KBinsDiscretizer(n_bins=10,encode='ordinal',strategy='uniform').fit_transform(X)
plt.scatter(Xd[:,0],Xd[:,1],c=y,cmap=plt.cm.RdYlBu,edgecolors='black')
plt.show()

dtc = DecisionTreeClassifier(random_state=0)
score1 = cross_val_score(dtc,X,y,cv=5)
score2 = cross_val_score(dtc,Xd,y,cv=5)
np.mean(score1),np.mean(score2)
np.std(score1),np.std(score2)

km = KBinsDiscretizer(n_bins=3,encode='ordinal',strategy='kmeans').fit_transform(X)
s =cross_val_score(dtc,km,y,cv=5)
np.mean(s),np.std(s)

#动手练习
import numpy as np
rnd = np.random.RandomState(42)
X = rnd.uniform(-3,3,size=100)
y = np.sin(X) + rnd.normal(size=len(X))/3
X = X.reshape(-1,1)
X

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题-设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
#离散化
kbd = KBinsDiscretizer(n_bins=10,encode='onehot')
X_binned = kbd.fit_transform(X)

#利用线性回归模型对原始数据进行预测
fig, (ax1, ax2) = plt.subplots(ncols=2,sharey=True,figsize=(10,4))
line = np.linspace(-3,3,1000,endpoint=False).reshape(-1,1)
lreg = LinearRegression().fit(X,y)
ax1.plot(line,lreg.predict(line),linewidth=2,color='green',label='线性回归')
dreg = DecisionTreeRegressor(min_samples_split=3,random_state=0).fit(X,y)
ax1.plot(line,dreg.predict(line),linewidth=2,color='blue',label='决策树回归')
ax1.plot(X[:,0],y,'o',c='k')
ax1.legend(loc='best')
ax1.set_ylabel('回归输出')
ax1.set_xlabel('输入特征')
ax1.set_title('未离散化结果')
#对离散化之后的数据进行预测
line_binned = kbd.transform(line)
lreg_binned = LinearRegression().fit(X_binned,y)
ax2.plot(line,lreg_binned.predict(line_binned),linewidth=2,color='green',linestyle='-',label='线性回归')
dreg_binned = DecisionTreeRegressor(min_samples_split=3,random_state=0).fit(X_binned,y)
ax2.plot(line,dreg_binned.predict(line_binned),linewidth=2,color='red',linestyle='-',label='决策树回归')
ax2.plot(X[:,0],y,'o',c='k')
ax2.vlines(kbd.bin_edges_[0],*plt.gca().get_ylim(),linewidth=1,alpha=0.2)
ax2.legend(loc='best')
ax2.set_ylabel('回归输出')
ax2.set_xlabel('输入特征')
ax2.set_title('已离散化结果')

plt.show()

#3.5.2有监督离散化
import entropy_based_binning as ebb
import numpy as np
A = np.array([[1,1,2,3,4],[1,1,0,1,0]])
ebb.bin_array(A,nbins=2,axis=1)


##4.1封装器法
#4.1.1循环特征选择
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.data import wine_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X,y = wine_data()
X.shape
X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y,test_size=0.3,random_state=1)
std = StandardScaler()
X_train_std = std.fit_transform(X_train)

knn = KNeighborsClassifier(n_neighbors=3)
sfs = SFS(estimator=knn,k_features=4,forward=True,floating=False,verbose=2,scoring='accuracy',cv=0)
sfs.fit(X_train_std,y_train)

sfs.subsets_

#项目案例
knn = KNeighborsClassifier(n_neighbors=3)
sfs1 = SFS(estimator=knn,
            k_features=4,
            forward=True,
            floating=True,
            verbose=2,
            scoring='accuracy',
            cv=0)
sfs1.fit(X_train_std,y_train)

import matplotlib.pyplot as plt
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
fig = plot_sfs(sfs.get_metric_dict(),kind='std_err')
plt.show()

sfs.get_metric_dict()

knn = KNeighborsClassifier(n_neighbors=3)
sfs2 = SFS(estimator=knn,
            k_features=(3,10),
            forward=True,
            floating=True,
            verbose=0,
            scoring='accuracy',
            cv=5)
sfs2.fit(X_train_std,y_train)
fig = plot_sfs(sfs2.get_metric_dict(),kind='std_err')
plt.show()

sfs2.subsets_[6]

#动手练习
import pandas as pd
df = pd.read_csv('./数据挖掘/practice/home/aistudio/data20528/housprice.csv')
df.shape

cols = list(df.select_dtypes(include=['int64','float64']).columns)
data = df[cols]

X_train,X_test,y_train,y_test = train_test_split(data.drop('SalePrice',axis=1),data['SalePrice'],test_size=0.2,random_state=1)
X_train.fillna(0,inplace=True) #用0填充缺失值
from sklearn.ensemble import RandomForestRegressor

sfs3 = SFS(RandomForestRegressor(),
            k_features=10,
            forward=True,
            verbose=0,
            cv=5,
            n_jobs=-1,
            scoring='r2')
sfs3.fit(X_train,y_train)
sfs3.k_feature_names_

##4.1.2穷举特征选择
mini_data = X_train[X_train.columns[list(sfs3.k_feature_idx_)]]
mini_data.shape
import numpy as np
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
efs = EFS(RandomForestRegressor(),min_features=1,max_features=5,scoring='r2',n_jobs=-1)
efs.fit(np.array(mini_data), y_train)
mini_data.columns[list(efs.best_idx_)]

#项目案例
path = './数据挖掘/practice/home/aistudio/data20528'
paribas_data = pd.read_csv(path+'/paribas_data.csv',nrows=20000)
paribas_data.shape

num_colums = ['int16','int32','int64','float16','float32','float64']
numerical_columns = list(paribas_data.select_dtypes(include=num_colums).columns)
paribas_data = paribas_data[numerical_columns]
paribas_data.shape

from sklearn.model_selection import train_test_split
train_features,test_features,train_labels,test_labels = train_test_split(
                paribas_data.drop(labels=['target','ID'],axis=1),
                paribas_data['target'],
                test_size=0.2,random_state=41
                ) 

correlated_features = set()
correlation_matrix = paribas_data.corr()
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i,j]) > 0.3:
            colname = correlation_matrix.columns[i]
            correlated_features.add(colname)
train_features.drop(labels=correlated_features,axis=1,inplace=True)
test_features.drop(labels=correlated_features,axis=1,inplace=True)
train_features.shape, test_features.shape

from mlxtend.feature_selection import ExhaustiveFeatureSelector
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.metrics import roc_auc_score

feature_selector = ExhaustiveFeatureSelector(RandomForestClassifier(n_jobs=-1),
                min_features=2,
                max_features=4,
                scoring='roc_auc',
                print_progress=True,
                cv=2)
import numpy as np
features = feature_selector.fit(np.array(train_features.fillna(0)),train_labels)
filtered_features = train_features.columns[list(features.best_idx_)]
filtered_features

#动手练习
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS

iris = datasets.load_iris()
X = iris.data
y = iris.target

efs = EFS(RandomForestClassifier(n_jobs=-1),
        min_features=2,max_features=4,
        print_progress=True,cv=2)
efs.fit(X,y)
efs.best_idx_

##递归特征消除
from sklearn.feature_selection import RFE
rfe = RFE(RandomForestRegressor(),n_features_to_select=5)
rfe.fit(np.array(mini_data),y_train)
rfe.ranking_

mini_data.columns[rfe.ranking_ == 1]

#项目案例
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
df_train = pd.read_csv(path+'/train.csv')
df_test = pd.read_csv(path+'/test.csv')
train_samples = df_train.shape[0]
test_samples = df_test.shape[0]
train_test = pd.concat((df_train,df_test),axis=0,ignore_index=True,sort=False)
features = [x for x in df_train.columns]
cat_features = [x for x in df_train.select_dtypes(include=['object']).columns]
num_features = [x for x in df_train.select_dtypes(exclude=['object']).columns]
print('\n Categorical features: %d' % len(cat_features))
print('\n Numerical features: %d' %len(num_features))

le = LabelEncoder()
for c in cat_features:
    train_test[c] = le.fit_transform(train_test[c])
X_train = train_test.iloc[:train_samples,:].drop(['id','loss'],axis=1)
X_test = train_test.iloc[train_samples:,:].drop(['id'],axis=1)

from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

y_train = df_train['loss']
rfr = RandomForestRegressor(n_estimators=100,max_features='sqrt',max_depth=12,n_jobs=-1)
rfecv = RFECV(estimator=rfr,
            step=10,
            cv=3,
            min_features_to_select=10,
            scoring='neg_mean_absolute_error',
            verbose=2)
strat_time = datetime.now()
rfecv.fit(X_train,y_train)
end_time = datetime.now()
m, s = divmod((end_time-strat_time).total_seconds(),60)
print('Time taken:{0}minutes and {1} seconds.'.format(m,round(s,2)))

import matplotlib.pyplot as plt
plt.figure()
plt.xlabel('Number of features tests x 10')
plt.ylabel('Cross-validation score(negative MAE)')
plt.plot(range(1,len(rfecv.grid_scores_)+1),rfecv.grid_scores_)
plt.show()

ranking =pd.DataFrame({'Features':features})
ranking.drop([0,131],inplace=True)
ranking['rank'] = rfecv.ranking_
ranking.sort_values('rank',inplace=True)
ranking.head(10)

#动手练习
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000,n_features=25,n_informative=3,n_clusters_per_class=1,random_state=0)

svc = SVC(kernel='linear')
rfecv = RFECV(estimator=svc,step=1,cv=StratifiedKFold(2),scoring='accuracy')
rfecv.fit(X,y)
print('Optimal number of features:%d' %rfecv.n_features_)

plt.figure()
plt.xlabel('Number of features selected')
plt.ylabel('Cross validation score(nb of correct classifications)')
plt.plot(range(1,len(rfecv.grid_scores_)+1),rfecv.grid_scores_)
plt.show()

##5.1无监督特征抽取
#5.1.1主成分分析
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
X[:4]
X.shape

from sklearn.decomposition import PCA
import numpy as np
pca = PCA()
X_pca = pca.fit_transform(X)
np.round(X_pca[:4],2)
pca.explained_variance_ratio_

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
X_pca[:4]
pca.explained_variance_ratio_.sum()

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X_train,X_test,y_train,y_test = train_test_split(X,iris.target,test_size=0.3,random_state=0)

clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)

X_train_pca,X_test_pca,y_train_pca,y_test_pca = train_test_split(X_pca,iris.target,test_size=0.3,random_state=0)
clf2 = DecisionTreeClassifier()
clf2.fit(X_train_pca,y_train_pca)
y_pred_pca = clf2.predict(X_test_pca)
accuracy2 = accuracy_score(y_test_pca,y_pred_pca)

print('dataset with 4 features:',accuracy)
print('dataset with 2 features:',accuracy2)

#项目案例
from scipy.io import loadmat
mnist = loadmat('./数据挖掘/practice/home/aistudio/data20537/mnist-original.mat')
mnist

mnist_data = mnist['data'].T
mnist_label = mnist['label'][0]
mnist_data.shape

from sklearn.decomposition import PCA
pca = PCA(0.95)
lower_dimensional_data = pca.fit_transform(mnist_data)
pca.n_components_
lower_dimensional_data.shape

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题-设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

inverse_data = pca.inverse_transform(lower_dimensional_data)
plt.figure(figsize=(8,4))
#原图
plt.subplot(1,2,1)
plt.imshow(mnist_data[1].reshape(28,28),
        cmap=plt.cm.gray,interpolation='nearest',
        clim=(0,255))
plt.xlabel('784 components',fontsize=14)
plt.title('原图',fontsize=20)

#154 principal components 
plt.subplot(1,2,2)
plt.imshow(inverse_data[1].reshape(28,28),
        cmap=plt.cm.gray,interpolation='nearest',
        clim=(0,255))
plt.xlabel('154 components',fontsize=14)
plt.title('特征抽取后的图',fontsize=20)
plt.show()

import time
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from scipy.io import loadmat
import pandas as pd

mnist = loadmat('./数据挖掘/practice/home/aistudio/data20537/mnist-original.mat')
mnist_data = mnist['data'].T
mnist_label = mnist['label'][0]
train_img,test_img,train_lbl,test_lbl = train_test_split(mnist_data,mnist_label,test_size=1/7.0,random_state=0)

scaler = StandardScaler()
scaler.fit(train_img)
train_img = scaler.transform(train_img)
test_img = scaler.transform(test_img)

def logisitic_reg(exp_var):
    #global train_img test_img
    pca = PCA(exp_var)
    pca.fit(train_img)
    lr = LogisticRegression(solver='lbfgs')
    lr.fit(pca.transform(train_img),train_lbl)
    lbl_pred = lr.predict(pca.transform(test_img))
    acc = accuracy_score(test_lbl,lbl_pred)
    return pca.n_components_, acc

v,n,a,t = [],[],[],[]
for i in [None,0.99,0.95,0.90,0.85]:
    start = time.time()
    components, accuracy = logisitic_reg(i)
    stop = time.time()
    delta = stop - start
    v.append(i)
    n.append(components)
    a.append(accuracy)
    t.append(delta)

df = pd.DataFrame({'var_ratio':v,
                    'n_components':n,
                    'accuracy':a,
                    'times':t})

df

import pandas as pd
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
# load dataset into Pandas DataFrame
df = pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width','target'])

from sklearn.preprocessing import StandardScaler
features = ['sepal length','sepal width','petal length','petal width']

# Separating out the features
x = df.loc[:, features].values

# Separating out the target
y = df.loc[:,['target']].values

# Standardizing the features
x = StandardScaler().fit_transform(x)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf,df[['target']]],axis=1)

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1',fontsize=15)
ax.set_ylabel('Principal Component 2',fontsize=15)
ax.set_title('2 component PCA',fontsize=20)

targets = ['Iris-setosa','Iris-versicolor','Iris-virginica']
colors = ['r','g','b']
for target,color in zip(targets,colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep,'principal component 1'],
                finalDf.loc[indicesToKeep,'principal component 2'],
                c = color,
                s = 50)
ax.legend(targets)
ax.grid()
plt.show()

pca.explained_variance_ratio_

#5.1.2因子分析
from sklearn.decomposition import FactorAnalysis
from sklearn import datasets
iris = datasets.load_iris()

fa = FactorAnalysis()
iris_fa = fa.fit(iris.data)
fa.components_

fa = FactorAnalysis(n_components=2)
iris_two = fa.fit_transform(iris.data)
iris_two[:4]

f = plt.figure(figsize=(5,5))
ax = f.add_subplot(1,1,1)
ax.scatter(iris_two[:,0],iris_two[:,1],c=iris.target)
ax.set_title('Factor Analysis 2 Components')
plt.show()

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X = iris.data
X_pca = pca.fit_transform(X)
f = plt.figure(figsize=(5,5))
ax = f.add_subplot(111)
ax.scatter(X_pca[:,0],X_pca[:,1],c=iris.target)
ax.set_title('PCA 2 Components')
plt.show()

#项目案例
import pandas as pd
df = pd.read_csv('./数据挖掘/practice/home/aistudio/data20537/bfi.csv')
df.columns

df.drop(['Unnamed: 0', 'gender', 'education', 'age'],axis=1,inplace=True)
df.info()

df.dropna(inplace=True)
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
chi_square_value,p_value = calculate_bartlett_sphericity(df)
chi_square_value,p_value

from factor_analyzer.factor_analyzer import calculate_kmo
kmo_all,kmo_model = calculate_kmo(df)
kmo_model

from factor_analyzer import FactorAnalyzer
fa = FactorAnalyzer(rotation=None)
fa.fit(df,25)
ev,v = fa.get_eigenvalues()
ev

import matplotlib.pyplot as plt
plt.scatter(range(1,df.shape[1]+1),ev)
plt.plot(range(1,df.shape[1]+1),ev)
plt.title('Scree Plot')
plt.xlabel('Factors')
plt.ylabel('Eigenvalue')
plt.grid()
plt.show()

#动手练习
fa = FactorAnalyzer(rotation="varimax", n_factors=5)
fa.fit(df)
fa.transform(df)

####预测心脏病
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import rainbow 
import warnings
warnings.filterwarnings('ignore')

#划分训练集
from sklearn.model_selection import train_test_split
#标准化
from sklearn.preprocessing import StandardScaler

#导入模型
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#加载数据
dataset = pd.read_csv('./数据挖掘/practice/home/aistudio/data18245/dataset.csv')
dataset.head()
dataset.info()
#共计303个样本，没有缺失值。13个特征，特征target是标签特征。—— 有监督学习

dataset.describe()
#每个特征的数值分布差异很大，观察最大值，特征age是77，而特征chol是564，差异较大。 这时，需要对特征进行规范.

#热力图
rcParams['figure.figsize'] = 20,20
plt.matshow(dataset.corr())
plt.yticks(np.arange(dataset.shape[1]),dataset.columns)
plt.xticks(np.arange(dataset.shape[1]),dataset.columns)
plt.colorbar()
plt.show()

#每个变量的直方图
dataset.hist()
plt.show()
#上面的直方图，可以看到每个特征都有不同的分布范围。 因此，在预测之前使用特征规范中的特征区间化方法，将各个特征数据进行变换，这对分类问题确实很突出。

#另外，对于标签而言，其中的值应该分布比较平衡，否则训练的模型就比较差了。
rcParams['figure.figsize'] = 8,6
plt.bar(dataset['target'].unique(),dataset['target'].value_counts(),color=['grey','green'])
plt.xticks([0,1])
plt.xlabel('Target Classes')
plt.ylabel('Count')
plt.title('Count of each Target Class')
plt.show()
#标签中的两个值虽然不是各50％，但是现在比例已经足够好，可以继续使用，不需要删除或增加数据。

##数据处理
#将分类型特征转化为虚拟变量
dataset = pd.get_dummies(dataset,columns=['sex','cp','fbs','restecg','exang','slope','ca','thal'])
#用StandardScaler()对特征进行区间化
standardscaler = StandardScaler()
columns_to_scale = ['age','trestbps','chol','thalach','oldpeak']
dataset[columns_to_scale] = standardscaler.fit_transform(dataset[columns_to_scale])

##机器学习
#划分数据集
y = dataset['target']
X = dataset.drop(['target'],axis=1)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=0)

#K近邻分类模型
knn_scores = []
for k in range(1,21):
    knn_classifier = KNeighborsClassifier( n_neighbors=k)
    knn_classifier.fit(X_train,y_train)
    knn_scores.append(knn_classifier.score(X_test,y_test))

plt.plot([k for k in range(1,21)], knn_scores,color='red')
for i in range(1,21):
    plt.text(i,knn_scores[i-1],(i,knn_scores[i-1]))
plt.xticks([i for i in range(1,21)])
plt.xlabel('Number of Neighbors(K)')
plt.ylabel('Scores')
plt.title('K Neighbor Classifier scores for different K values')
plt.show()

print('The Score for K neighbors classifier is {}% with {} neighbors.'.format(knn_scores[7]*100,8))

#支持向量机
svc_scores = []
kernels = ['linear','poly','rbf','sigmoid']
for i in range(len(kernels)):
    svc_classifier = SVC(kernel=kernels[i])
    svc_classifier.fit(X_train,y_train)
    svc_scores.append(svc_classifier.score(X_test,y_test))

colors = rainbow(np.linspace(0,1,len(kernels)))
plt.bar(kernels,svc_scores,color=colors)
for i in range(len(kernels)):
    plt.text(i,svc_scores[i],svc_scores[i])
plt.xlabel('Kernels')
plt.ylabel('Scores')
plt.title('Support Vector Classifier for different kernels')
plt.show()

print("The score for Support Vector Classifier is {}% with {} kernel.".format(svc_scores[0]*100, 'linear'))

#决策树分类模型
dt_scores =[]
for i in range(1,len(X.columns)+1):
    dt_classifier = DecisionTreeClassifier(max_features=i,random_state=0)
    dt_classifier.fit(X_train,y_train)
    dt_scores.append(dt_classifier.score(X_test,y_test))
plt.plot([i for i in range(1,len(X.columns)+1)],dt_scores,color='green')
for i in range(1,len(X.columns)+1):
    plt.text(i,dt_scores[i-1],(i,dt_scores[i-1]))
plt.xticks([i for i in range(1,len(X.columns)+1)])
plt.xlabel('Max features')
plt.ylabel('Scores')
plt.title('Decision Tree Classifier scores for different number of maximum features')
plt.show()

print('The score for decision tree classifier is {}% with {} maximum features.'.format(dt_scores[17]*100,[2,4,18]))

#随机森林分类模型
rf_scores = []
estimators = [10,100,200,500,1000]
for i in estimators:
    rf_classifier = RandomForestClassifier(n_estimators=i,random_state=0)
    rf_classifier.fit(X_train,y_train)
    rf_scores.append(rf_classifier.score(X_test,y_test))

colors = rainbow(np.linspace(0,1,len(estimators)))
plt.bar([i for i in range(len(estimators))],rf_scores,color=colors,width=0.8)
for i in range(len(estimators)):
    plt.text(i,rf_scores[i],rf_scores[i])
plt.xticks([i for i in range(len(estimators))],[str(estimator) for estimator in estimators])
plt.xlabel('Number of estimators')
plt.ylabel('Scores')
plt.title('Random Forest Classifier scores for different number of eatimators')
plt.show()

print('The score of Random Forest Classifier is {}% with {} estimators.'.format(rf_scores[1]*100,[100,500]))

###朴素贝叶斯
import numpy as np
X = np.array([[0,1,0,1],
            [1,1,1,0],
            [0,1,1,0],
            [0,0,0,1],
            [0,1,1,0],
            [0,1,0,1],
            [1,0,0,1]])
y = np.array([0,1,1,0,1,0,0])

#伯努利贝叶斯，适用于二分类
from sklearn.naive_bayes import BernoulliNB
clf = BernoulliNB()
clf.fit(X,y)

tomorrow = [[0,0,1,0]]
pre = clf.predict(tomorrow)
pre

#多分类的时候，不能使用伯努利朴素贝叶斯
from sklearn.datasets import make_blobs #生成数据的工具
from sklearn.model_selection import train_test_split
#生成500个样本，分为5个样本
X, y = make_blobs(n_samples=500,centers=5,random_state=9)
X_train,X_test,y_train,y_test = train_test_split(X,y)

nb = BernoulliNB()
nb.fit(X_train,y_train)
nb.score(X_test,y_test)

#高斯朴素贝叶斯
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)
gnb.score(X_test,y_test)

#再用上述数据测试另外一个：多项式朴素贝叶斯(以下代码会报错):原因在于MultinomialNB()要求数据中不能有负数，所以要进行区间变换
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
mnb.fit(X_train,y_train)
mnb.score(X_test,y_test)

#
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

mnb = MultinomialNB()
mnb.fit(X_train_scaled,y_train)
mnb.score(X_test_scaled,y_test)

#案例：乳腺癌筛查
from sklearn.datasets import load_breast_cancer #导入乳腺癌数据集
cancer = load_breast_cancer()

#分类名称：malignant--恶性，benign--良性
cancer.target_names
#特征
cancer['feature_names']
#分别得到X，y
X,y = cancer.data,cancer.target
#划分数据集
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=9)
#使用高斯朴素贝叶斯
gnb = GaussianNB()
gnb.fit(X_train,y_train)
gnb.score(X_test,y_test)

#实战：短信息分类。对sms信息进行分类，两个类别ham,spam
#读入数据
import pandas as pd
sms = pd.read_csv('./数据挖掘/practice/home/aistudio/data33700/sms.csv',sep='\t',header=None,names=['label','message'])
sms.shape
#前5个样本
sms.head()
#类别（标签）数量分布
#将标签转化为0，1
sms['label_num'] = sms['label'].map({'ham':0,'spam':1})
sms.head()
#划分数据集
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(sms['message'],sms['label_num'],random_state=1)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
#特征向量化
from sklearn.feature_extraction.text import CountVectorizer #引入向量化模型
vect = CountVectorizer() #向量化示例
vect.fit(X_train)
X_train_dtm = vect.transform(X_train)
X_test_dtm = vect.transform(X_test)
X_train_dtm
#创建机器学习模型
from sklearn.naive_bayes import MultinomialNB
mlnb = MultinomialNB()
mlnb.fit(X_train_dtm,y_train)
#评估
mlnb.score(X_test_dtm,y_test)

#对比，利用logistic回归
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train_dtm,y_train)
logreg.score(X_test_dtm,y_test)

####机器学习项目案例
#案例1：利用岭回归研究波士顿放假，Ridge Regression在损失函数中加入L2范数作为惩罚项来控制线性模型的复杂程度，提高泛化能力

#读取数据
from sklearn.datasets import load_boston
boston = load_boston()
boston.feature_names
#数据共506个样本，13个特征
boston.data.shape
#使用简单的线性回归模型
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(boston.data,boston.target)
pre = lin_reg.predict(boston.data[:10])
lin_reg.score(boston.data,boston.target)

#使用岭回归
from sklearn.linear_model import Ridge
ridge_reg = Ridge(alpha=0.5)
ridge_reg.fit(boston.data,boston.target)
ridge_reg.score(boston.data,boston.target)

import seaborn as sns
import matplotlib.pyplot as plt
def test_Ridge_alpha(*data):
    X_train,X_test,y_train,y_test = data
    alphas = [0.01,0.2,0.05,0.1,0.2,0.5,1,5,10,20,50,100,200,500,1000]
    scores = []
    for i,alpha in enumerate(alphas):
        ridge_reg = Ridge(alpha=alpha)
        ridge_reg.fit(X_train,y_train)
        scores.append(ridge_reg.score(X_test,y_test))
    sns.lineplot(x=alphas,y=scores)
    plt.show()
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(boston.data,boston.target,test_size=0.3,random_state=31)
test_Ridge_alpha(X_train,X_test,y_train,y_test)

#案例2:利用决策树回归预测波士顿放假
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score,cross_validate
from sklearn.tree import DecisionTreeRegressor

boston = load_boston()
X,y = boston.data,boston.target
features = boston.feature_names

regression_tree = DecisionTreeRegressor(min_samples_split=30,min_samples_leaf=10,random_state=0)
regression_tree.fit(X,y)
score = np.mean(cross_val_score(regression_tree,X,y))
print('Mean squared error: {0}'.format(round(abs(score),2)))

#案例3 Logistic回归实现对鸢尾花数据分类
import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()

sepal_length_list = iris.data[:,0] #花萼长度
sepal_width_list = iris.data[:,1]  #花萼宽度

#构建setosa,versicolor,virginica索引数组
setosa_index_list = iris.target == 0
versicolor_index_list = iris.target == 1
virginica_index_list = iris.target == 2

plt.scatter(sepal_length_list[setosa_index_list],sepal_width_list[setosa_index_list],
                color='red',marker='o',label='setosa')
plt.scatter(sepal_length_list[versicolor_index_list],sepal_width_list[versicolor_index_list],
                color='blue',marker='x',label='versicolor') 
plt.scatter(sepal_length_list[virginica_index_list],sepal_width_list[virginica_index_list],
                color='green',marker='+',label='virginica')   
plt.legend(loc='best',title='iris type')
plt.xlabel('sepal_length(cm)')
plt.show()    

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
#设置训练集和测试集
X_train,X_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size=0.5,random_state=1)
#创建一个Logistic回归分类器
logr = LogisticRegression()
#训练分类器
logr.fit(X_train,y_train)
#预测所属类别
category = logr.predict(X_test)

import numpy as np
#只考虑前两个特征，即花萼长度(sepal length)，花萼宽度(sepal width)
X = iris.data[:,0:2]
y = iris.target

logreg = LogisticRegression(C=1e5) #C:惩罚项系数的倒数，越小，正则化越大
logreg.fit(X,y)
#网格大小
h = 0.02
x_min ,x_max = X[:,0].min() -0.5 , X[:,0].max() + 0.5
y_min, y_max = X[:,1].min() -0.5, X[:,1].max() + 0.5
xx ,yy = np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
#调用ravel()函数将xx和yy平铺，然后使用np.c_将平铺后的列表拼接
#生成需要预测的特征矩阵，每一行的表示一个样本，每一列表示每个特征的取值
pre_data = np.c_[xx.ravel(),yy.ravel()]
Z = logreg.predict(pre_data)
Z = Z.reshape(xx.shape)
plt.figure(1,figsize=(8,6))

plt.pcolormesh(xx,yy,Z,cmap=plt.cm.Paired)
plt.scatter(X[:,0],X[:,1],c=y,edgecolors='k',cmap=plt.cm.Paired)
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.xlim(xx.min(),xx.max())
plt.ylim(yy.min(),yy.max())

plt.xticks(np.arange(x_min,x_max,h*10))
plt.yticks(np.arange(y_min,y_max,h*10))
plt.show()

###案例4：利用贝叶斯分类实现手写数字识别
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

digits = load_digits()

fig = plt.figure()
for i in range(25):
    ax = fig.add_subplot(5,5,i+1)
    ax.imshow(digits.images[i],cmap=plt.cm.gray_r,interpolation='nearest')
plt.show()
#划分数据集
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(digits.data,digits.target,test_size=0.3,random_state=0)




