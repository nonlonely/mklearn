# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
from pylab import *



data_train=pd.read_csv("../train.csv")
#初看数据
print(data_train)
#查看数据的统计信息
print(data_train.info())
#查看数据关于数值的统计信息
print(data_train.describe())


#通过上边的统计信息可以得到一些大概的知识，但是如果想要得到更加详细的知识，得需要更加统计意义的判断，因此这个时候就要画图来统计一下了
fig=plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数

#乘客各属性分布

#获救人数和未获救人数的分布
plt.subplot2grid((2,3),(0,0))
data_train.Survived.value_counts().plot(kind="bar")
#防止中文乱码
zhfont1 = matplotlib.font_manager.FontProperties(fname='C:\Windows\Fonts\simsun.ttc')
plt.title("获救情况(1为获救)",fontproperties=zhfont1)
plt.ylabel("人数",fontproperties=zhfont1)

#从等级看人口分布关系
plt.subplot2grid((2,3),(0,1))
data_train.Pclass.value_counts().plot(kind="bar")
plt.ylabel("人数",fontproperties=zhfont1)
plt.title("乘客等级分布",fontproperties=zhfont1)

#从年龄看人口分布
plt.subplot2grid((2,3),(0,2))
plt.scatter(data_train.Survived,data_train.Age)
plt.ylabel("年龄",fontproperties=zhfont1)
plt.grid(b=True,which='major',axis='y')
plt.title("按年龄看获救分布（1为年龄）",fontproperties=zhfont1)

#年龄与用户等级的年龄分布
plt.subplot2grid((2,3),(1,0),colspan=2)
data_train.Age[data_train.Pclass==1].plot(kind='kde')
data_train.Age[data_train.Pclass==2].plot(kind='kde')
data_train.Age[data_train.Pclass==3].plot(kind='kde')
plt.xlabel("年龄",fontproperties=zhfont1)
plt.ylabel("密度",fontproperties=zhfont1)
plt.title("各等级的乘客年龄分布",fontproperties=zhfont1)
plt.legend(("头等舱","2等舱","3等舱"),loc="best")

#各登船口岸上船人数
plt.subplot2grid((2,3),(1,2))
data_train.Embarked.value_counts().plot(kind='bar')
plt.title("各登船口岸上船人数",fontproperties=zhfont1)
plt.ylabel("人数",fontproperties=zhfont1)

#plt.show()

#属性与获救结果的关联统计

#看看各乘客等级的获救情况
fig=plt.figure()
fig.set(alpha=0.2)
Survived_0=data_train.Pclass[data_train.Survived==0].value_counts()
Survived_1=data_train.Pclass[data_train.Survived==1].value_counts()
df=pd.DataFrame({'获救':Survived_1,'未获救':Survived_0})
df.plot(kind='bar',stacked=True)
plt.title("各乘客等级的获救情况",fontproperties=zhfont1)
plt.xlabel("乘客等级",fontproperties=zhfont1)
plt.ylabel("人数",fontproperties=zhfont1)
#plt.show()

#看看各性别的获救情况
fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数

Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()
Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()
df=pd.DataFrame({u'男性':Survived_m, u'女性':Survived_f})
df.plot(kind='bar', stacked=True)
plt.title("按性别看获救情况",fontproperties=zhfont1)
plt.xlabel("性别",fontproperties=zhfont1)
plt.ylabel("人数",fontproperties=zhfont1)
#plt.show()

 #然后我们再来看看各种舱级别情况下各性别的获救情况
fig=plt.figure()
fig.set(alpha=0.65) # 设置图像透明度，无所谓
plt.title("根据舱等级和性别的获救情况",fontproperties=zhfont1)

ax1=fig.add_subplot(141)
data_train.Survived[data_train.Sex == 'female'][data_train.Pclass != 3].value_counts().plot(kind='bar', label="female highclass", color='#FA2479')
ax1.set_xticklabels(["获救", "未获救"], rotation=0,fontproperties=zhfont1)
ax1.legend(["女性/高级舱"], loc='best')

ax2=fig.add_subplot(142, sharey=ax1)
data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='female, low class', color='pink')
ax2.set_xticklabels(["未获救", "获救"], rotation=0,fontproperties=zhfont1)
plt.legend(["女性/低级舱"], loc='best')

ax3=fig.add_subplot(143, sharey=ax1)
data_train.Survived[data_train.Sex == 'male'][data_train.Pclass != 3].value_counts().plot(kind='bar', label='male, high class',color='lightblue')
ax3.set_xticklabels(["未获救", "获救"], rotation=0,fontproperties=zhfont1)
plt.legend(["男性/高级舱"], loc='best')

ax4=fig.add_subplot(144, sharey=ax1)
data_train.Survived[data_train.Sex == 'male'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='male low class', color='steelblue')
ax4.set_xticklabels(["未获救", "获救"], rotation=0,fontproperties=zhfont1)
plt.legend(["男性/低级舱"], loc='best')
plt.show()
"""

fig=plt.figure()
fig.set(alpha=0.2)

Survived_0=data_train.Embarked[data_train.Survived==0].value_counts()
Survived_1=data_train.Embarked[data_train.Survived==1].value_counts()
df=pd.DataFrame({'获救':Survived_1,'未获救':Survived_0})
df.plot(kind='bar',stacked=True)
zhfont1 = matplotlib.font_manager.FontProperties(fname='C:\Windows\Fonts\simsun.ttc')
plt.title("各登录港口乘客的获救情况",fontproperties=zhfont1)
plt.xlabel("登录港口",fontproperties=zhfont1)
plt.ylabel("人数",fontproperties=zhfont1)

#plt.show()

#下面我们来看看 堂兄弟/妹，孩子/父母有几人，对是否获救的影响。
g=data_train.groupby(['SibSp','Survived'])
df=pd.DataFrame(g.count()['PassengerId'])
print(df)
p=data_train.groupby(['Parch','Survived'])
dfp=pd.DataFrame(p.count()['PassengerId'])
print(dfp)

#ticket是船票编号，应该是unique的，和最后的结果没有太大的关系，先不纳入考虑的特征范畴把
#cabin只有204个乘客有值，我们先看看它的一个分布
print(data_train.Cabin.value_counts())

fig=plt.figure()
fig.set(alpha=0.2)

Survived_cabin=data_train.Survived[pd.notnull(data_train.Cabin)].value_counts()
Survived_nocabin=data_train.Survived[pd.isnull(data_train.Cabin)].value_counts()
df=pd.DataFrame({'有':Survived_cabin,'无':Survived_nocabin}).transpose()
df.plot(kind='bar',stacked=True)
plt.title("按Cabin有无看获救情况",fontproperties=zhfont1)
plt.xlabel("Cabin有无",fontproperties=zhfont1)
plt.ylabel("人数",fontproperties=zhfont1)
plt.show()


"""







































