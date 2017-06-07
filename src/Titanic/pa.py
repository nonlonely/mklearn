import pandas as pd
data_fram=pd.read_csv("train.csv")

#经过上边的过程已经得到了数组，因此下一步是出路数据，
#先处理性别，如果是male，则处理为0，如果是female，则处理为1
data_fram.loc[data_fram["Sex"]=="male","Sex"]=0
data_fram.loc[data_fram["Sex"]=="female","Sex"]=1

data_fram.loc[data_fram["Embarked"]=="S","Embarked"]=0
data_fram.loc[data_fram["Embarked"]=="C","Embarked"]=1
data_fram.loc[data_fram["Embarked"]=="Q","Embarked"]=2

#删除名字这一列
data_fram=data_fram.drop('Name', 1)

#处理年龄
#data_fram.loc[data_fram["Age"]==,"Age"]=30
#data_fram.loc[float(data_fram["Age"])<16 and float(data_fram["Age"])>0,"Age"]=0
#data_fram.loc[float(data_fram["Age"])>=16 and float(float(data_fram["Age"])<40),"Age"]=0
#data_fram.loc[float(data_fram["Age"])>=40,"Age"]=0
data_fram.Age[data_fram.Age>=40]=3
print(data_fram.head())