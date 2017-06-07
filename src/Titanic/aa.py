import csv
import random
from sklearn import cross_validation   #可视化学习的整个过程
from sklearn.tree import DecisionTreeClassifier
#读文件
def load_data(filename):
    reader = open(filename, 'rb')
    datafile = []
    for line in reader:
        datafile.append(line)
        #print line
    return datafile

#预处理
def data_clean(datafile):
    l = len(datafile)
    print(l)
    target = [0 for i in range(l)]
    datamat = [[0 for j in range(6)] for i in range(l)]
    for i in range(l):
        target[i] = datafile[i][0]
        datamat[i][0] = int(datafile[i][1]) #C
        if(datafile[i][3] == 'male'):
             datamat[i][1] = 1
        else:
            datamat[i][1] = 2#m


        t5 = int(datamat[i][4])
        if((0 < t5) and (t5 < 15)):
            datamat[i][2] = 0
        elif(15<=t5 and t5<40):
            datamat[i][2] = 1
        elif(40<=t5 and t5<100):
            datamat[i][2] = 2 #A


        if(datafile[i][5]>=2):
             datamat[i][3] = 2
        else:
             datamat[i][3] = datafile[i][5]#S


        if(datafile[i][6]>=2):
             datamat[i][4] = 2
        else:
             datamat[i][4] = datafile[i][6]#P


        if(datafile[i][10] == 'S'):
             datamat[i][5] = 0
        elif(datafile[i][10] == 'C'):
            datamat[i][5] = 1
        else:
            datamat[i][5] = 2#E

        #print target[i], datamat[i]
    return target, datamat


datafile = load_data('train.csv')
print(datafile[1])
print(datafile[1][1])
print(datafile[1][2])
train_target,train_data = data_clean(datafile)
print(train_target)
"""
#洗牌
r= random.randint(2147483647)
random.seed(r)
random.shuffle(train_data)
random.seed(r)
random.shuffle(train_target)
print('load finished')
"""
#'''
#交叉验证
X_train, X_test, y_train, y_test = cross_validation.train_test_split(train_data,train_target, test_size=0.5, random_state=0)
#clf = LogisticRegression()#LR分类 参数默认
#clf = RandomForestClassifier(n_estimators=8)
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_predicted = clf.predict(X_test).tolist()
print(y_test)
print(y_predicted)

#calculate_result(y_test,y_predicted)