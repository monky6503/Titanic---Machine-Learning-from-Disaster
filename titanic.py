import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# read data and check data
train_data = pd.read_csv("~/Desktop/titanic/train.csv")
test_data = pd.read_csv("~/Desktop/titanic/test.csv")
print(train_data)
print(test_data)
df_data = train_data.append(test_data)
print(df_data[:len(train_data)])
print(df_data[len(train_data):])
df_data.set_index("PassengerId",inplace=True)

df_data["Sex"] = df_data["Sex"].map({"female": 0, "male": 1})
df_data["Age"] = df_data["Age"].fillna(df_data["Age"].median())
df_data["Fare"] = df_data["Fare"].fillna(df_data["Fare"].median())
df_data["family_size"] = df_data["SibSp"] + df_data["Parch"] + 1
df_data["fare_bin6"] = pd.qcut(df_data["Fare"], 6,labels=[0,1,2,3,4,5])
df_data["Embarked"] = df_data["Embarked"].map({"S":0,"C":1,"Q":2})
df_data["Embarked"] = df_data["Embarked"].fillna(0)

def family(x):
    if x<=3:
        return 1
    elif x>=4 and x<=6:
        return 2
    else:
        return 3
df_data["family_1"] = df_data["family_size"].map(lambda x: family(x))

def age(x):
    if x <= 15:
        return 1
    else:
        return 0

df_data["age_1"] = df_data["Age"].map(lambda x: age(x))
cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']

def transtr(str1, str2):
    for i in str2:
        if str.find(str(str1), i) != -1:
            return str(i)
    return np.nan

df_data['Deck'] = df_data['Cabin'].map(lambda x: transtr(x, cabin_list))
df_data['Deck'] = df_data['Deck'].map({"A":0,"B":1,"C":2,"D":3,"E":4,"F":5,"T":6,"G":7,"NaN":8})
df_data['Deck'] = df_data['Deck'].fillna(8)
print(df_data.head())


import seaborn as sns


fig, axes = plt.subplots(2,3)
p1 = sns.countplot(ax=axes[0,0],data=df_data[:len(train_data)],x="Sex")
p1.set_xticklabels(["Female","Male"])
p2 = sns.countplot(ax=axes[0,1],data=df_data[:len(train_data)],x="Pclass")
p2.set_xticklabels(["1(Upper)","2(Middle)","3(Lower)"])
p3 = sns.histplot(ax=axes[0,2],data=df_data[:len(train_data)],x="Fare")
p4 = sns.countplot(ax=axes[1,0],data=df_data[:len(train_data)], x="Embarked")
p4.set_xticklabels(["S","C","Q"])
p5 = sns.countplot(ax=axes[1,1],data=df_data[:len(train_data)], x="family_1")
p5.set_xticklabels(["1~3","4~6","7 up"])
p5.set_xlabel("Family_number")
p6 = sns.countplot(ax=axes[1,2],data=df_data[:len(train_data)], x="Deck")
plt.show()

fig, axes = plt.subplots(1,3)
p7 = sns.countplot(ax=axes[0],data=df_data[:len(train_data)], x="Pclass", hue="Survived")
p7.set_xticklabels(["1(Upper)","2(Middle)","3(Lower)"])
p8 = sns.countplot(ax=axes[1],data=df_data[:len(train_data)], x="Sex", hue="Survived")
p8.set_xticklabels(["Female","Male"])
p9 = sns.countplot(ax=axes[2],data=df_data[:len(train_data)], x="fare_bin6", hue="Survived")
p9.set_xlabel("Fare_interval")
plt.show()

p10 = sns.boxplot(data=df_data[:len(train_data)],x="Pclass",y="Fare",hue="Survived",
                 flierprops={'marker':'o','markerfacecolor':'red','color':'black'})
p10.set_xticklabels(["1(Upper)","2(Middle)","3(Lower)"])
plt.show()

p11 = sns.countplot(data=df_data[:len(train_data)], x="Deck", hue="Pclass")
p11.set_xticklabels(['A', 'B', 'C', 'D', 'E', 'F', 'T',"G","Unknow"])
p11.legend(loc="upper left",title="Pclass")
plt.show()

p12 = sns.countplot(data=df_data[:len(train_data)], x="family_1", hue="Survived")
p12.set_xticklabels(["1","2","3"])
plt.show()

fig, axes = plt.subplots(1,2)
p13 = sns.countplot(ax=axes[0],data=df_data[:len(train_data)], x="Embarked", hue="Survived")
p13.set_xticklabels(["S","C","Q"])
p14 = sns.countplot(ax=axes[1],data=df_data[:len(train_data)], x="Embarked", hue="Pclass")
p14.set_xticklabels(["S","C","Q"])
plt.show()


print("*"*50)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df_data[:len(train_data)][["Pclass","Sex","age_1","family_1","fare_bin6","Embarked"]], df_data[:len(train_data)]["Survived"], test_size=0.2, random_state=0)


from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestClassifier


randomForestModel = RandomForestClassifier(n_estimators=250,random_state=0,oob_score=True)
cv_rf = cross_val_score(estimator = randomForestModel, X = x_train, y = y_train, cv = 10)
randomForestModel.fit(x_train,y_train)
randomForestModel.predict(x_train)

df = pd.DataFrame({"method":["cv","train","test","oob"],
                   "score":[cv_rf.mean(),randomForestModel.score(x_train,y_train),randomForestModel.score(x_test,y_test),randomForestModel.oob_score_]})
print(df)


pred = randomForestModel.predict(df_data[len(train_data):][["Pclass","Sex","age_1","family_1","fare_bin6","Embarked"]])


# with open("survived_predict.csv","w") as f:
#     f.write("PassengerId,Survived\n")
#     for i in range(len(pred)):
#         f.write(str(i+892)+","+str(int(pred[i]))+"\n")



