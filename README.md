# Titanic---Machine-Learning-from-Disaster

## Kaggle Challenge

The sinking of the Titanic is one of the most infamous shipwrecks in history.

On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.

While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.

In this challenge, we ask you to build a predictive model that answers the question: “what sorts of people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic class, etc).

## Dataset
訓練集資料(891筆)、測試集資料(418筆)

The data has been split into two groups:

training set (train.csv)
test set (test.csv)
The training set should be used to build your machine learning models. For the training set, we provide the outcome (also known as the “ground truth”) for each passenger. Your model will be based on “features” like passengers’ gender and class. You can also use feature engineering to create new features.

The test set should be used to see how well your model performs on unseen data. For the test set, we do not provide the ground truth for each passenger. It is your job to predict these outcomes. For each passenger in the test set, use the model you trained to predict whether or not they survived the sinking of the Titanic.

We also include gender_submission.csv, a set of predictions that assume all and only female passengers survive, as an example of what a submission file should look like.

## Data Dictionary
表格為資料的各個特徵
![截圖 2022-05-18 14 38 06](https://user-images.githubusercontent.com/97944035/168973785-214ec2d3-188e-4cbb-9f01-f52300c86c7a.png)


## 查看資料並填補缺失值
```{}
train_data = pd.read_csv("~/Desktop/titanic/train.csv")
test_data = pd.read_csv("~/Desktop/titanic/test.csv")
df_data = train_data.append(test_data)
print(df_data[:len(train_data)])
print(df_data[len(train_data):])
df_data.set_index("PassengerId",inplace=True)
print(df_data.isnull().sum())
```
### train data

![截圖 2022-05-18 21 49 45](https://user-images.githubusercontent.com/97944035/169056342-bcbf6c2e-21db-493c-976d-0171da4b7716.png)

### test data

![截圖 2022-05-18 21 50 14](https://user-images.githubusercontent.com/97944035/169056452-09f0bee1-d894-40dc-95ab-9f2b21342aa0.png)

### 檢查缺失值

![截圖 2022-05-18 21 49 29](https://user-images.githubusercontent.com/97944035/169056797-553a556e-602b-4752-a126-0824b2252dbe.png)


之後將缺失值補齊後，對類別變數進行轉換。

新增家庭人數及票價分區間這兩項新變數，再將年齡及家庭人數各區分成兩個區間及三個圈間。

![截圖 2022-05-18 22 06 45](https://user-images.githubusercontent.com/97944035/169061139-225005f9-d854-4ce1-9fe4-76809d7bc5c0.png)

接下來進行各變數的視覺化。

可以發現到社會階級越高、性別（女性）、船票較高者及所處的甲板位置都有較高的存活率

利用隨機森林對選取的變數進行分析，對模型進行10次的交叉驗證

將交叉驗證的結果與訓練集、測試及和score做比較

```{}
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df_data[:len(train_data)][["Pclass","Sex","age_1","family_1","fare_bin6","Embarked",'Deck']], df_data[:len(train_data)]["Survived"], test_size=0.2, random_state=0)


from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

randomForestModel = RandomForestClassifier(n_estimators=250,random_state=1,max_depth=12,oob_score=True)
cv_rf = cross_val_score(estimator = randomForestModel, X = x_train, y = y_train, cv = 10)
randomForestModel.fit(x_train,y_train)
randomForestModel.predict(x_train)


df = pd.DataFrame({"method":["cv","train","test","oob"],
                   "score":[cv_rf.mean(),randomForestModel.score(x_train,y_train),randomForestModel.score(x_test,y_test),randomForestModel.oob_score_]})
print(df)
```

![截圖 2022-05-18 22 16 15](https://user-images.githubusercontent.com/97944035/169063240-22cd6346-9303-4191-8da3-14c1ad449420.png)


