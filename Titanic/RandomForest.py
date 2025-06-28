import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
# pd.set_option('display.max_rows', 20)
np.set_printoptions(threshold = np.inf)

# データロード
train_data = pd.read_csv('./Titanic/data/train.csv')
train_data = train_data.drop('Name', axis = 1) # 列方向
X_train = train_data.iloc[:, 2:].values
y_train = train_data.iloc[:, 1].values

test_data = pd.read_csv('./Titanic/data/test.csv')
test_data = test_data.drop('Name', axis = 1)
X_test = test_data.iloc[:, 1:].values

# --- 前処理 ---
# 欠損値処理
imputer_age = SimpleImputer(missing_values = np.nan, strategy = 'mean')
X_train[:, 2:3] = imputer_age.fit_transform(X_train[:, 2:3])
X_test[:, 2:3] = imputer_age.transform(X_test[:, 2:3])

imputer_fare = SimpleImputer(missing_values = np.nan, strategy = 'mean')
X_train[:, -3:-2] = imputer_fare.fit_transform(X_train[:, -3:-2])
X_test[:, -3:-2] = imputer_fare.transform(X_test[:, -3:-2])

# imputer = SimpleImputer(strategy = 'most_frequent')
imputer = SimpleImputer(strategy = 'constant', fill_value = 'unknown')
X_train[:, -2:-1] = imputer.fit_transform(X_train[:, -2:-1])
X_test[:, -2:-1] = imputer.transform(X_test[:, -2:-1])

# OneHotEncoding (Sex, Embarked)
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [1, 8])], remainder = 'passthrough')
X_train = ct.fit_transform(X_train)
X_test = ct.transform(X_test)

# LabelEncodeing (Ticket, Cabin)
le = LabelEncoder()
X_train[:, -1] = le.fit_transform(X_train[:, -1])
X_test[:, -1] = le.fit_transform(X_test[:, -1])
X_train[:, -3] = le.fit_transform(X_train[:, -3])
X_test[:, -3] = le.fit_transform(X_test[:, -3])

# --- フィーチャースケーリング ---
sc = StandardScaler()
X_train[:, 5:] = sc.fit_transform(X_train[:, 5:])
X_test[:, 5:] = sc.transform(X_test[:, 5:])


# --- モデルの訓練 ---
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
X_test_passengerid = test_data.iloc[:, 0].values

survived_passenger = pd.DataFrame({
    'PassengerId': X_test_passengerid,
    'Survived': y_pred
})
survived_passenger.to_csv('./Titanic/data/submittion_randomforest.csv', index = False, encoding = 'utf-8')