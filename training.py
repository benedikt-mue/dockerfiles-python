from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np 
import pickle

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-m", "--model")

args = parser.parse_args()

train_df = pd.read_csv("./data/train.csv")
test_df = pd.read_csv("./data/test.csv")

relevant_features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

# Handle missing values
imputer = SimpleImputer(strategy='most_frequent')
train_df[relevant_features] = imputer.fit_transform(train_df[relevant_features])
test_df[relevant_features] = imputer.transform(test_df[relevant_features])

# Encode categorical variables as numeric
train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})
test_df['Sex'] = test_df['Sex'].map({'male': 0, 'female': 1})
train_df['Embarked'] = train_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
test_df['Embarked'] = test_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Transform skewed or non-normal features
# Instead of normalizing all of the numeric features, you could try using techniques like log transformation or Box-Cox transformation to make the distribution of a feature more normal
scaler = StandardScaler()
train_df[relevant_features] = scaler.fit_transform(train_df[relevant_features])
test_df[relevant_features] = scaler.transform(test_df[relevant_features])

# Split the data into features (X) and labels (y)
X_train = train_df[relevant_features]
y_train = train_df['Survived']
X_test = test_df[relevant_features]

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=33)


models = {
    'LogisiticRegression': LogisticRegression(),
    'KNeighborsClassifier': KNeighborsClassifier(),
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'RandomForestClassifier': RandomForestClassifier(),
    'SVC': SVC(gamma = 'auto'),
    'GaussianNB': GaussianNB(),
    'XGBClassifier': XGBClassifier()
}

model = models.get(args.model)

model.fit(X_train.values, y_train)

y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print("Accuracy: ", accuracy)


filename = f'models/{args.model}.pkl'
pickle.dump(model, open(filename, 'wb'))
