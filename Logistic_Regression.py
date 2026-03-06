import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


# load dataset
df = pd.read_csv("mushroom.csv")

print(df.head())
print(df.shape)

print(df["class"].value_counts())

# features and target
X = df.drop("class", axis=1)
y = df["class"]

# encode categorical features
X = pd.get_dummies(X)

print(X.head())
print(X.shape)

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(X_train.shape)
print(X_test.shape)

# train model
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

print(model.coef_)
print(model)

# prediction
y_pred = model.predict(X_test)

# evaluation
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:\n", cm)
