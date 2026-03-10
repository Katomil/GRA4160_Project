import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA



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

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

importance = model.coef_[0]
feature_names = X.columns

indices = np.argsort(np.abs(importance))[-10:]

plt.barh(range(len(indices)), importance[indices])
plt.yticks(range(len(indices)), feature_names[indices])
plt.title("Top 10 Important Features")
plt.show()

sns.countplot(x="odor", hue="class", data=df)
plt.title("Distribution of Odor by Mushroom Class")
plt.show()

X_encoded = pd.get_dummies(X)

plt.figure(figsize=(12,10))
sns.heatmap(X_encoded.corr(), cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

pca = PCA(n_components=2)
X_pca = pca.fit_transform(pd.get_dummies(X))

plt.scatter(X_pca[:,0], X_pca[:,1], c=(y=="p"), cmap="coolwarm")
plt.title("PCA Visualization of Mushrooms")
plt.show()
