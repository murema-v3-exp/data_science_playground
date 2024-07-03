import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
import sklearn.linear_model as lm
import sklearn.metrics  as metrics


# Load df from a csv file
df = pd.read_csv("Titanic-dataset.csv")

# Replacing null values in Age with the median age
mid_age = df["Age"].median()
df["Age"].fillna(mid_age, inplace=True)

#Replacing null values in embarked with the most common place
md_embarked = df["Embarked"].mode()
df["Embarked"].fillna(md_embarked, inplace=True)

# Change categorical variables into factors
df["Sex"] = df["Sex"].map({"male":0,"female":1})
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

# Dropping irrelevant columns
df.drop(columns=["Name","Cabin","Ticket","PassengerId"], inplace=True)

# Defining the variables of the model
explanatory_vars = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked_Q',"Embarked_S"]
response_var = "Survived"

print(df.info)
print(df["Survived"].value_counts())


# Splitting the data into training and testing sets 
x_train,x_test,y_train,y_test = train_test_split(df[explanatory_vars], df[response_var], test_size=0.2, random_state=42)

# Initializing an object for the model
model = lm.LogisticRegression(max_iter=200)

# fitting a training model
model.fit(x_train, y_train)

# male predictions on test set 
preds = model.predict(x_test)

# Check model performance
accuracy = metrics.accuracy_score(y_test, preds)
conf_matrix = metrics.confusion_matrix(y_test, preds)
class_report = metrics.classification_report(y_test, preds)

print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)

sns.heatmap(conf_matrix, annot=True, cmap='magma', fmt='g', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Example input data for prediction
new_data = pd.DataFrame({
    'Pclass': [3],
    'Sex': [0],  
    'Age': [25],
    'SibSp': [1],
    'Parch': [0],
    'Fare': [7.25],
    'Embarked_Q': [0],
    'Embarked_S': [1]
})

prediction = model.predict_proba(new_data)
print("Predicted Probability:", prediction[:,1])
