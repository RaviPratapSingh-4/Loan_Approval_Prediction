import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(r"loan_data_after_feature_enginnering.csv")

print(df)

df.info()

columns_to_extract = [
    "ApplicantIncome", 
    "CoapplicantIncome", 
    "LoanAmount", 
    "Loan_Amount_Term", 
    "Credit_History", 
    "Total_Income", 
    "LTI"
]

x = df[columns_to_extract]

print(x)

y = df["Loan_Status"]

x_train, x_test, y_train, y_test = train_test_split(x, y , test_size = 0.30, random_state = 4)

print(x_train)

print(x_test)

var = StandardScaler()
x_train = var.fit_transform(x_train)
x_test = var.transform(x_test)

print(x_train)

print(x_test)

logReg = LogisticRegression()
logReg.fit(x_train, y_train)
y_pred = logReg.predict(x_test)

print(y_pred)

print(accuracy_score(y_pred, y_test))

print(confusion_matrix(y_pred, y_test))

