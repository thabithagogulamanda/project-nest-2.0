import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pickle


df = pd.read_csv('Training Dataset.csv')


df['ApplicantIncome'] = pd.to_numeric(df['ApplicantIncome'], errors='coerce')
df['LoanAmount'] = pd.to_numeric(df['LoanAmount'], errors='coerce')



df = df.dropna(subset=['LoanAmount', 'ApplicantIncome'])


X = df[['ApplicantIncome','LoanAmount']]
y = df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LogisticRegression()
model.fit(X_train, y_train)

y_pred=model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"accuracy:{accuracy}")


with open('logistic_regression_model.pkl', 'wb') as f:
    pickle.dump(model, f)

LoanAmount=float(input("enter loanamount:"))
ApplicantIncome=float(input("enter ApplicantIncome:"))
user_input=np.array([[LoanAmount,ApplicantIncome]])

predicted_loanapproval = model.predict(user_input)
print("predicted_loanapproval :",predicted_loanapproval[0])
