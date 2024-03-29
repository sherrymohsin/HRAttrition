import streamlit as st
import pandas as pd
import time
from IPython.display import display
from sklearn import tree
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
#import joblib
from sklearn.tree import DecisionTreeClassifier

df_HR = pd.read_csv('HR_Attrition.csv')
    
    #Drop unnecessary columns:
df_HR = df_HR.drop(['Attrition Date'], axis=1)
df_HR = df_HR.drop(['EmployeeNumber'], axis=1)
df_HR = df_HR.drop(['Random Number'], axis=1)
df_HR = df_HR.drop(['EmployeeCount'], axis=1)


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Create a label encoder object
le = LabelEncoder()

le_count = 0
for col in df_HR.columns[1:]:
    if df_HR[col].dtype == 'object':
        if len(list(df_HR[col].unique())) <= 2:
            le.fit(df_HR[col])
            df_HR[col] = le.transform(df_HR[col])
            le_count += 1

# convert rest of categorical variable into dummy
df_HR = pd.get_dummies(df_HR, drop_first=True)

X=df_HR.drop(["Attrition"],axis=1)
y=df_HR["Attrition"].values

#Oversampling
from collections import Counter

from imblearn.over_sampling import RandomOverSampler

rus= RandomOverSampler(random_state = 42)
X_over, y_over = rus.fit_resample(X, y)

#Split the data into training and testing set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_over, y_over, test_size=0.2, random_state=42 ,stratify=y_over)

#Applying DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(criterion='entropy',max_depth= 3,random_state=42)
clf.fit(X_train,y_train)

#y_pred= clf.predict(X_test)


#Define UI
left_co, cent_co,last_co = st.columns(3)
with cent_co:
    st.image('Emp_Attr.jpg', width = 300)





st.title('HR Attrition App \n\n')
st.subheader('Input values.') 

cols=st.columns(3)
with cols[0]:
    input_Overtime=st.selectbox("Overtime",('Yes','No'))
with cols[1]:
    input_YearsWithCurrManager=st.number_input("Years with current manager", step=1, min_value=0, max_value=17)
with cols[2]:   
    input_Age=st.number_input("Age", step=1, min_value=18, max_value=60)
    

cols=st.columns(2)
with cols[0]:
    input_MonthlyIncome=st.number_input("Monthly income", step=1)
with cols[1]:   
    input_Department=st.selectbox("Department",('Sales','Research & Development','Human Resources'))

Department = 0    
if input_Department == 'Sales':
    Department = 1

Overtime = 0
if input_Overtime == 'Yes':
    Overtime = 1


df_sample = X_test[:1]
df_sample['OverTime'] = Overtime
df_sample['Age'] = input_Age
df_sample['MonthlyIncome'] = input_MonthlyIncome
df_sample['YearsWithCurrManager'] = input_YearsWithCurrManager
df_sample['Department_Sales'] = Department

def plot_decision_tree(model, feature_names, class_names):
    # plot_tree function contains a list of all nodes and leaves of the Decision tree
    tree = plot_tree(model, feature_names = feature_names, class_names = class_names,
                     rounded = True, proportion = True, precision = 2, filled = False, fontsize=10)
    
    # I return the tree for the next part
    return tree
    
def plot_decision_path_tree(model, X, class_names=None):
    fig = plt.figure(figsize=(10, 12))
    class_names = model.classes_.astype(str) if type(class_names) == type(None) else class_names
    feature_names = X.index if type(X) == type(pd.Series()) else X.columns
    
    # Getting the tree from the function programmed above
    tree = plot_decision_tree(model, feature_names, class_names)
    
    # Get the decision path of the wanted prediction 
    decision_path = model.decision_path(X)

    # Now remember the tree object contains all nodes and leaves so the logic here
    # is to loop into the tree and change visible attribute for components that 
    # are not in the decision path    
    
               
    for i in range(0,len(tree)):
        if i not in decision_path.indices:
            plt.setp(tree[i], color = 'grey')
        else:
            plt.setp(tree[i], color = 'green')    
    
    st.pyplot(plt)
    


def predict_churn(user_input, model, ):
    y_pred= model.predict(user_input)
    
    churn='Churn not expected'    
    if y_pred[:1] == 1:
        churn='Churn expected'
        
    st.write(churn)    
    
    #draw_tree
    display(user_input)
    plot_decision_path_tree(model, user_input, user_input.columns)   
    #['OverTime','Age','MonthlyIncome','YearsWithCurrManager','Department_Sales']
    
if st.button("Predict churn"):
    predict_churn(df_sample, clf)
    
