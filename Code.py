#!/usr/bin/env python
# coding: utf-8

# In[155]:


import numpy as np 
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
from patsy import dmatrices
import sklearn
import seaborn as sns




# In[156]:


dataframe=pd.read_csv("Data.csv")


# In[157]:


dataframe.head()


# In[208]:


dataframe.loc[:, 'NumCompaniesWorked'].mean()


# In[158]:


names = dataframe.columns.values 
#print(names)


# In[164]:


dataframe.describe()


# In[160]:


# explore data for Attrition by Age
plt.figure(figsize=(14,10))
plt.scatter(dataframe.Attrition,dataframe.Age, alpha=.55)
plt.title("Attrition by Age ")
plt.ylabel("Age")
plt.grid(which='major', axis='y')
#Change By RajGPT
#plt.show()


# In[161]:


# explore data for Left employees breakdown
plt.figure(figsize=(8,6))
dataframe.Attrition.value_counts().plot(kind='barh',color='blue',alpha=.65)
plt.title("Attrition breakdown ")
#plt.show()


# In[162]:


# explore data for Education Field distribution
plt.figure(figsize=(10,8))
dataframe.EducationField.value_counts().plot(kind='barh',color='g',alpha=.65)
plt.title("Education Field Distribution")
#plt.show()


# In[163]:


# explore data for Marital Status
plt.figure(figsize=(8,6))
dataframe.MaritalStatus.value_counts().plot(kind='bar',alpha=.5)
#plt.show()


# In[186]:


fig_dims = (12, 4)
fig, ax = plt.subplots(figsize=fig_dims)

#ax = axis
sns.countplot(x='Age', hue='Attrition', data = dataframe, palette="colorblind", ax = ax,  edgecolor=sns.color_palette("dark", n_colors = 1));


# In[165]:


dataframe.info()


# In[166]:


dataframe.columns


# In[167]:


# dataframe.std()
numeric_column = pd.to_numeric(dataframe['Attrition'], errors='coerce') #added by Raj
std_value = numeric_column.std()  #added by Raj


# In[168]:


dataframe['Attrition'].value_counts()


# In[169]:


dataframe['Attrition'].dtypes


# In[170]:


# Check the unique values in the 'Attrition' column
unique_values = dataframe['Attrition'].unique()

# Make sure the column contains only 'Yes' and 'No'
if set(unique_values) == {'Yes', 'No'}:
    # Replace 'Yes' with 1 and 'No' with 0
    dataframe['Attrition'].replace({'Yes': 1, 'No': 0}, inplace=True)
    dataframe['Attrition'].replace('No', 0, inplace=True)   # Replace 'No' with 0
else:
    print("The 'Attrition' column contains unexpected values.")

# Assuming your DataFrame is named 'dataframe'
# dataframe['Attrition'].replace('Yes', 1, inplace=True)  # Replace 'Yes' with 1
# dataframe['Attrition'].replace('No', 0, inplace=True)   # Replace 'No' with 0



# In[171]:


dataframe.head(10)


# In[172]:


# building up a logistic regression model
X = dataframe.drop(['Attrition'],axis=1)
X.head()
Y = dataframe['Attrition']
Y.head()


# In[173]:


dataframe['EducationField'].replace('Life Sciences',1, inplace=True)
dataframe['EducationField'].replace('Medical',2, inplace=True)
dataframe['EducationField'].replace('Marketing', 3, inplace=True)
dataframe['EducationField'].replace('Other',4, inplace=True)
dataframe['EducationField'].replace('Technical Degree',5, inplace=True)
dataframe['EducationField'].replace('Human Resources', 6, inplace=True)


# In[174]:


dataframe['EducationField'].value_counts()


# In[175]:


dataframe['Department'].value_counts()


# In[176]:


dataframe['Department'].replace('Research & Development',1, inplace=True)
dataframe['Department'].replace('Sales',2, inplace=True)
dataframe['Department'].replace('Human Resources', 3, inplace=True)


# In[177]:


dataframe['Department'].value_counts()


# In[178]:


dataframe['MaritalStatus'].value_counts()


# In[179]:


dataframe['MaritalStatus'].replace('Married',1, inplace=True)
dataframe['MaritalStatus'].replace('Single',2, inplace=True)
dataframe['MaritalStatus'].replace('Divorced',3, inplace=True)


# In[180]:


dataframe['MaritalStatus'].value_counts()


# In[181]:


#x.columns


# In[182]:


y=dataframe['Attrition']


# In[183]:


y.head()


# In[184]:


y, x = dmatrices('Attrition ~ Age + Department +                   DistanceFromHome + Education + EducationField + YearsAtCompany',
                  dataframe, return_type="dataframe")
#print (x.columns)


# In[185]:


dataframe.isnull().values.any()


# In[187]:


#for column in dataframe.columns:
   # if dataframe[column].dtype == object:
        #print(str(column) + ' : ' + str(dataframe[column].unique()))
        #print(dataframe[column].value_counts())
        #print("_________________________________________________________________")


# In[188]:


#Get the correlation of the columns
dataframe.corr()


# In[189]:


plt.figure(figsize=(14,14))  #14in by 14in
sns.heatmap(dataframe.corr(), annot=True, fmt='.0%')


# In[190]:


from sklearn.preprocessing import LabelEncoder

for column in dataframe.columns:
        if dataframe[column].dtype == np.number:
            continue
        dataframe[column] = LabelEncoder().fit_transform(dataframe[column])


# In[191]:


dataframe['Age_Years'] = dataframe['Age']
#Remove the first column called age 
dataframe = dataframe.drop('Age', axis = 1)
#Show the dataframe
dataframe


# In[192]:


X = dataframe.iloc[:, 1:dataframe.shape[1]].values 
Y = dataframe.iloc[:, 0].values


# In[193]:


y = np.ravel(y)


# In[194]:


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model = model.fit(x, y)

# check the accuracy on the training set
model.score(x, y)


# In[195]:


y.mean()


# In[196]:


X_train,X_test,y_train,y_test=sklearn.model_selection.train_test_split(x,y, test_size=0.3, random_state=0)
model2=LogisticRegression()
model2.fit(X_train, y_train)


# In[197]:


predicted= model2.predict(X_test)
#print (predicted)


# In[198]:


probs = model2.predict_proba(X_test)
#print (probs)


# In[199]:


from sklearn import metrics

#print (metrics.accuracy_score(y_test, predicted))
#print (metrics.roc_auc_score(y_test, probs[:, 1]))


# In[200]:


#print (metrics.confusion_matrix(y_test, predicted))
#print (metrics.classification_report(y_test, predicted))


# In[201]:


#print (X_train)


# In[202]:


#print(model.predict_proba(X_train))


# In[203]:

def calc(age,dept,dist,education,edufield,years):
    kk=[[1.0, age, dept, dist, education, edufield, years]]
     
    return model.predict_proba(kk)

#add random values to KK according to the parameters mentioned above to check the proabily of attrition of the employee
kk=[[1.0, 25.0, 1.0, 500.0, 3.0, 24.0, 1.0]]

print(model.predict_proba(kk))


# Integration  of LLM

#pip install replicate

import os

os.environ["REPLICATE_API_TOKEN"] = "r8_V26YzDNdYXYAriSA5FlfSd4SeTqysag1rZnKp"

import replicate

# Prompts
pre_prompt = "Compliance Monitoring and Enforcement through Log Analysis using Large Language Models"
prompt_input = "check for compliance with security policies and standards!"

# Generate LLM response
output = replicate.run('a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5', # LLM model
                        input={"prompt": f"{pre_prompt} {prompt_input} Assistant: ", # Prompts
                        "temperature":0.1, "top_p":0.9, "max_length":128, "repetition_penalty":1})  # Model parameters

full_response = ""

for item in output:
    full_response += item

print(full_response)

