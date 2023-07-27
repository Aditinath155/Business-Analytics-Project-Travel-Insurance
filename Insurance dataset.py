#!/usr/bin/env python
# coding: utf-8

# # Business Problem:- A tour & travels company is offering travel insurance package to their customers.The company wants to know which customers would be interested to buy it based on their database history.
# Maximize - Sales of insurance package

# In[33]:


#load important libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as pltx


# In[34]:


#load the file
data = pd.read_csv(r"C:\Users\Aditi nath\Downloads\EDA\TravelInsurancePrediction.csv")
data


# In[36]:


#EDA
data.info()


# #Dataset Description-
# TravelInsurance:-Our target variable/dependent is Travell insurance whether they take insurance or not.
# 
# Independent Variables
# 
# Age[Int] :- Age of the customer
# Employment Type[String] :- The sector in which customer is employed (Goverment,Private/Self Employeed).
# GraduateOrNot[String] :- Whether the customer is college graduate or not
# AnnualIncome[Int] :- The yearly income of the customer.
# FamilyMembers[Int] :- Number of members in customer’s family
# ChronicDiseases[Int] :- Whether the customer suffers from any major disease or conditions like Diabetes/high BP or Asthma, etc
# FrequentFlyer[String] :- Derived data based on customer’s history of booking air tickets 
# EverTravelledAbroad[String] :- Has the customer ever travelled to a foreign country
# 
# 

# In[37]:


#Checking missing value
data.isnull().sum() 
#no missing value


# In[38]:


#Lets do summary statistic
data.describe()
#no outliers in the data


# In[39]:


#No need to to check duplicate value in this dataset because age,annal income can be same


# In[40]:


#Check the unique values
for i in data.columns:
    print(i,'Unique values: ',data[i].unique())


# In[41]:


data.corr()


# In[42]:


#Data visualization
#Checking relationship between Age and travel insurance
sns.boxplot(x='TravelInsurance',y='Age',data=data)


# #Observation
# #The majority of individuals who purchased travel insurance are above 30 age .

# In[44]:


print(f"minimum annual income = {data['AnnualIncome'].min()}")
print(f"maximum annual income = {data['AnnualIncome'].max()}")


# In[45]:


#Relationship between travel insurance and age
sns.boxplot(x='TravelInsurance',y='AnnualIncome',data=data)

#The majority of individuals who purchased travel insurance fall within the 80000 to 140000 annual income group.


# In[46]:


#Relationship between Familymember and travel insurance
sns.boxplot(x='TravelInsurance',y='FamilyMembers',data=data)

#The ratio of Family member in respect to buying the insurance stays the same throughout the whole graph.


# In[47]:


employment_counts = data['Employment Type'].value_counts()
fig = plt.figure(figsize =(5, 7))
plt.pie(employment_counts, labels=employment_counts.index, autopct='%1.1f%%')
plt.show()


# In[48]:


Frequent_Flyer_counts = data['FrequentFlyer'].value_counts()

# Create a pie chart
fig = plt.figure(figsize=(5,7))
plt.pie(Frequent_Flyer_counts, labels=Frequent_Flyer_counts.index, autopct='%1.1f%%')
plt.axis('equal')  # Equal aspect ratio ensures a circular pie
plt.title('Frequent Flyer Distribution')
plt.show()


# In[49]:


#Making the dataset all numerical with map function
#Frequent flyer And ever travlelled abroad
# Yes : 1 , No : 0
data['FrequentFlyer'] = data['FrequentFlyer'].map({'Yes':1,'No':0})
data['EverTravelledAbroad']= data['EverTravelledAbroad'].map({'Yes':1,'No':0})

# Government Sector : 1, Private Sector/Self Employed : 0
data['Employment Type'] = data['Employment Type'].map({"Government Sector" : 1, "Private Sector/Self Employed" : 0})


# In[50]:


#the heat map of the correlation
plt.figure(figsize=(20,7))
sns.heatmap(data.corr(), annot=True, cmap='RdYlGn')


# In[51]:


#Observation:-
#Annual income and Ever trvelled abroad have positive correlation
#People who travelled abroad take insurance
#Another interesting finding, it looks like people who did travel abroad are more likely to be frequent flyers.
# because the travel to foreign countries is more expensive and, as we know already, the income plays a huge role in our problem. 


# In[52]:


#Rich people and frequent flyer relationship
pltx.histogram(data, x='AnnualIncome', color='FrequentFlyer', color_discrete_map={1:'#acc8fc', 0:'#6f6cd4'}, title='Rich people and frequent flyer relationship.')
#Yes-1 and No -0


# In[53]:


#Observation
#People with more income tend to travel more frequently and so they are also more likely to buy an insurance.


# In[54]:


#Annual income and employemnt type.
#Goverment employee- 1 and private/self employee = 0
pltx.histogram(data, x='AnnualIncome', color='Employment Type', color_discrete_map={1:'#acc8fc', 0:'#6f6cd4'}, title='Annual income and employemnt type.')


# In[55]:


#Observation
#Goverment employee are less paid than private employees so these can be reason behind goverment employee are not taking insurance as they less travel


# In[56]:


#How many people are in govt or private sector and are frequent flyer or not?
data_emp_fly = pd.crosstab(data['Employment Type'],data['FrequentFlyer'])
data_emp_fly


# In[57]:


#How many people are in govt or private sector and are purchase travel insurance or not?
data_insurance = pd.crosstab(data['Employment Type'],data['TravelInsurance'])
data_insurance


# In[58]:


data_emp_fly = pd.crosstab(data['FrequentFlyer'],data['TravelInsurance'])
data_emp_fly


# In[59]:


#Conclusion
#The conclusion here is easy to make, the interest around the insurance comes down to the budget.
#We don't know if the company reaches its target audience or not, it could be that the service is targeted to the mid-upper income group of people.

#Suggestion
#Awarness about the insurance through advertisment
#Discounts and Incentives
#Informative brochures that outline the benefits of travel insurance.
#Make sure to include details about coverage for trip cancellations, medical emergencies, lost baggage, and other relevant situations.

