#!/usr/bin/env python
# coding: utf-8

# ## Problem Understanding:
# 
# This challenge proposes a problem where a prediction is required to identify a shoper's likelihood to return to the stores considering his past shopping behaviour which would help the company to target specific customers by providing promotional offers,discounts etc which would in turn increase their revenue.
# 
# Dataset contains customer id's with their different visits to the mall on different dates. Units purchased and total spend details are provided for each visit.

# In[ ]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import os


# In[ ]:


# Reading data from AWS S3 bucket to Jupyter Notebook

import boto3
import csv

# get a handle on s3
session = boto3.Session(
                    aws_access_key_id='xxxxxxxxxxxxxxxxxx',
                    aws_secret_access_key='xxxxxxxxxxxxxxxxxxx',
                    region_name='xxxxxxxxxxxxx')
                    
s3 = session.resource('s3')

# get a handle on the bucket that holds your file
bucket = s3.Bucket('bucket_name') # example: energy_market_procesing

# get a handle on the object you want (i.e. your file)
obj = bucket.Object(key='dataset.csv') # example: market/zone1/data.csv

# get the object
response = obj.get()

# read the contents of the file
lines = response['Body'].read()

# saving the file data in a new file test.csv
with open('data.csv', 'wb') as file:
    file.write(lines)


# In[ ]:


# Reading the dataset
class read_dataset:
    
    def __init__(self):
        pass
    
    def dataset(self):
        dicPath = os.getcwd()
        dicPathData = os.path.join(dicPath,'data.csv')
        dataset = pd.read_csv(dicPathData ,parse_dates=['order_date'])
        print()
        print("Dataset:\n\n", dataset.head())
        print('**************************************************************************************')
        print()
        return dataset


# ### Data Preprocessing with
#     -> Exploratory Data Analysis
#     -> Type casting columns
#     -> Handling null values 
#     -> Feature Engineering 

# In[ ]:


class Data_Preprocessing:
    
    def __init__(self, dataset):
        self.dataset = dataset
    
    #EDA to see the monthly number of orders to the store
    def EDA(self):
        state_grp = self.dataset['order_date'].groupby(self.dataset.order_date.dt.to_period("M")).agg('count')
        return state_grp
    
    
    def typecasting(self):
        print("Datatypes of variables before Typecasting:\n\n", self.dataset.dtypes)
        print('**************************************************************************************')
        print()
        dataset['total_spend'] = pd.to_numeric(self.dataset.total_spend, errors='coerce')
        print("Datatypes of variables after Typecasting:\n\n", self.dataset.dtypes)
        print('**************************************************************************************')
        print()
    
    #Null values are handled by replacing them with the mean 
    def null_value(self):
        print("Null values replaced with Mean:\n\n", self.dataset.isnull().sum())
        print('**************************************************************************************')
        print()
        self.dataset = self.dataset.fillna(self.dataset.mean())
    
    #New column "count" is created to see the number of times, the customer visited the store.
    def feature_engineering(self):
        self.dataset['count'] = self.dataset.groupby(['cust_id','order_date'])['cust_id'].transform('count')
        print("Number of times customers visited to the store:\n\n", self.dataset.head())
        print('**************************************************************************************')
        print()
        return self.dataset


# ### Recency, Frequency, Monetary( RFM )
#     -> RFM model is a classic analytics and segmentation tool for identifying best customers.
# 
#     -> Fundamentals of RFM -
#         1) have made a purchase recently,
#         2) make regular or frequent purchases with you,
#         3) spend a large amount with you, are more likely to return to the store.
#         
#     -> In our dataset we do not need to generate any new training or testing attributes as we can easily RFM on our dataset, since we have the relavant information we need like
#         1) Recency - Order date is provided
#         2) Frequency - we have already computed frequencies(count) above
#         3) Montary value - Total spend and units purchased attributes are provided.  
#         
# 
#     

# In[ ]:


class RFM_table_segmentation:
    
    def __init__(self, dataset):
        self.dataset = dataset
    
    #RFM table is created by finding how recently the customer visited, and how frequently the customer bought the items, and the monetary value of each customer 
    def RFM_table(self):
        NOW = dt.datetime(2016,3,27)
        self.rfmTable = self.dataset.groupby('cust_id').agg({'order_date': lambda x: (NOW - x.max()).days, 'count': lambda x: len(x), 'total_spend': lambda x: x.sum()})
        self.rfmTable['order_date'] = self.rfmTable['order_date'].astype(int)
        self.rfmTable.rename(columns={'order_date': 'recency', 
                                 'count': 'frequency', 
                                 'total_spend': 'monetary_value'}, inplace=True)
        return self.rfmTable
    
    # Customers are segmented based on the quartiles to segregate customers into different areas in RFM table
    def RFM_segment(self):
        quantiles = self.rfmTable.quantile(q=[0.25,0.5,0.75])
        quantiles = quantiles.to_dict()
        segmented_rfm = self.rfmTable
        return quantiles, segmented_rfm
    

    


# In[ ]:


### RFM scores are calculated for each customer based on the quartile range and each customer is given a RFM score.
class RFM_Scores:  
    
    def __init__(self):
        pass
    
    def RScore(self, x, p, d):
        if x <= d[p][0.25]:
            return 1
        elif x <= d[p][0.50]:
            return 2
        elif x <= d[p][0.75]: 
            return 3
        else:
            return 4
    
    def FMScore(self, x, p, d):
        if x <= d[p][0.25]:
            return 4
        elif x <= d[p][0.50]:
            return 3
        elif x <= d[p][0.75]: 
            return 2
        else:
            return 1


# In[ ]:


# Customers are mapped to various RFM levels based on RFM scores for the store to target specific customers based on their RFM levels.
class RFM_level:
    
    def __init__(self):
        pass
    
    def rfm_level(self, segmented_rfm):
        if segmented_rfm['RFM_level_Score'] >= 9:
            return 'Can\'t Loose Them'
        elif ((segmented_rfm['RFM_level_Score'] >= 8) and (segmented_rfm['RFM_level_Score'] < 9)):
            return 'Champions'
        elif ((segmented_rfm['RFM_level_Score'] >= 7) and (segmented_rfm['RFM_level_Score'] < 8)):
            return 'Loyal'
        elif ((segmented_rfm['RFM_level_Score'] >= 6) and (segmented_rfm['RFM_level_Score'] < 7)):
            return 'Potential'
        elif ((segmented_rfm['RFM_level_Score'] >= 5) and (segmented_rfm['RFM_level_Score'] < 6)):
            return 'Promising'
        elif ((segmented_rfm['RFM_level_Score'] >= 4) and (segmented_rfm['RFM_level_Score'] < 5)):
            return 'Needs Attention'
        else:
            return 'Require Activation'
        
    
    # Mapping segmented customers into various RFM levels to help the store to target specific customers for Customer retention
    def RFM_level_aggregation(self, segmented_rfm):
        # Calculate average values for each RFM_Level, and return a size of each segment 
        rfm_level_agg = segmented_rfm.groupby('RFM_Level').agg({
            'recency': 'mean',
            'frequency': 'mean',
            'monetary_value': ['mean', 'count']}).round(1)
        
        segmented_rfm = segmented_rfm.sort_values('RFM_level_Score', ascending = False)
        print()
        print()
        print("Aggregated RFM scores of Customers:\n", rfm_level_agg)
        print()
        print('**************************************************************************************')
    
    # Plotting the Customers segmented on different RFM levels 
        plt.figure(figsize=(12,8))
        sns.countplot(x='RFM_Level', data=segmented_rfm)
        plt.ylabel('Count', fontsize=12)
        plt.xlabel('RFM_Level', fontsize=12)
        plt.xticks(rotation='vertical')
        plt.title('Customer Segmentation based on RFM Level', fontsize=15)
        plt.show()


# In[ ]:


# Main Function 
if __name__ == '__main__':
    
    # Reading Dataset
    d = read_dataset()
    dataset = d.dataset()
    
    # Data Preprocessing
    preprocessor = Data_Preprocessing(dataset)
    eda = preprocessor.EDA()
    eda.plot()
    preprocessor.typecasting()
    preprocessor.null_value()
    processed_dataset = preprocessor.feature_engineering()
    
    # RFM Table, Segmentation
    rfm_segment = RFM_table_segmentation(processed_dataset)
    rfm_table = rfm_segment.RFM_table()
    print("The RFM table is given by:\n", rfm_table)
    print()
    print('***************************************************************************************')
    print()
    quantiles, segmented_rfm = rfm_segment.RFM_segment()
    
    # RFM scores are determined and is applied across RFM table
    rfm_scores = RFM_Scores()
    segmented_rfm['r_quartile'] = segmented_rfm['recency'].apply(rfm_scores.RScore, args=('recency',quantiles,))
    segmented_rfm['f_quartile'] = segmented_rfm['frequency'].apply(rfm_scores.FMScore, args=('frequency',quantiles,))
    segmented_rfm['m_quartile'] = segmented_rfm['monetary_value'].apply(rfm_scores.FMScore, args=('monetary_value',quantiles,))
    print("Segmented RFM table is given by:\n", segmented_rfm.head())
    print()
    print('***************************************************************************************')
    
    
    # RFM scores are attached to the dataset
    segmented_rfm['RFMScore'] = segmented_rfm.r_quartile.map(str) + segmented_rfm.f_quartile.map(str) + segmented_rfm.m_quartile.map(str)
    
    segmented_rfm['RFM_level_Score'] = segmented_rfm[['r_quartile','f_quartile','m_quartile']].sum(axis=1)
    print("RFM table with RFM scores attached:\n", segmented_rfm.head())
    print()
    print('**************************************************************************************')
    print()
    
    # RFM levels are defined to map customers into various levels
    RFM_Level = RFM_level()
    segmented_rfm['RFM_Level'] = segmented_rfm.apply(RFM_Level.rfm_level, axis=1) 
    print()
    print("Final RFM table with Customers mapped to Different RFM levels:\n", segmented_rfm.head())
    print()
    print('**************************************************************************************')
    
    RFM_Level.RFM_level_aggregation(segmented_rfm)


# In[ ]:




