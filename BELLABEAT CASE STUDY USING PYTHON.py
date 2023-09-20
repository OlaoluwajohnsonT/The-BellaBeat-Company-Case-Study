#!/usr/bin/env python
# coding: utf-8

# # THE BELLABEAT CASE STUDY

# In this case study, I find myself on the marketing analytics team at Bellabeat, a company specializing in health-focused products for women. Our mission is to unlock growth opportunities for the company by analyzing smart device fitness data. Under the guidance of Bellabeat's co-founder and Chief Creative Officer, Urška Sršen, I'm tasked with delving into smart device usage trends.
# 
# The goal is to gain insights into how consumers utilize non-Bellabeat smart devices, with the aim of applying these findings to enhance our marketing strategy. I'm excited to contribute to Bellabeat's mission of empowering women through data-driven health and wellness solutions.
# 
# To begin, I'll explore the FitBit Fitness Tracker Data, a publicly available dataset containing minute-level data on physical activity, heart rate, and sleep monitoring from Fitbit users. However, I'll also consider supplementing this data to address potential limitations. My journey will involve data preparation, ensuring the dataset's integrity, and addressing any data-related challenges.
# 
# Ultimately, I aim to produce a comprehensive report with compelling visualizations and actionable recommendations. By leveraging these insights, we can shape Bellabeat's marketing strategy to resonate with our target audience and drive growth in the competitive smart device market.

# In[126]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import sklearn
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns
sklearn.__version__


# I am goint to import all the require datasets

# In[127]:


activity= pd.read_csv("C:\\Users\TOJMARK LTD\\DATA SCIENCE PROJECT\\BellaBeat Project\\dailyActivity_merged.csv")
calories= pd.read_csv("C:\\Users\\TOJMARK LTD\\DATA SCIENCE PROJECT\\BellaBeat Project\\dailyCalories_merged.csv")
intense= pd.read_csv("C:\\Users\\TOJMARK LTD\\DATA SCIENCE PROJECT\\BellaBeat Project\\dailyIntensities_merged.csv")
step= pd.read_csv("C:\\Users\\TOJMARK LTD\\DATA SCIENCE PROJECT\\BellaBeat Project\\dailySteps_merged.csv")
heartrate= pd.read_csv("C:\\Users\\TOJMARK LTD\\DATA SCIENCE PROJECT\\BellaBeat Project\\heartrate_seconds_merged.csv")
sleep= pd.read_csv("C:\\Users\\TOJMARK LTD\\DATA SCIENCE PROJECT\\BellaBeat Project\\sleepDay_merged.csv")
weight= pd.read_csv("C:\\Users\\TOJMARK LTD\\DATA SCIENCE PROJECT\\BellaBeat Project\\weightLogInfo_merged.csv")


# # Data Exploratory

# In[128]:


activity.head()


# In[129]:


calories.head()


# In[130]:


sleep.head()


# In[131]:


intense.head()


# Let confirm the shape of each dataset

# In[132]:


print("Activity shape:", activity.shape)
print("Sleep shape:",sleep.shape)
print("Intensity shape:",intense.shape)
print("Calories shape:",calories.shape)


# # Checking for Misssing values

# In[133]:


# Check for missing values in each column
activity.isna().sum()


# In[134]:


sleep.isna().sum()


# In[135]:


intense.isna().sum()


# It shows each dataset has no missing values, now let me futher exploring the dataset to familirise myself with it

# In[136]:


print(activity.describe())


# There are some interesting statistics related to physical activity and fitness tracking. The dataset comprises information from 940 individuals, and it offers valuable insights into their daily activity patterns.
# 
# On average, users take approximately 7,638 steps a day, covering a distance of around 5.49 kilometers. Interestingly, some users logged no activity, while others recorded a maximum of 36,019 steps. The data also reveals variations in activity intensity, with users engaging in very active, moderately active, and light active distances. Sedentary activity is almost negligible for most users.
# 
# In terms of minutes spent in different activity categories, users generally spend around 21 minutes in very active pursuits, 13 minutes in fairly active ones, and a significant portion of their day, approximately 991 minutes, in sedentary activities. Caloric burn averages at 2,303 calories.
# 
# These statistics paint a picture of diverse physical activity levels among users. It's clear that some individuals are highly active, while others are more sedentary. 

# In[137]:


activity.info()


# In[138]:


sleep.info()


# In[139]:


# Calculate the average for some specific columns on activity dataset and round them to 0 decimal places
average_total_steps = activity['TotalSteps'].mean()
average_total_distance = activity['TotalDistance'].mean()
average_sedentary_minutes = activity['SedentaryMinutes'].mean()
average_calories = activity['Calories'].mean()

# Round the average values to 0 decimal places
average_total_steps = round(average_total_steps, 0)
average_total_distance = round(average_total_distance, 0)
average_sedentary_minutes = round(average_sedentary_minutes, 0)
average_calories = round(average_calories, 0)

print(f"Average Total Steps: {average_total_steps}")
print(f"Average Total Distance: {average_total_distance}")
print(f"Average Sedentary Minutes: {average_sedentary_minutes}")
print(f"Average Calories: {average_calories}")


# In the dataset, the average daily steps recorded by users stand at 7,638, indicating moderate activity levels. Sedentary minutes are notably high, averaging around 991 minutes, highlighting a need for promoting more active lifestyles. Users also burn an average of 2,304 calories daily, reflecting diverse activity levels and dietary needs.

# In[140]:


# Calculate the average for the specific columns for sleep dataset
average_total_sleep_records = sleep['TotalSleepRecords'].mean()
average_total_minutes_asleep = sleep['TotalMinutesAsleep'].mean()
average_total_time_in_bed = sleep['TotalTimeInBed'].mean()

# Round the average values to 0 decimal places
average_total_sleep_records = round(average_total_sleep_records, 0)
average_total_minutes_asleep = round(average_total_minutes_asleep, 0)
average_total_time_in_bed = round(average_total_time_in_bed, 0)

print(f"Average Total Sleep Records: {average_total_sleep_records}")
print(f"Average Total Minutes Asleep: {average_total_minutes_asleep}")
print(f"Average Total Time In Bed: {average_total_time_in_bed}")


# # Number of Participant In Each Dataset From FitBit Fitness Tracker

# In[141]:


# Count the number of distinct users in the 'activity' dataset
distinct_users_activity = activity['Id'].nunique()
distinct_users_sleep = sleep['Id'].nunique()
distinct_users_intense = intense['Id'].nunique()
distinct_users_calories = calories['Id'].nunique()

print(f"Number of Distinct Users in 'activity' dataset: {distinct_users_activity}")
print(f"Number of Distinct Users in 'sleep' dataset: {distinct_users_sleep}")
print(f"Number of Distinct Users in 'sleep' dataset: {distinct_users_intense}")
print(f"Number of Distinct Users in 'sleep' dataset: {distinct_users_calories}")


# There are 33 participants apart from sleep data that has 24 

# # Merging Activity and Sleep Dataset 
# I need to merge activity dataset and sleep data before we can continue for our exploration. To get this done require us to convert the date to the same datetime format so as to be able to merge both data by date

# In[142]:


# Converting date columns to a consistent format
activity['ActivityDate'] = pd.to_datetime(activity['ActivityDate'])
sleep['SleepDay'] = pd.to_datetime(sleep['SleepDay'], format='%m/%d/%Y %I:%M:%S %p')

# Seting date columns as the index
activity.set_index('ActivityDate', inplace=True)
sleep.set_index('SleepDay', inplace=True)

# Merging the datasets on the date index
merged_data = pd.merge(activity, sleep, how='inner', left_index=True, right_index=True)


# In[143]:


merged_data.head()


# # Explorative analysis

# In[144]:


#Let first utlise pairplot to check for relationship of some specific columns
# Selecting the relevant numeric columns from merged_data
selected_columns = ['TotalDistance', 'TotalSteps', 'VeryActiveMinutes', 'LightlyActiveMinutes',
                    'SedentaryMinutes', 'TotalTimeInBed', 'Calories', 'TotalMinutesAsleep']
numeric_data = merged_data[selected_columns]

# Create a pairplot
sns.pairplot(numeric_data, diag_kind='kde')

# Show the plot
plt.show()


# In[145]:


# using corr function to check correlation of total steps and calories burn
correlation = merged_data['TotalSteps'].corr(merged_data['Calories'])

print(f"Correlation between Total Steps and Calories: {correlation:.2f}")


# In[147]:


#Let visualising it
x = merged_data['TotalSteps']
y = merged_data['Calories']

# Creating a scatter plot to check the relationship
plt.figure(figsize=(10, 5))
plt.scatter(x, y, alpha=0.5)
plt.title('Total Steps vs. Calories')
plt.xlabel('Total Steps')
plt.ylabel('Calories Burned')
plt.grid(True)
plt.show()


# It shows there is a correlation between the totoal steps and the calories

# In[105]:


# Assuming you have a merged dataset named 'merged_data'
x = merged_data['TotalMinutesAsleep']
y = merged_data['TotalTimeInBed']

# Create a scatter plot
plt.figure(figsize=(10, 5))
plt.scatter(x, y, alpha=0.5)
plt.title('Total Minutes Asleep vs. Total Time in Bed')
plt.xlabel('Total Minutes Asleep')
plt.ylabel('Total Time in Bed')
plt.grid(True)

# Show the scatter plot
plt.show()


# This is a strong correlation between the minute in bed and the total minute asleep. It indiscate that the perticipant falls asleep peacefully without no concern

# In[148]:


# Assuming you have a merged dataset named 'merged_data'
x = merged_data['SedentaryMinutes']
y = merged_data['TotalMinutesAsleep']

# using corr function to check correlation of total steps and calories burn
correlation = merged_data['SedentaryMinutes'].corr(merged_data['TotalMinutesAsleep'])

print(f"Correlation between Sedentary Minutes and Total Minutes Asleep: {correlation:.2f}")

# Create a scatter plot
plt.figure(figsize=(10, 5))
plt.scatter(x, y, alpha=0.5)
plt.title('Scatter Plot: Sedentary Minutes vs. Total Minutes Asleep')
plt.xlabel('Sedentary Minutes')
plt.ylabel('Total Minutes Asleep')
plt.grid(True)

# Show the scatter plot
plt.show()


# A correlation coefficient of -0.01 suggests a weak negative correlation between sedentary minutes and total minutes asleep. This implies that there is a minor tendency for users who engage in less exercise to experience slightly less sleepiness. However, it's important to note that the correlation is very weak and may not have significant practical implications.

# In[149]:


# Extract the day of the week from the index
merged_data['DayOfWeek'] = merged_data.index.day_name()

# Group the data by 'DayOfWeek' and calculate the mean of 'TotalSteps' for each day
grouped_data = merged_data.groupby('DayOfWeek')['TotalSteps'].mean()

# Define the order of days of the week
days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Sort the grouped data by the predefined order
grouped_data = grouped_data.reindex(days_order)

# Create a bar plot to visualize the mean steps for each day of the week
plt.figure(figsize=(10, 6))
grouped_data.plot(kind='bar', color='skyblue')
plt.title('Average Total Steps by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Average Total Steps')
plt.xticks(rotation=45)
plt.grid(axis='y')

# Show the bar plot
plt.show()


# The most active day is Saturday follows by monday

# # The Relationship Between Activity and Weight Dataset for BMI, WEIGHT, FAT
# To explore the relationship between activity and weight in the dataset, specifically looking at BMI (Body Mass Index), weight, and fat, I will perform correlation analysis and visualize these relationships using scatter plots and corr functions. 

# In[150]:


weight.head()


# In[151]:


weight.info()


# In[152]:


activity.info()


# In[153]:


# Merging the 'activity' and 'weight' datasets on the 'Id' column
merged_weight = pd.merge(activity, weight, on='Id', how='inner')

# Calculate the correlations
correlation_bmi = merged_weight['TotalSteps'].corr(merged_weight['BMI'])
correlation_weight = merged_weight['TotalSteps'].corr(merged_weight['WeightKg'])
correlation_fat = merged_weight['TotalSteps'].corr(merged_weight['Fat'])
print(correlation_bmi)
print(correlation_weight)
print(correlation_fat)


# The negative sign shows that there is negative relation between the total step to weight, bmi and fat contained. To finalise it, let check through visualisation

# In[154]:


# Create scatter plots to visualize the relationships
plt.figure(figsize=(12, 5))

# Scatter plot for BMI vs. Total Steps
plt.subplot(131)
plt.scatter(merged_weight['BMI'], merged_weight['TotalSteps'], alpha=0.5)
plt.title(f'Correlation = {correlation_bmi:.2f}')
plt.xlabel('BMI')
plt.ylabel('Total Steps')


# In[155]:


# Scatter plot for Weight vs. Total Steps
plt.subplot(132)
plt.scatter(merged_weight['WeightKg'], merged_weight['TotalSteps'], alpha=0.5)
plt.title(f'Correlation = {correlation_weight:.2f}')
plt.xlabel('Weight (Kg)')
plt.ylabel('Total Steps')


# In[156]:


# Scatter plot for Fat vs. Total Steps
plt.subplot(133)
plt.scatter(merged_weight['Fat'], merged_weight['TotalSteps'], alpha=0.5)
plt.title(f'Correlation = {correlation_fat:.2f}')
plt.xlabel('Fat')
plt.ylabel('Total Steps')

plt.tight_layout()

# Show the scatter plots
plt.show()


# # Let check for sendatory minutes with Fat, BMI and WEIGHT

# In[157]:


# Creating a scatter plot for Sedentary Minutes vs. Fat
plt.figure(figsize=(15, 5))

# Scatter plot for Sedentary Minutes vs. Fat
plt.subplot(131)
plt.scatter(merged_weight['SedentaryMinutes'], merged_weight['Fat'], alpha=0.5)
plt.title('Sedentary Minutes vs. Fat')
plt.xlabel('Sedentary Minutes')
plt.ylabel('Fat')

# Scatter plot for Sedentary Minutes vs. BMI
plt.subplot(132)
plt.scatter(merged_weight['SedentaryMinutes'], merged_weight['BMI'], alpha=0.5)
plt.title('Sedentary Minutes vs. BMI')
plt.xlabel('Sedentary Minutes')
plt.ylabel('BMI')

# Scatter plot for Sedentary Minutes vs. Weight
plt.subplot(133)
plt.scatter(merged_weight['SedentaryMinutes'], merged_weight['WeightKg'], alpha=0.5)
plt.title('Sedentary Minutes vs. Weight (Kg)')
plt.xlabel('Sedentary Minutes')
plt.ylabel('Weight (Kg)')

plt.tight_layout()

# Show the scatter plots
plt.show()


# In[158]:


# Creating a scatter plot for Total Time in Bed vs. Fat
plt.figure(figsize=(15, 5))

# Scatter plot for Total Time in Bed vs. Fat
plt.subplot(131)
plt.scatter(merged_weight['Calories'], merged_weight['Fat'], alpha=0.5)
plt.title('Calories Burn vs. Fat')
plt.xlabel('Calories')
plt.ylabel('Fat')

# Scatter plot for Total Time in Bed vs. BMI
plt.subplot(132)
plt.scatter(merged_weight['Calories'], merged_weight['BMI'], alpha=0.5)
plt.title('Calories Burn vs. BMI')
plt.xlabel('Calories')
plt.ylabel('BMI')

# Scatter plot for Total Time in Bed vs. Weight
plt.subplot(133)
plt.scatter(merged_weight['Calories'], merged_weight['WeightKg'], alpha=0.5)
plt.title('Calories Burn vs. Weight (Kg)')
plt.xlabel('Calories')
plt.ylabel('Weight (Kg)')

plt.tight_layout()

# Show the scatter plots
plt.show()


# In[159]:


# let calculate the correlation
correlation_fat = merged_weight['Calories'].corr(merged_weight['Fat'])
correlation_bmi = merged_weight['Calories'].corr(merged_weight['BMI'])
correlation_weight = merged_weight['Calories'].corr(merged_weight['WeightKg'])

print(f"Correlation between Calories burn and Fat: {correlation_fat:.2f}")
print(f"Correlation between  burnCalories burn and BMI: {correlation_bmi:.2f}")
print(f"Correlation between Calories burn and Weight (Kg): {correlation_weight:.2f}")


# In[160]:


# Group the data by 'DayOfWeek' and calculate the mean of 'TotalSteps' and 'BMI'
grouped_data = merged_weight.groupby('Id')[['TotalSteps', 'BMI']].mean()

# Calculate the correlation between average total steps and average BMI
correlation = grouped_data['TotalSteps'].corr(grouped_data['BMI'])

# Create a scatter plot to visualize the relationship
plt.figure(figsize=(8, 6))
plt.scatter(grouped_data['TotalSteps'], grouped_data['BMI'], alpha=0.5)
plt.title(f'Correlation = {correlation:.2f}')
plt.xlabel('Average Total Steps')
plt.ylabel('Average BMI')
plt.grid(True)

# Show the scatter plot
plt.show()


# The correlation coefficient of -0.74 indicates a strong negative relationship between active participants and variables such as weight, fat content, and BMI (Body Mass Index). This suggests that as activity levels increase, there is a tendency for weight, fat content, and BMI to decrease.

# # Findings
# 
# The average total steps taken by the participants in the 'activity' dataset is 7638, with an average total distance of 5.0 miles.
# 
# Participants spend an average of 991 minutes in sedentary activities and burn an average of 2304 calories.
#     
# The 'activity' dataset contains data from 33 distinct users, while the 'sleep' dataset contains data from 24 distinct users.
#     
# There is a strong positive correlation of 0.59 between total steps and calories burned, indicating that as the number of steps increases, the calories burned also increase.
#     
# A notable finding is a strong positive correlation between the time spent in bed and the total minutes asleep, suggesting that participants have restful sleep with no interruptions.
#     
# Surprisingly, there is a negligible correlation of -0.01 between sedentary minutes and total minutes asleep, indicating that sedentary behavior doesn't significantly affect sleep duration.
#     
# Saturday is the most active day of the week, followed by Monday.
#     
# There is a weak negative correlation of -0.16 between total steps and BMI, a moderate positive correlation of 0.26 between total steps and weight, and a strong negative correlation of -0.60 between total steps and body fat percentage.
#     
# Calories burned have a moderate positive correlation of 0.28 with body fat percentage and a weak positive correlation of 0.12 with BMI. However, there is a strong positive correlation of 0.65 between calories burned and weight (in kilograms).
#     
# The data suggests a negative relationship between participants' activity levels and their weight, body fat percentage, and BMI, indicating that more active participants tend to have lower weight and body fat percentages.
# 

# # Recommendations
# 
# Encourage users to increase their daily step count to help burn more calories and maintain a healthier weight.
# 
# Highlight the importance of regular physical activity, especially on Saturdays and Mondays when users tend to be more active.
# 
# Consider creating personalized fitness plans based on users' goals and body composition to help them achieve their desired weight and body fat percentage.
# 
# Develop targeted marketing campaigns to promote the relationshp between physical activity and calorie burn, emphasizing the health benefits.
# 
# Integrate weight management features into the app, allowing users to track their weight and receive recommendations for achieving their weight goals.
# 
# Offer nutrition and diet-related content within the app to complement users' fitness journeys and help them make healthier dietary choices.
# 
# 

# # Thank you

# In[ ]:




