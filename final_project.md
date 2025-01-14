# Data-Science-and-machine-learning-for-the-biosciences-Assessment

# Analysis and predictions regarding the characteristics of my research area based on satellite images, before commencing fieldwork

## INTRODUCTION

## METHODS
### - The data was obtained using Google Earth Engine (GEE), with the Mapbiomas User Toolkit for maps in Brazil.
### - The database was created by Elizabeth Renteria
### - Data set Book1. is a data set about 18 diferent sites (plots) from the Amazon forest in Brazil. These 18 transects are the same size, a radius of 2km, but each have a different composition of habitat type. Each of them can have a different combination between 11 habitat types. Having a maximal of 6 combination and a minimun or 1 single type. The measurements from the different habitat types in each transect was obtain using Google Earth Engine with satelitte images from 2013-2023 
### - I used a second data (centerb) base to explore a bigger area than contains all 18 plots plus the area between the plots. An area of 30,000 m2 radius. I found it important to also analize this data base, because it can contain information thta is important to undertand the main land use differences in the whole area and not only snips of it.

## Data Analysis
### - I used Python to analize and visualize the data.
### - First, I used clustering to explore the data with k-means.
### - Second, I created a Heatmap because the cluster plot is not very specific, so it would be better to have a heatmap, so we can see with more detail the difference in habitat composition.
### - Third, perform an ANOVA to see if the differences in the heatmaps are significant
### - Fourth, perform a time series analysis with a 5 year forecast, using the ARIMA function. I only used the first plot from the first database to show the forecast, because 18 plot is too long to plot.
### - Fifth, perform a linear regression only in the secon data base, to see the relationship between the area of the 4 most important habitat types in the last 11 years.

## RESULTS

## Clusters

![png](output_1_1.png)
### This plot should plot the 4 clusters of the Area in squared meters of each plot(BufferID). But since each of the transects have the same Area and only changes the type of habitat, I can argue that the plot actually could represent the levels of homogenity or heterogenity of the plot. Cluster 1 are the more homogenous so they have more area because its only distributed in 1 habitat type. Cluster 0 are plots that have their area divided in 5 or 6 habitat types. Cluster 2, 3-4 habitat types. Cluster 3, 2-3 habitat types (18 plots).

## Heatmaps

![png](output_1_2.png)
### In this heatmap we can see that our main habitat type is "Forest formation" in most plots, followed by "OilPalm". We can also observed than a lot of the other habitat types contribute with less than 1 percent to the plots composition (18 plots).

![png](output_1_3.png)
### In this heatmap we can see that our main habitat type is "Forest formation" all Years, followed by "OilPalm". The percentages dont really change across the years (18 plots).




## ANOVA

#####   ANOVA for Plot : F-statistic: 9.00751058069251, p-value: 1.3372846720700188e-21
![png](output_1_5.png)
### ANOVA for Plots. There is a significant difference between the plots, being plot TAI_02 and TAI_09 the ones that produce this difference. Plot TAI_02 and TAI_09 are the plots that are compound by only one habitat type (ForestFormation) (18 plots).

#####   ANOVA for Type: F-statistic: 464.7582383256251, p-value: 1.64e-321

![png](output_1_7.png)
### ANOVA for Habitat Type. There is a significant difference between the Habitat types, being ForestFormation and OilPalm the ones thta produce this difference.
#Both types are the one with the largest area_m2 (18 plots).

#####  ANOVA for Year: F-statistic: 0.04438257860079598, p-value: 0.9999961834517294
![png](output_1_9.png)
### ANOVA for Year. There was no a significant in the area differences between the 11 years (18 plots).

## Time Analysis and Forecast

![png](output_1_11.png)
# The area m2 of ForestFormation habitat type in the Plot TAI_01 , is expected to increase in the next 5 years. After checking the forecast by fittting an  ARIMA model on training data, I can conlude is accurate, since the forecast is inside the confidence interval (18 plots).

## Linear Regression

```python



```python
#########################################################################################################################################################
##############################                                SECOND DATA BASE               ####################################

# I used a second data base to explore a bigger area than contains all 18 plots plus the area between the plots. An area of 30,000 m2 radius.
# I found important to also analize this data base, because it can contain information thta is important to undertand the main land use differences
# in the whole area and not only snips of it 

center = pd.read_excel(r"C:\Users\vs24904\OneDrive - University of Bristol\Documents\centerb.xlsx")

##############################               First, I am going to use some clustering to explore the data            ##################################

# Group by BufferID and calculate the average area
center_avg_area = center.groupby('Type')['Area_m2'].mean().reset_index()
# Standardize the data
scaler = StandardScaler()
center_avg_area_scaled = scaler.fit_transform(center_avg_area[['Area_m2']])

# Apply K-Means clustering
kmeans = KMeans(n_clusters=4, random_state=0)
center_avg_area['Cluster'] = kmeans.fit_predict(center_avg_area_scaled)

# Plot the clusters
# Here the cluster plot indicates which habitat type has more area in total, in the big area that encompas the 18 plots.
# We can see that in the big area "ForestFormation" has the biggest squared meter area, followed by "Pasture" and "OilPalm"

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Type', y='Area_m2', hue='Cluster', data=center_avg_area, palette='viridis', s=100)
plt.title('K-Means Clustering of Habitat Type by Average Area m2')
plt.xlabel('Type')
plt.ylabel('Average Area (m²)')
plt.legend(title='Cluster')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('K-Means Clustering of Habitat Type by Average Area m2 center.png')
plt.show()

#########   HEAT MAP FOR HABITAT TYPE PER YEAR

# Aggregate by BufferID and Areas to calculate the average proportion across all years
center_avg_proportion = center.groupby(['Date', 'Type'])['Area_m2'].sum().reset_index()
# Calculate the proportion of Area_m2 for each Area within each Plot
center_avg_proportion['Proportion'] = center_avg_proportion.groupby('Date')['Area_m2'].transform(lambda x: x / x.sum())
# Pivot the data for heatmap
center_pivot_avg = center_avg_proportion.pivot_table(values='Proportion', index='Date', columns='Type', fill_value=0)

#Plot the heatmap
#In the heatmap we can see that our main habitat type is "Forest formation" all Years, followed by "OilPalm".
#The percentages dont really change across the years.
plt.figure(figsize=(12, 8))
sns.heatmap(center_pivot_avg, cmap='PiYG', annot=True, fmt='.2f', linewidths=0.5)
plt.title('Average Proportion of Total Area by Habitat Type for Each Year')
plt.xlabel('Habitat Type')
plt.ylabel('Year')
plt.xticks(rotation=80)
plt.tight_layout()
plt.savefig('Average Proportion of Total Area by Habitat Type for Each Year.png')
plt.show()

##############################              ARE THE DIFFERENCES IN THE HEAT MAPS SIGNIFICANT                                ####################


#################       ANOVA for Type

Type_cent_groups = [center[center['Type'] == Type]['Area_m2'].values for Type in center['Type'].unique()]
anova_cent_Type = f_oneway(*Type_cent_groups)
print("ANOVA for Type:")
print(f"F-statistic: {anova_cent_Type.statistic}, p-value: {anova_cent_Type.pvalue}")

# Perform Tukey's HSD test for Type
tukey_type = pairwise_tukeyhsd(endog=center['Area_m2'], groups=center['Type'], alpha=0.05)
# Display the results
print("\nTukey HSD Test for PLot:")
print(tukey_type)
# Visualize the Tukey HSD results
tukey_type.plot_simultaneous()
plt.title("Tukey HSD Test for Habitat Type")
plt.xlabel('Mean Difference')
plt.grid(True)
plt.savefig('Tukey HSD Test for Type center.png')
plt.show()
#There is a significant difference between the Habitat types, being ForestFormation, Pastures, OilPalm, and Floodplains the ones that produce this difference.
#These types are the one with the largest area_m2
# A difference can be noted between the whole area of the experiment and the selected plot. 
# If only we take only the plots under accountance we get a false overview of the area, beacause the plots were selected to get a gradient of type of forest
# But ForestFormation has by far the largest area. If we take also the whole area encompasing the plots, we can see that the area is actually pretty
# heterogenic, having a high area disturbance type habitats, like Pasture and Oilpalm

######################################################################################
#################      ANOVA for Year
year_groups = [center[center['Date'] == year]['Area_m2'].values for year in center['Date'].unique()]
anova_year = f_oneway(*year_groups)
print("\nANOVA for Year:")
print(f"F-statistic: {anova_year.statistic}, p-value: {anova_year.pvalue}")

# Perform Tukey's HSD test for Year
tukey_Year = pairwise_tukeyhsd(endog=center['Area_m2'], groups=center['Date'], alpha=0.05)
# Display the results
print("\nTukey HSD Test for Year:")
print(tukey_Year)
# Visualize the Tukey HSD results
tukey_Year.plot_simultaneous()
plt.title("Tukey HSD Test for Year")
plt.xlabel('Mean Difference')
plt.grid(True)
plt.savefig('Tukey HSD Test for Year center.png')
plt.show()
#There was no a significant in the area differences between the 11 years

#######################################################################################################################################################
#############################################################################################################
#####                                                    TIME SERIES AND FORECASTING

####    PASTURES
# Load the dataset
df = pd.read_excel(r"C:\Users\vs24904\OneDrive - University of Bristol\Documents\centerb.xlsx")

# Identify unique BufferIDs and Area
unique_dates = df['Date'].unique()
unique_buffer_ids = df['BufferID'].unique()
unique_areas = df['Type'].unique()

# Create a complete DataFrame with all combinations
all_combinations = pd.DataFrame(
    list(itertools.product(unique_dates, unique_buffer_ids, unique_areas)),
    columns=['Date', 'BufferID', 'Type'])
# Step 3: Merge with the original DataFrame and fill missing values with 0
df_complete = pd.merge(all_combinations, df, on=['Date', 'BufferID', 'Type'], how='left').fillna({'Area_m2': 0})

# Time series analysis for a specific BufferID
area = 'Pasture'# Example Area
df_filtered = df[(df['Type'] == area)].copy()

# Convert Date to numeric index for time series modeling
df_filtered['Date'] = pd.to_numeric(df_filtered['Date'])
df_filtered.sort_values('Date', inplace=True)
df_filtered.set_index('Date', inplace=True)

# Define the model with specified order (p, d, q)
model = ARIMA(df_filtered['Area_m2'], order=(1, 2, 1))
# Fit the model
model_fit = model.fit()
#print(model_fit.summary())

#Forecast
steps = 5
forecast = model_fit.forecast(steps=steps)
forecast_years = range(df_filtered.index[-1] + 1, df_filtered.index[-1] + 1 + steps)
forecast_df = pd.DataFrame({'Date': forecast_years, 'Forecasted_Area_m2': forecast})

#Plot and forecast
plt.figure(figsize=(12, 6))
plt.plot(df_filtered.index, df_filtered['Area_m2'], label='Observed', color='blue')
plt.plot(forecast_df['Date'], forecast_df['Forecasted_Area_m2'], label='Forecast', color='red', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Area (m2)')
plt.title(f'Time Series Forecast for Area: {area} in the Tailandia(Pará, Brazil)')
plt.legend()
plt.grid(True)
plt.savefig('time series forecast pasture tailandia.png')
plt.show()

print(forecast_df)

# The area m2 of Pasture habitat type in the site (Tailandia-Brazil), are expected to grow in the next 5 years
#I Checked the forecast using "ARIMA" and the forecast looks plausible


#####################################################################################################################################

#####                                                   test if forecast is correct or good



# Split data into training and testing sets (e.g., last 5 years for testing)
train_size = int(len(df_filtered) * 0.6)
train_data, test_data = df_filtered.iloc[:train_size], df_filtered.iloc[train_size:]

# Fit ARIMA model on the training data
arima_model_train = ARIMA(train_data['Area_m2'], order=(1, 2, 1))
arima_fit_train = arima_model_train.fit()

# Forecast for the test period
forecast_test = arima_fit_train.get_forecast(steps=len(test_data))
forecasted_values = forecast_test.predicted_mean
confidence_intervals = forecast_test.conf_int()

# Evaluation metrics
mae = mean_absolute_error(test_data['Area_m2'], forecasted_values)
mse = mean_squared_error(test_data['Area_m2'], forecasted_values)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((test_data['Area_m2'] - forecasted_values) / test_data['Area_m2'])) * 100

# Print evaluation metrics
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAPE: {mape}%")

# Plot the forecast with confidence intervals
plt.figure(figsize=(12, 6))
plt.plot(train_data.index, train_data['Area_m2'], label='Training Data', color='blue')
plt.plot(test_data.index, test_data['Area_m2'], label='Actual Test Data', color='green')
plt.plot(test_data.index, forecasted_values, label='Forecast', color='red', linestyle='--')
plt.fill_between(test_data.index, confidence_intervals.iloc[:, 0], confidence_intervals.iloc[:, 1], color='pink', alpha=0.3)
plt.xlabel('Year')
plt.ylabel('Area (m²)')
plt.title('ARIMA Pastures Forecast vs Actuals with Confidence Intervals')
plt.legend()
plt.grid(True)
plt.savefig('ARIMA Pastures Forecast vs Actuals with Confidence Intervals.png')
plt.show()


##############################################################################################################

####    ForestFormation

# Load the dataset
df = pd.read_excel(r"C:\Users\vs24904\OneDrive - University of Bristol\Documents\centerb.xlsx")
print(df)

# Step 1: Identify unique BufferIDs and Area
unique_dates = df['Date'].unique()
unique_buffer_ids = df['BufferID'].unique()
unique_areas = df['Type'].unique()

# Step 2: Create a complete DataFrame with all combinations
import itertools
all_combinations = pd.DataFrame(
    list(itertools.product(unique_dates, unique_buffer_ids, unique_areas)),
    columns=['Date', 'BufferID', 'Type'])
# Step 3: Merge with the original DataFrame and fill missing values with 0
df_complete = pd.merge(all_combinations, df, on=['Date', 'BufferID', 'Type'], how='left').fillna({'Area_m2': 0})

# Time series analysis for a specific BufferID
area = 'ForestFormation'# Example Area
df_filtered = df[(df['Type'] == area)].copy()

# Convert Date to numeric index for time series modeling
df_filtered['Date'] = pd.to_numeric(df_filtered['Date'])
df_filtered.sort_values('Date', inplace=True)
df_filtered.set_index('Date', inplace=True)

# Define the model with specified order (p, d, q)
model = ARIMA(df_filtered['Area_m2'], order=(1, 2, 1))
# Fit the model
model_fit = model.fit()
#print(model_fit.summary())

#Forecast
steps = 5
forecast = model_fit.forecast(steps=steps)
forecast_years = range(df_filtered.index[-1] + 1, df_filtered.index[-1] + 1 + steps)
forecast_df = pd.DataFrame({'Date': forecast_years, 'Forecasted_Area_m2': forecast})

#Plot and forecast
plt.figure(figsize=(12, 6))
plt.plot(df_filtered.index, df_filtered['Area_m2'], label='Observed', color='blue')
plt.plot(forecast_df['Date'], forecast_df['Forecasted_Area_m2'], label='Forecast', color='red', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Area (m2)')
plt.title(f'Time Series Forecast for Area: {area} in the Tailandia(Pará, Brazil)')
plt.legend()
plt.grid(True)
plt.savefig('Forest formation Forecast tailandia.png')
plt.show()

print(forecast_df)

# The area m2 of ForestFormation habitat type in the site (Tailandia-Brazil), is expected to decrease in the next 5 years
#I Checked the forecast using "ARIMA" and the forecast looks plausible, beacuse is inside the confidence intervals

#####################################################################################################################################

#####                                                   test if forecast is correct or good

# Split data into training and testing sets (e.g., last 5 years for testing)
train_size = int(len(df_filtered) * 0.5)
train_data, test_data = df_filtered.iloc[:train_size], df_filtered.iloc[train_size:]

# Fit SARIMA model on the training data
arima_model_train = ARIMA(train_data['Area_m2'], order=(1, 2, 1))
arima_fit_train = arima_model_train.fit()

# Forecast for the test period
forecast_test = arima_fit_train.get_forecast(steps=len(test_data))
forecasted_values = forecast_test.predicted_mean
confidence_intervals = forecast_test.conf_int()

# Evaluation metrics
mae = mean_absolute_error(test_data['Area_m2'], forecasted_values)
mse = mean_squared_error(test_data['Area_m2'], forecasted_values)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((test_data['Area_m2'] - forecasted_values) / test_data['Area_m2'])) * 100

# Print evaluation metrics
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAPE: {mape}%")

# Plot the forecast with confidence intervals
plt.figure(figsize=(12, 6))
plt.plot(train_data.index, train_data['Area_m2'], label='Training Data', color='blue')
plt.plot(test_data.index, test_data['Area_m2'], label='Actual Test Data', color='green')
plt.plot(test_data.index, forecasted_values, label='Forecast', color='red', linestyle='--')
plt.fill_between(test_data.index, confidence_intervals.iloc[:, 0], confidence_intervals.iloc[:, 1], color='pink', alpha=0.3)
plt.xlabel('Year')
plt.ylabel('Area (m²)')
plt.title('ARIMA ForestFormation Forecast vs Actuals with Confidence Intervals')
plt.legend()
plt.grid(True)
plt.savefig('ARIMA ForestFormation Forecast vs Actuals with Confidence Intervals center.png')
plt.show()


##############################################################################################################

####    OilPalm

# Load the dataset
df = pd.read_excel(r"C:\Users\vs24904\OneDrive - University of Bristol\Documents\centerb.xlsx")
print(df)

# Identify unique BufferIDs and Area
unique_dates = df['Date'].unique()
unique_buffer_ids = df['BufferID'].unique()
unique_areas = df['Type'].unique()

# Create a complete DataFrame with all combinations
import itertools
all_combinations = pd.DataFrame(
    list(itertools.product(unique_dates, unique_buffer_ids, unique_areas)),
    columns=['Date', 'BufferID', 'Type'])
# Merge with the original DataFrame and fill missing values with 0
df_complete = pd.merge(all_combinations, df, on=['Date', 'BufferID', 'Type'], how='left').fillna({'Area_m2': 0})

# Time series analysis for a specific BufferID
area = 'OilPalm'# Example Area
df_filtered = df[(df['Type'] == area)].copy()

# Convert Date to numeric index for time series modeling
df_filtered['Date'] = pd.to_numeric(df_filtered['Date'])
df_filtered.sort_values('Date', inplace=True)
df_filtered.set_index('Date', inplace=True)

# Define the model with specified order (p, d, q)
model = ARIMA(df_filtered['Area_m2'], order=(1, 2, 1))
# Fit the model
model_fit = model.fit()
#print(model_fit.summary())

#Forecast
steps = 5
forecast = model_fit.forecast(steps=steps)
forecast_years = range(df_filtered.index[-1] + 1, df_filtered.index[-1] + 1 + steps)
forecast_df = pd.DataFrame({'Date': forecast_years, 'Forecasted_Area_m2': forecast})

#Plot and forecast
plt.figure(figsize=(12, 6))
plt.plot(df_filtered.index, df_filtered['Area_m2'], label='Observed', color='blue')
plt.plot(forecast_df['Date'], forecast_df['Forecasted_Area_m2'], label='Forecast', color='red', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Area (m2)')
plt.title(f'Time Series Forecast for Area: {area} in the Tailandia(Pará, Brazil)')
plt.legend()
plt.grid(True)
plt.savefig('OilPalm Forecast tailandia.png')
plt.show()
print(forecast_df)


# The area m2 of OilPalm habitat type in the site (Tailandia-Brazil), is expected to decrease in the next 5 years, but after checking 
# the forecast using "ARIMA" it seems to be increasing, I think that the discrepancy is due the OilPalm habitat decresing very recently,
# so the train data only shows an increasing (80%) and the test data is decreasing, but since the forescast is done with the first 80%, only shows
# a expected increase


#####################################################################################################################################

#####                                                   test if forecast is correct or good

# Split data into training and testing sets (e.g., last 5 years for testing)
train_size = int(len(df_filtered) * 0.8)
train_data, test_data = df_filtered.iloc[:train_size], df_filtered.iloc[train_size:]

# Fit SARIMA model on the training data
arima_model_train = ARIMA(train_data['Area_m2'], order=(1, 2, 1))
arima_fit_train = arima_model_train.fit()

# Forecast for the test period
forecast_test = arima_fit_train.get_forecast(steps=len(test_data))
forecasted_values = forecast_test.predicted_mean
confidence_intervals = forecast_test.conf_int()

# Evaluation metrics
mae = mean_absolute_error(test_data['Area_m2'], forecasted_values)
mse = mean_squared_error(test_data['Area_m2'], forecasted_values)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((test_data['Area_m2'] - forecasted_values) / test_data['Area_m2'])) * 100

# Print evaluation metrics
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAPE: {mape}%")

# Plot the forecast with confidence intervals
plt.figure(figsize=(12, 6))
plt.plot(train_data.index, train_data['Area_m2'], label='Training Data', color='blue')
plt.plot(test_data.index, test_data['Area_m2'], label='Actual Test Data', color='green')
plt.plot(test_data.index, forecasted_values, label='Forecast', color='red', linestyle='--')
plt.fill_between(test_data.index, confidence_intervals.iloc[:, 0], confidence_intervals.iloc[:, 1], color='pink', alpha=0.3)
plt.xlabel('Year')
plt.ylabel('Area (m²)')
plt.title('ARIMA Oilpalm Forecast vs Actuals with Confidence Intervals')
plt.legend()
plt.grid(True)
plt.savefig('ARIMA Oilpalm Forecast vs Actuals with Confidence Intervals center.png')
plt.show()



##############################################################################################################

####    Floodplains

# Load the dataset
df = pd.read_excel(r"C:\Users\vs24904\OneDrive - University of Bristol\Documents\centerb.xlsx")
# Identify unique BufferIDs and Area
unique_dates = df['Date'].unique()
unique_buffer_ids = df['BufferID'].unique()
unique_areas = df['Type'].unique()

# Create a complete DataFrame with all combinations
import itertools
all_combinations = pd.DataFrame(
    list(itertools.product(unique_dates, unique_buffer_ids, unique_areas)),
    columns=['Date', 'BufferID', 'Type'])
# Merge with the original DataFrame and fill missing values with 0
df_complete = pd.merge(all_combinations, df, on=['Date', 'BufferID', 'Type'], how='left').fillna({'Area_m2': 0})

# Time series analysis for a specific BufferID
area = 'Floodplains'# Example Area
df_filtered = df[(df['Type'] == area)].copy()

# Convert Date to numeric index for time series modeling
df_filtered['Date'] = pd.to_numeric(df_filtered['Date'])
df_filtered.sort_values('Date', inplace=True)
df_filtered.set_index('Date', inplace=True)

# Define the model with specified order (p, d, q)
model = ARIMA(df_filtered['Area_m2'], order=(1, 2, 1))
# Fit the model
model_fit = model.fit()
#print(model_fit.summary())

#Forecast
steps = 5
forecast = model_fit.forecast(steps=steps)
forecast_years = range(df_filtered.index[-1] + 1, df_filtered.index[-1] + 1 + steps)
forecast_df = pd.DataFrame({'Date': forecast_years, 'Forecasted_Area_m2': forecast})

#Plot and forecast
plt.figure(figsize=(12, 6))
plt.plot(df_filtered.index, df_filtered['Area_m2'], label='Observed', color='blue')
plt.plot(forecast_df['Date'], forecast_df['Forecasted_Area_m2'], label='Forecast', color='red', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Area (m2)')
plt.title(f'Time Series Forecast for Habitat: {area} in the Tailandia(Pará, Brazil)')
plt.legend()
plt.grid(True)
plt.savefig('Floodplains Forecast tailandia.png')
plt.show()
print(forecast_df)


# The area m2 of Floodplains habitat type in the site (Tailandia-Brazil), is expected to decrease in the next 5 years, after checking 
# the forecast using "ARIMA" I can conlude is accurate, since the forecast is inside the confidence interval 

#####################################################################################################################################

#####                                                   test if forecast is correct or good

# Split data into training and testing sets (e.g., last 5 years for testing)
train_size = int(len(df_filtered) * 0.8)
train_data, test_data = df_filtered.iloc[:train_size], df_filtered.iloc[train_size:]

# Fit SARIMA model on the training data
arima_model_train = ARIMA(train_data['Area_m2'], order=(1, 2, 1))
arima_fit_train = arima_model_train.fit()

# Forecast for the test period
forecast_test = arima_fit_train.get_forecast(steps=len(test_data))
forecasted_values = forecast_test.predicted_mean
confidence_intervals = forecast_test.conf_int()

# Evaluation metrics
mae = mean_absolute_error(test_data['Area_m2'], forecasted_values)
mse = mean_squared_error(test_data['Area_m2'], forecasted_values)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((test_data['Area_m2'] - forecasted_values) / test_data['Area_m2'])) * 100

# Print evaluation metrics
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAPE: {mape}%")

# Plot the forecast with confidence intervals
plt.figure(figsize=(12, 6))
plt.plot(train_data.index, train_data['Area_m2'], label='Training Data', color='blue')
plt.plot(test_data.index, test_data['Area_m2'], label='Actual Test Data', color='green')
plt.plot(test_data.index, forecasted_values, label='Forecast', color='red', linestyle='--')
plt.fill_between(test_data.index, confidence_intervals.iloc[:, 0], confidence_intervals.iloc[:, 1], color='pink', alpha=0.3)
plt.xlabel('Year')
plt.ylabel('Area (m²)')
plt.title('ARIMA Floodplains Forecast vs Actuals with Confidence Intervals')
plt.legend()
plt.grid(True)
plt.savefig('ARIMA Floodplains Forecast vs Actuals with Confidence Intervals')
plt.show()



###########################################################################################################################################

#########                                          LINEAR REGRESSION

# ForestFormation

# Load the data
data = pd.read_excel(r"C:\Users\vs24904\OneDrive - University of Bristol\Documents\centerb.xlsx")

# Filter data for 'ForestFormation'
forest_data = data[data['Type'] == 'ForestFormation'].copy()

# Encode 'BufferID' as numerical for plotting and regression
forest_data.loc[:, 'BufferID_encoded'] = pd.factorize(forest_data['Date'])[0]

# Prepare data for regression
X = sm.add_constant(forest_data['BufferID_encoded'])  # Add constant for intercept
y = forest_data['Area_m2']
# Fit the regression model using statsmodels
model = sm.OLS(y, X).fit()
# Get the model summary
model_summary = model.summary()

# Predict area for plotting the regression line
forest_data.loc[:, 'Predicted_Area'] = model.predict(X)

# Plot the scatter plot with regression line
plt.figure(figsize=(12, 6))
sns.scatterplot(x='BufferID_encoded', y='Area_m2', data=forest_data, label='Actual Data', color='blue')
plt.plot(forest_data['BufferID_encoded'], forest_data['Predicted_Area'], color='red', label='Regression Line')
plt.xticks(np.arange(len(forest_data['Date'].unique())), forest_data['Date'].unique(), rotation=45)
plt.xlabel('Year')
plt.ylabel('Area (m²)')
plt.title("Scatter Plot with Regression Line: Area by Year for 'ForestFormation' habitat")
plt.legend()
plt.grid(True)
plt.savefig("Scatter Plot with Regression Line: Area by Year for 'ForestFormation' habitat")
plt.show()
print(model_summary)

# The linear regression indicates a signficative increase of 1.528e+07 m2 every year (ForestFormation area). 
# The R-squared is high, so the variance can be explain by this model


##################################################################
# OilPalm

# Load the data
data = pd.read_excel(r"C:\Users\vs24904\OneDrive - University of Bristol\Documents\centerb.xlsx")

# Filter data for 'ForestFormation'
forest_data = data[data['Type'] == 'OilPalm'].copy()

# Encode 'BufferID' as numerical for plotting and regression
forest_data.loc[:, 'BufferID_encoded'] = pd.factorize(forest_data['Date'])[0]

# Prepare data for regression
X = sm.add_constant(forest_data['BufferID_encoded'])  # Add constant for intercept
y = forest_data['Area_m2']
# Fit the regression model using statsmodels
model = sm.OLS(y, X).fit()
# Get the model summary
model_summary = model.summary()

# Predict area for plotting the regression line
forest_data.loc[:, 'Predicted_Area'] = model.predict(X)

# Plot the scatter plot with regression line
plt.figure(figsize=(12, 6))
sns.scatterplot(x='BufferID_encoded', y='Area_m2', data=forest_data, label='Actual Data', color='blue')
plt.plot(forest_data['BufferID_encoded'], forest_data['Predicted_Area'], color='red', label='Regression Line')
plt.xticks(np.arange(len(forest_data['Date'].unique())), forest_data['Date'].unique(), rotation=45)
plt.xlabel('Year')
plt.ylabel('Area (m²)')
plt.title("Scatter Plot with Regression Line: Area by Year for 'Oilpalm' habitat")
plt.legend()
plt.grid(True)
plt.savefig("Scatter Plot with Regression Line: Area by Year for 'Oilpalm' habitat")
plt.show()
print(model_summary)
# The linear regression indicates a signficative increase of 9.968e+06 m2 every year (Oilpalm area). 
# The R-squared is not that high, so  the variance cannot be fully explain by this model



# Pasture

# Load the data
data = pd.read_excel(r"C:\Users\vs24904\OneDrive - University of Bristol\Documents\centerb.xlsx")

# Filter data for 'ForestFormation'
forest_data = data[data['Type'] == 'Pasture'].copy()

# Encode 'BufferID' as numerical for plotting and regression
forest_data.loc[:, 'BufferID_encoded'] = pd.factorize(forest_data['Date'])[0]

# Prepare data for regression
X = sm.add_constant(forest_data['BufferID_encoded'])  # Add constant for intercept
y = forest_data['Area_m2']
# Fit the regression model using statsmodels
model = sm.OLS(y, X).fit()
# Get the model summary
model_summary = model.summary()

# Predict area for plotting the regression line
forest_data.loc[:, 'Predicted_Area'] = model.predict(X)

# Plot the scatter plot with regression line
plt.figure(figsize=(12, 6))
sns.scatterplot(x='BufferID_encoded', y='Area_m2', data=forest_data, label='Actual Data', color='blue')
plt.plot(forest_data['BufferID_encoded'], forest_data['Predicted_Area'], color='red', label='Regression Line')
plt.xticks(np.arange(len(forest_data['Date'].unique())), forest_data['Date'].unique(), rotation=45)
plt.xlabel('Year')
plt.ylabel('Area (m²)')
plt.title("Scatter Plot with Regression Line: Area by Year for 'Pasture' habitat")
plt.legend()
plt.grid(True)
plt.savefig("Scatter Plot with Regression Line: Area by Year for 'Pasture' habitat")
plt.show()
print(model_summary)
# The linear regression indicates a signficative increase of 9.929e+06 m2 every year (Pasture area). 
# The R-squared is not that high, so  the variance cannot be fully explain by this model


###################################################
# Floodplain

# Load the data
data = pd.read_excel(r"C:\Users\vs24904\OneDrive - University of Bristol\Documents\centerb.xlsx")

# Filter data for 'ForestFormation'
forest_data = data[data['Type'] == 'Floodplains'].copy()

# Encode 'BufferID' as numerical for plotting and regression
forest_data.loc[:, 'BufferID_encoded'] = pd.factorize(forest_data['Date'])[0]

# Prepare data for regression
X = sm.add_constant(forest_data['BufferID_encoded'])  # Add constant for intercept
y = forest_data['Area_m2']
# Fit the regression model using statsmodels
model = sm.OLS(y, X).fit()
# Get the model summary
model_summary = model.summary()

# Predict area for plotting the regression line
forest_data.loc[:, 'Predicted_Area'] = model.predict(X)

# Plot the scatter plot with regression line
plt.figure(figsize=(12, 6))
sns.scatterplot(x='BufferID_encoded', y='Area_m2', data=forest_data, label='Actual Data', color='blue')
plt.plot(forest_data['BufferID_encoded'], forest_data['Predicted_Area'], color='red', label='Regression Line')
plt.xticks(np.arange(len(forest_data['Date'].unique())), forest_data['Date'].unique(), rotation=45)
plt.xlabel('Year')
plt.ylabel('Area (m²)')
plt.title("Scatter Plot with Regression Line: Area by Year for 'FloodPlains'habitat")
plt.legend()
plt.grid(True)
plt.savefig("Scatter Plot with Regression Line: Area by Year for 'Floodplains' habitat")
plt.show()
print(model_summary)

# The linear regression indicates a signficative decrease of Floofplains area of 3.537e+06 m2, every year. 
#The high R-squared indicates that the variance can be explain by this model
```

    C:\Users\vs24904\AppData\Local\anaconda32\Lib\site-packages\sklearn\cluster\_kmeans.py:1411: UserWarning: KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS=1.
      warnings.warn(
    


    
![png](output_2_1.png)
    



    
![png](output_2_2.png)
    


    ANOVA for Type:
    F-statistic: 3518.154323541889, p-value: 5.477677187641807e-157
    
    Tukey HSD Test for PLot:
                                Multiple Comparison of Means - Tukey HSD, FWER=0.05                             
    ============================================================================================================
            group1                group2            meandiff     p-adj       lower            upper       reject
    ------------------------------------------------------------------------------------------------------------
              Floodplains       ForestFormation   931911605.1337    0.0   901097892.8823   962725317.3851   True
              Floodplains      ForestPlantation  -337407734.4385    0.0  -368221446.6899  -306594022.1871   True
              Floodplains             Grassland  -331212437.3262    0.0  -362026149.5776  -300398725.0748   True
              Floodplains         MosaicofCrops  -335764870.5882    0.0  -366578582.8396  -304951158.3369   True
              Floodplains               OilPalm   162326971.4439    0.0   131513259.1925   193140683.6952   True
              Floodplains OtherNonVegetatedArea  -337603459.2513    0.0  -368417171.5027     -306789747.0   True
              Floodplains               Pasture   278428724.5989    0.0   247615012.3476   309242436.8503   True
              Floodplains     RiverLakeandOcean  -330588565.6684    0.0  -361402277.9198  -299774853.4171   True
              Floodplains      SavannaFormation  -338442160.7487    0.0     -369255873.0  -307628448.4973   True
              Floodplains              Soybeans  -338372746.8449    0.0  -369186459.0963  -307559034.5935   True
              Floodplains   UrbanInfrastructure  -333717815.2941    0.0  -364531527.5455  -302904103.0427   True
              Floodplains               Wetland  -289465026.0963    0.0  -320278738.3476  -258651313.8449   True
          ForestFormation      ForestPlantation -1269319339.5722    0.0 -1300133051.8236 -1238505627.3208   True
          ForestFormation             Grassland -1263124042.4599    0.0 -1293937754.7113 -1232310330.2085   True
          ForestFormation         MosaicofCrops -1267676475.7219    0.0 -1298490187.9733 -1236862763.4705   True
          ForestFormation               OilPalm  -769584633.6898    0.0  -800398345.9412  -738770921.4385   True
          ForestFormation OtherNonVegetatedArea  -1269515064.385    0.0 -1300328776.6364 -1238701352.1337   True
          ForestFormation               Pasture  -653482880.5348    0.0  -684296592.7861  -622669168.2834   True
          ForestFormation     RiverLakeandOcean -1262500170.8021    0.0 -1293313883.0535 -1231686458.5508   True
          ForestFormation      SavannaFormation -1270353765.8823    0.0 -1301167478.1337  -1239540053.631   True
          ForestFormation              Soybeans -1270284351.9786    0.0   -1301098064.23 -1239470639.7272   True
          ForestFormation   UrbanInfrastructure -1265629420.4278    0.0 -1296443132.6792 -1234815708.1764   True
          ForestFormation               Wetland -1221376631.2299    0.0 -1252190343.4813 -1190562918.9786   True
         ForestPlantation             Grassland     6195297.1123    1.0   -24618415.1391    37009009.3637  False
         ForestPlantation         MosaicofCrops     1642863.8503    1.0   -29170848.4011    32456576.1016  False
         ForestPlantation               OilPalm   499734705.8824    0.0    468920993.631   530548418.1337   True
         ForestPlantation OtherNonVegetatedArea     -195724.8128    1.0   -31009437.0642    30617987.4385  False
         ForestPlantation               Pasture   615836459.0374    0.0   585022746.7861   646650171.2888   True
         ForestPlantation     RiverLakeandOcean     6819168.7701 0.9999   -23994543.4813    37632881.0214  False
         ForestPlantation      SavannaFormation    -1034426.3102    1.0   -31848138.5615    29779285.9412  False
         ForestPlantation              Soybeans     -965012.4064    1.0   -31778724.6578     29848699.845  False
         ForestPlantation   UrbanInfrastructure     3689919.1444    1.0    -27123793.107    34503631.3958  False
         ForestPlantation               Wetland    47942708.3422    0.0    17128996.0909    78756420.5936   True
                Grassland         MosaicofCrops     -4552433.262    1.0   -35366145.5134    26261278.9893  False
                Grassland               OilPalm   493539408.7701    0.0   462725696.5187   524353121.0214   True
                Grassland OtherNonVegetatedArea    -6391021.9251    1.0   -37204734.1765    24422690.3262  False
                Grassland               Pasture   609641161.9251    0.0   578827449.6738   640454874.1765   True
                Grassland     RiverLakeandOcean      623871.6578    1.0   -30189840.5936    31437583.9091  False
                Grassland      SavannaFormation    -7229723.4225 0.9999   -38043435.6738    23583988.8289  False
                Grassland              Soybeans    -7160309.5187 0.9999   -37974021.7701    23653402.7327  False
                Grassland   UrbanInfrastructure    -2505377.9679    1.0   -33319090.2193    28308334.2835  False
                Grassland               Wetland    41747411.2299 0.0008    10933698.9786    72561123.4813   True
            MosaicofCrops               OilPalm   498091842.0321    0.0   467278129.7807   528905554.2835   True
            MosaicofCrops OtherNonVegetatedArea    -1838588.6631    1.0   -32652300.9145    28975123.5883  False
            MosaicofCrops               Pasture   614193595.1872    0.0   583379882.9358   645007307.4385   True
            MosaicofCrops     RiverLakeandOcean     5176304.9198    1.0   -25637407.3316    35990017.1712  False
            MosaicofCrops      SavannaFormation    -2677290.1604    1.0   -33491002.4118    28136422.0909  False
            MosaicofCrops              Soybeans    -2607876.2567    1.0   -33421588.5081    28205835.9947  False
            MosaicofCrops   UrbanInfrastructure     2047055.2941    1.0   -28766656.9573    32860767.5455  False
            MosaicofCrops               Wetland     46299844.492 0.0001    15486132.2406    77113556.7434   True
                  OilPalm OtherNonVegetatedArea  -499930430.6952    0.0  -530744142.9466  -469116718.4438   True
                  OilPalm               Pasture   116101753.1551    0.0    85288040.9037   146915465.4065   True
                  OilPalm     RiverLakeandOcean  -492915537.1123    0.0  -523729249.3637  -462101824.8609   True
                  OilPalm      SavannaFormation  -500769132.1925    0.0  -531582844.4439  -469955419.9411   True
                  OilPalm              Soybeans  -500699718.2888    0.0  -531513430.5401  -469886006.0374   True
                  OilPalm   UrbanInfrastructure   -496044786.738    0.0  -526858498.9893  -465231074.4866   True
                  OilPalm               Wetland  -451791997.5401    0.0  -482605709.7915  -420978285.2887   True
    OtherNonVegetatedArea               Pasture   616032183.8503    0.0   585218471.5989   646845896.1016   True
    OtherNonVegetatedArea     RiverLakeandOcean     7014893.5829 0.9999   -23798818.6685    37828605.8343  False
    OtherNonVegetatedArea      SavannaFormation     -838701.4973    1.0   -31652413.7487     29975010.754  False
    OtherNonVegetatedArea              Soybeans     -769287.5936    1.0    -31582999.845    30044424.6578  False
    OtherNonVegetatedArea   UrbanInfrastructure     3885643.9572    1.0   -26928068.2942    34699356.2086  False
    OtherNonVegetatedArea               Wetland    48138433.1551    0.0    17324720.9037    78952145.4065   True
                  Pasture     RiverLakeandOcean  -609017290.2674    0.0  -639831002.5188   -578203578.016   True
                  Pasture      SavannaFormation  -616870885.3476    0.0   -647684597.599  -586057173.0962   True
                  Pasture              Soybeans  -616801471.4438    0.0  -647615183.6952  -585987759.1925   True
                  Pasture   UrbanInfrastructure   -612146539.893    0.0  -642960252.1444  -581332827.6417   True
                  Pasture               Wetland  -567893750.6952    0.0  -598707462.9466  -537080038.4438   True
        RiverLakeandOcean      SavannaFormation    -7853595.0802 0.9997   -38667307.3316    22960117.1712  False
        RiverLakeandOcean              Soybeans    -7784181.1765 0.9997   -38597893.4278    23029531.0749  False
        RiverLakeandOcean   UrbanInfrastructure    -3129249.6257    1.0    -33942961.877    27684462.6257  False
        RiverLakeandOcean               Wetland    41123539.5722  0.001    10309827.3208    71937251.8236   True
         SavannaFormation              Soybeans       69413.9037    1.0   -30744298.3476    30883126.1551  False
         SavannaFormation   UrbanInfrastructure     4724345.4545    1.0   -26089366.7968    35538057.7059  False
         SavannaFormation               Wetland    48977134.6524    0.0     18163422.401    79790846.9038   True
                 Soybeans   UrbanInfrastructure     4654931.5508    1.0   -26158780.7006    35468643.8022  False
                 Soybeans               Wetland    48907720.7487    0.0    18094008.4973       79721433.0   True
      UrbanInfrastructure               Wetland    44252789.1979 0.0003    13439076.9465    75066501.4492   True
    ------------------------------------------------------------------------------------------------------------
    


    
![png](output_2_4.png)
    


    
    ANOVA for Year:
    F-statistic: 4.867223506960202e-30, p-value: 1.0
    
    Tukey HSD Test for Year:
           Multiple Comparison of Means - Tukey HSD, FWER=0.05        
    ==================================================================
    group1 group2 meandiff p-adj      lower          upper      reject
    ------------------------------------------------------------------
      2013   2014      0.0   1.0 -492736391.7088 492736391.7088  False
      2013   2015      0.0   1.0 -492736391.7088 492736391.7088  False
      2013   2016     -0.0   1.0 -492736391.7088 492736391.7088  False
      2013   2017     -0.0   1.0 -492736391.7088 492736391.7088  False
      2013   2018      0.0   1.0 -492736391.7088 492736391.7088  False
      2013   2019     -0.0   1.0 -492736391.7088 492736391.7088  False
      2013   2020     -0.0   1.0 -492736391.7088 492736391.7088  False
      2013   2021     -0.0   1.0 -492736391.7088 492736391.7088  False
      2013   2022     -0.0   1.0 -492736391.7088 492736391.7088  False
      2013   2023      0.0   1.0 -492736391.7088 492736391.7088  False
      2014   2015      0.0   1.0 -492736391.7088 492736391.7088  False
      2014   2016     -0.0   1.0 -492736391.7088 492736391.7088  False
      2014   2017     -0.0   1.0 -492736391.7088 492736391.7088  False
      2014   2018      0.0   1.0 -492736391.7088 492736391.7088  False
      2014   2019     -0.0   1.0 -492736391.7088 492736391.7088  False
      2014   2020     -0.0   1.0 -492736391.7088 492736391.7088  False
      2014   2021     -0.0   1.0 -492736391.7088 492736391.7088  False
      2014   2022     -0.0   1.0 -492736391.7088 492736391.7088  False
      2014   2023      0.0   1.0 -492736391.7088 492736391.7088  False
      2015   2016     -0.0   1.0 -492736391.7088 492736391.7088  False
      2015   2017     -0.0   1.0 -492736391.7088 492736391.7088  False
      2015   2018      0.0   1.0 -492736391.7088 492736391.7088  False
      2015   2019     -0.0   1.0 -492736391.7088 492736391.7088  False
      2015   2020     -0.0   1.0 -492736391.7088 492736391.7088  False
      2015   2021     -0.0   1.0 -492736391.7088 492736391.7088  False
      2015   2022     -0.0   1.0 -492736391.7088 492736391.7088  False
      2015   2023     -0.0   1.0 -492736391.7088 492736391.7088  False
      2016   2017     -0.0   1.0 -492736391.7088 492736391.7088  False
      2016   2018      0.0   1.0 -492736391.7088 492736391.7088  False
      2016   2019     -0.0   1.0 -492736391.7088 492736391.7088  False
      2016   2020     -0.0   1.0 -492736391.7088 492736391.7088  False
      2016   2021     -0.0   1.0 -492736391.7088 492736391.7088  False
      2016   2022      0.0   1.0 -492736391.7088 492736391.7088  False
      2016   2023      0.0   1.0 -492736391.7088 492736391.7088  False
      2017   2018      0.0   1.0 -492736391.7088 492736391.7088  False
      2017   2019      0.0   1.0 -492736391.7088 492736391.7088  False
      2017   2020      0.0   1.0 -492736391.7088 492736391.7088  False
      2017   2021      0.0   1.0 -492736391.7088 492736391.7088  False
      2017   2022      0.0   1.0 -492736391.7088 492736391.7088  False
      2017   2023      0.0   1.0 -492736391.7088 492736391.7088  False
      2018   2019     -0.0   1.0 -492736391.7088 492736391.7088  False
      2018   2020     -0.0   1.0 -492736391.7088 492736391.7088  False
      2018   2021     -0.0   1.0 -492736391.7088 492736391.7088  False
      2018   2022     -0.0   1.0 -492736391.7088 492736391.7088  False
      2018   2023     -0.0   1.0 -492736391.7088 492736391.7088  False
      2019   2020     -0.0   1.0 -492736391.7088 492736391.7088  False
      2019   2021      0.0   1.0 -492736391.7088 492736391.7088  False
      2019   2022      0.0   1.0 -492736391.7088 492736391.7088  False
      2019   2023      0.0   1.0 -492736391.7088 492736391.7088  False
      2020   2021      0.0   1.0 -492736391.7088 492736391.7088  False
      2020   2022      0.0   1.0 -492736391.7088 492736391.7088  False
      2020   2023      0.0   1.0 -492736391.7088 492736391.7088  False
      2021   2022      0.0   1.0 -492736391.7088 492736391.7088  False
      2021   2023      0.0   1.0 -492736391.7088 492736391.7088  False
      2022   2023      0.0   1.0 -492736391.7088 492736391.7088  False
    ------------------------------------------------------------------
    


    
![png](output_2_6.png)
    


    C:\Users\vs24904\AppData\Local\anaconda32\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:473: ValueWarning: An unsupported index was provided. As a result, forecasts cannot be generated. To use the model for forecasting, use one of the supported classes of index.
      self._init_dates(dates, freq)
    C:\Users\vs24904\AppData\Local\anaconda32\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:473: ValueWarning: An unsupported index was provided. As a result, forecasts cannot be generated. To use the model for forecasting, use one of the supported classes of index.
      self._init_dates(dates, freq)
    C:\Users\vs24904\AppData\Local\anaconda32\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:473: ValueWarning: An unsupported index was provided. As a result, forecasts cannot be generated. To use the model for forecasting, use one of the supported classes of index.
      self._init_dates(dates, freq)
    C:\Users\vs24904\AppData\Local\anaconda32\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
      return get_prediction_index(
    C:\Users\vs24904\AppData\Local\anaconda32\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:837: FutureWarning: No supported index is available. In the next version, calling this method in a model without a supported index will result in an exception.
      return get_prediction_index(
    


    
![png](output_2_8.png)
    


        Date  Forecasted_Area_m2
    11  2024        7.484357e+08
    12  2025        7.928831e+08
    13  2026        8.371145e+08
    14  2027        8.811696e+08
    15  2028        9.250809e+08
    MAE: 110300072.24824476
    MSE: 1.3826009542965576e+16
    RMSE: 117584053.09805228
    MAPE: nan%
    

    C:\Users\vs24904\AppData\Local\anaconda32\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:473: ValueWarning: An unsupported index was provided. As a result, forecasts cannot be generated. To use the model for forecasting, use one of the supported classes of index.
      self._init_dates(dates, freq)
    C:\Users\vs24904\AppData\Local\anaconda32\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:473: ValueWarning: An unsupported index was provided. As a result, forecasts cannot be generated. To use the model for forecasting, use one of the supported classes of index.
      self._init_dates(dates, freq)
    C:\Users\vs24904\AppData\Local\anaconda32\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:473: ValueWarning: An unsupported index was provided. As a result, forecasts cannot be generated. To use the model for forecasting, use one of the supported classes of index.
      self._init_dates(dates, freq)
    C:\Users\vs24904\AppData\Local\anaconda32\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
      return get_prediction_index(
    C:\Users\vs24904\AppData\Local\anaconda32\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:837: FutureWarning: No supported index is available. In the next version, calling this method in a model without a supported index will result in an exception.
      return get_prediction_index(
    


    
![png](output_2_11.png)
    


         Date BufferID                   Type       Area_m2
    0    2013   center                Wetland  5.228134e+07
    1    2013  center               Grassland  7.807045e+06
    2    2013  center                 Pasture  6.034558e+08
    3    2013  center     UrbanInfrastructure  4.611600e+06
    4    2013  center   OtherNonVegetatedArea  1.161000e+05
    ..    ...      ...                    ...           ...
    138  2023   center               Soybeans  7.334294e+05
    139  2023   center       SavannaFormation  1.215000e+05
    140  2023   center          MosaicofCrops  1.465211e+06
    141  2023   center            Floodplains  3.191143e+08
    142  2023   center       ForestPlantation  1.186906e+06
    
    [143 rows x 4 columns]
    

    C:\Users\vs24904\AppData\Local\anaconda32\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:473: ValueWarning: An unsupported index was provided. As a result, forecasts cannot be generated. To use the model for forecasting, use one of the supported classes of index.
      self._init_dates(dates, freq)
    C:\Users\vs24904\AppData\Local\anaconda32\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:473: ValueWarning: An unsupported index was provided. As a result, forecasts cannot be generated. To use the model for forecasting, use one of the supported classes of index.
      self._init_dates(dates, freq)
    C:\Users\vs24904\AppData\Local\anaconda32\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:473: ValueWarning: An unsupported index was provided. As a result, forecasts cannot be generated. To use the model for forecasting, use one of the supported classes of index.
      self._init_dates(dates, freq)
    C:\Users\vs24904\AppData\Local\anaconda32\Lib\site-packages\statsmodels\tsa\statespace\sarimax.py:978: UserWarning: Non-invertible starting MA parameters found. Using zeros as starting parameters.
      warn('Non-invertible starting MA parameters found.'
    C:\Users\vs24904\AppData\Local\anaconda32\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
      return get_prediction_index(
    C:\Users\vs24904\AppData\Local\anaconda32\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:837: FutureWarning: No supported index is available. In the next version, calling this method in a model without a supported index will result in an exception.
      return get_prediction_index(
    


    
![png](output_2_14.png)
    


        Date  Forecasted_Area_m2
    11  2024        1.165483e+09
    12  2025        1.137640e+09
    13  2026        1.109796e+09
    14  2027        1.081950e+09
    15  2028        1.054104e+09
    MAE: 31708128.052738428
    MSE: 1251817592692463.8
    RMSE: 35381034.364366226
    MAPE: nan%
    

    C:\Users\vs24904\AppData\Local\anaconda32\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:473: ValueWarning: An unsupported index was provided. As a result, forecasts cannot be generated. To use the model for forecasting, use one of the supported classes of index.
      self._init_dates(dates, freq)
    C:\Users\vs24904\AppData\Local\anaconda32\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:473: ValueWarning: An unsupported index was provided. As a result, forecasts cannot be generated. To use the model for forecasting, use one of the supported classes of index.
      self._init_dates(dates, freq)
    C:\Users\vs24904\AppData\Local\anaconda32\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:473: ValueWarning: An unsupported index was provided. As a result, forecasts cannot be generated. To use the model for forecasting, use one of the supported classes of index.
      self._init_dates(dates, freq)
    C:\Users\vs24904\AppData\Local\anaconda32\Lib\site-packages\statsmodels\tsa\statespace\sarimax.py:866: UserWarning: Too few observations to estimate starting parameters for ARMA and trend. All parameters except for variances will be set to zeros.
      warn('Too few observations to estimate starting parameters%s.'
    C:\Users\vs24904\AppData\Local\anaconda32\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
      return get_prediction_index(
    C:\Users\vs24904\AppData\Local\anaconda32\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:837: FutureWarning: No supported index is available. In the next version, calling this method in a model without a supported index will result in an exception.
      return get_prediction_index(
    


    
![png](output_2_17.png)
    


         Date BufferID                   Type       Area_m2
    0    2013   center                Wetland  5.228134e+07
    1    2013  center               Grassland  7.807045e+06
    2    2013  center                 Pasture  6.034558e+08
    3    2013  center     UrbanInfrastructure  4.611600e+06
    4    2013  center   OtherNonVegetatedArea  1.161000e+05
    ..    ...      ...                    ...           ...
    138  2023   center               Soybeans  7.334294e+05
    139  2023   center       SavannaFormation  1.215000e+05
    140  2023   center          MosaicofCrops  1.465211e+06
    141  2023   center            Floodplains  3.191143e+08
    142  2023   center       ForestPlantation  1.186906e+06
    
    [143 rows x 4 columns]
    

    C:\Users\vs24904\AppData\Local\anaconda32\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:473: ValueWarning: An unsupported index was provided. As a result, forecasts cannot be generated. To use the model for forecasting, use one of the supported classes of index.
      self._init_dates(dates, freq)
    C:\Users\vs24904\AppData\Local\anaconda32\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:473: ValueWarning: An unsupported index was provided. As a result, forecasts cannot be generated. To use the model for forecasting, use one of the supported classes of index.
      self._init_dates(dates, freq)
    C:\Users\vs24904\AppData\Local\anaconda32\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:473: ValueWarning: An unsupported index was provided. As a result, forecasts cannot be generated. To use the model for forecasting, use one of the supported classes of index.
      self._init_dates(dates, freq)
    C:\Users\vs24904\AppData\Local\anaconda32\Lib\site-packages\statsmodels\tsa\statespace\sarimax.py:966: UserWarning: Non-stationary starting autoregressive parameters found. Using zeros as starting parameters.
      warn('Non-stationary starting autoregressive parameters'
    C:\Users\vs24904\AppData\Local\anaconda32\Lib\site-packages\statsmodels\tsa\statespace\sarimax.py:978: UserWarning: Non-invertible starting MA parameters found. Using zeros as starting parameters.
      warn('Non-invertible starting MA parameters found.'
    C:\Users\vs24904\AppData\Local\anaconda32\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
      return get_prediction_index(
    C:\Users\vs24904\AppData\Local\anaconda32\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:837: FutureWarning: No supported index is available. In the next version, calling this method in a model without a supported index will result in an exception.
      return get_prediction_index(
    


    
![png](output_2_20.png)
    


        Date  Forecasted_Area_m2
    11  2024        5.131608e+08
    12  2025        5.044460e+08
    13  2026        4.957316e+08
    14  2027        4.870173e+08
    15  2028        4.783029e+08
    MAE: 8671598.79203401
    MSE: 131579802076476.12
    RMSE: 11470823.94932797
    MAPE: nan%
    

    C:\Users\vs24904\AppData\Local\anaconda32\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:473: ValueWarning: An unsupported index was provided. As a result, forecasts cannot be generated. To use the model for forecasting, use one of the supported classes of index.
      self._init_dates(dates, freq)
    C:\Users\vs24904\AppData\Local\anaconda32\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:473: ValueWarning: An unsupported index was provided. As a result, forecasts cannot be generated. To use the model for forecasting, use one of the supported classes of index.
      self._init_dates(dates, freq)
    C:\Users\vs24904\AppData\Local\anaconda32\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:473: ValueWarning: An unsupported index was provided. As a result, forecasts cannot be generated. To use the model for forecasting, use one of the supported classes of index.
      self._init_dates(dates, freq)
    C:\Users\vs24904\AppData\Local\anaconda32\Lib\site-packages\statsmodels\tsa\statespace\sarimax.py:966: UserWarning: Non-stationary starting autoregressive parameters found. Using zeros as starting parameters.
      warn('Non-stationary starting autoregressive parameters'
    C:\Users\vs24904\AppData\Local\anaconda32\Lib\site-packages\statsmodels\tsa\statespace\sarimax.py:978: UserWarning: Non-invertible starting MA parameters found. Using zeros as starting parameters.
      warn('Non-invertible starting MA parameters found.'
    C:\Users\vs24904\AppData\Local\anaconda32\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
      return get_prediction_index(
    C:\Users\vs24904\AppData\Local\anaconda32\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:837: FutureWarning: No supported index is available. In the next version, calling this method in a model without a supported index will result in an exception.
      return get_prediction_index(
    


    
![png](output_2_23.png)
    


    C:\Users\vs24904\AppData\Local\anaconda32\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:473: ValueWarning: An unsupported index was provided. As a result, forecasts cannot be generated. To use the model for forecasting, use one of the supported classes of index.
      self._init_dates(dates, freq)
    C:\Users\vs24904\AppData\Local\anaconda32\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:473: ValueWarning: An unsupported index was provided. As a result, forecasts cannot be generated. To use the model for forecasting, use one of the supported classes of index.
      self._init_dates(dates, freq)
    C:\Users\vs24904\AppData\Local\anaconda32\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:473: ValueWarning: An unsupported index was provided. As a result, forecasts cannot be generated. To use the model for forecasting, use one of the supported classes of index.
      self._init_dates(dates, freq)
    C:\Users\vs24904\AppData\Local\anaconda32\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
      return get_prediction_index(
    C:\Users\vs24904\AppData\Local\anaconda32\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:837: FutureWarning: No supported index is available. In the next version, calling this method in a model without a supported index will result in an exception.
      return get_prediction_index(
    


    
![png](output_2_25.png)
    


        Date  Forecasted_Area_m2
    11  2024        3.137402e+08
    12  2025        3.083666e+08
    13  2026        3.029931e+08
    14  2027        2.976197e+08
    15  2028        2.922462e+08
    MAE: 5666639.169054608
    MSE: 33763819493611.793
    RMSE: 5810664.2902177535
    MAPE: nan%
    

    C:\Users\vs24904\AppData\Local\anaconda32\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:473: ValueWarning: An unsupported index was provided. As a result, forecasts cannot be generated. To use the model for forecasting, use one of the supported classes of index.
      self._init_dates(dates, freq)
    C:\Users\vs24904\AppData\Local\anaconda32\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:473: ValueWarning: An unsupported index was provided. As a result, forecasts cannot be generated. To use the model for forecasting, use one of the supported classes of index.
      self._init_dates(dates, freq)
    C:\Users\vs24904\AppData\Local\anaconda32\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:473: ValueWarning: An unsupported index was provided. As a result, forecasts cannot be generated. To use the model for forecasting, use one of the supported classes of index.
      self._init_dates(dates, freq)
    C:\Users\vs24904\AppData\Local\anaconda32\Lib\site-packages\statsmodels\tsa\statespace\sarimax.py:978: UserWarning: Non-invertible starting MA parameters found. Using zeros as starting parameters.
      warn('Non-invertible starting MA parameters found.'
    C:\Users\vs24904\AppData\Local\anaconda32\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
      return get_prediction_index(
    C:\Users\vs24904\AppData\Local\anaconda32\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:837: FutureWarning: No supported index is available. In the next version, calling this method in a model without a supported index will result in an exception.
      return get_prediction_index(
    


    
![png](output_2_28.png)
    


    C:\Users\vs24904\AppData\Local\anaconda32\Lib\site-packages\scipy\stats\_axis_nan_policy.py:430: UserWarning: `kurtosistest` p-value may be inaccurate with fewer than 20 observations; only n=11 observations were given.
      return hypotest_fun_in(*args, **kwds)
    


    
![png](output_2_30.png)
    


                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                Area_m2   R-squared:                       0.939
    Model:                            OLS   Adj. R-squared:                  0.932
    Method:                 Least Squares   F-statistic:                     138.8
    Date:                Tue, 14 Jan 2025   Prob (F-statistic):           9.02e-07
    Time:                        12:51:59   Log-Likelihood:                -195.19
    No. Observations:                  11   AIC:                             394.4
    Df Residuals:                       9   BIC:                             395.2
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ====================================================================================
                           coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------------
    const             1.347e+09   7.68e+06    175.484      0.000    1.33e+09    1.36e+09
    BufferID_encoded -1.528e+07    1.3e+06    -11.780      0.000   -1.82e+07   -1.23e+07
    ==============================================================================
    Omnibus:                        5.232   Durbin-Watson:                   1.340
    Prob(Omnibus):                  0.073   Jarque-Bera (JB):                2.380
    Skew:                          -1.125   Prob(JB):                        0.304
    Kurtosis:                       3.357   Cond. No.                         11.3
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    

    C:\Users\vs24904\AppData\Local\anaconda32\Lib\site-packages\scipy\stats\_axis_nan_policy.py:430: UserWarning: `kurtosistest` p-value may be inaccurate with fewer than 20 observations; only n=11 observations were given.
      return hypotest_fun_in(*args, **kwds)
    


    
![png](output_2_33.png)
    


                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                Area_m2   R-squared:                       0.772
    Model:                            OLS   Adj. R-squared:                  0.747
    Method:                 Least Squares   F-statistic:                     30.53
    Date:                Tue, 14 Jan 2025   Prob (F-statistic):           0.000368
    Time:                        12:51:59   Log-Likelihood:                -198.82
    No. Observations:                  11   AIC:                             401.6
    Df Residuals:                       9   BIC:                             402.4
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ====================================================================================
                           coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------------
    const             4.511e+08   1.07e+07     42.268      0.000    4.27e+08    4.75e+08
    BufferID_encoded  9.968e+06    1.8e+06      5.526      0.000    5.89e+06     1.4e+07
    ==============================================================================
    Omnibus:                        1.989   Durbin-Watson:                   0.403
    Prob(Omnibus):                  0.370   Jarque-Bera (JB):                1.051
    Skew:                          -0.408   Prob(JB):                        0.591
    Kurtosis:                       1.725   Cond. No.                         11.3
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    

    C:\Users\vs24904\AppData\Local\anaconda32\Lib\site-packages\scipy\stats\_axis_nan_policy.py:430: UserWarning: `kurtosistest` p-value may be inaccurate with fewer than 20 observations; only n=11 observations were given.
      return hypotest_fun_in(*args, **kwds)
    


    
![png](output_2_36.png)
    


                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                Area_m2   R-squared:                       0.664
    Model:                            OLS   Adj. R-squared:                  0.627
    Method:                 Least Squares   F-statistic:                     17.78
    Date:                Tue, 14 Jan 2025   Prob (F-statistic):            0.00225
    Time:                        12:51:59   Log-Likelihood:                -201.75
    No. Observations:                  11   AIC:                             407.5
    Df Residuals:                       9   BIC:                             408.3
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ====================================================================================
                           coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------------
    const             5.674e+08   1.39e+07     40.728      0.000    5.36e+08    5.99e+08
    BufferID_encoded  9.929e+06   2.35e+06      4.217      0.002     4.6e+06    1.53e+07
    ==============================================================================
    Omnibus:                        0.401   Durbin-Watson:                   0.842
    Prob(Omnibus):                  0.818   Jarque-Bera (JB):                0.492
    Skew:                           0.278   Prob(JB):                        0.782
    Kurtosis:                       2.126   Cond. No.                         11.3
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    

    C:\Users\vs24904\AppData\Local\anaconda32\Lib\site-packages\scipy\stats\_axis_nan_policy.py:430: UserWarning: `kurtosistest` p-value may be inaccurate with fewer than 20 observations; only n=11 observations were given.
      return hypotest_fun_in(*args, **kwds)
    


    
![png](output_2_39.png)
    


                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                Area_m2   R-squared:                       0.942
    Model:                            OLS   Adj. R-squared:                  0.935
    Method:                 Least Squares   F-statistic:                     145.8
    Date:                Tue, 14 Jan 2025   Prob (F-statistic):           7.30e-07
    Time:                        12:52:00   Log-Likelihood:                -178.82
    No. Observations:                  11   AIC:                             361.6
    Df Residuals:                       9   BIC:                             362.4
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ====================================================================================
                           coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------------
    const             3.563e+08   1.73e+06    205.574      0.000    3.52e+08     3.6e+08
    BufferID_encoded -3.537e+06   2.93e+05    -12.074      0.000    -4.2e+06   -2.87e+06
    ==============================================================================
    Omnibus:                        0.179   Durbin-Watson:                   1.005
    Prob(Omnibus):                  0.914   Jarque-Bera (JB):                0.258
    Skew:                           0.228   Prob(JB):                        0.879
    Kurtosis:                       2.403   Cond. No.                         11.3
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    
