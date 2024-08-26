#!/usr/bin/env python
# coding: utf-8

# **Python code for Dividing the historical data into a (75:25) Train-Test data split for Time Series XGBoost Forecasting Model**

# In[8]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'data' is the dictionary containing the historical data
data = {
    'Month': ['Apr-19', 'May-19', 'Jun-19', 'Jul-19', 'Aug-19', 'Sep-19', 'Oct-19', 'Nov-19', 'Dec-19',
              'Jan-20', 'Feb-20', 'Mar-20', 'Apr-20', 'May-20', 'Jun-20', 'Jul-20', 'Aug-20', 'Sep-20',
              'Oct-20', 'Nov-20', 'Dec-20', 'Jan-21', 'Feb-21', 'Mar-21', 'Apr-21', 'May-21', 'Jun-21',
              'Jul-21', 'Aug-21', 'Sep-21', 'Oct-21', 'Nov-21', 'Dec-21', 'Jan-22', 'Feb-22', 'Mar-22',
              'Apr-22', 'May-22', 'Jun-22', 'Jul-22', 'Aug-22', 'Sep-22', 'Oct-22', 'Nov-22', 'Dec-22',
              'Jan-23', 'Feb-23', 'Mar-23', 'Apr-23', 'May-23', 'Jun-23', 'Jul-23', 'Aug-23', 'Sep-23',
              'Oct-23', 'Nov-23', 'Dec-23', 'Jan-24', 'Feb-24', 'Mar-24'],
    'HDPE film': [1166, 1182, 1166, 1102, 1074, 1027, 1037, 988, 967, 958, 958, 819, 817, 757, 809, 878,
                  914, 911, 882, 948, 958, 1140, 1435, 1744, 1858, 1523, 1465, 1350, 1354, 1413, 1388,
                  1595, 1553, 1543, 1547, 1762, 1968, 1729, 1708, 1577, 1443, 1355, 1388, 1393, 1339,
                  1313, 1299, 1334, 1229, 1122, 1065, 1016, 1118, 1158, 1175, 1076, 1063, 1186, 1265, 1280],
    'LDPE film': [1096, 1120, 1105, 1047, 1029, 964, 985, 953, 952, 944, 953, 836, 828, 762, 840, 917,
                  935, 918, 912, 1024, 1051, 1288, 1562, 1920, 2082, 1900, 1816, 1710, 1748, 1767, 1486,
                  1914, 1872, 1851, 1845, 2033, 2225, 1932, 1893, 1660, 1547, 1407, 1486, 1431, 1394,
                  1352, 1330, 1349, 1261, 1095, 979, 986, 1090, 1162, 1221, 1067, 1059, 1290, 1351, 1060],
    'PET': [170, 200, 200, 180, 200, 160, 190, 150, 210, 130, 170, 230, 220, 210, 140, 220, 160, 90, 70,
            180, 300, 160, 100, 80, 140, 70, 70, 200, 160, 180, 290, 90, 70, 0, 360, 200, 140, 250, 10,
            90, 360, 450, 400, 400, 410, 470, 340, 220, 230, 310, 300, 310, 300, 250, 320, 190, 200, 260, 260, 400],
    'PP homo-polymer fiber': [1203, 1223, 1197, 1070, 1055, 1008, 1019, 979, 995, 968, 977, 917, 917,
                              847, 893, 938, 949, 936, 921, 997, 1012, 1194, 1434, 1844, 2066, 1962,
                              1630, 1538, 1631, 1631, 1275, 1712, 1697, 1676, 1654, 1860, 2043, 1790,
                              1648, 1466, 1248, 1191, 1275, 1265, 1220, 1227, 1250, 1275, 1221, 1121,
                              1062, 1019, 1065, 1141, 1145, 1057, 1029, 1237, 1254, 1250],
    'rPET': [360, 360, 350, 330, 320, 340, 340, 340, 400, 370, 330, 310, 380, 370, 360, 340, 290, 300,
             260, 300, 340, 380, 380, 380, 380, 390, 430, 480, 440, 420, 520, 520, 480, 720, 350, 390,
             360, 600, 820, 720, 560, 480, 490, 550, 700, 610, 700, 550, 550, 570, 480, 450, 460, 430,
             480, 450, 410, 380, 420, 410],
    'rLDPE Film': [550, 580, 520, 540, 500, 480, 510, 490, 510, 480, 470, 530, 530, 520, 500, 500, 430,
                   430, 470, 470, 450, 510, 470, 490, 530, 490, 470, 450, 420, 380, 340, 430, 470, 480,
                   430, 650, 640, 640, 790, 610, 500, 560, 600, 520, 490, 480, 470, 390, 410, 730, 180,
                   290, 180, 180, 290, 360, 410, 560, 550, 470],
    'rPP': [590, 600, 600, 580, 570, 570, 550, 570, 570, 560, 530, 520, 510, 530, 520, 460, 450, 490,
            480, 480, 510, 510, 520, 530, 550, 600, 620, 560, 570, 620, 650, 730, 630, 650, 530, 790,
            780, 670, 860, 830, 750, 790, 900, 840, 770, 780, 760, 740, 750, 740, 740, 700, 650, 640,
            640, 600, 580, 600, 620, 730],
    'rHDPE Film': [570, 580, 600, 620, 600, 560, 570, 600, 600, 570, 550, 520, 520, 510, 490, 510, 490,
                   490, 490, 490, 490, 500, 520, 580, 620, 620, 640, 600, 630, 620, 570, 580, 600, 750,
                   590, 710, 630, 740, 1010, 1020, 780, 770, 760, 720, 650, 760, 750, 730, 660, 640,
                   670, 630, 610, 610, 640, 580, 580, 580, 570, 520]
}

# Convert 'Month' column to datetime and set it as index
df = pd.DataFrame(data)
df['Month'] = pd.to_datetime(df['Month'], format='%b-%y')
df.set_index('Month', inplace=True)

# Split the data into training (75%) and validation (25%) sets
train_size = int(0.75 * len(df))
train_data = df.iloc[:train_size]
validation_data = df.iloc[train_size:]

# Set the color palette and style for better visualization
sns.set(style="whitegrid")

# Plotting the training and testing data using seaborn for better visualization
for compound in df.columns:
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=train_data[compound], label='Training Data', color='blue', linewidth=2.5)
    sns.lineplot(data=validation_data[compound], label='Testing Data', color='red', linewidth=2.5)
    plt.title(f'Training and Testing Data for {compound}', fontsize=16)
    plt.xlabel('Month', fontsize=14)
    plt.ylabel('Price', fontsize=14)
    plt.legend(title='Data Type', title_fontsize='13', fontsize='12')
    plt.show()


# **Specifications Guide European Chemicals (2024). [Polymer Market Chemical Guidelines]. S&P Global Commodity Insights.**

# In[16]:


import pandas as pd

# Define the data
data = {
    'Polymer Compounds': ['HDPE film', 'LDPE film', 'PET', 'PP homo-polymer fiber'],
    'Delivery Period (In days)': ['3-30 days', '3-30 days', '3-30 days', '3-30 days']
}

# Create DataFrame
df = pd.DataFrame(data)

# Print the DataFrame
print(df)


# **Python Code for Graphviz installer package installation**

# In[4]:


pip install graphviz


# **Python Code generating Relevant Statistics from Historical Data**

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'data' is the dictionary containing the historical data
data = {
    'Month': ['Apr-19', 'May-19', 'Jun-19', 'Jul-19', 'Aug-19', 'Sep-19', 'Oct-19', 'Nov-19', 'Dec-19',
              'Jan-20', 'Feb-20', 'Mar-20', 'Apr-20', 'May-20', 'Jun-20', 'Jul-20', 'Aug-20', 'Sep-20',
              'Oct-20', 'Nov-20', 'Dec-20', 'Jan-21', 'Feb-21', 'Mar-21', 'Apr-21', 'May-21', 'Jun-21',
              'Jul-21', 'Aug-21', 'Sep-21', 'Oct-21', 'Nov-21', 'Dec-21', 'Jan-22', 'Feb-22', 'Mar-22',
              'Apr-22', 'May-22', 'Jun-22', 'Jul-22', 'Aug-22', 'Sep-22', 'Oct-22', 'Nov-22', 'Dec-22',
              'Jan-23', 'Feb-23', 'Mar-23', 'Apr-23', 'May-23', 'Jun-23', 'Jul-23', 'Aug-23', 'Sep-23',
              'Oct-23', 'Nov-23', 'Dec-23', 'Jan-24', 'Feb-24', 'Mar-24'],
    'HDPE film': [1166, 1182, 1166, 1102, 1074, 1027, 1037, 988, 967, 958, 958, 819, 817, 757, 809, 878,
                  914, 911, 882, 948, 958, 1140, 1435, 1744, 1858, 1523, 1465, 1350, 1354, 1413, 1388,
                  1595, 1553, 1543, 1547, 1762, 1968, 1729, 1708, 1577, 1443, 1355, 1388, 1393, 1339,
                  1313, 1299, 1334, 1229, 1122, 1065, 1016, 1118, 1158, 1175, 1076, 1063, 1186, 1265, 1280],
    'LDPE film': [1096, 1120, 1105, 1047, 1029, 964, 985, 953, 952, 944, 953, 836, 828, 762, 840, 917,
                  935, 918, 912, 1024, 1051, 1288, 1562, 1920, 2082, 1900, 1816, 1710, 1748, 1767, 1486,
                  1914, 1872, 1851, 1845, 2033, 2225, 1932, 1893, 1660, 1547, 1407, 1486, 1431, 1394,
                  1352, 1330, 1349, 1261, 1095, 979, 986, 1090, 1162, 1221, 1067, 1059, 1290, 1351, 1060],
    'PET': [170, 200, 200, 180, 200, 160, 190, 150, 210, 130, 170, 230, 220, 210, 140, 220, 160, 90, 70,
            180, 300, 160, 100, 80, 140, 70, 70, 200, 160, 180, 290, 90, 70, 0, 360, 200, 140, 250, 10,
            90, 360, 450, 400, 400, 410, 470, 340, 220, 230, 310, 300, 310, 300, 250, 320, 190, 200, 260, 260, 400],
    'PP homo-polymer fiber': [1203, 1223, 1197, 1070, 1055, 1008, 1019, 979, 995, 968, 977, 917, 917,
                              847, 893, 938, 949, 936, 921, 997, 1012, 1194, 1434, 1844, 2066, 1962,
                              1630, 1538, 1631, 1631, 1275, 1712, 1697, 1676, 1654, 1860, 2043, 1790,
                              1648, 1466, 1248, 1191, 1275, 1265, 1220, 1227, 1250, 1275, 1221, 1121,
                              1062, 1019, 1065, 1141, 1145, 1057, 1029, 1237, 1254, 1250],
    'rPET': [360, 360, 350, 330, 320, 340, 340, 340, 400, 370, 330, 310, 380, 370, 360, 340, 290, 300,
             260, 300, 340, 380, 380, 380, 380, 390, 430, 480, 440, 420, 520, 520, 480, 720, 350, 390,
             360, 600, 820, 720, 560, 480, 490, 550, 700, 610, 700, 550, 550, 570, 480, 450, 460, 430,
             480, 450, 410, 380, 420, 410],
    'rLDPE Film': [550, 580, 520, 540, 500, 480, 510, 490, 510, 480, 470, 530, 530, 520, 500, 500, 430,
                   430, 470, 470, 450, 510, 470, 490, 530, 490, 470, 450, 420, 380, 340, 430, 470, 480,
                   430, 650, 640, 640, 790, 610, 500, 560, 600, 520, 490, 480, 470, 390, 410, 730, 180,
                   290, 180, 180, 290, 360, 410, 560, 550, 470],
    'rPP': [590, 600, 600, 580, 570, 570, 550, 570, 570, 560, 530, 520, 510, 530, 520, 460, 450, 490,
            480, 480, 510, 510, 520, 530, 550, 600, 620, 560, 570, 620, 650, 730, 630, 650, 530, 790,
            780, 670, 860, 830, 750, 790, 900, 840, 770, 780, 760, 740, 750, 740, 740, 700, 650, 640,
            640, 600, 580, 600, 620, 730],
    'rHDPE Film': [570, 580, 600, 620, 600, 560, 570, 600, 600, 570, 550, 520, 520, 510, 490, 510, 490,
                   490, 490, 490, 490, 500, 520, 580, 620, 620, 640, 600, 630, 620, 570, 580, 600, 750,
                   590, 710, 630, 740, 1010, 1020, 780, 770, 760, 720, 650, 760, 750, 730, 660, 640,
                   670, 630, 610, 610, 640, 580, 580, 580, 570, 520]
}

# Convert 'Month' column to datetime and set it as index
df = pd.DataFrame(data)
df['Month'] = pd.to_datetime(df['Month'], format='%b-%y')
df.set_index('Month', inplace=True)


# In[2]:


df.describe()


# In[3]:


df.head()


# In[4]:


df.tail()


# In[8]:


df


# **Python Code for Monte Carlo Forward Price Simulation model**

# In[10]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Set the aesthetic parameters in one step
sns.set(style="whitegrid")

# Define the data
data = {
    'Month': ['Apr-19', 'May-19', 'Jun-19', 'Jul-19', 'Aug-19', 'Sep-19', 'Oct-19', 'Nov-19', 'Dec-19',
              'Jan-20', 'Feb-20', 'Mar-20', 'Apr-20', 'May-20', 'Jun-20', 'Jul-20', 'Aug-20', 'Sep-20',
              'Oct-20', 'Nov-20', 'Dec-20', 'Jan-21', 'Feb-21', 'Mar-21', 'Apr-21', 'May-21', 'Jun-21',
              'Jul-21', 'Aug-21', 'Sep-21', 'Oct-21', 'Nov-21', 'Dec-21', 'Jan-22', 'Feb-22', 'Mar-22',
              'Apr-22', 'May-22', 'Jun-22', 'Jul-22', 'Aug-22', 'Sep-22', 'Oct-22', 'Nov-22', 'Dec-22',
              'Jan-23', 'Feb-23', 'Mar-23', 'Apr-23', 'May-23', 'Jun-23', 'Jul-23', 'Aug-23', 'Sep-23',
              'Oct-23', 'Nov-23', 'Dec-23', 'Jan-24', 'Feb-24', 'Mar-24'],
    'HDPE film': [1166, 1182, 1166, 1102, 1074, 1027, 1037, 988, 967, 958, 958, 819, 817, 757, 809, 878,
                  914, 911, 882, 948, 958, 1140, 1435, 1744, 1858, 1523, 1465, 1350, 1354, 1413, 1388,
                  1595, 1553, 1543, 1547, 1762, 1968, 1729, 1708, 1577, 1443, 1355, 1388, 1393, 1339,
                  1313, 1299, 1334, 1229, 1122, 1065, 1016, 1118, 1158, 1175, 1076, 1063, 1186, 1265, 1280],
    'LDPE film': [1096, 1120, 1105, 1047, 1029, 964, 985, 953, 952, 944, 953, 836, 828, 762, 840, 917,
                  935, 918, 912, 1024, 1051, 1288, 1562, 1920, 2082, 1900, 1816, 1710, 1748, 1767, 1486,
                  1914, 1872, 1851, 1845, 2033, 2225, 1932, 1893, 1660, 1547, 1407, 1486, 1431, 1394,
                  1352, 1330, 1349, 1261, 1095, 979, 986, 1090, 1162, 1221, 1067, 1059, 1290, 1351, 1060],
    'PET': [170, 200, 200, 180, 200, 160, 190, 150, 210, 130, 170, 230, 220, 210, 140, 220, 160, 90, 70,
            180, 300, 160, 100, 80, 140, 70, 70, 200, 160, 180, 290, 90, 70, 0, 360, 200, 140, 250, 10,
            90, 360, 450, 400, 400, 410, 470, 340, 220, 230, 310, 300, 310, 300, 250, 320, 190, 200, 260, 260, 400],
    'PP homo-polymer fiber': [1203, 1223, 1197, 1070, 1055, 1008, 1019, 979, 995, 968, 977, 917, 917,
                              847, 893, 938, 949, 936, 921, 997, 1012, 1194, 1434, 1844, 2066, 1962,
                              1630, 1538, 1631, 1631, 1275, 1712, 1697, 1676, 1654, 1860, 2043, 1790,
                              1648, 1466, 1248, 1191, 1275, 1265, 1220, 1227, 1250, 1275, 1221, 1121,
                              1062, 1019, 1065, 1141, 1145, 1057, 1029, 1237, 1254, 1250],
    'rPET': [360, 360, 350, 330, 320, 340, 340, 340, 400, 370, 330, 310, 380, 370, 360, 340, 290, 300, 260, 300, 340, 380, 380, 380, 380, 390, 430, 480, 440, 420, 520, 520, 480, 720, 350, 390, 360, 600, 820, 720, 560, 480, 490, 550, 700, 610, 700, 550, 550, 570, 480, 450, 460, 430, 480, 450, 410, 380, 420, 410],
    'rLDPE Film': [550, 580, 520, 540, 500, 480, 510, 490, 510, 480, 470, 530, 530, 520, 500, 500, 430, 430, 470, 470, 450, 510, 470, 490, 530, 490, 470, 450, 420, 380, 340, 430, 470, 480, 430, 650, 640, 640, 790, 610, 500, 560, 600, 520, 490, 480, 470, 390, 410, 730, 180, 290, 180, 180, 290, 360, 410, 560, 550, 470],
    'rPP': [590, 600, 600, 580, 570, 570, 550, 570, 570, 560, 530, 520, 510, 530, 520, 460, 450, 490, 480, 480, 510, 510, 520, 530, 550, 600, 620, 560, 570, 620, 650, 730, 630, 650, 530, 790, 780, 670, 860, 830, 750, 790, 900, 840, 770, 780, 760, 740, 750, 740, 740, 700, 650, 640, 640, 600, 580, 600, 620, 730],
    'rHDPE Film': [570, 580, 600, 620, 600, 560, 570, 600, 600, 570, 550, 520, 520, 510, 490, 510, 490, 490, 490, 490, 490, 500, 520, 580, 620, 620, 640, 600, 630, 620, 570, 580, 600, 750, 590, 710, 630, 740, 1010, 1020, 780, 770, 760, 720, 650, 760, 750, 730, 660, 640, 670, 630, 610, 610, 640, 580, 580, 580, 570, 520]
}

# Create DataFrame from the data dictionary
df = pd.DataFrame(data)
df['Month'] = pd.to_datetime(df['Month'], format='%b-%y')
df.set_index('Month', inplace=True)

# Data Preprocessing: Identify and treat outliers using the IQR method
def treat_outliers_iqr(df):
    for column in df.columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
        df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
    return df

# Treat outliers in the DataFrame
df = treat_outliers_iqr(df)

# Interest rate for forward contracts
interest_rate = 0.0244  # 2.44%

# Delivery period for forward contracts
delivery_period = (3, 30)  # 3-30 days

# Define parameters and initialize variables for simulations
num_scenarios = 10000  # Number of scenarios for Monte Carlo simulation
n_months_future = 60  # Simulate future prices for the next 5 years (12 months * 5 years)
dt = 1 / 252  # Assuming daily data; adjust as needed

# Function to perform Monte Carlo simulation with specified parameters
def monte_carlo_simulation(s0, mu, sigma, n, num_scenarios, dt):
    prices = np.zeros((num_scenarios, n))
    for i in range(num_scenarios):
        price = np.zeros(n)
        price[0] = s0
        for t in range(1, n):
            simulated_return = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.normal()
            next_price = price[t - 1] * np.exp(simulated_return)
            if next_price <= 0:
                prices[i, t:] = np.nan
                break
            price[t] = next_price
        prices[i, :t+1] = price[:t+1]
    return prices

# Adjust the start date for simulation to April 2024
start_date_future = pd.Timestamp('2024-04-01')

# Initialize simulated prices dictionary
simulated_prices_future = {}

# Loop over each polymer compound and perform simulation
for material in df.columns:
    # Simulate future prices with specified parameters starting from April 2024
    s0_future = df[material].iloc[-1]
    mu = 0.003  # Drift (mu) of 0.3%
    sigma = 0.27124  # Volatility (sigma) of 27.12%
    simulated_prices_future[material] = monte_carlo_simulation(s0_future, mu, sigma, n_months_future, num_scenarios, dt)

    # Create a DataFrame for plotting
    dates = pd.date_range(start=start_date_future, periods=n_months_future, freq='M')
    scenarios = np.arange(num_scenarios)
    multi_index = pd.MultiIndex.from_product([dates, scenarios], names=['Date', 'Scenario'])
    plot_data = pd.DataFrame(simulated_prices_future[material].T.flatten(), index=multi_index, columns=[material])
    plot_data = plot_data.reset_index()

    # Plot the distribution of future prices for each material until Mar-2029
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=plot_data, x='Date', y=material, hue='Scenario', palette='viridis', legend=None, alpha=0.2)
    plt.xlabel('Month/Year')
    plt.ylabel('Price')
    plt.title(f'Simulated Future Prices for {material} until Mar-2029')
    plt.ylim(0, None)  # Set y-axis limit to start from zero
    plt.show()

    # Calculate and print expected future prices at different time steps for each material until Mar-2029
    expected_prices_future = simulated_prices_future[material].mean(axis=0)
    print(f"\nExpected Future Prices at Different Time Steps for {material} until Mar-2029:")
    for month, price in zip(dates, expected_prices_future):
        print(f"{month.strftime('%B %Y')}: €{price:.2f}")

# Calculate and print MSE, MAE, RMSE, and MAPE for the overall model
def calculate_errors(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return mse, mae, rmse, mape

# Assuming we have actual future prices for comparison (for demonstration purposes, using the last known prices)
actual_prices = df.iloc[-1].values

# Calculate errors for each material
for material in df.columns:
    predicted_prices = simulated_prices_future[material][:, -1]  # Using the last simulated prices
    mse, mae, rmse, mape = calculate_errors(np.full(predicted_prices.shape, actual_prices[df.columns.get_loc(material)]), predicted_prices)
    print(f"\nError Metrics for {material}:")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")


# **Python Code for Quantile Curves based on Monte Carlo Simulations**

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm

# Set the aesthetic parameters
sns.set(style="whitegrid")

# Define the data
data = {
    'Month': ['Apr-19', 'May-19', 'Jun-19', 'Jul-19', 'Aug-19', 'Sep-19', 'Oct-19', 'Nov-19', 'Dec-19',
              'Jan-20', 'Feb-20', 'Mar-20', 'Apr-20', 'May-20', 'Jun-20', 'Jul-20', 'Aug-20', 'Sep-20',
              'Oct-20', 'Nov-20', 'Dec-20', 'Jan-21', 'Feb-21', 'Mar-21', 'Apr-21', 'May-21', 'Jun-21',
              'Jul-21', 'Aug-21', 'Sep-21', 'Oct-21', 'Nov-21', 'Dec-21', 'Jan-22', 'Feb-22', 'Mar-22',
              'Apr-22', 'May-22', 'Jun-22', 'Jul-22', 'Aug-22', 'Sep-22', 'Oct-22', 'Nov-22', 'Dec-22',
              'Jan-23', 'Feb-23', 'Mar-23', 'Apr-23', 'May-23', 'Jun-23', 'Jul-23', 'Aug-23', 'Sep-23',
              'Oct-23', 'Nov-23', 'Dec-23', 'Jan-24', 'Feb-24', 'Mar-24'],
    'HDPE film': [1166, 1182, 1166, 1102, 1074, 1027, 1037, 988, 967, 958, 958, 819, 817, 757, 809, 878,
                  914, 911, 882, 948, 958, 1140, 1435, 1744, 1858, 1523, 1465, 1350, 1354, 1413, 1388,
                  1595, 1553, 1543, 1547, 1762, 1968, 1729, 1708, 1577, 1443, 1355, 1388, 1393, 1339,
                  1313, 1299, 1334, 1229, 1122, 1065, 1016, 1118, 1158, 1175, 1076, 1063, 1186, 1265, 1280],
    'LDPE film': [1096, 1120, 1105, 1047, 1029, 964, 985, 953, 952, 944, 953, 836, 828, 762, 840, 917,
                  935, 918, 912, 1024, 1051, 1288, 1562, 1920, 2082, 1900, 1816, 1710, 1748, 1767, 1486,
                  1914, 1872, 1851, 1845, 2033, 2225, 1932, 1893, 1660, 1547, 1407, 1486, 1431, 1394,
                  1352, 1330, 1349, 1261, 1095, 979, 986, 1090, 1162, 1221, 1067, 1059, 1290, 1351, 1060],
    'PET': [170, 200, 200, 180, 200, 160, 190, 150, 210, 130, 170, 230, 220, 210, 140, 220, 160, 90, 70,
            180, 300, 160, 100, 80, 140, 70, 70, 200, 160, 180, 290, 90, 70, 0, 360, 200, 140, 250, 10,
            90, 360, 450, 400, 400, 410, 470, 340, 220, 230, 310, 300, 310, 300, 250, 320, 190, 200, 260, 260, 400],
    'PP homo-polymer fiber': [1203, 1223, 1197, 1070, 1055, 1008, 1019, 979, 995, 968, 977, 917, 917,
                              847, 893, 938, 949, 936, 921, 997, 1012, 1194, 1434, 1844, 2066, 1962,
                              1630, 1538, 1631, 1631, 1275, 1712, 1697, 1676, 1654, 1860, 2043, 1790,
                              1648, 1466, 1248, 1191, 1275, 1265, 1220, 1227, 1250, 1275, 1221, 1121,
                              1062, 1019, 1065, 1141, 1145, 1057, 1029, 1237, 1254, 1250],
}

# Create DataFrame from the data dictionary
df = pd.DataFrame(data)
df['Month'] = pd.to_datetime(df['Month'], format='%b-%y')  # Corrected date format
df.set_index('Month', inplace=True)

# Data Preprocessing: Identify and treat outliers using the IQR method
def treat_outliers_iqr(df):
    for column in df.columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
        df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
    return df

# Treat outliers in the DataFrame
df = treat_outliers_iqr(df)

# Interest rate for forward contracts
interest_rate = 0.0244  # 2.44%

# Delivery period for forward contracts
delivery_period = (3, 30)  # 3-30 days

# Define parameters and initialize variables for simulations
num_scenarios = 10000  # Number of scenarios for Monte Carlo simulation
n_months_future = 60  # Simulate future prices for the next 5 years (12 months * 5 years)
dt = 1 / 252  # Assuming daily data; adjust as needed

# Function to perform Monte Carlo simulation with specified parameters
def monte_carlo_simulation(s0, mu, sigma, n, num_scenarios, dt):
    prices = np.zeros((num_scenarios, n))
    for i in range(num_scenarios):
        price = np.zeros(n)
        price[0] = s0
        for t in range(1, n):
            simulated_return = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.normal()
            next_price = price[t - 1] * np.exp(simulated_return)
            if next_price <= 0:
                prices[i, t:] = np.nan
                break
            price[t] = next_price
        prices[i, :t+1] = price[:t+1]
    return prices

# Adjust the start date for simulation to April 2024
start_date_future = pd.Timestamp('2024-04-01')

# Initialize simulated prices dictionary
simulated_prices_future = {}

# Loop over each polymer compound and perform simulation
for material in ['HDPE film', 'LDPE film', 'PET', 'PP homo-polymer fiber']:
    # Simulate future prices with specified parameters starting from April 2024
    s0_future = df[material].iloc[-1]
    mu = 0.003  # Drift (mu) of 0.3%
    sigma = 0.27124  # Volatility (sigma) of 27.12%
    simulated_prices = monte_carlo_simulation(s0_future, mu, sigma, n_months_future, num_scenarios, dt)
    simulated_prices_future[material] = simulated_prices

    # Calculate quantiles for each month
    quantiles = np.percentile(simulated_prices, [25, 50, 75], axis=0)

    # Plot quantile curves
    plt.figure(figsize=(10, 6))
    months = pd.date_range(start=start_date_future, periods=n_months_future, freq='M')
    plt.plot(months, quantiles[0], color='blue', label='25th Percentile')
    plt.plot(months, quantiles[1], color='green', label='50th Percentile (Median)')
    plt.plot(months, quantiles[2], color='red', label='75th Percentile')
    plt.title(f'Quantile Curves for {material}')
    plt.xlabel('Month')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


# **Python Code for Time Series XGBoost Forecasting model**

# In[9]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Set the aesthetic parameters in one step
sns.set(style="whitegrid")

# Define the data
data = {
    'Month': ['Apr-19', 'May-19', 'Jun-19', 'Jul-19', 'Aug-19', 'Sep-19', 'Oct-19', 'Nov-19', 'Dec-19',
              'Jan-20', 'Feb-20', 'Mar-20', 'Apr-20', 'May-20', 'Jun-20', 'Jul-20', 'Aug-20', 'Sep-20',
              'Oct-20', 'Nov-20', 'Dec-20', 'Jan-21', 'Feb-21', 'Mar-21', 'Apr-21', 'May-21', 'Jun-21',
              'Jul-21', 'Aug-21', 'Sep-21', 'Oct-21', 'Nov-21', 'Dec-21', 'Jan-22', 'Feb-22', 'Mar-22',
              'Apr-22', 'May-22', 'Jun-22', 'Jul-22', 'Aug-22', 'Sep-22', 'Oct-22', 'Nov-22', 'Dec-22',
              'Jan-23', 'Feb-23', 'Mar-23', 'Apr-23', 'May-23', 'Jun-23', 'Jul-23', 'Aug-23', 'Sep-23',
              'Oct-23', 'Nov-23', 'Dec-23', 'Jan-24', 'Feb-24', 'Mar-24'],
    'HDPE film': [1166, 1182, 1166, 1102, 1074, 1027, 1037, 988, 967, 958, 958, 819, 817, 757, 809, 878,
                  914, 911, 882, 948, 958, 1140, 1435, 1744, 1858, 1523, 1465, 1350, 1354, 1413, 1388,
                  1595, 1553, 1543, 1547, 1762, 1968, 1729, 1708, 1577, 1443, 1355, 1388, 1393, 1339,
                  1313, 1299, 1334, 1229, 1122, 1065, 1016, 1118, 1158, 1175, 1076, 1063, 1186, 1265, 1280],
    'LDPE film': [1096, 1120, 1105, 1047, 1029, 964, 985, 953, 952, 944, 953, 836, 828, 762, 840, 917,
                  935, 918, 912, 1024, 1051, 1288, 1562, 1920, 2082, 1900, 1816, 1710, 1748, 1767, 1486,
                  1914, 1872, 1851, 1845, 2033, 2225, 1932, 1893, 1660, 1547, 1407, 1486, 1431, 1394,
                  1352, 1330, 1349, 1261, 1095, 979, 986, 1090, 1162, 1221, 1067, 1059, 1290, 1351, 1060],
    'PET': [170, 200, 200, 180, 200, 160, 190, 150, 210, 130, 170, 230, 220, 210, 140, 220, 160, 90, 70,
            180, 300, 160, 100, 80, 140, 70, 70, 200, 160, 180, 290, 90, 70, 0, 360, 200, 140, 250, 10,
            90, 360, 450, 400, 400, 410, 470, 340, 220, 230, 310, 300, 310, 300, 250, 320, 190, 200, 260, 260, 400],
    'PP homo-polymer fiber': [1203, 1223, 1197, 1070, 1055, 1008, 1019, 979, 995, 968, 977, 917, 917,
                              847, 893, 938, 949, 936, 921, 997, 1012, 1194, 1434, 1844, 2066, 1962,
                              1630, 1538, 1631, 1631, 1275, 1712, 1697, 1676, 1654, 1860, 2043, 1790,
                              1648, 1466, 1248, 1191, 1275, 1265, 1220, 1227, 1250, 1275, 1221, 1121,
                              1062, 1019, 1065, 1141, 1145, 1057, 1029, 1237, 1254, 1250],
    'rPET': [360, 360, 350, 330, 320, 340, 340, 340, 400, 370, 330, 310, 380, 370, 360, 340, 290, 300, 260, 300, 340, 380, 380, 380, 380, 390, 430, 480, 440, 420, 520, 520, 480, 720, 350, 390, 360, 600, 820, 720, 560, 480, 490, 550, 700, 610, 700, 550, 550, 570, 480, 450, 460, 430, 480, 450, 410, 380, 420, 410],
    'rLDPE Film': [550, 580, 520, 540, 500, 480, 510, 490, 510, 480, 470, 530, 530, 520, 500, 500, 430, 430, 470, 470, 450, 510, 470, 490, 530, 490, 470, 450, 420, 380, 340, 430, 470, 480, 430, 650, 640, 640, 790, 610, 500, 560, 600, 520, 490, 480, 470, 390, 410, 730, 180, 290, 180, 180, 290, 360, 410, 560, 550, 470],
    'rPP': [590, 600, 600, 580, 570, 570, 550, 570, 570, 560, 530, 520, 510, 530, 520, 460, 450, 490, 480, 480, 510, 510, 520, 530, 550, 600, 620, 560, 570, 620, 650, 730, 630, 650, 530, 790, 780, 670, 860, 830, 750, 790, 900, 840, 770, 780, 760, 740, 750, 740, 740, 700, 650, 640, 640, 600, 580, 600, 620, 730],
    'rHDPE Film': [570, 580, 600, 620, 600, 560, 570, 600, 600, 570, 550, 520, 520, 510, 490, 510, 490, 490, 490, 490, 490, 500, 520, 580, 620, 620, 640, 600, 630, 620, 570, 580, 600, 750, 590, 710, 630, 740, 1010, 1020, 780, 770, 760, 720, 650, 760, 750, 730, 660, 640, 670, 630, 610, 610, 640, 580, 580, 580, 570, 520]
}

# Create DataFrame from the data dictionary
df = pd.DataFrame(data)
df['Month'] = pd.to_datetime(df['Month'], format='%b-%y')
df.set_index('Month', inplace=True)

# Data Preprocessing: Identify and treat outliers using the IQR method
def treat_outliers_iqr(df):
    for column in df.columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
        df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
    return df

# Treat outliers in the DataFrame
df = treat_outliers_iqr(df)

# Function to create lag features
def create_lag_features(df, compound_name, n_lags):
    lag_df = pd.DataFrame()
    for lag in range(1, n_lags + 1):
        lag_df[f'lag_{lag}'] = df[compound_name].shift(lag)
    return lag_df.dropna()

# Function to forecast prices using XGBoost
def forecast_prices_xgb(compound_name, forecast_period=60, n_lags=12):
    compound_df = df[[compound_name]].copy()
    n_lags = min(n_lags, len(compound_df) - 1)
    lag_df = create_lag_features(compound_df, compound_name, n_lags)
    y = compound_df[compound_name].iloc[n_lags:]
    X_train, X_test, y_train, y_test = train_test_split(lag_df, y, test_size=0.25, random_state=42)
    
    # Initialize and train the XGBoost model
    reg = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.01, max_depth=3, objective='reg:squarederror', booster='gbtree')
    reg.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)
    
    # Forecast future prices
    future_dates = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=forecast_period, freq='M')
    future_df = pd.DataFrame(index=future_dates, columns=[compound_name])
    last_known_data = lag_df.iloc[-1:].values.reshape(1, -1)
    forecasted_prices = []
    for _ in future_dates:
        future_price = reg.predict(last_known_data)
        forecasted_prices.append(future_price[0])
        last_known_data = np.roll(last_known_data, -1)
        last_known_data[0, -1] = future_price
    future_df[compound_name] = forecasted_prices
    
    # Print forecasted prices
    print(f"Forecasted prices for {compound_name} from {future_dates[0]} to {future_dates[-1]}:")
    print(future_df)
    
    # Plot historical and forecasted prices
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df[compound_name], label='Historical Prices', color='blue')
    plt.plot(future_df.index, forecasted_prices, label='Forecasted Prices', color='red', linestyle='--')
    plt.title(f'Historical and Forecasted Prices for {compound_name}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    
    # Calculate and print error metrics
    y_pred = reg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    print(f"\nError Metrics for {compound_name}:")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# Forecast prices for all compounds
for compound in df.columns:
    forecast_prices_xgb(compound, forecast_period=60, n_lags=12)


# **Python Code for Decision Trees and Feature Engineering derived from the Time Series XGBoost Forecasting Model**

# In[17]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import plot_tree, plot_importance

# Set the aesthetic parameters in one step
sns.set(style="whitegrid")

# Assuming 'data' is the dictionary containing the historical data
data = {
    'Month': ['Apr-19', 'May-19', 'Jun-19', 'Jul-19', 'Aug-19', 'Sep-19', 'Oct-19', 'Nov-19', 'Dec-19',
              'Jan-20', 'Feb-20', 'Mar-20', 'Apr-20', 'May-20', 'Jun-20', 'Jul-20', 'Aug-20', 'Sep-20',
              'Oct-20', 'Nov-20', 'Dec-20', 'Jan-21', 'Feb-21', 'Mar-21', 'Apr-21', 'May-21', 'Jun-21',
              'Jul-21', 'Aug-21', 'Sep-21', 'Oct-21', 'Nov-21', 'Dec-21', 'Jan-22', 'Feb-22', 'Mar-22',
              'Apr-22', 'May-22', 'Jun-22', 'Jul-22', 'Aug-22', 'Sep-22', 'Oct-22', 'Nov-22', 'Dec-22',
              'Jan-23', 'Feb-23', 'Mar-23', 'Apr-23', 'May-23', 'Jun-23', 'Jul-23', 'Aug-23', 'Sep-23',
              'Oct-23', 'Nov-23', 'Dec-23', 'Jan-24', 'Feb-24', 'Mar-24'],
    'HDPE film': [1166, 1182, 1166, 1102, 1074, 1027, 1037, 988, 967, 958, 958, 819, 817, 757, 809, 878,
                  914, 911, 882, 948, 958, 1140, 1435, 1744, 1858, 1523, 1465, 1350, 1354, 1413, 1388,
                  1595, 1553, 1543, 1547, 1762, 1968, 1729, 1708, 1577, 1443, 1355, 1388, 1393, 1339,
                  1313, 1299, 1334, 1229, 1122, 1065, 1016, 1118, 1158, 1175, 1076, 1063, 1186, 1265, 1280],
    'LDPE film': [1096, 1120, 1105, 1047, 1029, 964, 985, 953, 952, 944, 953, 836, 828, 762, 840, 917,
                  935, 918, 912, 1024, 1051, 1288, 1562, 1920, 2082, 1900, 1816, 1710, 1748, 1767, 1486,
                  1914, 1872, 1851, 1845, 2033, 2225, 1932, 1893, 1660, 1547, 1407, 1486, 1431, 1394,
                  1352, 1330, 1349, 1261, 1095, 979, 986, 1090, 1162, 1221, 1067, 1059, 1290, 1351, 1060],
    'PET': [170, 200, 200, 180, 200, 160, 190, 150, 210, 130, 170, 230, 220, 210, 140, 220, 160, 90, 70,
            180, 300, 160, 100, 80, 140, 70, 70, 200, 160, 180, 290, 90, 70, 0, 360, 200, 140, 250, 10,
            90, 360, 450, 400, 400, 410, 470, 340, 220, 230, 310, 300, 310, 300, 250, 320, 190, 200, 260, 260, 400],
    'PP homo-polymer fiber': [1203, 1223, 1197, 1070, 1055, 1008, 1019, 979, 995, 968, 977, 917, 917,
                              847, 893, 938, 949, 936, 921, 997, 1012, 1194, 1434, 1844, 2066, 1962,
                              1630, 1538, 1631, 1631, 1275, 1712, 1697, 1676, 1654, 1860, 2043, 1790,
                              1648, 1466, 1248, 1191, 1275, 1265, 1220, 1227, 1250, 1275, 1221, 1121,
                              1062, 1019, 1065, 1141, 1145, 1057, 1029, 1237, 1254, 1250],
    'rPET': [360, 360, 350, 330, 320, 340, 340, 340, 400, 370, 330, 310, 380, 370, 360, 340, 290, 300,
             260, 300, 340, 380, 380, 380, 380, 390, 430, 480, 440, 420, 520, 520, 480, 720, 350, 390,
             360, 600, 820, 720, 560, 480, 490, 550, 700, 610, 700, 550, 550, 570, 480, 450, 460, 430,
             480, 450, 410, 380, 420, 410],
    'rLDPE Film': [550, 580, 520, 540, 500, 480, 510, 490, 510, 480, 470, 530, 530, 520, 500, 500, 430,
                   430, 470, 470, 450, 510, 470, 490, 530, 490, 470, 450, 420, 380, 340, 430, 470, 480,
                   430, 650, 640, 640, 790, 610, 500, 560, 600, 520, 490, 480, 470, 390, 410, 730, 180,
                   290, 180, 180, 290, 360, 410, 560, 550, 470],
    'rPP': [590, 600, 600, 580, 570, 570, 550, 570, 570, 560, 530, 520, 510, 530, 520, 460, 450, 490,
            480, 480, 510, 510, 520, 530, 550, 600, 620, 560, 570, 620, 650, 730, 630, 650, 530, 790,
            780, 670, 860, 830, 750, 790, 900, 840, 770, 780, 760, 740, 750, 740, 740, 700, 650, 640,
            640, 600, 580, 600, 620, 730],
    'rHDPE Film': [570, 580, 600, 620, 600, 560, 570, 600, 600, 570, 550, 520, 520, 510, 490, 510, 490,
                   490, 490, 490, 490, 500, 520, 580, 620, 620, 640, 600, 630, 620, 570, 580, 600, 750,
                   590, 710, 630, 740, 1010, 1020, 780, 770, 760, 720, 650, 760, 750, 730, 660, 640,
                   670, 630, 610, 610, 640, 580, 580, 580, 570, 520]
}

# Convert 'Month' column to datetime and set it as index
df = pd.DataFrame(data)
df['Month'] = pd.to_datetime(df['Month'], format='%b-%y')
df.set_index('Month', inplace=True)

# Data Preprocessing: Identify and treat outliers using the IQR method
def treat_outliers_iqr(df):
    for column in df.columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
        df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
    return df

# Treat outliers in the DataFrame
df = treat_outliers_iqr(df)


def create_lag_features(df, compound_name, n_lags):
    lag_df = pd.DataFrame()
    for lag in range(1, n_lags + 1):
        lag_df[f'lag_{lag}'] = df[compound_name].shift(lag)
    return lag_df.dropna()

def train_xgb_model(compound_name, n_lags=12):
    compound_df = df[[compound_name]].copy()
    n_lags = min(n_lags, len(compound_df) - 1)
    lag_df = create_lag_features(compound_df, compound_name, n_lags)
    y = compound_df[compound_name].iloc[n_lags:]
    X_train, X_test, y_train, y_test = train_test_split(lag_df, y, test_size=0.25, random_state=42)

    # Initialize and train the XGBoost model
    reg = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.01, max_depth=3, objective='reg:squarederror', booster='gbtree')
    reg.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)

    # Plot the first tree
    plt.figure(figsize=(20, 10))
    plot_tree(reg, num_trees=0, rankdir='LR')
    plt.title(f'Decision Tree for {compound_name}')
    plt.show()

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plot_importance(reg)
    plt.title(f'Feature Importance for {compound_name}')
    plt.show()

    return reg

# Train and visualize the model for all compounds
for compound in df.columns:
    train_xgb_model(compound, n_lags=12)


# In[4]:


pip install plotly


# **Python Code for 2-Dimensional plot for visualization of Forecasted Prices of Polymer Compounds from Monte Carlo Forward Price Simulation**

# In[2]:


import pandas as pd
import plotly.graph_objects as go
from scipy.signal import find_peaks

# Data for HDPE film
hdpe_data = {
    "Month/Year": [
        "April 2024", "May 2024", "June 2024", "July 2024", "August 2024", "September 2024", "October 2024", "November 2024", "December 2024",
        "January 2025", "February 2025", "March 2025", "April 2025", "May 2025", "June 2025", "July 2025", "August 2025", "September 2025", "October 2025", "November 2025", "December 2025",
        "January 2026", "February 2026", "March 2026", "April 2026", "May 2026", "June 2026", "July 2026", "August 2026", "September 2026", "October 2026", "November 2026", "December 2026",
        "January 2027", "February 2027", "March 2027", "April 2027", "May 2027", "June 2027", "July 2027", "August 2027", "September 2027", "October 2027", "November 2027", "December 2027",
        "January 2028", "February 2028", "March 2028", "April 2028", "May 2028", "June 2028", "July 2028", "August 2028", "September 2028", "October 2028", "November 2028", "December 2028",
        "January 2029", "February 2029", "March 2029"
    ],
        "HDPE film":[
        1280.00, 1280.05, 1280.11, 1280.21, 1280.25, 1280.46, 1280.45, 1280.31, 1280.63,
        1280.60, 1280.88, 1280.85, 1280.97, 1281.03, 1281.24, 1281.32, 1281.78, 1281.76, 1281.60, 1281.66, 1281.88,
        1281.99, 1281.85, 1282.05, 1281.95, 1281.65, 1281.31, 1281.32, 1281.52, 1281.43, 1281.19, 1280.97, 1281.09,
        1281.04, 1281.19, 1281.04, 1280.96, 1281.04, 1281.10, 1281.06, 1281.07, 1281.36, 1281.72, 1281.45, 1281.51,
        1281.35, 1281.08, 1281.32, 1281.15, 1281.07, 1281.29, 1281.00, 1280.72, 1280.93, 1281.50, 1281.56, 1281.52,
        1281.42, 1281.71, 1281.91
    ],

}

# Data for LDPE film
ldpe_data = {
    "Month/Year": hdpe_data["Month/Year"],
    "LDPE film":[
        1060.00, 1060.25, 1060.44, 1060.29, 1060.07, 1059.84, 1059.82, 1060.14, 1060.18,
        1060.40, 1060.71, 1060.70, 1060.72, 1060.75, 1061.26, 1061.46, 1061.38, 1061.52, 1061.69, 1061.93, 1062.06,
        1061.98, 1061.60, 1061.99, 1062.07, 1062.10, 1062.14, 1062.26, 1062.29, 1062.10, 1061.88, 1061.73, 1061.96,
        1062.03, 1062.09, 1061.73, 1061.98, 1061.62, 1061.43, 1061.31, 1061.45, 1061.70, 1061.53, 1061.53, 1061.69,
        1062.11, 1062.14, 1062.00, 1062.22, 1062.11, 1062.02, 1062.17, 1062.18, 1062.05, 1061.77, 1062.01, 1061.85,
        1061.61, 1061.08, 1060.91
    ],
}

# Data for PET
pet_data = {
    "Month/Year": hdpe_data["Month/Year"],
    "PET":[
        400.00, 399.95, 399.94, 399.93, 399.98, 399.97, 400.01, 399.97, 399.95,
        400.00, 399.99, 399.95, 400.05, 400.02, 399.99, 400.00, 399.92, 400.03, 400.05, 400.04, 400.07,
        400.06, 399.99, 399.97, 399.96, 399.93, 400.01, 400.11, 400.12, 400.07, 400.18, 400.34, 400.36,
        400.41, 400.45, 400.56, 400.56, 400.48, 400.39, 400.48, 400.37, 400.42, 400.46, 400.52, 400.48,
        400.40, 400.41, 400.37, 400.42, 400.39, 400.34, 400.36, 400.37, 400.33, 400.35, 400.27, 400.20,
        400.19, 400.11, 399.98
    ],
               
}

# Data for PP homo-polymer fiber
pp_data = {
    "Month/Year": hdpe_data["Month/Year"],
    "PP homo-polymer fiber":[
        1250.00, 1250.26, 1250.58, 1250.66, 1250.94, 1250.94, 1250.90, 1250.97, 1251.14,
        1250.97, 1251.22, 1251.38, 1251.58, 1251.68, 1251.64, 1251.69, 1251.63, 1251.99, 1251.77, 1251.79, 1252.01,
        1251.69, 1251.71, 1251.81, 1251.85, 1252.10, 1251.95, 1251.91, 1251.61, 1251.63, 1252.14, 1252.20, 1252.22,
        1252.06, 1252.15, 1252.10, 1251.97, 1252.08, 1252.05, 1252.16, 1251.72, 1251.65, 1251.68, 1251.33, 1251.20,
        1251.20, 1250.99, 1250.69, 1250.45, 1250.37, 1250.35, 1250.50, 1250.05, 1249.74, 1249.72, 1250.10, 1250.33,
        1250.44, 1250.32, 1250.15
    ],
}

# Create DataFrames
df_hdpe = pd.DataFrame(hdpe_data)
df_ldpe = pd.DataFrame(ldpe_data)
df_pet = pd.DataFrame(pet_data)
df_pp = pd.DataFrame(pp_data)

# Merge DataFrames on "Month/Year"
df = df_hdpe.merge(df_ldpe, on="Month/Year").merge(df_pet, on="Month/Year").merge(df_pp, on="Month/Year")

# Convert "Month/Year" to datetime
df["Month/Year"] = pd.to_datetime(df["Month/Year"], format="%B %Y")

# Function to plot time series with peaks and troughs
def plot_time_series_with_peaks(df, column, title):
    fig = go.Figure()

    # Add time series line
    fig.add_trace(go.Scatter(x=df["Month/Year"], y=df[column], mode='lines', name=column))

    # Find peaks and troughs
    peaks, _ = find_peaks(df[column])
    troughs, _ = find_peaks(-df[column])

    # Add peaks
    fig.add_trace(go.Scatter(
        x=df["Month/Year"].iloc[peaks],
        y=df[column].iloc[peaks],
        mode='markers',
        marker=dict(color='red', size=8),
        name='Peaks'
    ))

    # Add troughs
    fig.add_trace(go.Scatter(
        x=df["Month/Year"].iloc[troughs],
        y=df[column].iloc[troughs],
        mode='markers',
        marker=dict(color='blue', size=8),
        name='Troughs'
    ))

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Price (€)',
        showlegend=True
    )

    fig.show()

# Plot for each polymer compound
plot_time_series_with_peaks(df, "HDPE film", "Expected Forward HDPE Film Prices")
plot_time_series_with_peaks(df, "LDPE film", "Expected Forward LDPE Film Prices")
plot_time_series_with_peaks(df, "PET", "Expected Forward PET Prices")
plot_time_series_with_peaks(df, "PP homo-polymer fiber", "Expected Forward PP Homo-Polymer Fiber Prices")

df.describe()


# **Python Code for ascertaining the Historical Price Volatilities from Historical Data**

# In[22]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set the aesthetic parameters
sns.set(style="whitegrid")

# Define the data
data = {
    'Month': ['Apr-19', 'May-19', 'Jun-19', 'Jul-19', 'Aug-19', 'Sep-19', 'Oct-19', 'Nov-19', 'Dec-19',
              'Jan-20', 'Feb-20', 'Mar-20', 'Apr-20', 'May-20', 'Jun-20', 'Jul-20', 'Aug-20', 'Sep-20',
              'Oct-20', 'Nov-20', 'Dec-20', 'Jan-21', 'Feb-21', 'Mar-21', 'Apr-21', 'May-21', 'Jun-21',
              'Jul-21', 'Aug-21', 'Sep-21', 'Oct-21', 'Nov-21', 'Dec-21', 'Jan-22', 'Feb-22', 'Mar-22',
              'Apr-22', 'May-22', 'Jun-22', 'Jul-22', 'Aug-22', 'Sep-22', 'Oct-22', 'Nov-22', 'Dec-22',
              'Jan-23', 'Feb-23', 'Mar-23', 'Apr-23', 'May-23', 'Jun-23', 'Jul-23', 'Aug-23', 'Sep-23',
              'Oct-23', 'Nov-23', 'Dec-23', 'Jan-24', 'Feb-24', 'Mar-24'],
    'HDPE film': [1166, 1182, 1166, 1102, 1074, 1027, 1037, 988, 967, 958, 958, 819, 817, 757, 809, 878,
                  914, 911, 882, 948, 958, 1140, 1435, 1744, 1858, 1523, 1465, 1350, 1354, 1413, 1388,
                  1595, 1553, 1543, 1547, 1762, 1968, 1729, 1708, 1577, 1443, 1355, 1388, 1393, 1339,
                  1313, 1299, 1334, 1229, 1122, 1065, 1016, 1118, 1158, 1175, 1076, 1063, 1186, 1265, 1280],
    'LDPE film': [1096, 1120, 1105, 1047, 1029, 964, 985, 953, 952, 944, 953, 836, 828, 762, 840, 917,
                  935, 918, 912, 1024, 1051, 1288, 1562, 1920, 2082, 1900, 1816, 1710, 1748, 1767, 1486,
                  1914, 1872, 1851, 1845, 2033, 2225, 1932, 1893, 1660, 1547, 1407, 1486, 1431, 1394,
                  1352, 1330, 1349, 1261, 1095, 979, 986, 1090, 1162, 1221, 1067, 1059, 1290, 1351, 1060],
    'PET': [170, 200, 200, 180, 200, 160, 190, 150, 210, 130, 170, 230, 220, 210, 140, 220, 160, 90, 70,
            180, 300, 160, 100, 80, 140, 70, 70, 200, 160, 180, 290, 90, 70, 0, 360, 200, 140, 250, 10,
            90, 360, 450, 400, 400, 410, 470, 340, 220, 230, 310, 300, 310, 300, 250, 320, 190, 200, 260, 260, 400],
    'PP homo-polymer fiber': [1203, 1223, 1197, 1070, 1055, 1008, 1019, 979, 995, 968, 977, 917, 917,
                              847, 893, 938, 949, 936, 921, 997, 1012, 1194, 1434, 1844, 2066, 1962,
                              1630, 1538, 1631, 1631, 1275, 1712, 1697, 1676, 1654, 1860, 2043, 1790,
                              1648, 1466, 1248, 1191, 1275, 1265, 1220, 1227, 1250, 1275, 1221, 1121,
                              1062, 1019, 1065, 1141, 1145, 1057, 1029, 1237, 1254, 1250],
}

# Create DataFrame from the data dictionary
df = pd.DataFrame(data)
df['Month'] = pd.to_datetime(df['Month'], format='%b-%y')
df.set_index('Month', inplace=True)

# Data Preprocessing: Identify and treat outliers using the IQR method
def treat_outliers_iqr(df):
    for column in df.columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
        df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
    return df

# Treat outliers in the DataFrame
df = treat_outliers_iqr(df)

# Replace zeros with NaN for interpolation
df = df.replace(0, np.nan)

# Interpolate missing values using linear interpolation
df = df.interpolate(method='linear', limit_direction='both')

# Function to calculate historical volatility
def calculate_historical_volatility(prices):
    log_returns = np.log(prices / prices.shift(1))
    return log_returns.std() * np.sqrt(12)  # Annualize monthly data

# Calculate historical volatilities for each polymer compound
volatilities = pd.DataFrame()
for material in df.columns:
    volatilities[material] = df[material].rolling(window=len(df), min_periods=1).apply(calculate_historical_volatility)

# Plot the historical volatilities
plt.figure(figsize=(12, 8))
for material in volatilities.columns:
    plt.plot(volatilities.index, volatilities[material], label=material)

plt.xlabel('Date')
plt.ylabel('Historical Volatility (Annualized)')
plt.title('Historical Volatilities of Polymer Compounds')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Print the latest volatility values
print("Latest Historical Volatilities:")
print(volatilities.iloc[-1])


# **Python Code for Computing the Interquartile Ranges (IQRs) from Historical data**

# In[18]:


import pandas as pd
import numpy as np

# Assuming 'data' is the dictionary containing the historical data
data = {
    'Month': ['Apr-19', 'May-19', 'Jun-19', 'Jul-19', 'Aug-19', 'Sep-19', 'Oct-19', 'Nov-19', 'Dec-19',
              'Jan-20', 'Feb-20', 'Mar-20', 'Apr-20', 'May-20', 'Jun-20', 'Jul-20', 'Aug-20', 'Sep-20',
              'Oct-20', 'Nov-20', 'Dec-20', 'Jan-21', 'Feb-21', 'Mar-21', 'Apr-21', 'May-21', 'Jun-21',
              'Jul-21', 'Aug-21', 'Sep-21', 'Oct-21', 'Nov-21', 'Dec-21', 'Jan-22', 'Feb-22', 'Mar-22',
              'Apr-22', 'May-22', 'Jun-22', 'Jul-22', 'Aug-22', 'Sep-22', 'Oct-22', 'Nov-22', 'Dec-22',
              'Jan-23', 'Feb-23', 'Mar-23', 'Apr-23', 'May-23', 'Jun-23', 'Jul-23', 'Aug-23', 'Sep-23',
              'Oct-23', 'Nov-23', 'Dec-23', 'Jan-24', 'Feb-24', 'Mar-24'],
    'HDPE film': [1166, 1182, 1166, 1102, 1074, 1027, 1037, 988, 967, 958, 958, 819, 817, 757, 809, 878,
                  914, 911, 882, 948, 958, 1140, 1435, 1744, 1858, 1523, 1465, 1350, 1354, 1413, 1388,
                  1595, 1553, 1543, 1547, 1762, 1968, 1729, 1708, 1577, 1443, 1355, 1388, 1393, 1339,
                  1313, 1299, 1334, 1229, 1122, 1065, 1016, 1118, 1158, 1175, 1076, 1063, 1186, 1265, 1280],
    'LDPE film': [1096, 1120, 1105, 1047, 1029, 964, 985, 953, 952, 944, 953, 836, 828, 762, 840, 917,
                  935, 918, 912, 1024, 1051, 1288, 1562, 1920, 2082, 1900, 1816, 1710, 1748, 1767, 1486,
                  1914, 1872, 1851, 1845, 2033, 2225, 1932, 1893, 1660, 1547, 1407, 1486, 1431, 1394,
                  1352, 1330, 1349, 1261, 1095, 979, 986, 1090, 1162, 1221, 1067, 1059, 1290, 1351, 1060],
    'PET': [170, 200, 200, 180, 200, 160, 190, 150, 210, 130, 170, 230, 220, 210, 140, 220, 160, 90, 70,
            180, 300, 160, 100, 80, 140, 70, 70, 200, 160, 180, 290, 90, 70, 0, 360, 200, 140, 250, 10,
            90, 360, 450, 400, 400, 410, 470, 340, 220, 230, 310, 300, 310, 300, 250, 320, 190, 200, 260, 260, 400],
    'PP homo-polymer fiber': [1203, 1223, 1197, 1070, 1055, 1008, 1019, 979, 995, 968, 977, 917, 917,
                              847, 893, 938, 949, 936, 921, 997, 1012, 1194, 1434, 1844, 2066, 1962,
                              1630, 1538, 1631, 1631, 1275, 1712, 1697, 1676, 1654, 1860, 2043, 1790,
                              1648, 1466, 1248, 1191, 1275, 1265, 1220, 1227, 1250, 1275, 1221, 1121,
                              1062, 1019, 1065, 1141, 1145, 1057, 1029, 1237, 1254, 1250],
    'rPET': [360, 360, 350, 330, 320, 340, 340, 340, 400, 370, 330, 310, 380, 370, 360, 340, 290, 300,
             260, 300, 340, 380, 380, 380, 380, 390, 430, 480, 440, 420, 520, 520, 480, 720, 350, 390,
             360, 600, 820, 720, 560, 480, 490, 550, 700, 610, 700, 550, 550, 570, 480, 450, 460, 430,
             480, 450, 410, 380, 420, 410],
    'rLDPE Film': [550, 580, 520, 540, 500, 480, 510, 490, 510, 480, 470, 530, 530, 520, 500, 500, 430,
                   430, 470, 470, 450, 510, 470, 490, 530, 490, 470, 450, 420, 380, 340, 430, 470, 480,
                   430, 650, 640, 640, 790, 610, 500, 560, 600, 520, 490, 480, 470, 390, 410, 730, 180,
                   290, 180, 180, 290, 360, 410, 560, 550, 470],
    'rPP': [590, 600, 600, 580, 570, 570, 550, 570, 570, 560, 530, 520, 510, 530, 520, 460, 450, 490,
            480, 480, 510, 510, 520, 530, 550, 600, 620, 560, 570, 620, 650, 730, 630, 650, 530, 790,
            780, 670, 860, 830, 750, 790, 900, 840, 770, 780, 760, 740, 750, 740, 740, 700, 650, 640,
            640, 600, 580, 600, 620, 730],
    'rHDPE Film': [570, 580, 600, 620, 600, 560, 570, 600, 600, 570, 550, 520, 520, 510, 490, 510, 490,
                   490, 490, 490, 490, 500, 520, 580, 620, 620, 640, 600, 630, 620, 570, 580, 600, 750,
                   590, 710, 630, 740, 1010, 1020, 780, 770, 760, 720, 650, 760, 750, 730, 660, 640,
                   670, 630, 610, 610, 640, 580, 580, 580, 570, 520]
}

# Convert 'Month' column to datetime and set it as index
df = pd.DataFrame(data)
df['Month'] = pd.to_datetime(df['Month'], format='%b-%y')
df.set_index('Month', inplace=True)

# Function to calculate and print IQR and quartile ranges
def calculate_iqr_and_quartiles(df):
    for column in df.columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        print(f"{column}:")
        print(f"  Q1 (25th percentile): {Q1}")
        print(f"  Q3 (75th percentile): {Q3}")
        print(f"  IQR: {IQR}\n")

# Calculate and print IQR and quartile ranges
calculate_iqr_and_quartiles(df)


# **Python Code for Monte Carlo Simulation Normal Distribution and VaR Backtesting model validation and Expected Violations based on Historical Data**

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set the aesthetic parameters for seaborn
sns.set(style="whitegrid")

# Define the Historic data for VAR Backtesting
data = {
    'Month': ['Apr-19', 'May-19', 'Jun-19', 'Jul-19', 'Aug-19', 'Sep-19', 'Oct-19', 'Nov-19', 'Dec-19',
              'Jan-20', 'Feb-20', 'Mar-20', 'Apr-20', 'May-20', 'Jun-20', 'Jul-20', 'Aug-20', 'Sep-20',
              'Oct-20', 'Nov-20', 'Dec-20', 'Jan-21', 'Feb-21', 'Mar-21', 'Apr-21', 'May-21', 'Jun-21',
              'Jul-21', 'Aug-21', 'Sep-21', 'Oct-21', 'Nov-21', 'Dec-21', 'Jan-22', 'Feb-22', 'Mar-22',
              'Apr-22', 'May-22', 'Jun-22', 'Jul-22', 'Aug-22', 'Sep-22', 'Oct-22', 'Nov-22', 'Dec-22',
              'Jan-23', 'Feb-23', 'Mar-23', 'Apr-23', 'May-23', 'Jun-23', 'Jul-23', 'Aug-23', 'Sep-23',
              'Oct-23', 'Nov-23', 'Dec-23', 'Jan-24', 'Feb-24', 'Mar-24'],
    'HDPE film': [1166, 1182, 1166, 1102, 1074, 1027, 1037, 988, 967, 958, 958, 819, 817, 757, 809, 878,
                  914, 911, 882, 948, 958, 1140, 1435, 1744, 1858, 1523, 1465, 1350, 1354, 1413, 1388,
                  1595, 1553, 1543, 1547, 1762, 1968, 1729, 1708, 1577, 1443, 1355, 1388, 1393, 1339,
                  1313, 1299, 1334, 1229, 1122, 1065, 1016, 1118, 1158, 1175, 1076, 1063, 1186, 1265, 1280],
    'LDPE film': [1096, 1120, 1105, 1047, 1029, 964, 985, 953, 952, 944, 953, 836, 828, 762, 840, 917,
                  935, 918, 912, 1024, 1051, 1288, 1562, 1920, 2082, 1900, 1816, 1710, 1748, 1767, 1486,
                  1914, 1872, 1851, 1845, 2033, 2225, 1932, 1893, 1660, 1547, 1407, 1486, 1431, 1394,
                  1352, 1330, 1349, 1261, 1095, 979, 986, 1090, 1162, 1221, 1067, 1059, 1290, 1351, 1060],
    'PET': [170, 200, 200, 180, 200, 160, 190, 150, 210, 130, 170, 230, 220, 210, 140, 220, 160, 90, 70,
            180, 300, 160, 100, 80, 140, 70, 70, 200, 160, 180, 290, 90, 70, 0, 360, 200, 140, 250, 10,
            90, 360, 450, 400, 400, 410, 470, 340, 220, 230, 310, 300, 310, 300, 250, 320, 190, 200, 260, 260, 400],
    'PP homo-polymer fiber': [1203, 1223, 1197, 1070, 1055, 1008, 1019, 979, 995, 968, 977, 917, 917,
                              847, 893, 938, 949, 936, 921, 997, 1012, 1194, 1434, 1844, 2066, 1962,
                              1630, 1538, 1631, 1631, 1275, 1712, 1697, 1676, 1654, 1860, 2043, 1790,
                              1648, 1466, 1248, 1191, 1275, 1265, 1220, 1227, 1250, 1275, 1221, 1121,
                              1062, 1019, 1065, 1141, 1145, 1057, 1029, 1237, 1254, 1250],
    'rPET': [360, 360, 350, 330, 320, 340, 340, 340, 400, 370, 330, 310, 380, 370, 360, 340, 290, 300, 260, 300, 340, 380, 380, 380, 380, 390, 430, 480, 440, 420, 520, 520, 480, 720, 350, 390, 360, 600, 820, 720, 560, 480, 490, 550, 700, 610, 700, 550, 550, 570, 480, 450, 460, 430, 480, 450, 410, 380, 420, 410],
    'rLDPE Film': [550, 580, 520, 540, 500, 480, 510, 490, 510, 480, 470, 530, 530, 520, 500, 500, 430, 430, 470, 470, 450, 510, 470, 490, 530, 490, 470, 450, 420, 380, 340, 430, 470, 480, 430, 650, 640, 640, 790, 610, 500, 560, 600, 520, 490, 480, 470, 390, 410, 730, 180, 290, 180, 180, 290, 360, 410, 560, 550, 470],
    'rPP': [590, 600, 600, 580, 570, 570, 550, 570, 570, 560, 530, 520, 510, 530, 520, 460, 450, 490, 480, 480, 510, 510, 520, 530, 550, 600, 620, 560, 570, 620, 650, 730, 630, 650, 530, 790, 780, 670, 860, 830, 750, 790, 900, 840, 770, 780, 760, 740, 750, 740, 740, 700, 650, 640, 640, 600, 580, 600, 620, 730],
    'rHDPE Film': [570, 580, 600, 620, 600, 560, 570, 600, 600, 570, 550, 520, 520, 510, 490, 510, 490, 490, 490, 490, 490, 500, 520, 580, 620, 620, 640, 600, 630, 620, 570, 580, 600, 750, 590, 710, 630, 740, 1010, 1020, 780, 770, 760, 720, 650, 760, 750, 730, 660, 640, 670, 630, 610, 610, 640, 580, 580, 580, 570, 520]
}

# Create DataFrame from the data dictionary
df = pd.DataFrame(data)
df['Month'] = pd.to_datetime(df['Month'], format='%b-%y')
df.set_index('Month', inplace=True)

# Data Preprocessing: Identify and treat outliers using the IQR method
def treat_outliers_iqr(df):
    for column in df.columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
        df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
    return df

# Treat outliers in the DataFrame
df = treat_outliers_iqr(df)

# Monte Carlo Simulation function using Geometric Brownian Motion (GBM)
def monte_carlo_simulation(s0, mu, sigma, n, num_scenarios, dt):
    prices = np.zeros((num_scenarios, n))
    for i in range(num_scenarios):
        price = np.zeros(n)
        price[0] = s0
        for t in range(1, n):
            simulated_return = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.normal()
            next_price = price[t - 1] * np.exp(simulated_return)
            if next_price <= 0:
                prices[i, t:] = np.nan
                break
            price[t] = next_price
        prices[i, :t+1] = price[:t+1]
    return prices

# Simulation parameters
num_scenarios = 10000
n_months_future = 60
dt = 1 / 252

# Initialize simulated prices dictionary
simulated_prices_future = {}

# Loop over each polymer compound and perform simulation
for material in df.columns:
    s0_future = df[material].iloc[-1]
    mu = 0.003
    sigma = 0.27124
    simulated_prices_future[material] = monte_carlo_simulation(s0_future, mu, sigma, n_months_future, num_scenarios, dt)

# Function to calculate VaR and ETL
def calculate_var_etl(prices, confidence_level=0.95):
    sorted_prices = np.sort(prices)
    index_at_var = int((1 - confidence_level) * len(sorted_prices))
    var = sorted_prices[index_at_var]
    etl = sorted_prices[:index_at_var].mean()
    return var, etl

# Function to backtest VaR
def backtest_var(df, var, confidence_level=0.95):
    returns = df.pct_change().dropna()
    var_violations = (returns < -var).sum()
    expected_violations = (1 - confidence_level) * len(returns)
    return var_violations, expected_violations

# Calculate historical VaR for each material
historical_var = {}
for material in df.columns:
    returns = df[material].pct_change().dropna()
    historical_var[material] = calculate_var_etl(returns, confidence_level=0.95)[0]

# Backtest VaR for each material
backtest_results = {}
for material in df.columns:
    var_violations, expected_violations = backtest_var(df[material], historical_var[material])
    backtest_results[material] = (var_violations, expected_violations)

# Print backtest results
for material, result in backtest_results.items():
    print(f"\n{material}:")
    print(f"VaR Violations: {result[0]}")
    print(f"Expected Violations: {result[1]}")

# Plotting and calculating metrics for each material
for material in df.columns:
    prices = simulated_prices_future[material][:, -1]
    median_price = np.median(prices)
    var, etl = calculate_var_etl(prices)
    confidence_interval = np.percentile(prices, [2.5, 97.5])

    print(f"\n{material}:")
    print(f"Median Price: €{median_price:.2f}")
    print(f"95% Confidence Interval: €{confidence_interval[0]:.2f} - €{confidence_interval[1]:.2f}")
    print(f"Value at Risk (VaR) at 95% confidence: €{var:.2f}")
    print(f"Expected Tail Loss (ETL): €{etl:.2f}")

    plt.figure(figsize=(10, 6))
    sns.histplot(prices, bins=50, kde=True)
    plt.axvline(median_price, color='r', linestyle='dashed', linewidth=2, label='Median')
    plt.axvline(var, color='g', linestyle='dashed', linewidth=2, label='VaR (95%)')
    plt.axvline(etl, color='b', linestyle='dashed', linewidth=2, label='ETL')
    plt.title(f'Distribution of Simulated Prices for {material}')
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()


# **Python Code for Monte Carlo Simulation Normal Distribution and VaR Backtesting model validation and Expected Violations based on Forecasted Data**

# In[4]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Data for Forecasted Prices for backtesting

# Set the aesthetic parameters for seaborn
sns.set(style="whitegrid")

# Data for HDPE film
hdpe_data = {
    "Month/Year": [
        "April 2024", "May 2024", "June 2024", "July 2024", "August 2024", "September 2024", "October 2024", "November 2024", "December 2024",
        "January 2025", "February 2025", "March 2025", "April 2025", "May 2025", "June 2025", "July 2025", "August 2025", "September 2025", "October 2025", "November 2025", "December 2025",
        "January 2026", "February 2026", "March 2026", "April 2026", "May 2026", "June 2026", "July 2026", "August 2026", "September 2026", "October 2026", "November 2026", "December 2026",
        "January 2027", "February 2027", "March 2027", "April 2027", "May 2027", "June 2027", "July 2027", "August 2027", "September 2027", "October 2027", "November 2027", "December 2027",
        "January 2028", "February 2028", "March 2028", "April 2028", "May 2028", "June 2028", "July 2028", "August 2028", "September 2028", "October 2028", "November 2028", "December 2028",
        "January 2029", "February 2029", "March 2029"
    ],
    "HDPE film": [
        1280.00, 1280.05, 1280.11, 1280.21, 1280.25, 1280.46, 1280.45, 1280.31, 1280.63,
        1280.60, 1280.88, 1280.85, 1280.97, 1281.03, 1281.24, 1281.32, 1281.78, 1281.76, 1281.60, 1281.66, 1281.88,
        1281.99, 1281.85, 1282.05, 1281.95, 1281.65, 1281.31, 1281.32, 1281.52, 1281.43, 1281.19, 1280.97, 1281.09,
        1281.04, 1281.19, 1281.04, 1280.96, 1281.04, 1281.10, 1281.06, 1281.07, 1281.36, 1281.72, 1281.45, 1281.51,
        1281.35, 1281.08, 1281.32, 1281.15, 1281.07, 1281.29, 1281.00, 1280.72, 1280.93, 1281.50, 1281.56, 1281.52,
        1281.42, 1281.71, 1281.91
    ]
}

# Data for LDPE film
ldpe_data = {
    "Month/Year": hdpe_data["Month/Year"],
    "LDPE film": [
        1060.00, 1060.25, 1060.44, 1060.29, 1060.07, 1059.84, 1059.82, 1060.14, 1060.18,
        1060.40, 1060.71, 1060.70, 1060.72, 1060.75, 1061.26, 1061.46, 1061.38, 1061.52, 1061.69, 1061.93, 1062.06,
        1061.98, 1061.60, 1061.99, 1062.07, 1062.10, 1062.14, 1062.26, 1062.29, 1062.10, 1061.88, 1061.73, 1061.96,
        1062.03, 1062.09, 1061.73, 1061.98, 1061.62, 1061.43, 1061.31, 1061.45, 1061.70, 1061.53, 1061.53, 1061.69,
        1062.11, 1062.14, 1062.00, 1062.22, 1062.11, 1062.02, 1062.17, 1062.18, 1062.05, 1061.77, 1062.01, 1061.85,
        1061.61, 1061.08, 1060.91
    ]
}

# Data for PET
pet_data = {
    "Month/Year": hdpe_data["Month/Year"],
    "PET": [
        400.00, 399.95, 399.94, 399.93, 399.98, 399.97, 400.01, 399.97, 399.95,
        400.00, 399.99, 399.95, 400.05, 400.02, 399.99, 400.00, 399.92, 400.03, 400.05, 400.04, 400.07,
        400.06, 399.99, 399.97, 399.96, 399.93, 400.01, 400.11, 400.12, 400.07, 400.18, 400.34, 400.36,
        400.41, 400.45, 400.56, 400.56, 400.48, 400.39, 400.48, 400.37, 400.42, 400.46, 400.52, 400.48,
        400.40, 400.41, 400.37, 400.42, 400.39, 400.34, 400.36, 400.37, 400.33, 400.35, 400.27, 400.20,
        400.19, 400.11, 399.98
    ]
}

# Data for PP homo-polymer fiber
pp_data = {
    "Month/Year": hdpe_data["Month/Year"],
    "PP homo-polymer fiber": [
        1250.00, 1250.26, 1250.58, 1250.66, 1250.94, 1250.94, 1250.90, 1250.97, 1251.14,
        1250.97, 1251.22, 1251.38, 1251.58, 1251.68, 1251.64, 1251.69, 1251.63, 1251.99, 1251.77, 1251.79, 1252.01,
        1251.69, 1251.71, 1251.81, 1251.85, 1252.10, 1251.95, 1251.91, 1251.61, 1251.63, 1252.14, 1252.20, 1252.22,
        1252.06, 1252.15, 1252.10, 1251.97, 1252.08, 1252.05, 1252.16, 1251.72, 1251.65, 1251.68, 1251.33, 1251.20,
        1251.20, 1250.99, 1250.69, 1250.45, 1250.37, 1250.35, 1250.50, 1250.05, 1249.74, 1249.72, 1250.10, 1250.33,
        1250.44, 1250.32, 1250.15
    ]
}

# Create DataFrames
df_hdpe = pd.DataFrame(hdpe_data)
df_ldpe = pd.DataFrame(ldpe_data)
df_pet = pd.DataFrame(pet_data)
df_pp = pd.DataFrame(pp_data)

# Merge DataFrames on "Month/Year"
df = df_hdpe.merge(df_ldpe, on="Month/Year").merge(df_pet, on="Month/Year").merge(df_pp, on="Month/Year")

# Convert "Month/Year" to datetime
df["Month/Year"] = pd.to_datetime(df["Month/Year"], format="%B %Y")
df.set_index("Month/Year", inplace=True)

# Data Preprocessing: Identify and treat outliers using the IQR method
def treat_outliers_iqr(df):
    for column in df.columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
        df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
    return df

# Treat outliers in the DataFrame
df = treat_outliers_iqr(df)

# Monte Carlo Simulation function using Geometric Brownian Motion (GBM)
def monte_carlo_simulation(s0, mu, sigma, n, num_scenarios, dt):
    prices = np.zeros((num_scenarios, n))
    for i in range(num_scenarios):
        price = np.zeros(n)
        price[0] = s0
        for t in range(1, n):
            simulated_return = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.normal()
            next_price = price[t - 1] * np.exp(simulated_return)
            if next_price <= 0:
                prices[i, t:] = np.nan
                break
            price[t] = next_price
        prices[i, :t+1] = price[:t+1]
    return prices

# Simulation parameters
num_scenarios = 10000
n_months_future = 60
dt = 1 / 252

# Initialize simulated prices dictionary
simulated_prices_future = {}

# Loop over each polymer compound and perform simulation
for material in df.columns:
    s0_future = df[material].iloc[-1]
    mu = 0.003
    sigma = 0.27124
    simulated_prices_future[material] = monte_carlo_simulation(s0_future, mu, sigma, n_months_future, num_scenarios, dt)

# Function to calculate VaR and ETL
def calculate_var_etl(prices, confidence_level=0.95):
    sorted_prices = np.sort(prices)
    index_at_var = int((1 - confidence_level) * len(sorted_prices))
    var = sorted_prices[index_at_var]
    etl = sorted_prices[:index_at_var].mean()
    return var, etl

# Function to backtest VaR
def backtest_var(df, var, confidence_level=0.95):
    returns = df.pct_change().dropna()
    var_violations = (returns < -var).sum()
    expected_violations = (1 - confidence_level) * len(returns)
    return var_violations, expected_violations

# Calculate historical VaR for each material
historical_var = {}
for material in df.columns:
    returns = df[material].pct_change().dropna()
    historical_var[material] = calculate_var_etl(returns, confidence_level=0.95)[0]

# Backtest VaR for each material
backtest_results = {}
for material in df.columns:
    var_violations, expected_violations = backtest_var(df[material], historical_var[material])
    backtest_results[material] = (var_violations, expected_violations)

# Print backtest results
for material, result in backtest_results.items():
    print(f"\n{material}:")
    print(f"VaR Violations: {result[0]}")
    print(f"Expected Violations: {result[1]}")

# Plotting and calculating metrics for each material
for material in df.columns:
    prices = simulated_prices_future[material][:, -1]
    median_price = np.median(prices)
    var, etl = calculate_var_etl(prices)
    confidence_interval = np.percentile(prices, [2.5, 97.5])

    print(f"\n{material}:")
    print(f"Median Price: €{median_price:.2f}")
    print(f"95% Confidence Interval: €{confidence_interval[0]:.2f} - €{confidence_interval[1]:.2f}")
    print(f"Value at Risk (VaR) at 95% confidence: €{var:.2f}")
    print(f"Expected Tail Loss (ETL): €{etl:.2f}")

    plt.figure(figsize=(10, 6))
    sns.histplot(prices, bins=50, kde=True)
    plt.axvline(median_price, color='r', linestyle='dashed', linewidth=2, label='Median')
    plt.axvline(var, color='g', linestyle='dashed', linewidth=2, label='VaR (95%)')
    plt.axvline(etl, color='b', linestyle='dashed', linewidth=2, label='ETL')
    plt.title(f'Distribution of Simulated Prices for {material}')
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()


# **Summary Statistics for VaR Backtesting model validation comparing Expected VaR Violations in both Historical v/s Forecasted Data**

# In[7]:


import pandas as pd

# Data for VaR backtesting results
data = {
    "Polymer Type": ["HDPE film", "LDPE film", "PET", "PP homo-polymer fiber"],
    "VaR Violations (Historical)": [54, 54, 52, 54],
    "Expected Violations (Historical)": [2.95, 2.95, 2.95, 2.95],
    "VaR Violations (Forecasted)": [54, 56, 52, 56],
    "Expected Violations (Forecasted)": [2.95, 2.95, 2.95, 2.95]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Print the DataFrame
print(df)

# Describe the DataFrame
print("\nDescription of VaR Backtesting Results:")
print(df.describe())

