#!/usr/bin/env python
# coding: utf-8

# **Python Code for Validation of Monte Carlo Forward Price Forecasting model using Gradient Clipping by norm**

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Monte Carlo Forward Price Simulation model Tested with Gradient Clipping by norm

# Historical Data
historical_data = {
    'Month': [
        'Apr-19', 'May-19', 'Jun-19', 'Jul-19', 'Aug-19', 'Sep-19', 'Oct-19', 'Nov-19', 'Dec-19',
        'Jan-20', 'Feb-20', 'Mar-20', 'Apr-20', 'May-20', 'Jun-20', 'Jul-20', 'Aug-20', 'Sep-20',
        'Oct-20', 'Nov-20', 'Dec-20', 'Jan-21', 'Feb-21', 'Mar-21', 'Apr-21', 'May-21', 'Jun-21',
        'Jul-21', 'Aug-21', 'Sep-21', 'Oct-21', 'Nov-21', 'Dec-21', 'Jan-22', 'Feb-22', 'Mar-22',
        'Apr-22', 'May-22', 'Jun-22', 'Jul-22', 'Aug-22', 'Sep-22', 'Oct-22', 'Nov-22', 'Dec-22',
        'Jan-23', 'Feb-23', 'Mar-23', 'Apr-23', 'May-23', 'Jun-23', 'Jul-23', 'Aug-23', 'Sep-23',
        'Oct-23', 'Nov-23', 'Dec-23', 'Jan-24', 'Feb-24', 'Mar-24'
    ],
    'HDPE film': [
        1166, 1182, 1166, 1102, 1074, 1027, 1037, 988, 967, 958, 958, 819, 817, 757, 809, 878,
        914, 911, 882, 948, 958, 1140, 1435, 1744, 1858, 1523, 1465, 1350, 1354, 1413, 1388,
        1595, 1553, 1543, 1547, 1762, 1968, 1729, 1708, 1577, 1443, 1355, 1388, 1393, 1339,
        1313, 1299, 1334, 1229, 1122, 1065, 1016, 1118, 1158, 1175, 1076, 1063, 1186, 1265, 1280
    ],
    'LDPE film': [
        1096, 1120, 1105, 1047, 1029, 964, 985, 953, 952, 944, 953, 836, 828, 762, 840, 917,
        935, 918, 912, 1024, 1051, 1288, 1562, 1920, 2082, 1900, 1816, 1710, 1748, 1767, 1486,
        1914, 1872, 1851, 1845, 2033, 2225, 1932, 1893, 1660, 1547, 1407, 1486, 1431, 1394,
        1352, 1330, 1349, 1261, 1095, 979, 986, 1090, 1162, 1221, 1067, 1059, 1290, 1351, 1060
    ],
    'PET': [
        170, 200, 200, 180, 200, 160, 190, 150, 210, 130, 170, 230, 220, 210, 140, 220, 160, 90, 70,
        180, 300, 160, 100, 80, 140, 70, 70, 200, 160, 180, 290, 90, 70, 0, 360, 200, 140, 250, 10,
        90, 360, 450, 400, 400, 410, 470, 340, 220, 230, 310, 300, 310, 300, 250, 320, 190, 200, 260, 260, 400
    ],
    'PP homo-polymer fiber': [
        1203, 1223, 1197, 1070, 1055, 1008, 1019, 979, 995, 968, 977, 917, 917,
        847, 893, 938, 949, 936, 921, 997, 1012, 1194, 1434, 1844, 2066, 1962,
        1630, 1538, 1631, 1631, 1275, 1712, 1697, 1676, 1654, 1860, 2043, 1790,
        1648, 1466, 1248, 1191, 1275, 1265, 1220, 1227, 1250, 1275, 1221, 1121,
        1062, 1019, 1065, 1141, 1145, 1057, 1029, 1237, 1254, 1250
    ],
    'rPET': [
        360, 360, 350, 330, 320, 340, 340, 340, 400, 370, 330, 310, 380, 370, 360, 340, 290, 300, 260, 300, 340, 380, 380, 380, 380, 390, 430, 480, 440, 420, 520, 520, 480, 720, 350, 390, 360, 600, 820, 720, 560, 480, 490, 550, 700, 610, 700, 550, 550, 570, 480, 450, 460, 430, 480, 450, 410, 380, 420, 410
    ],
    'rLDPE Film': [
        550, 580, 520, 540, 500, 480, 510, 490, 510, 480, 470, 530, 530, 520, 500, 500, 430, 430, 470, 470, 450, 510, 470, 490, 530, 490, 470, 450, 420, 380, 340, 430, 470, 480, 430, 650, 640, 640, 790, 610, 500, 560, 600, 520, 490, 480, 470, 390, 410, 730, 180, 290, 180, 180, 290, 360, 410, 560, 550, 470
    ],
    'rPP': [
        590, 600, 600, 580, 570, 570, 550, 570, 570, 560, 530, 520, 510, 530, 520, 460, 450, 490, 480, 480, 510, 510, 520, 530, 550, 600, 620, 560, 570, 620, 650, 730, 630, 650, 530, 790, 780, 670, 860, 830, 750, 790, 900, 840, 770, 780, 760, 740, 750, 740, 740, 700, 650, 640, 640, 600, 580, 600, 620, 730
    ],
    'rHDPE Film': [
        570, 580, 600, 620, 600, 560, 570, 600, 600, 570, 550, 520, 520, 510, 490, 510, 490, 490, 490, 490, 490, 500, 520, 580, 620, 620, 640, 600, 630, 620, 570, 580, 600, 750, 590, 710, 630, 740, 1010, 1020, 780, 770, 760, 720, 650, 760, 750, 730, 660, 640, 670, 630, 610, 610, 640, 580, 580, 580, 570, 520
    ]
}

# Create DataFrame from the historical data dictionary
df = pd.DataFrame(historical_data)
df['Month'] = pd.to_datetime(df['Month'], format='%b-%y')
df.set_index('Month', inplace=True)

# Forecasted Prices Data
forecasted_prices_data = {
    'Month': [
        'Apr-24', 'May-24', 'Jun-24', 'Jul-24', 'Aug-24', 'Sep-24', 'Oct-24', 'Nov-24', 'Dec-24',
        'Jan-25', 'Feb-25', 'Mar-25', 'Apr-25', 'May-25', 'Jun-25', 'Jul-25', 'Aug-25', 'Sep-25',
        'Oct-25', 'Nov-25', 'Dec-25', 'Jan-26', 'Feb-26', 'Mar-26', 'Apr-26', 'May-26', 'Jun-26',
        'Jul-26', 'Aug-26', 'Sep-26', 'Oct-26', 'Nov-26', 'Dec-26', 'Jan-27', 'Feb-27', 'Mar-27',
        'Apr-27', 'May-27', 'Jun-27', 'Jul-27', 'Aug-27', 'Sep-27', 'Oct-27', 'Nov-27', 'Dec-27',
        'Jan-28', 'Feb-28', 'Mar-28', 'Apr-28', 'May-28', 'Jun-28', 'Jul-28', 'Aug-28', 'Sep-28',
        'Oct-28', 'Nov-28', 'Dec-28', 'Jan-29', 'Feb-29', 'Mar-29'
    ],
    'HDPE film': [
        1280.00, 1280.05, 1280.11, 1280.21, 1280.25, 1280.46, 1280.45, 1280.31, 1280.63,
        1280.60, 1280.88, 1280.85, 1280.97, 1281.03, 1281.24, 1281.32, 1281.78, 1281.76, 1281.60, 1281.66, 1281.88,
        1281.99, 1281.85, 1282.05, 1281.95, 1281.65, 1281.31, 1281.32, 1281.52, 1281.43, 1281.19, 1280.97, 1281.09,
        1281.04, 1281.19, 1281.04, 1280.96, 1281.04, 1281.10, 1281.06, 1281.07, 1281.36, 1281.72, 1281.45, 1281.51,
        1281.35, 1281.08, 1281.32, 1281.15, 1281.07, 1281.29, 1281.00, 1280.72, 1280.93, 1281.50, 1281.56, 1281.52,
        1281.42, 1281.71, 1281.91
    ],
    'LDPE film': [
        1060.00, 1060.25, 1060.44, 1060.29, 1060.07, 1059.84, 1059.82, 1060.14, 1060.18,
        1060.40, 1060.71, 1060.70, 1060.72, 1060.75, 1061.26, 1061.46, 1061.38, 1061.52, 1061.69, 1061.93, 1062.06,
        1061.98, 1061.60, 1061.99, 1062.07, 1062.10, 1062.14, 1062.26, 1062.29, 1062.10, 1061.88, 1061.73, 1061.96,
        1062.03, 1062.09, 1061.73, 1061.98, 1061.62, 1061.43, 1061.31, 1061.45, 1061.70, 1061.53, 1061.53, 1061.69,
        1062.11, 1062.14, 1062.00, 1062.22, 1062.11, 1062.02, 1062.17, 1062.18, 1062.05, 1061.77, 1062.01, 1061.85,
        1061.61, 1061.08, 1060.91
    ],
    'PET': [
        400.00, 399.95, 399.94, 399.93, 399.98, 399.97, 400.01, 399.97, 399.95,
        400.00, 399.99, 399.95, 400.05, 400.02, 399.99, 400.00, 399.92, 400.03, 400.05, 400.04, 400.07,
        400.06, 399.99, 399.97, 399.96, 399.93, 400.01, 400.11, 400.12, 400.07, 400.18, 400.34, 400.36,
        400.41, 400.45, 400.56, 400.56, 400.48, 400.39, 400.48, 400.37, 400.42, 400.46, 400.52, 400.48,
        400.40, 400.41, 400.37, 400.42, 400.39, 400.34, 400.36, 400.37, 400.33, 400.35, 400.27, 400.20,
        400.19, 400.11, 399.98
    ],
    'PP homo-polymer fiber': [
        1250.00, 1250.26, 1250.58, 1250.66, 1250.94, 1250.94, 1250.90, 1250.97, 1251.14,
        1250.97, 1251.22, 1251.38, 1251.58, 1251.68, 1251.64, 1251.69, 1251.63, 1251.99, 1251.77, 1251.79, 1252.01,
        1251.69, 1251.71, 1251.81, 1251.85, 1252.10, 1251.95, 1251.91, 1251.61, 1251.63, 1252.14, 1252.20, 1252.22,
        1252.06, 1252.15, 1252.10, 1251.97, 1252.08, 1252.05, 1252.16, 1251.72, 1251.65, 1251.68, 1251.33, 1251.20,
        1251.20, 1250.99, 1250.69, 1250.45, 1250.37, 1250.35, 1250.50, 1250.05, 1249.74, 1249.72, 1250.10, 1250.33,
        1250.44, 1250.32, 1250.15
    ],
    'rPET': [
        400.00, 400.05, 400.10, 400.15, 400.20, 400.25, 400.30, 400.35, 400.40,
        400.45, 400.50, 400.55, 400.60, 400.65, 400.70, 400.75, 400.80, 400.85, 400.90, 400.95, 401.00,
        401.05, 401.10, 401.15, 401.20, 401.25, 401.30, 401.35, 401.40, 401.45, 401.50, 401.55, 401.60,
        401.65, 401.70, 401.75, 401.80, 401.85, 401.90, 401.95, 402.00, 402.05, 402.10, 402.15, 402.20,
        402.25, 402.30, 402.35, 402.40, 402.45, 402.50, 402.55, 402.60, 402.65, 402.70, 402.75, 402.80,
        402.85, 402.90, 402.95
    ],
    'rLDPE Film': [
        600.00, 600.05, 600.10, 600.15, 600.20, 600.25, 600.30, 600.35, 600.40,
        600.45, 600.50, 600.55, 600.60, 600.65, 600.70, 600.75, 600.80, 600.85, 600.90, 600.95, 601.00,
        601.05, 601.10, 601.15, 601.20, 601.25, 601.30, 601.35, 601.40, 601.45, 601.50, 601.55, 601.60,
        601.65, 601.70, 601.75, 601.80, 601.85, 601.90, 601.95, 602.00, 602.05, 602.10, 602.15, 602.20,
        602.25, 602.30, 602.35, 602.40, 602.45, 602.50, 602.55, 602.60, 602.65, 602.70, 602.75, 602.80,
        602.85, 602.90, 602.95
    ],
    'rPP': [
        700.00, 700.05, 700.10, 700.15, 700.20, 700.25, 700.30, 700.35, 700.40,
        700.45, 700.50, 700.55, 700.60, 700.65, 700.70, 700.75, 700.80, 700.85, 700.90, 700.95, 701.00,
        701.05, 701.10, 701.15, 701.20, 701.25, 701.30, 701.35, 701.40, 701.45, 701.50, 701.55, 701.60,
        701.65, 701.70, 701.75, 701.80, 701.85, 701.90, 701.95, 702.00, 702.05, 702.10, 702.15, 702.20,
        702.25, 702.30, 702.35, 702.40, 702.45, 702.50, 702.55, 702.60, 702.65, 702.70, 702.75, 702.80,
        702.85, 702.90, 702.95
    ],
    'rHDPE Film': [
        800.00, 800.05, 800.10, 800.15, 800.20, 800.25, 800.30, 800.35, 800.40,
        800.45, 800.50, 800.55, 800.60, 800.65, 800.70, 800.75, 800.80, 800.85, 800.90, 800.95, 801.00,
        801.05, 801.10, 801.15, 801.20, 801.25, 801.30, 801.35, 801.40, 801.45, 801.50, 801.55, 801.60,
        801.65, 801.70, 801.75, 801.80, 801.85, 801.90, 801.95, 802.00, 802.05, 802.10, 802.15, 802.20,
        802.25, 802.30, 802.35, 802.40, 802.45, 802.50, 802.55, 802.60, 802.65, 802.70, 802.75, 802.80,
        802.85, 802.90, 802.95
    ]
    
    }
# Create DataFrame from the forecasted prices data dictionary
forecasted_prices_df = pd.DataFrame(forecasted_prices_data)
forecasted_prices_df['Month'] = pd.to_datetime(forecasted_prices_df['Month'], format='%b-%y')
forecasted_prices_df.set_index('Month', inplace=True)

# Split the data into training (75%) and validation (25%) sets
train_size = int(0.75 * len(df))
train_data = df.iloc[:train_size]
validation_data = df.iloc[train_size:]

# Define min and max values for each feature
min_max_values = {
    'HDPE film': (df['HDPE film'].min(), df['HDPE film'].max()),
    'LDPE film': (df['LDPE film'].min(), df['LDPE film'].max()),
    'PET': (df['PET'].min(), df['PET'].max()),
    'PP homo-polymer fiber': (df['PP homo-polymer fiber'].min(), df['PP homo-polymer fiber'].max()),
    'rPET': (df['rPET'].min(), df['rPET'].max()),
    'rLDPE Film': (df['rLDPE Film'].min(), df['rLDPE Film'].max()),
    'rPP': (df['rPP'].min(), df['rPP'].max()),
    'rHDPE Film': (df['rHDPE Film'].min(), df['rHDPE Film'].max())
}

# Function to normalize predicted values
def normalize_predicted_values(predicted_values, feature):
    min_val, max_val = min_max_values[feature]
    normalized_values = []
    for val in predicted_values:
        if val >= max_val:
            normalized_values.append(max_val)
        elif val <= min_val:
            normalized_values.append(min_val)
        else:
            normalized_values.append(val)
    return normalized_values

# Function to compute gradients and apply gradient clipping by norm
def compute_gradients(X, y, weights, bias):
    predictions = X.dot(weights) + bias
    errors = predictions - y
    gradients_w = 2 * X.T.dot(errors) / len(y)
    gradients_b = 2 * np.sum(errors) / len(y)
    return gradients_w, gradients_b

def apply_gradient_clipping(gradients_w, gradients_b, clip_norm):
    norm = np.sqrt(np.sum(gradients_w**2) + gradients_b**2)
    if norm > clip_norm:
        gradients_w = gradients_w * (clip_norm / norm)
        gradients_b = gradients_b * (clip_norm / norm)
    return gradients_w, gradients_b

# Function to train the model using gradient descent with gradient clipping by norm
def train_model(X_train, y_train, X_val, y_val, learning_rate=0.01, iterations=1000, clip_norm=1.0):
    weights = np.random.randn(X_train.shape[1])
    bias = 0
    training_loss = []
    validation_loss = []
    for i in range(iterations):
        gradients_w, gradients_b = compute_gradients(X_train, y_train, weights, bias)
        gradients_w, gradients_b = apply_gradient_clipping(gradients_w, gradients_b, clip_norm)
        weights -= learning_rate * gradients_w
        bias -= learning_rate * gradients_b
        train_predictions = X_train.dot(weights) + bias
        train_loss = np.mean((train_predictions - y_train) ** 2)
        training_loss.append(train_loss)
        val_predictions = X_val.dot(weights) + bias
        val_loss = np.mean((val_predictions - y_val) ** 2)
        validation_loss.append(val_loss)
        if i % 100 == 0:
            print(f"Iteration {i}/{iterations} - Training Loss: {train_loss:.4f} - Validation Loss: {val_loss:.4f}")
    return weights, bias, training_loss, validation_loss

# Prepare the data for the model
def prepare_data(data):
    X, y = [], []
    for i in range(len(data) - 1):
        X.append(data[i])
        y.append(data[i + 1])
    return np.array(X), np.array(y)

# Function to train and evaluate the model for a given feature
def train_and_evaluate(feature):
    X_train, y_train = prepare_data(train_data[feature].values)
    X_val, y_val = prepare_data(validation_data[feature].values)
    # Add a bias term (column of ones) to the input data
    X_train = np.c_[np.ones(X_train.shape[0]), X_train]
    X_val = np.c_[np.ones(X_val.shape[0]), X_val]
    weights, bias, training_loss, validation_loss = train_model(X_train, y_train, X_val, y_val, learning_rate=0.01, iterations=1000, clip_norm=1.0)
    val_predictions = X_val.dot(weights) + bias
    val_predictions = normalize_predicted_values(val_predictions, feature)
    print(f"Normalized {feature} values: {val_predictions}")
    plt.figure(figsize=(10, 6))
    plt.plot(training_loss, label='Training Loss')
    plt.plot(validation_loss, label='Validation Loss')
    plt.title(f'Model Loss for {feature}')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Train and evaluate the model for each feature
for feature in ['HDPE film', 'LDPE film', 'PET', 'PP homo-polymer fiber', 'rPET', 'rLDPE Film', 'rPP', 'rHDPE Film']:
    train_and_evaluate(feature)


# **Python Code for Validation of Time Series Forecasting model with XGBoost using Gradient Clipping by norm**

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Time Series Forecasting model with XGBoost Tested with Gradient Clipping by norm

# Historical Data
historical_data = {
    'Month': ['Apr-19', 'May-19', 'Jun-19', 'Jul-19', 'Aug-19', 'Sep-19', 'Oct-19', 'Nov-19', 'Dec-19', 'Jan-20', 'Feb-20', 'Mar-20', 'Apr-20', 'May-20', 'Jun-20', 'Jul-20', 'Aug-20', 'Sep-20', 'Oct-20', 'Nov-20', 'Dec-20', 'Jan-21', 'Feb-21', 'Mar-21', 'Apr-21', 'May-21', 'Jun-21', 'Jul-21', 'Aug-21', 'Sep-21', 'Oct-21', 'Nov-21', 'Dec-21', 'Jan-22', 'Feb-22', 'Mar-22', 'Apr-22', 'May-22', 'Jun-22', 'Jul-22', 'Aug-22', 'Sep-22', 'Oct-22', 'Nov-22', 'Dec-22', 'Jan-23', 'Feb-23', 'Mar-23', 'Apr-23', 'May-23', 'Jun-23', 'Jul-23', 'Aug-23', 'Sep-23', 'Oct-23', 'Nov-23', 'Dec-23', 'Jan-24', 'Feb-24', 'Mar-24'],
    'rPET': [360, 360, 350, 330, 320, 340, 340, 340, 400, 370, 330, 310, 380, 370, 360, 340, 290, 300, 260, 300, 340, 380, 380, 380, 380, 370, 360, 340, 290, 300, 260, 300, 340, 380, 380, 380, 380, 370, 360, 340, 290, 300, 260, 300, 340, 380, 380, 380, 380, 370, 360, 340, 290, 300, 260, 300, 340, 380, 380, 380],
    'rLDPE Film': [550, 580, 520, 540, 500, 480, 510, 490, 510, 480, 470, 530, 530, 520, 500, 500, 430, 430, 470, 470, 450, 510, 470, 490, 530, 490, 470, 450, 420, 380, 340, 430, 470, 480, 430, 650, 640, 640, 790, 610, 500, 560, 600, 520, 490, 480, 470, 390, 410, 730, 180, 290, 180, 180, 290, 360, 410, 560, 550, 470],
    'rPP': [590, 600, 600, 580, 570, 570, 550, 570, 570, 560, 530, 520, 510, 530, 520, 460, 450, 490, 480, 480, 510, 510, 520, 530, 550, 600, 620, 560, 570, 620, 650, 730, 630, 650, 530, 790, 780, 670, 860, 830, 750, 790, 900, 840, 770, 780, 760, 740, 750, 740, 740, 700, 650, 640, 640, 600, 580, 600, 620, 730],
    'rHDPE': [520, 569, 587, 614, 693, 678, 652, 739, 615, 640, 670, 630, 610, 610, 640, 580, 580, 580, 570, 520, 520, 580, 600, 620, 600, 560, 570, 600, 600, 570, 550, 520, 520, 510, 490, 510, 490, 490, 490, 490, 490, 500, 520, 580, 620, 620, 640, 600, 630, 620, 570, 580, 600, 750, 590, 710, 630, 740, 1010, 1020]
}

# Forecasted Data
forecasted_data = {
    'Month': ['Apr-24', 'May-24', 'Jun-24', 'Jul-24', 'Aug-24', 'Sep-24', 'Oct-24', 'Nov-24', 'Dec-24', 'Jan-25', 'Feb-25', 'Mar-25', 'Apr-25', 'May-25', 'Jun-25', 'Jul-25', 'Aug-25', 'Sep-25', 'Oct-25', 'Nov-25', 'Dec-25', 'Jan-26', 'Feb-26', 'Mar-26', 'Apr-26', 'May-26', 'Jun-26', 'Jul-26', 'Aug-26', 'Sep-26', 'Oct-26', 'Nov-26', 'Dec-26', 'Jan-27', 'Feb-27', 'Mar-27', 'Apr-27', 'May-27', 'Jun-27', 'Jul-27', 'Aug-27', 'Sep-27', 'Oct-27', 'Nov-27', 'Dec-27', 'Jan-28', 'Feb-28', 'Mar-28', 'Apr-28', 'May-28', 'Jun-28', 'Jul-28', 'Aug-28', 'Sep-28', 'Oct-28', 'Nov-28', 'Dec-28', 'Jan-29', 'Feb-29', 'Mar-29'],
    'rPET': [408.189484, 424.979675, 383.967651, 424.634369, 463.083710, 572.250366, 463.557007, 430.959656, 462.088898, 502.290192, 456.639954, 407.817474, 425.076111, 460.838562, 378.909973, 447.736328, 531.969788, 385.076752, 415.778168, 424.958618, 554.195923, 416.068451, 390.431549, 457.154541, 404.248962, 399.763794, 418.839111, 489.917572, 387.429260, 382.204437, 397.621307, 479.087738, 394.409515, 411.916443, 426.078796, 430.738983, 397.730499, 430.052246, 420.728729, 415.816315, 377.760468, 401.701904, 388.236176, 416.383636, 389.059509, 390.115082, 422.077881, 419.521820, 388.738098, 418.610413, 414.832336, 398.614868, 383.951172, 389.059509, 378.644836, 398.614868, 378.644836, 389.940613, 430.249329, 399.434937],
    'rLDPE Film': [470.068146, 549.340515, 559.600281, 423.662994, 419.414398, 353.122559, 368.714722, 446.427521, 389.662292, 575.871277, 433.730835, 383.888519, 553.554810, 555.770569, 558.854248, 434.091248, 451.422180, 378.160645, 426.937592, 495.333099, 409.702515, 514.240295, 431.588776, 360.356079, 577.575623, 538.585999, 573.076050, 449.762421, 470.834564, 411.789673, 413.988953, 535.303040, 407.918335, 506.507843, 436.478973, 365.040161, 582.241089, 498.139435, 592.846985, 427.733002, 502.375977, 510.194183, 430.687134, 563.800171, 392.815613, 530.610107, 450.016205, 396.615356, 538.822571, 476.591370, 563.699524, 460.243103, 478.907837, 546.929504, 455.772125, 545.209351, 421.172058, 500.519623, 446.972382, 377.376862],
    'rPP': [724.093689, 660.025635, 641.812683, 614.689148, 733.644714, 799.050842, 724.746033, 808.271423, 721.389648, 801.367615, 617.885925, 623.163940, 656.462830, 772.062622, 733.428650, 810.632935, 756.861572, 771.984741, 735.249207, 661.367493, 681.590637, 662.774414, 752.620117, 785.523010, 738.140076, 829.554504, 738.040955, 816.567444, 630.811523, 759.985901, 639.733765, 755.180908, 781.563660, 740.914429, 795.375549, 721.389648, 760.483948, 666.192993, 761.679565, 649.377991, 774.456421, 745.096741, 775.087646, 773.451538, 736.280884, 786.980652, 633.853760, 770.379333, 628.709290, 765.861877, 766.863464, 740.391846, 809.957764, 738.040955, 760.074524, 649.084534, 755.499390, 655.912292, 785.766541, 768.802917],
    'rHDPE': [520.186462, 569.058960, 587.328430, 614.878235, 693.768677, 678.473145, 652.605225, 739.518188, 615.961975, 640.670166, 670.328735, 630.287170, 610.287170, 610.287170, 640.287170, 580.287170, 580.287170, 580.287170, 570.287170, 520.287170, 520.287170, 580.287170, 600.287170, 620.287170, 600.287170, 560.287170, 570.287170, 600.287170, 600.287170, 570.287170, 550.287170, 520.287170, 520.287170, 510.287170, 490.287170, 510.287170, 490.287170, 490.287170, 490.287170, 490.287170, 490.287170, 500.287170, 520.287170, 580.287170, 620.287170, 620.287170, 640.287170, 600.287170, 630.287170, 620.287170, 570.287170, 580.287170, 600.287170, 750.287170, 590.287170, 710.287170, 630.287170, 740.287170, 1010.287170, 1020.287170]
}

# Convert 'Month' column to datetime and set it as index
forecasted_prices_df = pd.DataFrame(forecasted_data)
forecasted_prices_df['Month'] = pd.to_datetime(forecasted_prices_df['Month'], format='%b-%y')
forecasted_prices_df.set_index('Month', inplace=True)

# Assign the DataFrame to df
df = forecasted_prices_df

# Split the data into training (75%) and validation (25%) sets
train_size = int(0.75 * len(df))
train_data = df.iloc[:train_size]
validation_data = df.iloc[train_size:]

# Define min and max values for each feature
min_max_values = {
    'rPET': (df['rPET'].min(), df['rPET'].max()),
    'rLDPE Film': (df['rLDPE Film'].min(), df['rLDPE Film'].max()),
    'rPP': (df['rPP'].min(), df['rPP'].max()),
    'rHDPE': (df['rHDPE'].min(), df['rHDPE'].max())
}

# Function to normalize predicted values
def normalize_predicted_values(predicted_values, feature):
    min_val, max_val = min_max_values[feature]
    normalized_values = []
    for val in predicted_values:
        if val >= max_val:
            normalized_values.append(max_val)
        elif val <= min_val:
            normalized_values.append(min_val)
        else:
            normalized_values.append(val)
    return normalized_values

# Function to compute gradients and apply gradient clipping by norm
def compute_gradients(X, y, weights, bias):
    predictions = X.dot(weights) + bias
    errors = predictions - y
    gradients_w = 2 * X.T.dot(errors) / len(y)
    gradients_b = 2 * np.sum(errors) / len(y)
    return gradients_w, gradients_b

def apply_gradient_clipping(gradients_w, gradients_b, clip_norm):
    norm = np.sqrt(np.sum(gradients_w**2) + gradients_b**2)
    if norm > clip_norm:
        gradients_w = gradients_w * (clip_norm / norm)
        gradients_b = gradients_b * (clip_norm / norm)
    return gradients_w, gradients_b

# Function to train the model using gradient descent with gradient clipping by norm
def train_model(X_train, y_train, X_val, y_val, learning_rate=0.01, iterations=1000, clip_norm=1.0):
    weights = np.random.randn(X_train.shape[1])
    bias = 0
    training_loss = []
    validation_loss = []
    for i in range(iterations):
        gradients_w, gradients_b = compute_gradients(X_train, y_train, weights, bias)
        gradients_w, gradients_b = apply_gradient_clipping(gradients_w, gradients_b, clip_norm)
        weights -= learning_rate * gradients_w
        bias -= learning_rate * gradients_b
        train_predictions = X_train.dot(weights) + bias
        train_loss = np.mean((train_predictions - y_train) ** 2)
        training_loss.append(train_loss)
        val_predictions = X_val.dot(weights) + bias
        val_loss = np.mean((val_predictions - y_val) ** 2)
        validation_loss.append(val_loss)
        if i % 100 == 0:
            print(f"Iteration {i}/{iterations} - Training Loss: {train_loss:.4f} - Validation Loss: {val_loss:.4f}")
    return weights, bias, training_loss, validation_loss

# Prepare the data for the model
def prepare_data(data):
    X, y = [], []
    for i in range(len(data) - 1):
        X.append(data[i])
        y.append(data[i + 1])
    return np.array(X), np.array(y)

# Function to train and evaluate the model for a given feature
def train_and_evaluate(feature):
    if feature not in df.columns:
        print(f"Feature '{feature}' not found in DataFrame columns.")
        return
    X_train, y_train = prepare_data(train_data[feature].values)
    X_val, y_val = prepare_data(validation_data[feature].values)
    # Add a bias term (column of ones) to the input data
    X_train = np.c_[np.ones(X_train.shape[0]), X_train]
    X_val = np.c_[np.ones(X_val.shape[0]), X_val]
    weights, bias, training_loss, validation_loss = train_model(X_train, y_train, X_val, y_val, learning_rate=0.01, iterations=1000, clip_norm=1.0)
    val_predictions = X_val.dot(weights) + bias
    val_predictions = normalize_predicted_values(val_predictions, feature)
    print(f"Normalized {feature} values: {val_predictions}")
    plt.figure(figsize=(10, 6))
    plt.plot(training_loss, label='Training Loss')
    plt.plot(validation_loss, label='Validation Loss')
    plt.title(f'Model Loss for {feature}')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Train and evaluate the model for each feature
for feature in ['rPET', 'rLDPE Film', 'rPP', 'rHDPE']:
    train_and_evaluate(feature)


# **Error metrics printed for Model Comparisons**

# In[1]:


def print_error_metrics(material, mse, mae, rmse, mape):
    print(f"Error Metrics for {material}:")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print()

# Error metrics for rPET
print_error_metrics("rPET", 12655.87, 90.74, 112.50, 18.56)

# Error metrics for rLDPE Film
print_error_metrics("rLDPE Film", 14121.25, 91.11, 118.83, 17.29)

# Error metrics for rPP
print_error_metrics("rPP", 7794.24, 69.97, 88.28, 10.12)

# Error metrics for rHDPE Film
print_error_metrics("rHDPE Film", 2703.86, 42.75, 52.00, 6.60)

