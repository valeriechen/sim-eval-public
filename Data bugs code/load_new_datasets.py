import sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

def load_occupancy_dataset(has_bug, bug_type):

    dtypes = [
            ("date", "category"),
            ("Temperature", "float32"),
            ("Humidity", "float32"),
            ("Light", "float32"),
            ("CO2", "float32"),
            ("HumidityRatio", "float32"),
            ("Occupancy", "category")
    ]

    data = pd.read_csv(
        "occupancy.txt", delimiter=',', skiprows=[0],
        names=[d[0] for d in dtypes],
            dtype=dict(dtypes))

    #data.fillna(data.mean(), inplace=True)

    filt_dtypes = list(filter(lambda x: not (x[0] in ["Occupancy", "date"]), dtypes))
    y = data["Occupancy"] == "1"
    y = LabelEncoder().fit_transform(y)

    data= data.drop(["Occupancy", "date"], axis=1)

    for k, dtype in filt_dtypes:
        if dtype == "category":
            data[k] = data[k].cat.codes

    X = data

    if has_bug:
        perc = float(bug_type[-4:])
        print(perc)

        #ADDS BUGS TO EVERYTHING
        inds = np.random.choice(X.shape[0], int(perc*X.shape[0])).tolist()
        for i in range(len(inds)):
          y[inds[i]] = random.choice([0, 1])

    scaler = MinMaxScaler()
    train_X_scaled = scaler.fit_transform(X)

    inds = np.random.choice(X.shape[0], 10000)  
    X_subset = X[inds,:]
    y_subset = y[inds]
    X_t_subset = train_X_scaled[inds,:]

    inds = np.random.choice(X.shape[0], 1500)  
    X_subset1 = X[inds,:]
    y_subset1 = y[inds]

    return X_subset, y_subset, X_subset1, y_subset1, X_t_subset

def load_shopper_dataset(has_bug, bug_type):

    dtypes = [
        ("Administrative", "float32"),
        ("Administrative Duration", "float32"),
        ("Informational", "float32"),
        ("Informational Duration", "float32"),
        ("Product Related", "float32"),
        ("Product Related Duration", "float32"),
        ("Bounce Rate", "float32"),
        ("Exit Rate", "float32"),
        ("Page Value", "float32"),
        ("Special Day", "float32"),
        ("Month", "category"),
        ("OperatingSystems", "category"),
        ("Browser", "category"),
        ("Region", "category"),
        ("TrafficType", "category"),
        ("VisitorType", "category"),
        ("Weekend", "category"),
        ("Revenue", "category")

    ]

    data = pd.read_csv(
        "online_shoppers_intention.csv", delimiter=',', skiprows=[0],
        names=[d[0] for d in dtypes],
            dtype=dict(dtypes))

    filt_dtypes = list(filter(lambda x: not (x[0] in ["Revenue"]), dtypes))
    y = data["Revenue"] == "TRUE"
    y = LabelEncoder().fit_transform(y)

    data= data.drop(["Revenue"], axis=1)

    for k, dtype in filt_dtypes:
        if dtype == "category":
            data[k] = data[k].cat.codes

    X = data

    if has_bug:
        perc = float(bug_type[-4:])
        print(perc)

        #ADDS BUGS TO EVERYTHING
        inds = np.random.choice(X.shape[0], int(perc*X.shape[0])).tolist()
        for i in range(len(inds)):
          y[inds[i]] = random.choice([0, 1])

    scaler = MinMaxScaler()
    train_X_scaled = scaler.fit_transform(X)

    inds = np.random.choice(X.shape[0], 10000)  
    X_subset = X[inds,:]
    y_subset = y[inds]
    X_t_subset = train_X_scaled[inds,:]

    inds = np.random.choice(X.shape[0], 1500)  
    X_subset1 = X[inds,:]
    y_subset1 = y[inds]

    return X_subset, y_subset, X_subset1, y_subset1, X_t_subset

def load_bank_dataset(has_bug, bug_type):

    dtypes = [
        ("Age", "float32"),
        ("Job", "category"),
        ("Marital", "category"),
        ("Education", "category"),
        ("Default", "category"),
        ("Balance", "float32"),
        ("Housing", "category"),
        ("Loan", "category"),
        ("contact", "category"),
        ("Month", "category"),
        ("DayOfWeek", "category"),
        ("Duration", "float32"),
        ("Campaign", "float32"),
        ("PDays", "float32"),
        ("Previous", "float32"),
        ("poutcome", "category"),
        ("y", "category")
    ]

    data = pd.read_csv(
        "bank-full.csv",skiprows=[0], delimiter=';',
        names=[d[0] for d in dtypes],
            dtype=dict(dtypes))

    filt_dtypes = list(filter(lambda x: not (x[0] in ["y", "poutcome", "contact"]), dtypes))
    y = data["y"] == "yes"
    y = LabelEncoder().fit_transform(y)

    data= data.drop(["y", "poutcome", "contact"], axis=1)

    for k, dtype in filt_dtypes:
        if dtype == "category":
            data[k] = data[k].cat.codes

    X = data

    if has_bug:
        perc = float(bug_type[-4:])
        print(perc)

        #ADDS BUGS TO EVERYTHING
        inds = np.random.choice(X.shape[0], int(perc*X.shape[0])).tolist()
        for i in range(len(inds)):
          y[inds[i]] = random.choice([0, 1])

    scaler = MinMaxScaler()
    train_X_scaled = scaler.fit_transform(X)

    inds = np.random.choice(X.shape[0], 10000)  
    X_subset = X[inds,:]
    y_subset = y[inds]
    X_t_subset = train_X_scaled[inds,:]

    inds = np.random.choice(X.shape[0], 1500)  
    X_subset1 = X[inds,:]
    y_subset1 = y[inds]

    return X_subset, y_subset, X_subset1, y_subset1, X_t_subset





