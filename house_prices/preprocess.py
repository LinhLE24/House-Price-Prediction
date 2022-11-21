import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib
import sys
sys.path.append('..')


def read_dataset(path):
    df_master = pd.read_csv(path)
    df = df_master.copy()
    return df


def replace_missing_value(df):
    for col in df.columns:
        if df[col].dtypes == 'object':
            df[col].fillna(df[col].mode()[0], inplace=True)
        elif df[col].dtypes == 'int64' or df[col].dtypes == 'float64':
            df[col].fillna(df[col].mean(), inplace=True)
    return df


def get_train_test_sets(df):
    df = df[['SalePrice', 'GrLivArea', '1stFlrSF', 'MasVnrArea', 'LotFrontage', 'GarageArea', 'TotalBsmtSF', 'OverallQual', 'Street', 'MSZoning', 'HouseStyle']]
    X = df.drop('SalePrice', axis=1)
    y = df['SalePrice'].values.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    return X_train, y_train, X_test, y_test


def classify_category(df):
    cat_features = df.select_dtypes(include=['object']).columns.tolist()
    num_features = df.select_dtypes(exclude=['object']).columns.tolist()
    return cat_features, num_features


def encoding_category_features_with_fitting(df, categorical_col):
    fitting_encoder = pd.DataFrame(ohe.fit_transform(df[categorical_col]))
    fitting_encoder.columns = ohe.get_feature_names_out(categorical_col)
    fitting_encoder.index = np.arange(1, len(df)+1)
    df.drop(categorical_col, axis=1, inplace=True)
    df = pd.concat([df.reset_index(drop=True), fitting_encoder.reset_index(drop=True)], axis=1)
    return df


def encoding_category_features(df, categorical_col):
    encoder = pd.DataFrame(ohe.transform(df[categorical_col]))
    encoder.columns = ohe.get_feature_names_out(categorical_col)
    encoder.index = np.arange(1, len(df)+1)
    df.drop(categorical_col, axis=1, inplace=True)
    df = pd.concat([df.reset_index(drop=True), encoder.reset_index(drop=True)], axis=1)
    return df


def scaling_numeric_features_with_fitting(df, numerical_col):
    df.loc[:, numerical_col] = sc.fit_transform(df[numerical_col])
    return df


def scaling_numeric_features(df, numerical_col):
    df.loc[:, numerical_col] = sc.transform(df[numerical_col])
    return df


def train_model(X, y):
    global model
    model = multiple_regression.fit(X, y)
    return model


# Regression
regressor = LinearRegression()
joblib.dump(regressor, '../models/model.joblib', compress=0, protocol=None, cache_size=None)
multiple_regression = joblib.load('../models/model.joblib', mmap_mode=None)

# OneHotEncoder
one_hot_encoder = OneHotEncoder(sparse=False)
joblib.dump(one_hot_encoder, '../models/one_hot_encoder.joblib', compress=0, protocol=None, cache_size=None)
ohe = joblib.load('../models/one_hot_encoder.joblib', mmap_mode=None)

# Scaling
scaler = StandardScaler()
joblib.dump(scaler, '../models/scaler.joblib', compress=0, protocol=None, cache_size=None)
sc = joblib.load('../models/scaler.joblib', mmap_mode=None)


def prediction(X_val):
    y_pred = model.predict(X_val)
    return y_pred


def evaluate_model(y_pred, y_val):
    y_pred = y_pred.squeeze()
    y_val = y_val.squeeze()
    print("Mean square error (MSE): %.2f" % np.mean((y_pred - y_val) ** 2))
    print("Root mean square error (RMSE): %.2f" % np.sqrt(np.mean((y_pred - y_val) ** 2)))
    print("Mean absolute error (MAE): %.2f" % np.mean(abs(y_pred - y_val)))
    print("Coefficient of determination (R^2): %.2f" % r2_score(y_val, y_pred))
    return evaluate_model
