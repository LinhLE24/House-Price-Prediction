from house_prices.preprocess import (replace_missing_value, classify_category, encoding_category_features, scaling_numeric_features, prediction)
import sys
sys.path.append('..')


def make_predictions(data):
    data = replace_missing_value(data)
    print("Missing data ? :", data.isnull().values.any())
    data = data[['GrLivArea', '1stFlrSF', 'MasVnrArea', 'LotFrontage', 'GarageArea', 'TotalBsmtSF', 'OverallQual', 'Street', 'MSZoning', 'HouseStyle']]
    cat_col_inference, num_col_inference = classify_category(data)
    data = encoding_category_features(data, cat_col_inference)
    data = scaling_numeric_features(data, num_col_inference)
    prediction(data)
    return prediction(data)
