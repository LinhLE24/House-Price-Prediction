from house_prices.preprocess import (replace_missing_value, get_train_test_sets, classify_category, encoding_category_features_with_fitting, scaling_numeric_features_with_fitting, train_model, encoding_category_features, scaling_numeric_features, evaluate_model, prediction)
import sys
sys.path.append('..')


def build_model(data):
    data = replace_missing_value(data)
    print("Missing data ? :", data.isnull().values.any())
    X_train, y_train, X_test, y_test = get_train_test_sets(data)
    cat_col_train, num_col_train = classify_category(X_train)
    X_train = encoding_category_features_with_fitting(X_train, cat_col_train)
    X_train = scaling_numeric_features_with_fitting(X_train, num_col_train)
    train_model(X_train, y_train)
    cat_col_test, num_col_test = classify_category(X_test)
    X_test = encoding_category_features(X_test, cat_col_test)
    X_test = scaling_numeric_features(X_test, num_col_test)
    y_pred = prediction(X_test)
    metrics = evaluate_model(y_pred, y_test)
    return train_model(X_train, y_train), metrics
