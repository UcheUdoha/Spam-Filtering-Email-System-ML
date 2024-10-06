from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report
from Load_Processing import load_and_preprocess_data

def linear_regression_classifier(X_train, y_train, X_test, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_class = (y_pred >= 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred_class)
    report = classification_report(y_test, y_pred_class)
    return accuracy, report

def main():
    filepath = 'email.csv'
    X_train, X_test, y_train, y_test = load_and_preprocess_data(filepath)
    accuracy, report = linear_regression_classifier(X_train, y_train, X_test, y_test)
    print(f'Linear Regression (Classification) Accuracy: {accuracy}')
    print('Linear Regression (Classification) Report:')
    print(report)

if __name__ == '__main__':
    main()

