from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from Load_Processing import load_and_preprocess_data

def logistic_regression_classifier(X_train, y_train, X_test, y_test):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report

def main():
    filepath = 'email.csv'
    X_train, X_test, y_train, y_test = load_and_preprocess_data(filepath)
    accuracy, report = logistic_regression_classifier(X_train, y_train, X_test, y_test)
    print(f'Logistic Regression Accuracy: {accuracy}')
    print('Logistic Regression Classification Report:')
    print(report)

if __name__ == '__main__':
    main()