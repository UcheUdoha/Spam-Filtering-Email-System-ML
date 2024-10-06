import unittest
from Load_Processing import load_and_preprocess_data
from Logistic_R import logistic_regression_classifier
from Linear_R import linear_regression_classifier

class TestClassifiers(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        filepath = 'email.csv'
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = load_and_preprocess_data(filepath)

    def test_logistic_regression(self):
        accuracy, report = logistic_regression_classifier(self.X_train, self.y_train, self.X_test, self.y_test)
        print(f'Logistic Regression Accuracy: {accuracy}')
        print('Logistic Regression Classification Report:')
        print(report)
        self.assertGreaterEqual(accuracy, 0.9)  # Example assertion

    def test_linear_regression(self):
        accuracy, report = linear_regression_classifier(self.X_train, self.y_train, self.X_test, self.y_test)
        print(f'Linear Regression (Classification) Accuracy: {accuracy}')
        print('Linear Regression (Classification) Report:')
        print(report)
        self.assertGreaterEqual(accuracy, 0.9)  # Example assertion

if __name__ == '__main__':
    unittest.main()