# Hala Almutairi        CpE 420: Data Mining    Project 1

# Import
import sys
import pandas as pd
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score


# Load And Prepro Data

def load_data(file_path, encoders=None):
    df = pd.read_csv(file_path, header=None)

    if encoders is None:
        encoders = {}
        for col in df.columns:
            # Only Encode Object Columns (categorical)
            if df[col].dtype == 'object':
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                encoders[col] = le
        return df, encoders
    else:
        for col in df.columns:
            if col in encoders:
                # Unseen Labels 
                df[col] = df[col].apply(
                    lambda x: encoders[col].transform([x])[0]
                    if x in encoders[col].classes_
                    else -1
                )
        return df


def split_xy(df):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y



# Task 4
def print_results(y_test, y_pred):
    for i in range(len(y_test)):
        object_id = i + 1
        predicted = y_pred[i]
        true = y_test.iloc[i]
        accuracy = 1 if predicted == true else 0
        print(f"ID={object_id:5d}, predicted={predicted:3d}, true={true:3d}, accuracy={accuracy:4.2f}")

def evaluate(y_test, y_pred, start_time):
    end_time = time.time()

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    runtime = end_time - start_time

    print("\nSummary:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"Runtime  : {runtime:.4f} seconds\n")

# Decision Tree
def decision_tree(train_file, test_file):
    train_df, encoders = load_data(train_file)
    test_df = load_data(test_file, encoders)
    X_train, y_train = split_xy(train_df)
    X_test, y_test = split_xy(test_df)
    model = DecisionTreeClassifier()

    print("\n=====Decision Tree=====")
    start_time = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print_results(y_test, y_pred)
    evaluate(y_test, y_pred, start_time)

# Random Forest
def random_forest(train_file, test_file):
    train_df, encoders = load_data(train_file)
    test_df = load_data(test_file, encoders)

    X_train, y_train = split_xy(train_df)
    X_test, y_test = split_xy(test_df)
    model = RandomForestClassifier(n_estimators=100)

    print("\n=====Random Forest=====")
    start_time = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print_results(y_test, y_pred)
    evaluate(y_test, y_pred, start_time)

# Naive Bayes
def naive_bayes(train_file, test_file):
    train_df, encoders = load_data(train_file)
    test_df = load_data(test_file, encoders)

    X_train, y_train = split_xy(train_df)
    X_test, y_test = split_xy(test_df)
    model = GaussianNB()

    print("\n=====Naive Bayes=====")
    start_time = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print_results(y_test, y_pred)
    evaluate(y_test, y_pred, start_time)


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: python3 DTvsRFvsNB.py <training_file> <test_file>")
        sys.exit(1)

    train_file = sys.argv[1]
    test_file = sys.argv[2]

    print("Start\n")
    decision_tree(train_file, test_file)
    random_forest(train_file, test_file)
    naive_bayes(train_file, test_file)