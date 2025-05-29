import os
import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def load_data(path):
    df = pd.read_csv(path)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y

def run_svm(train_path, test_path):
    X_train, y_train = load_data(train_path)
    X_test, y_test = load_data(test_path)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = SVC(kernel='rbf', C=1.0, gamma='scale')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("Classification Report:\n", classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix Heatmap')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    print("Accuracy:", accuracy_score(y_test, y_pred))

def evaluate_svm_model(file_pairs, base_path):
    for train_file, test_file in file_pairs:
        print(f"\n----- ĐÁNH GIÁ MÔ HÌNH SVM VỚI BỘ {train_file[:-4]} - {test_file[:-4]} -----")
        train_path = f"{base_path}\\{train_file}"
        test_path = f"{base_path}\\{test_file}"


        start_time = time.time()
        run_svm(train_path, test_path)
        end_time = time.time()

        print(f"Thời gian chạy bộ {train_file[:-4]}: {end_time - start_time:.2f} giây")
