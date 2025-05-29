
from svm_utils import evaluate_svm_model

# Original data files
file_pairs = [
    ("train41.csv", "test41.csv"),
    ("train64.csv", "test64.csv"),
    ("train73.csv", "test73.csv")
]

base_path = ".\data\original_data"
print("----- ĐÁNH GIÁ MÔ HÌNH SVM VỚI DỮ LIỆU GỐC -----")
evaluate_svm_model(file_pairs, base_path)

# # PCA data files
# file_pairs = [
#     ("train_pca_41.csv", "test_pca_41.csv"),
#     ("train_pca_64.csv", "test_pca_64.csv"),
#     ("train_pca_73.csv", "test_pca_73.csv")
# ]
# base_path = ".\\data\\PCA_data"
# print("\n----- ĐÁNH GIÁ MÔ HÌNH SVM VỚI DỮ LIỆU PCA -----")
# evaluate_svm_model(file_pairs, base_path)

# # LDA data files
# file_pairs = [
#     ("train_lda_41.csv", "test_lda_41.csv"),
#     ("train_lda_64.csv", "test_lda_64.csv"),
#     ("train_lda_73.csv", "test_lda_73.csv")
# ]
# base_path = ".\\data\\LDA_data"
# print("\n----- ĐÁNH GIÁ MÔ HÌNH SVM VỚI DỮ LIỆU LDA -----")
# evaluate_svm_model(file_pairs, base_path)
