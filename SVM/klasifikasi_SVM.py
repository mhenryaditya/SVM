import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.combine import SMOTETomek
import joblib

def SVM(data):
    data = pd.read_csv(data)

    # Memisahkan fitur dan label
    X = data.drop(columns=["class_label"])  # Semua kolom kecuali 'class_label' adalah fitur
    y = data["class_label"]                 # Kolom 'class_label' adalah target

    # Pembagian dataset (80% latih, 20% uji)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Balancing
    sampler = SMOTETomek(random_state=42)
    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)

    # Melatih model SVM
    model = LinearSVC(C=0.5, random_state=42)
    model.fit(X_resampled, y_resampled)

    # Prediksi pada data uji
    y_pred = model.predict(X_test)

    # Simpan model
    joblib.dump(model, './data/dump/svm_model.pkl')

    # Evaluasi
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Akurasi: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Simpan hasil prediksi (opsional)
    results = pd.DataFrame({
        "Actual": y_test.reset_index(drop=True),
        "Predicted": y_pred
    })
    results.to_csv("./data/hasil_klasifikasi/svm_tfidf_results.csv", index=False)
