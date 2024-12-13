import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

def ektraksi_fitur(input_csv):
    # Baca dataset hasil preprocessing
    df = pd.read_csv(input_csv)

    # Pastikan kolom 'text' ada dalam dataset
    if 'text' not in df.columns:
        raise ValueError("Kolom 'text' tidak ditemukan dalam dataset. Pastikan file CSV memiliki kolom 'text'.")

    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()

    # Transformasi teks menjadi matriks TF-IDF
    tfidf_matrix = vectorizer.fit_transform(df['text'])

    # Mendapatkan fitur (kata unik)
    feature_names = vectorizer.get_feature_names_out()

    # Konversi matriks TF-IDF ke DataFrame untuk visualisasi
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

    # Gabungkan dengan label untuk referensi
    tfidf_df['class_label'] = df['class_label']

    # Simpan vectorizer
    joblib.dump(vectorizer, './data/dump/tfidf_vectorizer.pkl')

    # Simpan hasil ke file baru
    output_csv = "./data/dataekstraksi/tfidf_features.csv"
    tfidf_df.to_csv(output_csv, index=False)

    print(f"Ekstraksi Fitur selesai, Hasil TF-IDF disimpan dalam file: {output_csv}")