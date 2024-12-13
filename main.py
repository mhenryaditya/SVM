from preprocessing.main_prepocessing import preprocessing
from preprocessing.preprocessing_text import preprocessing_text
from preprocessing.preprocessing_text2 import to_base_form
from ekstraksi_fitur.ekstraksi_fitur import ektraksi_fitur
from SVM.klasifikasi_SVM import SVM
import joblib

# Data
# read file csv
input_file1 = './data/imdb.csv'
output_file1 = './data/prepdata/processed_text.csv'
input_file2 = './data/prepdata/processed_text.csv'
output_file2 = './data/prepdata/processed_text2.csv'
text_column = 'text'
input_csv = "./data/prepdata/processed_text2.csv"
data = "./data/dataekstraksi/tfidf_features.csv"

# 1. Preprocessing
print("Memulai Preprocessing...")
preprocessing(input_file1, input_file2, output_file1, output_file2, text_column)

# 2. Ekstraksi Fitur
print("Memulai Ekstraksi Fitur...")
ektraksi_fitur(input_csv)

# 3. Train and save model
print("Training data...")
SVM(data)

# 4. Predict new data
print("Predicting new data...")

# contoh kalimat pengetesan
new_texts = [
    "The cinematography was breathtaking, and every scene felt like a masterpiece of art.",
    "The dialogue was poorly written and sounded unnatural, making it hard to connect with the characters.",
    "I thoroughly enjoyed the storyline; it was both compelling and emotionally resonant.",
    "The plot was overly convoluted, leaving me confused and uninterested in the outcome.",
    "The characters were well-developed, and their arcs were satisfying from start to finish.",
    "I found the acting to be lackluster, with the cast delivering uninspired performances.",
    "This movie exceeded my expectations with its smart dialogue and engaging plot twists.",
    "The film dragged on for far too long, and the ending felt rushed and unsatisfying.",
    "A beautifully crafted film that combines stunning visuals with a heartwarming message.",
    "The soundtrack was jarring and didn’t match the tone of the scenes, which was distracting."
]


vectorizer = joblib.load('./data/dump/tfidf_vectorizer.pkl')
model = joblib.load('./data/dump/svm_model.pkl')

processed_texts1 = [preprocessing_text(text) for text in new_texts]
processed_texts2 = [to_base_form(text) for text in new_texts]
new_features = vectorizer.transform(processed_texts2)
prediction = model.predict(new_features)
for text, label in zip(new_texts, prediction):
    print(f"Text: \"{text}\" -> Prediction: {label}")