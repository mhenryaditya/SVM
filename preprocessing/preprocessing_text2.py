import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import pandas as pd
import re

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

irregular_verbs = {
    "was": "be", "were": "be", "been": "be",
    "ate": "eat", "eaten": "eat",
    "ran": "run", "running": "run",
    "saw": "see", "seen": "see",
    "bought": "buy", "buying": "buy",
    "thought": "think", "thinking": "think",
    "gave": "give", "given": "give",
    "took": "take", "taken": "take",
    "made": "make", "making": "make",
    "went": "go", "gone": "go",
    "came": "come", "coming": "come",
    "wrote": "write", "written": "write",
    "spoke": "speak", "spoken": "speak",
    "chose": "choose", "chosen": "choose",
    "flew": "fly", "flown": "fly",
    "broke": "break", "broken": "break",
}

def get_wordnet_pos(tag):
    if tag.startswith('V'):  # Hanya memproses kata kerja
        return wordnet.VERB
    return None  # Abaikan tag lain

# Fungsi untuk mengubah kata kerja menjadi bentuk dasar (V1)
def to_base_form(text):
    if not text:  # Jika teks kosong
        return ""
    
    # Tokenisasi teks
    tokens = word_tokenize(text)
    
    # POS Tagging
    pos_tags = pos_tag(tokens)
    
    # Lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Proses untuk setiap token
    base_tokens = []
    for token, tag in pos_tags:
        wordnet_pos = get_wordnet_pos(tag)
        if wordnet_pos:  # Jika token adalah kata kerja
            # Gunakan kamus irregular verbs jika tersedia
            base_form = irregular_verbs.get(token.lower(), token.lower())
            # Lemmatize jika tidak ditemukan di kamus irregular
            base_form = lemmatizer.lemmatize(base_form, wordnet_pos)
            base_tokens.append(base_form)
        else:
            # Simpan kata lain tanpa perubahan
            base_tokens.append(token.lower())
    
    # Gabungkan kembali menjadi teks
    text = ' '.join(base_tokens)

    # Hilangkan "i" yang tidak diinginkan
    text = re.sub(r'\bi\b', '', text)  # Hilangkan "i" yang berdiri sendiri
    text = re.sub(r'\bi ', '', text)  # Hilangkan "i " yang berada di awal kata
    text = re.sub(r' i\b', '', text)  # Hilangkan " i" yang berada di akhir kata
    text = re.sub(r' i ', ' ', text)  # Hilangkan " i " yang berada di tengah kata
    
    return text.strip()  # Hilangkan spasi tambahan di awal dan akhir teks