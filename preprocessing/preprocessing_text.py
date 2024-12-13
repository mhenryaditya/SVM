import pandas as pd
import re
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from spellchecker import SpellChecker

import nltk

irregular_words = {
    # Bentuk jamak ke tunggal
    "men": "man",
    "women": "woman",
    "children": "child",
    "feet": "foot",
    "teeth": "tooth",
    "geese": "goose",
    "mice": "mouse",
    "oxen": "ox",
    "indices": "index",
    "criteria": "criterion",
    "data": "datum",

    # Bentuk kata kerja tidak beraturan
    "was": "be", "were": "be",
    "been": "be", "being": "be",
    "ran": "run", "running": "run",
    "ate": "eat", "eaten": "eat",
    "drank": "drink", "drunk": "drink",
    "saw": "see", "seen": "see",
    "gave": "give", "given": "give",
    "went": "go", "gone": "go",
    "got": "get", "gotten": "get",
    "did": "do", "done": "do",
    "made": "make",
    "spoke": "speak", "spoken": "speak",
    "took": "take", "taken": "take",
    "brought": "bring",
    "thought": "think",
    "caught": "catch",
    "taught": "teach",
    "bought": "buy",
    "fought": "fight",
    "sought": "seek",
    "built": "build",
    "spent": "spend",
    "left": "leave",
    "kept": "keep",
    "slept": "sleep",
    "wept": "weep",
    "lit": "light",
    "lost": "lose",
    "won": "win",
    "began": "begin", "begun": "begin",
    "chose": "choose", "chosen": "choose",
    "rode": "ride", "ridden": "ride",
    "wrote": "write", "written": "write",
    "broke": "break", "broken": "break",
    "froze": "freeze", "frozen": "freeze",
    "fell": "fall", "fallen": "fall",
    "rose": "rise", "risen": "rise",

    # Superlative dan Comparative
    "better": "good",
    "best": "good",
    "worse": "bad",
    "worst": "bad",
    "more": "much",
    "most": "much",
    "less": "little",
    "least": "little",
    "farther": "far",
    "farthest": "far",
    "further": "far",
    "furthest": "far",
}

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default ke noun


def preprocessing_text(text):
    if pd.isnull(text):  # Jika nilai kosong, return kosong
        return ""
    
    # Membersihkan teks
    text = re.sub(r'\d+', '', text)  # Menghapus angka
    text = re.sub(r'\s+', ' ', text).strip()  # Menghapus spasi ekstra
    
    # Tokenisasi
    tokens = word_tokenize(text)
    
    # Lowercase semua teks
    tokens = [token.lower() for token in tokens]
    
    # Menghapus karakter spesial
    tokens = [re.sub(r'[^\w\s]', '', token) for token in tokens if token]
    
    # Menghapus stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # POS Tagging
    pos_tags = pos_tag(tokens)  # Dapatkan POS tag untuk setiap token
    
    # Lemmatization dengan POS
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token, get_wordnet_pos(tag)) for token, tag in pos_tags]
    
    # Normalisasi menggunakan kamus kata tidak standar
    tokens = [irregular_words[token] if token in irregular_words else token for token in tokens]
    
    # Memperbaiki kesalahan kata
    spell = SpellChecker()
    tokens = [spell.correction(token) if token not in spell else token for token in tokens]
    tokens = [token for token in tokens if token]  # Hapus nilai None
    
    # Gabungkan kembali menjadi teks
    return ' '.join(tokens)