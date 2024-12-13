import pandas as pd
from preprocessing.preprocessing_text import preprocessing_text
from preprocessing.preprocessing_text2 import to_base_form

def preprocessing(input_file1, input_file2, output_file1, output_file2, text_column):
    print("Memulai preprocessing1...")

    # read data
    df = pd.read_csv(input_file1)

    # process each row on text column
    df['text'] = df[text_column].apply(preprocessing_text)

    # save file
    df[['text', 'class_label']].to_csv(output_file1, index=False)

    print(f"Preprocessing1 selesai, Data berhasil diproses dan disimpan ke {output_file1}")


    print("Memulai preprocessing2...")

    # read data
    df = pd.read_csv(input_file2)

    # process each row on text column
    df['text'] = df[text_column].apply(to_base_form)

    # save file
    df[['text', 'class_label']].to_csv(output_file2, index=False)

    print(f"Preprocessing2 selesai, Data berhasil diproses dan disimpan ke {output_file2}")