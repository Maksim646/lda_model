import os


import prepairng_text
import useful as useful
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import chardet

directory_path = '/mnt/c/Users/Максим/Downloads/txt_files_main_words'

texts = []

for filename in os.listdir(directory_path):
    if filename.endswith('.txt'):
        filepath = os.path.join(directory_path, filename)
        with open(filepath, 'rb') as file:
            raw_data = file.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']
            print(f"Определенная кодировка для {filename}: {encoding}")

        try:
            with open(filepath, 'r', encoding=encoding, errors='replace') as file:
                content = file.read()
                texts.append(content)
        except UnicodeDecodeError as e:
            print(f"Ошибка декодирования файла {filename} с кодировкой {encoding}: {e}")
            texts.append("")


texts = prepairng_text.preparing_text(texts)
tokens = useful.tokenize(texts)
tokens = useful.lemmatize(tokens)

dictionary= useful.create_dictionary(tokens)
corpus = useful.create_corpus(tokens, dictionary)


lda = useful.lda_model(corpus, dictionary)

vis = gensimvis.prepare(lda, corpus, dictionary)
pyLDAvis.save_html(vis, 'lda_visualization.html')



