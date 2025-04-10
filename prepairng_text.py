import useful as useful


def preparing_text(texts=[]):
    prepared_text = []
    for text in texts:
        text = text.lower()
        text = useful.remove_digits_from_text(text)
        text = useful.remove_useless_chars_from_text(text)
        prepared_text.append(text)
    return prepared_text


