from pathlib import Path
from itertools import islice


def starts_with_cyrillic_lowercase(text):
    if not text:  # Check for an empty string
        return False
    first_char = text[0]
    # Check that the character is a lowercase letter and is within the Cyrillic range
    return first_char.islower() and 'а' <= first_char <= 'я' or first_char == 'ё'


def is_only_cyrillic(text):
    if not text:  # Check for an empty string
        return False
    return all(char.isalpha() and (('а' <= char.lower() <= 'я') or char.lower() == 'ё')
               for char in text)


def get_word(line):
    split = line.split(' ')
    try:
        word = split[0]
        if is_only_cyrillic(word):
            return word
        return None
    
    except IndexError:
        return None


def clean_dictionary(input_file_name, output_file_name, input_path, start=0, end=1_000_000):
    full_input_path = Path(input_path).joinpath(input_file_name)
    full_output_path = Path(input_path).joinpath(output_file_name)
    
    if not full_input_path.exists():
        return 'Input file did not found'
    
    with open(full_input_path, 'r', encoding='utf-8') as in_file, \
            open(full_output_path, 'w', encoding='utf-8') as outfile:
        for i, line in enumerate(islice(in_file, start, end + 1), start=start):
            word = get_word(line)
            if word is not None:
                outfile.write(f'{word}\n')


def clean_ru_dictionary(input_file_name, output_file_name, input_path, separator=' ', start=0, end=1_000_000):
    full_input_path = Path(input_path).joinpath(input_file_name)
    full_output_path = Path(input_path).joinpath(output_file_name)
    if not full_input_path.exists():
        return 'Input file did not found'
    
    with open(full_input_path, 'r', encoding='utf-8') as in_file, \
            open(full_output_path, 'w', encoding='utf-8') as outfile:
        
        for i, line in enumerate(islice(in_file, start, end + 1), start=start):
            word = line.split(' ')
            word = word[0][:-1]
            if all(char.isalpha() and ('а' <= char.lower() <= 'я') for char in word):
                outfile.write(f'{word}{separator}')


def clean_en_dictionary(input_file_name, output_file_name, input_path, start=0, end=1_000_000):
    full_input_path = Path(input_path).joinpath(input_file_name)
    full_output_path = Path(input_path).joinpath(output_file_name)
    if not full_input_path.exists():
        return 'Input file did not found'
    
    with open(full_input_path, 'r', encoding='utf-8') as in_file, \
            open(full_output_path, 'w', encoding='utf-8') as outfile:
        
        for i, line in enumerate(islice(in_file, start, end + 1), start=start):
            word = line.split(' ')
            word = word[0][:-1]
            if all(char.isalpha() and ('a' <= char.lower() <= 'z') for char in word):
                outfile.write(f'{word} ')


def read_dictionary(input_file_name, input_path):
    full_input_path = Path(input_path).joinpath(input_file_name)
    if not full_input_path.exists():
        return 'Input file did not found'
    
    bigrams = set()
    
    with open(full_input_path, 'r', encoding='utf-8') as file:
        for word in file:
            word = word[:-1]
            pair = tuple(sorted(word[:2]))
            word_lst = list(word.lower())
            bgs = list(zip(word_lst, word_lst[1:]))
            a, *rest = pair
            b = rest[0] if rest else None
            for bg in bgs:
                bigrams.add(bg)
            if b:
                # bigrams.add(a)
                # bigrams.add(b)
                # bigrams.add((a, b))
                ...
    
    return bigrams


def generate_embraced_spaces_dict(ru_with_spaces_path: Path, en_with_spaces_path: Path, output_path: Path):
    with open(ru_with_spaces_path, 'r', encoding='utf-8') as ru_with_spaces_file:
        ru_with_spaces_dict = ru_with_spaces_file.read()
    
    with open(en_with_spaces_path, 'r', encoding='utf-8') as en_with_spaces_file:
        en_with_spaces_dict = en_with_spaces_file.read()
    
    out_data_ru = ''
    ru_dict_len = len(ru_with_spaces_dict)
    en_dict_len = len(en_with_spaces_dict)
    for i, element in enumerate(ru_with_spaces_dict):
        if element == ' ' and i < ru_dict_len - 1:
            left = ru_with_spaces_dict[i - 1]
            right = ru_with_spaces_dict[i + 1]
            out_data_ru += f'{left}{element}{right} '
    
    out_data_en = ''
    for i, element in enumerate(en_with_spaces_dict):
        if element == ' ' and i < en_dict_len - 1:
            left = en_with_spaces_dict[i - 1]
            right = en_with_spaces_dict[i + 1]
            out_data_en += f'{left}{element}{right} '
    
    output_path_ru = (output_path
                      / 'ru_spaces_embraced.txt')
    output_path_en = output_path / 'en_spaces_embraced.txt'
    
    with open(output_path_ru, 'w', encoding='utf-8') as out_file:
        out_file.write(out_data_ru)
    
    with open(output_path_en, 'w', encoding='utf-8') as out_file:
        out_file.write(out_data_en)


if __name__ == '__main__':
    from sliding_window import sliding_window_generator
    
    # test_string = 'abcdef'
    #
    # raw_ru_dictionary_name = 'ru_dict.txt'
    # prepared_ru_dictionary_name = 'ru_dict_prepared.txt'
    # input_ru_path = '../assets/lang/ru'
    #
    # raw_en_dictionary_name = 'raw_en_dict.txt'
    # prepared_en_dictionary_name = 'en_dict_prepared.txt'
    # input_en_path = '../assets/lang/en'
    #
    # clean_ru_dictionary(raw_ru_dictionary_name, prepared_ru_dictionary_name, input_ru_path)
    # clean_en_dictionary(raw_en_dictionary_name, prepared_en_dictionary_name, input_en_path)
    
    # for w in sliding_window_generator(test_string, 5):
    #     print(w)
    
    ru_dict_path = Path('./datasets/early/ru/ru_spaced_dict.txt')
    en_dict_path = Path('./datasets/early/en/en_spaced_dict.txt')
    
    clean_ru_dictionary('ru_dict.txt', 'ru_dict_nl.txt', Path('../assets/lang/ru'), separator='')
    
    # generate_embraced_spaces_dict(ru_dict_path, en_dict_path, Path('./datasets/early/punct'))