import numpy as np

print('word2vec python implementation')

vocab_hash_size = int(3e7)  # 30 million

example = 'here is some text to parse through so we can try things out as we go'
print(example)

window_radius = 1
window_size = 2 * window_radius + 1
blank = '=blank='

def process_sentence(sentence):
    tokens = sentence.split() 
    for _ in range(window_radius):
        tokens.append(blank)
        tokens.insert(0, blank)
    return tokens

def get_context(example, position, window_radius):
    window_start = position - window_radius
    window_end = position + window_radius
    context = example[window_start:window_end + 1]
    return context

example_tokenized = process_sentence(example)
print(example_tokenized)

for i, word in enumerate(example_tokenized):
    if i < window_radius:
        continue 
    if i > len(example_tokenized) - window_radius - 1:
        continue
    print('Current word:', word)
    context = get_context(example_tokenized, i, window_radius)
    print('Context:', context)


def InitUnigramTable():
    return []

def init_mapping(tokens):
    word_to_id = dict()
    id_to_word = dict()

    for i, token in enumerate(set(tokens)):
        word_to_id[token] = i
        id_to_word[i] = token

    return word_to_id, id_to_word


