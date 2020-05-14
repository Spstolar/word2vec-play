"""
This extends code found at https://towardsdatascience.com/word2vec-from-scratch-with-numpy-8786ddd49e72
"""
import numpy as np
import re
import matplotlib.pyplot as plt

explanation = False
if explanation:
    print('Word2Vec Python implementation.')

MAX_STRING = 100
EXP_TABLE_SIZE = 1000
MAX_EXP = 6
MAX_SENTENCE_LENGTH = 1000
MAX_CODE_LENGTH = 40

vocab_hash_size = int(3e7)  # 30 million

class VocabWord:
    def __init__(self, word_freq, point, word, code, codelen):
        self.word_freq = word_freq
        self.point = point
        self.word = word
        self.code = code
        self.codelen = codelen


binary = 0
cbow = 1 
debug_mode = 2
window = 5
min_count = 5
num_threads = 12
min_reduce = 1

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

# for i, word in enumerate(example_tokenized):
#     if i < window_radius:
#         continue 
#     if i > len(example_tokenized) - window_radius - 1:
#         continue
#     print('Current word:', word)
#     context = get_context(example_tokenized, i, window_radius)
#     print('Context:', context)


def InitUnigramTable():
    return []

# Start word2vec from scratch with numpy implementation
def tokenize(text):
    # so things that are word characters or strings with carets (^) or single quotes (') are counted as words
    # [^abc] means except abc, but if the caret is not at the beginning, it is the same as escaping the caret
    pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
    return pattern.findall(text.lower())

def init_mapping(tokens):
    word_to_id = dict()
    id_to_word = dict()

    for i, token in enumerate(set(tokens)):
        word_to_id[token] = i
        id_to_word[i] = token

    return word_to_id, id_to_word


def generate_training_data(tokens, word_to_id, window_size):
    """For a collection of tokens return X a list of input (center)
    tokens and Y a list of correspond output contexts.

    Arguments:
        tokens {array} -- list of doc tokens
        word_to_id {dict} -- map from words in vocab to index
        window_size {int} -- context size
    """
    N = len(tokens)
    X, Y = [], []

    for i in range(N):
        # grab the window_size tokens before token i, stopping
        # if you reach the beginning of the sentence, as well as the
        # window_size tokens after token i, stopping if you reach the
        # end
        context_indices = list(range(max(0, i - window_size), i)) + \
            list(range(i + 1, min(N, i + window_size + 1)))
        for j in context_indices:
            X.append(word_to_id[tokens[i]])
            Y.append(word_to_id[tokens[j]])

    X = np.array(X)
    X = np.expand_dims(X, axis=0) # add a buffer array, so that X = X'[0]
    Y = np.array(Y)
    Y = np.expand_dims(Y, axis=0)

    return X, Y

doc = "After the deduction of the costs of investing " \
    "beating the stock market is a loser's game."

tokens = tokenize(doc)
word_to_id, id_to_word = init_mapping(tokens)
X, Y = generate_training_data(tokens, word_to_id, 3)

print('X.shape', X.shape)
vocab_size = len(id_to_word)
m = Y.shape[1]
# create one-hot encoding for Y
Y_one_hot = np.zeros((vocab_size, m))
if explanation:
    print('Doc:', doc)
    print('X flattened and converted to words:', [id_to_word[word] for word in X.flatten()])
    print('Y flattened and converted to words:', [id_to_word[word] for word in Y.flatten()])
# flatten unrolls a tensor into a single array
# so, e.g., a matrix becomes the list of the entries (ordered by row and column)
Y_one_hot[Y.flatten(), np.arange(m)] = 1
print(Y_one_hot)

def init_word_embedding(vocab_size, embedding_size):
    word_embedding = np.random.randn(vocab_size, embedding_size) * 0.01
    return word_embedding

def init_dense(input_size, ouput_size):
    W = np.random.randn(ouput_size, input_size) * 0.01
    return W

def init_params(vocab_size, embedding_size):
    word_embedding = init_word_embedding(vocab_size, embedding_size)
    W = init_dense(embedding_size, vocab_size)

    parameters = {}
    parameters['word_embedding'] = word_embedding
    parameters['W'] = W

    return parameters

# Forward Propagation:
# 1. obtain input word's vector rep
# 2. pass word embedding to dense layer
# 3. apply softmax to output of dense layer

def indices_to_word_vec(indices, parameters):
    m = indices.shape[1]
    word_embedding = parameters['word_embedding']
    word_vec = word_embedding[indices.flatten(), :].T

    assert(word_vec.shape == (word_embedding.shape[1], m))

    return word_vec

def linear_dense(word_vec, parameters):
    m = word_vec.shape[1]
    W = parameters['W']
    Z = np.dot(W, word_vec)

    assert(Z.shape == (W.shape[0], m))

    return W, Z

def softmax(Z):
    softmax_out = np.divide(
        np.exp(Z),
        np.sum(np.exp(Z), axis=0, keepdims=True) + 0.001
    )

    assert(softmax_out.shape == Z.shape)

    return softmax_out

def forward_propagation(indices, parameters):
    word_vec = indices_to_word_vec(indices, parameters)
    W, Z = linear_dense(word_vec, parameters)
    softmax_out = softmax(Z)

    caches = {}
    caches['indices'] = indices
    caches['word_vec'] = word_vec
    caches['W'] = W
    caches['Z'] = Z

    return softmax_out, caches

def cross_entropy(softmax_out, Y):
    m = softmax_out.shape[1]
    cost = -(1 / m) * np.sum(np.sum(Y * np.log(softmax_out + 0.001), axis=0, keepdims=True), axis=1)
    return cost

def softmax_backward(Y, softmax_out):
    dL_dZ = softmax_out - Y

    assert(dL_dZ.shape == softmax_out.shape)
    return dL_dZ

def delta_cross_entropy(X, y):
    """https://deepnotes.io/softmax-crossentropy#derivative-of-cross-entropy-loss-with-softmax

    Arguments:
        X {array} -- output from fully connected layer (num_examples x num_classes)
        y {array} -- labels (num_examples x 1)
    """
    m = y.shape[0]
    grad = softmax(X)
    grad[range(m), y] -= 1
    grad = grad / m
    return grad

def dense_backward(dL_dZ, caches):
    W = caches['W']
    word_vec = caches['word_vec']
    m = word_vec.shape[1]

    dL_dW = (1 / m) * np.dot(dL_dZ, word_vec.T)
    dL_dword_vec = np.dot(W.T, dL_dZ)

    return dL_dW, dL_dword_vec

def backward_propagation(Y, softmax_out, caches):
    dL_dZ = softmax_backward(Y, softmax_out)
    dL_dW, dL_dword_vec = dense_backward(dL_dZ, caches)

    gradients = dict()
    gradients['dL_dZ'] = dL_dZ
    gradients['dL_dW'] = dL_dW
    gradients['dL_dword_vec'] = dL_dword_vec

    return gradients

def update_parameters(parameters, caches, gradients, learning_rate):
    vocab_size, embedding_size = parameters['word_embedding'].shape
    indices = caches['indices']
    word_embedding = parameters['word_embedding']
    dL_dword_vec = gradients['dL_dword_vec']
    m = indices.shape[-1]

    word_embedding[indices.flatten(), :] -= dL_dword_vec.T * learning_rate

    parameters['W'] -= learning_rate * gradients['dL_dW']


def skipgram_model_training(X, Y, vocab_size, embedding_size, learning_rate, epochs, batch_size=256, parameters=None, print_cost=True, plot_cost=True):
    costs = []
    m = X.shape[1]

    if parameters is None:
        parameters = init_params(vocab_size, embedding_size)
    
    for epoch in range(epochs):
        epoch_cost = 0
        batch_indicies = list(range(0, m, batch_size))
        np.random.shuffle(batch_indicies)
        for i in batch_indicies:
            X_batch = X[:, i:i+batch_size]
            Y_batch = Y[:, i:i+batch_size]

            softmax_out, caches = forward_propagation(X_batch, parameters)
            gradients = backward_propagation(Y_batch, softmax_out, caches)
            update_parameters(parameters, caches, gradients, learning_rate)
            cost = cross_entropy(softmax_out, Y_batch)
            epoch_cost += np.squeeze(cost)

        costs.append(epoch_cost)
        if print_cost and epoch % (epochs // 500) == 0:
            print(f"Cost after epoch {epoch}: {epoch_cost}.")
        if epoch % (epochs // 100) == 0:
            learning_rate *= 0.98

    if plot_cost:
        plt.plot(np.arange(epochs), costs)
        plt.xlabel('# of epochs')
        plt.ylabel('cost')

    return parameters

skipgram_model_training(X, Y, vocab_size, embedding_size=10, learning_rate=0.9, epochs=500, batch_size=2)