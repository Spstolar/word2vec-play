"""
This extends code found at https://towardsdatascience.com/word2vec-from-scratch-with-numpy-8786ddd49e72
"""
import numpy as np
import re
import matplotlib.pyplot as plt

explanation = False
if explanation:
    print('Word2Vec Python implementation.')

class Data:
    def __init__(self, document, window_size=3):
        self.tokens = self.tokenize(document)
        self.word_to_id, self.id_to_word = self.init_mapping(self.tokens)
        self.window_size = window_size
        self.X, self.Y = self.generate_training_data(self.tokens, self.window_size)

        self.vocab_size = len(self.id_to_word)
        self.m = self.Y.shape[1]
        self.y_one_hot = self.one_hot_encode(self.Y)
        if explanation:
            print('Doc:', doc)
            print('X flattened and converted to words:', [self.id_to_word[word] for word in self.X.flatten()])
            print('Y flattened and converted to words:', [self.id_to_word[word] for word in self.Y.flatten()])

    def one_hot_encode(self, Y):
        # create one-hot encoding for Y
        Y_one_hot = np.zeros((self.vocab_size, Y.shape[1]))

        # flatten unrolls a tensor into a single array
        # so, e.g., a matrix becomes the list of the entries (ordered by row and column)
        Y_one_hot[Y.flatten(), np.arange(self.m)] = 1

        return Y_one_hot

    def tokenize(self, text):
        # so things that are word characters or strings with carets (^) or single quotes (') are counted as words
        # [^abc] means except abc, but if the caret is not at the beginning, it is the same as escaping the caret
        pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
        return pattern.findall(text.lower())

    def init_mapping(self, tokens):
        word_to_id = dict()
        id_to_word = dict()

        for i, token in enumerate(set(tokens)):
            word_to_id[token] = i
            id_to_word[i] = token

        return word_to_id, id_to_word


    def generate_training_data(self, tokens, window_size):
        """For a collection of tokens return X a list of input (center)
        tokens and Y a list of correspond output contexts.

        Arguments:
            tokens {array} -- list of doc tokens
            window_size {int} -- context size
        """
        N = len(tokens)
        X, Y = [], []

        # TODO: this is sort of "cheating" by breaking up a multiclass
        # problem into a bunch in single-class ones. Not sure what difference
        # that makes!
        for i in range(N):
            # grab the window_size tokens before token i, stopping
            # if you reach the beginning of the sentence, as well as the
            # window_size tokens after token i, stopping if you reach the
            # end
            context_indices = list(range(max(0, i - window_size), i)) + \
                list(range(i + 1, min(N, i + window_size + 1)))
            for j in context_indices:
                X.append(self.word_to_id[tokens[i]])
                Y.append(self.word_to_id[tokens[j]])

        X = np.array(X)
        X = np.expand_dims(X, axis=0) # add a buffer array, so that X = X'[0]
        Y = np.array(Y)
        Y = np.expand_dims(Y, axis=0)

        return X, Y


class Model:
    def __init__(self, data, embedding_size, learning_rate, verbose=True):
        self.data = data
        self.X = self.data.X
        self.y = self.data.y_one_hot
        self.vocab_size = self.data.vocab_size
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.super_verbose = False
        self.cost_printing = 500
        self.learning_rate_change = 500

        if self.verbose:
            print(f"The vocabulary has {self.vocab_size} words.")
            print("Initializing weights...")
            print(f"Word embedding vectors matrix: {self.vocab_size} x {self.embedding_size}.")
            print(f"Dense layer weights: {self.embedding_size} x {self.vocab_size}.")
        self.word_embedding = self.generate_random_matrix(self.vocab_size, self.embedding_size)
        self.dense_weights = self.generate_random_matrix(self.vocab_size, self.embedding_size)
    
    def generate_random_matrix(self, num_rows, num_columns):
        weights = np.random.randn(num_rows, num_columns) * 0.01
        return weights

    def indices_to_word_vec(self, indices):
        # m is how many examples are being used
        # when just doing a verbose example we are passing [word_index] 
        # when it's a batch we are passing [[word_1_index, ..., word_n_index]]
        if len(indices.shape) == 1:
            m = 1
        else:
            m = indices.shape[1]
        # We flatten the indices so we're just injesting a simple list.
        # We grab those rows from our word embedding and then take the transpose
        # for matrix products.
        word_vec = self.word_embedding[indices.flatten(), :].T

        assert(word_vec.shape == (self.word_embedding.shape[1], m))
        return word_vec

    def linear_dense(self, word_vec):
        # m - how many words are in the batch
        m = word_vec.shape[1]
        Z = np.dot(self.dense_weights, word_vec)

        assert(Z.shape == (self.dense_weights.shape[0], m))

        return Z

    def softmax(self, Z):
        softmax_out = np.divide(
            np.exp(Z),
            np.sum(np.exp(Z), axis=0, keepdims=True) + 0.001
        )

        assert(softmax_out.shape == Z.shape)

        return softmax_out

    def forward_propagation(self, indices):
        """
        Forward Propagation:
        1. obtain input word's vector rep
        2. pass word embedding to dense layer
        3. apply softmax to output of dense layer

        Arguments:
            indices {list}  -- this is a list which contains the list of indices for the batch

        Returns:
            [type] -- [description]
        """
        if self.super_verbose:
            print('indices for forward prop', indices)
        # for batches, word_vec is a matrix whose columns are the word embeddings
        word_vec = self.indices_to_word_vec(indices)
        # apply the dense layer weights to the word vec
        Z = self.linear_dense(word_vec)
        # apply softmax to each column of Z
        softmax_out = self.softmax(Z)

        cache = {}
        cache['indices'] = indices
        cache['word_vec'] = word_vec
        cache['Z'] = Z

        return softmax_out, cache

    def cross_entropy(self, softmax_out, Y):
        m = softmax_out.shape[1]
        cost = np.sum(np.sum(Y * np.log(softmax_out + 0.001), axis=0, keepdims=True), axis=1)
        cost *= -(1 / m)
        return cost

    def softmax_backward(self, Y, softmax_out):
        # if Y.ndim == 1:
            # Y = Y.reshape(-1, 1)
        dL_dZ = softmax_out - Y

        assert(dL_dZ.shape == softmax_out.shape)
        return dL_dZ

    def dense_backward(self, dL_dZ, word_vec):
        W = self.dense_weights
        m = word_vec.shape[1]

        dL_dW = (1 / m) * np.dot(dL_dZ, word_vec.T)
        dL_dword_vec = np.dot(W.T, dL_dZ)
        if self.super_verbose:
            print('-' * 80)
            print('dense_backward')
            print('-' * 80)
            print('dL_dW', dL_dW)
            print('wordvec', word_vec)
            print('dL_dZ', dL_dZ)
            print('dl_dwordvec', dL_dword_vec)

        return dL_dW, dL_dword_vec

    def backward_propagation(self, Y, softmax_out, cache):
        dL_dZ = self.softmax_backward(Y, softmax_out)
        word_vec = cache['word_vec']
        dL_dW, dL_dword_vec = self.dense_backward(dL_dZ, word_vec)

        gradients = dict()
        gradients['dL_dZ'] = dL_dZ
        gradients['dL_dW'] = dL_dW
        gradients['dL_dword_vec'] = dL_dword_vec

        return gradients

    def update_weights(self, cache, gradients):
        indices = cache['indices']
        dL_dword_vec = gradients['dL_dword_vec']

        self.word_embedding[indices.flatten(), :] -= dL_dword_vec.T * self.learning_rate
        self.dense_weights -= self.learning_rate * gradients['dL_dW']

    def skipgram_model_training(self, epochs, batch_size=256, print_cost=True, plot_cost=True):
        costs = []
        # m - the number of center words in our training set
        m = self.X.shape[1]
        
        for epoch in range(epochs):
            epoch_cost = 0
            batch_start_indices = list(range(0, m, batch_size))
            np.random.shuffle(batch_start_indices)
            for i in batch_start_indices:
                X_batch = self.X[:, i:i+batch_size]
                Y_batch = self.y[:, i:i+batch_size]

                # print(X_batch)
                # print(Y_batch)
                softmax_out, cache = self.forward_propagation(X_batch)
                gradients = self.backward_propagation(Y_batch, softmax_out, cache)
                self.update_weights(cache, gradients)
                cost = self.cross_entropy(softmax_out, Y_batch)
                epoch_cost += np.squeeze(cost)

            costs.append(epoch_cost)
            if print_cost and epoch % (epochs // self.cost_printing) == 0:
                print(f"Cost after epoch {epoch}: {epoch_cost}.")
                # print(gradients)
                if np.isnan(epoch_cost):
                    print('Cost was nan, breaking...')
                    break
            if epoch % (epochs // self.learning_rate_change) == 0:
                self.learning_rate *= 0.98

        if plot_cost:
            plt.plot(np.arange(epochs), costs)
            plt.xlabel('# of epochs')
            plt.ylabel('cost')


    def train(self):
        self.skipgram_model_training(epochs=500, batch_size=2)

    def verbose_example(self):
        X_example = self.X[:, 3]
        y_example = self.y[:, 3].reshape(-1, 1)
        y_example_index = self.data.Y[:, 3]
        center_word = [self.data.id_to_word[word] for word in X_example.flatten()]
        context_words = [self.data.id_to_word[word] for word in y_example_index.flatten()]
        print('Center word:', X_example, center_word)
        print('Context:', y_example_index, context_words)

        print("Propagating the example through the network.")
        softmax_out, cache = self.forward_propagation(X_example)
        print('Word embedding:', cache['word_vec'])
        print('Dense layer output:', cache['Z'])
        print('Softmax output:', softmax_out)
        cross_entropy_result = self.cross_entropy(softmax_out, y_example)
        print("Cross entropy loss:", cross_entropy_result)

        gradients = self.backward_propagation(y_example, softmax_out, cache)
        print('Gradients:', gradients)

if __name__ == "__main__":
    doc = "After the deduction of the costs of investing " \
        "beating the stock market is a loser's game."

    data = Data(doc)
    model = Model(data, 20, 0.9)
    model.train()