import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class ELMPCAClassifier(ClassifierMixin, BaseEstimator):
    w_input_: np.array
    means_: np.array
    eig_vec_: np.array
    explained_var_: np.array
    w_output_: np.array

    def __init__(self, neurons: int, expl_var_threshold: float = 0.999):
        self.neurons = neurons
        self.expl_var_threshold = expl_var_threshold

    def fit(self, X, y):
        # generate random hidden layer weights [store input weights]
        self.w_input_ = np.random.uniform(size=(X.shape[1] + 1, self.neurons))

        # calculate hidden layer input
        hidden_input = (
            np.hstack([np.ones(shape=(X.shape[0], 1)), X])
            @ self.w_input_
        )

        # calculate hidden layer output
        hidden_output = np.tanh(hidden_input)
        hidden_output = np.hstack([
            np.ones(shape=(X.shape[0], 1)),
            hidden_output
        ])

        # calculate hidden pca layer input [store training data mean and eigen vectors]
        self.means_ = hidden_output.mean(axis=0)
        cov = np.cov((hidden_output - self.means_).T)
        eig_val, self.eig_vec_ = np.linalg.eig(np.round(cov, 5))

        # remove columns based on column weight norm
        self.explained_var_ = eig_val / np.sum(eig_val)
        num_var = np.array(sorted(enumerate(self.explained_var_), key=lambda x: x[1], reverse=True))
        num_var[:, 1] = np.cumsum(num_var[:, 1]) <= self.expl_var_threshold
        selector_binary = np.array(sorted(num_var, key=lambda x: x[0], reverse=False))[:, 1]

        selector_matrix = np.eye(self.explained_var_.shape[0]) * selector_binary

        self.eig_vec_ = self.eig_vec_ @ selector_matrix

        # calculate hidden pca layer output
        pca_input = np.dot(hidden_output - self.means_, self.eig_vec_)
        pca_output = np.hstack([
            np.ones(shape=(X.shape[0], 1)),
            pca_input
        ])

        # calculate output layer weights [store output weights]
        self.w_output_ = np.linalg.pinv(pca_output) @ y

        return self

    def predict(self, X):
        # calculate hidden layer input [use input weights]
        hidden_input = (
            np.hstack([np.ones(shape=(X.shape[0], 1)), X])
            @ self.w_input_
        )

        # calculate hidden layer output
        hidden_output = np.tanh(hidden_input)
        hidden_output = np.hstack([
            np.ones(shape=(X.shape[0], 1)),
            hidden_output
        ])

        # calculate pca layer output
        pca_input = np.dot(hidden_output - self.means_, self.eig_vec_)
        pca_output = np.hstack([
            np.ones(shape=(X.shape[0], 1)),
            pca_input
        ])

        # calculate output layer input values
        output_input = pca_output @ self.w_output_

        # output layer activation function is linear
        return output_input.reshape(-1) > 0


class ELMRegClassifier(ClassifierMixin, BaseEstimator):
    w_input_: np.array
    w_output_: np.array

    def __init__(self, neurons: int, reg: float = 1e3):
        self.neurons = neurons
        self.reg = reg

    def fit(self, X, y):
        # GENERATE RANDOM HIDDEN LAYER WEIGHTS
        self.w_input_ = np.random.uniform(size=(X.shape[1] + 1, self.neurons))

        # CALCULATE HIDDEN LAYER INPUT
        hidden_input = (
                np.hstack([np.ones(shape=(X.shape[0], 1)), X])
                @ self.w_input_
        )

        # CALCULATE HIDDEN LAYER OUTPUT
        hidden_output = np.tanh(hidden_input)
        hidden_output = np.hstack([
            np.ones(shape=(X.shape[0], 1)),
            hidden_output
        ])

        # CALCULATE OUTPUT LAYER WEIGHTS
        aux = np.linalg.pinv(
            (hidden_output.T @ hidden_output)
            + (self.reg * np.identity(self.neurons + 1))
        )
        self.w_output_ = aux @ hidden_output.T @ y

        return self

    def predict(self, X):
        # CALCULATE HIDDEN LAYER INPUT
        hidden_input = (
                np.hstack([np.ones(shape=(X.shape[0], 1)), X])
                @ self.w_input_
        )

        # CALCULATE HIDDEN LAYER OUTPUT
        hidden_output = np.tanh(hidden_input)
        hidden_output = np.hstack([
            np.ones(shape=(X.shape[0], 1)),
            hidden_output
        ])

        # CALCULATE OUTPUT LAYER INPUT VALUES
        output_input = hidden_output @ self.w_output_

        # OUTPUT LAYER ACTIVATEION FUNCTION IS LINEAR
        return output_input.reshape(-1) > 0


class ELMClassifier(ClassifierMixin, BaseEstimator):
    w_input_: np.array
    w_output_: np.array

    def __init__(self, neurons: int):
        self.neurons = neurons

    def fit(self, X, y):
        # GENERATE RANDOM HIDDEN LAYER WEIGHTS
        self.w_input_ = np.random.uniform(size=(X.shape[1] + 1, self.neurons))

        # CALCULATE HIDDEN LAYER INPUT
        hidden_input = (
                np.hstack([np.ones(shape=(X.shape[0], 1)), X])
                @ self.w_input_
        )

        # CALCULATE HIDDEN LAYER OUTPUT
        hidden_output = np.tanh(hidden_input)
        hidden_output = np.hstack([
            np.ones(shape=(X.shape[0], 1)),
            hidden_output
        ])

        # CALCULATE OUTPUT LAYER WEIGHTS
        self.w_output_ = np.linalg.pinv(hidden_output) @ y

        return self

    def predict(self, X):
        # CALCULATE HIDDEN LAYER INPUT
        hidden_input = (
                np.hstack([np.ones(shape=(X.shape[0], 1)), X])
                @ self.w_input_
        )

        # CALCULATE HIDDEN LAYER OUTPUT
        hidden_output = np.tanh(hidden_input)
        hidden_output = np.hstack([
            np.ones(shape=(X.shape[0], 1)),
            hidden_output
        ])

        # CALCULATE OUTPUT LAYER INPUT VALUES
        output_input = hidden_output @ self.w_output_

        # OUTPUT LAYER ACTIVATEION FUNCTION IS LINEAR
        return output_input.reshape(-1) > 0
