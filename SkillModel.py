import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm as cm
from sklearn.cluster import KMeans
from scipy.signal import butter
import scipy.signal as signal

class SkillModel(object):

    def __init__(self, name, num_states, num_obs, empty=False):
        self.name = name

        self.N = num_states
        self.M = num_obs

        if empty:
            self.A = np.zeros((self.N, self.N))
            self.B = np.zeros((self.N, self.M))
            self.pi = np.zeros((self.N, 1))
        else:
            # Use random initialization
            self.A = np.random.rand(self.N, self.N)
            self.A = self.A / np.sum(self.A, axis=1, keepdims=True)
            self.B = np.random.rand(self.N, self.M)
            self.B = self.B / np.sum(self.B, axis=1, keepdims=True)
            self.pi = np.random.rand(self.N, 1)
            self.pi = self.pi / np.sum(self.pi)

    def forward_one_sequence_with_scale(self, obs_sequence):
        T = obs_sequence.shape[0]
        ct = np.zeros((T, ))
        thresh = 10e-6
        alpha = np.zeros((self.N, T))
        Ot = obs_sequence[0]
        alpha[:, [0]] = np.maximum(self.pi * self.B[:, [Ot]], thresh)
        ct[0] = 1 / np.sum(alpha[:, 0])
        alpha[:, 0] = alpha[:, 0] * ct[0]

        for i in range(1, T):
            Ot = obs_sequence[i]
            pi = np.dot(alpha[:, [i - 1]].T, self.A).T
            alpha[:, [i]] = np.maximum(pi * self.B[:, [Ot]], thresh)

            ct[i] = 1 / np.sum(alpha[:, i])
            alpha[:, i] = alpha[:, i] * ct[i]

        log_sequence_prob = - np.sum(np.log(ct))
        return alpha, ct, log_sequence_prob

    def backward_one_sequence_with_scale(self, obs_sequence, ct):
        T = obs_sequence.shape[0]
        beta = np.zeros((self.N, T))
        beta[:, -1] = 1
        beta[:, -1] = beta[:, -1] * ct[-1]
        thresh = 10e-6

        for i in range(T - 1, 0, -1):
            Ot = obs_sequence[i]
            beta[:, [i - 1]] = np.maximum(np.dot(self.A, self.B[:, [Ot]] * beta[:, [i]]), thresh)
            beta[:, i - 1] = beta[:, i - 1] * ct[i - 1]

        return beta

    def reevaluate_model(self, alpha, beta, obs_sequence):
        T = alpha.shape[1]   
        xsi = np.zeros((self.N, self.N, T - 1))
        gamma = alpha * beta / np.sum(alpha * beta, axis=0, keepdims=True)

        for i in range(T - 1):
            Ot = obs_sequence[i + 1]
            xsi_numerator = alpha[:, [i]] * self.A * beta[:, [i + 1]].T * self.B[:, [Ot]].T
            xsi_denominator = np.sum(xsi_numerator)

            xsi[:, :, i] = xsi_numerator / xsi_denominator

        self.pi = gamma[:, [0]]
        self.A = np.sum(xsi, axis=2) / np.sum(gamma[:, :-1], axis=1, keepdims=True)

        gamma_sum_over_t = np.sum(gamma, axis=1, keepdims=True)
        for k in range(self.M):
            mask = obs_sequence[np.newaxis, :] == k
            self.B[:, [k]] = np.sum(gamma * mask, axis=1, keepdims=True) / gamma_sum_over_t

    def baum_welch(self, obs_sequence):
        max_iter = 1000
        epsilon = 10e-6

        alpha, ct, log_sequence_prob = self.forward_one_sequence_with_scale(obs_sequence)
        beta = self.backward_one_sequence_with_scale(obs_sequence, ct)
        self.reevaluate_model(alpha, beta, obs_sequence)

        log_prob_prev = log_sequence_prob

        for i in range(max_iter):
            alpha, ct, log_sequence_prob = self.forward_one_sequence_with_scale(obs_sequence)
            beta = self.backward_one_sequence_with_scale(obs_sequence, ct)
            self.reevaluate_model(alpha, beta, obs_sequence)

            if np.abs(log_sequence_prob - log_prob_prev) < epsilon:
                break

            log_prob_prev = log_sequence_prob

def train_a_model(name, num_states, num_obs, obs_sequences):
    model = SkillModel(name, num_states, num_obs)
    model.baum_welch(obs_sequences)

    return model

def train_models(folder, num_states, num_obs, classifier=None):
    if classifier:
        kmeans, obs_sequences = discretize_raw(folder, num_obs)
    else: 
        kmeans, obs_sequences = discretize_raw(folder, num_obs, classifier)

    one_shot_models = []
    for name, obs_sequence in obs_sequences.items():
        model = SkillModel(name, num_states, num_obs)
        model.baum_welch(obs_sequence)
        one_shot_models.append(model)

    levels = ["E","I","N"]
    models = []
    counts = []
    for i in range(len(levels)):
        name = levels[i]
        model = SkillModel(name, num_states, num_obs, empty=True)
        models.append(model)
        counts.append(0)

    for model in one_shot_models:
        for i in range(len(levels)):
            if levels[i] in model.name:
                models[i].A = models[i].A + model.A
                models[i].B = models[i].B + model.B
                models[i].pi = models[i].pi + model.pi
                counts[i] = counts[i] + 1
                break

    for i in range(len(levels)):
        models[i].A = models[i].A / counts[i]
        models[i].B = models[i].B / counts[i]
        models[i].pi = models[i].pi / counts[i]

    return kmeans, models

def predict(models, obs_sequence):
    K = len(models)
    log_probs = np.zeros((K, ))

    for i in range(K):
        _, _, log_prob = models[i].forward_one_sequence_with_scale(obs_sequence)
        log_probs[i] = log_prob

    probs = np.exp(log_probs / obs_sequence.shape[0])
    probs = probs / np.sum(probs)
    top_guess = log_probs[log_probs.argsort()][-1]
    second_guess = log_probs[log_probs.argsort()][-2]

    if top_guess - second_guess < 8:
        label = "Unknown_N"
    else:
        number = np.argmax(log_probs)
        label = models[number].name

    confidence_naive = (top_guess - second_guess) / (- top_guess)
    P1 = np.exp(top_guess / obs_sequence.shape[0])
    P2 = np.exp(second_guess / obs_sequence.shape[0])
    confidence_timed = (P1 - P2) / P1
    return label, log_probs, probs, confidence_naive, confidence_timed

def plot_correlation(models):
    N = models[0].A.shape[0]
    M = models[0].B.shape[1]

    fig, axs = plt.subplots(1, 3, figsize=(16, 10))
    axs = axs.ravel()
    cmap = cm.get_cmap('hot')
    labels = list(range(1, N + 1))

    for i in range(3):
        name = models[i].name
        A = models[i].A
        cax = axs[i].imshow(A, interpolation="nearest", cmap=cmap)
        axs[i].set_title("Correlation Matrix for " + name)
        axs[i].set_xticklabels(labels)
        axs[i].set_yticklabels(labels)
        ticks = np.linspace(np.min(A), np.max(A), 5)
        fig.colorbar(cax, ax=axs[i], ticks=ticks)

    plt.suptitle("Model Zoo with " + str(N) + " states and " + str(M) + " observations")
    filename = "Correlation_Matrix"

    plt.savefig(filename)
    plt.close()
    return 0

def plot_probs(name, label, confidence_naive, confidence_timed, log_probs, probs):
    levels = [" ","E","I","N"]
    ind = np.array([0, 1, 2])

    fig = plt.figure(figsize=(12, 5))
    fig.tight_layout()

    ax1 = fig.add_subplot(121)
    ax1.bar(ind, log_probs, facecolor="skyblue", align="center")
    ax1.set_xticklabels(levels)
    for x, y in zip(ind, log_probs):
        ax1.text(x, y, '%.1f' % y, ha='center', va='bottom')
    ax1.set_title("Log likelihoods for each group")

    ax2 = fig.add_subplot(122)
    ax2.bar(ind, probs, facecolor="orangered", align="center")
    ax2.set_xticklabels(levels)
    for x, y in zip(ind, probs):
        ax2.text(x, y, '%.4f' % y, ha='center', va='bottom')
    ax2.set_title("Normalized likelihoods for each group")

    confidence_naive = float('%.3f' % confidence_naive)
    confidence_timed = float('%.3f' % confidence_timed)

    plt.suptitle("Trial name: " + name + "  Prediction: " + label +
                 "     Confidence 1: " + str(confidence_naive) + "   Confidence 2: " + str(confidence_timed))

    filename = name + "_predictions"

    plt.savefig(filename)

    plt.close()
    return 0

def discretize_raw(folder, M, classifier=None):
    sequences = {}
    obs_sequences = {}

    X = np.zeros((0, 6))

    for root, dirs, files in os.walk(folder):
        for file in files:
            if ".csv" not in file:
                continue
        
            name = file.split(".csv")[0]
            raw_sequence = pd.read_csv(os.path.join(root, file), sep=',',header = None)[[0,1,2,9,10,11]]
            raw_sequence = raw_sequence.as_matrix()

            filtered = butter_lowpass_filter(raw_sequence)
            sequences[name] = filtered
            X = np.concatenate((X, raw_sequence), axis=0)

    if classifier is None:
        classifier = KMeans(n_clusters=M, random_state=0).fit(X)

    for name, sequence in sequences.items():
        obs_sequences[name] = classifier.predict(sequence)

    return classifier, obs_sequences

def butter_lowpass_filter(data):
    lowcut = 0.4
    order = 2
    y = np.zeros(data.shape)
    b, a = butter(order, lowcut, output='ba')
    for i in range(0, data.shape[1]):
        y[:,i] = signal.filtfilt(b, a, data[:,i])
        
    return y