# -*- coding: utf-8 -*-
"""
Created on Mon Sept 29 13:38:35 2018

@author: xngu0004
"""

import numpy as np
import pickle as p
import SkillModel as SM
from sklearn.metrics import classification_report

f_test = open("./data/dataY_test.csv")
y_test = np.loadtxt(fname = f_test, delimiter = ',')
my_pred = []

def main(retrain, recluster, S, O):

    train_folder = "train_data/"
    val_folder = "val_data/"

    model_file = "./models/models_" + str(S) + "_" + str(O) + ".pkl"
    kmeans_file = "./models/kmeans_" + str(O) + ".pkl"

    if recluster:
        kmeans, _ = SM.discretize_raw(train_folder, O, classifier=None)
        with open(kmeans_file, 'wb') as file:
            p.dump(kmeans, file)
    else:
        with open(kmeans_file, "rb") as file:
            kmeans = p.load(file)

    if retrain:
        _, models = SM.train_models(train_folder, S, O, classifier=kmeans)
        with open(model_file, 'wb') as file:
            p.dump(models, file)
    else:
        with open(model_file, 'rb') as file:
            models = p.load(file)

    _, obs_sequences = SM.discretize_raw(val_folder, O, kmeans)
    levels = ["E","I","N"]
    print("--------------o0o-------------")
    for name, obs_sequence in obs_sequences.items():
        print("Trial: ", name)
        pred, log_probs, probs, confidence_naive, confidence_timed = SM.predict(models, obs_sequence)
        print("Prediction: ", pred)
        my_pred.append(pred)
        for i in range(3):
            print("Confidence for ", levels[i], ": ", log_probs[i], " Prob: ", probs[i]*100)
            
        print("--------------o0o-------------")
        SM.plot_probs(name, pred, confidence_naive, confidence_timed, log_probs, probs)

    SM.plot_correlation(models)
    y_pred = np.array(my_pred)
    print('Accuracy:' + str(np.sum(y_pred == y_test)/y_test.shape[0]))
    print(classification_report(y_test, y_pred))
    return 0

if __name__ == "__main__":
    
    main(retrain = True, recluster = True, S = 8, O = 15)
    print("Done.")
