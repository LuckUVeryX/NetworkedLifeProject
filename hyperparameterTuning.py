import mainRBM
import projectLib as lib
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import os

training = lib.getTrainingData()
validation = lib.getValidationData()
# You could also try with the chapter 4 data
# training = lib.getChapter4Data()

trStats = lib.getUsefulStats(training)
vlStats = lib.getUsefulStats(validation)

# ! Parameters
K = 5

# SET PARAMETERS HERE!!!
# number of hidden units
# TODO Hyper parameter tuning F, (number of hidden units)
F = 15
epochs = 3

# * We are using adaptive learning rate instead of a fixed gradientLearningRate
# //gradientLearningRate = 0.1
# * Use this to select ideal learning rate at epoch 1
initialLearningRate = [0.5, 0.1]
#  TODO Hyper parameter tuning
# ? Range from 0.01 to 1
learningRateDecay = [0.5, 0.1]

# * Set the regularization strength here
# TODO Hyper parameter tuning
# ? Range from 0 to 0.05
regularization = 0.01

# * Momemntum
# TODO Hyper parameter tuning
# ? 0 to 1
momentum = 0.9


def hyperparameterTuning():
    # Initialise Variables
    results = []
    fig, axs = plt.subplots(len(initialLearningRate),
                            len(learningRateDecay))

    for i in range(len(initialLearningRate)):
        for j in range(len(learningRateDecay)):

            print("----------Training with decay rate {} and initial learning rate {}----------".format(
                initialLearningRate[i], learningRateDecay[j]))

            train_loss, val_loss = mainRBM.main(K=K,
                                                F=F,
                                                epochs=epochs,
                                                initialLearningRate=initialLearningRate[i],
                                                learningRateDecay=learningRateDecay[j],
                                                regularization=regularization,
                                                momentum=momentum)

            results.append({"Validation Loss": min(val_loss),
                            "Init Learn Rate": initialLearningRate[i],
                            "Learn Rate Decay": learningRateDecay[j]})

            # plot the evolution of training and validation RMSE
            # axs.legend(loc='upper right')
            axs[i, j].plot(train_loss)
            axs[i, j].plot(val_loss)
            axs[i, j].set(xlabel='epoch', ylabel='RMSE')
            axs[i, j].set_title('LR {} & Decay {}'.format(
                initialLearningRate[i], learningRateDecay[j]))

    # * Code to output predictions without overwriting
    now = datetime.now()
    date = now.strftime("%d%m%Y")
    time = now.strftime("%H%M")
    if not os.path.exists('predictions/{}/{}/'.format(date, time)):
        os.makedirs('predictions/{}/{}/'.format(date, time))

    # Output CSV
    df = pd.DataFrame(results)
    df = df.sort_values(by='Validation Loss', ascending=True)
    df.to_csv("predictions/{}/{}/results.csv".format(date, time))

    for ax in axs.flat:
        ax.label_outer()
    line_labels = ["Training Loss", "Validation Loss"]
    fig.legend(labels=line_labels)
    plt.show()


hyperparameterTuning()
