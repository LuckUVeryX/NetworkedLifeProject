import mainRBM
import projectLib as lib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
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
epochs = 2

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
regularization = [0.01, 0]

# * Momemntum
# TODO Hyper parameter tuning
# ? 0 to 1
momentum = 0.9


def hyperparameterTuning():
    # Initialise Variables
    results = []
    x = 0
    y = 0

    # ! Update based on the number of parameters testing
    num_plots = len(initialLearningRate) * len(learningRateDecay)
    num_plot_dimension = int(np.ceil(np.sqrt(num_plots)))
    print(num_plot_dimension)

    fig, axs = plt.subplots(num_plot_dimension, num_plot_dimension)

    # ! Loop over the different parameters
    for i in range(len(initialLearningRate)):
        for j in range(len(learningRateDecay)):

            # ! Modify print statement to reflect training parameters
            print("----------Training with decay rate {} and initial learning rate {}----------".format(
                initialLearningRate[i], learningRateDecay[j]))

            # ! Update the training function
            train_loss, val_loss = mainRBM.main(K=K,
                                                F=F,
                                                epochs=epochs,
                                                initialLearningRate=initialLearningRate[i],
                                                learningRateDecay=learningRateDecay[j],
                                                regularization=regularization,
                                                momentum=momentum)

            # ! Add parameter to dictionary
            results.append({"Validation Loss": min(val_loss),
                            "Init Learn Rate": initialLearningRate[i],
                            "Learn Rate Decay": learningRateDecay[j]})

            # plot the evolution of training and validation RMSE
            # ! Update the title of plots
            axs[x, y].plot(train_loss)
            axs[x, y].plot(val_loss)
            axs[x, y].set(xlabel='epoch', ylabel='RMSE')
            axs[x, y].set_title('LR {} & Decay {}'.format(
                initialLearningRate[i], learningRateDecay[j]))

            # Update the index to plot plots
            if x < num_plot_dimension - 1:
                x += 1
            else:
                y += 1
                x = 0

    # * Code to output predictions without overwriting
    now = datetime.now()
    date = now.strftime("%d%m%Y")
    time = now.strftime("%H%M")
    if not os.path.exists('predictions/{}/'.format(date)):
        os.makedirs('predictions/{}/'.format(date))

    # Output CSV
    df = pd.DataFrame(results)
    df = df.sort_values(by='Validation Loss', ascending=True)
    df.to_csv("predictions/{}/{}.csv".format(date, time))

    # Output Plot
    for ax in axs.flat:
        ax.label_outer()
    line_labels = ["Training Loss", "Validation Loss"]
    fig.legend(labels=line_labels)
    plt.savefig("predictions/{}/{}.pdf".format(date, time))
    plt.show()


hyperparameterTuning()
