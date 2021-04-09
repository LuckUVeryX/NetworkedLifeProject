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

# * Parameters
# Ratings from 1-5
K = 5

epochs = 50

# TODO Hyper parameter tuning
# number of hidden units
F = [15]

initialLearningRate = [0.5, 0.1]

# ? Range from 0.01 to 1
learningRateDecay = [0.1, 0.5]

# ? Range from 0 to 0.05
regularization = [0.01, 0.05]

# ? 0 to 1
momentum = [0.9, 0.99]


def get_plot_dimension():
    num_plots = len(F) * len(initialLearningRate) * \
        len(learningRateDecay) * len(regularization) * len(momentum)

    return int(np.ceil(np.sqrt(num_plots)))


def update_plot_location(x, y, dimension):
    if x < dimension - 1:
        x += 1
    else:
        y += 1
        x = 0
    return x, y


def get_current_date_and_time():
    now = datetime.now()
    date = now.strftime("%d%m%Y")
    time = now.strftime("%H%M")
    return date, time


def hyperparameterTuning():
    # Initialise Variables
    results = []
    x = 0
    y = 0

    plot_dimension = get_plot_dimension()
    fig, axs = plt.subplots(
        plot_dimension, plot_dimension, sharex=True, sharey=True, figsize=(60, 20))  # ! Resize this if training with many different parameters

    # Loop over the different parameters
    for a in range(len(F)):
        for b in range(len(initialLearningRate)):
            for c in range(len(learningRateDecay)):
                for d in range(len(regularization)):
                    for e in range(len(momentum)):

                        print("----------Training with F {}, initLearningRate {}, learningRateDecay {}, regularization {}, momentum {}----------".format(
                            F[a], initialLearningRate[b], learningRateDecay[c], regularization[d], momentum[e]))

                        train_loss, val_loss = mainRBM.main(K=K,
                                                            epochs=epochs,
                                                            F=F[a],
                                                            initialLearningRate=initialLearningRate[b],
                                                            learningRateDecay=learningRateDecay[c],
                                                            regularization=regularization[d],
                                                            momentum=momentum[e])

                        # Append results in the form of dictionary
                        results.append({"Validation Loss": min(val_loss),
                                        "F": F[a],
                                        "Init Learn Rate": initialLearningRate[b],
                                        "Learn Rate Decay": learningRateDecay[c],
                                        "Regularization": regularization[d],
                                        "Momentum": momentum[e]
                                        })

                        # Plot the evolution of training and validation RMSE
                        axs[x, y].plot(train_loss)
                        axs[x, y].plot(val_loss)
                        axs[x, y].set(xlabel='epoch', ylabel='RMSE')
                        axs[x, y].set_title('F {}, LR {}, Decay {}, Reg {}, Mmt {}'.format(
                            F[a], initialLearningRate[b], learningRateDecay[c], regularization[d], momentum[e]))

                        # Update the index to plot plots
                        x, y = update_plot_location(x, y, plot_dimension)

    # Code to output predictions without overwriting by using current date time
    date, time = get_current_date_and_time()
    if not os.path.exists('predictions/{}/'.format(date)):
        os.makedirs('predictions/{}/'.format(date))

    # Output CSV
    print('--- Writing to CSV ---')
    df = pd.DataFrame(results)
    df = df.sort_values(by='Validation Loss', ascending=True)
    df.to_csv("predictions/{}/{}.csv".format(date, time))

    # Output Plot
    print("--- Plotting in progress ---")
    for ax in axs.flat:
        ax.label_outer()
    line_labels = ["Training Loss", "Validation Loss"]
    fig.legend(labels=line_labels)
    print("--- Saving plots... ---")
    plt.savefig("predictions/{}/{}.pdf".format(date, time))
    # print("--- Loading Plot... ---")
    plt.show()


# Run main hyperparameter tuning function
hyperparameterTuning()
