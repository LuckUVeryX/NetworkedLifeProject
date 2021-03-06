import rbm
import mainRBM
import projectLib as lib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from datetime import datetime

training = lib.getTrainingData()
validation = lib.getValidationData()

trStats = lib.getUsefulStats(training)
vlStats = lib.getUsefulStats(validation)

# Parameters
# Ratings from 1-5
K = 5

epochs = 200

# Hyper parameter tuning
F = [15, 50]
initialLearningRate = [0.3, 0.1]
learningRateDecay = [0.3, 0.1]
regularization = [0.05, 0.1]
momentum = [0.9, 0.99]
batchNumber = [20, 25, 30, 35, 40, 45, 50]


def getPlotDimension():
    num_plots = len(F) * len(initialLearningRate) * len(learningRateDecay) * len(regularization) * len(momentum) * len(batchNumber)
    return int(np.ceil(np.sqrt(num_plots)))


def updatePlotLocation(x, y, dimension):
    if x < dimension - 1:
        x += 1
    else:
        y += 1
        x = 0
    return x, y


def hyperparameterTuning():
    print("--- Commencing Hyper Parameter Tuning")
    startTime = datetime.now().replace(microsecond=0)
    # Initialise Variables
    results = []
    bestValLoss = np.inf

    x = 0
    y = 0

    plotDimension = getPlotDimension()
    # ! Resize this if training with many different parameters
    # ? Might want to test with epochs = 1 to see if pdf will fit the plots
    fig, axs = plt.subplots(plotDimension, plotDimension, sharex=True, sharey=True, figsize=(40, 20))

    # Loop over the different parameters
    for a in range(len(F)):
        for b in range(len(initialLearningRate)):
            for c in range(len(learningRateDecay)):
                for d in range(len(regularization)):
                    for e in range(len(momentum)):
                        for f in range(len(batchNumber)):
                            print("--- Training with F {}, initLearningRate {}, decay {}, regularization {}, momentum {}, batchNumber {}".format(
                                F[a], initialLearningRate[b], learningRateDecay[c], regularization[d], momentum[e], batchNumber[f]))

                        trainLoss, valLoss, trainedWeights, trainedHiddenBias, trainedVisibleBias, best_vlRMSE = mainRBM.main(K=K,
                                                                                                                              F=F[a],
                                                                                                                              epochs=epochs,
                                                                                                                              initialLearningRate=initialLearningRate[
                                                                                                                              b],
                                                                                                                              learningRateDecay=learningRateDecay[
                                                                                                                              c],
                                                                                                                              regularization=regularization[
                                                                                                                              d],
                                                                                                                              momentum=momentum[
                                                                                                                              e],
                                                                                                                              batchNumber=batchNumber[
                                                                                                                              f])

                        # Append results in the form of dictionary
                        results.append({"Validation Loss": min(valLoss),
                                        "F": F[a],
                                        "Init Learn Rate": initialLearningRate[b],
                                        "Decay": learningRateDecay[c],
                                        "Regularization": regularization[d],
                                        "Momentum": momentum[e],
                                        "batchNumber": batchNumber[f]
                                        })

                        # Save the weights and biases of the best model
                        if min(valLoss) < bestValLoss:
                            bestValLoss = min(valLoss)
                            bestWeights = trainedWeights
                            bestHiddenBias = trainedHiddenBias
                            bestVisibleBias = trainedVisibleBias

                        # Plot the evolution of training and validation RMSE
                        axs[x, y].plot(trainLoss)
                        axs[x, y].plot(valLoss)
                        axs[x, y].set(xlabel='epoch', ylabel='RMSE')
                        axs[x, y].set_title('F {}, LR {}, Decay {}, Reg {}, Mmt {}, Batch {}'.format(
                            F[a], initialLearningRate[b], learningRateDecay[c], regularization[d], momentum[e], batchNumber[f]))

                        # Update the index to plot plots
                        x, y = updatePlotLocation(x, y, plotDimension)

    # Code to output predictions without overwriting by using current date time
    date, time = mainRBM.getCurrentDateAndTime()
    if not os.path.exists('predictions/{}/'.format(date)):
        os.makedirs('predictions/{}/'.format(date))

    # Output CSV
    print('--- Writing to CSV')
    df = pd.DataFrame(results)
    df = df.sort_values(by='Validation Loss', ascending=True)
    df.to_csv("predictions/{}/{}.csv".format(date, time))

    # Output best ratings
    print("--- Predicting ratings...")
    bestPredictedRatings = np.array(
        [rbm.predictForUserWithBias(user, bestWeights, bestHiddenBias, bestVisibleBias, training) for user in trStats["u_users"]])

    print("--- Saving predictions")
    np.savetxt("predictions/{}/{}_bestPredictedRatings.txt".format(date, time), bestPredictedRatings)

    # Output Plot
    print("--- Plotting in progress")
    for ax in axs.flat:
        ax.label_outer()
    lineLabels = ["Training Loss", "Validation Loss"]
    fig.legend(labels=lineLabels)

    print("--- Saving plots...")
    plt.savefig("predictions/{}/{}.pdf".format(date, time))

    endTime = datetime.now().replace(microsecond=0)
    print("--- Finished hyperparameter tuning ---")
    print("--- Time Taken ---")
    print("--- {} ---".format(endTime-startTime))

    plt.show()


# Run main hyperparameter tuning function
hyperparameterTuning()
