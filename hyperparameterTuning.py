import mainRBM
import projectLib as lib
import matplotlib.pyplot as plt
import pandas as pd

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
epochs = 5

# * We are using adaptive learning rate instead of a fixed gradientLearningRate
# //gradientLearningRate = 0.1
# * Use this to select ideal learning rate at epoch 1
initialLearningRate = [1, 0.1]
#  TODO Hyper parameter tuning
# ? Range from 1 to 5
learningRateDecay = [0.1, 0.01]

# * Set the regularization strength here
# TODO Hyper parameter tuning
# ? Range from 0 to 0.05
regularization = 0.01

# * Momemntum
# TODO Hyper parameter tuning
# ? 0 to 1
momentum = 0.01


def hyperparameterTuning():
    results = []
    for learningRate in initialLearningRate:
        for decay in learningRateDecay:
            print("----------Training with decay rate {}----------".format(decay))
            val_loss = mainRBM.main(K=K, F=F, epochs=epochs, initialLearningRate=learningRate,
                                    learningRateDecay=decay, regularization=regularization, momentum=momentum)
            results.append({"Validation Loss": val_loss,
                            "Initial Learning Rate": initialLearningRate, "Learning Rate Decay": decay})
    plt.show()
    return results


results = hyperparameterTuning()
df = pd.DataFrame(results)
df.sort_values(by='Validation Loss', ascending=True)
df.to_csv("predictions/results.csv")
