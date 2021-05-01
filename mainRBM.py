import numpy as np
import rbm
import projectLib as lib
import matplotlib.pyplot as plt
from datetime import datetime

training = lib.getTrainingData()
validation = lib.getValidationData()
# You could also try with the chapter 4 data
# training = lib.getChapter4Data()

trStats = lib.getUsefulStats(training)
vlStats = lib.getUsefulStats(validation)

K = 5

# SET PARAMETERS HERE!!!

# Number of hidden units
F = 15

# Number of epochs
epochs = 200

# We are using adaptive learning rate instead of a fixed gradientLearningRate
# Use this to select ideal learning rate at epoch 1
initialLearningRate = 3
learningRateDecay = 0.1

# Regularization 
regularization = 0.05

# Momentum
momentum = 0.4

# Mini-Batch
batchNumber = 35

def getCurrentDateAndTime():
    now = datetime.now()
    date = now.strftime("%d%m%Y")
    time = now.strftime("%H%M")
    return date, time

def main(K, F, epochs, initialLearningRate, learningRateDecay, regularization, momentum, batchNumber):
    best_vlRMSE = 2
    # Initialise all our arrays
    W = rbm.getInitialWeights(trStats["n_movies"], F, K)
    grad = np.zeros(W.shape)
    posprods = np.zeros(W.shape)
    negprods = np.zeros(W.shape)
    # imagine bias as additional hidden and visible unit
    # bias is a term to be added for each visible unit, for each hidden unit, and there are 5 ratings
    # Ref: Salakhutdinov et al. research paper
    hidden_bias = rbm.getInitialHiddenBias(F)  # b_j is bias of hidden feature j
    hidden_bias_grad = np.zeros(hidden_bias.shape)

    # b_ik is the bias of rating k for movie i
    visible_bias = rbm.getInitialVisibleBias(trStats["n_movies"], K)
    visible_bias_grad = np.zeros(visible_bias.shape)

    # create arrays to store our loss for each epoch
    # ! Made training loss inf for init
    trainLoss = [np.inf, np.inf]
    valLoss = [np.inf, np.inf]

    # store best weights 
    trainedWeights = W
    # store best biases
    trainedHiddenBias = hidden_bias
    trainedVisibleBias = visible_bias

    for epoch in range(1, epochs):
        visitingOrder = np.array(trStats["u_users"])
        np.random.shuffle(visitingOrder)
        batches = np.array_split(visitingOrder, batchNumber)

        for batch in batches:
            # keep track previous gradient for weights
            last_grad = grad
            grad = np.zeros(W.shape)

            # keep track of previous gradient for biases
            last_hidden_bias_grad = hidden_bias_grad
            last_visible_bias_grad = visible_bias_grad
            visible_bias_grad = np.zeros(visible_bias.shape)
            hidden_bias_grad = np.zeros(hidden_bias.shape)

            for user in batch:
                # get the ratings of that user
                ratingsForUser = lib.getRatingsForUser(user, training)

                # build the visible input
                v = rbm.getV(ratingsForUser)

                # get the weights associated to movies the user has seen
                weightsForUser = W[ratingsForUser[:, 0], :, :]
                # get the visible bias associated to movies the user has seen
                visibleBiasForUser = visible_bias[ratingsForUser[:, 0], :]

                ### LEARNING ###
                # propagate visible input to hidden units
                posHiddenProb = rbm.visibleToHiddenVecBias(v, weightsForUser, hidden_bias)
                # get positive gradient
                # note that we only update the movies that this user has seen!
                posprods[ratingsForUser[:, 0], :, :] = rbm.probProduct(v, posHiddenProb)

                ### UNLEARNING ###
                # sample from hidden distribution
                sampledHidden = rbm.sample(posHiddenProb)
                # propagate back to get "negative data"
                negData = rbm.hiddenToVisibleBias(sampledHidden, weightsForUser, visibleBiasForUser)
                # propagate negative data to hidden units
                negHiddenProb = rbm.visibleToHiddenVecBias(negData, weightsForUser, hidden_bias)
                # get negative gradient
                # note that we only update the movies that this user has seen!
                negprods[ratingsForUser[:, 0], :, :] = rbm.probProduct(negData, negHiddenProb)

                # we average over the number of users in the batch
                # implement L2 regularization; reference: https://sudonull.com/post/128613-Regularization-in-a-restricted-Boltzmann-machine-experiment
                grad[ratingsForUser[:, 0], :, :] += (rbm.getAdaptiveLearningRate(lr0=initialLearningRate, epoch=epoch, k=learningRateDecay) * \
                    (posprods[ratingsForUser[:, 0], :, :] -
                    negprods[ratingsForUser[:, 0], :, :] -
                    regularization * W[ratingsForUser[:, 0], :, :]))/len(batch)

                # calculate the gradient wrt biases
                # refer to update rule for biases: https://stats.stackexchange.com/questions/139138/updating-bias-with-rbms-restricted-boltzmann-machines
                # gradient for hidden bias
                hidden_bias_grad += (rbm.getAdaptiveLearningRate(lr0=initialLearningRate, epoch=epoch, k=learningRateDecay) * \
                    (posHiddenProb -
                    negHiddenProb -
                    regularization * hidden_bias))/len(batch)

                # gradient for visible bias
                visible_bias_grad[ratingsForUser[:, 0], :] += (rbm.getAdaptiveLearningRate(lr0=initialLearningRate, epoch=epoch, k=learningRateDecay) * \
                    (v -
                    negData -
                    regularization * visible_bias[ratingsForUser[:, 0], :]))/len(batch)

            # give some inertia to the gradient updates, limiting the risk that your gradient starts oscillating
            W[ratingsForUser[:, 0], :, :] += (1-momentum) * grad[ratingsForUser[:, 0], :, :] + momentum * last_grad[ratingsForUser[:, 0], :, :]

            hidden_bias += (1-momentum) * hidden_bias_grad + \
                    momentum * last_hidden_bias_grad

            visible_bias[ratingsForUser[:, 0], :] += (1-momentum) * visible_bias_grad[ratingsForUser[:, 0], :] + \
                    momentum * last_visible_bias_grad[ratingsForUser[:, 0], :]

        # Print the current RMSE for training and validation sets
        # this allows you to control for overfitting e.g
        # We predict over the training set
        trRHat = rbm.predictWithBias(trStats["movies"], trStats["users"], W, hidden_bias, visible_bias, training)
        trRMSE = lib.rmse(trStats["ratings"], trRHat)
        trainLoss.append(trRMSE)

        # We predict over the validation set
        valRHat = rbm.predictWithBias(vlStats["movies"], vlStats["users"], W, hidden_bias, visible_bias, training)
        vlRMSE = lib.rmse(vlStats["ratings"], valRHat)
        valLoss.append(vlRMSE)

        # If val loss is lower than what we have seen so far, update the weights and biases
        # Early stopping implementation
        if valLoss[-1] == min(valLoss):
            trainedWeights = W
            trainedHiddenBias = hidden_bias
            trainedVisibleBias = visible_bias

        print("--- EPOCH %d" % epoch)
        print("Training loss = %f" % trRMSE)
        print("Validation loss = %f" % vlRMSE)
        print("Learning Rate = %f" % rbm.getAdaptiveLearningRate(lr0=initialLearningRate, epoch=epoch, k=learningRateDecay))
        if vlRMSE < best_vlRMSE:
            best_vlRMSE = vlRMSE
        print("best_vlRMSE= %f" % best_vlRMSE)
        print("")

    return trainLoss, valLoss, trainedWeights, trainedHiddenBias, trainedVisibleBias, best_vlRMSE


# Only runs when mainRBM is called, not when imported
if __name__ == "__main__":
    startTime = datetime.now().replace(microsecond=0)

    # * Function to train model
    trainLoss, valLoss, trainedWeights, trainedHiddenBias, trainedVisibleBias, best_vlRMSE = main(K, F, epochs, initialLearningRate, learningRateDecay, regularization, momentum, batchNumber)
        
    if best_vlRMSE < 1.07:
        print("---Getting Predicting ratings...")
        predictedRatings = np.array([rbm.predictForUserWithBias(user, trainedWeights, trainedHiddenBias, trainedVisibleBias, training) for user in trStats["u_users"]])

        date, time = getCurrentDateAndTime()
        print("--- Saving predictions")
        np.savetxt("predictions/predictedRatings.txt", predictedRatings)

        endTime = datetime.now().replace(microsecond=0)
        print("--- Finished running code")
        print("--- Time Taken")
        print("--- {}".format(endTime-startTime))
    else:
        print("Model not good enough")
