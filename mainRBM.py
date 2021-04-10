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
# number of hidden units
# TODO Hyper parameter tuning F, (number of hidden units)
F = 15
epochs = 200

# * We are using adaptive learning rate instead of a fixed gradientLearningRate
# //gradientLearningRate = 0.1
# * Use this to select ideal learning rate at epoch 1
initialLearningRate = 3.0
#  TODO Hyper parameter tuning
# ? Range from 1 to 5
learningRateDecay = 0.1

# * Set the regularization strength here
# TODO Hyper parameter tuning
# ? Range from 0 to 0.05
regularization = 0.05

# * Momemntum
# TODO Hyper parameter tuning
# ? 0 to 1
momentum = 0.3


def get_current_date():
    now = datetime.now()
    date = now.strftime("%d%m%Y")
    return date


def main(K, F, epochs, initialLearningRate, learningRateDecay, regularization, momentum, makePredictions):
    # Initialise all our arrays
    W = rbm.getInitialWeights(trStats["n_movies"], F, K)
    grad = np.zeros(W.shape)
    posprods = np.zeros(W.shape)
    negprods = np.zeros(W.shape)
    # imagine bias as additional hidden and visible unit
    # bias is a term to be added for each visible unit, for each hidden unit, and there are 5 ratings
    # Ref: Salakhutdinov et al. research paper
    hidden_bias = rbm.getInitialHiddenBias(
        F)  # b_j is bias of hidden feature j
    hidden_bias_grad = np.zeros(hidden_bias.shape)

    # b_ik is the bias of rating k for movie i
    visible_bias = rbm.getInitialVisibleBias(trStats["n_movies"], K)
    visible_bias_grad = np.zeros(visible_bias.shape)

    # create arrays to store our loss for each epoch
    # ! Made training loss inf for init
    train_loss = [np.inf, np.inf]
    val_loss = [np.inf, np.inf]

    # store best weights
    bestWeights = W
    # store best biases
    best_hidden_bias = hidden_bias
    best_visible_bias = visible_bias

    for epoch in range(1, epochs):
        # # ! Keep training until training lost stops decreasing
        # epoch = 0
        # while True:
        #     epoch += 1
        #     if abs(train_loss[-1]-train_loss[-2]) < 0.0001 and epoch > 3:
        #         break
        # in each epoch, we'll visit all users in a random order
        visitingOrder = np.array(trStats["u_users"])
        np.random.shuffle(visitingOrder)

        # keep track previous gradient for weights
        last_grad = grad
        # keep track of previous gradient for biases
        last_hidden_bias_grad = hidden_bias_grad
        last_visible_bias_grad = visible_bias_grad

        for user in visitingOrder:
            # get the ratings of that user
            ratingsForUser = lib.getRatingsForUser(user, training)

            # build the visible input
            v = rbm.getV(ratingsForUser)

            # get the weights associated to movies the user has seen
            weightsForUser = W[ratingsForUser[:, 0], :, :]
            # get the visible bias associated to movies the user has seen
            visible_biasForUser = visible_bias[ratingsForUser[:, 0], :]

            ### LEARNING ###
            # propagate visible input to hidden units
            posHiddenProb = rbm.visibleToHiddenVecBias(
                v, weightsForUser, hidden_bias)
            # get positive gradient
            # note that we only update the movies that this user has seen!
            posprods[ratingsForUser[:, 0], :,
                     :] = rbm.probProduct(v, posHiddenProb)

            ### UNLEARNING ###
            # sample from hidden distribution
            sampledHidden = rbm.sample(posHiddenProb)
            # propagate back to get "negative data"
            negData = rbm.hiddenToVisibleBias(
                sampledHidden, weightsForUser, visible_biasForUser)
            # propagate negative data to hidden units
            negHiddenProb = rbm.visibleToHiddenVecBias(
                negData, weightsForUser, hidden_bias)
            # get negative gradient
            # note that we only update the movies that this user has seen!
            negprods[ratingsForUser[:, 0], :, :] = rbm.probProduct(
                negData, negHiddenProb)

            # we average over the number of users in the batch (if we use mini-batch)
            # implement L2 regularization; reference: https://sudonull.com/post/128613-Regularization-in-a-restricted-Boltzmann-machine-experiment
            grad[ratingsForUser[:, 0], :, :] = rbm.getAdaptiveLearningRate(lr0=initialLearningRate, epoch=epoch, k=learningRateDecay) * \
                (posprods[ratingsForUser[:, 0], :, :] -
                 negprods[ratingsForUser[:, 0], :, :] -
                 regularization * W[ratingsForUser[:, 0], :, :])

            # give some inertia to the gradient updates, limiting the risk that your gradient starts oscillating
            W[ratingsForUser[:, 0], :, :] += (1-momentum) * grad[ratingsForUser[:, 0], :, :] + \
                momentum * last_grad[ratingsForUser[:, 0], :, :]

            # calculate the gradient wrt biases
            # refer to update rule for biases: https://stats.stackexchange.com/questions/139138/updating-bias-with-rbms-restricted-boltzmann-machines
            # gradient for hidden bias
            hidden_bias_grad = rbm.getAdaptiveLearningRate(lr0=initialLearningRate, epoch=epoch, k=learningRateDecay) * \
                (posHiddenProb -
                 negHiddenProb -
                 regularization * hidden_bias)
            # give some inertia to gradient updates
            hidden_bias += (1-momentum) * hidden_bias_grad + \
                momentum * last_hidden_bias_grad

            # gradient for visible bias
            visible_bias_grad[ratingsForUser[:, 0], :] = rbm.getAdaptiveLearningRate(lr0=initialLearningRate, epoch=epoch, k=learningRateDecay) * \
                (v -
                 negData -
                 regularization * visible_bias[ratingsForUser[:, 0], :])
            # give some inertia to gradient updates
            visible_bias[ratingsForUser[:, 0], :] += (1-momentum) * visible_bias_grad[ratingsForUser[:, 0], :] + \
                momentum * last_visible_bias_grad[ratingsForUser[:, 0], :]

        # Print the current RMSE for training and validation sets
        # this allows you to control for overfitting e.g
        # We predict over the training set
        tr_r_hat = rbm.predictWithBias(
            trStats["movies"], trStats["users"], W, hidden_bias, visible_bias, training)
        trRMSE = lib.rmse(trStats["ratings"], tr_r_hat)
        train_loss.append(trRMSE)

        # We predict over the validation set
        vl_r_hat = rbm.predictWithBias(
            vlStats["movies"], vlStats["users"], W, hidden_bias, visible_bias, training)
        vlRMSE = lib.rmse(vlStats["ratings"], vl_r_hat)
        val_loss.append(vlRMSE)

        # If Val loss is lower than what we have seen so far, update the best weights
        if val_loss[-1] <= min(val_loss):
            bestWeights = W
            best_hidden_bias = hidden_bias
            best_visible_bias = visible_bias

        print("### EPOCH %d ###" % epoch)
        print("Training loss = %f" % trRMSE)
        print("Validation loss = %f" % vlRMSE)

        # ! Print statement to track learning rate. Comment out for submission
        print("Learning Rate = %f" % rbm.getAdaptiveLearningRate(
            lr0=initialLearningRate, epoch=epoch, k=learningRateDecay))
        print("")

    ### END ###
    if makePredictions:
        predictedRatings = np.array(
            [rbm.predictForUserWithBias(user, bestWeights, best_hidden_bias, best_visible_bias, training) for user in trStats["u_users"]])
        np.savetxt("predictions/predictedRatings_{}.txt".format(get_current_date()),
                   predictedRatings)

    return train_loss, val_loss


# * Function to train model
main(K, F, epochs, initialLearningRate,
     learningRateDecay, regularization, momentum, True)
