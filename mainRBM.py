import numpy as np
import rbm
import projectLib as lib
import matplotlib.pyplot as plt

training = lib.getTrainingData()
validation = lib.getValidationData()
# You could also try with the chapter 4 data
# training = lib.getChapter4Data()

trStats = lib.getUsefulStats(training)
vlStats = lib.getUsefulStats(validation)

# look at the data statistics
print("train stats users movies ratings: ",
      trStats['n_users'], trStats['n_movies'], trStats['n_ratings'])
print("val stats: ", vlStats['n_users'],
      vlStats['n_movies'], vlStats['n_ratings'])

K = 5

# SET PARAMETERS HERE!!!
# number of hidden units
# TODO Hyper parameter tuning F, (number of hidden units)
F = 30
epochs = 30

# * We are using adaptive learning rate instead of a fixed gradientLearningRate
# //gradientLearningRate = 0.1
# * Use this to select ideal learning rate at epoch 1
initialLearningRate = 2

# * Set the regularization strength here
regularization = 0.05

# * Momemntum
momentum = 3

# Initialise all our arrays
W = rbm.getInitialWeights(trStats["n_movies"], F, K)
grad = np.zeros(W.shape)
posprods = np.zeros(W.shape)
negprods = np.zeros(W.shape)

# create arrays to store our loss for each epoch
train_loss = []
val_loss = []

# store best weights
bestWeights = W

for epoch in range(1, epochs):
    # in each epoch, we'll visit all users in a random order
    visitingOrder = np.array(trStats["u_users"])
    np.random.shuffle(visitingOrder)

    # keep track previous gradient
    last_grad = grad

    for user in visitingOrder:
        # get the ratings of that user
        ratingsForUser = lib.getRatingsForUser(user, training)

        # build the visible input
        v = rbm.getV(ratingsForUser)

        # get the weights associated to movies the user has seen
        weightsForUser = W[ratingsForUser[:, 0], :, :]

        ### LEARNING ###
        # propagate visible input to hidden units
        posHiddenProb = rbm.visibleToHiddenVec(v, weightsForUser)
        # get positive gradient
        # note that we only update the movies that this user has seen!
        posprods[ratingsForUser[:, 0], :,
                 :] = rbm.probProduct(v, posHiddenProb)

        ### UNLEARNING ###
        # sample from hidden distribution
        sampledHidden = rbm.sample(posHiddenProb)
        # propagate back to get "negative data"
        negData = rbm.hiddenToVisible(sampledHidden, weightsForUser)
        # propagate negative data to hidden units
        negHiddenProb = rbm.visibleToHiddenVec(negData, weightsForUser)
        # get negative gradient
        # note that we only update the movies that this user has seen!
        negprods[ratingsForUser[:, 0], :, :] = rbm.probProduct(
            negData, negHiddenProb)

        # we average over the number of users in the batch (if we use mini-batch)
        # implement L2 regularization; reference: https://sudonull.com/post/128613-Regularization-in-a-restricted-Boltzmann-machine-experiment
        grad[ratingsForUser[:, 0], :, :] = rbm.getAdaptiveLearningRate(lr0=initialLearningRate, epoch=epoch) * \
            (posprods[ratingsForUser[:, 0], :, :] -
             negprods[ratingsForUser[:, 0], :, :] -
             regularization * W[ratingsForUser[:, 0], :, :])

        # give some inertia to the gradient updates, limiting the risk that your gradient starts oscillating
        W[ratingsForUser[:, 0], :, :] += grad[ratingsForUser[:, 0], :, :] + \
            momentum * last_grad[ratingsForUser[:, 0], :, :]

    # Print the current RMSE for training and validation sets
    # this allows you to control for overfitting e.g
    # We predict over the training set
    tr_r_hat = rbm.predict(trStats["movies"], trStats["users"], W, training)
    trRMSE = lib.rmse(trStats["ratings"], tr_r_hat)
    train_loss.append(trRMSE)

    # We predict over the validation set
    vl_r_hat = rbm.predict(vlStats["movies"], vlStats["users"], W, training)
    vlRMSE = lib.rmse(vlStats["ratings"], vl_r_hat)
    val_loss.append(vlRMSE)

    # If Val loss is lower than what we have seen so far, update the best weights
    if val_loss[-1] <= min(val_loss):
        bestWeights = W

    print("### EPOCH %d ###" % epoch)
    print("Training loss = %f" % trRMSE)
    print("Validation loss = %f" % vlRMSE)

    # ! Print statement to track learning rate. Comment out for submission
    print("Learning Rate = %f" % rbm.getAdaptiveLearningRate(
        lr0=initialLearningRate, epoch=epoch))

# plot the evolution of training and validation RMSE
plt.figure(figsize=(8, 8))
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('RMSE')
plt.ylim([0, 2.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

### END ###
# This part you can write on your own
# you could plot the evolution of the training and validation RMSEs for example

predictedRatings = np.array(
    [rbm.predictForUser(user, bestWeights, training) for user in trStats["u_users"]])
np.savetxt("predictions/predictedRatings.txt", predictedRatings)
