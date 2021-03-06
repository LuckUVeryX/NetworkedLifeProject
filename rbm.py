import numpy as np
import projectLib as lib

# set highest rating
K = 5


def softmax(x):
    # Numerically stable softmax function
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def ratingsPerMovie(training):
    movies = [x[0] for x in training]
    u_movies = np.unique(movies).tolist()
    return np.array([[i, movie, len([x for x in training if x[0] == movie])] for i, movie in enumerate(u_movies)])


def getV(ratingsForUser):
    # ratingsForUser is obtained from the ratings for user library
    # you should return a binary matrix ret of size m x K, where m is the number of movies
    #   that the user has seen. ret[i][k] = 1 if the user
    #   has rated movie ratingsForUser[i, 0] with k stars
    #   otherwise it is 0
    ret = np.zeros((len(ratingsForUser), K))
    for i in range(len(ratingsForUser)):
        ret[i, ratingsForUser[i, 1]-1] = 1.0
    return ret


def getInitialWeights(m, F, K):
    # m is the number of visible units
    # F is the number of hidden units
    # K is the highest rating (fixed to 5 here)
    return np.random.normal(0, 0.1, (m, F, K))


def getInitialHiddenBias(F):
    # F is the number of hidden units
    return np.random.normal(0, 0.1, (F))


def getInitialVisibleBias(m, K):
    # m is the number of visible units
    # K is the highest rating (fixed to 5 here)
    return np.random.normal(0, 0.1, (m, K))


def sig(x):
    ### TO IMPLEMENT ###
    # x is a real vector of size n
    # ret should be a vector of size n where ret_i = sigmoid(x_i)
    return 1 / (1 + np.exp(-x))


def visibleToHiddenVec(v, w):
    ### TO IMPLEMENT ###
    # v is a matrix of size m x 5. Each row is a binary vector representing a rating
    #    OR a probability distribution over the rating
    # w is a list of matrices of size m x F x 5
    # ret should be a vector of size F
    m, F, K = w.shape
    h = np.zeros(F)
    for h_j in range(F):
        score = np.sum(v*w[:, h_j, :])
        prob = sig(score)
        h[h_j] = prob
    return h


def visibleToHiddenVecBias(v, w, b):
    ### TO IMPLEMENT ###
    # v is a matrix of size m x 5. Each row is a binary vector representing a rating
    #    OR a probability distribution over the rating
    # w is a list of matrices of size m x F x 5
    # b is the hidden bias of size F
    # ret should be a vector of size F
    m, F, K = w.shape
    h = np.zeros(F)
    for h_j in range(F):
        score = np.sum(v*w[:, h_j, :]) + b[h_j]
        prob = sig(score)
        h[h_j] = prob
    return h


def hiddenToVisible(h, w):
    ### TO IMPLEMENT ###
    # h is a binary vector of size F
    # w is an array of size m x F x 5
    # ret should be a matrix of size m x 5, where m
    #   is the number of movies the user has seen.
    #   Remember that we do not reconstruct movies that the user
    #   has not rated! (where reconstructing means getting a distribution
    #   over possible ratings).
    #   We only do so when we predict the rating a user would have given to a movie.
    m, F, K = w.shape
    v = np.zeros((m, 5))
    for movie in range(m):
        score = np.matmul(h, w[movie, :, :])  # 1 x F * F x 5
        prob = softmax(score)
        v[movie, ] = prob
    return v


def hiddenToVisibleBias(h, w, b):
    ### TO IMPLEMENT ###
    # h is a binary vector of size F
    # w is an array of size m x F x 5
    # b is visible bias of size m x 5
    # ret should be a matrix of size m x 5, where m
    #   is the number of movies the user has seen.
    #   Remember that we do not reconstruct movies that the user
    #   has not rated! (where reconstructing means getting a distribution
    #   over possible ratings).
    #   We only do so when we predict the rating a user would have given to a movie.
    m, F, K = w.shape
    v = np.zeros((m, 5))
    for movie in range(m):
        # (1 x F * F x 5) + (1 X 5) 
        score = np.matmul(h, w[movie, :, :]) + b[movie,:]
        prob = softmax(score)
        v[movie, ] = prob
    return v


def probProduct(v, p):
    # v is a matrix of size m x 5
    # p is a vector of size F, activation of the hidden units
    # returns the gradient for visible input v and hidden activations p
    ret = np.zeros((v.shape[0], p.size, v.shape[1]))
    for i in range(v.shape[0]):
        for j in range(p.size):
            for k in range(v.shape[1]):
                ret[i, j, k] = v[i, k] * p[j]
    return ret


def sample(p):
    # p is a vector of real numbers between 0 and 1
    # ret is a vector of same size as p, where ret_i = Ber(p_i)
    # In other word we sample from a Bernouilli distribution with
    # parameter p_i to obtain ret_i
    samples = np.random.random(p.size)
    return np.array(samples <= p, dtype=int)


def getPredictedDistribution(v, w, wq):
    ### TO IMPLEMENT ###
    # This function returns a distribution over the ratings for movie q, if user data is v
    # v is the dataset of the user we are predicting the movie for
    #   It is a m x 5 matrix, where m is the number of movies in the
    #   dataset of this user.
    # w is the weights array for the current user, of size m x F x 5
    # wq is the weight matrix of size F x 5 for movie q
    #   If W is the whole weights array, then wq = W[q, :, :]
    # You will need to perform the same steps done in the learning/unlearning:
    #   - Propagate the user input to the hidden units
    #   - Sample the state of the hidden units
    #   - Backpropagate these hidden states to obtain
    #       the distribution over the movie whose associated weights are wq
    # ret is a vector of size 5
    posHiddenProb = visibleToHiddenVec(v, w)
    sampledHidden = sample(posHiddenProb)
    # same logic as a single for loop in the hiddenToVisible function
    v = np.zeros((1, 5))
    score = np.matmul(sampledHidden, wq)
    prob = softmax(score)
    v = prob
    return v


def getPredictedDistributionWithBias(v, w, wq, hidden_bias, visible_bias, vbq):
    ### TO IMPLEMENT ###
    # This function returns a distribution over the ratings for movie q, if user data is v
    # v is the dataset of the user we are predicting the movie for
    #   It is a m x 5 matrix, where m is the number of movies in the
    #   dataset of this user.
    # w is the weights array for the current user, of size m x F x 5
    # wq is the weight matrix of size F x 5 for movie q
    #   If W is the whole weights array, then wq = W[q, :, :]
    # visible_bias is the bias array for the current user, of size m x 5
    # vbq is the bias matrix of size 1 x 5 for movie q
    #   If visible_bias is the whole weights array, then vbq = visible_bias[q, :]
    # You will need to perform the same steps done in the learning/unlearning:
    #   - Propagate the user input to the hidden units
    #   - Sample the state of the hidden units
    #   - Backpropagate these hidden states to obtain
    #       the distribution over the movie whose associated weights are wq
    # ret is a vector of size 5
    posHiddenProb = visibleToHiddenVecBias(v, w, hidden_bias)
    sampledHidden = sample(posHiddenProb)
    # same logic as a single for loop in the hiddenToVisibleBias function
    v = np.zeros((1, 5))
    score = np.matmul(sampledHidden, wq) + vbq
    prob = softmax(score)
    v = prob
    return v


def predictRatingMax(ratingDistribution):
    ### TO IMPLEMENT ###
    # ratingDistribution is a probability distribution over possible ratings
    #   It is obtained from the getPredictedDistribution function
    # This function is one of three you are to implement
    # that returns a rating from the distribution
    # We decide here that the predicted rating will be the one with the highest probability
    prediction = np.where(ratingDistribution ==
                          np.amax(ratingDistribution))[0][0] + 1
    return prediction


def predictRatingExp(ratingDistribution):
    ### TO IMPLEMENT ###
    # ratingDistribution is a probability distribution over possible ratings
    #   It is obtained from the getPredictedDistribution function
    # This function is one of three you are to implement
    # that returns a rating from the distribution
    # We decide here that the predicted rating will be the expectation over
    # the softmax applied to ratingDistribution
    ratings = np.array((1, 2, 3, 4, 5))
    prediction = np.dot(ratingDistribution, ratings)
    return prediction


def predictMovieForUser(q, user, W, training, predictType="exp"):
    # movie is movie idx
    # user is user ID
    # type can be "max" or "exp"
    ratingsForUser = lib.getRatingsForUser(user, training)
    v = getV(ratingsForUser)
    ratingDistribution = getPredictedDistribution(
        v, W[ratingsForUser[:, 0], :, :], W[q, :, :])
    if predictType == "max":
        return predictRatingMax(ratingDistribution)
    else:
        return predictRatingExp(ratingDistribution)


def predictMovieForUserWithBias(q, user, W, hidden_bias, visible_bias, training, predictType="exp"):
    # movie is movie idx
    # user is user ID
    # type can be "max" or "exp"
    ratingsForUser = lib.getRatingsForUser(user, training)
    v = getV(ratingsForUser)
    ratingDistribution = getPredictedDistributionWithBias(
        v, W[ratingsForUser[:, 0], :, :], W[q, :, :], hidden_bias, visible_bias[ratingsForUser[:, 0], :],visible_bias[q, :])
    if predictType == "max":
        return predictRatingMax(ratingDistribution)
    else:
        return predictRatingExp(ratingDistribution)


def predict(movies, users, W, training, predictType="exp"):
    # given a list of movies and users, predict the rating for each (movie, user) pair
    # used to compute RMSE
    return [predictMovieForUser(movie, user, W, training, predictType=predictType) for (movie, user) in zip(movies, users)]


def predictWithBias(movies, users, W, hidden_bias, visible_bias,training, predictType="exp"):
    # given a list of movies and users, predict the rating for each (movie, user) pair
    # used to compute RMSE
    return [predictMovieForUserWithBias(movie, user, W, hidden_bias,visible_bias, training, predictType=predictType) for (movie, user) in zip(movies, users)]


training = lib.getTrainingData()
trStats = lib.getUsefulStats(training)


def predictForUser(user, W, training, predictType="exp"):
    # TO IMPLEMENT
    # given a user ID, predicts all movie ratings for the user
    return [predictMovieForUser(movie, user, W, training, predictType=predictType) for movie in trStats["u_movies"]]


def predictForUserWithBias(user, W, hidden_bias, visible_bias, training, predictType="exp"):
    # TO IMPLEMENT
    # given a user ID, predicts all movie ratings for the user
    return [predictMovieForUserWithBias(movie, user, W, hidden_bias, visible_bias, training, predictType=predictType) for movie in trStats["u_movies"]]


def getAdaptiveLearningRate(lr0, epoch, k):
    # * Using Time-based decay
    # ? Perhaps explore different type of adaptive learning rates
    # https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1
    return lr0/(1+k*(epoch-1))
