import numpy as np
import projectLib as lib

# shape is movie,user,rating
training = lib.getTrainingData()
validation = lib.getValidationData()

# some useful stats
trStats = lib.getUsefulStats(training)
vlStats = lib.getUsefulStats(validation)
rBar = np.mean(trStats["ratings"])

# we get the A matrix from the training dataset


def getA(training):
    A = np.zeros(
        (trStats["n_ratings"], trStats["n_movies"] + trStats["n_users"]))
    for i in range(trStats["n_ratings"]):
        A[i][training[i][0]] = 1  # movie ID
        A[i][training[i][1]] = 1  # user ID
    return A

# we also get c


def getc(rBar, ratings):
    return ratings - rBar


# apply the functions
A = getA(training)
c = getc(rBar, trStats["ratings"])

# compute the estimator b


def param(A, c):
    # (A_transpose A)^-1 A_transpose c
    inverse = np.linalg.pinv(np.matmul(A.transpose(), A))
    b = np.matmul(inverse, np.matmul(A.transpose(), c))
    return b

# compute the estimator b with a regularisation parameter l
# note: lambda is a Python keyword to define inline functions
#       so avoid using it as a variable name!


def param_reg(A, c, l):
    # (A_transpose A + lambda I)^-1 A_transpose c
    A_transpose_A = np.matmul(A.transpose(), A)
    identity = np.identity(A_transpose_A.shape[0])
    inverse = np.linalg.pinv(A_transpose_A + l*identity)
    b = np.matmul(inverse, np.matmul(A.transpose(), c))
    return b

# from b predict the ratings for the (movies, users) pair


def predict(movies, users, rBar, b):
    n_predict = len(users)
    p = np.zeros(n_predict)
    for i in range(0, n_predict):
        rating = rBar + b[movies[i]] + b[trStats["n_movies"] + users[i]]
        if rating > 5:
            rating = 5.0
        if rating < 1:
            rating = 1.0
        p[i] = rating
    return p


# Unregularised version (<=> regularised version with l = 0)
# b = param(A, c)


# Regularised version
l = 1
b = param_reg(A, c, l)

print("Linear regression, l = %f" % l)
print("RMSE for training %f" % lib.rmse(
    predict(trStats["movies"], trStats["users"], rBar, b), trStats["ratings"]))
print("RMSE for validation %f" % lib.rmse(
    predict(vlStats["movies"], vlStats["users"], rBar, b), vlStats["ratings"]))
