import numpy as np
import projectLib as lib
from datetime import datetime
import os

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
    print (A)
    # have every linear combination between movie ID and user ID based on the n_rating
    for i in range(trStats["n_ratings"]):
        A[i][training[i][0]] = 1  # movie ID
        # since the 1st 97 index refers to trStats["n_movies"]
        # hence add 97(n_movies) + n_rating for every possible combination
        A[i][trStats["n_movies"] + training[i][1]] = 1  # user ID
    return A

# we also get c


def getc(rBar, ratings):
    return ratings - rBar


# apply the functions
A = getA(training)
c = getc(rBar, trStats["ratings"])
l = 1
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
    try:
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
    except: 
        p = 'error'
        return (p)


# Unregularised version (<=> regularised version with l = 0)
# b = param(A, c)


# Regularised version
def linearmodel(A,c,l):
    b = param_reg(A, c, l)
    print("Linear regression, l = %f" % l)
    predicted_training = predict(trStats["movies"], trStats["users"], rBar, b)
    actual_training = trStats["ratings"]
    train_loss = lib.rmse(predicted_training, actual_training)
    print("RMSE for training %f" % train_loss)

    predicted_val = predict(vlStats["movies"], vlStats["users"], rBar, b)
    actual_val = vlStats["ratings"]
    val_loss = lib.rmse(predicted_val ,actual_val)
    print("RMSE for validation %f" % val_loss)
    return(b,l,train_loss, val_loss,)

def get_current_date_and_time():
    now = datetime.now()
    date = now.strftime("%d%m%Y")
    time = now.strftime("%H%M")
    return date, time

def createfolder(foldername):
    foldername = str(foldername)
    if not os.path.exists(foldername):
        os.makedirs(foldername)

# Only runs when linearRegression is called, not when imported
if __name__ == "__main__":
    start_time = datetime.now().replace(microsecond=0)


    # * Function to train model
    b,l,train_loss, val_loss = linearmodel(A,c,l)
    print("--- Predicting ratings...")
    predicted_ratings = np.array(
        [predict(trStats["movies"], trStats["u_users"] , rBar, b)])

    date, time = get_current_date_and_time()
    print("--- Saving predictions")
    # check and create folder
    createfolder('predictions')
    createfolder('predictions/' + str(date))
    np.savetxt("predictions/{}/{}_predictedRatings.txt".format(date, time),
               predicted_ratings,fmt='%s')

    end_time = datetime.now().replace(microsecond=0)
    print("--- Finished training model")
    print("--- Time Taken")
    print("--- {}".format(end_time-start_time))