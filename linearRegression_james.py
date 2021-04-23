import numpy as np
import projectLib as lib
import matplotlib.pyplot as plt

# shape is movie,user,rating
training = lib.getTrainingData()
validation = lib.getValidationData()

#some useful stats
trStats = lib.getUsefulStats(training)
vlStats = lib.getUsefulStats(validation)
rBar = np.mean(trStats["ratings"])

# we get the A matrix from the training dataset
def getA(training):
    A = np.zeros((trStats["n_ratings"], trStats["n_movies"] + trStats["n_users"]))
    for index in range(len(A)):
        A[index][training[index][0]] = 1
        A[index][trStats["n_movies"] + training[index][1]] = 1
    print(len(A[0]))
    return A

# we also get c
def getc(rBar, ratings):
    alist = []
    for index in range(len(ratings)):
        cij = ratings[index] - rBar
        alist.append(cij)
    c = np.array(alist)
    return c

# apply the functions
A = getA(training)
c = getc(rBar, trStats["ratings"])

# compute the estimator b
def param(A, c):
    return np.matmul(np.linalg.pinv(np.matmul(A.T, A)), np.matmul(A.T, c))


# compute the estimator b with a regularisation parameter l
# note: lambda is a Python keyword to define inline functions
#       so avoid using it as a variable name!
def param_reg(A, c, l):
    return np.matmul(np.linalg.pinv(np.matmul(A.T, A) + l*np.identity(A.shape[1])), np.matmul(A.T, c))

# from b predict the ratings for the (movies, users) pair
def predict(movies, users, rBar, b):
    n_predict = len(users)
    p = np.zeros(n_predict)
    for i in range(0, n_predict):
        rating = rBar + b[movies[i]] + b[trStats["n_movies"] + users[i]]
        if rating > 5: rating = 5.0
        if rating < 1: rating = 1.0
        p[i] = rating
    return p

# Unregularised version (<=> regularised version with l = 0)
#b = param(A, c)

# Regularised version
l_list = [x for x in range(0,101)]
val_rmse_list = []
trn_rmse_list = []
for l in l_list:
    b = param_reg(A, c, l)
    val_rmse = lib.rmse(predict(vlStats["movies"], vlStats["users"], rBar, b), vlStats["ratings"])
    trn_rmse = lib.rmse(predict(trStats["movies"], trStats["users"], rBar, b), trStats["ratings"])
    val_rmse_list.append(val_rmse)
    trn_rmse_list.append(trn_rmse)
    print("Linear regression, l = %f" % l)
    print("RMSE for training %f" % lib.rmse(predict(trStats["movies"], trStats["users"], rBar, b), trStats["ratings"]))
    print("RMSE for validation %f" % lib.rmse(predict(vlStats["movies"], vlStats["users"], rBar, b), vlStats["ratings"]))

plt.plot(l_list, trn_rmse_list, label = "Training Loss RMSE", marker='o')
plt.plot(l_list, val_rmse_list, label = "Validation Loss RMSE", marker='o')
plt.xlabel('Regularisation')
plt.ylabel('RMSE')
the_title = "Plot of the rmse loss against regularisation \n min trRMSE " + str(round(min(trn_rmse_list),3)) + " & min valRMSE " + str(round(min(val_rmse_list),3))
plt.title(the_title)
plt.legend()
plt.show()