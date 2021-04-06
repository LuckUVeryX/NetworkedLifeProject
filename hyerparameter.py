import numpy as np
import rbm
import projectLib as lib
import matplotlib.pyplot as plt

#addtional functions
import pandas as pd
import datetime
import os.path
import time

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
# need to change to what was suggested from Ryan and
# fixed parameters
epochs = 100

# parameters that need tunning
# function to create list of float or integers
def parameters_list(_start, _end, _incre):
    para_list = []

    # caculate the number of elements in the list
    num_of_element = _end + _incre - _start
    num_of_element = int(num_of_element/ _incre)

    # increment the number of elements
    for i in range(1,num_of_element+1):
        # append the increment to the 
        para_list.append(_start)
        _start = round((_start + _incre),7)

    return(para_list)

# number of hidden units
# TODO Hyper parameter tuning F, (number of hidden units)
# F in range of 8 to 50, increment of 1
F_list = parameters_list(8,50,1)

# * We are using adaptive learning rate instead of a fixed gradientLearningRate
# //gradientLearningRate = 0.1

# * Use this to select ideal learning rate at epoch 1
initialLearningRate_list = parameters_list(0,10,0.1)
#  TODO Hyper parameter tuning
# ? Range from 1 to 5
learningRateDecay_list = parameters_list(1,5,0.1)

# * Set the regularization strength here
# TODO Hyper parameter tuning
# ? Range from 0 to 0.05
regularization_list = parameters_list(0,0.05,0.01)

# * Momemntum
# TODO Hyper parameter tuning
# ? 0 to 1
momentum_list = parameters_list(0.1,1,0.1)



# RBM codes starts here
def rbm_model(_F,_initialLearningRate, _learningRateDecay, _regularization, _momentum):

    # Initialise all our arrays
    W = rbm.getInitialWeights(trStats["n_movies"], _F, K)
    grad = np.zeros(W.shape)
    posprods = np.zeros(W.shape)
    negprods = np.zeros(W.shape)
    # imagine bias as additional hidden and visible unit
    # bias is a term to be added for each visible unit, for each hidden unit, and there are 5 ratings
    # Ref: Salakhutdinov et al. research paper
    hidden_bias = rbm.getInitialHiddenBias(_F) # b_j is bias of hidden feature j
    hidden_bias_grad = np.zeros(hidden_bias.shape)

    visible_bias = rbm.getInitialVisibleBias(trStats["n_movies"], K) # b_ik is the bias of rating k for movie i
    visible_bias_grad = np.zeros(visible_bias.shape)

    # create arrays to store our loss for each epoch
    train_loss = []
    val_loss = []

    # store best weights
    bestWeights = W
    # store best biases
    best_hidden_bias = hidden_bias
    best_visible_bias = visible_bias

    for epoch in range(1, epochs):
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
            visible_biasForUser = visible_bias[ratingsForUser[:,0],:]

            ### LEARNING ###
            # propagate visible input to hidden units
            posHiddenProb = rbm.visibleToHiddenVecBias(v, weightsForUser, hidden_bias)
            # get positive gradient
            # note that we only update the movies that this user has seen!
            posprods[ratingsForUser[:, 0], :,
                    :] = rbm.probProduct(v, posHiddenProb)

            ### UNLEARNING ###
            # sample from hidden distribution
            sampledHidden = rbm.sample(posHiddenProb)
            # propagate back to get "negative data"
            negData = rbm.hiddenToVisibleBias(sampledHidden, weightsForUser, visible_biasForUser)
            # propagate negative data to hidden units
            negHiddenProb = rbm.visibleToHiddenVecBias(negData, weightsForUser, hidden_bias)
            # get negative gradient
            # note that we only update the movies that this user has seen!
            negprods[ratingsForUser[:, 0], :, :] = rbm.probProduct(
                negData, negHiddenProb)

            # we average over the number of users in the batch (if we use mini-batch)
            # implement L2 regularization; reference: https://sudonull.com/post/128613-Regularization-in-a-restricted-Boltzmann-machine-experiment
            grad[ratingsForUser[:, 0], :, :] = rbm.getAdaptiveLearningRate(lr0=_initialLearningRate, epoch=epoch, k=_learningRateDecay) * \
                (posprods[ratingsForUser[:, 0], :, :] -
                negprods[ratingsForUser[:, 0], :, :] -
                _regularization * W[ratingsForUser[:, 0], :, :])

            # give some inertia to the gradient updates, limiting the risk that your gradient starts oscillating
            W[ratingsForUser[:, 0], :, :] += (1-_momentum) * grad[ratingsForUser[:, 0], :, :] + \
                _momentum * last_grad[ratingsForUser[:, 0], :, :]

            # calculate the gradient wrt biases
            # refer to update rule for biases: https://stats.stackexchange.com/questions/139138/updating-bias-with-rbms-restricted-boltzmann-machines
            # gradient for hidden bias
            hidden_bias_grad = rbm.getAdaptiveLearningRate(lr0=_initialLearningRate, epoch=epoch, k=_learningRateDecay) * \
                (posHiddenProb - 
                negHiddenProb - 
                _regularization * hidden_bias)
            # give some inertia to gradient updates
            hidden_bias += (1-_momentum) * hidden_bias_grad + \
                _momentum * last_hidden_bias_grad
            
            # gradient for visible bias
            visible_bias_grad[ratingsForUser[:,0],:] = rbm.getAdaptiveLearningRate(lr0=_initialLearningRate, epoch=epoch, k=_learningRateDecay) * \
                (v - 
                negData -
                _regularization * visible_bias[ratingsForUser[:,0],:])
            # give some inertia to gradient updates
            visible_bias[ratingsForUser[:,0],:] += (1-_momentum) * visible_bias_grad[ratingsForUser[:,0],:] + \
                _momentum * last_visible_bias_grad[ratingsForUser[:,0],:]


        # Print the current RMSE for training and validation sets
        # this allows you to control for overfitting e.g
        # We predict over the training set
        tr_r_hat = rbm.predictWithBias(trStats["movies"], trStats["users"], W, hidden_bias, visible_bias, training)
        trRMSE = lib.rmse(trStats["ratings"], tr_r_hat)
        train_loss.append(trRMSE)

        # We predict over the validation set
        vl_r_hat = rbm.predictWithBias(vlStats["movies"], vlStats["users"], W, hidden_bias, visible_bias, training)
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
            lr0=_initialLearningRate, epoch=epoch, k=_learningRateDecay))
        print("")

    # plot the evolution of training and validation RMSE
    # plt.figure(figsize=(8, 8))
    # plt.plot(train_loss, label='Training Loss')
    # plt.plot(val_loss, label='Validation Loss')
    # plt.legend(loc='upper right')
    # plt.ylabel('RMSE')
    # plt.ylim([0, 2.0])
    # plt.title('Training and Validation Loss')
    # plt.xlabel('epoch')
    # plt.show()

    #store the paras in dict
    # the parameter name of interest
    parameter_list = ["F", "initialLearningRate", 'learningRateDecay','regularization' , 'momentum','min_train_loss','min_val_loss' ]
    # the value of a single run of the rbm
    value_list = [_F,_initialLearningRate, _learningRateDecay, _regularization, _momentum,min(train_loss),min(val_loss)]
    # dict to store the values
    parameter_dict ={}
    # index to ensure the value_list values would be store properly
    index = 0
    for key in parameter_list:
        parameter_dict[key] = value_list[index]
        # after values are stored increase index by 1
        index = index + 1

    return(parameter_dict)
### END ###
# This part you can write on your own
# you could plot the evolution of the training and validation RMSEs for example

#predictedRatings = np.array(
#     [rbm.predictForUserWithBias(user, bestWeights, best_hidden_bias, best_visible_bias, training) for user in trStats["u_users"]])
#np.savetxt("predictions/predictedRatings.txt", predictedRatings)



def finding_rbm_parameters(_F_list, _initialLearningRate_list, _learningRateDecay_list, 
                            _regularization_list, _momentum_list):
    #measure run time for the program
    start_time  = datetime.datetime.now()
    
    # dict of list to store parameters
    #list of parameter to be stored
    parameter_list = ["F", "initialLearningRate", 'learningRateDecay','regularization' , 'momentum','min_train_loss','min_val_loss' ]
    _stored_parameter = {}
    # create a ls in dict for each element in parameter_list
    for para in parameter_list:
        _stored_parameter[para] = []

    # create folder if folder does not exits
    # getting the date of running py
    today = str(datetime.date.today())
    # namming foldername based on today date
    foldername = 'rbm_results_'+ today
    # check if folder exist, if have return TRUE
    folder_exists = os.path.exists(foldername)
    if not folder_exists:
        # addtional condition to ensure
        os.makedirs(foldername)

    # number of combination of hyper parameters
    num_combi = len(_F_list)*len(_initialLearningRate_list)*len(_learningRateDecay_list)
    num_combi = num_combi*len(_regularization_list)*len(_momentum_list)
    current_left = num_combi

    # iterate in _F_list
    for _F in _F_list:
        # iterate in initialLearningRate
        for _initialLearningRate in _initialLearningRate_list:
            # iterate in _learningRateDecay_list
            for _learningRateDecay in _learningRateDecay_list:
                #iterate in _regularization_list
                for _regularization in _regularization_list:
                    # ieterate in _momentum_list
                    for _momentum in _momentum_list:
                        # para to print
                        para_print = ''
                        para_print = 'F: ' + str(_F) + ' ' + 'InitialLearningRate: ' + str(_initialLearningRate) + ' '
                        para_print = para_print + 'LearningRateDecay: ' + str(_learningRateDecay) + ' ' + 'Regularization: ' + str(_regularization) + ' '
                        para_print = para_print + 'momentum: ' + str(_momentum) 
                        print(para_print)
                        # run the rbm with the different combination of the parameters
                        rbm_results = rbm_model(_F,_initialLearningRate, _learningRateDecay, _regularization, _momentum)
                        
                        #storing the rbm results into the dict to convert to df for exporting
                        for _para in _stored_parameter:
                            _stored_parameter[_para].append(rbm_results[_para])
                        
                        # minus one for each combination done
                        current_left = current_left - 1

                #save results incase system crash
                # convert _stored_parameter to df and export as to csv
                _stored_parameter_df = pd.DataFrame(data = _stored_parameter)
                # the outputfile name with today date
                outputfilename = 'stored_parameter_' + today + '.csv'
                # the file path
                filesavepath = foldername + '/' + outputfilename
                # exporting the results to csv
                _stored_parameter_df.to_csv(filesavepath)

                #save percentage complete as the title of a txt file
                percent_left =  round((current_left/ num_combi),2)
                # update the txtfilename containing the run date and run duration
                updatepercent = foldername + '/' + str(num_combi) + '_' +str(percent_left) + '.txt'
                percent = open(updatepercent,"w+")
                percent.close()

    # save overall results into a folder
    # convert _stored_parameter to df and export as to csv
    _stored_parameter_df = pd.DataFrame(_stored_parameter)
    # the outputfile name with today date
    outputfilename = 'stored_parameter_' + today + '.csv'
    # the file path
    filesavepath = foldername + '/' + outputfilename
    # exporting the results to csv
    _stored_parameter_df.to_csv(filesavepath)
    
    #save run time as the title of a txt file
    runtime = str(datetime.datetime.now() - begin_time)
    runtime = runtime.replace(":", "_")

    ## commented out to check this change would work if used datetime.datetime.now() - begin_time
    # # convert the seconds to minutes and hours
    # runtime = time.strftime("%H_%M_%S", time.gmtime(runtime))
    
    # update the txtfilename containing the run date and run duration
    txtfilename = str(today) + '_runtime_' + runtime + '.txt'
    # create the txt file
    f= open(txtfilename,"w+")
    # close the txt files
    f.close() 

    # return _stored_parameter dict 
    return(_stored_parameter)


## testing parameter based on smaller parameter range
epochs = 10

# TODO Hyper parameter tuning F, (number of hidden units)
# F in range of 8 to 50, increment of 1
# F can only interger as rmb function would product: TypeError: 'float' object cannot be interpreted as an integer
F_list = parameters_list(8,9,1)

# * We are using adaptive learning rate instead of a fixed gradientLearningRate
# //gradientLearningRate = 0.1

# * Use this to select ideal learning rate at epoch 1
# initialLearningRate_list = [0.01, 0.1]

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
## fix rest, vary initialLearningRate_list 
# initialLearningRate_list with increment of 0.001
initialLearningRate_list = parameters_list(0.001,0.005,0.001)
# initialLearningRate_list with increment of 0.01
initialLearningRate_list.extend(parameters_list(0.01,0.05,0.01))
# extend initialLearningRate_list with element of incerment 0.1
initialLearningRate_list.extend(parameters_list(0.1,0.5,0.1))
# extend initialLearningRate_list with element of incerment 1
initialLearningRate_list.extend(parameters_list(1,5,1))
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#

#  TODO Hyper parameter tuning
# ? Range from 1 to 5
# learningRateDecay_list = parameters_list(1,5,1)
# learningRateDecay_list = [0.0001,0.001,0.01,0.1]
## Fix learningRateDecay_list
learningRateDecay_list = [0.01, 0.1]

# * Set the regularization strength here
# TODO Hyper parameter tuning
# ? Range from 0 to 0.05
# regularization_list = parameters_list(0,0.05,0.01)
## Fix regularization_list
regularization_list = [0, 0.01, 0.02]

# * Momemntum
# TODO Hyper parameter tuning
# ? 0 to 1
# momentum_list = parameters_list(0.5,1,0.1)
# momentum_list = [0.5,0.9,0.99]
## Fix momentum_list
momentum_list = [0.5,0.99]

abc = finding_rbm_parameters(F_list, initialLearningRate_list, learningRateDecay_list, 
                            regularization_list, momentum_list)

print(abc)
