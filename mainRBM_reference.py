import numpy as np
import rbm
import projectLib as lib
import matplotlib.pyplot as plt

#addtional functions
import pandas as pd
import datetime
import os.path

training = lib.getTrainingData()
validation = lib.getValidationData()
# You could also try with the chapter 4 data
# training = lib.getChapter4Data()

trStats = lib.getUsefulStats(training)
vlStats = lib.getUsefulStats(validation)

K = 5

# SET PARAMETERS HERE!!!
# number of hidden units in range

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


F_list = parameters_list(1,10,1)
epochs_list = parameters_list(10,15,1)
gradientLearningRate_list = parameters_list(0.001,0.005,0.001)

# this function only takes the list of F, epcochs and gradientLeanringRate to test the hyperparameter
# it is to improve the readablity of the code
def rbm_model(_F, _epochs, _gradientLearningRate):
    print(' the current rbm model para')
    print('F: ' + str(_F) + ', epochs: ' +  str(_epochs) + ', gradientLearningRate: ' + str(_gradientLearningRate))
    # Initialise all our arrays
    W = rbm.getInitialWeights(trStats["n_movies"], _F, K)
    grad = np.zeros(W.shape)
    posprods = np.zeros(W.shape)
    negprods = np.zeros(W.shape)

    # create arrays to store our loss for each epoch
    train_loss = []
    val_loss = []

    #create folder to sort plots
    # create file to save all the plots
    current_date_and_time = datetime.datetime.now()
    current_date_string = str(current_date_and_time)[:10]
    # current_time_string = str(current_date_and_time)[11:13] + str(current_date_and_time)[14:16]
    # current_date_time_string = current_date_string + '_' + current_time_string

    # folder name
    _foldername = 'plot_files_' +  current_date_string
    folder_exists = os.path.exists(_foldername) 
    # create folder if the folder do no exists
    if not folder_exists:
        # addtional condition to ensure
        os.makedirs(_foldername)
    

    for epoch in range(1, _epochs):
        # in each epoch, we'll visit all users in a random order
        visitingOrder = np.array(trStats["u_users"])
        np.random.shuffle(visitingOrder)

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
            posprods[ratingsForUser[:, 0], :, :] = rbm.probProduct(v, posHiddenProb)

            ### UNLEARNING ###
            # sample from hidden distribution
            sampledHidden = rbm.sample(posHiddenProb)
            # propagate back to get "negative data"
            negData = rbm.hiddenToVisible(sampledHidden, weightsForUser)
            # propagate negative data to hidden units
            negHiddenProb = rbm.visibleToHiddenVec(negData, weightsForUser)
            # get negative gradient
            # note that we only update the movies that this user has seen!
            negprods[ratingsForUser[:, 0], :, :] = rbm.probProduct(negData, negHiddenProb)

            # we average over the number of users in the batch (if we use mini-batch)
            grad[ratingsForUser[:, 0], :, :] = _gradientLearningRate * (posprods[ratingsForUser[:, 0], :, :] - negprods[ratingsForUser[:, 0], :, :])

            W[ratingsForUser[:, 0], :, :] += grad[ratingsForUser[:, 0], :, :]

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

        print("### EPOCH %d ###" % epoch)
        print("Training loss = %f" % trRMSE)
        print("Validation loss = %f" % vlRMSE)

    # plot the evolution of training and validation RMSE
    plt.figure(figsize=(8, 8))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('RMSE')
    plt.ylim([0,2.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    # plt.show()

    # we would like to save the image stead of showing on the terminal
    # save plot into folder created for the plots 
    plotname = str(_F) + '_' + str(_epochs) + '_' + str(_gradientLearningRate) + '.png'
    plt.savefig(_foldername + '/' + plotname )

    #returning the result for the
    return([_F, _epochs, _gradientLearningRate, min(train_loss), min(val_loss)])

### END ###

def finding_rbm_parameters(_F_list,_epochs_list,_gradientLearningRate_list):
    # dict of list to store parameters
    _stored_parameter = {}
    _stored_parameter['F'] = []
    _stored_parameter['epochs'] = []
    _stored_parameter['gradientLearningRate'] = []
    _stored_parameter['train_loss'] = []
    _stored_parameter['val_loss'] = []


    # literate in _F_list
    for _F in _F_list:

    # iterate in _epochs_list
        for  _epochs in _epochs_list:

    # iterate in _gradientLearningRate_list
            for _gradientLearningRate in _gradientLearningRate_list:
                # this functions return: _F, _epochs, _gradientLearningRate, min(train_loss), min(val_loss)
                rbm_results = rbm_model(_F, _epochs, _gradientLearningRate)

                #storing the rbm results into the dict to convert to df to export
                index_rbm_results = 0
                for _para in _stored_parameter:
                   _stored_parameter[_para] = rbm_results[index_rbm_results]
                   index_rbm_results = index_rbm_results + 1

    #convert dict to df and export df
    _stored_parameter_df  = pd.DataFrame.from_dict(_stored_parameter)
    # naming the file using current date
    current_date_and_time = datetime.datetime.now()
    current_date_string = str(current_date_and_time)[:10]
    current_time_string = str(current_date_and_time)[11:13] + str(current_date_and_time)[14:16]
    current_date_time_string = current_date_string + '_' + current_time_string

    output_filename = 'hyperparameter_selection_' + current_date_time_string + '.xlsx'
    _stored_parameter_df.to_excel(output_filename)

    # print the best parameter for trging_loss/ vaild_loss and return the df
    # best para for trging_loss
    _best_trg_loss_para_df = _stored_parameter_df[_stored_parameter_df.traing_loss == _stored_parameter_df.train_loss.max()]

    # best para for vaild_loss
    _best_vaild_loss_para_df = _stored_parameter_df[_stored_parameter_df.val_loss == _stored_parameter_df.val_loss.max()]

    return([_stored_parameter_df,_best_trg_loss_para_df,_best_vaild_loss_para_df])

# intitalize the function here
# we want the best _F, _epochs, _gradientLearningRate for the lower training lose and the vaildation loss
# output consist of _stored_parameter_df,best_trg_loss_para_df,best_vaild_loss_para_df
para_dfs = finding_rbm_parameters(F_list,epochs_list,gradientLearningRate_list)
stored_parameter_df = para_dfs[0]
best_trg_loss_para_df = para_dfs[1]
best_vaild_loss_para_df = para_dfs[2]

# This part you can write on your own
# you could plot the evolution of the training and validation RMSEs for example

# predictedRatings = np.array([rbm.predictForUser(user, W, training) for user in trStats["u_users"]])
# np.savetxt("predictedRatings.txt", predictedRatings)
