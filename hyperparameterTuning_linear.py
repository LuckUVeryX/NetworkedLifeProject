import linearRegression
import projectLib as lib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from datetime import datetime

training = lib.getTrainingData()
validation = lib.getValidationData()
# You could also try with the chapter 4 data
# training = lib.getChapter4Data()

trStats = lib.getUsefulStats(training)
vlStats = lib.getUsefulStats(validation)
rBar = np.mean(trStats["ratings"])

# * Parameters
A = linearRegression.getA(training)
c = linearRegression.getc(rBar, trStats["ratings"])
# TODO Hyper parameter tuning
# number of hidden units
L = list(np.arange(0, 5.01, 0.01))

def hyperparameterTuning():
    print("--- Commencing Hyper Parameter Tuning")
    start_time = datetime.now().replace(microsecond=0)
    # Initialise Variables
    results = []
    train_loss = []
    val_loss = []

    # Loop over the different parameters
    for l in L:
        print("--- Training with L {}".format(l))
        b,l2,train_loss_1, val_loss_1 = linearRegression.linearmodel(A = A,c= c, l=l)
        # convert val_loss float to list
        train_loss.extend([train_loss_1])
        val_loss.extend([val_loss_1])
        # Append results in the form of dictionary
        results.append({"Training Loss" : train_loss_1, "Validation Loss": val_loss_1,"L": l})

    # Plot the evolution of training and validation RMSE
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.xlabel('L')
    plt.ylabel('RMSE')
    plt.title('L {}'.format(l))
    plt.legend('Training RMSE','Vaildation RMSE')

    # Code to output predictions without overwriting by using current date time
    date, time = linearRegression.get_current_date_and_time()
    if not os.path.exists('predictions/{}/'.format(date)):
        os.makedirs('predictions/{}/'.format(date))

    # Output CSV
    print('--- Writing to CSV')
    df = pd.DataFrame(results)
    df = df.sort_values(by='Validation Loss', ascending=True)
    df.to_csv("predictions/{}/{}.csv".format(date, time))

    # Output best ratings
    print("--- Predicting ratings...")
    bestPredictedRatings = np.array(
        [linearRegression.predict(trStats["movies"], trStats["u_users"] , rBar, b)])
    print("--- Saving predictions")
    np.savetxt("predictions/{}/{}_bestPredictedRatings.txt".format(date, time),
               bestPredictedRatings)

    # Save & Output Plot
    print("--- Saving plots...")
    plt.savefig("predictions/{}/{}.pdf".format(date, time))

    print("--- Plotting in progress")
    plt.show(block = False)
    plt.close()

    end_time = datetime.now().replace(microsecond=0)
    print("--- Finished hyperparameter tuning ---")
    print("--- Time Taken ---")
    print("--- {} ---".format(end_time-start_time))

# Run main hyperparameter tuning function
hyperparameterTuning()
