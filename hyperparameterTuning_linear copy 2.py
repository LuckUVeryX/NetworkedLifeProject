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
L = list(np.arange(0, 60, 1))
# L = list(np.arange(7, 8, 0.1))

def get_plot_dimension():
    num_plots = len(L)
    return int(np.ceil(np.sqrt(num_plots)))


def update_plot_location(x, y, dimension):
    if x < dimension - 1:
        x += 1
    else:
        y += 1
        x = 0
    return x, y


def hyperparameterTuning():
    print("--- Commencing Hyper Parameter Tuning")
    start_time = datetime.now().replace(microsecond=0)
    # Initialise Variables
    results = []
    train_loss = []
    val_loss = []
    best_val_loss = np.inf

    x = 0
    y = 0

    plot_dimension = get_plot_dimension()
    # ! Resize this if training with many different parameters
    # ? Might want to test with epochs = 1 to see if pdf will fit the plots
    fig, axs = plt.subplots(
        plot_dimension, plot_dimension, sharex=True, sharey=True, figsize=(40, 20))

    # Loop over the different parameters
    for l in L:
        print("--- Training with L {}".format(l))
        b,l2,train_loss_1, val_loss_1 = linearRegression.linearmodel(A = A,c= c, l=l)
        # convert val_loss float to list
        train_loss.extend([train_loss_1])
        val_loss.extend([val_loss_1])
        # Append results in the form of dictionary
        results.append({"Training Loss" : train_loss_1, "Validation Loss": val_loss_1,"L": l})

        # Save the weights and biases of the best model
        if min(val_loss) < best_val_loss:
            best_val_loss = min(val_loss)
            best_hidden_bias = b

        # Plot the evolution of training and validation RMSE
        axs[x, y].plot(train_loss)
        axs[x, y].plot(val_loss)
        axs[x, y].set(xlabel='l', ylabel='RMSE')
        axs[x, y].set_title('L {}'.format(l))

        # Update the index to plot plots
        x, y = update_plot_location(x, y, plot_dimension)

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

    # Output Plot
    print("--- Plotting in progress")
    for ax in axs.flat:
        ax.label_outer()
        # make the axis ticks more obvious 
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    line_labels = ["Training Loss", "Validation Loss"]
    fig.legend(labels=line_labels)

    print("--- Saving plots...")
    plt.savefig("predictions/{}/{}.pdf".format(date, time))

    end_time = datetime.now().replace(microsecond=0)
    print("--- Finished hyperparameter tuning ---")
    print("--- Time Taken ---")
    print("--- {} ---".format(end_time-start_time))

    # plt.show()


# Run main hyperparameter tuning function
hyperparameterTuning()
