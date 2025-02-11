import pandas as pd
from functions.plotting import plot_acc_vs_npoints

#############################################################################
###                                Options                                ###
#############################################################################

data_file = 'scripts/conic_error_vs_num_circles_csv'

#############################################################################
###                                  Main                                 ###
#############################################################################

if __name__ == "__main__":

    # Import data
    df=pd.read_csv(data_file, index_col=0)

    # Plot the data
    plot_acc_vs_npoints(df)


    
