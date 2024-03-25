#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ----           load libraries           ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import glob
from dask_jobqueue import PBSCluster
from dask.distributed import Client

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ----  server request to aid processing  ----
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_cluster(account,cores=30):    
    """Spin up a dask cluster.

    Keyword arguments:
    account -- your account number, e.g. 'UCSB0021'
    cores -- the number of processors requested

    Returns:
    client -- can be useful to inspect client.cluster or run client.close()
    """

    cluster = PBSCluster(
    # The number of cores you want
    cores=1,
    # Amount of memory
    memory='10GB',
     # How many processes
    processes=1,
    # The type of queue to utilize (/glade/u/apps/dav/opt/usr/bin/execcasper)
    queue='casper', 
    # Use your local directory
    local_directory = '$TMPDIR',
    # Specify resources
    resource_spec='select=1:ncpus=1:mem=10GB',
    # Input your project ID here
    account = account,
    # Amount of wall time
    walltime = '02:00:00',
    )

    # Scale up
    cluster.scale(cores)
    
    # Setup your client
    client = Client(cluster)

    return client



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ----     cluster reading function       ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#modify the function if you want to pass the parameter
def read_all_simulations(var):
    '''prepare cluster list and read to create ensemble(group of data)
    use preprocess to select only certain dimension and a variable'''
    # read all simulations as a list
    cluster_list= sorted(glob.glob('/glade/campaign/cgd/tss/projects/PPE/PPEn11_LHC/transient/hist/PPEn11_transient_LHC[0][0-5][0-9][0-9].clm2.h0.2005-02-01-00000.nc'))
    cluster_list = cluster_list[1:len(cluster_list)]

    def preprocess(ds, var):
        '''using this function in xr.open_mfdataset as preprocess
        ensures that when only these four things are selected 
        before the data is combined'''
        return ds[['lat', 'lon', 'time', var]]
    
    #read the list and load it for the notebook
    ds = xr.open_mfdataset( cluster_list, 
                                   combine='nested',
                                   preprocess = lambda ds: preprocess(ds, var),
                                   parallel= True, 
                                   concat_dim="ens")
    return ds


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ----     load data stored in casper     ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #----  Gridcell Landareas Data     -----
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# reading, storing, subsetting
landarea_file = '/glade/campaign/cgd/tss/projects/PPE/helpers/sparsegrid_landarea.nc'

landarea_ds = xr.open_dataset(landarea_file)

landarea = landarea_ds['landarea']
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #----        Parameter Data.       ----
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# x parameter data for assessment
df = pd.read_csv('/glade/campaign/asp/djk2120/PPEn11/csvs/lhc220926.txt',index_col=0)
# the only dimension here is the 'member' aka file index id [LCH0001-500]
# convert to xr.ds
params = xr.Dataset(df)

# subset wrangle user selected parameter c
def subset_param(param):

    # xr.da subset of parameter data 
    return param_avg  = params[param]


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ----     correct time-parsing bug       ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def fix_time(da):
    
    '''fix CESM monthly time-parsing bug'''
    
    yr0 = str(da['time.year'][0].values)
    
    da['time'] = xr.cftime_range(yr0,periods=len(da.time),freq='MS',calendar='noleap')
    
    return da


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ----  weigh dummy landarea by gridcell  ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #----  Weight Gridcells by Landarea ----
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def weight_landarea_gridcells(da,landarea):

    # weigh landarea variable by mean of gridcell dimension
    da['landarea'] = da.weighted(landarea).mean(dim = 'gridcell')             

    return da                                          

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ----     weigh dummy data time dim      ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#------Weighted Averages by Time---
def yearly_weighted_average(da):
    # Get the array of number of days from the main dataset
    days_in_month = da['time.daysinmonth']

    # Multiply each month's data by corresponding days in month
    weighted_month = (days_in_month*da).groupby("time.year").sum(dim = 'time')

    # Total days in the year
    days_per_year = days_in_month.groupby("time.year").sum(dim = 'time')

    # Calculate weighted average for the year
    return weighted_month / days_per_year
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ----    Subset User Selection Funct     ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def subset_var_cluster(var):
    '''Subset the selected variable 
    (s) between 2005-2010 [for now, will be time range]
    as a xr.da.'''
    
    # Read in and wrangle user selected variable cluster
    da_v = read_all_simulations(var)
    # feb. ncar time bug
    da = fix_time(da_v)
    # convert xr.ds to xr.da
    return da = da[var]


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ----      Subset Var Wrangle Funct      ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def wrangle_var_cluster(da):
    '''Weight gridcell dimension by landarea 
    and weight time dimension by the days per each month
    over the total number of days in a year. Globally average
    the selected variable between 2005-2010 
    [for now, will be time range]
    as a xr.da.'''
    # weight gridcell dim by global land area
    da_global = weight_landarea_gridcells(da, landarea)
    # weight time dim by days in month
    da_global_ann = yearly_weighted_average(da_global)
    # take global avg for variable over year dimension
    return var_avg = da_global_ann.mean(dim='year')


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ----  User Selected ML Plotting Funct   ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# function to plot a cluster for to build on for ml
def cluster_ml_plot(param_avg, var_avg):
    '''Basic plotting to aid with gpr assessment, 
    building on currently.'''

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ----         Fit Model            ----
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate predictions and confidence intervals for each variable using the trained model
   # y_pred, sigma = gpr_best.predict(x_pred, return_std=True)     # to be included once upper portion of workflow optimized
   
    # standard 0:1, set constant
    x_pred = np.linspace(0,1,20)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ----         Plot Model           ----
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    plt.figure(figsize=(10, 6))
    plt.scatter(x_test, y_test, color='#62c900ff', label='Observed data')
    plt.plot(x_pred, y_pred, color='#134611', label='GPR Prediction')
    plt.fill_between(x_test.flatten(),
                     y_pred - 1.96 * sigma,, y_pred + 1.96 * sigma,
                     alpha=0.5,
                     color='#9d6b53',
                     label = '95% Confidence Interval')
    # Set plot labels and title
    plt.xlabel(f"Perturbed Parameter: ({param_avg:})")  # ideas for later: param_name = getattr(param_avg)
    plt.ylabel(f"Variable: ({var_avg:})")
    plt.title('Parameter Value Uncertainty Estimation')
    plt.legend()
    # Show the plot
    plt.show()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ---- User Selected Panel Plotting Funct ---- # adding this because working to implement into ml wkflw 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# function to plot a cluster for the dashboard
def cluster_panel_plot(param_avg, var_avg):
'''Basic plotting to aid with dashboard set up 
    building on currently.'''
    data = pd.DataFrame({'x': param_avg, 'y': var_avg})
    
    # Create scatter plot using hvplot
    hvplot.scatter(
        x='x', y='y', color='#62c900ff', alpha=0.8,
        xlabel=param, ylabel=var, title='2005-2010 Global Average'
    )
    return pn.panel(scatter_plot)
