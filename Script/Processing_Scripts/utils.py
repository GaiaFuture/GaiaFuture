#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ----           load libraries           ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dask_jobqueue import PBSCluster
from dask.distributed import Client

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ----  server request to aid processing  ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
#-------Gridcell Landareas Data-----
# reading, storing, subsetting
landarea_file = '/glade/campaign/cgd/tss/projects/PPE/helpers/sparsegrid_landarea.nc'

landarea_ds = xr.open_dataset(landarea_file)

landarea = landarea_ds['landarea']

#-------Dummy Variable Data---------
# dummy data to have stored for preloaded visual on 
dummy_filepath = '/glade/campaign/cgd/tss/projects/PPE/PPEn11_OAAT/CTL2010/hist/PPEn11_CTL2010_OAAT0000.clm2.h0.2005-02-01-00000.nc'


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
#------Weight Gridcells by Landarea---
def weight_landarea_gridcells(da,landarea):

    # weigh landarea variable by mean of gridcell dimension
    da['landarea'] = da.weighted(landarea).mean(dim = 'gridcell')              # changed to da['landarea'] instead of weighted_avg_area. should it be da.landarea?

    return da                                           # This now works when importing utils (changed from weighted_avg_area)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ----     weigh dummy data time dim      ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#------Weighted Averages by Time---
def yearly_weighted_average(da):
    # Get the array of number of days from the main dataset
    days_in_month = da['time.daysinmonth']

    # Multiply each month's data by corresponding days in month
    weighted_sum = (days_in_month*da).groupby("time.year").sum(dim = 'time')

    # Total days in the year
    total_days = days_in_month.groupby("time.year").sum(dim = 'time')

    # Calculate weighted average for the year
    da['time'] = weighted_sum / total_days            # This now works when importing utils. (changed from return weighted_sum / total_days)

    return da