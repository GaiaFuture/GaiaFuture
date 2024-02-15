import xarray as xr
import numpy as np
from dask_jobqueue import PBSCluster
from dask.distributed import Client
def get_cluster(account,cores=30):    
    """Spin up a dask cluster.

    Keyword arguments:
    account -- your account number, e.g. 'UCSB0021'
    cores -- the number of processors requested

    Returns:
    client -- can be useful to inspect client.cluster or run client.close()
    """

    cluster = PBSCluster(
    cores=1, # The number of cores you want
    memory='10GB', # Amount of memory
    processes=1, # How many processes
    queue='casper', # The type of queue to utilize (/glade/u/apps/dav/opt/usr/bin/execcasper)
    local_directory='$TMPDIR', # Use your local directory
    resource_spec='select=1:ncpus=1:mem=10GB', # Specify resources
    account=account, # Input your project ID here
    walltime='02:00:00', # Amount of wall time
    )

    # Scale up
    cluster.scale(cores)
    
    # Setup your client
    client = Client(cluster)

    return client
    
def fix_time(ds):
    '''fix CESM monthly time-parsing bug'''
    yr0=str(ds['time.year'][0].values)
    ds['time']=xr.cftime_range(yr0,periods=len(ds.time),freq='MS',calendar='noleap')
    return ds

#------Weight Gridcells by Landarea---
def weight_landarea_gridcells(da,landarea):
    weighted_avg_area = da.weighted(landarea).mean(dim = 'gridcell') 
    return weighted_avg_area

#------Weighted Averages by Time---
def yearly_weighted_average(da):
    # Get the array of number of days from the main dataset
    days_in_month = da['time.daysinmonth']
    weighted_sum = (days_in_month*da).groupby("time.year").sum(dim = 'time') # Multiply each month's data by corresponding days in month
    total_days = days_in_month.groupby("time.year").sum(dim = 'time')  # Total days in the year
    return weighted_sum / total_days  # Calculate weighted average for the year