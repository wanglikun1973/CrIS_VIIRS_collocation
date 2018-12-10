import numpy as np
import glob
import geo
import time
import pdb

start_time = time.time()

dataDir='./data/'

# get CrIS files 
cris_sdr_files = sorted(glob.glob(dataDir+'SCRIS*'))
cris_geo_files = sorted(glob.glob(dataDir+'GCRSO*'))

# get VIIRS files 
viirs_sdr_files = sorted(glob.glob(dataDir+'SVM15*'))
viirs_geo_files = sorted(glob.glob(dataDir+'GMODO*'))

# read VIIRS data 
viirs_lon, viirs_lat, viirs_satAzimuth, viirs_satRange, viirs_satZenith = geo.read_viirs_geo(viirs_geo_files)
viirs_bt, viirs_rad, viirs_sdrQa = geo.read_viirs_sdr(viirs_sdr_files)


# read CrIS data 
cris_lon, cris_lat, cris_satAzimuth, cris_satRange, cris_satZenith = geo.read_cris_geo(cris_geo_files)

cris_realLW, cris_realMW, cris_realSW, cris_sdrQa, cris_geoQa, cris_dayFlag = geo.read_cris_sdr(cris_sdr_files , sdrFlag=True)

# compute CrIS Pos Vector in EFEC on the Earth Surface 
cris_pos= np.zeros(np.append(cris_lat.shape, 3))
cris_pos[:, :, :, 0], cris_pos[:, :, :, 1], cris_pos[:, :, :, 2] \
    = geo.LLA2ECEF(cris_lon, cris_lat, np.zeros_like(cris_lat))

# compute CrIS LOS Vector in ECEF 
cris_east, cris_north, cris_up = geo.RAE2ENU(cris_satAzimuth, cris_satZenith, cris_satRange)

cris_los= np.zeros(np.append(cris_lat.shape, 3))
cris_los[:, :, :, 0], cris_los[:, :, :, 1], cris_los[:, :, :, 2] = \
    geo.ENU2ECEF(cris_east, cris_north, cris_up, cris_lon, cris_lat)

# compute viirs POS vector in ECEF
viirs_pos= np.zeros(np.append(viirs_lat.shape, 3))
viirs_pos[:, :, 0], viirs_pos[:, :, 1], viirs_pos[:, :, 2] = \
    geo.LLA2ECEF(viirs_lon, viirs_lat, np.zeros_like(viirs_lat))

# cris_los is pointing from pixel to satellite, we need to
#   change from satellite to pixel
cris_los = -1.0*cris_los

# using Kd-tree to find the closted pixel of VIIRS for each CrIS FOV
dy, dx = geo.match_cris_viirs(cris_los, cris_pos, viirs_pos, viirs_sdrQa)
print("collocation are done in --- %s seconds --- for %d files " % (time.time() - start_time, len(cris_sdr_files)))

# collocation is done

##############################################################################
# showing the collocated images 
#############################################################################
start_time = time.time()

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib.colors as colors
import matplotlib.cm as cmx



m = Basemap(resolution='l', projection='cyl',  \
		llcrnrlon=cris_lon.min(), llcrnrlat=cris_lat.min(),  
        urcrnrlon=cris_lon.max(), urcrnrlat=cris_lat.max())
m.drawcoastlines()
m.drawcountries()
m.drawstates()

# meridians on bottom and left
parallels = np.arange(0.,81,10.)
m.drawparallels(parallels,labels=[False,True,True,False])
meridians = np.arange(10.,351.,20.)
m.drawmeridians(meridians,labels=[True,False,False,True])

# create color map 
jet = cm = plt.get_cmap('jet') 
cNorm  = colors.Normalize(vmin=220, vmax=310)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

# show collocated pixels 
for k, j, i in np.ndindex(cris_lat.shape):
	
	ix=dx[k,j,i]
	iy=dy[k,j,i]
	vcolorVal = np.squeeze(scalarMap.to_rgba(viirs_bt[iy, ix]))
	vx, vy = m(viirs_lon[iy, ix], viirs_lat[iy, ix])
	cs1 = m.scatter(vx, vy, s=0.5, c=vcolorVal, edgecolor='none', cmap='jet', marker='.')

plt.savefig('myfig', dpi=600)    

print("making plots is using --- %s seconds " % (time.time() - start_time))



 
