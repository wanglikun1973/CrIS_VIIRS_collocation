import numpy as np
from numpy import linalg as LA
from numpy import sqrt, sin, cos, deg2rad, arctan2, \
    arcsin, rad2deg
import xml.etree.ElementTree as etree
import h5py

from pykdtree.kdtree import KDTree


WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_B = WGS84_A*(1.0 - WGS84_F)
WGS84_E2 = 2 * WGS84_F - WGS84_F ** 2

#Rotational angular velocity of Earth in radians/sec from IERS
#   Conventions (2003).
ANGVEL = 7.2921150e-5;

def LLA2ECEF(lonIn, latIn, altIn):
    """
	Transform lon,lat,alt (WGS84 degrees, meters) to  ECEF
	x,y,z (meters)
    """
    lonRad = deg2rad(np.asarray(lonIn, dtype=np.float64) ) 
    latRad = deg2rad(np.asarray(latIn, dtype=np.float64) )
    alt    = np.asarray(altIn, dtype=np.float64) 
    a, b, e2 = WGS84_A, WGS84_B, WGS84_E2

    ## N = Radius of Curvature (meters), defined as:
    N = a/sqrt(1.0-e2*(sin(latRad)**2.0))
            
    ##$ calcute X, Y, Z
    x=(N+alt)*cos(latRad)*cos(lonRad)
    y=(N+alt)*cos(latRad)*sin(lonRad)
    z=(b**2.0/a**2.0*N + altIn)*sin(latRad)

    return x, y, z 


def RAE2ENU(azimuthIn, zenithIn, rangeIn):
    """
    Transform azimuth, zenith, range to ENU x,y,z (meters)
    """
    azimuth = deg2rad(np.asarray(azimuthIn, dtype=np.float64))
    zenith  = deg2rad(np.asarray(zenithIn, dtype=np.float64))
    r       = np.asarray(rangeIn, dtype=np.float64)

    # up 
    up = r*cos(zenith)
  
    # projection on the x-y plane 
    p = r*sin(zenith)  
  
    # north 
    north = p*cos(azimuth)
 
    # east
    east = p*sin(azimuth)   

    return east, north, up


def ENU2ECEF (east, north, up, lon, lat):
    """
    Convert local East, North, Up (ENU) coordinates to the (x,y,z) Earth Centred Earth Fixed (ECEF) coordinates
    Reference is here:  
    http://www.navipedia.net/index.php/Transformations_between_ECEF_and_ENU_coordinates
    Note that laitutde should be geocentric latitude instead of geodetic latitude 
    Note: 

    On June 16 2015
    This note from https://en.wikipedia.org/wiki/Geodetic_datum 
    Note: \ \phi is the geodetic latitude. A prior version of this page showed use of the geocentric latitude (\ \phi^\prime).
    The geocentric latitude is not the appropriate up direction for the local tangent plane. If the
    original geodetic latitude is available it should be used, otherwise, the relationship between geodetic and geocentric
    latitude has an altitude dependency, and is captured by ...
    """    

    x0 = np.asarray(east, dtype=np.float64)
    y0 = np.asarray(north, dtype=np.float64)
    z0 = np.asarray(up, dtype=np.float64)

    lm = deg2rad(np.asarray(lon, dtype=np.float64))
    ph = deg2rad(np.asarray(lat, dtype=np.float64))

    x=-1.0*x0*sin(lm)-y0*cos(lm)*sin(ph)+z0*cos(lm)*cos(ph)
    y= x0*cos(lm) -y0*sin(lm)*sin(ph)+z0*sin(lm)*cos(ph)
    z= x0*0       +y0*cos(ph)        +z0*sin(ph)   

    return x, y, z
        
##################################################################################### 
def match_cris_viirs(crisLos, crisPos, viirsPos, viirsMask):
    """
    Match crisLos with viirsPos using the method by Wang et al. (2016)
    Wang, L., D. A. Tremblay, B. Zhang, and Y. Han, 2016: Fast and Accurate 
      Collocation of the Visible Infrared Imaging Radiometer Suite 
      Measurements and Cross-track Infrared Sounder Measurements. 
      Remote Sensing, 8, 76; doi:10.3390/rs8010076.     
    """
    
    # Derive Satellite Postion 
    crisSat = crisPos - crisLos 
        
        # using KD-tree to find best matched points 
    
    # build kdtree to find match index 
    pytree_los = KDTree(viirsPos.reshape(viirsPos.size/3, 3))
    dist_los, idx_los = pytree_los.query(crisPos.reshape(crisPos.size/3, 3) , sqr_dists=False)
    
    my, mx = np.unravel_index(idx_los, viirsPos.shape[0:2])
    
    
    idy, idx  = find_match_index(crisLos.reshape(crisLos.size/3, 3),\
                                     crisSat.reshape(crisSat.size/3, 3),\
                                     viirsPos, viirsMask, mx, my)
        
    idy = np.array(idy).reshape(crisLos.shape[0:crisLos.ndim-1])
    idx = np.array(idx).reshape(crisLos.shape[0:crisLos.ndim-1])

    return idy, idx

##############################################################################################
# Satellite data reader 
# read CrIS SDR files 
def read_cris_sdr (filelist, sdrFlag=False):

    """
    Read JPSS CrIS SDR and return LW, MW, SW Spectral. Note that this method
    is very fast but can't open too many files (<1024) simultaneously.  
    """

    sdrs = [h5py.File(filename) for filename in filelist]
    real_lw = np.concatenate([f['All_Data']['CrIS-SDR_All']['ES_RealLW'][:] for f in sdrs])
    real_mw = np.concatenate([f['All_Data']['CrIS-SDR_All']['ES_RealMW'][:] for f in sdrs])
    real_sw = np.concatenate([f['All_Data']['CrIS-SDR_All']['ES_RealSW'][:] for f in sdrs])
    
    if not sdrFlag: 
        return real_lw, real_mw, real_sw
    else:  
        QF3_CRISSDR = np.concatenate([f['All_Data']['CrIS-SDR_All']['QF3_CRISSDR'][:] for f in sdrs])
        QF4_CRISSDR = np.concatenate([f['All_Data']['CrIS-SDR_All']['QF4_CRISSDR'][:] for f in sdrs])

        #sdrQa = shift(shift(qf3,-6),6)
        sdrQa = QF3_CRISSDR & 0b00000011
    
        #GeoQa = shift(shift(shift(qf3, 2),-7), 7)
        geoQa = (QF3_CRISSDR & 0b00000100) >> 2

        # dayFlag = shift(shift(qf4, -7), 7)
        dayFlag = QF4_CRISSDR & 0b00000001
        return real_lw, real_mw, real_sw, sdrQa, geoQa, dayFlag
            
####################################################################################    
## read CrIS GOE files     
def read_cris_geo (filelist, ephemeris = False):
    
    """
    Read JPSS CrIS Geo files and return Longitude, Latitude, SatelliteAzimuthAngle, SatelliteRange, SatelliteZenithAngle.
        if ephemeris=True, then return forTime, midTime, satellite position, velocity, attitude 
    """
    
    geos = [h5py.File(filename) for filename in filelist]
    
    if ephemeris == False:  
        Latitude  = np.concatenate([f['All_Data']['CrIS-SDR-GEO_All']['Latitude'] [:] for f in geos])
        Longitude = np.concatenate([f['All_Data']['CrIS-SDR-GEO_All']['Longitude'][:] for f in geos])
        SatelliteAzimuthAngle = np.concatenate([f['All_Data']['CrIS-SDR-GEO_All']['SatelliteAzimuthAngle'][:] for f in geos])
        SatelliteRange = np.concatenate([f['All_Data']['CrIS-SDR-GEO_All']['SatelliteRange'][:] for f in geos])
        SatelliteZenithAngle = np.concatenate([f['All_Data']['CrIS-SDR-GEO_All']['SatelliteZenithAngle'][:] for f in geos])
        return Longitude, Latitude, SatelliteAzimuthAngle, SatelliteRange, SatelliteZenithAngle
    if ephemeris == True:
        FORTime  = np.concatenate([f['All_Data']['CrIS-SDR-GEO_All']['FORTime'] [:] for f in geos])
        MidTime  = np.concatenate([f['All_Data']['CrIS-SDR-GEO_All']['MidTime'] [:] for f in geos])
        SCPosition  = np.concatenate([f['All_Data']['CrIS-SDR-GEO_All']['SCPosition'] [:] for f in geos])
        SCVelocity  = np.concatenate([f['All_Data']['CrIS-SDR-GEO_All']['SCVelocity'] [:] for f in geos])
        SCAttitude  = np.concatenate([f['All_Data']['CrIS-SDR-GEO_All']['SCAttitude'] [:] for f in geos])
        return FORTime, MidTime, SCPosition, SCVelocity, SCAttitude

#################################################################
## READ VIIRS Geofiles
 
def read_viirs_geo (filelist, ephemeris=False, hgt=False):

    """
    Read JPSS VIIRS Geo files and return Longitude, Latitude, SatelliteAzimuthAngle, SatelliteRange, SatelliteZenithAngle.
        if ephemeris=True, then return midTime, satellite position, velocity, attitude 
    """

    if type(filelist) is str: filelist = [filelist]
    if len(filelist) ==0: return None
    
    # Open user block to read Collection_Short_Name
    with h5py.File(filelist[0], 'r') as fn:
            user_block_size = fn.userblock_size
 
    with open(filelist[0], 'rU') as fs:
            ub_text = fs.read(user_block_size)
    ub_xml = etree.fromstring(ub_text.rstrip('\x00'))
    
    #print(ub_text)
    #print(etree.tostring(ub_xml))
    CollectionName = ub_xml.find('Data_Product/N_Collection_Short_Name').text+'_All'    
    #print(CollectionName)

    # read the data
    geos = [h5py.File(filename, 'r') for filename in filelist]
    
    if not ephemeris:
        Latitude  = np.concatenate([f['All_Data'][CollectionName]['Latitude'][:]  for f in geos])
        Longitude = np.concatenate([f['All_Data'][CollectionName]['Longitude'][:] for f in geos])
        SatelliteAzimuthAngle = np.concatenate([f['All_Data'][CollectionName]['SatelliteAzimuthAngle'][:] for f in geos])
        SatelliteRange = np.concatenate([f['All_Data'][CollectionName]['SatelliteRange'][:] for f in geos])
        SatelliteZenithAngle = np.concatenate([f['All_Data'][CollectionName]['SatelliteZenithAngle'][:] for f in geos])
        Height = np.concatenate([f['All_Data'][CollectionName]['Height'][:] for f in geos])
        if hgt: 
            return Longitude, Latitude, SatelliteAzimuthAngle, SatelliteRange, SatelliteZenithAngle, Height
        else: 
            return Longitude, Latitude, SatelliteAzimuthAngle, SatelliteRange, SatelliteZenithAngle
    
    if ephemeris: 
        MidTime  = np.concatenate([f['All_Data'][CollectionName]['MidTime'] [:] for f in geos])
        SCPosition  = np.concatenate([f['All_Data'][CollectionName]['SCPosition'][:] for f in geos])
        SCVelocity  = np.concatenate([f['All_Data'][CollectionName]['SCVelocity'][:] for f in geos])
        SCAttitude  = np.concatenate([f['All_Data'][CollectionName]['SCAttitude'][:] for f in geos])
        return MidTime, SCPosition, SCVelocity, SCAttitude 

####################################################################################        
## READ VIIRS SDR files
def read_viirs_sdr (filelist):
    
    """
    READ VIIRS SDR files
    """
        
    if type(filelist) is str: filelist = [filelist]
    if len(filelist) == 0: return None
    
    # Opne userbloack to read Collection_Short_Name
    with h5py.File(filelist[0], 'r') as fn:
            user_block_size = fn.userblock_size
 
    with open(filelist[0], 'rU') as fn:
            ub_text = fn.read(user_block_size)

    ub_xml = etree.fromstring(ub_text.rstrip('\x00'))

    
    #print(etree.tostring(ub_xml, pretty_print=True))
    CollectionName = ub_xml.find('Data_Product/N_Collection_Short_Name').text+'_All'    
    #print(CollectionName)
    
    s='All_Data/'+CollectionName+'/'

    # Read datasets
    sdrs = [h5py.File(filename, 'r') for filename in filelist]
    
    if 'BrightnessTemperature' in sdrs[0][s].keys(): 
        BrightnessTemperature = np.concatenate([f[s+'BrightnessTemperature'] for f in sdrs])
        BT = BrightnessTemperature
        
        if 'BrightnessTemperatureFactors' in sdrs[0][s].keys(): 
            BrightnessTemperatureFactors=np.concatenate([f[s+'BrightnessTemperatureFactors'] for f in sdrs])
            BT = BrightnessTemperature * BrightnessTemperatureFactors[0] + BrightnessTemperatureFactors[1]
        
    if 'Reflectance' in sdrs[0][s].keys(): 
    
        Reflectance = np.concatenate([f[s+'Reflectance'] for f in sdrs])
        ReflectanceFactors=np.concatenate([f[s+'ReflectanceFactors'] for f in sdrs])
        BT = Reflectance * ReflectanceFactors[0] + ReflectanceFactors[1]    
    
    Radiance = np.concatenate([f[s+'Radiance'] for f in sdrs])
    
    if 'RadianceFactors' in sdrs[0][s].keys(): 
        RadianceFactors=np.concatenate([f[s+'RadianceFactors'] for f in sdrs])
        RAD = Radiance * RadianceFactors[0] + RadianceFactors[1]
    else: 
        RAD = Radiance
    
    if CollectionName.find('VIIRS-I') >= 0:
        qaStr = 'QF1_VIIRSIBANDSDR' 
    else:   qaStr = 'QF1_VIIRSMBANDSDR' 
    QF1_VIIRSBANDSDR = np.concatenate([f[s+qaStr] for f in sdrs])
        
    return BT, RAD, QF1_VIIRSBANDSDR
##############################################################################################
    
def find_match_index (cris_los, cris_sat, viirs_pos_in, viirs_sdrQa_in, \
                      mx, my, fovDia=0.963):


        nLine, nPixel = viirs_pos_in.shape[0:2]
        crisShape = cris_los.shape[0:cris_los.ndim]

        # setup parameters
        cos_half_fov=cos(deg2rad(fovDia/2.0))
        if nPixel == 3200: nc = np.round(deg2rad(0.963/2)*833.0/0.75*4).astype(np.int)
        if nPixel == 6400: nc = np.round(deg2rad(0.963/2)*833.0/0.375*4).astype(np.int)


        # return list
        index_x = []
        index_y = []

        for i in range(0, mx.size):

                xd = mx[i]
                yd = my[i]

                xb = 0        if xd-nc <0        else xd-nc
                xe = nPixel-1 if xd+nc >nPixel-1 else xd+nc

                yb = 0        if yd-nc <0        else yd-nc
                ye = nLine-1  if yd+nc >nLine-1  else yd+nc

                viirs_pos = viirs_pos_in[yb:ye, xb:xe, : ]
                viirs_Qa  = viirs_sdrQa_in[yb:ye, xb:xe]
                viirs_los = viirs_pos  - cris_sat[i, :]
                temp = np.dot(viirs_los, cris_los[i, :])
                temp = temp / LA.norm(viirs_los, axis=2)
                cos_angle = temp / LA.norm(cris_los[i, :])

                iy, ix = np.where ( (viirs_Qa == 0) & (cos_angle > cos_half_fov) )

                index_x.append(ix+xb)
                index_y.append(iy+yb)

        return index_y, index_x
        
    
                    
    
    
    
        
    
    
    
    
        
    

