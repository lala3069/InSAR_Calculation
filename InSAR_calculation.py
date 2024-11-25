import cgm_library as cgm
import numpy as np
import xarray as xr
import math
import pygmt
import pandas as pd
from Tectonic_Utils.geodesy import insar_vector_functions as ivf


def common_coordinates(track1_structure, track2_structure, decimal = 3, where = False):
    '''
    Find common geographic coordinates of different tracks of a satellite mission
    Input: 
        track1: read out hdf5 structure of the first track 
        track2: read out hdf5 structure of the second track
        decimal: geographic decimal place (default = 3)
    Output: 
        common latitude, common longitude (array)
        if where = True will also gave the pixel location for those coordinates. (array)
    '''

    lon1 = np.round(track1_structure[0]['lon'], decimal)
    lat1 = np.round(track1_structure[0]['lat'], decimal)

    lon2 = np.round(track2_structure[0]['lon'], decimal)
    lat2 = np.round(track2_structure[0]['lat'], decimal)

    common_lon = np.intersect1d(lon1, lon2)
    common_lat = np.intersect1d(lat1, lat2)

    lat_idx_1 = np.searchsorted(lat1, common_lat)
    lon_idx_1 = np.searchsorted(lon1, common_lon)
    lat_idx_2 = np.searchsorted(lat2, common_lat)
    lon_idx_2 = np.searchsorted(lon2, common_lon)

    if where == True:
        return common_lat, common_lon, lat_idx_1, lon_idx_1, lat_idx_2, lon_idx_2 
    else:
        return common_lat, common_lon
        
    
def common_velocity(track1_structure, track2_structure, decimal = 3):
    '''
    Find velocities at common pixels of different tracks of a satellite mission
    Input: 
        track1: read out hdf5 structure of the first track 
        track2: read out hdf5 structure of the second track
        decimal: geographic decimal place (default = 3) 
    Output: 
        track 1 common pixels velocities, track 2 common pixels velocities (array)    
    '''
    y, x, lat_idx_1, lon_idx_1, lat_idx_2, lon_idx_2  = common_coordinates(track1_structure, track2_structure, decimal = decimal, where = True)
    v_1 = track1_structure[0]['velocities']
    v_2 = track2_structure[0]['velocities']
   
    v_1_common = v_1[np.ix_(lat_idx_1, lon_idx_1)]
    v_2_common = v_2[np.ix_(lat_idx_2, lon_idx_2)]

    return v_1_common, v_2_common
    

def common_lkv(track1_structure, track2_structure, decimal = 3):
    '''
    Find look vector components at common pixels of different tracks of a satellite mission
    Input: 
        track1: read out hdf5 structure of the first track 
        track2: read out hdf5 structure of the second track
        decimal: geographic decimal place (default = 3) 
    Output: 
        track 1 common pixels look vector components, track 2 common pixels look vector components (array)   
        [lkv_E1, lkv_N1, lkv_U1], [lkv_E2, lkv_N2, lkv_U2]
    '''
    y, x, lat_idx_1, lon_idx_1, lat_idx_2, lon_idx_2  = common_coordinates(track1_structure, track2_structure, decimal = decimal, where = True)

    lkv_E1 = track1_structure[0]['lkv_E']
    lkv_N1 = track1_structure[0]['lkv_N']
    lkv_U1 = track1_structure[0]['lkv_U']

    lkv_E2 = track2_structure[0]['lkv_E']
    lkv_N2 = track2_structure[0]['lkv_N']
    lkv_U2 = track2_structure[0]['lkv_U']
    
    # Step 4: Assign velocities from A166 and D173 to the grids
    lkv_E1_common = lkv_E1[np.ix_(lat_idx_1, lon_idx_1)]
    lkv_N1_common = lkv_N1[np.ix_(lat_idx_1, lon_idx_1)]
    lkv_U1_common = lkv_U1[np.ix_(lat_idx_1, lon_idx_1)]
    
    lkv_E2_common = lkv_E2[np.ix_(lat_idx_2, lon_idx_2)]
    lkv_N2_common = lkv_N2[np.ix_(lat_idx_2, lon_idx_2)]
    lkv_U2_common = lkv_U2[np.ix_(lat_idx_2, lon_idx_2)]

    return [lkv_E1_common, lkv_N1_common, lkv_U1_common], [lkv_E2_common, lkv_N2_common, lkv_U2_common]
    

def flight_and_incidence_angles(lkv):
    '''
    Find flight heading angles and incidence angles of each pixel
    Input: 
       lkv: look vector components array in format [lkv_E, lkv_N, lkv_U]
    Output: 
        2D array of flight heading angles, 2D array of incidence angles
    '''
    
    lkv_E = lkv[0]
    lkv_N = lkv[1]
    lkv_U = lkv[2]
    fa = np.zeros_like(lkv_E)
    ia = np.zeros_like(lkv_E)
    
    for i in range(lkv_E.shape[0]):  # Loop over rows
        for j in range(lkv_E.shape[1]):  # Loop over columns
            flight_angle, incidence_angle = ivf.look_vector2flight_incidence_angles(lkv_E[i, j], 
                                                                                    lkv_N[i, j], 
                                                                                    lkv_U[i, j])
            fa[i, j] = flight_angle
            ia[i, j] = incidence_angle

    return fa, ia
    

def velocity_projection(track1_structure, track2_structure, azimuth = None, decimal = 3):
    '''
    Calculate any azimuth projection velocity and vertical velocity from common pixels of 2 different satellite tracks
    Input: 
        track1: read out hdf5 structure of the first track 
        track2: read out hdf5 structure of the second track
        azimuth: projected azimuth heading in degree
        decimal: geographic decimal place (default = 3) 
    Output: 
       2D array of azimuth projected velocity,  2D array of vertical velocity
       
    '''
    
    Ue_azi = np.cos(math.radians(azimuth))
    Un_azi = np.sin(math.radians(azimuth))

    #find common velocity, flight angle and incidence angle
    v_1_common, v_2_common = common_velocity(track1_structure, track2_structure, decimal = decimal)
    lkv_1_common, lkv_2_common = common_lkv(track1_structure, track2_structure, decimal = decimal)
    fa_1, ia_1 = flight_and_incidence_angles(lkv_1_common)
    fa_2, ia_2 = flight_and_incidence_angles(lkv_2_common)

    #Create empty array
    vert_v = np.zeros_like(fa_1)  # For flight angles
    azi_v = np.zeros_like(fa_1)  # For incidence angles
    los_1 = np.zeros_like(fa_1)
    los_2 = np.zeros_like(fa_1)
    
    for i in range(fa_1.shape[0]):  # Loop over rows
        for j in range(fa_1.shape[1]):  # Loop over columns
          #def3D_into_LOS(U_e, U_n, U_u, flight_angle, incidence_angle, look_direction='right')
            
            #test data
            los_1_vert = ivf.def3D_into_LOS(0, 0, 1, fa_1[i,j], ia_1[i,j])          #all_vertical
            los_1_azi = ivf.def3D_into_LOS(Ue_azi, Un_azi, 0, fa_1[i,j], ia_1[i,j])  #all_azi
            
            los_2_vert = ivf.def3D_into_LOS(0, 0, 1, fa_2[i,j], ia_2[i,j])  #u_vertical
            los_2_azi = ivf.def3D_into_LOS(Ue_azi, Un_azi, 0, fa_2[i,j], ia_2[i,j]) #_fault
            
            #G matrix
            G = np.array([[los_1_vert, los_1_azi],
                          [los_2_vert, los_2_azi]])
            
            #Actual los data
            los_1 = v_1_common[i,j]
            los_2 = v_2_common[i,j]
            
            # d matrix
            d = np.array([[los_1],
                         [los_2]])
            
            # solve for m
            m = np.dot(np.linalg.inv(G),d) # m = invert G * d
    
            vert_v[i,j] = m[0][0]
            azi_v[i,j] = m[1][0]

    return  azi_v, vert_v

def grid_spacing(input_grid):
    spacing = float(pygmt.grdinfo(input_grid).split('\n')[5].split()[6])
    return spacing

def xyz_to_grid(data_frame, spacing):
    '''
    pyGMT xyz2grd with automated calculate spacing and region
    Input:
        data frame with column x, y, z 
    
    '''
    data_frame.columns = ['lon','lat','z']
    region = [np.min(data_frame['lon']), np.max(data_frame['lon']), np.min(data_frame['lat']), np.max(data_frame['lat'])]
    grid = pygmt.xyz2grd(x = data_frame['lon'].values, y = data_frame['lat'].values, z = data_frame['z'].values, 
                                  spacing = spacing,
                                  region= region)
    return grid


def grid_resample(input_grid, lon_space = None, lat_space = None):
    '''
    Resample grid to specific spacing
    Input: 
        input_grid: original grid file path or grid data
        lon_space: new longitude spacing
        lat_space: new latitude spacing
    Output: 
       Resampled grid with new spacing 
    '''
    new_grid = pygmt.grdsample(grid = input_grid, translate = True, spacing = [lon_space , lat_space])
    return new_grid
    

def grid_normalize(input_grid, ref_lon, ref_lat, spacing,output_type = 'grid', invert = False):
    '''
    Normalize grid pixel value with specific reference pixel value
    Input:
        input_grid: original grid file path or grid data
        ref_lon: langitude of the reference pixel
        ref_lat: latitude of the reference pixel
        output_type: 'pandas' or 'grid'
        invert: invert z value to negative
    Output: 
        Normalize data frame or grid
        
    '''
    
    data = pygmt.grd2xyz(grid = input_grid, output_type='pandas')
    data.columns = ['lon','lat','z']

    #round data
    data['lon'] = np.round(data['lon'], 3)
    data['lat'] = np.round(data['lat'], 3)

    #find ref pixel
    ref_data = data[data['lon'] == ref_lon]
    ref_data = ref_data[ref_data['lat'] == ref_lat]
    ref_value = ref_data['z'].values[0]
    print(f'Reference pixel value is {ref_value}')

    #Normalize 
    norm_df = data
    norm_df['z'] = data['z'] - ref_value

    if invert == True:
        norm_df['z'] = norm_df['z'].values * -1
    
    if output_type == 'grid':
        norm_grid = xyz_to_grid(norm_df, spacing = spacing)
        return norm_grid
        
    if output_type == 'pandas':
        return norm_df


def grid_common(grid_1, grid_2, grid1_space = None, grid2_space = None, full = True, output_type = 'grid'):
    '''
    Find the common pixels from 2 grids.
    Input:
        grid_1: First grid file path or grid data
        grid_2: Second grid file path or grid data
            both grids should have the same spacing or need to resample to specific spacing
        resample: resample both grids to new spacing
        grid1_space: longitude spacing for resampling in degree
        grid2_space: latitude spacing for resampling in degree
        output_type: 'pandas' or 'grid'
    Output: 
        grid of data frame with common pixels location
        
    '''     
    data1 = pygmt.grd2xyz(grid = grid_1, output_type='pandas')
    data2 = pygmt.grd2xyz(grid = grid_2, output_type='pandas')
    data1.columns = ['lon','lat','z']
    data2.columns = ['lon','lat','z']

    #round data
    data1['lon'] = np.round(data1['lon'], 3)
    data1['lat'] = np.round(data1['lat'], 3)
    data2['lon'] = np.round(data2['lon'], 3)
    data2['lat'] = np.round(data2['lat'], 3)
   
    # Step 1: Find common longitude and latitude
    common_lon = np.intersect1d(data1['lon'], data2['lon'])
    common_lat = np.intersect1d(data1['lat'], data2['lat'])
    
    
    data1_filtered = data1[(data1['lon'].isin(common_lon)) & (data1['lat'].isin(common_lat))]
    data2_filtered = data2[(data2['lon'].isin(common_lon)) & (data2['lat'].isin(common_lat))]
    
    filtered_region = [np.min(common_lon), np.max(common_lon), np.min(common_lat), np.max(common_lat)]

    if full == False:
        data1_filtered['z'] =  data1_filtered['z'].values *  data2_filtered['z'].values / data2_filtered['z'].values
        data2_filtered['z'] =  data2_filtered['z'].values *  data1_filtered['z'].values / data1_filtered['z'].values

    if output_type == 'grid':
        grid_1 = xyz_to_grid(data1_filtered, grid1_space)
        grid_2 = xyz_to_grid(data2_filtered, grid2_space)
        return grid_1, grid_2

    if output_type == 'pandas':
        return data1_filtered, data2_filtered


def grid_difference(grid_1, grid_2, spacing = None, output_type = 'grid'):
    '''
    Calculate pixel value (z) difference at common pixels from 2 grids.
    Input:
        grid_1: First grid file path or grid data
        grid_2: Second grid file path or grid data
            both grids should have the same spacing or need to resample to specific spacing
        spacing: spacing of grid in degree
        output_type: 'pandas' or 'grid'
    Output: 
        grid of data frame of difference of z pixel values at common pixels location
        
    '''
    data1, data2 = grid_common(grid_1, grid_2, output_type = 'pandas')

    difference_z = data2['z'].values - data1['z'].values
    difference = {'lon': data1['lon'].values, 'lat': data1['lat'].values, 'z': difference_z }
    difference_df = pd.DataFrame(difference)

    if output_type == 'grid':
        difference_grid = xyz_to_grid(difference_df, spacing)
        return difference_grid
    if output_type == 'pandas':
        return difference_df
    


