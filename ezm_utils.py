import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from dlc_utils import angle_between

def get_quadrant(df,bodypart='centroid',p_cutoff = 0.95,strategy='remove_uncertain',closed_corner = 'ZM_tl_in',open_corner='ZM_tr_in'):
    """
        moving clockwise around the EZM, when we pass the closed_corner, we 
        reach the closed_section which opens up again when we pass the 
        open_corner.
    """
    resnet_name = df.keys()[0][0]
    
    X = df[resnet_name,bodypart,'x']
    Y = df[resnet_name,bodypart,'y']
    p = df[resnet_name,bodypart,'likelihood']
    good_loc = p>p_cutoff
    
    closed_coord = np.asarray([df[resnet_name,closed_corner,'x'].mean(),df[resnet_name,closed_corner,'y'].mean()])
    open_coord = np.asarray([df[resnet_name,open_corner,'x'].mean(),df[resnet_name,open_corner,'y'].mean()])    
    zm_center = np.asarray([df[resnet_name,'ZM_center','x'].mean(),df[resnet_name,'ZM_center','y'].mean()])
    
    location = np.full_like(X,np.nan)
    
    # loop through the locations
    for idx,(x,y) in enumerate(zip(X,Y)):
        if not good_loc[idx]: continue
        
        angle_from_cc = angle_between(zm_center-closed_coord,zm_center-np.asarray([x,y]))
        angle_from_oc = angle_between(zm_center-open_coord,zm_center-np.asarray([x,y]))
        if angle_from_cc<=90 and angle_from_oc<=90: #Q1
            location[idx] = 1
        elif angle_from_cc>90 and angle_from_oc<=90: #Q2
            location[idx] = 0
        elif angle_from_cc>90 and angle_from_oc>90: #Q3
            location[idx] = 1
        elif angle_from_cc<=90 and angle_from_oc>90: #Q4
            location[idx] = 0
    body_loc = (X,Y)
    return location, body_loc, zm_center, open_coord, closed_coord