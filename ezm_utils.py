import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from dlc_utils import angle_between,get_all_videos_in_session,get_analysis_dfs,load_session_details,get_subject_from_session

def get_quadrant(df,bodypart='centroid',p_cutoff = 0.95,strategy='remove_uncertain',closed_corner = 'ZM_tl_in',open_corner='ZM_tr_in',time_filter=None,frame_rate=20.0):
    """
        moving clockwise around the EZM, when we pass the closed_corner, we 
        reach the closed_section which opens up again when we pass the 
        open_corner.
        
        outputs 0 if closed; 1 if open
    """
    resnet_name = df.keys()[0][0]
    
    X = df[resnet_name,bodypart,'x']
    Y = df[resnet_name,bodypart,'y']
    p = df[resnet_name,bodypart,'likelihood']
    good_loc = p>p_cutoff
    
    if time_filter:
        X = X[:int(time_filter*frame_rate)] 
        Y = Y[:int(time_filter*frame_rate)] 
        p = p[:int(time_filter*frame_rate)] 
        good_loc = good_loc[:int(time_filter*frame_rate)] 
    
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
            location[idx] = 0
        elif angle_from_cc>90 and angle_from_oc<=90: #Q2
            location[idx] = 1
        elif angle_from_cc>90 and angle_from_oc>90: #Q3
            location[idx] = 0
        elif angle_from_cc<=90 and angle_from_oc>90: #Q4
            location[idx] = 1
    body_loc = (X,Y)
    return location, body_loc, zm_center, open_coord, closed_coord

def show_analysis(session=r'Y:\Data 2018-2019\Complement 4 - schizophrenia Project\2019 Adult Behavior C4_for revisions\EZM\ALC_060519_2_mC4\ALC_060519_2_57G_EZM',output=r"C:\Users\User\Desktop\Code\annotated_SI_part.avi",
    speedX=10,):
    import cv2
    import tqdm
    import collections
    import sys

    vids = get_all_videos_in_session(session)
    df = get_analysis_dfs(session,vids)
    location, body_loc, zm_center, open_coord, closed_coord = get_quadrant(df)


    try:
        curr_filename = os.path.join(session,vids[0]+'.avi')
        cap = cv2.VideoCapture(curr_filename)
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        writer = cv2.VideoWriter(output,fourcc, 1/cap.get(2), 
                                 (int(cap.get(3)),int(cap.get(4))))
        for idx,row in tqdm.tqdm(df.iterrows(),total=df.shape[0]):
            if not idx%int(speed)==0:continue
            if row['source'][0] in curr_filename:
                # cv is open to the right place
                pass
            else:
                cap.release()
                curr_filename = os.path.join(session,row['source'][0]+'.avi')
                cap=cv2.VideoCapture(curr_filename)
            # get the frame
            cap.set(1,row['frame_number'][0]) # 1==set frame number
            succ,img = cap.read()
            
            # draw the object circles
            img = cv2.circle(img,(int(zm_center[0]),int(zm_center[1])),5,(0,0,255))
            img = cv2.circle(img,(int(open_coord[0]),int(open_coord[1])),5,(0,0,255))
            img = cv2.circle(img,(int(closed_coord[0]),int(closed_coord[1])),5,(0,0,255))
            
            if not np.isnan(location[idx]):
                if location[idx]==1:
                    img = cv2.circle(img,(int(body_loc[0][idx]),int(body_loc[1][idx])),5,(255,0,0))
                else:
                    img = cv2.circle(img,(int(body_loc[0][idx]),int(body_loc[1][idx])),5,(0,255,0))
            
            # add to writer
            writer.write(img)
    except Exception as er:

        raise er.with_traceback(sys.exc_info()[2])
    finally:
        # for index, row in df.iterrows():
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
        
def run_open_probability_for_all_data():
    details = load_session_details()
    details['EZM_open_probability'] = np.nan
    
    # OF
    base_folder = r'Y:\Data 2018-2019\Complement 4 - schizophrenia Project\2019 Adult Behavior C4_for revisions\EZM'
    litters = ['ALC_050319_1_control','ALC_050319_2_mC4','ALC_051719_1_control','ALC_051719_2_mC4','ALC_060519_1_control','ALC_060519_2_mC4','ALC_070519_1_control'] # 'ALC_070519_1_control'
    
    for litter in litters:
         # get the subject folders
         sessions = os.listdir(os.path.join(base_folder,litter))
         for session in sessions:
             sub = get_subject_from_session(session)
             sess_folder = os.path.join(base_folder,litter,session)
             vids = get_all_videos_in_session(sess_folder)
             df = get_analysis_dfs(sess_folder,vids)
             q_loc, body_loc, zm_center, open_coord, closed_coord = get_quadrant(df,time_filter=300) #### change time_filter=None if you dont want a time filter
             
             details.loc[sub,'EZM_open_probability'] = np.nansum(q_loc)/np.count_nonzero(~np.isnan(q_loc))

             print(litter,':',session)
    details.to_csv(r'C:\Users\User\Desktop\Code\dlc_utils\EZM_open_probability.csv')   