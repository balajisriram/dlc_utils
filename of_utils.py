import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from dlc_utils import get_all_videos_in_session,get_analysis_dfs,euclidean_dist,is_odd,load_session_details,get_subject_from_session
from scipy.ndimage import median_filter as medfilt

def get_animal_speed(df,bodypart='centroid',p_cutoff=0.95,strategy='raw',frame_rate=20.0,arena_s1ze_in_mm=450,time_filter=None):
    resnet_name = df.keys()[0][0]
    
    t_l_corner = np.asarray([df[resnet_name,'box_tl','x'].mean(),-df[resnet_name,'box_tl','y'].mean()])
    t_r_corner = np.asarray([df[resnet_name,'box_tr','x'].mean(),-df[resnet_name,'box_tl','y'].mean()])
    arena_length_in_pix = euclidean_dist(t_l_corner,t_r_corner)
    
    centroid_x = df[(resnet_name,'centroid','x')]
    centroid_y = df[(resnet_name,'centroid','y')]
    centroid_p = df[(resnet_name,'centroid','likelihood')]
    
    good_data = (centroid_p>p_cutoff)
    
    # remove data from the initial section - there is no animal and estimates are terrible
    # criterion is atleast 1 second of reliable data 
    remove_until = 0
    for i in range(1,len(good_data)):
        if np.any(good_data[:i]) and np.all(good_data[i:i+int(frame_rate)]):
            remove_until=i
            break
            
    body_location =  [(k,l) for k,l in zip(centroid_x,centroid_y)]
    body_location[:remove_until] = np.full((remove_until,1),np.nan)
    body_speed = np.full_like(body_location,np.nan)
    for i in range(remove_until+1,len(good_data)):
        body_speed[i] = euclidean_dist(body_location[i-1],body_location[i])
    body_speed = body_speed.astype(np.float64)
    if time_filter:
        n_to_keep = int(frame_rate*time_filter)
        body_speed = body_speed[:n_to_keep]
    if strategy=='raw':
        # do nothing to the distance
        pass
    elif strategy=='raw+outlier_removal':
        # do nothing to the distance
        body_speed[body_speed>np.nanquantile(body_speed,0.999)] = np.nan
    elif strategy=='median_filter':
        # median filter
        filter_duration = 0.1 # 200 ms
        filter_dur_in_frames = int(filter_duration*frame_rate)
        if not is_odd(filter_dur_in_frames):filter_dur_in_frames = filter_dur_in_frames+1
        body_speed = medfilt(body_speed,size=filter_dur_in_frames)
    elif strategy=='median_filter+outlier_removal':
        # median filter
        filter_duration = 0.1 # 200 ms
        filter_dur_in_frames = int(filter_duration*frame_rate)
        if not is_odd(filter_dur_in_frames):filter_dur_in_frames = filter_dur_in_frames+1
        body_speed = medfilt(body_speed,size=filter_dur_in_frames)
        body_speed[body_speed>np.nanquantile(body_speed,0.999)] = np.nan
    # body_speed is in delta(pix)/frame. Convert to delta(mm)/s        
    body_speed = body_speed*(arena_s1ze_in_mm/arena_length_in_pix)*frame_rate
    return body_speed


def run_body_speed_for_all_data():
    details = load_session_details()
    details['OF_body_speed_max'] = np.nan
    details['OF_body_speed_mean'] = np.nan
    details['OF_body_distance'] = np.nan
                 
    # OF
    base_folder = r'Y:\Data 2018-2019\Complement 4 - schizophrenia Project\2019 Adult Behavior C4_for revisions\OF'
    litters = ['ALC_050319_1_control','ALC_050319_2_mC4','ALC_051719_1_control','ALC_051719_2_mC4','ALC_060519_1_control','ALC_060519_2_mC4','ALC_070519_1_control'] # 'ALC_060519_1_control','ALC_060519_2_mC4',
    
    for litter in litters:
         # get the subject folders
         sessions = os.listdir(os.path.join(base_folder,litter))
         for session in sessions:
             sub = get_subject_from_session(session)
             sess_folder = os.path.join(base_folder,litter,session)
             vids = get_all_videos_in_session(sess_folder)
             df = get_analysis_dfs(sess_folder,vids)
             bs = get_animal_speed(df,time_filter=300)
             
             details.loc[sub,'OF_body_speed_max'] = np.nanmax(bs)/1000.
             details.loc[sub,'OF_body_speed_mean'] = np.nanmean(bs)/1000.
             details.loc[sub,'OF_body_distance'] = np.nansum(bs/20.)/1000.

             print(litter,':',session)
    details.to_csv(r'C:\Users\User\Desktop\Code\dlc_utils\OF_speed_distance.csv')       
    
    
if __name__=='__main__':
    run_body_speed_for_all_data()
    # sess = r'Y:\Data 2018-2019\Complement 4 - schizophrenia Project\2019 Adult Behavior C4_for revisions\NOR\ALC_050319_1_control\ALC_050319_1_41C_NOR\ALC_050319_1_41C_NOR_Trial1'
    # vids = get_all_videos_in_session(sess)
    # df = get_analysis_dfs(sess,vids)
    # fig,ax = plt.subplots(5,1,sharex=True,sharey=True)
    
    # bs = get_animal_speed(df,strategy='raw')
    # ax[0].plot(bs);
    # ax[0].set_title('raw')
    # bs1 = bs
    
    # bs = get_animal_speed(df,strategy='raw+outlier_removal')
    # ax[1].plot(bs);
    # ax[1].set_title('raw+outlier')
    
    # bs = get_animal_speed(df,strategy='median_filter')
    # ax[2].plot(bs);
    # ax[2].set_title('median')
    
    # bs = get_animal_speed(df,strategy='median_filter+outlier_removal')
    # ax[3].plot(bs);
    # ax[3].set_title('median+outlier')
    
    # plt.show()
    
    # plt.hist(bs1,35);plt.show()
    
    # import pdb
    # pdb.set_trace()