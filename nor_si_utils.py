import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path

from dlc_utils import get_all_videos_in_session,get_analysis_dfs,plot_pos_vs_frame,load_session_details

def get_object_locations(df,fractional_size=0.33):
    resnet_name = df.keys()[0][0]
    t_l_corner = np.asarray([df[resnet_name,'tl_corner','x'].mean(),-df[resnet_name,'tl_corner','y'].mean()])
    t_r_corner = np.asarray([df[resnet_name,'tr_corner','x'].mean(),-df[resnet_name,'tr_corner','y'].mean()])
    b_l_corner = np.asarray([df[resnet_name,'bl_corner','x'].mean(),-df[resnet_name,'bl_corner','y'].mean()])
    b_r_corner = np.asarray([df[resnet_name,'br_corner','x'].mean(),-df[resnet_name,'br_corner','y'].mean()])
    
    d_top = t_r_corner-t_l_corner
    d_bottom = b_r_corner-b_l_corner
    d_left = t_l_corner-b_l_corner
    d_right = t_r_corner-b_r_corner
    
    bl = b_l_corner+fractional_size*d_bottom/2+fractional_size*d_left/2
    tr = t_r_corner-fractional_size*d_top/2-fractional_size*d_right/2
    bl = (int(bl[0]),-int(bl[1]))
    tr = (int(tr[0]),-int(tr[1]))
    return bl,tr
    

def get_occupancy(df,bodypart='centroid',p_cutoff = 0.95,strategy='remove_uncertain',time_filter = None,exclude_out_of_box = False,plot_on=True,fractional_size = 0.33):
    resnet_name = df.keys()[0][0]
    x = df[resnet_name,bodypart,'x']
    y = df[resnet_name,bodypart,'y']
    p = df[resnet_name,bodypart,'likelihood']
    
    if plot_on: ax = plt.subplot()
    # filter by time
    if not time_filter:
        pass
    else:
        frames_to_count = time_filter*20
        x = x[:frames_to_count]
        y = y[:frames_to_count]
        p = p[:frames_to_count]
    
    if strategy=='remove_uncertain':
        good_x = x[p>p_cutoff]
        good_y = y[p>p_cutoff]
    else:
        ValueError('unknown strategy')
    if plot_on: ax.plot(good_x,-good_y,'k.',alpha=0.1)

    # plot the corners
    spotx = (resnet_name,'tl_corner','x'); spoty = (resnet_name,'tl_corner','y');el_tl = patches.Ellipse((df[spotx].mean(),-df[spoty].mean()),width=2*df[spotx].std(),height=2*df[spoty].std(),color=(0,0,0))
    spotx = (resnet_name,'tr_corner','x'); spoty = (resnet_name,'tr_corner','y');el_tr = patches.Ellipse((df[spotx].mean(),-df[spoty].mean()),width=2*df[spotx].std(),height=2*df[spoty].std(),color=(0,0,0))
    spotx = (resnet_name,'bl_corner','x'); spoty = (resnet_name,'bl_corner','y');el_bl = patches.Ellipse((df[spotx].mean(),-df[spoty].mean()),width=2*df[spotx].std(),height=2*df[spoty].std(),color=(0,0,0))
    spotx = (resnet_name,'br_corner','x'); spoty = (resnet_name,'br_corner','y');el_br = patches.Ellipse((df[spotx].mean(),-df[spoty].mean()),width=2*df[spotx].std(),height=2*df[spoty].std(),color=(0,0,0))
    if plot_on: ax.add_patch(el_tl)
    if plot_on: ax.add_patch(el_tr)
    if plot_on: ax.add_patch(el_bl)
    if plot_on: ax.add_patch(el_br)
    
    # now to calculate the squares of intertest
    t_l_corner = np.asarray([df[resnet_name,'tl_corner','x'].mean(),-df[resnet_name,'tl_corner','y'].mean()])
    t_r_corner = np.asarray([df[resnet_name,'tr_corner','x'].mean(),-df[resnet_name,'tr_corner','y'].mean()])
    b_l_corner = np.asarray([df[resnet_name,'bl_corner','x'].mean(),-df[resnet_name,'bl_corner','y'].mean()])
    b_r_corner = np.asarray([df[resnet_name,'br_corner','x'].mean(),-df[resnet_name,'br_corner','y'].mean()])
    
    d_top = t_r_corner-t_l_corner
    d_bottom = b_r_corner-b_l_corner
    d_left = t_l_corner-b_l_corner
    d_right = t_r_corner-b_r_corner
    
    # print(d_top)
    # print(d_bottom)
    # print(d_left)
    # print(d_right)
    
    verts = [b_l_corner,b_l_corner+fractional_size*d_bottom,b_l_corner+fractional_size*d_bottom+fractional_size*d_left,b_l_corner+fractional_size*d_left,b_l_corner]
    codes = [Path.MOVETO,Path.LINETO,Path.LINETO,Path.LINETO,Path.CLOSEPOLY,]
    b_l_square = Path(verts,codes)
    
    verts = [t_r_corner,t_r_corner-fractional_size*d_top,t_r_corner-fractional_size*d_top-fractional_size*d_right,t_r_corner-fractional_size*d_right,t_r_corner]
    codes = [Path.MOVETO,Path.LINETO,Path.LINETO,Path.LINETO,Path.CLOSEPOLY,]
    t_r_square = Path(verts,codes)
    
    verts = [b_l_corner,b_r_corner,t_r_corner,t_l_corner,b_l_corner]
    codes = [Path.MOVETO,Path.LINETO,Path.LINETO,Path.LINETO,Path.CLOSEPOLY,]
    big_square = Path(verts,codes)
    
    b_l_patch = patches.PathPatch(b_l_square,facecolor='orange',alpha=0.2)
    t_r_patch = patches.PathPatch(t_r_square,facecolor='cyan',alpha=0.2)
    big_patch = patches.PathPatch(big_square,facecolor='grey',alpha=0.05)
    if plot_on: ax.add_patch(b_l_patch)
    if plot_on: ax.add_patch(t_r_patch)
    if plot_on: ax.add_patch(big_patch)
    
    # check which points are in each paths
    points = list(zip(good_x,-good_y))
    pts_in_bl = b_l_square.contains_points(points)
    pts_in_tr = t_r_square.contains_points(points)
    pts_in_big = big_square.contains_points(points)
    if plot_on: ax.plot(good_x[pts_in_bl],-good_y[pts_in_bl],color='orange',marker='.',alpha=0.1)
    if plot_on: ax.plot(good_x[pts_in_tr],-good_y[pts_in_tr],color='cyan',marker='.',alpha=0.1)

    # calculate occupancy
    bl_occupancy = good_x[pts_in_bl].size/good_x[pts_in_big].size
    tr_occupancy = good_x[pts_in_tr].size/good_x[pts_in_big].size
    
    # if plot_on:
        # bl_ext = b_l_square.get_extents
        # tr_ext = b_l_square.get_extents
        # print(bl_ext)
        # plt.text((bl_ext[0]+bl_ext[1])/2,(bl_ext[2]+bl_ext[3])/2,'{0}'.format(bl_occupancy),fontsize=12)
        # plt.text((tr_ext[0]+tr_ext[1])/2,(tr_ext[2]+tr_ext[3])/2,'{0}'.format(tr_occupancy),fontsize=12)
    if plot_on: ax.axis('equal')
    if plot_on: plt.show()
    return bl_occupancy,tr_occupancy

def get_subject_from_session(sess):
    splits = sess.split('_')
    sub = splits[0]
    for split in splits[1:-1]:
        sub = sub+'_'
        sub = sub+split
    return sub

def get_latency(df,bodypart='centroid',p_cutoff=0.95,strategy='set_to_nan',time_filter = None,exclude_out_of_box = False,plot_on=True,fractional_size = 0.33):
    resnet_name = df.keys()[0][0]
    x = df[resnet_name,bodypart,'x']
    y = df[resnet_name,bodypart,'y']
    p = df[resnet_name,bodypart,'likelihood']
    
    if plot_on: ax = plt.subplot()
    # filter by time
    if not time_filter:
        pass
    else:
        frames_to_count = time_filter*20
        x = x[:frames_to_count]
        y = y[:frames_to_count]
        p = p[:frames_to_count]
    
    if strategy=='set_to_nan':
        x[p<p_cutoff]=np.nan
        y[p<p_cutoff]=np.nan
    else:
        ValueError('unknown strategy')
        
    # plot the corners
    spotx = (resnet_name,'tl_corner','x'); spoty = (resnet_name,'tl_corner','y');el_tl = patches.Ellipse((df[spotx].mean(),-df[spoty].mean()),width=2*df[spotx].std(),height=2*df[spoty].std(),color=(0,0,0))
    spotx = (resnet_name,'tr_corner','x'); spoty = (resnet_name,'tr_corner','y');el_tr = patches.Ellipse((df[spotx].mean(),-df[spoty].mean()),width=2*df[spotx].std(),height=2*df[spoty].std(),color=(0,0,0))
    spotx = (resnet_name,'bl_corner','x'); spoty = (resnet_name,'bl_corner','y');el_bl = patches.Ellipse((df[spotx].mean(),-df[spoty].mean()),width=2*df[spotx].std(),height=2*df[spoty].std(),color=(0,0,0))
    spotx = (resnet_name,'br_corner','x'); spoty = (resnet_name,'br_corner','y');el_br = patches.Ellipse((df[spotx].mean(),-df[spoty].mean()),width=2*df[spotx].std(),height=2*df[spoty].std(),color=(0,0,0))
    if plot_on: ax.add_patch(el_tl)
    if plot_on: ax.add_patch(el_tr)
    if plot_on: ax.add_patch(el_bl)
    if plot_on: ax.add_patch(el_br)
    
    # now to calculate the squares of intertest
    t_l_corner = np.asarray([df[resnet_name,'tl_corner','x'].mean(),-df[resnet_name,'tl_corner','y'].mean()])
    t_r_corner = np.asarray([df[resnet_name,'tr_corner','x'].mean(),-df[resnet_name,'tr_corner','y'].mean()])
    b_l_corner = np.asarray([df[resnet_name,'bl_corner','x'].mean(),-df[resnet_name,'bl_corner','y'].mean()])
    b_r_corner = np.asarray([df[resnet_name,'br_corner','x'].mean(),-df[resnet_name,'br_corner','y'].mean()])
    
    d_top = t_r_corner-t_l_corner
    d_bottom = b_r_corner-b_l_corner
    d_left = t_l_corner-b_l_corner
    d_right = t_r_corner-b_r_corner
    
    verts = [b_l_corner,b_l_corner+fractional_size*d_bottom,b_l_corner+fractional_size*d_bottom+fractional_size*d_left,b_l_corner+fractional_size*d_left,b_l_corner]
    codes = [Path.MOVETO,Path.LINETO,Path.LINETO,Path.LINETO,Path.CLOSEPOLY,]
    b_l_square = Path(verts,codes)
    
    verts = [t_r_corner,t_r_corner-fractional_size*d_top,t_r_corner-fractional_size*d_top-fractional_size*d_right,t_r_corner-fractional_size*d_right,t_r_corner]
    codes = [Path.MOVETO,Path.LINETO,Path.LINETO,Path.LINETO,Path.CLOSEPOLY,]
    t_r_square = Path(verts,codes)
    
    verts = [b_l_corner,b_r_corner,t_r_corner,t_l_corner,b_l_corner]
    codes = [Path.MOVETO,Path.LINETO,Path.LINETO,Path.LINETO,Path.CLOSEPOLY,]
    big_square = Path(verts,codes)
    
    b_l_patch = patches.PathPatch(b_l_square,facecolor='orange',alpha=0.2)
    t_r_patch = patches.PathPatch(t_r_square,facecolor='cyan',alpha=0.2)
    big_patch = patches.PathPatch(big_square,facecolor='grey',alpha=0.05)
    if plot_on: ax.add_patch(b_l_patch)
    if plot_on: ax.add_patch(t_r_patch)
    if plot_on: ax.add_patch(big_patch)
    
    # check which points are in each paths
    points = list(zip(x,-y))
    pts_in_bl = b_l_square.contains_points(points)
    pts_in_tr = t_r_square.contains_points(points)
    pts_in_big = big_square.contains_points(points)
    
    idx_bl = np.where(pts_in_bl==True)[0][0]
    idx_tr = np.where(pts_in_tr==True)[0][0]
    if idx_tr<idx_bl:
        choice = 'tr'
    else:
        choice= 'bl'
        
    latency_tr = idx_tr/20.0
    latency_bl = idx_bl/20.0
    
    return choice,latency_bl,latency_tr


def run_occupancy_for_all_data():
    details = load_session_details()
    details['NOR_T1_object1_occupancy'] = np.nan
    details['NOR_T1_nonobject_occupancy'] = np.nan
    details['NOR_T2_object1_occupancy'] = np.nan
    details['NOR_T2_object2_occupancy'] = np.nan
    details['SI_T1_subject1_occupancy'] = np.nan
    details['SI_T1_nonsubject_occupancy'] = np.nan
    details['SI_T2_subject1_occupancy'] = np.nan
    details['SI_T2_subject2_occupancy'] = np.nan
    
    # NOR
    base_folder = r'Y:\Data 2018-2019\Complement 4 - schizophrenia Project\2019 Adult Behavior C4_for revisions\NOR'
    litters = ['ALC_050319_1_control','ALC_050319_2_mC4','ALC_051719_1_control','ALC_051719_2_mC4','ALC_060519_1_control','ALC_060519_2_mC4','ALC_070519_1_control']
    
    for litter in litters:
         # get the subject folders
         sessions = os.listdir(os.path.join(base_folder,litter))
         for session in sessions:
             sub = get_subject_from_session(session)
             trials = os.listdir(os.path.join(base_folder,litter,session))
             for trial in trials:
                 sess_folder = os.path.join(base_folder,litter,session,trial)
                 vids = get_all_videos_in_session(sess_folder)
                 df = get_analysis_dfs(sess_folder,vids)
                 bl,tr = get_occupancy(df,plot_on=False,time_filter=300)
                 
                 # now comes the annoying if then statements
                 
                 # trial1 or trial 2?
                 if 'Trial1' in trial or 'TRIAL1' in trial or 'trial1' in trial:
                     if details.loc[sub,'NOR_obj1']=='bl':
                         #This is trial 1 and the object is in bl
                         o_pref = bl
                         o_nonpref = tr
                         details.loc[sub,'NOR_T1_object1_occupancy'] = bl
                         details.loc[sub,'NOR_T1_nonobject_occupancy'] = tr
                     else:
                         #This is trial 1 and the object is in tr
                         o_pref = tr
                         o_nonpref = bl
                         details.loc[sub,'NOR_T1_object1_occupancy'] = tr
                         details.loc[sub,'NOR_T1_nonobject_occupancy'] = bl
                 elif 'Trial2' in trial or 'TRIAL2' in trial or 'trial2' in trial:
                     if details.loc[sub,'NOR_obj1']=='bl':
                         #This is trial 2 and object1 was in bl in trial1 and 2
                         o_pref = bl
                         o_nonpref = tr
                         details.loc[sub,'NOR_T2_object1_occupancy'] = bl
                         details.loc[sub,'NOR_T2_object2_occupancy'] = tr
                     else:
                         #This is trial 2 and object1 was in tr in trial1 and 2
                         o_pref = tr
                         o_nonpref = bl
                         details.loc[sub,'NOR_T2_object1_occupancy'] = tr
                         details.loc[sub,'NOR_T2_object2_occupancy'] = bl
                 else:
                     print(sess_folder)
                     ValueError('why am I here?')
                 print(litter,':',session,':',trial,':opref:',o_pref,':ononpref:',o_nonpref)
                 
    # SI
    base_folder = r'Y:\Data 2018-2019\Complement 4 - schizophrenia Project\2019 Adult Behavior C4_for revisions\SocialInteraction'
    litters = ['ALC_050319_1_control','ALC_050319_2_mC4','ALC_051719_1_control','ALC_051719_2_mC4','ALC_060519_1_control','ALC_060519_2_mC4','ALC_070519_1_control']
    
    for litter in litters:
         # get the subject folders
         sessions = os.listdir(os.path.join(base_folder,litter))
         for session in sessions:
             sub = get_subject_from_session(session)
             trials = os.listdir(os.path.join(base_folder,litter,session))
             for trial in trials:
                 sess_folder = os.path.join(base_folder,litter,session,trial)
                 vids = get_all_videos_in_session(sess_folder)
                 df = get_analysis_dfs(sess_folder,vids)
                 bl,tr = get_occupancy(df,plot_on=False,fractional_size=0.36,time_filter=300)
                 
                 # now comes the annoying if then statements
                 
                 # trial1 or trial 2?
                 if 'Trial1' in trial or 'TRIAL1' in trial:
                     if details.loc[sub,'SI_mouse1']=='bl':
                         #This is trial 1 and the subject is in bl
                         o_pref = bl
                         o_nonpref = tr
                         details.loc[sub,'SI_T1_subject1_occupancy'] = bl
                         details.loc[sub,'SI_T1_nonsubject_occupancy'] = tr
                     else:
                         #This is trial 1 and the object is in tr
                         o_pref = tr
                         o_nonpref = bl
                         details.loc[sub,'SI_T1_subject1_occupancy'] = tr
                         details.loc[sub,'SI_T1_nonsubject_occupancy'] = bl
                 elif 'Trial2' in trial or 'TRIAL2' in trial:
                     if details.loc[sub,'SI_mouse1']=='bl':
                         #This is trial 2 and object1 was in bl in trial1 and 2
                         o_pref = bl
                         o_nonpref = tr
                         details.loc[sub,'SI_T2_subject1_occupancy'] = bl
                         details.loc[sub,'SI_T2_subject2_occupancy'] = tr
                     else:
                         #This is trial 2 and object1 was in tr in trial1 and 2
                         o_pref = tr
                         o_nonpref = bl
                         details.loc[sub,'SI_T2_subject1_occupancy'] = tr
                         details.loc[sub,'SI_T2_subject2_occupancy'] = bl
                 else:
                     print(sess_folder)
                     ValueError('why am I here?')
                 print(litter,':',session,':',trial,':opref:',o_pref,':ononpref:',o_nonpref)
    details.to_csv(r'C:\Users\User\Desktop\Code\C4_behavior_utils\SIandNOR_details_NOR_SI_150s.csv')

def run_latency_for_all_data():
    details = load_session_details()
    details['NOR_T1_choice'] = np.nan
    details['NOR_T1_object1_latency'] = np.nan
    details['NOR_T1_nonobject_latency'] = np.nan
    details['NOR_T2_choice'] = np.nan
    details['NOR_T2_object1_latency'] = np.nan
    details['NOR_T2_object2_latency'] = np.nan
    details['SI_T1_choice'] = np.nan
    details['SI_T1_subject1_latency'] = np.nan
    details['SI_T1_nonsubject_latency'] = np.nan
    details['SI_T2_choice'] = np.nan
    details['SI_T2_subject1_latency'] = np.nan
    details['SI_T2_subject2_latency'] = np.nan
    
    # NOR
    base_folder = r'Y:\Data 2018-2019\Complement 4 - schizophrenia Project\2019 Adult Behavior C4_for revisions\NOR'
    litters = ['ALC_050319_1_control','ALC_050319_2_mC4','ALC_051719_1_control','ALC_051719_2_mC4','ALC_060519_1_control','ALC_060519_2_mC4','ALC_070519_1_control']
    
    for litter in litters:
         # get the subject folders
         sessions = os.listdir(os.path.join(base_folder,litter))
         for session in sessions:
             sub = get_subject_from_session(session)
             trials = os.listdir(os.path.join(base_folder,litter,session))
             for trial in trials:
                 sess_folder = os.path.join(base_folder,litter,session,trial)
                 vids = get_all_videos_in_session(sess_folder)
                 df = get_analysis_dfs(sess_folder,vids)
                 choice,bl,tr = get_latency(df,plot_on=False,time_filter=300)
                 
                 # now comes the annoying if then statements
                 
                 # trial1 or trial 2?
                 if 'Trial1' in trial or 'TRIAL1' in trial or 'trial1' in trial:
                     if details.loc[sub,'NOR_obj1']=='bl':
                         #This is trial 1 and the object is in bl
                         o_pref = bl
                         o_nonpref = tr
                         if choice=='bl':details.loc[sub,'NOR_T1_choice'] = 'object'
                         else: details.loc[sub,'NOR_T1_choice'] = 'nonobject'
                         details.loc[sub,'NOR_T1_object1_latency'] = bl
                         details.loc[sub,'NOR_T1_nonobject_latency'] = tr
                     else:
                         #This is trial 1 and the object is in tr
                         o_pref = tr
                         o_nonpref = bl
                         if choice=='bl':details.loc[sub,'NOR_T1_choice'] = 'nonobject'
                         else: details.loc[sub,'NOR_T1_choice'] = 'object'
                         details.loc[sub,'NOR_T1_object1_latency'] = tr
                         details.loc[sub,'NOR_T1_nonobject_latency'] = bl
                 elif 'Trial2' in trial or 'TRIAL2' in trial or 'trial2' in trial:
                     if details.loc[sub,'NOR_obj1']=='bl':
                         #This is trial 2 and object1 was in bl in trial1 and 2
                         o_pref = bl
                         o_nonpref = tr
                         if choice=='bl':details.loc[sub,'NOR_T2_choice'] = 'familiar'
                         else: details.loc[sub,'NOR_T2_choice'] = 'novel'
                         details.loc[sub,'NOR_T2_object1_latency'] = bl
                         details.loc[sub,'NOR_T2_object2_latency'] = tr
                     else:
                         #This is trial 2 and object1 was in tr in trial1 and 2
                         o_pref = tr
                         o_nonpref = bl
                         if choice=='bl':details.loc[sub,'NOR_T2_choice'] = 'novel'
                         else: details.loc[sub,'NOR_T2_choice'] = 'familiar'
                         details.loc[sub,'NOR_T2_object1_latency'] = tr
                         details.loc[sub,'NOR_T2_object2_latency'] = bl
                 else:
                     print(sess_folder)
                     ValueError('why am I here?')
                 print(litter,':',session,':',trial,':opref:',o_pref,':ononpref:',o_nonpref)
                 
    # SI
    base_folder = r'Y:\Data 2018-2019\Complement 4 - schizophrenia Project\2019 Adult Behavior C4_for revisions\SocialInteraction'
    litters = ['ALC_050319_1_control','ALC_050319_2_mC4','ALC_051719_1_control','ALC_051719_2_mC4','ALC_060519_1_control','ALC_060519_2_mC4','ALC_070519_1_control']
    
    for litter in litters:
         # get the subject folders
         sessions = os.listdir(os.path.join(base_folder,litter))
         for session in sessions:
             sub = get_subject_from_session(session)
             trials = os.listdir(os.path.join(base_folder,litter,session))
             trials = [x for x in trials if x !='.DS_Store']
             for trial in trials:
                 sess_folder = os.path.join(base_folder,litter,session,trial)
                 vids = get_all_videos_in_session(sess_folder)
                 df = get_analysis_dfs(sess_folder,vids)
                 choice,bl,tr = get_latency(df,plot_on=False,fractional_size=0.36,time_filter=300)
                 
                 # now comes the annoying if then statements
                 
                 # trial1 or trial 2?
                 if 'Trial1' in trial or 'TRIAL1' in trial:
                     if details.loc[sub,'SI_mouse1']=='bl':
                         #This is trial 1 and the subject is in bl
                         o_pref = bl
                         o_nonpref = tr
                         if choice=='bl':details.loc[sub,'SI_T1_choice'] = 'subject'
                         else: details.loc[sub,'SI_T1_choice'] = 'nonsubject'
                         details.loc[sub,'SI_T1_subject1_latency'] = bl
                         details.loc[sub,'SI_T1_nonsubject_latency'] = tr
                     else:
                         #This is trial 1 and the object is in tr
                         o_pref = tr
                         o_nonpref = bl
                         if choice=='bl':details.loc[sub,'SI_T1_choice'] = 'nonsubject'
                         else: details.loc[sub,'SI_T1_choice'] = 'subject'
                         details.loc[sub,'SI_T1_subject1_latency'] = tr
                         details.loc[sub,'SI_T1_nonsubject_latency'] = bl
                 elif 'Trial2' in trial or 'TRIAL2' in trial:
                     if details.loc[sub,'SI_mouse1']=='bl':
                         #This is trial 2 and object1 was in bl in trial1 and 2
                         o_pref = bl
                         o_nonpref = tr
                         if choice=='bl':details.loc[sub,'SI_T2_choice'] = 'familiar'
                         else: details.loc[sub,'SI_T2_choice'] = 'novel'
                         details.loc[sub,'SI_T2_subject1_latency'] = bl
                         details.loc[sub,'SI_T2_subject2_latency'] = tr
                     else:
                         #This is trial 2 and object1 was in tr in trial1 and 2
                         o_pref = tr
                         o_nonpref = bl
                         if choice=='bl':details.loc[sub,'SI_T2_choice'] = 'novel'
                         else: details.loc[sub,'SI_T2_choice'] = 'familiar'
                         details.loc[sub,'SI_T2_subject1_latency'] = tr
                         details.loc[sub,'SI_T2_subject2_latency'] = bl
                 else:
                     print(sess_folder)
                     ValueError('why am I here?')
                 print(litter,':',session,':',trial,':opref:',o_pref,':ononpref:',o_nonpref)
    details.to_csv(r'C:\Users\User\Desktop\Code\C4_behavior_utils\SIandNOR_details_NOR_SI_latency_300s.csv')
    
    
if __name__=='__main__':
    import cv2
    # get the occupancy
    folder = r'C:\Users\bsriram\Desktop\Data\ACM_Data\AllData\SI_sample'
    vids = get_all_videos_in_session(folder)
    df = get_analysis_dfs(folder,vids)
    plot_pos_vs_frame(df)
    bl,tr = get_occupancy(df,plot_on=True,fractional_size=0.36)
    print('bl',bl)
    print('tr',tr)
    
    # get the latency
    # folder = r'Y:\Data 2018-2019\Complement 4 - schizophrenia Project\2019 Adult Behavior C4_for revisions\NOR\ALC_051719_1_control\ALC_051719_1_45C_NOR\ALC_051719_1_45C_NOR_Trial2'
    # vids = get_all_videos_in_session(folder)
    # df = get_analysis_dfs(folder,vids)
    # bl,tr = get_latency(df,plot_on=True,fractional_size=0.33)
    
    # run_occupancy_for_all_data()
    # run_latency_for_all_data()
    