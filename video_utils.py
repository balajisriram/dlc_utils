import cv2
import tqdm
import collections
import sys
import os
import pandas as pd
import numpy as np
from scipy.ndimage import median_filter as medfilt
from dlc_utils import is_odd, angle_between
import matplotlib.pyplot as plt
import time

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle

import ruptures as rpt


import pprint
ppr = pprint.PrettyPrinter(indent=2,depth=2).pprint

base_folder = r'C:\Data\BehaviorAnalysis\BehaviorAnalysis'

def rotate(image, angle, center = None, scale = 1.0):
    """ Obtained this implementation from : https://stackoverflow.com/questions/9041681/opencv-python-rotate-image-by-x-degrees-around-specific-point  """
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)
    
    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (2*w, 2*h))
    
    return rotated
    
def filter_from_df(df, p_cutoff=0.95, smoothing=('raw',np.nan), features=('centroid','l_ear','r_ear'),
                   frame_rate=20.0):
    resnet_name = df.keys()[0][0]
    output = dict()
    for feature in features:
        feature_details = dict()
        feature_x = df[(resnet_name,feature,'x')]
        feature_y = df[(resnet_name,feature,'y')]
        feature_p = df[(resnet_name,feature,'likelihood')]
        
        # remove the unreliable
        feature_x.loc[feature_p<p_cutoff] = np.nan
        feature_y.loc[feature_p<p_cutoff] = np.nan

        # smoothing
        if smoothing[0]=='raw':
            # do nothing to the distance
            pass
        elif smoothing[0]=='median_filter':
            # median filter
            filter_duration = smoothing[1] # 200 ms
            filter_dur_in_frames = int(filter_duration*frame_rate)
            if not is_odd(filter_dur_in_frames):filter_dur_in_frames = filter_dur_in_frames+1
            
            feature_x = medfilt(feature_x,size=filter_dur_in_frames)
            feature_y = medfilt(feature_y,size=filter_dur_in_frames)
            
        # now add to the data
        feature_details['x'] = feature_x
        feature_details['y'] = feature_y
        output[feature] = feature_details
    return output

def get_rotated_coordinates(df,p_cutoff=0.95,points=['centroid','snout','tail_base','l_ear','r_ear'],zero_to='centroid',align_to='snout'):
    resnet_name = df.keys()[0][0]
    rotated_values = dict()
    rotated_values['raw_centroid'] = dict()
    rotated_values['raw_centroid']['x'] = []
    rotated_values['raw_centroid']['y'] = []
    rotated_values['angle_from_veridical'] = []
    for point in points:
        rotated_values[point] = dict()
        rotated_values[point]['x'] = []
        rotated_values[point]['y'] = []
        
    for idx,row in tqdm.tqdm(df.iterrows(),total=df.shape[0]):
        zero_coord=(df[(resnet_name,zero_to,'x')][idx],df[(resnet_name,zero_to,'y')][idx])
        
        zero_p = df[(resnet_name,zero_to,'likelihood')][idx]
        align_coord=(df[(resnet_name,align_to,'x')][idx],df[(resnet_name,align_to,'y')][idx])
        align_p = df[(resnet_name,align_to,'likelihood')][idx]
        
        if zero_p<p_cutoff or align_p<p_cutoff: # no point in aligning if the point is unreliable
            rotated_values['angle_from_veridical'].append(np.nan)
            rotated_values['raw_centroid']['x'].append(np.nan)
            rotated_values['raw_centroid']['y'].append(np.nan)
            for point in points:
                rotated_values[point]['x'].append(np.nan)
                rotated_values[point]['y'].append(np.nan)
            continue
        else:
            rotated_values['raw_centroid']['x'].append(df[(resnet_name,zero_to,'x')][idx])
            rotated_values['raw_centroid']['y'].append(df[(resnet_name,zero_to,'y')][idx])
            # angle from veridical will be calculated later
            
        # there is something to align    
        zero2align = (align_coord[0]-zero_coord[0],align_coord[1]-zero_coord[1])
        veridical1 = [0.,1.]
        veridical2 = [1.,0.]
        angle_from_veridical1 = angle_between(zero2align,veridical1,type='radians')
        angle_from_veridical2 = angle_between(zero2align,veridical2,type='radians')
        
        if angle_from_veridical2>(np.pi/2):
            angle_from_veridical = np.pi-angle_from_veridical1
        else:
            angle_from_veridical = angle_from_veridical1 + np.pi
        rotated_values['angle_from_veridical'].append(angle_from_veridical)
        ROTATION_MATRIX = np.asarray([[np.cos(angle_from_veridical), -np.sin(angle_from_veridical)],[np.sin(angle_from_veridical),np.cos(angle_from_veridical)]])
        
        for point in points:
            if df[(resnet_name,point,'likelihood')][idx]>p_cutoff:
                p1 = np.asarray([df[(resnet_name,point,'x')][idx]-zero_coord[0],df[(resnet_name,point,'y')][idx]-zero_coord[1]]) # zero out the points
                p1_rot = np.matmul(ROTATION_MATRIX,np.transpose(p1))
                rotated_values[point]['x'].append(p1_rot[0])
                rotated_values[point]['y'].append(p1_rot[1])
            else:
                rotated_values[point]['x'].append(np.nan)
                rotated_values[point]['y'].append(np.nan)

    df_rotated=pd.DataFrame()

    df_rotated[('raw_centroid','x')] = rotated_values['raw_centroid']['x']
    df_rotated[('raw_centroid','y')] = rotated_values['raw_centroid']['y']

    df_rotated['angle_from_veridical'] = rotated_values['angle_from_veridical']
    
    for point in points:
        df_rotated[(point,'x')] = rotated_values[point]['x']
        df_rotated[(point,'y')] = rotated_values[point]['y']
    return df_rotated

def get_angle_from_veridical(df,p_cutoff=0.95,zero_to='centroid',align_to='snout',interp_missing=False):
    resnet_name = df.keys()[0][0]
    angle_from_veridical = []
    angle_exists = []
    
    # runs for missing data
    run_started = False
    runs = []
    current_run = []
    
    for idx,row in tqdm.tqdm(df.iterrows(),total=df.shape[0]):
        zero_coord=(df[(resnet_name,zero_to,'x')][idx],df[(resnet_name,zero_to,'y')][idx])
        zero_p = df[(resnet_name,zero_to,'likelihood')][idx]
        align_coord=(df[(resnet_name,align_to,'x')][idx],df[(resnet_name,align_to,'y')][idx])
        align_p = df[(resnet_name,align_to,'likelihood')][idx]
        
        if zero_p<p_cutoff or align_p<p_cutoff: # no point in aligning if the point is unreliable
            angle_from_veridical.append(np.nan)
            angle_exists.append(False)
            # check if a previous run has started, if not add 
            if run_started:
                current_run.append(idx)
            else:
                # this is a new missing data
                current_run.append(idx)
                run_started = True                
            continue
        else: 
            # check if it comes in here during a run
            if run_started:
                # came in here for the first time after a run of missing values
                run_started = False
                runs.append(current_run)
                current_run = [] # reset to empty so that the next run can be contained here
                
            # there is something to align    
            zero2align = (align_coord[0]-zero_coord[0],align_coord[1]-zero_coord[1])
            veridical1 = [0.,1.]
            veridical2 = [1.,0.]
            angle_from_veridical1 = angle_between(zero2align,veridical1,type='radians')
            angle_from_veridical2 = angle_between(zero2align,veridical2,type='radians')

            if angle_from_veridical2>(np.pi/2):
                angle_from_veridical.append(np.pi-angle_from_veridical1)
            else:
                angle_from_veridical.append(angle_from_veridical1 + np.pi)
            angle_exists.append(True)
    # print('length_angle_from_veridiccal',len(angle_from_veridical))
    # print('length_angle_exists',len(angle_exists))
    
    starting_run = []
    if interp_missing:
        # use the runs to fill in the values
        for this_run in runs:
            if this_run[0]==0: 
                starting_run = this_run
                continue # initial section doesn't have animals
            start_ang = angle_from_veridical[this_run[0]-1]
            stop_ang = angle_from_veridical[this_run[-1]+1]
            n_missing = len(this_run)
            
            angle_from_veridical[this_run[0]-1:this_run[-1]+2] = np.linspace(start_ang,stop_ang,num=n_missing+2) # +2 because 1:3==[1,2]
            angle_exists[this_run[0]:this_run[-1]+1] = np.ones(n_missing,dtype=bool)
    
    df['angle_from_veridical'] = angle_from_veridical
    df['angle_exists'] = angle_exists
    return df, starting_run

def annotate_video_basic(base=base_folder,video_file='ALC_050319_1_41B.avi',analysis_file='ALC_050319_1_41B.analysis',
                   output_file='ALC_050319_1_41B_annotated.avi',speedX=1,
                   skeleton=[('centroid','snout'),('centroid','tail_base'),('l_ear','r_ear'),('box_tl','box_tr'),('box_tr','box_br'),('box_br','box_bl'),('box_bl','box_tl')]):
    
    analysis = os.path.join(base,analysis_file)
    video = os.path.join(base,video_file)
    output = os.path.join(base,output_file)
    
    
    # get the analysis
    df = pd.read_pickle(analysis)
    
    # get points of interest
    points = set()
    for bone in skeleton:
        points = points.union({*bone})
    
    out = filter_from_df(df,features=points) # points=('centroid','l_ear','r_ear')

    try:
        cap = cv2.VideoCapture(video)
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        writer = cv2.VideoWriter(output,fourcc, 1/cap.get(2), 
                                 (int(cap.get(3)),int(cap.get(4))))
        for idx,row in tqdm.tqdm(df.iterrows(),total=df.shape[0]):
            if not idx%int(speedX)==0:continue # speeds up by speedX
            # get the frame
            cap.set(1,idx) # 1==set frame number
            succ,img = cap.read()
            
            # plot the skeleton
            # draw the object circles
            for bone in skeleton:
                bone_edge1 = bone[0]
                bone_edge2 = bone[1]
                if not np.isnan(out[bone_edge1]['x'][idx]) and not np.isnan(out[bone_edge2]['x'][idx]):
                    p1 = (int(out[bone_edge1]['x'][idx]),int(out[bone_edge1]['y'][idx]))
                    p2 = (int(out[bone_edge2]['x'][idx]),int(out[bone_edge2]['y'][idx]))
                    
                    img = cv2.line(img,p1,p2,(0,0,255),2)
            # add to writer
            writer.write(img)
    except Exception as er:

        raise er.with_traceback(sys.exc_info()[2])
    finally:
        # for index, row in df.iterrows():
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

def annotate_extra(base=base_folder,video_file='ALC_050319_1_41B_annotated.avi',analysis_file='ALC_050319_1_41B.analysis',
                   output_file='ALC_050319_1_41B_annotated_skeletonAligned.avi',speedX=1,skeleton=[('centroid','snout'),('centroid','tail_base'),('l_ear','r_ear')]):
    
    analysis = os.path.join(base,analysis_file)
    video = os.path.join(base,video_file)
    output = os.path.join(base,output_file)
    
    
    # get the analysis
    df = pd.read_pickle(analysis)
    
    # get points of interest
    points = set()
    for bone in skeleton:
        points = points.union({*bone})
    
    out = filter_from_df(df,features=points) # points=('centroid','l_ear','r_ear')
    
    try:
        cap = cv2.VideoCapture(video)
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        writer = cv2.VideoWriter(output,fourcc, 1/cap.get(2), 
                                 (int(cap.get(3))+200,int(cap.get(4))))
        for idx,row in tqdm.tqdm(df.iterrows(),total=df.shape[0]):
            if not idx%int(speedX)==0:continue # speeds up by speedX
            # get the frame
            cap.set(1,idx) # 1==set frame number
            succ,img = cap.read()
            img_size=img.shape
            img2 = np.zeros((img_size[0],img_size[1]+200,img_size[2]))
            img2[:,:img_size[1],:] = img
            
            # plot the skeleton
            if not np.isnan(out['centroid']['x'][idx]) and not np.isnan(out['snout']['x'][idx]):
                centroid = [int(out['centroid']['x'][idx]),int(out['centroid']['y'][idx])]
                snout = [int(out['snout']['x'][idx]),int(out['snout']['y'][idx])]
                
                centroid2snout = [snout[0]-centroid[0],snout[1]-centroid[1]]
                veridical1 = [0.,1.]
                veridical2 = [1.,0.]
                angle_from_veridical1 = angle_between(centroid2snout,veridical1,type='radians')
                angle_from_veridical2 = angle_between(centroid2snout,veridical2,type='radians')
                
                if angle_from_veridical2>(np.pi/2):
                    angle_from_veridical = 3*np.pi-angle_from_veridical1
                else:
                    angle_from_veridical = angle_from_veridical1 + np.pi
                ROTATION_MATRIX = np.asarray([[np.cos(angle_from_veridical), -np.sin(angle_from_veridical)],[np.sin(angle_from_veridical),np.cos(angle_from_veridical)]])
                
                for bone in skeleton:
                    bone_edge1 = bone[0]
                    bone_edge2 = bone[1]
                    if not np.isnan(out[bone_edge1]['x'][idx]) and not np.isnan(out[bone_edge2]['x'][idx]):
                        p1 = np.asarray([int(out[bone_edge1]['x'][idx])-int(out['centroid']['x'][idx]),int(out[bone_edge1]['y'][idx])-int(out['centroid']['y'][idx])])
                        p1_rot = np.matmul(ROTATION_MATRIX,np.transpose(p1))
                        p2 = np.asarray([int(out[bone_edge2]['x'][idx])-int(out['centroid']['x'][idx]),int(out[bone_edge2]['y'][idx])-int(out['centroid']['y'][idx])])
                        p2_rot = np.matmul(ROTATION_MATRIX,np.transpose(p2))
                        
                        p1_final = (int(p1_rot[0].astype(int)+img_size[0]+100),int(p1_rot[1].astype(int)+img_size[1]/2))
                        p2_final = (int(p2_rot[0].astype(int)+img_size[0]+100),int(p2_rot[1].astype(int)+img_size[1]/2))
                        img2 = cv2.line(img2,p1_final,p2_final,(0,0,255),2)
                                
                text_pos = (int(img_size[0]+100),int(img_size[1]/4))
                img2 = cv2.putText(img2,'{0:2.0f}'.format(np.rad2deg(angle_from_veridical)),text_pos,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1)
            # if idx%1000==0:
                # import matplotlib.pyplot as plt
                # plt.imshow(img2[:,:,2]);plt.show()
                # import pdb;pdb.set_trace()
                
                # add to writer
            writer.write(img2.astype(np.uint8))
    except Exception as er:

        raise er.with_traceback(sys.exc_info()[2])
    finally:
        # for index, row in df.iterrows():
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

def make_centered_video(base=base_folder,video_file='ALC_050319_1_41B.avi',analysis_file='ALC_050319_1_41B.analysis',
                        output_file='ALC_050319_1_41B_centered.avi',zero_to='centroid',align_to='snout',size=(100,100)):
    analysis = os.path.join(base,analysis_file)
    video = os.path.join(base,video_file)
    output = os.path.join(base,output_file)
    
    # get the analysis
    df = pd.read_pickle(analysis)
    resnet_name = df.keys()[0][0]
    
    # get the angles
    df = get_angle_from_veridical(df,zero_to=zero_to,align_to=align_to)
    
    try:
        cap = cv2.VideoCapture(video)
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        writer = cv2.VideoWriter(output,fourcc, 1/cap.get(2),(size[0],size[1]))
        # writer = cv2.VideoWriter(output,fourcc, 1/cap.get(2),(2*int(cap.get(3)),2*int(cap.get(4))))
        for idx,row in tqdm.tqdm(df.iterrows(),total=df.shape[0]):
            if not df['angle_exists'][idx]:continue
            # get the frame
            cap.set(1,idx) # 1==set frame number
            succ,img = cap.read()
            
            center = (np.int(df[(resnet_name,zero_to,'x')][idx]),np.int(df[(resnet_name,zero_to,'y')][idx]))
            angle = np.rad2deg(df['angle_from_veridical'][idx])
            rot_img = rotate(img,-angle,center=center)
            rot_img_zoom = rot_img[center[1]-size[1]/2:center[1]+size[1]/2,center[0]-size[0]/2:center[0]+size[0]/2,:]
            # cv2.circle(rot_img,(center[0],center[1]),5,(0,255,0),-1)
            # add to writer
            writer.write(rot_img_zoom)
    except Exception as er:

        raise er.with_traceback(sys.exc_info()[2])
    finally:
        # for index, row in df.iterrows():
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
        
def get_video_matrix(base=base_folder,video_file='ALC_050319_1_41B.avi',analysis_file='ALC_050319_1_41B.analysis',
                        zero_to='centroid',align_to='snout',size_requested=(100,100),idx_lim=(None,None), scale=None):
    """
        img.shape=(#rows, #columns, #channels)
        size = (#rows,#columns)
        output_matrix = (#height, #width,#frames,#channels)
    """
    
    if scale is None:
        size = size_requested
        need_scaling = False
    else:
        size = tuple([2*x for x in size_requested]) # we are going to get a larger picture, zoom (in or out) as required and then select the relevant portions
        need_scaling = True
    print('need_scaling',need_scaling)
    analysis = os.path.join(base,analysis_file)
    video = os.path.join(base,video_file)
    
    # get the analysis
    df = pd.read_pickle(analysis)
    resnet_name = df.keys()[0][0]
    
    
    # get the angles
    df,start_run = get_angle_from_veridical(df,zero_to=zero_to,align_to=align_to,interp_missing=True)
    
    # rationalize the size of the size
    size = [f if f%2==0 else f+1 for f in size]
    
    cap = cv2.VideoCapture(video)
    vid_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_mat = np.zeros((size_requested[1],size_requested[0],vid_length,3),order='C',dtype=np.uint8)
    bad_idx = []
    do_pdb = True
    for idx,row in tqdm.tqdm(df.iterrows(),total=df.shape[0]):
        try:
            if idx in start_run: continue
            if not df['angle_exists'][idx]:
                print('No angle_predicted. adding {0} in bad_idx'.format(idx))
                bad_idx.append(idx)
                continue

            # get the frame
            cap.set(1,idx) # 1==set frame number
            succ,img = cap.read()
            
            # pad frame to ensure that the required data exists
            pad_length=np.max(size)
            img_pad = np.zeros((img.shape[0]+2*pad_length,img.shape[1]+2*pad_length,3),dtype=np.uint8)
            img_pad[pad_length:-pad_length,pad_length:-pad_length,:] = img
            
            center = (np.int(df[(resnet_name,zero_to,'x')][idx]),np.int(df[(resnet_name,zero_to,'y')][idx]))
            angle = np.rad2deg(df['angle_from_veridical'][idx])
            rot_img = rotate(img_pad,-angle,center=(center[0]+pad_length,center[1]+pad_length))
            
            rot_img_zoom = rot_img[center[1]+pad_length-size[1]//2:center[1]+pad_length+size[1]//2,center[0]+pad_length-size[0]//2:center[0]+pad_length+size[0]//2,:]
            
            if need_scaling:
                rot_img_zoom1 = cv2.resize(rot_img_zoom,dsize=(0,0),fx=scale[0],fy=scale[1])
                h_z,w_z,c_z = rot_img_zoom1.shape
                t_edge = np.int(h_z/2-size_requested[1]/2)
                b_edge = t_edge+size_requested[1]
                
                l_edge = np.int(w_z/2-size_requested[0]/2)
                r_edge = l_edge+size_requested[0]
                
                rot_img_zoom1 = rot_img_zoom1[t_edge:b_edge,l_edge:r_edge,:]
                
            # if idx%1000==0:
                # temp1 = rot_img_zoom[50:150,20:60,:]
                # breakpoint()
            
            video_mat[:,:,idx,:] = rot_img_zoom1
            
        except Exception as er:
            print('adding {0} in bad_idx'.format(idx))
            bad_idx.append(idx)
            print(er)
    # for index, row in df.iterrows():
    cap.release()
    cv2.destroyAllWindows()
    print(bad_idx)
    print('bad_idx length=',len(bad_idx))
    return video_mat

def test_array_reshape():
    temp = np.ones((10,10,100,3),dtype=np.uint8)
    for i in range(100):
        for j in range(10):
            for k in range(10):
                val = np.uint8(np.random.rand()*255)
                temp[j,k,i,0] = val
                temp[j,k,i,1] = val
                temp[j,k,i,2] = val
        
    temp = np.reshape(temp,(100,100,3))
    plt.imshow(temp)
    plt.show()

def reshape_array(inp,how='for_eigen_mice',params=None):
    
    if how=='for_pca':
        h,w,f,c = inp.shape
        out = np.zeros((h*w*c,f),order='C',dtype=np.uint8)
        inp = np.reshape(inp,(h*w,f,c),order='C')
        for chan in range(c):
            out[chan*h*w:(chan+1)*h*w,:] = inp[:,:,chan]
        out = out.transpose()
    elif how=='for_eigen_mice':
        assert params is not None, 'Need params'
        h,w,c = params
        n_eig = inp.shape[0]
        out = np.zeros((h,w,c,n_eig))
        for curr_eig in range(n_eig):
            for chan in range(c):
                out[:,:,chan,curr_eig] = np.reshape(inp[curr_eig,chan*h*w:(chan+1)*h*w], (h,w),order = 'C')
            out[:,:,:,curr_eig] = (out[:,:,:,curr_eig]-np.min(out[:,:,:,curr_eig]))/(np.max(out[:,:,:,curr_eig])-np.min(out[:,:,:,curr_eig])) # normalize each frame
    elif how=='unflatten':
        assert params is not None, 'Need params'
        h,w,c = params
        n_frames = inp.shape[1]
        out = np.zeros((h,w,c,n_frames))
        for frame in range(n_frames):
            for chan in range(c):
                out[:,:,chan,frame] = np.reshape(inp[chan*h*w:(chan+1)*h*w,frame], (h,w), order='C')
    elif how=='flattened_image':
        h,w,f,c = inp.shape
        out = np.reshape(inp,(h*w,f,c), order='C')
    elif how=='flatten_image_and_color':
        h,w,f,c = inp.shape
        out = np.zeros((h*w*c,f,c),order='C',dtype=np.uint8)
        inp = np.reshape(inp,(h*w,f,c),order='C')
        for chan in range(c):
            out[chan*h*w:(chan+1)*h*w,:,chan] = inp[:,:,chan]
    return out

def get_distance_between_points(df, p1='l_ear',p2='r_ear'):
    resnet_name = df.keys()[0][0]
    out=  np.sqrt(np.square(df[(resnet_name,p1,'x')]-df[(resnet_name,p2,'x')])+ np.square(df[(resnet_name,p1,'y')]-df[(resnet_name,p2,'y')]))
    qual_chk = np.bitwise_and(df[(resnet_name,p1,'likelihood')]>0.95,df[(resnet_name,p2,'likelihood')]>0.95)
    
    out[~qual_chk] = np.nan
    
    return out

def make_dual_plot(f1,f2):
    out = np.zeros_like(f1,dtype=np.uint8)
    out[:,:,0] = cv2.cvtColor(f1,cv2.COLOR_BGR2GRAY)
    out[:,:,1] = cv2.cvtColor(f2,cv2.COLOR_BGR2GRAY)
    plt.imshow(out)
    plt.show()

if __name__=='__main__':
    # base = r'C:\Users\bsriram\OneDrive\Desktop\Data\ACM_Data\OpenField'
    # files = ['ALC_050319_1_41B','ALC_050319_1_41C','ALC_050319_1_41R','ALC_050319_1_42B','ALC_050319_1_42C','ALC_050319_1_42R','ALC_050319_1_43B',
             # 'ALC_050319_1_43C','ALC_050319_1_43G','ALC_050319_1_43R','ALC_050319_2_44B','ALC_050319_2_44C','ALC_050319_2_44R','ALC_050319_2_45B','ALC_050319_2_45R',
             # 'ALC_050319_2_46B','ALC_050319_2_46C','ALC_050319_2_46R','ALC_051719_1_42Bk','ALC_051719_1_42G','ALC_051719_1_45Bk','ALC_051719_1_45C','ALC_051719_2_53B',
             # 'ALC_051719_2_53C','ALC_051719_2_53G','ALC_051719_2_53R','ALC_051719_2_54B','ALC_051719_2_54C','ALC_051719_2_54R','ALC_051719_2_55B','ALC_051719_2_55C',
             # 'ALC_051719_2_55R','ALC_060519_1_49B','ALC_060519_1_49C','ALC_060519_1_49R','ALC_060519_2_48B','ALC_060519_2_48C','ALC_060519_2_48R','ALC_060519_2_57B',
             # 'ALC_060519_2_57C','ALC_060519_2_57G','ALC_060519_2_57R','ALC_060519_2_58B','ALC_060519_2_58C','ALC_060519_2_58R','ALC_070519_1_21B','ALC_070519_1_21C',
             # 'ALC_070519_1_21R','ALC_070519_1_31B','ALC_070519_1_31C','ALC_070519_1_31R','ALC_070519_1_60B','ALC_070519_1_60C','ALC_070519_1_60G','ALC_070519_1_60R']
    files = ['ALC_070519_1_60G',]
    
    
    
    ## GET THE INTER KEYPOINT AVERAGE DISTANCE
    get_keypoint_distance=False
    if get_keypoint_distance:
        fig,ax = plt.subplots(nrows=2,ncols=2)
        ax = ax.flatten()
        m_e2e = []
        sd_e2e = []
        m_c2s = []
        sd_c2s = []
        output = []
        for i,file in enumerate(files):
            this_subject = {}
            # print('Running analysis for '+ file)
            analysis_file = file+'.analysis'
            analysis = os.path.join(base_folder,analysis_file)
            df = pd.read_pickle(analysis)
            
            dist_data = get_distance_between_points(df,p1='centroid',p2='snout')
            dist_data[dist_data>np.nanquantile(dist_data,q=0.99)] = np.nan # remove quantiles
            mthis_c2s = np.nanmean(dist_data)
            sdthis_c2s = np.nanstd(dist_data)
            m_c2s.append(mthis_c2s)
            sd_c2s.append(sdthis_c2s)
            
            dist_data = get_distance_between_points(df,p1='l_ear',p2='r_ear')
            dist_data[dist_data>np.nanquantile(dist_data,q=0.99)] = np.nan # remove quantiles
            mthis_e2e = np.nanmean(dist_data)
            sdthis_e2e = np.nanstd(dist_data)
            m_e2e.append(mthis_e2e)
            sd_e2e.append(sdthis_e2e)
            
            this_subject['subject_id'] = file
            this_subject['m_c2s'] = mthis_c2s
            this_subject['sd_c2s'] = sdthis_c2s
            this_subject['m_e2e'] = mthis_e2e
            this_subject['sd_e2e'] = sdthis_e2e
            
            print(file,mthis_c2s,mthis_e2e)
            output.append(this_subject)
        
        ords = np.argsort(m_e2e)
        for i,ord in enumerate(ords):
           ax[0].plot(i+1,m_e2e[ord],'kd')
           ax[0].plot([i+1,i+1],[m_e2e[ord]+sd_e2e[ord],m_e2e[ord]-sd_e2e[ord]],'k')
        ax[0].set_ylabel('Ear to ear distance')
        
        ords = np.argsort(m_c2s)
        for i,ord in enumerate(ords):
           ax[3].plot(m_c2s[ord],i+1,'kd')
           ax[3].plot([m_c2s[ord]+sd_c2s[ord],m_c2s[ord]-sd_c2s[ord]],[i+1,i+1],'k')
        ax[3].set_xlabel('Centroid to Snout distance')
        
        ax[1].plot(m_c2s,m_e2e,'kd')
        temp = np.corrcoef(m_c2s,m_e2e) 
        ax[2].text(0.5,0.5,'ear-ear:{0:.1f}-{1:.1f}\n\ncentroid-snout:{2:.1f}-{3:.1f}\n\nCorr coef = {4:.2f}'.format(np.min(m_e2e),np.max(m_e2e),np.min(m_c2s),np.max(m_c2s),temp[0,1]),
            horizontalalignment='center',
             verticalalignment='center',
             multialignment='center')
        df2 = pd.DataFrame(output)
        df2.to_pickle(os.path.join(base_folder,'waypoint_df.data'))
        breakpoint()
    
    ## ANNOTATE BASIC AND EXTRA
    annotate_videos=False
    if annotate_videos:
        for i,file in enumerate(files):
            print('Running analysis for '+ file)
            video_file = file+'.avi'
            analysis_file = file+'.analysis'
            output_file = file+'_annotated.avi'
            annotate_video_basic(base=base,video_file=video_file,analysis_file=analysis_file,output_file=output_file,speedX=1)
            
            video_file = file+'_annotated.avi'
            analysis_file = file+'.analysis'
            output_file = file+'_annotated_skeletonAligned.avi'
            annotate_extra(base=base,video_file=video_file,analysis_file=analysis_file,output_file=output_file,speedX=1)
    
    # GET THE VIDEO MATRIX
    get_video=False
    if get_video:
        for file in files:
            matrix = get_video_matrix(size_requested=(40,100),video_file=file+'.avi',analysis_file=file+'.analysis')
            print('matrix shape=({0})'.format(matrix.shape))
            breakpoint()
            
    # VIEW SAMPLE FRAMES
    view_sample_frames = False
    if view_sample_frames:
        for file in files:
            n_samples = 9
            fig,ax = plt.subplots(nrows=1,ncols=n_samples)
            ax = ax.flatten()
            matrix = get_video_matrix(size_requested=(40,100),video_file=file+'.avi',analysis_file=file+'.analysis')
            print('matrix shape=({0})'.format(matrix.shape))
            for i in range(n_samples):
                which = np.random.randint(200,high=18000)
                ax[i].imshow(matrix[:,:,which,:])
                ax[i].set_axis_off()
                ax[i].set_title('fr#:{0}'.format(which))
    
    # FLATTEN VIDEO MATRIX FOR VIEWING
    flatten_for_viewing = False
    if flatten_for_viewing:
        for file in files:
            matrix = get_video_matrix(size_requested=(40,100),video_file=file+'.avi',analysis_file=file+'.analysis')
            mat_flat = reshape_array(matrix,how='flattened_image')
            plt.imshow(mat_flat)
            plt.show()
    
    # FLATTEN VIDEO MATRIX FOR VIEWING
    flatten_for_viewing_with_color = False
    if flatten_for_viewing_with_color:
        for file in files:
            matrix = get_video_matrix(size_requested=(40,100),video_file=file+'.avi',analysis_file=file+'.analysis')
            mat_flat = reshape_array(matrix,how='flatten_image_and_color')
            plt.imshow(mat_flat)
            plt.show()
    
    
    # FLATTEN VIDEO MATRIX GREYSCALE
    flatten_for_pca = False
    if flatten_for_pca:
        for file in files:
            matrix = get_video_matrix(size_requested=(40,100),video_file=file+'.avi',analysis_file=file+'.analysis')
            mat_flat = reshape_array(matrix,how='for_pca')
            plt.imshow(mat_flat,cmap='gray')
            plt.show()
    
    # PERFORM PCA AND EVALUATE MODEL QUALITY
    perform_pca = False
    if perform_pca:
        for file in files:
            fig,ax = plt.subplots(nrows=1,ncols=3)
            ax = ax.flatten()
            
            matrix = get_video_matrix(size_requested=(40,100),video_file=file+'.avi',analysis_file=file+'.analysis')
            mat_flat = reshape_array(matrix,how='for_pca')
            
            start_time = time.time()
            #mat_flat = StandardScaler().fit_transform(mat_flat)
            ax[0].imshow(mat_flat,cmap='gray')
            
            n_pca = 300
            pca = PCA(n_components = n_pca)
            prin_comps = pca.fit_transform(mat_flat)
            # percent_variance = np.round(pca.explained_variance_ratio_* 100, decimals =2)
            ax[1].imshow(pca.inverse_transform(prin_comps),cmap='gray')
            
            ax[2].imshow(mat_flat-pca.inverse_transform(prin_comps),cmap='gray')
            
            
            print('that took {0} seconds'.format(time.time()-start_time))
            plt.show()
            
    # PCA TO GET EIGEN MICE
    get_eigen_mice = False
    if get_eigen_mice:
        for file in files:
            fig,ax = plt.subplots(nrows=7,ncols=20)
            fig.subplots_adjust(hspace=0.1, wspace=0.1)
            ax = ax.flatten()
            
            matrix = get_video_matrix(size_requested=(40,100),video_file=file+'.avi',analysis_file=file+'.analysis')
            mat_flat = reshape_array(matrix,how='for_pca')
            
            start_time = time.time()
            
            n_pca = 300
            pca = PCA(n_components = n_pca)
            prin_comps = pca.fit_transform(mat_flat)
            
            eig_vals = pca.components_
            params = (100,40,3) # height, width, channels
            eig_vals = reshape_array(eig_vals,how='for_eigen_mice',params=params)
            
            for i in range(140):
                ax[i].imshow(eig_vals[:,:,:,i])
                ax[i].set_axis_off()
            
            print('that took {0} seconds'.format(time.time()-start_time))
            fig.show()
            
    
    # PCA FOR EVALUATING FRAMES
    evaluate_indiv_frames = False
    if evaluate_indiv_frames:
        for file in files:
            # which frames to focus on?
            which_frames = [4598, 6337, 7723, 4794, 16288, 1714, 1453, 4153, 2430]
            fig,ax = plt.subplots(nrows=2,ncols=9)
            fig.subplots_adjust(hspace=0.1, wspace=0.1)
            ax = ax.flatten()
            
            matrix = get_video_matrix(size_requested=(40,100),video_file=file+'.avi',analysis_file=file+'.analysis')
            mat_flat = reshape_array(matrix,how='for_pca')
            
            start_time = time.time()
            
            n_pca = 300
            pca = PCA(n_components = n_pca)
            prin_comps = pca.fit_transform(mat_flat)
            
            reconstructed = pca.inverse_transform(prin_comps)
            orig_mat = reshape_array(mat_flat[which_frames,:].transpose(),'unflatten',params=(100,40,3))
            recon_mat = reshape_array(reconstructed[which_frames,:].transpose(),'unflatten',params=(100,40,3))
            
            for i in range(9):
                ax[i].imshow(np.uint8(orig_mat[:,:,:,i]));
                ax[i+9].imshow(np.uint8(recon_mat[:,:,:,i]));
                ax[i].set_axis_off();
                ax[i+9].set_axis_off()
            
            print('that took {0} seconds'.format(time.time()-start_time))
            fig.show()
            
            breakpoint()
    
    # MAKE AND SAVE THE RAW MATRICES
    save_raw_matrices = False
    if save_raw_matrices:
        for file in files:
            print('getting matrix for {0}'.format(file))
            matrix = get_video_matrix(size=(40,100),video_file=file+'.avi',analysis_file=file+'.analysis')
            np.save(os.path.join(base_folder,file+'.raw_matrix'),matrix)
    
    # TRY LOADING THE DATA TO A MATRIX
    load_all_data=False
    if load_all_data:
        total_frames = 0
        for file in files:
            temp = np.load(os.path.join(base_folder,file+'.raw_matrix.npy'))
            th,w,f,c = temp.shape
            total_frames += f
            print('matrix for {0} - shape={1}'.format(file,temp.shape))
        print('total frames = {0}'.format(total_frames))
        frame_data = np.zeros((total_frames,12000),dtype=np.uint8)
        subj_identity = np.zeros((total_frames,),dtype=np.uint8)
        subj_lut = {}
        current_frame = 0
        for i,file in enumerate(files):
            temp = np.load(os.path.join(base_folder,file+'.raw_matrix.npy'))
            temp = reshape_array(temp,how='for_pca')
            frame_data[current_frame:current_frame+temp.shape[0],:] = temp
            subj_identity[current_frame:current_frame+temp.shape[0]] = i
            current_frame += temp.shape[0]
            print('matrix for {0} - shape={1}'.format(file,temp.shape))
        np.savez(os.path.join(base_folder,'all_data_unnormalized'),frame=frame_data,id=subj_identity,lut=files)
    
    # NAIVE BAYES
    run_naive_bayes = False
    if run_naive_bayes:
        temp = np.load(os.path.join(base_folder,'all_data_unnormalized.npz'))
        X = temp['frame']
        y = temp['id']
        perf = []
        n_labels = np.unique(y).size
        n_subsamples = 10
        conf_all = np.zeros((n_labels,n_labels,n_subsamples))
        conf_shuffle_all = np.zeros((n_labels,n_labels,n_subsamples))
        for i in range(n_subsamples):
            print('subsampling the original data: #{0}'.format(i))
            N_subsample = 100000
            sub_idx = shuffle(np.arange(0,X.shape[0]))[:N_subsample]
            X_this = X[sub_idx,:]
            y_this = y[sub_idx]
            print('Doing a PCA on the features')
            X_this = PCA(n_components=300, whiten=True).fit_transform(X_this)
            X_this -= np.min(X_this) # because naive bayes fails with negative numbers
            print('doing the split')
            X_train, X_test, y_train, y_test = train_test_split(X_this, y_this, test_size=0.25, random_state=i)
            print('running bayes')
            gnb = MultinomialNB()
            y_pred = gnb.fit(X_train, y_train).predict(X_test)
            # get the confusion matrix for real data
            conf_mat = confusion_matrix(y_test,y_pred)
            conf_shuffle = np.zeros_like(conf_mat,dtype=np.float)
            for j in range(10):
                # shuffle the y_test for random predictions
                y_shuffle = shuffle(y_test)
                conf_shuffle += confusion_matrix(y_test,y_shuffle)
            conf_shuffle /= 10
            conf_all[:,:,i] = conf_mat
            conf_shuffle_all[:,:,i] = conf_shuffle
        breakpoint()
        # do some plotting
        fig,ax = plt.subplots(1,1)
        ax.imshow(np.mean(conf_all/conf_shuffle_all,axis=2))
        ax.set_xlabel('y_true')
        ax.set_ylabel('y_predicted')
        
        
        
        breakpoint()
        
    # MLP CLASSIFIER
    run_mlp_classifier = False
    if run_mlp_classifier:
        temp = np.load(os.path.join(base_folder,'all_data_unnormalized.npz'))
        X = temp['frame']
        y = temp['id']
        perf = []
        n_labels = np.unique(y).size
        n_subsamples = 10
        conf_all = np.zeros((n_labels,n_labels,n_subsamples))
        conf_shuffle_all = np.zeros((n_labels,n_labels,n_subsamples))
        for i in range(n_subsamples):
            print('subsampling the original data: #{0}'.format(i))
            N_subsample = 100000
            sub_idx = shuffle(np.arange(0,X.shape[0]))[:N_subsample]
            X_this = X[sub_idx,:]
            y_this = y[sub_idx]
            print('Doing a PCA on the features')
            X_this = PCA(n_components=300, whiten=True).fit_transform(X_this)
            print('doing the split')
            X_train, X_test, y_train, y_test = train_test_split(X_this, y_this, test_size=0.25, random_state=i)
            print('running mlp')
            mlp_clf = MLPClassifier(max_iter=1000,hidden_layer_sizes=(100,25,12),verbose=True,learning_rate='adaptive')
            y_pred = mlp_clf.fit(X_train, y_train).predict(X_test)
            # get the confusion matrix for real data
            conf_mat = confusion_matrix(y_test,y_pred)
            conf_shuffle = np.zeros_like(conf_mat,dtype=np.float)
            for j in range(10):
                # shuffle the y_test for random predictions
                y_shuffle = shuffle(y_test)
                conf_shuffle += confusion_matrix(y_test,y_shuffle)
            conf_shuffle /= 10
            conf_all[:,:,i] = conf_mat
            conf_shuffle_all[:,:,i] = conf_shuffle
        breakpoint()
    
    # NORMALIZE SIZES
    normalize_sizes = False
    if normalize_sizes:
        E2E_ideal = 13.
        C2S_ideal = 35.
        for file in files:
            print('Getting distances for '+ file)
            analysis_file = file+'.analysis'
            analysis = os.path.join(base_folder,analysis_file)
            df = pd.read_pickle(analysis)
            
            dist_data = get_distance_between_points(df,p1='centroid',p2='snout')
            dist_data[dist_data>np.nanquantile(dist_data,q=0.99)] = np.nan # remove quantiles
            m_c2s = np.nanmean(dist_data)
            
            dist_data = get_distance_between_points(df,p1='l_ear',p2='r_ear')
            dist_data[dist_data>np.nanquantile(dist_data,q=0.99)] = np.nan # remove quantiles
            m_e2e = np.nanmean(dist_data)
            
            e2e_scale = E2E_ideal/m_e2e
            c2s_scale = C2S_ideal/m_c2s
            print('scales:',(e2e_scale,c2s_scale))
            matrix = get_video_matrix(size_requested=(40,100),video_file=file+'.avi',analysis_file=file+'.analysis', scale=(c2s_scale,e2e_scale))
            np.save(os.path.join(base_folder,file+'.raw_matrix_norm'),matrix)
    
    # MAKE DUAL VIDEOS FOR NORMALIZED MICE
    make_dual_video = False
    if make_dual_video:
        df2 = pd.read_pickle(os.path.join(base_folder,'waypoint_df.data'))
        try:
            for file in files:
                orig = np.load(os.path.join(base_folder,file+'.raw_matrix.npy'))
                norm = np.load(os.path.join(base_folder,file+'.raw_matrix_norm.npy'))
                out = np.zeros((100,40,3),dtype=np.float)
                f_nos = norm.shape[2]
                fourcc = cv2.VideoWriter_fourcc(*'DIVX')
                writer = cv2.VideoWriter(os.path.join(base_folder,file+'_norm_combined.avi'),fourcc, 20,(40,100))
                max_vals_orig = []
                max_vals_norm = []
                for idx in tqdm.tqdm(range(f_nos)):
                    out[:,:,0] = cv2.cvtColor(orig[:,:,idx,:],cv2.COLOR_BGR2GRAY)
                    out[:,:,1] = cv2.cvtColor(norm[:,:,idx,:],cv2.COLOR_BGR2GRAY)
                    # out = norm[:,:,idx,:].astype(np.float)-orig[:,:,idx,:].astype(np.float)
                    out[out<0] = 0
                    if idx%1000==0:
                        make_dual_plot(orig[:,:,idx,:],norm[:,:,idx,:])
                    # max_vals_orig.append(np.max(orig[:,:,idx,:]))
                    # max_vals_norm.append(np.max(norm[:,:,idx,:]))
                    writer.write(np.uint8(out))
        finally:
            writer.release()
            cv2.destroyAllWindows()
        breakpoint()
    
    # LOAD NORMALIZED DATA TO MATRIX
    load_all_norm_data=False
    if load_all_norm_data:
        total_frames = 0
        for file in files:
            temp = np.load(os.path.join(base_folder,file+'.raw_matrix_norm.npy'))
            th,w,f,c = temp.shape
            total_frames += f
            print('matrix for {0} - shape={1}'.format(file,temp.shape))
        print('total frames = {0}'.format(total_frames))
        frame_data = np.zeros((total_frames,12000),dtype=np.uint8)
        subj_identity = np.zeros((total_frames,),dtype=np.uint8)
        current_frame = 0
        for i,file in enumerate(files):
            temp = np.load(os.path.join(base_folder,file+'.raw_matrix.npy'))
            temp = reshape_array(temp,how='for_pca')
            frame_data[current_frame:current_frame+temp.shape[0],:] = temp
            subj_identity[current_frame:current_frame+temp.shape[0]] = i
            current_frame += temp.shape[0]
            print('matrix for {0} - shape={1}'.format(file,temp.shape))
        np.savez(os.path.join(base_folder,'all_data_normalized'),frame=frame_data,id=subj_identity,lut=files)
    
    # NAIVE BAYES
    run_naive_bayes_normalized = False
    if run_naive_bayes_normalized:
        temp = np.load(os.path.join(base_folder,'all_data_normalized.npz'))
        X = temp['frame']
        y = temp['id']
        perf = []
        n_labels = np.unique(y).size
        n_subsamples = 10
        conf_all = np.zeros((n_labels,n_labels,n_subsamples))
        conf_shuffle_all = np.zeros((n_labels,n_labels,n_subsamples))
        for i in range(n_subsamples):
            print('subsampling the original data: #{0}'.format(i))
            N_subsample = 100000
            sub_idx = shuffle(np.arange(0,X.shape[0]))[:N_subsample]
            X_this = X[sub_idx,:]
            y_this = y[sub_idx]
            print('Doing a PCA on the features')
            X_this = PCA(n_components=300, whiten=True).fit_transform(X_this)
            X_this -= np.min(X_this) # because naive bayes fails with negative numbers
            print('doing the split')
            X_train, X_test, y_train, y_test = train_test_split(X_this, y_this, test_size=0.25, random_state=i)
            print('running bayes')
            gnb = MultinomialNB()
            y_pred = gnb.fit(X_train, y_train).predict(X_test)
            # get the confusion matrix for real data
            conf_mat = confusion_matrix(y_test,y_pred)
            conf_shuffle = np.zeros_like(conf_mat,dtype=np.float)
            for j in range(10):
                # shuffle the y_test for random predictions
                y_shuffle = shuffle(y_test)
                conf_shuffle += confusion_matrix(y_test,y_shuffle)
            conf_shuffle /= 10
            conf_all[:,:,i] = conf_mat
            conf_shuffle_all[:,:,i] = conf_shuffle
        breakpoint()
        # do some plotting
        fig,ax = plt.subplots(1,1)
        ax.imshow(np.mean(conf_all/conf_shuffle_all,axis=2))
        ax.set_xlabel('y_true')
        ax.set_ylabel('y_predicted')
        
        breakpoint()
    
    # MLP CLASSIFIER FOR NORMALIZED
    run_mlp_classifier_normalized = False
    
    if run_mlp_classifier_normalized:
        temp = np.load(os.path.join(base_folder,'all_data_normalized.npz'))
        X = temp['frame']
        y = temp['id']
        perf = []
        n_labels = np.unique(y).size
        n_subsamples = 10
        conf_all = np.zeros((n_labels,n_labels,n_subsamples))
        conf_shuffle_all = np.zeros((n_labels,n_labels,n_subsamples))
        for i in range(n_subsamples):
            print('subsampling the original data: #{0}'.format(i))
            N_subsample = 100000
            sub_idx = shuffle(np.arange(0,X.shape[0]))[:N_subsample]
            X_this = X[sub_idx,:]
            y_this = y[sub_idx]
            print('Doing a PCA on the features')
            X_this = PCA(n_components=300, whiten=True).fit_transform(X_this)
            print('doing the split')
            X_train, X_test, y_train, y_test = train_test_split(X_this, y_this, test_size=0.25, random_state=i)
            print('running mlp')
            mlp_clf = MLPClassifier(max_iter=1000,hidden_layer_sizes=(100,25,12),verbose=True,learning_rate='adaptive')
            y_pred = mlp_clf.fit(X_train, y_train).predict(X_test)
            # get the confusion matrix for real data
            conf_mat = confusion_matrix(y_test,y_pred)
            conf_shuffle = np.zeros_like(conf_mat,dtype=np.float)
            for j in range(10):
                # shuffle the y_test for random predictions
                y_shuffle = shuffle(y_test)
                conf_shuffle += confusion_matrix(y_test,y_shuffle)
            conf_shuffle /= 10
            conf_all[:,:,i] = conf_mat
            conf_shuffle_all[:,:,i] = conf_shuffle
        breakpoint()
    
    # CHANGE POINT DETECTION
    run_cpd = True
    if run_cpd:
        for file in files:
            temp = np.load(os.path.join(base_folder,file+'.raw_matrix_norm.npy'))
            temp = reshape_array(temp,how='for_pca')
            
            temp_pca = PCA(n_components=300, whiten=True).fit_transform(temp)
            
            fig,ax = plt.subplots(3,1,sharex=True)
            
            ax[0].imshow(temp.transpose(),aspect='auto',cmap='bwr')
            ax[1].imshow(temp_pca.transpose(),aspect='auto',cmap='bwr')

            
            breakpoint()
            
    # plt.bar(x=range(n_pca),height=np.cumsum(percent_variance))
    # plt.plot([0,n_pca],[85,85],'k--')
    # plt.show()
    # fig,ax = plt.subplots(nrows=1,ncols=9)
    # for i in range(9): which = np.random.randint(200,high=18000);ax[i].imshow(matrix[:,:,which,:]);ax[i].set_axis_off();ax[i].set_title('fr#:{0}'.format(which)); # ax[i].set_yticks(None);
    # plt.show()
    # breakpoint()
    

    # h,w,l,f = matrix.shape
    # vid_siz = (w,h)
        
    # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    # writer = cv2.VideoWriter(r'C:\Users\balaji\OneDrive\Desktop\temp.avi',fourcc, 20.,vid_siz)
    
    
    # for i in range(f):
        # writer.write(matrix[:,:,:,i])
        
    # writer.release()
    # test_array_reshape()
    
    
        
    