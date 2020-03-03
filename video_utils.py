import cv2
import tqdm
import collections
import sys
import os
import pandas as pd
import numpy as np
from scipy.ndimage import median_filter as medfilt
from dlc_utils import is_odd, angle_between


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
    for point in points:
        rotated_values[point]['x'] = []
        rotated_values[point]['y'] = []
        
    for idx,row in tqdm.tqdm(df.iterrows(),total=df.shape[0]):
        zero_coord=(df[(resnet_name,zero_to,'x')][idx],df[(resnet_name,zero_to,'y')][idx])
        zero_p = df[(resnet_name,zero_to,'likelihood')][idx]
        align_coord=(df[(resnet_name,align_to,'x')][idx],df[(resnet_name,align_to,'y')][idx])
        align_p = df[(resnet_name,align_to,'likelihood')][idx]
        
        if zero_p<p_cutoff or align_p<p_cutoff: # no point in aligning if the point is unreliable
            for point in points:
                rotated_values[point]['x'].append(np.nan)
                rotated_values[point]['y'].append(np.nan)
            continue
        # there is something to align    
        zero2align = (align_coord[0]-zero_coord[0],align_coord[1]-zero_coord[1])
        veridical1 = [0.,1.]
        veridical2 = [1.,0.]
        angle_from_veridical1 = angle_between(zero2align,veridical1,type='radians')
        angle_from_veridical2 = angle_between(zero2align,veridical2,type='radians')
        
        if angle_from_veridical2>(np.pi/2):
            angle_from_veridical = 3*np.pi-angle_from_veridical1
        else:
            angle_from_veridical = angle_from_veridical1 + np.pi
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


def annotate_video_basic(base=r'C:\Users\balaji\Desktop\ALC_OF',video_file='ALC_050319_1_41B.avi',analysis_file='ALC_050319_1_41B.analysis',
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

def annotate_extra(base=r'C:\Users\balaji\Desktop\ALC_OF',video_file='ALC_050319_1_41B_annotated.avi',analysis_file='ALC_050319_1_41B.analysis',
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
        
if __name__=='__main__':
    base = r'C:\Users\bsriram\Desktop\Data\ACM_Data\OpenField'
    
    # files = ['ALC_050319_1_41B','ALC_050319_1_41B_OF','ALC_050319_1_41C','ALC_050319_1_41R','ALC_050319_1_42B','ALC_050319_1_42C','ALC_050319_1_42R','ALC_050319_1_43B',
             # 'ALC_050319_1_43C','ALC_050319_1_43G','ALC_050319_1_43R','ALC_050319_2_44B','ALC_050319_2_44C','ALC_050319_2_44R','ALC_050319_2_45B','ALC_050319_2_45R',
             # 'ALC_050319_2_46B','ALC_050319_2_46C','ALC_050319_2_46R','ALC_051719_1_42Bk','ALC_051719_1_42G','ALC_051719_1_45Bk','ALC_051719_1_45C','ALC_051719_2_53B',
             # 'ALC_051719_2_53C','ALC_051719_2_53G','ALC_051719_2_53R','ALC_051719_2_54B','ALC_051719_2_54C','ALC_051719_2_54R','ALC_051719_2_55B','ALC_051719_2_55C',
             # 'ALC_051719_2_55R','ALC_060519_1_49B','ALC_060519_1_49C','ALC_060519_1_49R','ALC_060519_2_48B','ALC_060519_2_48C','ALC_060519_2_48R','ALC_060519_2_57B',
             # 'ALC_060519_2_57C','ALC_060519_2_57G','ALC_060519_2_57R','ALC_060519_2_58B','ALC_060519_2_58C','ALC_060519_2_58R','ALC_070519_1_21B','ALC_070519_1_21C',
             # 'ALC_070519_1_21R','ALC_070519_1_31B','ALC_070519_1_31C','ALC_070519_1_31R','ALC_070519_1_60B','ALC_070519_1_60C','ALC_070519_1_60G','ALC_070519_1_60R']
    files = ['ALC_050319_1_41B',]
    for file in files:
        print('Running analysis for '+ file)
        video_file = file+'.avi'
        analysis_file = file+'.analysis'
        output_file = file+'_annotated.avi'
        annotate_video_basic(base=base,video_file=video_file,analysis_file=analysis_file,output_file=output_file,speedX=1)
        
        video_file = file+'_annotated.avi'
        analysis_file = file+'.analysis'
        output_file = file+'_annotated_skeletonAligned.avi'
        annotate_extra(base=base,video_file=video_file,analysis_file=analysis_file,output_file=output_file,speedX=1)