import cv2
import tqdm
import collections
import sys
import os
import pandas as pd
import numpy as np
from scipy.ndimage import median_filter as medfilt


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
        feature_x[feature_p<p_cutoff] = np.nan
        feature_y[feature_p<p_cutoff] = np.nan
        
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


def annotate_video(base=r'C:\Users\balaji\Desktop\ALC_OF',video_file='ALC_050319_1_41B.avi',analysis_file='ALC_050319_1_41B.analysis',
                   output_file='ALC_050319_1_41B_annotated.avi',speedX=1,
                   skeleton=[('centroid','snout'),('l_ear','r_ear'),('box_tl','box_tr'),('box_tr','box_br'),('box_br','box_bl'),('box_bl','box_tl')]):
    
    analysis = os.path.join(base,analysis_file)
    video = os.path.join(base,video_file)
    output = os.path.join(base,output_file)
    
    
    # get the analysis
    df = pd.read_pickle(analysis)
    
    # get points of interest
    points = set()
    for bone in skeleton:
        points = points.union({*bone})
    print(points)
    
    out = filter_from_df(df,features=points) # points=('centroid','l_ear','r_ear')
    
    
    # import pdb; pdb.set_trace()
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
            
            # if not np.isnan(out['centroid']['x'][idx]):
                # img = cv2.circle(img,(int(out['centroid']['x'][idx]),int(out['centroid']['y'][idx])),5,(0,0,255))
            # if not np.isnan(out['l_ear']['x'][idx]):
                # img = cv2.circle(img,(int(out['l_ear']['x'][idx]),int(out['l_ear']['y'][idx])),5,(0,0,255))
            # if not np.isnan(out['r_ear']['x'][idx]):
                # img = cv2.circle(img,(int(out['r_ear']['x'][idx]),int(out['r_ear']['y'][idx])),5,(0,0,255))
            
            # add to writer
            writer.write(img)
    except Exception as er:

        raise er.with_traceback(sys.exc_info()[2])
    finally:
        # for index, row in df.iterrows():
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
        
if __name__=='__main__':
    base = r'C:\Users\bsriram\Desktop\Data\ACM_Data\OpenField'
    annotate_video(base = base,video_file='ALC_050319_1_41B_OF.avi', analysis_file='ALC_050319_1_41B_OF.analysis',
                   output_file='ALC_050319_1_41B_OF_annotated.avi',speedX=3)