import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter as medfilt
import cv2 
import sys

def euclidean_dist(p1,p2):
    return np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

def amplitude(v):
    return np.sqrt(v[0]**2+v[1]**2)

def is_odd(num):
   return num % 2 != 0
   
def get_all_videos_in_session(folder):
    videos = [f.split('.',1)[0] for f in os.listdir(folder) if f.endswith('avi')]
    returned_list = sorted(videos,key=lambda e: int(e[8:]))
    return returned_list
    
def angle_between(v1,v2,type='degrees'):
    cos_ang = (v1[0]*v2[0]+v1[1]*v2[1])/(amplitude(v1)*amplitude(v2))
    if type=='degrees':return np.degrees(np.arccos(cos_ang))
    elif type=='radians':return np.arccos(cos_ang)

def load_session_details(loc = r'C:\Users\User\Desktop\Code\dlc_utils\SIandNOR_details.csv'):
    data = pd.read_csv(loc)
    data = data.set_index('id')
    return data

def get_subject_from_session(sess):
    splits = sess.split('_')
    sub = splits[0]
    for split in splits[1:-1]:
        sub = sub+'_'
        sub = sub+split
    return sub
    
def get_body_length(df, p_cutoff=0.95,strategy='median_filter',frame_rate=20.0):
    resnet_name = df.keys()[0][0]
    centroid_x = df[(resnet_name,'centroid','x')]
    centroid_y = df[(resnet_name,'centroid','y')]
    centroid_p = df[(resnet_name,'centroid','likelihood')]
    
    l_ear_x = df[(resnet_name,'l_ear','x')]
    l_ear_y = df[(resnet_name,'l_ear','y')]
    l_ear_p = df[(resnet_name,'l_ear','likelihood')]
    
    r_ear_x = df[(resnet_name,'r_ear','x')]
    r_ear_y = df[(resnet_name,'r_ear','y')]
    r_ear_p = df[(resnet_name,'r_ear','likelihood')]
    
    good_data = (centroid_p>p_cutoff) & (l_ear_p>p_cutoff) & (r_ear_p>p_cutoff)
    
    # remove data from the initial section - there is no animal and estimates are terrible
    # criterion is atleast 1 second of reliable data 
    remove_until = 0
    for i in range(1,len(good_data)):
        if np.any(good_data[:i]) and np.all(good_data[i:i+int(frame_rate)]):
            remove_until=i
            break
    body_length = np.full((df.shape[0],1),np.nan)
    body_length_arrow_start = [(k,l) for k,l in zip(body_length,body_length)]
    body_length_arrow_end = [(k,l) for k,l in zip(body_length,body_length)]
    
    # import pdb
    # pdb.set_trace()
    # get the distance for the other data
    for i in range(remove_until,len(good_data)):
        body_length_arrow_start[i] = (int(centroid_x[i]),int(centroid_y[i]))
        body_length_arrow_end[i] = (int((l_ear_x[i]+r_ear_x[i])/2),int((l_ear_y[i]+r_ear_y[i])/2))
        body_length[i] = euclidean_dist(body_length_arrow_start[i],body_length_arrow_end[i])
    if strategy=='raw':
        # do nothing to the distance
        pass
    elif strategy=='median_filter':
        filter_duration = 0.2 # 200 ms
        filter_dur_in_frames = int(filter_duration*frame_rate)
        if not is_odd(filter_dur_in_frames):filter_dur_in_frames = filter_dur_in_frames+1
        body_length = medfilt(body_length,size=filter_dur_in_frames)
    else:
        raise ValueError("Unknownstrategy= {0}. Must be one of 'raw' ,'median_filter'".format(strategy))
    df['body_length'] = body_length
    df['bl_arrow_start'] =  body_length_arrow_start  
    df['bl_arrow_end'] =  body_length_arrow_end  
    return df
    

def get_analysis_dfs(folder,vids):
    full_df = pd.DataFrame()
    for v in vids:
        analysis = [f for f in os.listdir(folder) if f.startswith(v+'DeepCut') and f.endswith('.h5')]
        df = pd.read_hdf(os.path.join(folder,analysis[0]))
        df['frame_number'] = np.arange(df.shape[0])
        df['source'] = v
        
        if full_df.empty: full_df = df
        else: full_df = full_df.append(df,ignore_index=True)
    return full_df

def plot_centroid(df,bodypart='centroid',p_cutoff = 0.95,strategy='remove_uncertain',linewidth=0.5,time_filter=None,frame_rate=20.0):
    resnet_name = df.keys()[0][0]
    x = df[resnet_name,bodypart,'x']
    y = df[resnet_name,bodypart,'y']
    p = df[resnet_name,bodypart,'likelihood']
    
    good_x = x[p>p_cutoff]
    good_y = y[p>p_cutoff]
    if time_filter:
        good_x = good_x[:int(time_filter*frame_rate)]
        good_y = good_y[:int(time_filter*frame_rate)]
    plt.plot(good_x,-good_y,'k',linewidth=linewidth)
    plt.axis('equal')
    plt.show()
    
def plot_pos_vs_frame(df,bodypart='centroid',p_cutoff = 0.95,strategy='remove_uncertain'):
    resnet_name = df.keys()[0][0]
    x = df[resnet_name,bodypart,'x']
    y = df[resnet_name,bodypart,'y']
    p = df[resnet_name,bodypart,'likelihood']
    frame = df.index
    
    good_frame = frame[p>p_cutoff]
    good_x = x[p>p_cutoff]
    good_y = y[p>p_cutoff]
    plt.plot(good_frame,good_x,'r')
    plt.plot(good_frame,good_y,'g')
    plt.legend(['x','y'])
    plt.show()
    
def compress_video(inp,out):
    try:
        cap = cv2.VideoCapture(inp)
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        writer = cv2.VideoWriter(out,fourcc, int(1/cap.get(2)),(int(cap.get(3)),int(cap.get(4))))
        while True:
            succ,img = cap.read()
            if not succ: break
            writer.write(img)
    except Exception as er:
        raise er.with_traceback(sys.exc_info()[2])
    finally:
        # for index, row in df.iterrows():
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

def combine_videos(inps,out):
    try:
        cap = cv2.VideoCapture(inps[0]) # get date from the first video
        print(inps)
        #print(int(1/cap.get(2)))
        print(int(cap.get(3)))
        print(int(cap.get(4)))
        
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        writer = cv2.VideoWriter(out,fourcc, int(1/cap.get(2)),(int(cap.get(3)),int(cap.get(4))))
        for inp in inps:
            cap.release()
            cap = cv2.VideoCapture(inp)
            while True:
                succ,img = cap.read()
                if not succ: break
                writer.write(img)
            print(inp)
    except Exception as er:
        raise er.with_traceback(sys.exc_info()[2])
    finally:
        # for index, row in df.iterrows():
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

if __name__=='__main__':
    folder = r'C:\Data\SocialBehavior\orig'
    vids =  get_all_videos_in_session(folder)
    
    vids_path = [os.path.join(folder,v+'.avi') for v in vids]
    out = os.path.join(folder,'behavCam.avi')
    
    # for vid in vids:
        # compress_video(os.path.join(folder,vid+'.avi'),os.path.join(folder,vid+'_compressed.avi'))
        
    combine_videos(vids_path,out)
    # df = get_analysis_dfs(folder,vids)
    # import pdb
    
    # pdb.set_trace()
    # plot_centroid(df,linewidth=0.25,time_filter=300)