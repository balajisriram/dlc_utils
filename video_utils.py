import cv2
import tqdm
import collections
import sys
import pandas as pd


def filter_from_df(df, p_cutoff=0.95, smoothing=('median_filter',0.2), points=('centroid','l_ear','r_ear')):
    resnet_name = df.keys()[0][0]
    
    for feature in features:
        feature_x = df[(resnet_name,feature,'x')]
        feature_y = df[(resnet_name,feature,'y')]
        feature_p = df[(resnet_name,feature,'likelihood')]

def annotate_video(video=r'C:\Users\balaji\Desktop\ALC_OF\ALC_050319_1_41B.avi',analysis=r'C:\Users\balaji\Desktop\ALC_OF\ALC_050319_1_41B.analysis',
    output=r"C:\Users\balaji\Desktop\ALC_OF\ALC_050319_1_41B_annotated.avi",speedX=10,):
    
    df = pd.read_hdf(analysis)
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