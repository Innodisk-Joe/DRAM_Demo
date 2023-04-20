import os
import re
import cv2
import time
import numpy as np
import utils
import argmanager
import DRAM
from classification.model import DRAM_plugin

EXTENSIONS=['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.bmp','.BMP']
args=argmanager.get_args()

args.mode="file"
args.iVit=True

csv_path=None
for row in utils.Initialization_config(args.configpath,f'PATH'):
    if row[0]=='csv':
        csv_path=row[1]


dram_manager=DRAM.DRAM_alignment(csv_path)

if args.mode=='file':
    if not os.path.exists(args.dstpath):
        os.makedirs(args.dstpath)
    if os.path.isfile(args.srcpath):
        basename = os.path.basename(args.srcpath)
        if os.path.splitext(basename)[1] in EXTENSIONS :
            frame_dstpath=os.path.join(args.dstpath,os.path.splitext(basename)[0])
            if not os.path.exists(frame_dstpath):
                os.makedirs(frame_dstpath)
            frame=cv2.imread(args.srcpath)
            result,dst_shape=dram_manager.auto_alignment(frame,args.type)
            origin_result=result.copy()
            img_w,img_h=dst_shape[:2]
            transrole=np.array([(img_w)/dram_manager.width,(img_h)/dram_manager.height]).astype(np.float32)
            start_time=time.time()
            for key, value in dram_manager.data[f'{args.type}'].items():
                dataclass_type=re.findall(r'[a-zA-Z]+|\d+',str(key))
                if 'box' in value:
                    dataclass_path= os.path.join(frame_dstpath,dataclass_type[0])
                    if not os.path.exists(dataclass_path):
                        os.makedirs(os.path.join(dataclass_path))
                    box=np.array(value['box']*transrole,dtype=np.int32)
                    plugin_name=f'{str(key)}_{os.path.splitext(basename)[0]}.jpg'
                    cv2.imwrite(f'{os.path.join(dataclass_path,plugin_name)}',result.copy()[box[0][1]:box[1][1],box[0][0]:box[1][0]])
                    font_scale=utils.get_optimal_font_scale(box[1][0]-box[0][0],str(key),cv2.FONT_HERSHEY_PLAIN)
                    cv2.putText(origin_result, str(key), (box[0][0], box[0][1]), cv2.FONT_HERSHEY_PLAIN,font_scale, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.rectangle(origin_result,(box[0][0],box[0][1]), (box[1][0],box[1][1]), (0, 255, 0), 3)
                else:
                    cir_point=np.array(value['point']*transrole,dtype=np.int32)
                    cv2.circle(origin_result, cir_point, 10, (0, 0, 255), 3)
            cv2.imwrite(f'{os.path.join(frame_dstpath,"alignment.jpg")}',result.copy())
            cv2.imwrite(f'{os.path.join(frame_dstpath,"check.jpg")}',origin_result.copy())           
    else:
        for root,dirs,files in os.walk(args.srcpath): 
            dst_path=os.path.join(args.dstpath,os.path.relpath(root,args.srcpath))
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)
            for basename in files:
                frame_dstpath=os.path.join(dst_path,os.path.splitext(basename)[0])
                if not os.path.exists(frame_dstpath):
                    os.makedirs(frame_dstpath)
                if os.path.splitext(basename)[1] in EXTENSIONS :
                    frame_dstpath=os.path.join(dst_path,os.path.splitext(basename)[0])
                    if not os.path.exists(frame_dstpath):
                        os.makedirs(frame_dstpath)
                    frame=cv2.imread(os.path.join(root,basename))
                    result,dst_shape=dram_manager.auto_alignment(frame,args.type)
                    origin_result=result.copy()
                    img_w,img_h=dst_shape[:2]
                    transrole=np.array([(img_w)/dram_manager.width,(img_h)/dram_manager.height]).astype(np.float32)
                    start_time=time.time()
                    for key, value in dram_manager.data[f'{args.type}'].items():
                        dataclass_type=re.findall(r'[a-zA-Z]+|\d+',str(key))
                        if 'box' in value:
                            dataclass_path= os.path.join(frame_dstpath,dataclass_type[0])
                            if not os.path.exists(dataclass_path):
                                os.makedirs(os.path.join(dataclass_path))
                            box=np.array(value['box']*transrole,dtype=np.int32)
                            plugin_name=f'{str(key)}_{os.path.splitext(basename)[0]}.jpg'
                            cv2.imwrite(f'{os.path.join(dataclass_path,plugin_name)}',result.copy()[box[0][1]:box[1][1],box[0][0]:box[1][0]])
                            font_scale=utils.get_optimal_font_scale(box[1][0]-box[0][0],str(key),cv2.FONT_HERSHEY_PLAIN)
                            cv2.putText(origin_result, str(key), (box[0][0], box[0][1]), cv2.FONT_HERSHEY_PLAIN,font_scale, (0, 255, 0), 2, cv2.LINE_AA)
                            cv2.rectangle(origin_result,(box[0][0],box[0][1]), (box[1][0],box[1][1]), (0, 255, 0), 3)
                        else:
                            cir_point=np.array(value['point']*transrole,dtype=np.int32)
                            cv2.circle(origin_result, cir_point, 10, (0, 0, 255), 3)
                    cv2.imwrite(f'{os.path.join(frame_dstpath,"alignment.jpg")}',result.copy())
                    cv2.imwrite(f'{os.path.join(frame_dstpath,"check.jpg")}',origin_result.copy())
   
