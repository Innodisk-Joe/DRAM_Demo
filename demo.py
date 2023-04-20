import os
import re
import cv2
import time
import numpy as np
import utils
import argmanager
import DRAM
import classification
from multiprocessing.pool import ThreadPool


EXTENSIONS=['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.bmp','.BMP']
args=argmanager.get_args()

args.iVit=True

csv_path=None
for row in utils.Initialization_config(args.configpath,f'PATH'):
    if row[0]=='csv':
        csv_path=row[1]

dram_manager=DRAM.DRAM_alignment(csv_path)

Analysis_Threadpool=ThreadPool(os.cpu_count()*3)
check=time.time()
AImodel_manager={}
tmp=[]
if os.path.isfile(os.path.join(os.getcwd(),f'config.ini')):
    def creat_model(manager,p_type,direction,m_path,ivit):
        manager[f'{p_type}'][f'{direction}']['model'].append(classification.DRAM_plugin(m_path,ivit))
    class_type=None
    for key,path in utils.Initialization_config(os.path.join(os.getcwd(),f'config.ini'),f'AI_model'): 
        class_type=key.split('_')[0]
        if key.split('_')[1]=='h':
            AImodel_manager.setdefault(f'{class_type}',{}).setdefault(f'h',{}).setdefault(f'path',f'{path}')
            AImodel_manager.setdefault(f'{class_type}',{}).setdefault(f'h',{}).setdefault(f'model',[])
            if dram_manager.data['count'][f'{args.type}'][f'{class_type.upper()}']['h']>0:
                model_count=range(dram_manager.data['count'][f'{args.type}'][f'{class_type.upper()}']['h'])
                for i in model_count:
                    # AImodel_manager[f'{class_type}']['h']['model'].append(DRAM_plugin(f'{path}',args.iVit))
                    tmp.append(Analysis_Threadpool.apply_async(creat_model,[AImodel_manager,class_type,'h',f'{path}',args.iVit]))
                    # creat_model(AImodel_manager,class_type,'h',f'{path}',args.iVit)
        elif key.split('_')[1]=='v':
            AImodel_manager.setdefault(f'{class_type}',{}).setdefault(f'v',{}).setdefault(f'path',f'{path}')
            AImodel_manager.setdefault(f'{class_type}',{}).setdefault(f'v',{}).setdefault(f'model',[])
            if dram_manager.data['count'][f'{args.type}'][f'{class_type.upper()}']['v']>0:
                model_count=range(dram_manager.data['count'][f'{args.type}'][f'{class_type.upper()}']['v'])
                for i in model_count:
                    # AImodel_manager[f'{class_type}']['v']['model'].append(DRAM_plugin(f'{path}',args.iVit))
                    tmp.append(Analysis_Threadpool.apply_async(creat_model,[AImodel_manager,class_type,'v',f'{path}',args.iVit]))
                    # creat_model(AImodel_manager,class_type,'v',f'{path}',args.iVit)
for t in tmp:
    t.wait()

print(f'All_model_load_time:{time.time()-check}')

if args.mode == 'file':
    if not os.path.exists(args.dstpath):
        os.makedirs(args.dstpath)

if args.mode=='file':
    if os.path.isfile(args.srcpath):
        cv2.namedWindow("image",cv2.WINDOW_NORMAL)
        basename = os.path.basename(args.srcpath)
        if os.path.splitext(basename)[1] in EXTENSIONS :
            frame=cv2.imread(args.srcpath)
            result,dst_shape=dram_manager.auto_alignment(frame,args.type)
            result_vals=[]
            origin_result=result.copy()
            img_w,img_h=dst_shape[:2]
            transrole=np.array([(img_w)/dram_manager.width,(img_h)/dram_manager.height]).astype(np.float32)
            inference_thread={}
            for key, value in dram_manager.data[f'{args.type}'].items():
                dataclass_type=re.findall(r'[a-zA-Z]+|\d+',str(key))
                if 'box' in value:
                    if dataclass_type[0].lower() in AImodel_manager:
                        box=np.array(value['box']*transrole,dtype=np.int32)
                        if value['direction'] == 'h':
                            count=len(AImodel_manager[dataclass_type[0].lower()]['h']['model'])
                            inference_thread[f'{key}']=Analysis_Threadpool.apply_async(AImodel_manager[dataclass_type[0].lower()]['h']['model'][int(dataclass_type[1])%count].detect,[origin_result.copy()[box[0][1]:box[1][1],box[0][0]:box[1][0]],box])                                                     
                        elif value['direction'] == 'v':
                            count=len(AImodel_manager[dataclass_type[0].lower()]['v']['model'])
                            inference_thread[f'{key}']=Analysis_Threadpool.apply_async(AImodel_manager[dataclass_type[0].lower()]['v']['model'][int(dataclass_type[1])%count].detect,[origin_result.copy()[box[0][1]:box[1][1],box[0][0]:box[1][0]],box])
            start_time=time.time()
            for thread_key,model_val in inference_thread.items():
                result_vals.append(model_val.get())
            if len(result_vals)>0:
                for result_val in result_vals:
                    if re.match('\w+ng',result_val[0]):
                        font_scale=utils.get_optimal_font_scale(result_val[2][2]-result_val[2][0],str(result_val[0]),cv2.FONT_HERSHEY_PLAIN)
                        cv2.putText(result, str(result_val[0]), (result_val[2][0], result_val[2][1]), cv2.FONT_HERSHEY_PLAIN,font_scale, (0, 0, 255), 2, cv2.LINE_AA)
                        cv2.rectangle(result,(result_val[2][0],result_val[2][1]), (result_val[2][2],result_val[2][3]), (0, 0, 255), 3)
                frame=result
            cv2.imshow("image",frame)
            cv2.waitKey(0)
    else:
        cv2.namedWindow("image",cv2.WINDOW_NORMAL)
        for root,dirs,files in os.walk(args.srcpath): 
            dst_path=os.path.join(args.dstpath,os.path.relpath(root,args.srcpath))
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)
            for basename in files:
                if os.path.splitext(basename)[1] in EXTENSIONS :
                    start_time=time.time()
                    filename=os.path.splitext(basename)[0]
                    frame=cv2.imread(os.path.join(root,basename))
                    result,dst_shape=dram_manager.auto_alignment(frame,args.type)
                    result_vals=[]
                    origin_result=result.copy()
                    img_w,img_h=dst_shape[:2]
                    transrole=np.array([(img_w)/dram_manager.width,(img_h)/dram_manager.height]).astype(np.float32)
                    inference_thread={}
                    for key, value in dram_manager.data[f'{args.type}'].items():
                        dataclass_type=re.findall(r'[a-zA-Z]+|\d+',str(key))
                        if 'box' in value:
                            if dataclass_type[0].lower() in AImodel_manager:
                                box=np.array(value['box']*transrole,dtype=np.int32)
                                if value['direction'] == 'h':
                                    count=len(AImodel_manager[dataclass_type[0].lower()]['h']['model'])
                                    inference_thread[f'{key}']=Analysis_Threadpool.apply_async(AImodel_manager[dataclass_type[0].lower()]['h']['model'][int(dataclass_type[1])%count].detect,[origin_result.copy()[box[0][1]:box[1][1],box[0][0]:box[1][0]],box])                                                     
                                    # print(int(dataclass_type[1])%count)
                                elif value['direction'] == 'v':
                                    count=len(AImodel_manager[dataclass_type[0].lower()]['v']['model'])
                                    inference_thread[f'{key}']=Analysis_Threadpool.apply_async(AImodel_manager[dataclass_type[0].lower()]['v']['model'][int(dataclass_type[1])%count].detect,[origin_result.copy()[box[0][1]:box[1][1],box[0][0]:box[1][0]],box])
                                    # print(int(dataclass_type[1])%count)
                    start_time=time.time()
                    for thread_key,model_val in inference_thread.items():
                        result_vals.append(model_val.get())
                    if len(result_vals)>0:
                        for result_val in result_vals:
                            if re.match('\w+ng',result_val[0]):
                                font_scale=utils.get_optimal_font_scale(result_val[2][1][0]-result_val[2][0][0],str(result_val[0]),cv2.FONT_HERSHEY_PLAIN)
                                cv2.putText(result, str(result_val[0]), (result_val[2][0][0], result_val[2][0][1]), cv2.FONT_HERSHEY_PLAIN,font_scale, (0, 0, 255), 2, cv2.LINE_AA)
                                cv2.rectangle(result,(result_val[2][0][0],result_val[2][0][1]), (result_val[2][1][0],result_val[2][1][1]), (0, 0, 255), 3)
                        frame=result
            
                    fps = 1 / (time.time()-start_time)
                    print( "Estimated frames per second : {0}".format(fps))
                    cv2.imwrite(os.path.join(dst_path,f"{filename}_crop.jpg"),frame)
                    cv2.imshow("image",frame)
                    cv2.waitKey(0)
elif args.mode=='camera':
    from Video_Stream import Video_Stream
    webcam=Video_Stream('/dev/video0',args.mode)
    webcam.start()
    cv2.namedWindow("Video",cv2.WINDOW_NORMAL)
    result_vals=[]
    while True:
        ret,frame=webcam.result_read()
        if ret:
            if cv2.waitKey(1) & 0xFF == ord('s'):
                utils.save_frame(frame)
            elif cv2.waitKey(1) & 0xFF == ord('a'):
                try:
                    result,dst_shape=dram_manager.auto_alignment(frame,args.type)
                except:
                    continue
                result_vals=[]
                origin_result=result.copy()
                img_w,img_h=dst_shape[:2]
                transrole=np.array([(img_w)/dram_manager.width,(img_h)/dram_manager.height]).astype(np.float32)
                inference_thread={}
                for key, value in dram_manager.data[f'{args.type}'].items():
                    dataclass_type=re.findall(r'[a-zA-Z]+|\d+',str(key))
                    if 'box' in value:
                        if dataclass_type[0].lower() in AImodel_manager:
                            box=np.array(value['box']*transrole,dtype=np.int32)
                            if value['direction'] == 'h':
                                count=len(AImodel_manager[dataclass_type[0].lower()]['h']['model'])
                                inference_thread[f'{key}']=Analysis_Threadpool.apply_async(AImodel_manager[dataclass_type[0].lower()]['h']['model'][int(dataclass_type[1])%count].detect,[origin_result.copy()[box[0][1]:box[1][1],box[0][0]:box[1][0]],box])                                                     
                            elif value['direction'] == 'v':
                                count=len(AImodel_manager[dataclass_type[0].lower()]['v']['model'])
                                inference_thread[f'{key}']=Analysis_Threadpool.apply_async(AImodel_manager[dataclass_type[0].lower()]['v']['model'][int(dataclass_type[1])%count].detect,[origin_result.copy()[box[0][1]:box[1][1],box[0][0]:box[1][0]],box])
                start_time=time.time()
                for thread_key,model_val in inference_thread.items():
                    result_vals.append(model_val.get())
                result_time=time.time()

            elif cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                webcam.stop()
                break
            
            if len(result_vals)>0:
                for result_val in result_vals:
                    if re.match('\w+ng',result_val[0]):       
                        font_scale=utils.get_optimal_font_scale(result_val[2][1][0]-result_val[2][0][0],str(result_val[0]),cv2.FONT_HERSHEY_PLAIN)
                        cv2.putText(result, str(result_val[0]), (result_val[2][0][0], result_val[2][0][1]), cv2.FONT_HERSHEY_PLAIN,font_scale, (0, 0, 255), 2, cv2.LINE_AA)
                        cv2.rectangle(result,(result_val[2][0][0],result_val[2][0][1]), (result_val[2][1][0],result_val[2][1][1]), (0, 0, 255), 3)
                frame=result
                if time.time()-result_time>10:
                    result_vals=[]
                    result_time=time.time()
            cv2.imshow('Video', frame)