import cv2
import threading
import time
from utils import stop_thread

class Video_Stream(object):
    def __init__(self,uri,stream_type):
        self.uri=uri
        self.stream_type=stream_type
        self.source_ready=False
        self.status=False
        self.stopped=False
        self.analysis_status=False
        self.analysis_display=True
        self.frame_lock=threading.RLock()
        self.result_frame_lock=threading.RLock()
        self.Webcamsource=threading.Thread(target=self.source, args=())
        self.Webcamresult=threading.Thread(target=self.result, args=())
        if self.stream_type=="rtsp":
            gstreamer_pipeline = ('rtspsrc location={} latency=200 ! rtph264depay ! decodebin3 '
                                  '! videoconvert ! video/x-raw, format=BGR ! appsink drop=1').format(self.uri)
        elif self.stream_type=="camera":
            gstreamer_pipeline = ('v4l2src device={} ! video/x-raw, width=5472, height=3648, format=BGR ! decodebin3 '
                                  '! videoconvert ! appsink drop=1').format(self.uri)
        try:
            self.stream = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)
        except:
            raise Exception("videocapture error")
        self.ret, self.frame = self.stream.read()
        self.result_frame=self.frame.copy()
        self.Webcamsource=threading.Thread(target=self.source, args=(),daemon=True)
        self.Webcamresult=threading.Thread(target=self.result, args=(),daemon=True)
    
    def start(self):
        self.Webcamsource.start()
        time.sleep(1e-5)
        self.Webcamresult.start()
        
    def source(self):    
        while True:
            start_time=time.time()         
            if self.stopped:
                break
            ret, frame = self.stream.read()
            if ret==True:
                self.frame_lock.acquire()
                try:
                    self.ret=ret
                    self.frame=frame   
                finally:
                    self.frame_lock.release()
                    self.status=True
            else:
                self.status=False
                print("no_video wait 15 sec")
                time.sleep(10)
                self.stream.release()
                del self.stream
                time.sleep(5)
                if self.stream_type=="rtsp":
                    gstreamer_pipeline = ('rtspsrc location={} latency=200 ! rtph264depay ! decodebin3 '
                                  '! videoconvert ! video/x-raw, format=BGR ! appsink drop=1').format(self.uri)
                elif self.stream_type=="camera":
                    gstreamer_pipeline = ('v4l2src device={} ! image/jpeg, width=1280, height=720, framerate=30/1 ! decodebin3 '
                                  '! videoconvert ! video/x-raw, format=BGR ! appsink drop=1').format(self.uri)
                try:
                    self.stream = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)
                except:
                    raise Exception("videocapture error")
            if((1/30)-(time.time()-start_time))>0:
                time.sleep(abs((1/30)-(time.time()-start_time)))       
            else:
                continue 

    def source_read(self):
        self.frame_lock.acquire()
        try:
            if self.status==True:               
                ret=self.ret
                frame=self.frame.copy()   
            else:  
                ret=False
                frame=self.frame.copy()
        finally:
            self.frame_lock.release()
            return ret, frame


    def clear_frame_read(self,resolution,format):
        if self.status==True:
            ret, frame = self.source_read() 
            width=frame.shape[1]
            if resolution=="source":
                image=cv2.imencode(f".{format}",frame)[1]
            else:
                resize=int(resolution)/width
                if resize>0:
                    image=cv2.resize(frame,None,fx=resize,fy=resize,interpolation=None)
                    image=cv2.imencode(f".{format}",image)[1]
                else:
                    image=cv2.imencode(f".{format}",frame)[1]
        else:
            ret=False
            frame=self.frame.copy()
        return ret, image

    def result(self):    
        while True:
            start_time=time.time() 
            if self.status==True:
                if self.ret:
                    if self.stopped:
                        break
    
                    self.result_frame_lock.acquire()
                    try:
                        ret, self.result_frame = self.source_read()
                        if self.analysis_status == True:
                            if self.analysis_model.face_recognition_ready==True:
                                analysis_result=self.analysis_model.analtysis_read()
                                if self.analysis_display == True:
                                    self.result_frame=self.analysis_model.analtysis_draw_bbox(self.result_frame,analysis_result)
                    finally:
                        self.result_frame_lock.release()
            if((1/30)-(time.time()-start_time))>0:
                time.sleep(abs((1/30)-(time.time()-start_time)))
            else:
                time.sleep(0.00000001)
                continue    
           
    
    def result_read(self):
        self.result_frame_lock.acquire()
        try:
            ret=self.ret
            frame=self.result_frame.copy()[50:1875,200:4450]
        finally:
            self.result_frame_lock.release()
            return ret,frame
    
    def set_analysis_model(self,analysis_model:None):
        try:
            if analysis_model != None:
                self.analysis_model=analysis_model
                self.analysis_status=True
                return True
            else:
                return False
        except:
            return False
    
    def stop(self):
        self.stopped = True       
        if self.Webcamsource.is_alive() == True:
            stop_thread(self.Webcamsource)
        if self.Webcamresult.is_alive() == True:
            stop_thread(self.Webcamresult)
        self.stream.release() 
    
if __name__ == '__main__':
    src=f'/dev/video0'
    VideoStream = Video_Stream(src,"camera")
    #src=f'rtsp://admin:admin@172.16.21.1:554/snl/live/1/1/n'
    #VideoStream = Video_Stream(src,"rtsp")
    VideoStream.start()
    cv2.namedWindow("Video",cv2.WINDOW_NORMAL)
    while True:
        ret,frame=VideoStream.result_read()
        start_time=time.time()
        if ret:
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('s'):
                save_time=time.time()
                msec_time=str(save_time).split(".")[1]
                local_time = time.localtime(int(save_time))
                cv2.imwrite(f'IMG_{time.strftime("%Y%m%d%H%M%S", local_time)}{msec_time}.jpg',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                VideoStream.stop()
                break
        if((1/30)-(time.time()-start_time))>0:
            # print(((1/30)-(time.time()-start_time)))
            time.sleep(abs((1/30)-(time.time()-start_time)))
        else:
            # print(((1/30)-(time.time()-start_time)))
            continue 
    print("Ahoy")
        
