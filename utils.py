import os
import csv
import time
import re
import cv2
import inspect
import ctypes
import configparser
from typing import Iterable

def Initialization_config(path,header):
    try:
        config = configparser.ConfigParser()
        config.read(path)
        config_items=config.items(header)      
        return config_items
    except:
        return False 

def get_optimal_font_scale(width,text,fontface):
    for scale in range(59,10,-1):
        textSize = cv2.getTextSize(text, fontFace=fontface, fontScale=scale/10, thickness=1)
        new_width = textSize[0][0]
        if (new_width < width*0.8):
            return scale/10
    return 1

# def csvdatabase(path):
#     if os.path.isfile(path):
#         result={}
#         with open(path, newline='') as csvfile:
#             rows = csv.reader(csvfile)
#             for row in rows:
#                 if re.match('[a-zA-Z]+\d+',row[0]):
#                     result[f'{row[0]}']=row[1:]
#         return result
#     else:
#         raise BaseException('no csv')

def list_flatten(items):
    """Yield items from any nested iterable; see Reference."""
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            for sub_x in list_flatten(x):
                yield sub_x
        else:
            yield x

def save_frame(frame):
    save_time=time.time()
    msec_time=str(save_time).split(".")[1]
    local_time = time.localtime(int(save_time))
    cv2.imwrite(f'IMG_{time.strftime("%Y%m%d%H%M%S", local_time)}{msec_time}.jpg',frame)

def _async_raise(tid, exctype):
    """raises the exception, performs cleanup if needed"""
    tid = ctypes.c_long(tid)
    if not inspect.isclass(exctype):
        exctype = type(exctype)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        # """if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"""
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")


def stop_thread(thread):
    _async_raise(thread.ident, SystemExit)