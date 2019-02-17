# -*- coding: utf-8 -*-
"""
Transform the data type from ascii to ubyte format (8 bits unsigned binary) 
and save to new files, which would reduce the data size to 1/3, and would 
save the data transforming time when read by the python

@author: Marmot
"""
from os import getcwd
from os import path
sys.path.append(path.split(getcwd())[0])
from TOOLS.CIKM_TOOLS import *
import numpy as np
import time
import pandas as pd
import os
from multiprocessing import Pool
from multiprocessing import cpu_count

def write_file_multi_thread(input_file,set_name,output_file):
    Img_ind= 0
    value_list = list()
    with open(input_file) as f:
        for content in f:
            Img_ind = Img_ind +1
            print('transforming ' + set_name  + ': '  + str(Img_ind).zfill(5))
            line = content.split(',')
            title = line[0] + '    '+line[1]
            if set_name =="train":
                value_list.append(float(line[1]))
            data_write = np.asarray(line[2].strip().split(' '))
            data_write=data_write.astype(np.ubyte)
            data_write = (data_write + 1).astype(np.ubyte)
            if data_write.max()>255:
                print('Error,too large')
            if data_write.min()<0:
                print('Error,too small')
            f = open(output_file, "ab")
            f.write(data_write.tobytes())
            f.close()
    return value_list

if __name__=="__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    data_folder = r"C:\Users\qiaos\Desktop\CIKM 2017\\"
    set_list = ['train', 'testA', 'testB']
    size_list = [10000, 2000, 2000]
    time1 = time.time()
    value_list = list()
    mypool = Pool(processes=cpu_count())
    result_multi_thread = list()
    for set_name, set_size in zip(set_list, size_list):
        output_file = data_folder + set_name + '_ubyte.txt'
        input_file = data_folder + set_name + '.txt'
        if os.path.exists(output_file):
            os.remove(output_file)
        result_multi_thread.append(mypool.apply_async(write_file_multi_thread, (input_file, set_name, output_file)))
    mypool.close()
    mypool.join()
    for res in result_multi_thread:
        a=res.get()
        if len(a) > 0:
            value_list = a
    time2 = time.time()
    print('total elapse time:' + str(time2 - time1))

    value_list = pd.DataFrame(value_list, columns=['value'])
    value_list.to_csv(data_folder + 'train_label.csv', index=False, header=False)



