# -*- coding: utf-8 -*-
"""
Package features in step3.2 and step3.3 and save as numpy array

@author: Marmot
"""

import sys
sys.path.append(r"D:\Qiao\github\CIKM-Cup-2017\code")
from TOOLS.CIKM_TOOLS import *

set_name_list = ['train', 'testAB']

for set_name in set_name_list: 
    input_file = data_folder + set_name +'_sample_data'
    input_size = pd.read_csv(data_folder  + set_name + '_sample_size.csv')
    if (set_name =='train'):
        pic_sample = pd.read_csv(data_folder + set_name+ '_pic_sample.csv')

        train_label = pd.read_csv(data_folder  + 'train_label.csv',names= ['value'])
        train_label['PIC_IND'] = train_label.index + 1
        pic_sample = pd.merge(pic_sample, train_label, how = 'left',on = 'PIC_IND')

    if set_name =='testAB':
        pic_sample = pd.read_csv(data_folder + set_name+ 'B_pic_sample.csv')
    
    F3 = pd.read_csv(data_folder + set_name + '_F3.csv')
    pic_sample = pd.merge(pic_sample, F3 , on = 'PIC_IND', how = 'left')
    
    F2 = pd.read_csv(data_folder + set_name + '_F2.csv')
    pic_sample = pd.merge(pic_sample, F2 , on = 'PIC_IND', how = 'left')
    
    #%% general descirption
    velo_ = list( map(lambda x:'V'+ str(x).zfill(2), np.arange(1,7)))
    coord_ = list(map(lambda x:'C'+ str(x).zfill(2), np.arange(1,6)))
    N_centroids = 8
    kp_ = list(map(lambda x:'kp'+ str(x).zfill(2), np.arange(1,N_centroids+1)) )
    hist_ = list(map(lambda x:'H'+ str(x).zfill(2), np.arange(1,4)))
    bin_ = list(map(lambda x:'B'+ str(x).zfill(2), np.arange(1,8))  )
    M_ =list( map(lambda x:'M'+ str(x).zfill(2), np.arange(1,4)) )
    R_ = list(map(lambda x:'R'+ str(x).zfill(2), np.arange(1,3)) )
    N_ =list( map(lambda x:'N'+ str(x).zfill(2), np.arange(1,10 )) )
#    NP_ = list(map(lambda x:'NP'+ str(x).zfill(2), np.arange(2,20 )))
    

    GS_ = velo_  + coord_  + hist_ + bin_ + M_ + R_  + N_ 
    #%% time space description  
    time_diff_list = np.asarray([2,4])
    cover_diff_ = list( map(lambda x:'COV_DIFF'+ str(x).zfill(2), time_diff_list) )
    mean_diff_ = list( map(lambda x:'MEA_DIFF'+ str(x).zfill(2), time_diff_list) )
    std_diff_ = list( map(lambda x:'STD_DIFF'+ str(x).zfill(2), time_diff_list) )
    max_diff_ = list( map(lambda x:'MAX_DIFF'+ str(x).zfill(2), time_diff_list))
    height_diff_list = np.asarray([3,4])
    cover_diff_H = list( map(lambda x:'COV_DIFF_H'+ str(x).zfill(2), height_diff_list) )
    mean_diff_H =list(  map(lambda x:'MEA_DIFF_H'+ str(x).zfill(2), height_diff_list) )
    std_diff_H = list( map(lambda x:'STD_DIFF_H'+ str(x).zfill(2), height_diff_list) )
    max_diff_H = list( map(lambda x:'MAX_DIFF_H'+ str(x).zfill(2), height_diff_list) )
    time_list = np.asarray([11,15])
    cover_ = list( map(lambda x:'COV'+ str(x).zfill(2), time_list) )
    mean_ =list(  map(lambda x:'MEA'+ str(x).zfill(2), time_list) )
    std_ = list( map(lambda x:'STD'+ str(x).zfill(2), time_list) )
    max_ = list( map(lambda x:'MAX'+ str(x).zfill(2), time_list) )
        
    TS_ = cover_diff_ + mean_diff_ + std_diff_ + max_diff_ + cover_diff_H + mean_diff_H + std_diff_H + max_diff_H + cover_ + mean_ + std_ + max_
    TS_ = cover_diff_  + max_diff_ + cover_diff_H   + max_diff_H + cover_ + mean_ + max_   
    
    feature_list = velo_  + coord_  + hist_ + bin_ + M_ + R_  + N_ 
    feature_list = GS_ + TS_
    
    feature_list_pd = pd.read_csv(data_folder + set_name + '_image_PICIND.csv')
    pic_sample = pic_sample[pic_sample.PIC_IND.isin(feature_list_pd.PIC_IND)]
    
    X_ = pic_sample[feature_list].values
    if set_name =='train':    
        scaler = preprocessing.MinMaxScaler().fit(X_)
        X = scaler.transform(X_) 
    if set_name =='testAB':       
        X = scaler.transform(X_)   
    np.save(data_folder + set_name + '_flat.npy',X) 
