# -*- coding: utf-8 -*-
"""
Model ensemble and submit  

@author: Marmot
"""

import sys
if len(sys.argv) > 1:
    sys.path.append(sys.argv[1])
else:
    sys.path.append(r"D:\Qiao\github\CIKM-Cup-2017\code")
from TOOLS.CIKM_TOOLS import *

cnn = pd.read_csv(data_folder + 'result_cnn.csv')
nn = pd.read_csv(data_folder + 'result_nn.csv')
gbdt = pd.read_csv(data_folder + 'result_gbdt.csv')
result_full = 0.8*cnn+ 0.1*gbdt + 0.1*nn

nnpatch = pd.read_csv(data_folder + 'result_nnpatch.csv')
gbdtpatch = pd.read_csv(data_folder + 'result_gbdtpatch.csv')
result_patch = 0.8*nnpatch + 0.2*gbdtpatch

result = pd.concat([result_full,result_patch])

result = result.sort_values(by = 'PIC_IND' , ascending = [1])

submit0 = pd.DataFrame({'PIC_IND':np.arange(1,2001,1.0)})
submit0 = pd.merge(submit0,result, how = 'left' , on = 'PIC_IND')
submit0 = submit0.fillna(method = 'ffill')

submit0 = submit0.sort_values(by = ['PIC_IND'],ascending = [1])
print(submit0)
submit0['value'].to_csv(data_folder + 'submit.csv', header = False , index = False)
#submit0['value'].to_csv('../submit_official/' + 'submit.csv', header = False , index = False)
print(submit0.mean())