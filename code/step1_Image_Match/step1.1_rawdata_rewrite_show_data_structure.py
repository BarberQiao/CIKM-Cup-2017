#study from here: https://tianchi.aliyun.com/forum/postDetail?spm=5176.12282027.0.0.7d03311fbi5iFb&postId=980
from matplotlib import pyplot
from numpy import fromfile
from numpy import ubyte
H_ind = 4
T_ind = 15
TH_ind = 59
Img_size = 101

demo_file_path = r"C:\Users\qiaos\Desktop\CIKM 2017\testB_ubyte.txt"
f=open(demo_file_path,"r")
data = fromfile(f,count = Img_size*Img_size,dtype=ubyte)
print(data.shape)
data = data.reshape(Img_size,Img_size)
print(data.shape)
pyplot.imshow(data.reshape(Img_size,Img_size))
pyplot.show()
f.close()