import cv2
import os
import numpy as np
from numpy import *
import sys
os.chdir(sys.path[0])  #vscode读取相对路径
np.set_printoptions(threshold=np.inf)  #numpy显示全部矩阵


def changeImg(path):  #将图像处理成相同大小的像素块,再拉成一维数组
    img = cv2.imread(path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  #灰度化
    ret,thresh=cv2.threshold(img,127,255,cv2.THRESH_BINARY)  #二值化
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  #轮廓检测
    #cv2.drawContours(img,contours,-1,(0,0,255),1)  #画出轮廓
    #print('轮廓数量',len(contours)) #轮廓数量,这里的图片都是有三个轮廓的
    #print('轮廓',contours[1]) 
    x,y,w,h = cv2.boundingRect(contours[1])  #有三个轮廓，第一个是最外层的，第二个是数字边缘的，所以i取1
    #print('最终裁剪的xywh',[x,y,w,h])
    img =img[y:y+h,x:x+w]  #裁剪图像
    #cv2.imshow("裁剪后的Img", img)  #裁剪后的图像
    #print(img)
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_NEAREST)#标准化图像大小，28*28
    ##ret,thresh=cv2.threshold(img,127,255,cv2.THRESH_BINARY)  #resize后发现像素值可能并非全是0和255还需要二值化，可能是插值法的问题
    ##img = thresh
    #cv2.imshow("标准化后的Img", img)  #标准化后的图像
    #print('标准化后的Img',img)
    img = img.ravel()  #拉成一维数组
    for i in range(784):  #将255改成1*784的一维数组
        if img[i] == 255:
            img[i] = 1
    #print(img)  #即一维的只有0和1的图像数组
    return img

#path = 'train_picture/0-1.bmp'
#changeImg(path)

#'''
def knn(k,testdata,traindata,labels):  #欧氏距离模板匹配算法
    traindatasize=traindata.shape[0]#获取行数
    dif=tile(testdata,(traindatasize,1))-traindata #将行数扩展和训练集一样，并求数组差
    sqdif=dif**2 #每个值的平方  
    sumsqdif=sqdif.sum(axis=1) #每一行求和,得到一个记录了每一行和的一维数组
    distance=sumsqdif**0.5  #开根号
    sortdistance=distance.argsort() #排序返回数组下标,argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引号)
    count={}
    for i in range(10):#count清零
        count[i]=0
    for i in range(0,k):
        vote=labels[sortdistance[i]]
        count[vote]+=1 
    sortcount=sorted(count.items(),key=lambda x:x[1],reverse=True) #对众数按第二个元素进行降序排序
    return sortcount[0][0]

def seplabel(fname):#读取文件名字里的真实数字
    filestr=fname.split(".")[0]#提取文件名，返回标签值
    labels=int(filestr.split("-")[0])
    return labels

def traindata():
    labels=[]
    trainfile = os.listdir('train_picture/')  #训练文件夹下的图片文件
    num = len(trainfile)  #因该有九个图像文件
    #创建一个数组存放训练数据，行为文件总数，列为784，为一个手写体的内容 zeros创建规定大小的数组
    trainarr = zeros((num,784))  #里面放图像数组为一行，num列
    for i in range(0,num):
        thisfname = trainfile[i]  #求取文件名
        thislabel = seplabel(thisfname)  #读出真实数字
        labels.append(thislabel)
        trainarr[i] = changeImg('train_picture/'  + thisfname)
    return trainarr,labels  #返回文件的内容和名字里的值
   #将训练集的灰度图像信息保存在一个数组里

def text(): #用文件夹中1-9的图训练，用第十张图测试
    trainarr,labels=traindata()  #返回文件的内容和名字里的值
    testlist=os.listdir('text_picture/') #获取测试文件名
    tnum=len(testlist)#测试集总数
    right_count=0
    for i in range(tnum):
        thisname=testlist[i]
        test_arr=changeImg("text_picture/" + thisname)
        rknn = knn(k=3,testdata=test_arr,traindata=trainarr,labels=labels)
        lab = seplabel(str(thisname))
        if rknn == lab:
            right_count += 1
            print(str(thisname) + "  :  " + str(rknn)+'识别成功')
        else:
            print(str(thisname) + "  :  " + str(rknn) + '识别失败')
    print('正确率为: {:.2%}'.format(right_count/tnum))

text()  #开始测试
#'''
cv2.waitKey(0)
cv2.destroyAllWindows()