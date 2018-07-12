# caffe_root = 'E:/caffe27'
# caffe的执行文件 ：bin目录下的各种exe文件
# 目录结构
## data
### data/tmp  
主要文件存储

- **train_lmdb**   
训练数据集文件 lmdb格式（label，image） 

- **test_lmdb**   
测试数据集文件 lmdb格式（label，image） 

- **cate.txt**   
label存储文件  格式(idx\tlabel\n)

- **mymodel.caffemodel**   
caffe模型存储 即各个层上的权重参数

- **solver.prototxt**   
整个训练控制文件，最大迭代数量。学习速度

- **test.prototxt**   
测试时的网络结构

- **train_mean.binaryproto**   
训练数据（全体数据?）的RGB？通道均值，二进制文件

- **train_mean.pyc**   
数据的均值文件，pyc格式方便调用

- **train.prototxt**   
训练网络的结构文件
### data/train
训练数数据集、下属目录circle，combine，complex，simple，word，ploygon六个分类，都是自定义的（随便分的）

### last
没有处理的图片

## main
- main_train_plt.py   
主要训练文件，单步训练，生成图表
- main_train.py    
一键训练，极品宝刀点击就送
- main.py   
[官方教程的CSDN翻译版本](https://blog.csdn.net/jnulzl/article/details/52077915)   
照抄的。。。第一次尝试的时候

- test_cate.py   
main.py翻版，测试一下last里面的数据
## pycaffe
全抄的素材代码，，，，，等会删除
## test
各种测试代码，获取目录啊，查看网络结构啊，不方便在跑得时候弄的都在这
## tools
- ConvertMean.py

计算数据文件的mean，转化为为python可以调用的pyc格式

- GetProto.py

代码方式得到train.prototxt，test.prototxt文件，用于定义训练规程中的网络结构

- GetSolver.py

代码方式生成solver.prototxt文件，用于调配整个训练过程，例如最大训练次数，学习率等

- ImgProcess.py

主要就是pre_process_img函数，直方图均值化，重新定义大小

- GetLabel.py

由文件目录，生成label标签,存为文件
