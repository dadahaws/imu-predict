#### imu-predict

##文件夹说明：
####	1．文件夹informer_op中的　main_informer_quat.py 是整个模型的入口
######训练命令　python -u main_informer_quat.py --model informer --data euroc_data


##实验思路：
实验动机来源于以下两篇文章：
１．IDOL: Inertial Deep Orientation-Estimation and Localization　AAAI 2021
本篇文章的imu预测模型分为两个部分。
第一部分预测旋转，网络采用LSTM。输入原始imu数据，加速度量测值，角速度量测值，磁力计量测值。输出以四元数表示的旋转(四维向量)
损失函数见论文。使用tensor定义好四元数的减法乘法运算即可。
第二部分是回归位移偏差量。这个思路与TILO,RONIN　基本一致。网络采用双向LSTM. 输入是经过降噪后的加速度值与角速度值（注意此处这两个值使用的网络是预测旋转的一部分）

２．IMU Data Processing For Inertial Aided Navigation: A Recurrent Neural Network Based Approach　ICRA 2021
本篇文章Ｉｍｕ预测模型回归的是预积分那三个中间量，α，β，γ。同样分为两个部分。
第一部分是使用神经网络对加速度，角速度进行降噪处理。详见这篇文章　Denoising IMU Gyroscopes with Deep Learning for Open-Loop Attitude Estimation
第二部分是采用传统方式（本篇文章文献１）积分，作者给出不用网络进行积分的原因是网络效果较差。（个人认为是可以的，比如使用transformer）

总结思路如下：
结合以上两篇文章的优点，分为以下两个步骤：
１．单独预测旋转
２．使用informer进行位移或者预积分值的预测。


	
##实验计划
###   step 1:
###	６月１３日——６月２０日
	建立lossfunction 回归旋转（四元数表示）q
如果lossfunction 建立正确，则改写informer进行预测

###　 step 2:
###      ６月２０日－６月２７日
	回归预积分那几个量，目前已经从vins－mono中获取预积分。但该预积分是否正确还需验证

###    总结前两周：
lossfunction 建立完成，但是与作者提出的略有出入，是否需要将四元数表示换算到so3中？，但是整体目标达到。采用四元数的方式回归旋转向量。
模型改造完毕,基本掌握了transformer的修改方法，删除了原作informer的对时间序列的处理（用不到），以及词向量维度过高等问题。下一步是根据数据集，精细化调整模型结构。
接下来要解决两个问题：1.是否能通过解码器直接预测出位移？　如果可以，imu预测位姿问题得到解决。２：是否能解决静态输出位移。


#### step 3:
###    6/27-7/4
###制作数据：现在需要的数据有　１．预积分的几个真值 





