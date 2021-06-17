#### imu-predict

##文件夹说明：
####	１．文件夹ground_truth　　采用数据集euroc 中的V1_01_easy 作为整个实验的测试 
####	２．文件夹operation_test　一个简单的训练网络，目的是验证当前旋转lossfunction　的可行性

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
##   step 1:
###	６月１３日——６月２０日
	建立lossfunction 回归旋转（四元数表示）q
如果lossfunction 建立正确，则改写informer进行预测

##　 step 2:
###      ６月２０日－６月２７日
	回归预积分那几个量，目前已经从vins－mono中获取预积分。但该预积分是否正确还需验证



