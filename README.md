#### imu-predict

说明：
	１．文件夹ground_truth　　采用数据集euroc 中的V1_01_easy 作为整个实验的测试 
	２．文件夹operation_test　一个简单的训练网络，目的是验证当前旋转lossfunction　的可行性
	

#   step 1:
#	６月１３日——６月２０日
	建立lossfunction 回归旋转（四元数表示）q
如果lossfunction 建立正确，则改写informer进行预测


#　 step 2:
#       ６月２０日－６月２７日
	回归预积分那几个量，目前已经从vins－mono中获取预积分。但该预积分是否正确还需验证



