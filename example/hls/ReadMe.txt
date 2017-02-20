1:LeNet_v2
	经典的LeNet网络实现MNIST手写字分类。所有权重和中间数据缓存在FPGA BRAM中。

2:LeNet_Stream_v3
	经典的LeNet网络实现MNIST手写字分类。大部分权重和中间结果数据，通过AXI Master总线缓存在DDR中而不是FPGA BRAM中。
	
3:LSTM_v2
	经典的LSTM网络实现MNIST手写字分类。
	