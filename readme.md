##环境

python=3.7
tensorflow=1.14

##预处理

1、运行prepocess_to_csv.py可以将mat格式数据转为csv，得到data.csv

##预测

2、运行lstm.py是使用lstm预测太阳黑子数量

运行bilstm.py是使用bilstm

运行lstm_seq2seq.py是运行lstm和seq2seq结合的方法预测

运行cnn_seq2seq.py是运行cnn和seq2seq结合和attention结合的方法预测。