Liner、RNN、LSTM的构造方法\输入\输出构造参数
    pytorch中所有模型分为构造参数和输入和输出构造参数两种类型。

    模型构造参数主要限定了网络的结构，如对循环网络，则包括输入维度、隐层\输出维度、层数；对卷积网络，无论卷积层还是池化层，都不关心输入维度，其构造方法只涉及卷积核大小\步长等。这里的参数决定了模型持久化后的大小.
    输入和输出的构造参数一般和模型训练相关，都需指定batch大小，seq大小（循环网络）\chanel大小（卷积网络），以及输入\输出维度，如果是RNN还需涉及h0​和c0的初始化等。（ps：input_dim在循环网络模型构造中已经指定了，其实可以省略掉，从这点看Pytorch的api设计还是有点乱，包括batch参数的默认位置在cnn中和rnn中也不相同，囧）。这里的参数决定了模型训练效果。

Liner

    Liner(x_dim,y_dim)
    – 输入x，程序输入(batch,x)
    – 输出y, 程序输出(batch,y)
RNN
    对于最简单的 RNN，我们可以使用下面两种方式去调用，分别是 torch.nn.RNNCell() 和 torch.nn.RNN()，这两种方式的区别在于 RNNCell() 只能接受序列中单步的输入，且必须传入隐藏状态，而 RNN() 可以接受一个序列的输入，默认会传入全 0 的隐藏状态，也可以自己申明隐藏状态传入。
    RNN(input_dim ,hidden_dim ,num_layers ，…)
		– input_dim 表示输入 xt的特征维度
		– hidden_dim 表示输出的特征维度，如果没有特殊变化，相当于out
		– num_layers 表示网络的层数
		– nonlinearity 表示选用的非线性激活函数，默认是 ‘tanh’
		– bias 表示是否使用偏置，默认使用
		– batch_first 表示输入数据的形式，默认是 False，就是这样形式，(seq, batch, feature)，也就是将序列长度放在第一位，batch 放在第二位
		– dropout 表示是否在输出层应用 dropout
		– bidirectional 表示是否使用双向的 rnn，默认是 False
	输入xt,h0​,
		– xt​[seq,batch,input_dim],
		– h0[层数×方向,batch,h_dim]
	输出ht,output
		– output[seq,batch,h_dim * 方向]
		– ht​[层数 * 方向，batch,h_dim]
LSTM
	LSTM(x_dim,h_dim,layer_num)
	输入: xt​,(h0​, c_0)
		– xt​(seq,batch,x_dim)
		– (h0, c_0),为每个批次的每个x设置隐层状态和记忆单元的初值，其维度都是（num_layers * num_directions，batch,h_dim）
	输出: output, (hn,cn​)
		– output，每个时刻的LSTM网络的最后一层的输出，维度（seq_len, batch, hidden_size * num_directions）
		– (hn,cn​)，最后时刻的隐层状态和基于单元状态，维度(num_layers * num_directions, batch, hidden_size)

GRU
	GRU(x_dim,h_dim,layer_num,…)
	输入:xt,h0
		xt[seq,batch,x_dim]
		ho[层数×方向,batch,h_dim]
	输出:out,ht
		out[seq,batch,h_dim方向]
		ht[层数方向，batch,h_dim]