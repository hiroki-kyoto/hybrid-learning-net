# hybrid-learning-neural-network
HLNN(hybrid-learning-neural-network)
HLNN is created for Pattern Recognition and Pattern Transformation.
input patter [x]
HLNN(x) -> Label (PR), or HLNN(x) -> Pattern (PT)
the first requires a fuzzy network; second requires a BP neural network.
HLNN is composed of :
1.Unsupervised Learning Layers (SOM)
2.Supervised Learning Layers (Baysian Network)

Algorithm:
1.[Training] input patterns with or without labels ( no feedback )
2.train patterns into a SOM network, this is called clustering process.
3.if input pattern has labels, then using output layer of last process
  to train supervised learning layers, either Baysian or BP network.
4.if input pattern carrys no label at all, then it is done for current
  pattern training.
5.[Applying] input patterns to be recognized or transformed.
6.call procedure of [2] above.
7.using output state of affected neurons in SOM output layer to update
  supervised network output state, and output it as prediction [y].

[notice]
There're alternatives for this model, we could apply Tree-SOM or Kernel-SOM,
Neural Gas, to replace SOM, and we could also employ other supervised model
to in supervised process.
[Copyright: Xiang Chao (向超)]
[Statement]
This project contains algorithm and model design that only belongs to author,
and these code come from project that accompanies a published paper. If you
want to use it, please contact the auther first at : xiangchao215@mails.ucas.ac.cn.

