# hybrid-learning-neural-network
<h2>HLNN(hybrid-learning-neural-network)</h2>
<div>
<span> HLNN is created for Pattern Recognition and Pattern Transformation.
input patter [x] </span>
<span> HLNN(x) -> Label (PR), or HLNN(x) -> Pattern (PT) </span>
<span> the first requires a fuzzy network; second requires a BP neural network.</span>
<span> HLNN is composed of : </span>
<ul>
<li> 1. Unsupervised Learning Layers (SOM) </li>
<li> 2. Supervised Learning Layers (Baysian Network) </li>
</ul>
</div>

<div>
<h3> Algorithm: </h3>
<ul>
<li> 1. [Training] input patterns with or without labels ( no feedback ) </li>
<li> 2. train patterns into a SOM network, this is called clustering process. </li>
<li> 3.if input pattern has labels, then using output layer of last process 
  to train supervised learning layers, either Baysian or BP network.</li>
<li> 4.if input pattern carrys no label at all, then it is done for current
  pattern training.</li>
<li> 5.[Applying] input patterns to be recognized or transformed. </li>
<li> 6.call procedure of [2] above. </li>
<li> 7.using output state of affected neurons in SOM output layer to update
  supervised network output state, and output it as prediction [y].</li>
</ul>
</div>

<div>
<h3> [notice]</h3>
<p>There're alternatives for this model, we could apply Tree-SOM or Kernel-SOM,
 Neural Gas, to replace SOM, and we could also employ other supervised model
 to in supervised process.
</p>
<span> [Copyright: Xiang Chao (向超)]</span>
<span> [Statement] </span>
<p> This project contains algorithm and model design that only belongs to author,
and these code come from project that accompanies a published paper. If you
want to use it, please contact the auther first at : 
[xiangchao215 at mails dot ucas dot ac dot cn]
(please translate into normal email address by yourself, escape for mail address
spider to avoid spam, thanks!)
</p>
</div>

