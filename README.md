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
<table>
	<tr>
		<th>TOPIC ORDER</th>
		<th>TOPIC CONTENT</th>
		<th>EXPERIMENT ABSTRACT</th>
	</tr>
	<tr>
		<td>001</td>
		<td>SINGLE-LAYER MODEL</td>
		<td>CLASSIFICATION OF RANDOMLY GENERATED GRAPH</td>
	</tr>
	<tr>
		<td>002</td>
		<td>HYBRID LEARNING ON DEEP-NEURAL-NETWORK</td>
		<td>OBJECT RECOGNITION ON IMAGES USING DHLNN AND DBPNN</td>
	</tr>
	<tr>
		<td>003</td>
		<td>HLNN APPLICATION ON ROBOTICS CONTROL SYSTEM</td>
		<td>PATH PLAN APPLICATION USING DHLNN AND DNN WITH THE AID OF SENSOR-ARRAY AND ROTOR-ARRAY</td>
	</tr>
	<tr>
		<td>004</td>
		<td>HLNN EXTENSION : A SIGNAL ORIENTED MODEL COMPARING WITH LSTM</td>
		<td>APPLICATION OF SIGNAL-ORIENTED HLNN WORKING ON TEXT UNDERSTANDING COMPARING WITH LSTM MODEL</td>
	</tr>
</table>
</div>

<div>
<h3> [notice]</h3>
<p>There're alternatives for this model, we could apply Tree-SOM or Kernel-SOM,
 Neural Gas, to replace SOM, and we could also employ other supervised model
 to in supervised process.
</p>
<span> [Copyright: Yamasaki Hiroki(山崎ひろき)</span>
<span> [Statement] </span>
<p> This project contains algorithm and model design that only belongs to the author,
and these code come from project that accompanies a published paper. If you
want to use it, please contact the auther first at : <br/>
[yamasaki\_hiroki\_kyoto at yahoo dot co dot jp] <br/>
(please translate into normal email address by yourself, escape for mail address
spider to avoid spam, thanks!)
</p>
</div>

