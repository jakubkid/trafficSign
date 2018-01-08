# **Traffic Sign Recognition** 

## Report


[//]: # (Image References)

[ExampleSign]: ./raportPics/ExampleSign.png "Example Sign"
[imagesComparison]: ./raportPics/imagesComparison.png "Images Comparison Test"
[imagesComparisonTrainingSet]: ./raportPics/imagesComparisonTrainingSet.png "Images Comparison Training"
[softmaxColor]: ./raportPics/softmaxColor.png "Softmax Result Color Image"
[softmaxGrayscale]: ./raportPics/softmaxGrayscale.png "Softmax Result Grayscale Image"
[germanSigns]: ./raportPics/germanSigns.png "Softmax Result Grayscale Image"
[networkState]: ./raportPics/networkState.png "network State"


---
### Reflection

#### 1. README
The aim of this project is to create algorithm correctly detecting one of 43 German traffic signs using Convolutional neural network [project code](https://github.com/jakubkid/trafficSign/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Data set summary

I used the pandas library and python build in len function to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32, 32, 3
* The number of unique classes/labels in the data set is 43

#### 2. Sign Example

Here is an example sign from dataset 

![Example Sign][ExampleSign]

### Design and Test Model Architecture

#### 1. Preprocessing

As a first step, I decided to convert the images to grayscale because from my experiments it produced better results. As a next step I normalized image to cover full range from 0 to 255

Here is an example of a traffic sign image before and after gray scaling and normalization.

![Image preprocessing][imagesComparisonTrainingSet]

As a last step, I normalized the (-1.0,-1,0) range to improve neutral network training.


#### 2. Model Architecture.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling 2x2      	| 2x2 stride, outputs 14x14x6					|
| Convolution 5x5	    | 1x1 stride, same padding, outputs10x10x16  	|
| RELU					|												|
| Max pooling 2x2      	| 2x2 stride, outputs 5x5x16					|
| Flatten Layer1&2		| 5x5x16+14x14x6 = 1576							|
| Fully connected		| 1576->120    									|
| RELU					|												|
| Fully connected		| 120->84    									|
| sigmoid				|												|
| Fully connected		| 84->43    									|
 


#### 3. Model training

To train the model, I used AdamOptimizer to minimize loss operation of training set. I set batch size to 128 and with 20 epochs. Learning rate was started with 0.001 and after each epoch it is halved if validation accuracy dropped. See log of training below:

>EPOCH 1 ...
>Validation Accuracy = 0.751
>
>EPOCH 2 ...
>Validation Accuracy = 0.876
>
>EPOCH 3 ...
>Validation Accuracy = 0.918
>
>EPOCH 4 ...
>Validation Accuracy = 0.928
>
>EPOCH 5 ...
>Validation Accuracy = 0.934
>
>EPOCH 6 ...
>Validation Accuracy = 0.937
>
>EPOCH 7 ...
>Validation Accuracy = 0.938
>
>EPOCH 8 ...
>Validation Accuracy = 0.943
>
>EPOCH 9 ...
>Validation Accuracy = 0.949
>
>Learning rate dropped to 0.0005
>EPOCH 10 ...
>Validation Accuracy = 0.946
>
>EPOCH 11 ...
>Validation Accuracy = 0.954
>
>Learning rate dropped to 0.00025
>EPOCH 12 ...
>Validation Accuracy = 0.952
>
>Learning rate dropped to 0.000125
>EPOCH 13 ...
>Validation Accuracy = 0.952
>
>Learning rate dropped to 6.25e-05
>EPOCH 14 ...
>Validation Accuracy = 0.950
>
>EPOCH 15 ...
>Validation Accuracy = 0.951
>
>EPOCH 16 ...
>Validation Accuracy = 0.952
>
>EPOCH 17 ...
>Validation Accuracy = 0.952
>
>EPOCH 18 ...
>Validation Accuracy = 0.952
>
>Learning rate dropped to 3.125e-05
>EPOCH 19 ...
>Validation Accuracy = 0.951
>
>EPOCH 20 ...
>Validation Accuracy = 0.952
>
>Test Accuracy = 0.930
>Train Accuracy = 1.000



#### 4. Architecture

My final model results were:
* Training set accuracy of 1.000
* Validation set accuracy of 0.952
* Test set accuracy of 0.930

The way to final architecture:
* Firstly slightly modified LeNet lab (RGB image input and 43 outputs instead of 10) was chosen 
* This architecture was achieving results around 0.90
* Replacing last RELU with sigmoid was enough to achieve 0.93
* After reading about [example algorithm](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) feed forward from layer1 output to layer3 input was added which improved accuracy to 0.94
* Modifying this model to accept normalized grayscale images improved accuracy to 0.95


### New Images

#### 1. Selected signs.

Here are five German traffic signs that I found on the web:

![German Signs][germanSigns]

20kmh limit might be hard to detect because is captured together with other sign. 30kmh limit might be hard to detect because it is obscured with light. 50kmh limit animal and turn warning might be hard to detect because are very dark.

#### 2. New images predictions

Here are the results of the prediction with grayscale images:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Speed limit 20kmh 	| Speed limit 20kmh  							|
| Speed limit 30kmh  	| Speed limit 30kmh 							|
| Speed limit 50kmh		| Speed limit 50kmh								|
| Warning animal   		| Warning animal 				 				|
| Warning right turn    | Warning right turn    	    				|

Here are the results of the prediction with RGB images:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Speed limit 20kmh 	| Speed limit 20kmh  							|
| Speed limit 30kmh  	| Speed limit 30kmh 							|
| Speed limit 50kmh		| Speed limit 100kmh					    	|
| Warning animal   		| Warning animal 				 				|
| Warning right turn    | Warning right turn    	    				|

Even though RGB validation accuracy was 0.942 (grayscale 0.952) it had troubles with dark images and detected wrongly 50kmh limit.

#### 3. Model certainty 
First five softmax probabilities for grayscale images:

![First five softmax probabilities for grayscale images][softmaxGrayscale]

First five softmax probabilities for RGB images:

![First five softmax probabilities for RGB images][softmaxColor]

From this plots it is visible that model is very sure for it detection when grayscale images are provided. For RGB images model is not certain and sometimes wrong when image is dark.


### Visualizing the Neural Network 
#### 1. Network internal state

Layer 1 and layer 2 state for 50kmh limit:

![network State][networkState]

Layer 1 mainly detects edges and circles, layer 2 output is quite puzzling (black magic).
