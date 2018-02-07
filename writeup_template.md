# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report



### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 27839
* The size of the validation set is 6960
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the data distribution in the German Traffis sign dataset.
![alt text](https://github.com/z78406/Traffic-sign-classifier/blob/master/2.png)


Here is a visualization of every type of traffig signs.
![alt text](https://github.com/z78406/Traffic-sign-classifier/blob/master/1.png)

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Generally speaking, I just do the work that normalize the data pixel into [-1,1]. As another trial, I try to use the ImageDataGenerator which is provided by Kreas to augment the input data by rotating, translating etc.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution1     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| Relu				|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 2	    | 1x1 stride, valid padding, outputs 10x10x16       									|
| Relu		|         									|
| Max pooling				| 2x2 stride,  outputs 5x5x16        									|
|	Flatten					|								outputs 400				|
|	Fully Connected					|					outputs 120							|
| Leaky Relu |        |
| Dropout    |  |
|	Fully Connected					|					outputs 84							|
| Sigmoid|        |
|	Fully Connected					|					outputs 43							|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I try to use 2 different methods. The first method is traditional training process include foward and backward propogation. I set the batch_size as 128 and use mini-batch SGD to update the weight.The optimizer is AdamOptimizer. Another method is adding data augmentation in the training process. To say it concretly, I use ImageDataGenerato, which is provided in Keras, to randomly rotate,zoom and shift the input data in each batch training stages.
During the training stages, I found that traditional training stages runs much faster than training combing data augmentation( only in CPU mode).
Also the traditional training network guarantees at least 98% accuracy in the training set as well as validation set.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.1%
* validation set accuracy of 98.3% 
* test set accuracy of 92.7%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 
I basically implement the architecture by adding and modifying on the LeNet architecture, which was used to identify the digits input. However, in practice I found that the accuracy was not as good as expectation. To improve it, I try to add dropout layer to reduce overfitting and try a new Rely layer called "leaky-relu" as well as sigmoid function as activation layer. The training accuracy is up to 99.2% and testing accuracy is 92.7% even in an iteration of 20 epoches. I believe that more convolutional layer as well as epoches would continue to improve the testing accuracy.
### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web, which is included in the test_german file:

![alt text](https://github.com/z78406/Traffic-sign-classifier/blob/master/3.png)

For the speed limit sign, I think the difficulty comes from distiguishing the digits within the speed limit catagories.
For the stop entry sign, I think the difficulty comes from distiguishing the signs from other types of prohibit signs.
For the turn right sign, I think the difficulty comes from the lack of enogh samples in the dataset as other common signs.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| speed limit sign(50)     		| speed limit sign(50)   									| 
| stop entry    			| stop entry 										|
| speed limit sign(30) 				|speed limit sign(80) 										|
| Turn right ahead      		| speed limit sign(80) 					 			|
| speed limit sign(20) 		| speed limit sign(80)    							|


The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. This compares much less accurate to the testing accuracy.
Disccusion: The results seems to be disappointed. But it is reasonable. From the prediction result we can see that, the model can recognize the major class type like speed limit signs, but have some difficulty distinguishing the subtle categories, like 20 and 30 speed limit signs. And it also mistakenly identify the turn right sign. I believe some of the reasons come from the fact that the training iteration is not enuogh. I only train the NN in 20 iterations. More iterations may bring the result of more detailed classification. The second reason lies in the fact that I did not add data augmentation in the final because it is time consuming to run in CPU mode. Maybe I can test it later in GPU mode and see if it can bring a improvement in the classification. The third reason maybe that the data distribution is not equal, which means that some of the sign classes may not be recognized as enough as the others, resulting in the wrong classification.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The prediction softmax probabilities are shown below:
![alt text](https://github.com/z78406/Traffic-sign-classifier/blob/master/4.png)

From the output softmax probability we can find that for the first two new signs, the confidence is pretty high. Also the result is correct. Nevertheless, the last three new signs are predicted wrongly. All were considered class 5, which is speed limit 80. This indicates that the model need to be improved either increasing the iteration times or adding more layers. 






