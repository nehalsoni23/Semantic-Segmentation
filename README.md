# Semantic Segmentation
### Introduction
The goal of the project is to label the pixels of a road in images using a Fully Convolutional Network (FCN). [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) is used to train the neural network to identify road in an image at pixel level. And a fully convolutional version of a pre-trained vgg model is utilized.

### Following are the list of functions used to build FCN:

1. load_vgg - To load the pre-trained vgg model and weights. In the pre-trained vgg model, a fully connected layer is replaced with a 1x1 convolutional layer in order to preserve spatial information.
2. layers - In this function, 1x1 convolution is used as the first layer to reduce the size of the filter to a number of classes. Subsequent transpose layers are added for upsampling the image along with skip layers.
3. optimize - This function calculates cross entropy loss using labels and uses tensorflow Adam optimize to minimize the loss.
4. train_nn - It takes images, labels, epochs, batch_size and other parameters to train the neural network.
5. run - This function is responsible for creating the session, initializing the parameters and invoking all other functions as specified above.

### Results ###

[image1]: ./images/epoch_1.png "Logs of Epoch 1"

[image2]: ./images/epoch_40.png "Logs of Epoch 40"

[image3]: ./images/epoch_50.png "Logs of Epoch 50"

[image4]: ./images/008.png "Result 1"

[image5]: ./images/013.png "Result 2"

[image6]: ./images/032.png "Result 3"

[image7]: ./images/087.png "Result 4"

<h3 align="center"> Training Logs <h3>

![alt text][image1]

<div>
	<div class="left">![alt text][image2]</div>
	<div class="right">![alt text][image3]</div>
</div>

<h3 align="center"> Output images <h3>

![alt text][image4]

![alt text][image5]

![alt text][image6]

![alt text][image7]


### Setup
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
##### Implement
Implement the code in the `main.py` module indicated by the "TODO" comments.
The comments indicated with "OPTIONAL" tag are not required to complete.
##### Run
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

### Submission
1. Ensure you've passed all the unit tests.
2. Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view).
3. Submit the following in a zip file.
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Newest inference images from `runs` folder  (**all images from the most recent run**)
 
 ### Tips
- The link for the frozen `VGG16` model is hardcoded into `helper.py`.  The model can be found [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip)
- The model is not vanilla `VGG16`, but a fully convolutional version, which already contains the 1x1 convolutions to replace the fully connected layers. Please see this [forum post](https://discussions.udacity.com/t/here-is-some-advice-and-clarifications-about-the-semantic-segmentation-project/403100/8?u=subodh.malgonde) for more information.  A summary of additional points, follow. 
- The original FCN-8s was trained in stages. The authors later uploaded a version that was trained all at once to their GitHub repo.  The version in the GitHub repo has one important difference: The outputs of pooling layers 3 and 4 are scaled before they are fed into the 1x1 convolutions.  As a result, some students have found that the model learns much better with the scaling layers included. The model may not converge substantially faster, but may reach a higher IoU and accuracy. 
- When adding l2-regularization, setting a regularizer in the arguments of the `tf.layers` is not enough. Regularization loss terms must be manually added to your loss function. otherwise regularization is not implemented.