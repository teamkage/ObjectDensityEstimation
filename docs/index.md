---
layout: default
---
<h1 align="center"><b>Object Density Estimation</b></h1>

<h4 align="center">Deepak GR, Aashish Sood, Akshay Deshpande, Arvind Hudli </h4>

## **Introduction**

We consider the problem of estimating probabilistic density of objects in images. Given an image, our goal is to recover a density function F as a real function of pixels in the image. Most of the research in the domain are concentrated on the problem of object counting. Although, we aim to address the problem of object counting, we model our problem with preference in preserving the spatial structure of the image. The existing methods focus more on reducing the error in counting and hence, lose the spatial structure of the densities, due to downsample strides in the convolution/pooling operations. By spatial structure, we mean the output of the network is not of same size as the input image, hence there is no 1:1 mapping between the output densities and the input image.

## **Preparing Data**
We will talk about how we structure our data as it is important to understand the later sections. First, we have images which has objects of interest. Next, we annotate these objects with a single dot, naturally at its center of mass. We will now fit a probabilistic density on the object, Gaussian density in this case, centered around the dot annotation on the object. Gaussian density makes natural sense because the probability to find the object is more at the center and then decreases as you move away from the objects center of mass. We do this by convolving our dot annotate image with an Guassian kernel of a suitable sigma. The sigma of the density would depend on the spread of the object in the image. We use the image as input and the Guassian convolved density image as ground truth label to learn the mapping between the objects and its distribution.

 <table style="width:100%">
  <tr>
    <td><img src="data/image-1.png"></td>
    <td><img src="data/image-1-dot.png"></td>

  </tr>
  <tr>
    <td><img src="data/image-1-gt.png"></td>
    <td><img src="data/image-1-gt-3d.png"></td>
  </tr>
</table>

This is where the concept of object counting comes into the picture. Since, we fit one probability density per object, the sum of all pixels in the density image would be equal to the number of objects in the image. Hence, we formulate our problem based on the estimation of these densities, the accuracy of their peaks, the accuracy of their location and the count metric.

## **Problem Statement**
We started our work by understanding the model from [2], which we refer to as the Base Model, which has shown state of the art results for many counting datasets. The base model doesn't preserve the spatial structure of the objects in the image. Although, we progressed to build a model which is entirely different than the Base Model, based on our goal and evolution of the project. The base model, like many others in the domain, are optmized for counting performance. They don't preserve the location information of the objects of interest. The base model has 2 pooling layers and the size of output density of the network is reduced by 4. This density is then scaled up to match the dimension of the input patch. These overlapping patched are then averaged over to get the final density map. This resize and average operation of the density patches leads to either damping of the guassian peak, or overestimating the peak, by maintaining the count. This is shown in the figure below. If the Guassian in red is the true guassian, then the estimated density might be any of green, blue or yellow. But, it is important to note that the area under the density is still one and hence it doesn't affect count performance. Another problem is the lose of spatial information of the object because of resize and average operation.
<p style="text-align:center"><img src="data/guass.png" width="270" height="200"></p>
<p style="text-align:center"><img src="data/base-prob.png"></p>

## **Our Approach**
At a high level, we propose a sliding window based approach to address the problem of preserving the spatial structure. Given a image patch, we predict the value of the density at the center of the patch and slide through the image. This approach would predict densities at every pixel in a sliding window fashion, and we stitch all the pixels together to get the final density image. Since, we predict in a pixel wise manner, we don’t need to have skip connection to preserve the spatial information and no information is lost in the intermediate layers. Later, we also show that the output can be generalized to a NxN grid around the center of the patch, to make the density more smooth. All our results were achieved with N=1, ie estimating the density at the center of the patch.
We consider UCSD Pedestrian dataset and our object of interest are the people walking on the walkway.


## **Network #1: Initial model**
We start off with a basic model and transit to our final model based on the result of our experiments. Our initial model predicts the density at every pixel in a sliding window fashion. A patch is extracted from the input image and it is fed as input to the network. The size of the image patch is chosen based on the size of the object, whose densities we are trying to estimate. For every image patch, we predict the density of the image patch at the center of the patch. We slide over the image to predict the densities of all the pixels in the image. The size of the image patch was odd since the center is well defined.

<p style="text-align:center"><img src="data/center-wise_single-loss.png"></p>

### **Initial model shortcomings**
Although the model looks simple, it was very hard to train. Consider the following samples,

<p style="text-align:center"><img src="data/problem1.png"></p>

In the above samples, there are people in the patch, but the density at the center is zero. The network would learn the object density model only if the object is at the center of the patch. Since, more than 80% of the pixels in the image belong to the background, a random sampling of patches in the image during training would result in the network learning more of the background density than the object of interest, which is not what we want. This shows us that, even though we would like to estimate local density at the center, we also need to take into account the global density of the patch. This calls for a dual loss approach, one loss takes care of global density and another loss takes care of localized density, ie density at the center.

## **Network #2: Improved**
In this improved network, we optimize 2 losses. One for the global density of the patch itself and another one for the local density of the patch at its center. Given a patch, we estimate the density of the whole patch of size 18x18, we call this patch-wise density. The other fully-connected path in the network estimates the density at the center, we call this center-wise density. The patch-wise loss trains our the network to extract better features and the fc layers performs the final estimation. This network improved our results but still had its shortcoming. We discuss them in the next section.
<p style="text-align:center"><img src="data/dual-loss_shared-fc.png"></p>
<p style="text-align:center"><img src="data/loss.png"></p>

### **Improved model shortcomings**
We employed 2 different approaches to minimizing the dual losses.

#### **Iterative loss switching**
In this method, during training, the losses were iteratively switched every X iterations. Even for different values for X, both losses continued to fight eachother and didn't converge. By convergence, we mean to say that while one loss was being minimized the other one increased and this happened continously during training.

#### **Combined loss training**
Here, both losses were combined using a weighted sum. Since, the output is a probability density value, we deal with losses which are very low to start with. Since, the patch-wise loss takes in 18x18 patch, the correspoding loss is higher than the center-wise loss which is just a L2 norm of the predicted center density and center ground truth density, which is in orders less than the patch-wise loss. This results in the network minimizing the patch-wise loss more than the center-wise loss, which is not desired since, its the minimization of the center-wise loss that gives us accurate peaks and spatial location. Although, scaling center-wise loss improved the situation, the scale value had to be large and it didn't make it robous enough for other datasets. Model trained with this performed badly on the frames where there was high number objects with high overlap between them.

In general, what was observed was that both losses tried to undo each other in every epoch and didn't give the feeling of convergence. We noticed that the shared fc layers might be getting undone by each loss. Hence, we tried with having separate fc layers for each losses, so that they don't interfere with eachother and also converge. This lead to our next and final model.

## **Network #3: Final Architecture**
The ouput of the features extraction Convolution layers are fed to sepearate fc layers for both losses. This ensure that the fc layers of 2 losses are trained separately and help achieve convergence. We train our network with iterative loss switching approach, we switch losses every 3 iterations. We will show later how this helps our losses to converge. The patch-wise loss trains convolution layers help extract global patch features, while the center-wise loss help extract the local feature at the center. The sepearate fc layers then help in predicting the density.
<p style="text-align:center"><img src="data/final-model.png"></p>

## **Training**
<p style="text-align:center"><img src="data/train.png"></p>
Patches of size 37x37 are densely extracted from the image. The dot annotated image is convolved using a gaussian kernel. This can be any probability density function, even sum of two densities, as long as they are normalized such that the integration of all desnsity values of any object equals to one. The density patches corresponding to the image patches are then extracted from the density image (shown in green boxes). Since, we have two loss functions, the center pixel of these density patches are also extracted to train center-wise loss (shown in red). Later we show how we can extract any nxn grid around the center depending on how smooth we want our estimation to be by sacrificing peak accuracy.

## **Training with Dual Loss**
<p style="text-align:center"><img src="data/loss.png"></p>  
We have two loss functions. They are minimized in an iterative fashion by switching the losses every 3 iterations (emperically chosen). We kick start the training with the patch-wise loss, to train the model to learn the global density in the patch and then switch to localized center density after 3 iterations, and iterate.

<p style="text-align:center"><img src="data/loss-1.png"></p>

Here, we show the evolution of the predicted density with epochs. As mentioned before, training is kickstarted with patch-wise loss and hence you can see the structure emerging on the patch-wise predicted density after the first epoch. This continues to evolve until 3 iterations. Even though the center-wise loss has been trained yet, you can still see some minute structure emerging from the center-wise predicted density. At epoch 4, the once the training starts minimizing center-wise loss, the predicted density structure gets more clear. This continues for 30 epochs.
<p style="text-align:center"><img src="data/loss-2.png"></p>      

Below is the video showing the evolution for the whole 30 epochs. You can notice how the counting error keeps flipping based on which loss was being trained, as the losses fight it out. But, later the model converges to an stable training error. This is where the separate fc layers for each loss help. They help reduce the influence of the one loss over the other. This method works because both losses are correlated in some way, ie they are sort of achieving the same thing in a different way. Hence, the network learns that there is way to minimize both the losses without clashing.  

<h4 align="center"><b>Dual loss convergence (Video)</b></h4>
<video id="epoch" height="auto" width="740" src="data/epoch.mp4" controls></video>

## **Testing**
During testing, we extract patches of the same size of 37x37 in a sliding window fashion. These patches are then input the network and predict both the global density and local center density. The global patch density, which is of 18x18 is resized to input size and averaged over all overlapping density patches. But, there is no resizing involved with the center-wise predicted density. Since, the density is predicted at every pixel, the size of the final density image is of the same size as the original image. Hence, the network outputs 2 densitities, one with patch-wise density and then the other with center-wise density.  

<p style="text-align:center"><img src="data/test.png"></p>

## **Results**

Below video showcases the perfomance of our model in comparision with the base model from [2]. We present brief detail in the later section. The weighted density in the last column is the weighted sum of patch-wise and center-wise density, with higher weight to the center-wise density. This was done to make the center-wise density more smooth. Althouth the center-wise density is the one which has accurate peaks and preserves spatial information, achieves the best results on UCSD maximal test dataset.
<h4 align="center"><b>Result Summary (Video)</b></h4>      
<video id="results" height="auto" width="740" src="data/results.mp4" controls></video>

<p style="text-align:center"><img src="data/table.png" width="475" height="400"></p>

**NOTE**  

  1. The above results were achieved on UCSD maximal test dataset. Trained with frames 600:5:1400 and test data was same as mentioned in [2]  

  2. Because of the sliding window approach, 18 pixels (half of patch size) were skipped around the border. The only count it would miss are the ones entering the frame in the bottom-left corner of the screen. The ground truth numbers were adjusted accordingly to ensure fair comparision. This needs to be fixed, but shouldn't cause our error to increase too much.  

  3. All the other methods mentioned in the table above test their model on a varied set of datasets. We limited our experiments to UCSD pedestrian dataset, and above listed comparison are for the same dataset. The performance of our model for other datasets still remains to be experimented.


## **Analysis**

From the below picture, it can be seen that the patch-wise density is more smooth but has inaccurate peaks due to resize and averaging patch densities. The center-wise density is more rough, but the peaks closely resemble the ones from ground truth density, it also preserves the location to acceptable accuracy. The weighted density is just a smoother version of center-wise density. It is obtained by weighted average of patch-wise and center-wise density, with the latter having more weight. This is one of the attempt to make use of both densitites. Although, estimating the center-wise density is much faster than estimating the patch density.  
<p style="text-align:center"><img src="data/summ-res.png"></p>
<p style="text-align:center"><img src="data/analysis.png"></p>
<p style="text-align:center"><img src="data/final-model-generic.png"></p>
How was it achieved, explain the weighted density approach. Show good peaks in center approach. Show smoothness in patch

## **Perspective Distortion improvement**
<p style="text-align:center"><img src="data/distort.png"></p>
<p style="text-align:center"><img src="data/multi-scale.png"></p>

## **Applications**
<p style="text-align:center"><img src="data/count.png"></p>
<p style="text-align:center"><img src="data/track.png"></p>

## **Reference**

[1] Victor Lempitsky and Andrew Zisserman. Learning To Count Objects in Images. Advances in Neural Information Processing Systems, 2010.  
[2] Daniel O ̃noro-Rubio and Roberto J. L ́opez-Sastre. Towards perspective-free object counting with deep learning. ECCV, 2016.  
[3] Cong Zhang, Hongsheng Li, Xiaogang Wang, Xiaokang Yang. Cross-scene Crowd Counting via Deep Convolutional Neural Networks. CVPR, 2015.  
[4] Di Kang, Zheng Ma, Antoni B. Chan. Beyond Counting: Comparisons of Density Maps for Crowd Analysis Tasks - Counting, Detection, and Tracking. arXiv 2017.  
[5] Lokesh Boominathan, Srinivas S S Kruthiventi, R. Venkatesh Babu. CrowdNet: A Deep Convolutional Network for Dense Crowd Counting. ACM, 2016.  
[6] Mark Marsden, Kevin McGuinness, Suzanne Little and Noel E. O’Connor. Fully Convolutional Crowd Counting On Highly Congested Scenes. arXiv, 2016.  
[7] Lingke Zeng, Xiangmin Xu, Bolun Cai, Suo Qiu, Tong Zhang. Multi-scale Convolutional Neural Networks for Crowd Counting. arXiv, 2017.  
[8] Guanbin Li, Yizhou Yu. Visual Saliency Based on Multiscale Deep Features. arXiv, 2015.  
