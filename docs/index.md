---
layout: default
---
# Object Density Estimation

### ***Team Kage***

## **Introduction**

We consider the problem of estimating probabilistic density of objects in images. Given an image, our goal is to recover a density function F as a real function of pixels in the image. Most of the research in the domain are concentrated on the problem of object counting. Although, we aim to address the problem of object counting, we model our problem with preference in preserving the spatial structure of the image. The existing methods focus more on reducing the error in counting and hence, lose the spatial structure of the densities, due to downsample strides in the convolution/pooling operations. By spatial structure, we mean the output of the network is not of same size as the input image, hence there is no 1:1 mapping between the output densities and the input image.

## **Preparing Data**
We will talk about how we structure our data as it is important to understand the later sections. First, we have images which has objects of interest. Next, we annotate these objects with a single dot, naturally at its center of mass. We will now fit a probabilistic density on the object, Gaussian density in this case, centered around the dot annotation on the object. Gaussian density makes natural sense because the probability to find the object is more at the center and then decreases as you move away from the objects center of mass. We do this by convolving our dot annotate image with an Guassian kernel of a suitable sigma. The sigma of the density would depend on the spread of the object in the image. We use the image as input and the Guassian convolved density image as ground truth label to learn the mapping between the objects and its distribution.

 <table style="width:100%">
  <tr>
    <th><img src="data/image-1.png"></th>
    <th><img src="data/image-1-gt.png"></th>
  </tr>
  <tr>
    <td><img src="data/image-1-dot.png"></td>
    <td><img src="data/image-1-gt-3d.png"></td>
  </tr>
</table> 

This is where the concept of object counting comes into the picture. Since, we fit one probability density per object, the sum of all pixels in the density image would be equal to the number of objects in the image. Hence, we formulate our problem based on the estimation of these densities, the accuracy of their peaks, the accuracy of their location and the count metric.

## **Problem Statement**
We started our work by understanding the model from [2], which we refer to as the Base Model, which has shown state of the art results for many counting datasets. The base model doesn't preserve the spatial structure of the objects in the image. Although, we progressed to build a model which is entirely different than the Base Model, based on our goal and evolution of the project. The base model, like many others in the domain, are optmized for counting performance. They don't preserve the location information of the objects of interest. The base model has 2 pooling layers and the size of output density of the network is reduced by 4. This density is then scaled up to match the dimension of the input patch. These overlapping patched are then averaged over to get the final density map. This resize and average operation of the density patches leads to either damping of the guassian peak, or overestimating the peak, by maintaining the count. This is shown in the figure below. If the Guassian in red is the true guassian, then the estimated density might be any of green, blue or yellow. But, it is important to note that the area under the density is still one and hence it doesn't affect count performance. Another problem is the lose of spatial information of the object because of resize and average operation.
<p style="text-align:center"><img src="data/guass.png"></p>
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
Here, both losses were combined using a weighted sum. Since, the output is a probability density value, we deal with losses which are very low to start with. Since, the patch-wise loss takes in 18x18 patch, the correspoding loss is higher than the center-wise loss which is just a L2 norm of the predicted center density and center ground truth density, which is in orders less than the patch-wise loss. This results in the network minimizing the patch-wise loss more than the center-wise loss, which is not desired since, its the minimization of the center-wise loss that gives us accurate peaks and spatial location. Although, scaling center-wise loss improved the situation, the scale value had to be large and it didn't make it robous enough for other datasets.

In general, what was observed was that both losses tried to undo each other in every epoch and didn't give the feeling of convergence. We noticed that the shared fc layers might be getting undone by each loss. Hence, we tried with having separate fc layers for each losses, so that they don't interfere with eachother and also converge. This lead to our next and final model.

## **Network #3: Final Architecture**
The ouput of the features extraction Convolution layers are fed to sepearate fc layers for both losses. This ensure that the fc layers of 2 losses are trainined separately and help achieve convergence. We train our network with iterative loss switching approach, we switch losses every 3 iterations. We will show later how this helps our losses to converge where they done fight. The patch-wise loss trains convolution layers help extract global patch features, while the center-wise loss help extract the local feature at the center. The sepearate fc layers then help in predicting the density.
<p style="text-align:center"><img src="data/final-model.png"></p>

## **Training**
<p style="text-align:center"><img src="data/train.png"></p>

## **Training with Dual Loss**
<p style="text-align:center"><img src="data/loss.png"></p>

<p style="text-align:center"><img src="data/loss-1.png"></p>

<p style="text-align:center"><img src="data/loss-2.png"></p>

<video id="epoch" height="auto" width="740" src="data/epoch.mp4" controls></video>

## **Testing**
<p style="text-align:center"><img src="data/test.png"></p>

## Results
<p style="text-align:center"><img src="data/table.png"></p>
<video id="results" height="auto" width="740" src="data/results.mp4" controls></video>


## Goal?
<p style="text-align:center"><img src="data/summ-res.png"></p>
<p style="text-align:center"><img src="data/analysis.png"></p>
<p style="text-align:center"><img src="data/final-model-generic.png"></p>
How was it achieved, explain the weighted density approach. Show good peaks in center approach. Show smoothness in patch

## Perspective Distortion improvement:
<p style="text-align:center"><img src="data/distort.png"></p>
<p style="text-align:center"><img src="data/multi-scale.png"></p>

## Applications:
<p style="text-align:center"><img src="data/count.png"></p>
<p style="text-align:center"><img src="data/track.png"></p>

## Reference:

[1] Victor Lempitsky and Andrew Zisserman. Learning To Count Objects in Images. Advances in Neural Information Processing Systems, 2010.  
[2] Daniel O ̃noro-Rubio and Roberto J. L ́opez-Sastre. Towards perspective-free object counting with deep learning. ECCV, 2016.  
[3] Cong Zhang, Hongsheng Li, Xiaogang Wang, Xiaokang Yang. Cross-scene Crowd Counting via Deep Convolutional Neural Networks. CVPR, 2015.  
[4] Di Kang, Zheng Ma, Antoni B. Chan. Beyond Counting: Comparisons of Density Maps for Crowd Analysis Tasks - Counting, Detection, and Tracking. arXiv 2017.  
[5] Lokesh Boominathan, Srinivas S S Kruthiventi, R. Venkatesh Babu. CrowdNet: A Deep Convolutional Network for Dense Crowd Counting. ACM, 2016.  
[6] Mark Marsden, Kevin McGuinness, Suzanne Little and Noel E. O’Connor. Fully Convolutional Crowd Counting On Highly Congested Scenes. arXiv, 2016.  
[7] Lingke Zeng, Xiangmin Xu, Bolun Cai, Suo Qiu, Tong Zhang. Multi-scale Convolutional Neural Networks for Crowd Counting. arXiv, 2017.  
[8] Guanbin Li, Yizhou Yu. Visual Saliency Based on Multiscale Deep Features. arXiv, 2015.  
