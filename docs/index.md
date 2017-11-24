---
layout: default
---
# Object Density Estimation

#### Team Kage

## Intro

We consider the problem of estimating probabilistic density of objects in images. Given an image, our goal is to recover a density function F as a real function of pixels in the image. Most of the research in the domain are concentrated on the problem of object counting. Although, we aim to address the problem of object counting, we model our problem with preference in preserving the spatial structure of the image. The existing methods focus more on reducing the error in counting and hence, lose the spatial structure of the densities, due to downsample strides in the convolution/pooling operations. By spatial structure, we mean the output of the network is not of same size as the input image, hence there is no 1:1 mapping between the output densities and the input image.

## Preparing Data:
Gaussian convolution

## Problem:
3d density diagram of base and ground truth
Peak problem, explain how guassian gets smoothened with resize
Density squash, peaks not predominant

## Approach:
We started off understand the model form [], which we refer to as the base. The base model doesn’t preserve the spatial structure of the image. Although, we progressed to build a model that was entirely different than theirs based on our goal.
At a high level, we propose a sliding window based approach to address the problem of preserving the spatial structure. This approach would predict densities at every pixel in a sliding window fashion, and we stitch all the pixels together to get the final density image. Since, we predict in a pixel wise manner, we don’t need to have skip connection to preserve the spatial information and no information is lost in the intermediate layers.


## Network 1:
We start off with a basic model and transit to our final model based on the result of our experiments.
Our initial model predicted the density at every pixel in a sliding window fashion. A patch is extracted from the input image and it is fed as input to the network. The size of the image patch we chose is correlated based on the size of the object, whose densities we are trying to estimate. For every image patch, we predict the density of the image patch at the center of the patch. We slide over the image to predict the densities of all the pixels in the image. The size of the image patch was odd since the center is well defined.

<img src="data/final-model.png">

### Problems?

## Network 2:
With dual loss function

### Problems?

## Final Architecture:
Explain nxn smoothing factor, combination of 2 outputs

### Training:
How?

### Testing:
How?

### Iterative Loss switching:
How?
Convergence video

## Results
<video id="sampleMovie" src="data/epoch.mp4" controls></video>


##Goal?
How was it achieved, explain the weighted density approach. Show good peaks in center approach. Show smoothness in patch

##Perspective Distortion improvement:

##Applications:
Counting, tracking, detection.

##Future research:
Semi supervised, flownet

##Reference:

