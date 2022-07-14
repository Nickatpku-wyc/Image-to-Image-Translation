### Image-to-Image Translation

#### Introduction

Image-to-image translation is a very important application direction of GAN, and the common image restoration and super-resolution are actually examples of image-to-image translation. What we will do is an image-to-image translation task based on pix2pix, which was published in CVPR2017, a classic paper applying GAN to supervised image-to-image translation, where supervision means that the training data are paired.

pix2pix implements image translation based on GAN, or more precisely on cGAN (conditional GAN, also called conditional GAN), because cGAN can guide image generation by adding conditional information, so that the input image can be used as a condition in image translation to learn a mapping from the input image to the output image, resulting in a specified output image . Other GAN-based image translation, on the other hand, because the generator of the GAN algorithm is based on a random noise to generate the image, it is difficult to control the output, and therefore basically guides the image generation by other constraints instead of using cGAN, which is the difference between pix2pix and other GAN-based image translation.

Regarding the design of the network structure of the generator and discriminator. In the pix2pix model, the generator uses U-Net, which is a very widely used network structure in the field of image segmentation and is able to fully fuse features; whereas the generator structure commonly used in the original GAN is the encoder-decoder type.

The discriminator uses PatchGAN, which outputs a predicted probability value for each region (patch) of the input image, equivalent to the evolution from determining whether the input is true or false to determining whether the N*N size regions of the input are true or false.



#### Methods

Here I've listed what technique I've used in my experiments, more details can be found there.

**Hyper-parameter tuning**: We first tried different No of epochs with the No of epochs decay, since the baseline has trained too less. Based on that, we changed the learning rate and batch size together that we found that these two hyper-parameters will effect the weight adjustment together.  Except these, we also considered the learning policy.

**Data Augmentation**: For data augmentation, we tried flip, scale and crop images

**NN architectures**: We tried two NN architectures which having good performance for Image-to-Image Translation-- ResNet and UNet. And there are 4 architectures in all.

**Loss functions designs.**: We tried different loss terms for training.



#### Experiments & Analysis

We got the performance the baseline model on validation set : 0.4906.

Now we want to improve our model. The idea of the improvement is a top-down structure. We first tried different NN architectures, also with the loss functions and other subnetwork design.	

##### Hyper-Parameters tuning

As we mentioned before, the very first problem is, there is too less epochs for training. So, without changing the other parameters, we try to increase the number of epochs by more and, accordingly, change the corresponding parameters.

| Experiment No. |   Policy    |   Parameter    | Final ssim score | Best ssim score |
| :------------: | :---------: | :------------: | :--------------: | :-------------: |
|       1        | More epochs | n_epochs = 50  |      0.5002      |     0.5019      |
|       2        | More epochs | n_epochs = 100 |      0.5175      |     0.5175      |
|       3        | Less epochs | n_epochs = 75  |      0.5113      |     0.5272      |
|       4        | More epochs | n_epochs = 150 |      0.4987      |     0.5077      |
|       5        | Less epochs | n_epochs = 125 |      0.5035      |     0.5035      |

The chosen of No of epochs is binary search. And when we get n_epochs = 100, It seems like there is no significant difference than the last experiment, so we stop here about epoch temporarily.

Now, let's do with learning rate and batch size. We already know something about learning rate:

* small learning rate: Many iterations till convergence, and may trapped in local minimum
* large learning rate: Overshooting, No convergence 

And actually, there are also another hyper-parameter connected with learning rate -- batch size. In practice, we usually tuning these two parameters together.

About the batch size, we clear know what effect it has to the performance of our model:

* small batch size: Improving the generalization of models, the noise from a small batch size helps to escape the sharp minimum
* large batch size: Reduced training time and improved stability of our model. Meanwhile, large batch size gradients are more stable to compute as the model training curve will be smoother

It's worth to mention that large batch size performance degrades because the training time is not long enough, it is not essentially a problem of batch size, the parameters are updated less at the same epochs and therefore require longer iterations.

Let we take a look at the SGD algorithm we used:

<img src="C:\Users\Nicka\AppData\Roaming\Typora\typora-user-images\image-20220317162736534.png" alt="image-20220317162736534" style="zoom:33%;" />

We can write it as(without Adaptive Learning Rate):
$$
w\leftarrow w - \alpha J(\theta),\ \ 
J(\theta)=\frac{1}{B}\sum_{i=1}^B\nabla e(x_i,y_i)
$$
So we can find that the adjustment of weight is depending on learning rate and batch size simultaneously. And these two are in turn directly related to each other as numerators and denominators. 

Which means usually when we increase the batch size to N times the original, to ensure that the updated weights are equal after the same samples, the learning rate should be increased to N times the original according to the linear scaling rule, and vice versa.

And Smith S L and Le Q V in 2017 showed that for a fixed learning rate, there exists an optimal batch size that maximise test accuracy, which is positively related to the learning rate and the size of the training set.

So our aim is to find this optimal combination, and the specific experimental strategy is that if the learning rate is increased, then the batch size should preferably increase as well, so that the convergence is more stable.

To verify our assuming, we adjust the learning rate first, then preferably increase batch size.

| Experiment No. |      Policy       |        Parameter        | Final ssim score | Best ssim score |
| :------------: | :---------------: | :---------------------: | :--------------: | :-------------: |
|       6        |    Increase lr    | lr=4e-4, batch size=32  |      04923       |     0.4923      |
|       7        | change batch size | lr=4e-4, batch size=64  |  out of memory   |  out of memory  |
|       8        | change batch size | lr=4e-4, batch size=16  |      0.4899      |     0.5014      |
|       9        |    Increase lr    | lr=1e-3, batch size=16  |   not converge   |  not converge   |
|       10       |    decrease lr    | lr=2e -4, batch size=32 |      0.4975      |     0.5079      |
|       11       | change batch size | lr=2e-4, batch size=16  |      0.4897      |     0.5079      |
|       12       | change batch size | lr=2e-4, batch size=48  |  out of memory   |  out of memory  |

Then we tried different lr policys. 

| Experiment No. |       Policy        |        Parameter        | Final ssim score | Best ssim score |
| :------------: | :-----------------: | :---------------------: | :--------------: | :-------------: |
|       13       |  lr_policy = step   | lr=2e -4, batch size=32 |      0.4779      |     0.4779      |
|       14       | lr_policy = plateau | lr=2e -4, batch size=32 |      0.4907      |     0.5033      |
|       15       | lr_policy = cosine  | lr=2e -4, batch size=32 |      0.4825      |     0.4964      |

It seems not work, so we drop it.



##### NN architectures & Subnetwork design

For Image-to-Image Translation, actually, there are two NN architectures having good performance -- Resnet and Unet.

We already tried Unet_256 as default, now let's try others . And for new NNs, we also need to find out the best lr + batch size.

| Experiment No. |       Policy        |         Parameter          | Final ssim score | Best ssim score |
| :------------: | :-----------------: | :------------------------: | :--------------: | :-------------: |
|       16       |   netG = unet_256   |          default           |      0.4906      |     0.4915      |
|       17       |   netG = unet_128   |          default           |      0.4998      |     0.5037      |
|       18       | netG=resnet_6blocks |          default           |  out of memory   |  out of memory  |
|       19       | netG=resnet_6blocks | batch_size = 4, lr = 1e-4  |      0.5466      |     0.5523      |
|       20       | netG=resnet_6blocks | batch_size = 8, lr = 2e-4  |      0.5633      |     0.5633      |
|       21       | netG=resnet_9blocks |          default           |  out of memory   |  out of memory  |
|       22       | netG=resnet_9blocks | batch_size = 8, lr = 2e-4  |      0.5837      |     0.5977      |
|       23       | netG=resnet_9blocks | batch_size = 16, lr = 4e-4 |      0.6033      |     0.6274      |
|       24       |    netD = pixel     | batch_size = 16, lr = 4e-4 |      0.5897      |     0.5922      |
|       25       |   netD = 3_layers   | batch_size = 16, lr = 4e-4 |      0.6019      |     0.6175      |
|       26       |   netD = 2_layers   | batch_size = 16, lr = 4e-4 |      0.6185      |     0.6204      |
|       27       |    loss function    |      lambda_L1 200.0       |      0.6343      |     0.6390      |



##### Data augmentation

Then we tried to add some more data augmentation operation for our data.

| Experiment No. |     Policy      |           Parameter           | Final ssim score | Best ssim score |
| :------------: | :-------------: | :---------------------------: | :--------------: | :-------------: |
|       28       |      crop       |         crop_size=256         |      0.6115      |     0.6204      |
|       29       |   scale_width   |        load_width=192         |      0.6233      |     0.6233      |
|       30       | resize_and_crop | load_size=256 & crop_size=192 |      0.6275      |     0.6292      |
|       31       |      flip       |        no_flip = false        |      0.6351      |     0.6401      |

So we find that the best combination is to flip the images for data augmentation without other scaling or croping.

Till now, we have down all experiments to improve the model. 

![image-20220414224415790](C:\Users\Nicka\AppData\Roaming\Typora\typora-user-images\image-20220414224415790.png)



#### Reference

[1] Isola, P., Zhu, J. Y., Zhou, T., & Efros, A. A. (2017). Image-to-image translation with conditional adversarial networks. *arXiv preprint*.

[2] [Background: What is a Generative Model?  | Generative Adversarial Networks  | Google Developers](https://developers.google.com/machine-learning/gan/generative)

[3] Mao, X., & Li, Q. (2020). Generative adversarial networks (gans). In *Generative Adversarial Networks for Image Generation* (pp. 1–7). Springer Singapore.
