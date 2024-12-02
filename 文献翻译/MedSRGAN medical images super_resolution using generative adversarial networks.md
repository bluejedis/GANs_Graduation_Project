# MedSRGAN: medical images super-resolution using generative adversarial networks  


# Abstract  

Super-resolution (SR) in medical imaging is an emerging application in medical imaging due to the needs of high quality images acquired with limited radiation dose, such as low dose Computer Tomography (CT), low field magnetic resonance imaging (MRI). However, because of its complexity and higher visual requirements of medical images, SR is still a challenging task in medical imaging. In this study, we developed a deep learning based method called Medical Images SR using Generative Adversarial Networks (MedSRGAN) for SR in medical imaging. A novel convolutional neural network, Residual Whole Map Attention Network (RWMAN) was developed as the generator network for our MedSRGAN in extracting the useful information through different channels, as well as paying more attention on meaningful regions. In addition, a weighted sum of content loss, adversarial loss, and adversarial feature loss were fused to form a multi-task loss function during the MedSRGAN training. 242 thoracic CT scans and 110 brain MRI scans were collected for training and evaluation of MedSRGAN. The results showed that MedSRGAN not only preserves more texture details but also generates more realistic patterns on reconstructed SR images. A mean opinion score (MOS) test on CT slices scored by five experienced radiologists demonstrates the efficiency of our methods.  

Keywords  Medical images  $\cdot$    Super-resolution (SR)  $\cdot$  Deep learning  $\cdot$  Generative adversarial networks (GAN)  
# 1 Introduction  

Medical images, including computed tomography (CT), magnetic resonance imaging (MRI), positron emission tomography (PET), are popular for clinical applications such as noninvasive diseases diagnosis, anatomic imaging and treatment planning [ 2 ,  7 ,  17 ,  24 ,  25 ,  32 , 37 ,  38 ,  43 ,  47 ]. However, there are some limitations in these imaging technologies. For example, the radiation injury is inevitable during CT scanning. Low-dose CT (LDCT) is a clinical recommended technology for reducing the radiation injury to patients, but at the expense of low image resolution and noise contamination. MRI is known as its expressive capacity in providing anatomical, metabolic, and functional information for various regions of human bodies. However, one of the major disadvantages of MRI is the long acquisition duration which makes the image quality is susceptible to patient movement. In order to obtain image with higher single noise ratio (SNR), its spatial resolution tends to be coarse than CT images. Furthermore, low magnetic field of MRI scanner also constrains the spatial resolution of MRI. PET is able to detect the tumor in its early stage accurately by imaging the different absorption capacity of labeled compounds with radioisotope in different parts of the body. The imaging theory of PET is based on detecting the quantity of photon pairs produced by annihilation reaction of radioisotope. Due to the scarcity of photon pairs and noise interference, the low-resolution problem is also a challenge for PET imaging.  

To improve the image quality with low radiation dose and technology limitations, one of the solutions is retrieving image details based on low resolution (LR) image to reconstruct high-resolution (HR) [ 12 ,  29 ], such process is widely known as super-resolution (SR). Although clinical obtained LR medical images contain realistic and fundamental information, they lose many details including high-frequency components, certain patterns like tissue background and textures which are extremely necessary for reconstructing HR images. Therefore, SR for medical images is still a big challenge and hardly used in practice. In this work, we propose a novel medical image SR framework based on deep learning and show its effectiveness.  

# 1.1 Related works  

In recent years, a large range of successful neural network based algorithms were proposed for SR on natural images. According to different purposes, these methods can be mainly divided into two main categories: high peak signal-to-noise ratio (PSNR) / structural similarity (SSIM) and high perceptual quality. Methods with higher PSNR/SSIM were mainly achieved by network structure innovations such as deeper networks, various feature connections and feature flowing strategies inside networks [ 8 ,  26 ,  27 ,  33 ,  41 ,  49 ]. With the help of mean square error (MSE) loss, they have got prominent results on PSNR/SSIM metrics, but SR images still contradict with human observation because human eyes are good at catching high frequency components rather than telling pixel-level difference. On the other hand, methods with higher perceptual quality [ 30 ,  45 ] were mainly come out with perceptual loss [ 22 ] and GAN [ 11 ]. These methods performed well on perceptual index metrics but compromise on PSNR/SSIM metrics.  

Deep learning based medical image restoring researches gradually get more attentions in medical imaging research communities in recent years, these researches include image translation between different imaging systems [ 1 ,  19 ], super resolution [ 3 ,  4 ,  31 ,  34 ,  35 ,  39 ,  48 ], and image de-noising [ 1 ,  10 ,  20 ].  
# 1.1.1 High PSNR/SSIM of natural image  

Dong et al. [ 8 ] proposed Super-Resolution Convolutional Neural Network (SRCNN) for singleimage SR (SISR), which was the first convolutional neural network (CNN) research in SISR problem. SRCNN first enlarge input image to target size with bicubic interpolation, then feed it into a simple network with three convolutional layers. The method was already able to achieve higher PSNR/SSIM than any other traditional computer vision algorithms. This research reveals that endto-end learning methods can effectively learn those necessary nonlinear transformation on pixel level when enlarging an image. Since then, deep learning has become the main stream for SISR on natural image. Many attempts have been done in this field to achieve higher performance on PSNR/SSIM, such as using recursive convolutional network for feature extraction [ 26 ], stacking more convolutional filters and adding long residual connections [ 27 ], and removing batch normalization layers to preserve range flexibility from networks [ 33 ]. Shi et al. [ 41 ] proposed an efficient up-sampling module for SR, called pixel shuffle module, which simply rearranges  $H\times W\times r^{2}$    image tensor into    $r H\times r W\times1$   to enlarge the height and width of image features. This process does not contain learnable parameters, and it reduce the channels of feature. Researchers often put an extra convolutional layer before pixel shuffle module to increase the feature channels, which makes upsampling process learnable and preserves feature channels. Based on these ideas, Zhang et al. [ 49 ] designed a very deep CNN with stacked channel attention modules, called Residual Channel Attention Network (RCAN), and achieved state-of-the-art PSNR/SSIM.  

All these methods are considered as PSNR-oriented methods because they only take mean square error (MSE) or mean absolute error (MAE) as loss function during the network training, which is an intuitive idea to increase PSNR metric. However, the generated images would still become over-smoothed [ 30 ,  45 ]. Since PSNR often disagrees with visual evaluation by human observers, such results are intolerable in medical imaging applications, which may hinder related clinical applications such as disease diagnosis and lesion detection.  

# 1.1.2 High perceptual quality of natural image  

To restore feasible high-frequency components and obtain better visual results in SISR, researchers apply loss on feature level [ 22 ] and use GAN [ 11 ] to make images as realistic as possible. Loss on feature level, also known as perceptual loss, is widely used on many image reconstruction tasks including pixel-to-pixel image translation (pix2pix) [ 18 ], image style transfer [ 9 ,  21 ] and super resolution [ 30 ,  45 ]. It takes a pre-trained neural network to extract features for image outputs and ground truth targets separately, and then calculates loss on these two features, to guide networks to learn semantic features from target images. GAN is initially used in fake image synthesis [ 11 ]. It usually contains two independent neural networks with different parameters, one for image generation, and the other for real/fake image discrimination. The target for discrimination network is whether an image is real or produced by generation network, which means fake, while the target for generation network is the adversarial form of that for disc rim in at or network, so it makes fake images get close to real images. In the training process, generation network gradually narrows the gap between produced images and real images until disc rim in at or can no longer tell the differences.  

Ledig et al. [ 30 ] first proposed a GAN-based SR framework (SRGAN) which contains a generator network for high resolution image generation, a disc rim in at or network for recognizing generated images from real world images, and loss function which includes perceptual loss and GAN loss. SRGAN had got much better visual results than previous PSNR-oriented methods though it compromised some PSNR/SSIM values. In Wang et al. ’ s work ESRGAN (EnhancedSRGAN) [ 45 ], they proposed a densely connected [ 16 ] Residual in Residual Dense Block (RRDB) network for generator and used relativistic disc rim in at or [ 23 ] to determine whether an image is more realistic than the other image, and they won PIRM-SR Challenge (ECCV2018) on high perceptual quality group.  
# 1.1.3 Reconstruction works for medical image  

Aforementioned researches have achieved great progress in SISR tasks, however, none of these studies considered the characteristics of medical images nor evaluated their performances on medical images. It is undoubtedly infeasible to directly apply their models in medical images SR due to the totally different data distribution, patterns and textures.  

For medical image reconstruction works, Armanoius et al. [ 1 ] adapted ideas from pix2pix research and proposed a GAN-based framework named MedGAN for PET-CT translation, MRI motion correction and PET de-noising. You et al. [ 48 ] proposed a complicated GAN-based framework with ideas from cycle GAN [ 50 ] for   $2\times$   single-slice CT SR, and it requires independent LR and HR datasets. Due to unnecessary radiation injuries for collecting more CT scans on the same patients, they only used a tibia dataset from fresh-frozen cadaveric ankle specimens and a public abdominal dataset with a small amount of scans for experiments. Furthermore, they only focused on very small local patches instead of the whole CT scans, so the practicality of this approach is limited in clinical applications. As for MRI SR, Chen et al. [ 3 ,  4 ] proposed 3D Densely Connected SuperResolution Networks (DCSRN) to restore HR features of structural brain MRI and they further developed a GAN framework for guiding DCSRN training in order to further improve the SR quality. Mardani et al. [ 35 ] proposed GAN for Compressive Sensing (GANCS) to retrieve higher quality images with improved fine texture details. Although visually reliable results were observed in these methods, they did not evaluate whether these produced images would be acceptable by radiologists nor affect clinical diagnosis.  

# 1.2 Our contribution  

In this work, we develop a medical images SR framework using generative adversarial networks (MedSRGAN) for reconstructing reliable and visually realistic SR medical images, which takes a LR medical image as input to generate a  $4\times$   SR image. In MedSRGAN, we employ an improved medical image generation network, Residual Whole Map Attention Network (RWMAN) as SR image generator, a new pairwise disc rim in at or to distinguish the pairs of both HR/SR and LR images as well as a novel multi-task loss function combining the content loss, adversarial loss, and adversarial feature loss for guiding the SR image to obtain more reliability and feasibility. Our methods were evaluated on 242 thoracic CT scans and 110 brain MRI scans, and our results would bring marginal influence on disease diagnosis, which reveal that SR based medical imaging systems are possible to be applied in practice.  

# 2 Methods  

MedSRGAN is a typical GAN-based architecture. As is shown in Fig.  1 , it consists of two neural networks: generator and disc rim in at or, and two independent loss functions for each network. The targets of disc rim in at or are from whether an image is ground truth or generated image, while targets of generator come from disc rim in at or and image contents.  
![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/ceafd5ad4172f11e1c9222cca9105cb18b194c628933526ed836ef9d38451215.jpg)  
Fig. 1  Basic framework of MedSRGAN. Arrows of different colors defined how data flow forward for calculation, backward for parameters updating, and where to make comparison for loss function  

# 2.1 Generator structure  

As shown in Fig.  2 , we named our generator as Residual Whole Map Attention Network (RWMAN), which is a modification of RCAN [ 49 ] by replacing its Residual Channel Attention Block (RCAB) with Residual Whole Map Attention Block (RWMAB). RWMAN extracts abundant features on LR scale and up-samples after these steps. It consists of an input convolution, RWMAB groups with long and short residual connections, and up-sampling module with two Conv-Pixel Shuffle blocks.  

In RCAN [ 49 ], RCAB uses global average pooling to squeeze the whole feature map into a single value through all channels at first, then makes these values learnable in the following layers, and finally treats them as weights for previous channels, such process is regarded as channel attention. Attention mechanism has improved the performance of RNN models such as LSTM and GRU in many NLP tasks [ 6 ], and it is now widely used in other different domains including recommending system [ 5 ,  13 ] and computer vision [ 15 ,  44 ,  46 ]. Generally, this mechanism apply weights on current feature adaptively by adding a sub-network on this feature, and channel attention learns weights through channels of features. This step helps neural network adaptively learn how to use information wisely through different channels, but it neglects attentions on different regions of an image. Generally, the whole map of a natural image should be equally focused on to reconstruct a higher resolution one, since all pixels may reflect the meaningful information in real world. However, for many medical images, only those regions with useful information should be wisely considered, surrounded areas such as air and vacant areas are meaningless. We expect the neural network not only extracts useful information through different channels, but also pays more attention on meaningful regions.  

In RWMAB, we use a  $1\times1$   convolutional layer to obtain a tensor having the same shape of the input image tensor, followed by a sigmoid activation to form weights range from 0 to 1 for every pixel through all channels. This sub-network structure helps our model enlarge or lower the effect of each pixel adaptively, to enable our GAN-based loss function make influence. We stack 128 such neural network blocks with long (from the beginning to end) and short (for each 16 blocks) residual connections to reduce the training difficulties [ 14 ].  
![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/de070c8e9ef97de16578220d61283c70a3fe5254c6277fca33fb34bc6ddfbcd6.jpg)  
Fig. 2  The generator structure Residual Whole Map Attention Network (RWMAN). For each convolutional layer, channel (c) and strides (s) are pointed out above.  $\imath_{\times2},$   on pixel shuffle module means it double the height and width of an feature by pixel rearrangements. The red dashed box represents an optional noise input. This network outputs a   $4\times$   super-resolution image for each input image  

In up-sample module, we use two   $2\times$   Conv-Pixel Shuffle [ 41 ] modules to generate a   $4\times$  image feature, with the number of feature channels kept, then reduce channels by a   $3\times3$  convolutional layer to make output image.  

Additionally, a random Gaussian noise with zero mean and unit variation was introduced as an additional channel for perturbation before input a LR image into neural networks, as shown in Fig.  2 . This is an optional operation which is denoted by    $\because$   in this study. For instance, the MedSRGAN with additional noise channel is denoted by   $\mathrm{MedSRGAN+}$  . GAN framework taking noise and meaningful information as input is regarded as a standard conditional GAN [ 18 ,  36 ]. By doing so, feature maps inside the neural networks are provided with some randomness, and it may help neural networks to be more adaptive on generating more feasible patterns in homogeneous regions, especially in tissue backgrounds, while the major content of images still come from input LR images.  

For convenience, HR image generated by our neural network with corresponding LR image is named SR (super-resolution) image in the following descriptions:  

$$
S R=G(L R)
$$  

where  $G(\cdot)$   is the operation of a given generator. With this notation, HR represents ground truth image while SR means generated image in this study.  

# 2.2 Disc rim in at or structure  

Instead of having a single HR/SR image as input of disc rim in at or, we used image pair (LR, HR/SR) as input to discriminate the SR image with a given LR image. Fig.  3  shows the architecture of our disc rim in at or. With this design, the disc rim in at or is specified to learn the pairwise information of both HR/SR and LR images by concatenating feature maps extracted from the LR and HR pathway, and outputs the probability of a (LR, HR) or (LR, SR) pair as an real pair.  

With CT slice pair of (LR, HR) as 1 and (LR, SR) as 0, the training stage for disc rim in at or is expressed as:  

$$
\begin{array}{r}{D(L R,H R){\rightarrow}1}\\ {D(L R,S R){\rightarrow}0}\end{array}
$$  

where  $D(\cdot,\cdot)$   is the operation of disc rim in at or.  
![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/2e6487bfaaab47b8a1bdb154e4739b50ab5a8085669e916bf0b37e20f1aac570.jpg)  
Fig. 3  Disc rim in at or Structure. For each convolutional layer, channel (c) and strides (s) are pointed out above, and all LeakyReLU used    $\alpha\!=\!0.2$   on negative axis.  $D_{i}(\cdot,\cdot)$  ,  $i\in\{1,2,...,5\}$   indicates we output features in after these layers to calculate adversarial feature loss  

# 2.3 Loss function  

During the training, either a generator or a disc rim in at or is trained alternately at each iteration with different data to avoid model parameters falling into a local optimum. In this study, Binary Cross Entropy (BCE) loss is used to train disc rim in at or:  

$$
L_{d}=-\mathrm{log}(D(L R,H R))–\mathrm{log}(1–D(L R,S R))
$$  

where  $D(\cdot,\cdot)$   is the operation of disc rim in at or.  

A weighted sum of content loss, adversarial loss and adversarial feature loss is employed as loss function for generator training:  

$$
{\cal L}_{g}={\cal L}_{c o n t}+\lambda_{1}{\cal L}_{a d\nu}+\lambda_{2}{\cal L}_{a d\nu f e a t}
$$  

where    $\lambda_{1}$   and  $\lambda_{2}$   are tunable hyper parameters for balancing the impacts of each part of loss.  

# 2.3.1 Content loss  

This part of loss is mainly responsible for the content restoration of an image. Generally, to obtain higher PSNR, content loss between HR and SR can be simply set as the L1 loss or MSE loss:  

$$
L1=\frac{1}{N^{2}}\sum_{i=1}^{N}\sum_{j=1}^{N}|H R(i,j)–S R(i,j)|
$$  

$$
M S E=\frac{1}{N^{2}}\sum_{i=1}^{N}\sum_{j=1}^{N}\left(H R(i,j)–S R(i,j)\right)^{2}
$$  

where  $N$   represents the size of images, assuming that each image is  $N$   by  $N.$  .  

As shown in Eq. ( 5 ) and ( 9 ), minimizing L1 or MSE is a PSNR-oriented optimization, but it fails to reconstruct the high frequency content and preserve visual realistic characteristics [ 30 , 45 ]. For medical images, texture features are very important because these high-frequency information are decisive in human visual judgment for the quality of images. Thus, merely using PSNR-oriented loss function is not sufficient for medical image SR.  
A well pre-trained neural network, such as VGG-16 or VGG-19 [ 42 ], has been used to extract texture features on a hidden layer to calculate content loss in many tasks including image style transfer [ 9 ,  21 ], image translation [ 18 ] and image SR [ 22 ,  30 ,  45 ]. In this work, we used a pre-trained VGG-19 to calculate content loss. Rather than obtaining loss on a single VGG hidden block, we used five hidden blocks to fully extract texture features on each semantic level and combine losses with more weights on shallower blocks and less weights on deeper blocks. This is because deeper features bring more abstract semantic information, while shallow features show relatively more concrete texture features information, as is shown in Fig.  4 . With    $V_{i,j}$   representing the VGG-19 network operation by the j-th convolution before the i-th max-pooling layer, our content loss is defined as:  

$$
L_{c o n t}=\lambda_{L1}\cdot L_{1}(H R,S R)+\sum_{k=1}^{5}w_{k}\cdot M S E\Big(V_{(i,j)_{k}}(H R),V_{(i,j)_{k}}(S R)\Big)
$$  

where  $\lambda_{L1}$   is a tunable parameter for balancing the direct L1 loss between HR and SR image, weights    $\begin{array}{r}{w=\left\{\frac{1}{2}\,,\frac{1}{4}\,,\frac{1}{8}\,,\frac{1}{16}\,,\frac{1}{16}\right\}}\end{array}$   and layer numbers   $(i,j)=\{(1,2),(2,2),(3,4),(4,4),(5,4)\}$   for 5 hidden blocks,  $L1(\cdot,\cdot)$   and  $M S E(\cdot,\cdot)$  (·, ·) represents the calculation of the L1 norm (mean absolute error) and mean square error of two features respectively. Followed by the analysis of VGG feature maps in ESRGAN [ 45 ], we used feature maps before ReLU activation for all hidden blocks to avoid the information missing after ReLU activation.  

# 2.3.2 Adversarial loss  

Adversarial loss aims to help generator make SR image closer to realistic HR image as much as possible to deceive the disc rim in at or. The adversarial loss  $L_{a d\nu}$   has a reciprocal form of    $L_{d}$  , aiming to make  $D(L R,S R)$   get closer to 1:  

$$
L_{a d\nu}=-\mathrm{log}(1\!-\!D(L R,H\!R))\!-\!\mathrm{log}(D(L R,S\!R))
$$  

where  $D(\cdot,\cdot)$   is the operation of disc rim in at or.  

# 2.3.3 Adversarial feature loss  

Like VGG based content loss, we also view our disc rim in at or as a feature extractor and calculated the loss of hidden blocks between HR and SR images and between (HR, LR) and (SR, LR) pairs. Image examples are shown in Fig.  5 . Minimizing the adversarial feature loss is  

![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/bc73b34cbeea95a91282df23444789670fd177c9cf41aac1cc72c4cff3783dcb.jpg)  
Fig. 4  Output examples of VGG-19 with   $\mathbf{V}_{i,\,j}$   denotes the  $\mathrm{j}$  -th convolution before the i-th maxpooling layer (before ReLU activation) on one CT slice image, which shows that shallow layers get concrete texture information while deep layers get abstract information.  ‘ ch’  with a number indicates the channel number of  $\mathrm{V}_{i,.}$  j feature map.  Best viewed on screen with zoom in  
![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/3d07d84e6ee9f3c4f5cf5c06c0c00f80827208032262703b7e9a04623b91e64f.jpg)  

to minimizing differences of texture features viewed by disc rim in at or so that it helps generator benefit from disc rim in at or features to make more realistic patterns.  

With  $D_{k}(\cdot,\cdot)$   represents the   $\mathrm{k\Omega}$  -th hidden block inside the disc rim in at or as is shown in Fig.  3 , our adversarial feature loss is defined as:  

$$
L_{a d y f e a t}=\sum_{k=1}^{5}w_{k}\cdot M S E\big(D_{k}(L R,H R),D_{k}(L R,S R)\big)
$$  

where weights  $\begin{array}{r}{w=\left\{\frac{1}{2},\frac{1}{4},\frac{1}{8},\frac{1}{16},\frac{1}{16}\right\}}\end{array}$    for 5 hidden blocks, and    $M S E(\cdot,\cdot)$   represents the calculation of the mean square error of two features.  

# 3 Experiments  

In this study, MedSRGAN and MedSRGAN  $^+$   were compared with bicubic interpolation and CNN based SR frameworks including RCAN [ 49 ], ESRGAN [ 45 ] and PSNR-oriented RWMAN (denoted as RWMAN (P) in experiments). RCAN is a PSNR-oriented method and has achieved state-of-the-art PSNR and SSIM on several typical SISR datasets such as Set5, Set14 and Urban100, ESRGAN won PIRM-SR Challenge (at ECCV2018 Workshop, a super resolution competition) on high perceptual quality group, which means they did best on high frequency components reconstruction and got highest perceptual index. RWMAN (P) is the RWMAN trained using MSE loss on pixel level.  

# 3.1 Evaluation  

PSNR and SSIM are still used for the SR framework evaluation in all methods and the SR community is lack of reliable objective metrics that can imitate judgments by human observation. PSNR and SSIM are calculated as:  
$$
M S E=\frac{1}{N^{2}}\sum_{i=1}^{N}\sum_{j=1}^{N}\left(x(i,j)–y(i,j)\right)^{2}
$$  

$$
\mathit{P S N R}=10\cdot\log_{10}\left(\frac{M A X^{2}}{M S E}\right)
$$  

$$
\mathit{S S I M}=\frac{\left(2\mu_{x}\mu_{y}+c_{1}\right)\left(2\sigma_{x y}+c_{2}\right)}{\left(\mu_{x}^{2}+\mu_{y}^{2}+c_{1}\right)\left(\sigma_{x}+\sigma_{y}+c_{2}\right)}
$$  

where  $x,\,y,\,N$   $M A X$   is the maximum value of gray scale;    $\mu,~\sigma$   represent mean and variance,    $\sigma_{x y}$   is the covariance of two images; and two constants    $c_{1}\,{=}\,(0.01\cdot M A X)^{2}$  ,  $c_{2}\,{=}\,(0.03\cdot M A X)^{2}$    are calculated with the convention of SSIM.  

Since PSNR and SSIM often contradict with individual visual judgment, we also perform mean opinion score (MOS) test to qualitatively evaluate visual quality of SR images. Five experienced radiologists are required to give an integer score range from 1 to 5, which represent very annoying, annoying, slightly annoying, perceptible but not annoying, and imperceptible, to an image dataset with hidden tags, which includes 100 randomly selected HR CT slice images and corresponding SR images generated by MedSRGAN and other reference methods.  

# 3.2 CT experiments  

# 3.2.1 Data and training details  

242 thoracic CT scans from LUNA 16 Challenge [ 40 ] are used in our experiments. 219 randomly selected scans (52,073 slices) are used for training and the remaining 23 scans (5889 slices) for testing. All CT slices are   $512\times512$   and used as HR references. LR slices are obtained by applying   $4\times4$   average pooling on corresponding HR slices and have size of   $128\times128$  . Moreover, 30 additional thoracic CT scans (10,732 slices) collected from one China hospital are used for testing MedSRGAN on clinical data.  

All networks take a   $128\times128$   CT slice as input and output a   $512\times512$   CT slice. Our networks are fully convolutional, so they can take CT slices of arbitrary size as input and generate   $4\times$   slices. The intensity of LR and HR CT images is clipped to  $[-1024,\ 1024]$   HU (Hounsfield Unit), and then linearly scaled into range of [0, 1]. To train the generator, we set    $\lambda_{1}=10^{-2}$  ,    $\lambda_{2}=10^{-4}$    and    $\lambda_{L1}\!=\!0$   for loss function (4), and used Adam [ 28 ] optimizer with    $\beta_{1}=0.9$   and    $\beta_{2}\,{=}\,0.999$  . We set the weight of content loss as two orders of magnitude larger than the weight of adversarial losses, such setting is also applied in GAN based methods like SRGAN and ESRGAN, and we tuned    $\lambda_{1}$   and    $\lambda_{2}$   during experiments but effects were limited on generated images. The initial learning rate was set to   $10^{-4}$    and halved every   $50k$   iterations. These experiments were conducted on two NVIDIA Titan   $\mathrm{X\mathfrak{p}}$   GPUs.  
# 3.2.2 MOS test  

Medical images constitute a core portion of the information a radiologist utilizes to render diagnostic and treatment decision. The science of medical image perception is dedicated to understanding and improving the clinical interpretation process. In this study, we focus our evaluation on content-based subjective views, which is our MOS test, from five experienced radiologists in simulated clinical environment.   $100\;\mathrm{CT}$   slices are randomly selected from our test set and to form a MOS test dataset. Among these slice images, half used   $[-160,240]$  HU display window (tissue window) and the other half used [ − 1200, 0] HU (lung window), which are two significant display windows in thoracic CT viewing. For each slice, the HR image (original image) and SR images obtained by methods including bicubic interpolation, RCAN (P), RWMAN (P), ESRGAN, MedSRGAN and   $\mathrm{MedSRGAN+}$   are scored, thus our MOS dataset contains   $100\times(1+6)\,{=}\,700$   CT slice images. These images were randomly shuffled and assured not containing any information that could be implications of which methods to avoid subjective biases before being scored by five experienced radiologists. Results and distribution of their scores are shown in Table  1  and Fig.  6 .  

# 3.2.3 Analysis of results  

The average PNSR/SSIM for all test images using different methods are shown in the Table  2 , and these results were calculated after subjecting the whole [ − 1024, 1024] HU display window to [0, 1]. The result shows that using GAN-based methods can achieve almost the same PSNR/ SSIM performance comparing with PSNR-oriented methods in this task.  

Fig.  7  shows the image examples as well as PSNR/SSIM values. We observe that PSNRoriented RCAN generate over-smoothed images though it has higher PSNR/SSIM, while the patterns (the first row of Fig.  7 ) of images generated by ESRGAN are more visually unnatural, and the textures (the second row of Fig.  7 ) are less obvious than images generated by MedSRGAN. Moreover, these images of ESRGAN are still confronted with serious chessboard artifacts problem, this reveal that densely connected convolutional blocks are probably not suitable for CT SR. After all, the proposed MedSRGAN performs better in rebuilding these missing information and its generated images are more visually feasible.  

Besides, the MOS test of different methods with two display windows are shown in Table  1 and Fig.  6 . We observed that under both tissue window and lung window, the average MOS of our methods are better than other methods on visual judgments, no matter using noise channel  

Table 1  Results of MOS test for CT. (P) after the title of methods indicates this model was trained on PSNRoriented loss (MSE loss) and the postfix '+' denotes noise channel was included.  Boldface  and underline indicate the best and the second-best performance,  italics  indicates scores on HR images 
![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/fde8eb8a4e6ca169da3c4f621871affcb6ec30f3b6f8172aef678d067da9cd5b.jpg)  
![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/5a3437b79910113abdd972f19f6c4a41d3d98008200630ae21acb8077ecc590e.jpg)  
Fig. 6  MOS test for CT on tissue window   $[-160,240]$   and lung window  $[-1200,0]$  . (P) after the title of methods indicates this model was trained on PSNR-oriented loss (MSE loss) and the postfix    $\because$   denotes noise channel was included  

or not. Specifically, from results shown in the first graph of Fig.  6  we found that models with noise channel input helps generator make more natural artifacts and patterns when displaying on tissue window, thus it got a higher score, but it encountered less visual quality when displaying on lung window (see the second graph of Fig.  6 ). So there may exist a trade-off between generating natural patterns and restoring textures with using noise channel or not.  

Table 2  Average PSNR and SSIM of our experiments on  $^{23}\mathrm{~CT}$  additional 30 clinical CT scans (10,732 slices), and 30 brain MRI scans (2,118 slices) for testing. Models in ‘ LUNA 16 Test Set ’  and  ‘ Clinical CT Data ’  were trained with LUNA 16 training set, as stated in Section3.2.1 . Models for  ‘ Brain MRI ’  were trained with brain MRI training data, as stated in Section3.3.1 . (P) after the title of methods indicates this model was trained on PSNR-oriented loss (MSE loss) and the postfix    $\iota_{+},$   denotes noise channel was included.  Boldface  and underline in  ‘ PSNR ’  and  ‘ SSIM ’  column indicate the best and the second best performance on PSNR and SSIM 
![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/77fdf636758f595a2fc2227b91705c48a2aadc1a9235b0969f111ee8cfb54595.jpg)  
# 3.2.4 Ablation study  

Since our method takes PSNR-oriented RCAN as baseline model, the impacts of our meaningful modifications are worthy of comparative analysis. The progress is shown in Table  3  and a brief discussion is illustrated as follows.  

Replace RCAB with RWMAB  By comparing to natural images, some regions in medical images may be worthy of attention in generating high resolution images. We used RWMAB for whole image feature map attentions, to enable the network to focus on those meaningful regions. To better show its effect, we trained a PSNR-oriented RWMAN for comparison, which simply used MSE loss on pixel level for training RWMAN. We observe that RWMAN (P) got a marginal decrease in PSNR and SSIM but a higher increase in MOS test.  

Use GAN-based framework  As illustrated, although PSNR-oriented RWMAN performs not as well as RCAN on PSNR and SSIM, it performs more superior than RCAN in MOS test. Nevertheless, images generated with PSNR-oriented RWMAN are tend to be over-smoothed and lack of high-frequency details. So GAN-based framework with loss function (4) is introduced to allow the generator to gain enhancement from adversarial learning and produce more realistic results.  

Add input Noise Channel  With adversarial learning, SR images generated by MedSRGAN become closer to the real HR images, but the images are still monotonous in pattern generating, especially when displaying images under the tissue window. In order to deal with the problem, we introduce a noise channel as an additional input for making some randomness in network training. We observe MedSRGAN with noise input (MedSRGAN+) got a marginal increase in MOS test under tissue window displaying.  

![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/4d55a92dfd597ae78fcd824f5a42b33e6f3eba762c6dc1a845e10f2501e43160.jpg)  

![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/7244f72b128b1b8378039319c7a62898fceba99f83c317ff6acabfa9ff98114e.jpg)  
(PSNR/SSIM) (29.931/0.830) (31.074/0.884) (30.167/0.864) (30.066/0.868)  

Fig. 7  Results of different methods on  $96\times96$   image patches cropped from  $512\times512$   slices. (P) after the title of methods indicates this model was trained on PSNR-oriented loss (MSE loss) and the postfix    $\iota_{+},$   denotes noise channel was included. Image patches in yellow box are shown in the first row and its display window is  $[-160$  , 240] HU (tissue window), image patches in red box are shown in the second row and its display window is  $[-1200$  PSNR/SSIM. For GAN-based methods, images generated by ESRGAN still have the problem of  chessboard artifacts  (which is difficult to find but extremely noteworthy) while MedSRGAN+ gets rid of these annoying artifacts and looks closer to HR image both on patterns and textures.  Best viewed on screen with zoom in  
Table 3  The ablation study in our experiments. (P) after the title of methods indicates this model was trained on PSNR-oriented loss (MSE loss) and the postfix  $\because$   denotes noise channel was included.  Boldface  and underline indicate the best and second best performance. This table shows the progress on visual quality of our modifications, especially in MOS test 
![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/992235bd3c84cb5e00c209f3e16aca9c79b8041d69e5488b1d701fac5ead9b6e.jpg)  

# 3.2.5 Nodule detection test  

We believe that well rebuilt SR images should not affect real world diagnosis. To better demonstrate the effect on diagnosis, the lung nodule detection test is executed on HR and MedSRGAN  $^+$   generated SR images on a sophisticated CAD (computer aided detection) system. The detection system was well trained with reliable lung nodule annotations and has been used in practice. The test set, mentioned in Section3.2.1 , includes 23 scans with 35 annotated lung nodules from LUNA 16. The distribution of nodules diameter and detection results are shown in Fig.  8 . In total, CAD reports 165 nodules on HR scans with 31 true positive (TP) nodules, and 156 nodules on SR scans with  $32\;\mathrm{TPs}$  , which show that rebuilt SR images almost do not affect the nodule detection in this test.  

# 3.2.6 Results on clinical data  

MedSRGAN framework was also tested on 30 additional clinical CT scans (10,732 slices) from one China hospital, which were obtained by TOSHIBA Aquilion CT scanner and with size of   $512\times512$  . These data may confront a problem of different data distribution from LUNA16 [ 40 ] Dataset because different CT scanners may produce different noise patterns or other artifacts. However, our methods are also able to generate images with high visual quality, as is shown in Fig.  9  with HR reference and results generated by different methods. The first row of Fig.  9  are displayed on   $[-160,\,240]$   HU (tissue window) and the second row are displayed on   $[-1200,\,0]$   HU (lung window). The average PNSR/SSIM of SR using different methods for all 10,732 slices are also collected in Table  2 . Although there are only marginal differences between MedSRGAN and other methods (except for bicubic interpolation) on the average PSNR and SSIM, MedSRGAN outperforms any other methods in reconstructing patterns and textures information on these data and our results are closer to the real images visually as well, which visual results are shown in Fig.  9 .  

# 3.3 MRI experiments  

# 3.3.1 Data and training details  

To demonstrate the flexibility of our methods, 110 brain MRI scans from one China hospital were collected for MRI experiments. All MRI scans were obtained by PHILIPS Ingenia  $3.0\;\mathrm{T}$  
![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/d1cdb0d5f75f1c5829c8ed23cb21d94903e5cec84024ea3f93e48b00c7ebca02.jpg)  
Fig. 8  The nodule detection results on test set. Nodules are divided into three groups according to their diameters to show the effect of SR images on small, medium and large nodules. The SR scans were generated by MedSRGAN+ framework  

MR system and with size of   $672\times672$  , 80 scans (7,661 slices) are used for training and the rest 30 scans (2,118 slices) for testing.  

The imaging method of MRI is quite different from CT and the value of each voxel in MRI does not have specific physical meaning, so zero-mean normalization (subtract the mean and divided by the standard deviation) was applied to each MRI scan before training and testing. LR MRI slices are obtained by   $4\times4$   average pooling on original HR slices. We set  $\lambda_{1}\!=\!5\times$   $10^{-2}$  ,    $\lambda_{2}\,{=}\,5\times10^{-3}$    and    $\lambda_{L1}=10^{-2}$    for loss function (4) in training MRI super-resolution models, and used Adam [ 28 ] optimizer with    $\beta_{1}\!=\!0.9$   and    $\beta_{2}\,{=}\,0.999$  . The initial learning rate was set to   $10^{-4}$    and halved every   $20k$   iterations.  

# 3.3.2 Results  

The average PNSR/SSIM for all brain MRI test images using different methods are shown in the Table  2 . All metrics are calculated on cropped images to eliminate the influence of nonbody area, which would make PSNR/SSIM very high but meaningless. The quantitative results illustrate that all methods have similar performances in PSNR/SSIM except for bicubic interpolation, though the RWMAN (P) has the highest values.  

Fig.  10  shows some patches from original MRI images as well as their PSNR and SSIM on these cropped images. We observe that PSNR-oriented RCAN still result in over-smoothed SR images though it gets high PSNR/SSIM, but in some patches MedSRGAN and MedSRGAN+ can get higher PSNR and SSIM than RCAN. Images generated by ESRGAN still present less feasible artifacts, and   $\mathrm{MedSRGAN+}$   performs better in rebuilding these details, which results look closer to the real HR images.  

# 3.4 Efficiency and parameter counts  

We performed an experiment on the running speed of these methods, which results are listed in Table  4 . Note that RCAN [ 49 ] uses 10 residual groups with each contains 20 RCABs in their experiments, so do we use this configuration in training RCAN (P) and RWMAN (P) in our experiments. The counts and storage of parameters are also collected in Table  4 .  
![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/f1d4053a820d8331c6ae2fd08e48983dcd319544d3616cf646502b27c93e1baf.jpg)  
(PSNR/SSIM) (22.540/0.891) (23.197/0.913) (23.547/0.920) (23.358/0.915)  

Fig. 9  Results of different methods on   $96\times96$   image patches cropped from  $512\times512$   slices on a slice of clinical data. (P) after the title of methods indicates this model was trained on PSNR-oriented loss (MSE loss) and the postfix    $\iota_{+},$   denotes noise channel was included. Image patches in red box are shown in the first row and its display window is  $[-160$  and its display window is  $[-1200$  , 0] HU (lung window). PSNR-oriented RCAN brings over-smoothed images. For GAN-based methods, images generated by ESRGAN still have the problem of  chessboard artifacts  while MedSRGAN+ gets rid of these annoying artifacts and looks closer to HR image both on patterns and textures. These results show that our model performs visually better even though on data set with different distributions. Best viewed on screen with zoom in  

It was found that, 1) it is almost real-time for MedSRGAN framework and ESRGAN to generate an image on GPU based machines; 2) RCAN has no advantages on running speed, which can be explained with its additional channel decreasing and increasing convolutions in RCAB structure, while our network does not contain such step; 3) ESRGAN slightly performed better than MedSRGAN on speed, but the counts of parameter is larger. In conclusion, our method not only got prominent visual results but also is advantageous when taking speed and parameter counts into consideration.  

# 4 Discussion  

In this study, we presented MedSRGAN as an end-to-end framework for medical image superresolution tasks. MedSRGAN is comprised of a novel neural network for SR medical images generation, a pair-wise disc rim in at or and a GAN-based novel loss function. The framework ensures sufficient contents and details as well as realistic for viewing on rebuilt images to a great extent.  

However, we would like to point out that SR images of MedSRGAN are still not perfect because it is impossible for it to rebuild all information in HR based on LR images merely, though MedSRGAN has already got rid of blurring, over-smooth problems and strange artifacts. SR for image generation is actually an under determined problem, which means an output SR image incorporates much more information than its corresponding LR image. To interpret where these extra information come from, we reviewed the whole process of our experiments. In the training stage of MedSRGAN framework, each LR-HR pair was treated as a training sample and was fed to the training process independently, and a loss function was employed to make sure generated SR images get close to HR images as far as possible. As a  
![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/dce7d05af7d38c579e5ab2b14839d642e94c0fba523b00807c005c398105b612.jpg)  

![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/ecc2d081c7d694083be34bede35b1e5e90ec370c49ec130b9b8113186f0708ea.jpg)  

![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/af4efbadaf91c3982a93a1dfbde818883ef3d32450a2998a03a57c4bb8ab0d01.jpg)  

![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/506f42751fcb604f1d79ddaef08d9c55a717b18c2d685826027580e5672026e9.jpg)  
(PSNR/SSIM) (18.823/0.931) (22.011/0.968) (21.117/0.961) (20.077/0.950) (21.562/0.965)  

Fig. 10  Results of different methods on  $48\times48$   image patches cropped from original MRI slice. (P) after the title of methods indicates this model was trained on PSNR-oriented loss (MSE loss) and the postfix    $\iota_{+},$   denotes noise channel was included. The display window is clipped to [0, 480]. Image patches in red box are shown in the first row and image patches in yellow box are shown in the second row. PSNR-oriented RCAN brings over-smoothed images though it has high PSNR/SSIM. For GAN-based methods, images generated by our method look closer to HR image both on patterns and textures. These results show that our model performs better on generating feasible patterns.  Best viewed on screen with zoom in  

result, the general missing information were encoded into neural networks. With the assumption that LR images contain sufficient information for radiologists to make diagnosis, these encoded information hidden in neural networks can be illustrated as high frequency edges, specific artifacts and patterns of a certain kind of medical images. In a specific modality of medical images, such high frequency edges, specific artifacts and patterns are actually  

Table 4  Speed of SR methods and their counts as well as storage of parameters. (P) after the title of RCAN indicates it was trained on PSNR-oriented loss (MSE loss). RCAN (P) uses 10 residual groups with each contains 20 RCABs, as proposed in [ 49 ]. The postfix    $\because$   denotes noise channel was included.  Boldface  and underline indicate the best and second best performance 
![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/d93fac6e49c731c30cd914b1e1b70d5cde6b9acd04b3ebb0b7cae350bfea8c0e.jpg)  
monotonous and stereotyped when comparing to natural images, which make it relatively easy to learn in machine learning frameworks. But it must be pointed out that, in case of extremely tiny focuses which are smaller than  $4\times4$   pixels in high resolution images, no method is able to generate this imaging because all information of such focuses were unfortunately lost. This situation could happen if we take a CT scan on low-resolution scale.  

To find out whether SR images affect diagnosis or not, we performed a simple nodule detection test on both HR CT scans and MedSRGAN generated SR CT scans, as is demonstrated on Section3.2.5 . The results showed that this nodule detection system also performed well in rebuilt SR CT scans, because it obtained almost identical detection results on true positive nodules of different sizes. Also, SR CT scans generated by our MedSRGAN are the closest to real HR scans in human viewing, as was demonstrated in MOS test. Hence, our methods are at least effective and feasible in human viewing and preserving nodules information. However, these evaluations may not cover all diagnosis situations, and we believe more experiments for evaluation should be conducted.  

# 5 Conclusions  

In this study, we presented a GAN-based SR framework for common medical images (MedSRGAN), which included Residual Whole Map Attention Network (RWMAN) as generator, disc rim in at or for image pairs and a novel loss function for training generator. The results show that our method can be directly used in SR reconstruction of CT and MRI images, and their results preserve more feasible texture details and generate realistic patterns on HR images. The reliable and effective results of MedSRGAN imply the feasibility of using the SR method for retrieving more image details on clinical obtained low-resolution images, such as LDCT, low magnetic field MR, and MR spectroscopic imaging. In the future, we will conduct more in-depth studies of MedSRGAN, such as investigating a more sophisticated method to evaluate the SR reconstruction performance.  

Acknowledgments  This work was supported in part by the National Key R&D Program of China under Grant 2018YFC1704206, Grant 2016YFB0200602, in part by the NSFC under Grant 81971691, Grant 81801809, Grant 81830052, Grant 81827802, Grant U1811461, and Grant 11401601, in part by the Science and Technology Program of Guangzhou under Grant 20180420053, in part by the Science and Technology Innovative Project of Guangdong Province under Grant 2016 B 030307003, Grant 2015 B 010110003, and Grant 2015 B 020233008, in part by the Science and Technology Planning Project of Guangdong Province under Key Grant 2017 B 020210001, in part by the Guangzhou Science and Technology Creative Project under Key Grant 201604020003, in part by the Guangdong Province Key Laboratory of Computational Science Open Grant 2018009, in part by the Construction Project of Shanghai Key Laboratory of Molecular Imaging 18 DZ 2260400, and in part by China postdoctoral science foundation No.2019 M 653185.  
