# Image_Captioning
The image captioning project aims at creating a neural network with Convolution Neural Network (CNN) - Recurrent Neural Network (RNN) architecture to automatically generate 
caption for an image. The Convolution Neural Network (CNN) acts as the encoder which sends image feature map to an embedding layer to adjust the input size. Meanwhile, the 
Recurrent Neural Network (RNN) acts as the decoder which takes the resized feature map from embedding layer and then process it through Long-Short Term Memory (LSTM) component.

## Installation
1. Clone the repository and download the Jupyter Notebook through **https://jupyter.org/**. It is highly recommended to open and run this project through Jupyter Notebook
```
git clone https://github.com/zilinli0130/Image_Captioning

```

2. Install PyTorch and torchvision; this should install the latest version of PyTorch.
	
	- __Linux__ or __Mac__: 
	```
	conda install pytorch torchvision -c pytorch 
	```
	- __Windows__: 
	```
	conda install pytorch-cpu -c pytorch
	pip install torchvision
	```

3. Install a few required pip packages, which are specified in the requirements text file (including OpenCV).
```
pip install -r requirements.txt
```
 


## Usage

1. The `1_Preliminaries.ipynb` file is to transform the images from COCO Dataset to a standard size and translate the training caption text to token vector. The token vocabulary
consists of 9955 words generated from the project's dataset. Each element inside the token vector corresponds to a unqiue word from the token vocabulary hash table. The size of feature
map after the CNN processing is `torch.size([10, 3, 224, 224])` and the size of token vector is `torch.size([10, 128])`. It tells us that the image is in RGB format with depth of 3 and the 
`batch_size` is 10. Moreover, each caption consists of 12 words. The batch size of 10 is just an example and it will increase to `batch_size = 128` in the training phase. The `embed_size = 256`
represents the size of feature map after the embedding layer process, and size of feature map becomes `torch.size([10, 256])` before heading into RNN decoder. The `embed_size ` also will increase
in the training phase.

2. The detail of training phase contains in `2_Training.ipynb`. The training images will be stacked into batches with a fixed size and set as the input to CNN encoder. The provided pre-trained CNN 
encoder, namely ResNet-50, processes the image batches through a bunch of convolution layers with acitviation, pooling layers and fully connected layers. It is not necessary to output the image classes 
as a distribution scores. Instead, the feature maps will be flatten through an embedding layer and turn out as a vector with dimension of `(batch_size,embed_size)`. Meanwhile, the image captions are also 
stacked into batches with the same size of images, each words inside those captions will be transform into numerical values since the RNN architecture is not able to process pure words. The caption vectors 
are also reshaped through the embedding layers and turn out as an embedded token vector with dimension of `(batch_size,caption_length)`. Both of image vector and embedded token vector are then passed 
to RNN for further tranining process. The output of RNN should has the dimension of `(batch_size,caption_length,vocab_size)`. We can treat this dimension as a (i, j, k) structure which tells us that the output 
describes how likely the jth token in ith caption will be the kth vocabulary in vocabulary set. I remained most of the parameters as default, the embed_size and hidden_size are suggested to be 512 according to 
the paper (arXiv:1411.4555v2 [cs.CV] 20 Apr 2015). Moreover, I increase the batch size from 64 to 256 at an increment of 64, it turns out that model with batch_size of 128 performs better.

3. The `3_Inference.ipynb` is for testing phase. Some trainble parameters include batch_size and learning rate. This is acutally a trial and error process, I tried to increase the batch_size from 32 to 256 at an increment of 32, it turns out that batch_size = 128 works better. Moreover, the learning rate is also an importnant parameter since it decides the frequency for the model to adjust training weights. A large learning rate will make the weight adjustment oscillates obviously and doesn't reduce the error at all, however, a small learning rate will make the training process time-intensive and inefficient. I tried to increase the learning rate at the order of 0.00005, 0.0005, 0.005, 0.05, 0.5.The training process is extremely slow and I research on the default learning rate for Adam optimizer in Pytorch, the value is 1.0. Therefore, I set the batch_size ot 128 and keep the learning rate as default for Adam optimizer.
I did a research on some popular optimizer such as SGD and Adam. SGD uses a variant gradient approach which calculates only small set of data or ramdomly choosed data. It performs well with a small learning rate and is not competitive when the learning rate is large. Therefore, the training process might be slow expectedly. On the other hand, Adam optimizer combines the advantages of SGD and the learning rate is adaptive during the training process. It would choose an appropriate learning rate at different situations. Therefore, Adam is more efficient and less time-intensive, I choose the Adam as my training optimizer.
      
