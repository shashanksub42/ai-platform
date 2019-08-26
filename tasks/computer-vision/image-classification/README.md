# Image Classification

## Classified the CIFAR10 dataset using ResNet model.

81% accuracy achieved after 10 epochs, running PyTorch on the GPU. 
MLFlow used to log the parameters and save the model artifact.

## Model Architecture
I have implemented the ResNet-18 architecture. This architecture consisists of one convolution layer followed by four residual blocks, each with two convolution layers and a shortcut block, which is used for downsampling the inputs. 

### Residual Block
1. **Convolution Layer 1**
    - Input and output channels are the same as they are passed when calling this block, as well as the stride. Padding is set to 1. This layer downsamples the images if the input and output size are different and sends it to the next layer. If the input and output size are the same, then no downsampling is done. 
2. **Convolution Layer 2**
    - In this layer, the input channels are equal to the output channels. This means there is no downsampling done in this channel. It uses a 3x3 kernel, with stride = 1 and padding = 1.
3. **Shortcut Layer**
    - Convolution Layer:  This block is executed when the input channels are not equal to the output channels or when the stride is not equal to 1. This layer takes the original inputs and downsamples them. Then adds them together with the outputs of the previous two convolution layers. Kernel size is 1x1, stride = 1 and padding = 0.
    
### ResNet 
1. **Convolution Layer**
    - This layer takes the input images which have dimensions 3x32x32 (3 channels - red, green and blue - and 32x32 pixels in the image). It will convolve 64 filters, each of size 3x3x3. Since padding is set to 1 and stride is also set to 1, the output of this layer will be 64x32x32
2. **Residual Block 1**
    - This layer takes the 64x32x32 output from the convolution layer. No downsampling will occur as the input and output dimensions of this block are the same, and the stride is also equal to 1. Therefore, the output size of block 1 will be 64x32x32.
3. **Residual Block 2**
    - This layer takes the output from block 2 and downsamples them. Since stride is set to 2, and padding is 1, the output of this layer will be 128x16x16
4. **Residual Block 3**
    - This layer will also downsample the outputs from block 2, as the input and output size to this block differ, and stride is also not equal to 1. With stride = 2, padding = 1 and kernel = (3,3), the output of this block will be 256x8x8
5. **Residual Block 4**
    - Outputs of block 3 are downsampled to 512x4x4
6. **Fully connected layer**
    - Outputs of block 4 are flattened to a 512 size vector and then spewed out as 10 classes. 


## Running the model
1. Navigate to **tasks/computer-vision/image-classification** inside the **ai-platform** directory
2. Inside command line, run the command `python cifar-proto.py`.
