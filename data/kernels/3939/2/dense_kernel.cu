#include "includes.h"
//Training of the CNN is done using Keras. After training for 10 epochs, the obtained accuracy on the training data set is 99.70 and on the test data set is 99.14.
//This model implements the following layes in order- 2DConvolution---->Maxpooling---->2D Convolution---->Maxpooling---->Fully_connected layer---->Fully_connected layer.
//The image is a 28*28 greyscale image. The specifications of the layers are as follows:
//Layer_0: Convolution: 32 3*3 kernels with no padding and 1 stride.
//Layer_1: Maxpooling: 2*2 filters with with no padding and 1 stride.
//Layer_2: Convolution: 64 3*3 kernels with no padding and 1 stride.
//Layer_3: Maxpooling: 2*2 filters with with no padding and 1 stride.
//Layer_4: Flattening
//Layer_5: Fully connected / dense layer with 1024 output units.
//Layer_6: Dropout (done during training only).
//Layer_7: Fully connected / dense layer with 10 output units.

//All arrays and matrices are designed to be row ordered in this implementation.



//Kernel that does convolution. This convolution is done by each thread identifying that patch or portion of the image that it is responsible for its result and does the multiplication and addition of it's patche's values with the suitable kernel.
//The depth of the output image is the number of kernels.

//Kernel that does maxpooling.

//This kernel implements the fully connected layers.


__global__ void dense_kernel(int num_input, int num_output, double* gpu_in, double* weights, double* biases, double* gpu_out, int num_classes)
{
int tid = blockDim.x*blockIdx.x + threadIdx.x;
if (tid >= num_output) return;
double sum = 0.0l;
//The weights are extracted from Keras such that all the weights to one output node appears together, followed by weights to the next node and so on.
//Thus, each output node will be a multiply add of adjacent weight values with the input nodes.
for (int count = 0; count < num_input; count++) {
sum += gpu_in[count] * weights[tid*num_input + count];
}
sum += biases[tid];

//Activation: If the layer is the final layer, then don't do anything, otherwise relu activation max(0,value) is taken.
if ((num_output) != num_classes) {
if (sum < 0.0) {
sum = 0.0l;
}
}
gpu_out[tid] = sum;
}