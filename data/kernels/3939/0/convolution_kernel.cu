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


__global__ void convolution_kernel(int h, int w, int d, double* gpu_in, int k_h, int k_w, int k_d, double* kernel_weights, double* kernel_biases, int num_kernels, int op_h, int op_w, int op_d, double* gpu_out)
{
//Identifying threads by their IDs.
int row = blockDim.y*blockIdx.y + threadIdx.y;
int col = blockDim.x*blockIdx.x + threadIdx.x;
int deep = blockDim.z *blockIdx.z + threadIdx.z;
//Return if thread out of bounds
if (row >= op_h || col >= op_w || deep >= op_d) return;
double out=0.0;
int kernel_pointer = 0;
//Each thread/each output node identifies the corresponding element in the matrix that it is responsible to multiply-add.
for (int depth_pointer = 0; depth_pointer < k_d; depth_pointer++) {
for (int row_pointer = 0; row_pointer < k_h; row_pointer++) {
for (int column_pointer = 0; column_pointer < k_w; column_pointer++) {
out += gpu_in[((row*w + col) + row_pointer * w + column_pointer + h * w*depth_pointer)] * kernel_weights[kernel_pointer + deep * k_h*k_w*k_d];
kernel_pointer++;
}
}
}
//Bias addition and relu activation. One bias is applied to one output image layer, since one bias is applicable to one kernel.
//Relu activation : relu(a)=max(0,a). If the value is less than 0 then it becomes 0, else it is retained.
if (out + kernel_biases[deep] < 0.0)
gpu_out[row*op_w + col + deep * op_h*op_w] = 0.0l;
else
gpu_out[row*op_w + col + deep * op_h*op_w] = out + kernel_biases[deep];

}