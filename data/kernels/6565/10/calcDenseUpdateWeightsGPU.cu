#include "includes.h"
__global__ void calcDenseUpdateWeightsGPU( float *weights, float *biases, float *gradients, float *dW, float *dB, int batch_size, int in_size_x, int in_size_y, int in_size_z, int out_size_x, int out_size_y, int out_size_z, float learning_rate, float momentum )
{
int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

if ( id < out_size_x ) {
int w_size_x = in_size_x*in_size_y*in_size_z;
// int w_size_y = out_size_x;

for( int h = 0; h < w_size_x; ++h ){
// int index = id * (w_size_x * w_size_y) + h;
int index = h * out_size_x + id;
weights[index] = weights[index] - learning_rate * dW[index];
}

biases[id] = biases[id] - learning_rate * dB[id];

for( int b = 0; b < batch_size; ++b ){
int index = (b * out_size_x + id) * 2;
gradients[index+1] = gradients[index] + gradients[index+1] * momentum;
}
}

/* original
for (int i=0; i<weigts_data_num; ++i){
weights.data[i] = weights.data[i] - lr * 	dW.data[i];
}

for (int i=0; i<out.size.x; ++i){
biases.data[i] = biases.data[i] - lr * 	dB.data[i];
}

for ( int i = 0; i < out.size.x * in.size.b; ++i ){
GradientObject& grad = gradients[ i ];
grad.grad_prev = (grad.grad + grad.grad_prev * _momentum);
}
*/
}