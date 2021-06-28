#include "includes.h"
__global__ void SoftmaxLossBackprop( const float *label, int num_labels, int batch_size, float *diff ) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if( idx >= batch_size ) {
return;
}

const int label_value = static_cast<int>(label[ idx ]);

// For each item in the batch, decrease the result of the label's value by 1
diff[ idx * num_labels + label_value ] -= 1.0f;
}