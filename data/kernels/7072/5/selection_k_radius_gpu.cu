#include "includes.h"
__global__ void selection_k_radius_gpu(int b, int m, int k, float radius, const int* idx, const float* val, int* idx_out, float* val_out){
int batch_index = blockIdx.x;
int stride = batch_index * m * k;
idx += stride;
val += stride;
idx_out += stride;
val_out += stride;
for(int i = threadIdx.x; i < m;i += blockDim.x) {

for(int j = 0;j < k;j ++) {
if(val[i * k + j] < radius) {
idx_out[i * k + j] = idx[i * k + j];
val_out[i * k + j] = val[i * k + j];
} else {
idx_out[i * k + j] = idx[i * k ];
val_out[i * k + j] = val[i * k ];
}
}
}
}