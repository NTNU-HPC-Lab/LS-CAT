#include "includes.h"
__global__ void update_wout(double * weights_out_d, double *weights_out_delta_d, int bit){

//__shared__ double weights_out_delta_ds[10 * 55];

int tix = threadIdx.x;
int tiy = threadIdx.y;

int offset = OUTPUTS * H_HEIGHT;
//weights_out_delta_ds[tiy*offset+tix] = weights_out_delta_d[tiy*offset+tix];

for(int s=32; s > 0; s>>=1){
//int index = 2 * s * tiy;

if(tiy < s && (tiy+s) < blockDim.y)
weights_out_delta_d[tiy*offset+tix] += weights_out_delta_d[(tiy+s)*offset+tix];

__syncthreads();
}

if(tiy == 0){
weights_out_d[tix] -= (alpha_d * weights_out_delta_d[tix] / (true_sample*55.0));
}
__syncthreads();
weights_out_delta_d[tiy*offset+tix] = 0.0;
}