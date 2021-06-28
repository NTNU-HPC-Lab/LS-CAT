#include "includes.h"
__global__ void error_reduc(double *error_d, int bit){
__shared__ double error_ds[55];
int tix = threadIdx.x;
error_ds[tix] = error_d[tix];

__syncthreads();

for(int s = 32; s > 0; s>>=1){
//int index = 2 * s * threadIdx.x;

if(tix < s && (tix+s) < true_sample){
error_ds[tix] += error_ds[tix + s];
}

__syncthreads();
}


if(tix == 0){
//printf("GPU Error before divide: %f\n",error_d[0]);
error_ds[tix] /= 55.0;
printf("GPU Error: %f\n", error_ds[tix] * 100.0);
}

error_d[tix] = 0.0;

}