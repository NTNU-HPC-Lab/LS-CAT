#include "includes.h"
__global__ void cudaGetShiftedMidPrice(int N_inst, int batch_size, float *alphas, float *mid, float *shifted_prc){
int b_sz = blockDim.x, b_id = blockIdx.x, t_id = threadIdx.x;
if(b_id < N_inst){
for(int i=t_id; i<batch_size; i += b_sz){
shifted_prc[b_id * batch_size + i] = (1. + alphas[b_id * batch_size + i]) * mid[i];
}
}
}