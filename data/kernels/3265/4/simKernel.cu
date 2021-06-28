#include "includes.h"
__global__ void simKernel(int N_stgy, int N_batch, float *alpha, float *mid, float *gap, int *late, int *pos, int *rest_lag, float *prof, float *last_prc, int *cnt, float fee){
int global_i = blockIdx.x*blockDim.x + threadIdx.x;
if( global_i >= N_stgy) return;
int start = global_i*N_batch + rest_lag[global_i], end = global_i*N_batch + N_batch, i;
for(i = start; i<end; ++i) if(alpha[i]*mid[i%N_batch]>gap[i%N_batch] + fee || alpha[i]*mid[i%N_batch]<-gap[i%N_batch] - fee){
if(alpha[i]*mid[i%N_batch]>gap[i%N_batch]+fee && pos[global_i]<1){
last_prc[global_i] = mid[i%N_batch] + gap[i%N_batch] + fee;
prof[global_i] -= (1-pos[global_i])*last_prc[global_i];
cnt[global_i] += 1-pos[global_i];
pos[global_i] = 1;
i += late[i%N_batch];
}
else if(alpha[i]*mid[i%N_batch]<-gap[i%N_batch]-fee && pos[global_i]>-1){
last_prc[global_i] = mid[i%N_batch] - gap[i%N_batch] - fee;
prof[global_i] += (pos[global_i]+1)*last_prc[global_i];
cnt[global_i] += pos[global_i]+1;
pos[global_i] = -1;
i += late[i%N_batch];
}
}
rest_lag[global_i] = i-end;
}