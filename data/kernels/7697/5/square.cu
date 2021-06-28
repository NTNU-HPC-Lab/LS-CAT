#include "includes.h"
__global__ void square( int *d_num_steps, unsigned long long *d_fact, double *d_out){
int idx = threadIdx.x;
int num_steps = *d_num_steps;
for(int k=idx+1; k< num_steps; k+=blockDim.x){
d_out[idx] += (double) k*0.5/ (double) d_fact[k-1];
}

}