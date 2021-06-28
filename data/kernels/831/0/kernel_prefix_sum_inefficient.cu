#include "includes.h"
# include <bits/stdc++.h>
# include <cuda.h>

#define SIZE 60// Global Size
#define BLOCK_SIZE 1024
using namespace std;

//::::::::::::::::::::::::::::::::::::::::::GPU::::::::::::::::::::::::::::::::

// :::: Kernel




// :::: Calls
__global__ void kernel_prefix_sum_inefficient(double *g_idata,double *g_odata,int l){ // Sequential Addressing technique

__shared__ double sdata[BLOCK_SIZE];
// each thread loads one element from global to shared mem

unsigned int tid = threadIdx.x;
unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

if(i<l && tid !=0){
sdata[tid] = g_idata[i-1];
}else{
sdata[tid] = 0;
}

// do reduction in shared mem
for(unsigned int s=1;s<=tid;s *=2){
__syncthreads();
sdata[tid]+=sdata[tid-s];
}

// write result for this block to global mem
g_odata[i] = sdata[tid];
}