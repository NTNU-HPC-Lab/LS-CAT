#include "includes.h"
// %%cu
// as data type is int, sum might overflow (depending on rand(), but the seq and parallel answers are still equal, or change int to long long (too lazy sorry))
#define THREADS_PER_BLOCK 256
using namespace std;

__global__ void calculate(int *arr_in, int* arr_out, int sz, int option){
int ind = threadIdx.x;
int dim = blockDim.x;
extern __shared__ int shared_mem[];
int actual_ind = blockIdx.x*blockDim.x + ind;
if(actual_ind < sz){
shared_mem[ind] = arr_in[actual_ind];
}else{
if(option == 0 || option == 3)
shared_mem[ind] = 0;
else if(option == 1){//maximum
shared_mem[ind] = -INT_MAX;
}else{//minimum
shared_mem[ind] = INT_MAX;
}
}
__syncthreads();
for(int i=dim/2 ; i > 0 ; i=i/2){
if(ind<i){
if(option == 0 || option == 3)
shared_mem[ind]+=shared_mem[ind+i];
else if(option == 1){
shared_mem[ind]=max(shared_mem[ind],shared_mem[ind+i]);
}else{
shared_mem[ind]=min(shared_mem[ind],shared_mem[ind+i]);
}
}
__syncthreads();
}
arr_out[blockIdx.x]=shared_mem[0];
}