#include "includes.h"

#define getPos(a,k) (((a)>>(k-1))&1)

extern "C" {



}
__global__ void replace(int * input_T, int * output_T, int * prefix_T, int * prefix_helper_T, int n, int k, int blockPower) {
for(int i = 0; i<blockPower; i++) {
int oldpos = threadIdx.x + 1024*blockIdx.x + i*1024*gridDim.x;
if(oldpos >= n) return ;


int newpos = prefix_T[oldpos] + prefix_helper_T[blockIdx.x + i*gridDim.x];

if(getPos(input_T[oldpos],k) == 0) {
newpos = oldpos - newpos;
} else {
newpos = prefix_helper_T[(n+1023)/1024] + newpos - 1;
}

output_T[newpos] = input_T[oldpos];
}

}