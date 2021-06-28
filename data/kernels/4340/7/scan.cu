#include "includes.h"
__global__ void scan(float * input, float * output,  int len) {
//@@ Load a segment of the input vector into shared memory
__shared__ float sh_input[2048];


int tx = threadIdx.x;
int tx2= tx + blockDim.x;
int bdimx = blockDim.x;
int i = 2*blockIdx.x*blockDim.x + tx;

int start = 2*blockIdx.x*blockDim.x;
int Col1 = start + tx;
int Col2 = start + bdimx + tx;

if( Col2 < len)
{
// Collaborative loading of A
sh_input[tx] = input[ Col1];
sh_input[tx2] = input[ Col2];
}
else if ( Col1 < len)
{	// Control divergence at the edge
sh_input[tx] = input[ Col1];
sh_input[tx2]= 0.0f;
}
else
{	// Control divergence at the edge
sh_input[tx] = 0.0f;
sh_input[tx2]= 0.0f;
}
__syncthreads();
//output[Col1] = sh_input[tx]; output[Col2] = sh_input[tx2];

unsigned int stride; int index;
//  @@ Traverse the reduction tree down
for (stride = 1;stride <= 2*bdimx ; stride *= 2)
{
index = (tx +1)* stride*2 -1;
if (index  < 2*bdimx)
sh_input[index] += sh_input[index-stride];
__syncthreads();
}

//@@ Traverse the reduction tree up
for ( stride = bdimx/2;  stride > 0; stride/=2)
{
__syncthreads();
index = (tx +1)* stride*2 -1;
if (index + stride < 2*bdimx)
sh_input[index+stride] += sh_input[index];
}

//@@ Write the computed sum of the block to the output vector at the
//@@ correct index
__syncthreads();
output[i] = sh_input[tx];
if ( i + bdimx < len)
{
output[i + bdimx] = sh_input[tx2];
}

}