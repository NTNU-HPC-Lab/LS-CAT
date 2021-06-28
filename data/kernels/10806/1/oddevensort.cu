#include "includes.h"
__global__ void oddevensort(int *in, int *out, int size)
{
bool oddeven=true;
__shared__ bool swappedodd;
__shared__ bool swappedeven;
int temp;
swappedodd=true;
swappedeven=true;

while(true)
{
if(oddeven==true)
{
printf(" \n Swapping at odd locations ");
__syncthreads();
swappedodd=false;
__syncthreads();

int idx=threadIdx.x + blockIdx.x * blockDim.x;
if(idx < (size / 2))
{
if(in[2 * idx] > in[2 * idx +1])
{
printf("\n Thread Id %d : is swapping %d <-> %d  \n Thread Id %d : [%d] <-> [%d] \n ", idx, in[2 * idx] ,  in[2 * idx + 1], idx, 2 * idx, (2 * idx +1));

temp = in[2 * idx];
in [2 * idx]= in[2 * idx + 1];
in [2 * idx + 1]=temp;
swappedodd = true;
}
}

__syncthreads();
}

else
{
//printf("Swapping at even locations \n ");
__syncthreads();
swappedeven=false;
__syncthreads();

int idx=threadIdx.x + blockIdx.x * blockDim.x;
if(idx < (size / 2) - 1)
{
if(in[2 * idx + 1] > in[2 * idx +2])
{
printf("\n Thread Id %d : is swapping %d <-> %d  \n Thread Id %d : [%d] <-> [%d] \n ", idx, in[2 * idx + 1] ,  in[2 * idx + 2], idx, 2 * idx + 1, (2 * idx +2));

temp = in[2 * idx + 1];
in [2 * idx + 1]= in[2 * idx + 2];
in [2 * idx + 2] = temp;
swappedeven=true;
}
}
__syncthreads();
}

if(!(swappedodd || swappedeven ))
break;
oddeven = !oddeven;
}

__syncthreads();

int idx =threadIdx.x;

if(idx < size)
out[idx] = in[idx];
}