#include "includes.h"
__global__ void scatterKernel( const unsigned int *d_In, const unsigned int *d_FalseKeyAddresses, unsigned int *d_Out, const unsigned int totalFalses, size_t size, unsigned int bitPos)
{
int threadsPerBlock = blockDim.x * blockDim.y;

int blockId = blockIdx.x + (blockIdx.y * gridDim.x);

int threadId = threadIdx.x + (threadIdx.y * blockDim.x);

int myId = (blockId * threadsPerBlock) + threadId;

if (myId < size)
{
unsigned int _myFalseKeyAddress =
d_FalseKeyAddresses[myId];

// Calculate true sort key address
int _trueSortKeyAddress =
myId - _myFalseKeyAddress + totalFalses;

// True sort key?
unsigned int _trueSortKey =
(d_In[myId] >> bitPos) & 0x1;

int _destinationAddress =
_trueSortKey ?
_trueSortKeyAddress :
_myFalseKeyAddress;

d_Out[_destinationAddress] =
d_In[myId];

}
}