#include "includes.h"
// richu shaji abraham richursa
using namespace std;
__device__ int function(int value , int bit ,int bitset)
{
if(bitset == 1 )
{
if((value & bit)  != 0)
{
return 1;
}
else
return 0;
}
else
{
if((value & bit) == 0)
{
return 1;
}
else
{
return 0;
}
}
}
__global__ void predicateDevice(int *d_array , int *d_predicateArrry , int d_numberOfElements,int bit,int bitset)
{
int index = threadIdx.x + blockIdx.x*blockDim.x;
if(index < d_numberOfElements)
{

d_predicateArrry[index] = function(d_array[index],bit,bitset);
}
}