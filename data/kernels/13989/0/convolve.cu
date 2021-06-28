#include "includes.h"

/*
* Week 3
* Parallel Programming
* 2011-2012
* University of Birmingham
*
* This is a first step towards implementing "parallel reduce".
* Reducing means using an operation to aggregate the values of
* a data type, such an array or a list.
*
* For example, to calculate the sum we aggregate addition:
*     a1 + a2 + a3 + a4 ...
* To calculate the maximum we aggregate the max operation:
*     max (a1, max(a2, max(a3, ...
* Note that the order in which the device map, which is parallel,
* and the host map, which is sequential, will differ, therefore the
* operation needs to be associative.
* Operations such as +, * or max are associative, but function of
* two arguments, in general, are not!
*/




using namespace std;


const int ITERS = 500;




/*
* Reference CPU implementation, taken from http://www.songho.ca/dsp/convolution/convolution.html
*/
__global__ void convolve(float* data_in, float* data_out, float* kernel, int kernelSize, int BLOCK_SIZE)
{
int tx = threadIdx.x;
int bk = blockIdx.x;
int pos = (bk * BLOCK_SIZE) + tx;
data_out[pos] = 0;

for(int i = 0; i < kernelSize; i++){
if(pos - i >= 0) {
data_out[pos] += kernel[i] * data_in[pos - i];
}
}

}