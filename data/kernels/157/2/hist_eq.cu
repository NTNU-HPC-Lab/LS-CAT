#include "includes.h"

#define HISTOGRAM_LENGTH 256












__global__ void hist_eq(unsigned char * deviceCharImg, float * output, float* cdf, float cdfmin, int size)
{
int bx = blockIdx.x;
int tx = threadIdx.x;


int i = tx+blockDim.x*bx;

if(i < size)
{
deviceCharImg[i] = min(max(255*(cdf[deviceCharImg[i]] - cdfmin)/(1 - cdfmin),0.0),255.0);

output[i] = (float) (deviceCharImg[i]/255.0);

}
}