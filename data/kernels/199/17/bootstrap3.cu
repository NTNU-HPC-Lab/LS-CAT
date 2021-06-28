#include "includes.h"
__global__ void bootstrap3(int bins, int num_els, int num_boots, float *g_idata, double *g_odata, unsigned int *g_irand)
{
float myResample;

int constant = ( 4294967295 / ( bins ) );
int id = threadIdx.x + blockDim.x * blockIdx.x;
int dmid = bins * ( blockDim.y * blockIdx.y + threadIdx.y );
for (int i = 0; i < bins; i++)
{

int rid = g_irand[id * bins + i] / constant;

myResample += g_idata[dmid + rid];
}
dmid = num_boots * ( blockDim.y * blockIdx.y + threadIdx.y );
g_odata[dmid + threadIdx.x + blockDim.x * blockIdx.x] = ( (double) myResample / (double) num_els );

}