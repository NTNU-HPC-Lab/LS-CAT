#include "includes.h"
__global__ void complexmult_kernal(float *afft, const float *bfft, int totaltc)
{

const uint ridx = 2*(threadIdx.x + (blockIdx.x + blockIdx.y*gridDim.x)*MAX_THREADS);

if(ridx < totaltc){
const uint iidx = ridx + 1;
//maybe use float2 to improve coalessing....

float afftr = afft[ridx];
float affti = afft[iidx];
float bfftr = bfft[ridx];
float bffti = bfft[iidx];

afft[ridx] = afftr*bfftr - affti*bffti;  //real portion
afft[iidx] = affti*bfftr + afftr*bffti; //imaginary portion
}

}