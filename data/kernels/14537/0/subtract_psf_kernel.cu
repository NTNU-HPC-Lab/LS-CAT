#include "includes.h"
__global__ void subtract_psf_kernel( double * res_p_trans , const double * psf_p_trans , const int stopx , const int stopy , const int diff , const int linsize , const double peak_x_gain ) {
const int
x =  threadIdx.x + (blockIdx.x * blockDim.x)
, y =  threadIdx.y + (blockIdx.y * blockDim.y)
, tid = y * linsize + x
;
if (x < stopx && y < stopy) res_p_trans[tid] -= peak_x_gain * psf_p_trans[tid + diff];
}