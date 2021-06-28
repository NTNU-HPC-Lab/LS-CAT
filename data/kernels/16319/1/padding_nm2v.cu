#include "includes.h"
__global__ void padding_nm2v( float *nm2v_re, float *nm2v_im, int nfermi, int norbs, int nvirt, int vstart)
{
int i = blockIdx.x * blockDim.x + threadIdx.x; //nocc
int j = blockIdx.y * blockDim.y + threadIdx.y; //nvirt

if (i > vstart && i < nfermi)
{
if ( j < norbs - vstart )
{
nm2v_re[i*nvirt + j] = 0.0;
nm2v_im[i*nvirt + j] = 0.0;
}
}

}