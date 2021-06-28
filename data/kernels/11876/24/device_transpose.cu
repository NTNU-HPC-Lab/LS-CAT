#include "includes.h"
__global__ void device_transpose ()
{
int ivis, ihid ;

ivis = blockIdx.x * blockDim.x + threadIdx.x ;
if (ivis >= d_n_inputs)
return ;

ihid = blockIdx.y ;

d_wtr[ivis*d_nhid_cols+ihid] = d_w[ihid*d_n_inputs_cols+ivis] ;
}