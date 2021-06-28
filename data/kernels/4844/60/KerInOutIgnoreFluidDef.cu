#include "includes.h"
__global__ void KerInOutIgnoreFluidDef(unsigned n,typecode cod,typecode codnew,typecode *code)
{
const unsigned p=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
if(p<n){
if(code[p]==cod)code[p]=codnew;
}
}