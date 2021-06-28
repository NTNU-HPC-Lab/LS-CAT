#include "includes.h"
__global__ void KerCalcRidp(unsigned n,unsigned ini,unsigned idini,unsigned idfin,const unsigned *idp,unsigned *ridp)
{
unsigned p=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
if(p<n){
p+=ini;
const unsigned id=idp[p];
if(idini<=id && id<idfin)ridp[id-idini]=p;
}
}