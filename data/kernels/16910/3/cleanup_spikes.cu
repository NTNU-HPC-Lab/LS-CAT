#include "includes.h"
const int  Nthreads = 1024, maxFR = 5000, NrankMax = 6;
//////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////
__global__ void	cleanup_spikes(const double *Params, const float *err, const int *ftype, float *x, int *st, int *id, int *counter){

int lockout, indx, tid, bid, NT, tid0,  j, t0;
volatile __shared__ float sdata[Nthreads+2*81+1];
bool flag=0;
float err0, Th;

lockout   = (int) Params[4] - 1;
tid 		= threadIdx.x;
bid 		= blockIdx.x;

NT      	=   (int) Params[0];
tid0 		= bid * blockDim.x ;
Th 		= (float) Params[2];

while(tid0<NT-Nthreads-lockout+1){
if (tid<2*lockout)
sdata[tid] = err[tid0 + tid];
if (tid0+tid+2*lockout<NT)
sdata[tid+2*lockout] = err[2*lockout + tid0 + tid];
else
sdata[tid+2*lockout] = 0.0f;

__syncthreads();

err0 = sdata[tid+lockout];
t0 = tid+lockout         + tid0;
if(err0 > Th*Th && t0<NT-lockout-1){
flag = 0;
for(j=-lockout;j<=lockout;j++)
if(sdata[tid+lockout+j]>err0){
flag = 1;
break;
}
if(flag==0){
indx = atomicAdd(&counter[0], 1);
if (indx<maxFR){
st[indx] = t0;
id[indx] = ftype[t0];
x[indx]  = err0;
}
}
}

tid0 = tid0 + blockDim.x * gridDim.x;
}
}