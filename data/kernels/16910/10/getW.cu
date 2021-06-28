#include "includes.h"
const int  Nthreads = 1024,  NrankMax = 3, nt0max = 71, NchanMax = 1024;

//////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
__global__ void getW(const double *Params, double *wtw, double *W){

int Nfilt, nt0, tid, bid, i, t, Nrank,k, tmax;
double x, x0, xmax;
volatile __shared__ double sW[nt0max*NrankMax], swtw[nt0max*nt0max], xN[1];

nt0       = (int) Params[4];
Nrank       = (int) Params[6];
Nfilt    	=   (int) Params[1];
tmax = (int) Params[11];

tid 		= threadIdx.x;
bid 		= blockIdx.x;

for (k=0;k<nt0;k++)
swtw[tid + k*nt0] = wtw[tid + k*nt0 + bid * nt0 * nt0];
for (k=0;k<Nrank;k++)
sW[tid + k*nt0] = W[tid + bid * nt0  + k * nt0*Nfilt];
__syncthreads();


// for each svd
for(k=0;k<Nrank;k++){
for (i=0;i<100;i++){
// compute projection of wtw
x = 0.0f;
for (t=0;t<nt0;t++)
x+= swtw[tid + t*nt0] * sW[t + k*nt0];

__syncthreads();
if (i<99){
sW[tid + k*nt0] = x;
__syncthreads();

if (tid==0){
x0 = 0.00001f;
for(t=0;t<nt0;t++)
x0+= sW[t + k*nt0] * sW[t + k*nt0];
xN[0] = sqrt(x0);
}
__syncthreads();

sW[tid + k*nt0] = x/xN[0];
__syncthreads();
}
}

// now subtract off this svd from wtw
for (t=0;t<nt0;t++)
swtw[tid + t*nt0] -= sW[t+k*nt0] * x;

__syncthreads();
}


xmax = sW[tmax];
__syncthreads();

sW[tid] = - sW[tid] * copysign(1.0, xmax);

// now write W back
for (k=0;k<Nrank;k++)
W[tid + bid * nt0  + k * nt0*Nfilt] = sW[tid + k*nt0];

}