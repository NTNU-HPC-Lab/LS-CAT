#include "includes.h"
const int  Nthreads = 1024,  NrankMax = 3, nt0max = 71, NchanMax = 1024;

//////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
__global__ void reNormalize(const double *Params, const double *A, const double *B, double *W, double *U, double *mu){

int Nfilt, nt0, tid, bid, Nchan,k, Nrank, imax, t, ishift, tmax;
double x, xmax, xshift, sgnmax;

volatile __shared__ double sW[NrankMax*nt0max], sU[NchanMax*NrankMax], sS[NrankMax+1],
sWup[nt0max*10];

nt0       = (int) Params[4];
Nchan     = (int) Params[9];
Nfilt     = (int) Params[1];
Nrank     = (int) Params[6];
tmax = (int) Params[11];
bid 	  = blockIdx.x;

tid 		= threadIdx.x;
for(k=0;k<Nrank;k++)
sW[tid + k*nt0] = W[tid + bid*nt0 + k*Nfilt*nt0];

while (tid<Nchan*Nrank){
sU[tid] = U[tid%Nchan + bid*Nchan  + (tid/Nchan)*Nchan*Nfilt];
tid += blockDim.x;
}

__syncthreads();

tid 		= threadIdx.x;
if (tid<Nrank){
x = 0.0f;
for (k=0; k<Nchan; k++)
x += sU[k + tid*Nchan] * sU[k + tid*Nchan];
sS[tid] = sqrt(x);
}
// no need to sync here
if (tid==0){
x = 0.0000001f;
for (k=0;k<Nrank;k++)
x += sS[k] * sS[k];
sS[Nrank] = sqrt(x);
mu[bid] = sqrt(x);
}

__syncthreads();

// now re-normalize U
tid 		= threadIdx.x;

while (tid<Nchan*Nrank){
U[tid%Nchan + bid*Nchan  + (tid/Nchan)*Nchan*Nfilt] = sU[tid] / sS[Nrank];
tid += blockDim.x;
}

/////////////
__syncthreads();

// now align W
xmax = 0.0f;
imax = 0;
for(t=0;t<nt0;t++)
if (abs(sW[t]) > xmax){
xmax = abs(sW[t]);
imax = t;
}

tid 		= threadIdx.x;
// shift by imax - tmax
for (k=0;k<Nrank;k++){
ishift = tid + (imax-tmax);
ishift = (ishift%nt0 + nt0)%nt0;

xshift = sW[ishift + k*nt0];
W[tid + bid*nt0 + k*nt0*Nfilt] = xshift;
}

__syncthreads();
for (k=0;k<Nrank;k++){
sW[tid + k*nt0] = W[tid + bid*nt0 + k*nt0*Nfilt];
}

/////////////
__syncthreads();

// now align W. first compute 10x subsample peak
tid 		= threadIdx.x;
if (tid<10){
sWup[tid] = 0;
for (t=0;t<nt0;t++)
sWup[tid] += A[tid + t*10] * sW[t];
}
__syncthreads();

xmax = 0.0f;
imax = 0;
sgnmax = 1.0f;
for(t=0;t<10;t++)
if (abs(sWup[t]) > xmax){
xmax = abs(sWup[t]);
imax = t;
sgnmax = copysign(1.0f, sWup[t]);
}

// interpolate by imax
for (k=0;k<Nrank;k++){
xshift = 0.0f;
for (t=0;t<nt0;t++)
xshift += B[tid + t*nt0 +nt0*nt0*imax] * sW[t + k*nt0];

if (k==0)
xshift = -xshift * sgnmax;

W[tid + bid*nt0 + k*nt0*Nfilt] = xshift;
}

}