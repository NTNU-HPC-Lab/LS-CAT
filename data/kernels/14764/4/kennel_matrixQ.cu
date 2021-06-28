#include "includes.h"
__device__ double atomicAdd_dB(double* address, double val)
{
unsigned long long int* address_as_ull =
(unsigned long long int*)address;
unsigned long long int old = *address_as_ull, assumed;

do {
assumed = old;
old = atomicCAS(address_as_ull, assumed,
__double_as_longlong(val +
__longlong_as_double(assumed)));

// Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
} while (assumed != old);

return __longlong_as_double(old);
}
__device__ double atomicAdd_d(double* address, double val)
{
unsigned long long int* address_as_ull =
(unsigned long long int*)address;
unsigned long long int old = *address_as_ull, assumed;

do {
assumed = old;
old = atomicCAS(address_as_ull, assumed,
__double_as_longlong(val +
__longlong_as_double(assumed)));

// Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
} while (assumed != old);

return __longlong_as_double(old);
}
__global__ void kennel_matrixQ(double *d_P, double *d_px, double *d_py, double *d_Q, int Ng)
{
int rownum,colnum;
int tid = threadIdx.x + blockDim.x * blockIdx.x;
double temp;

rownum = tid/Ng;
colnum = tid%Ng;
d_Q[(tid+Ng)]=0;	//Ng is added to point in next row

for(int k=0; k<Ng; ++k)
{
if (d_px[rownum] && d_py[k])  // make sure to protect division by zero
{
temp=d_P[k+rownum*Ng]*d_P[k+colnum*Ng]/d_px[rownum]/d_py[k];
atomicAdd_dB(&d_Q[(tid+Ng)],temp);
}
}
}