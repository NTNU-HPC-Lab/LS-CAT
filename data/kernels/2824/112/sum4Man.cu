#include "includes.h"
__global__ void sum4Man(float *A, float *B, float *C, const int N)
{

int i = blockIdx.x * blockDim.x + threadIdx.x;
float x[4],a[4],b[4],c[4];

a[0] = A[i];
b[0] = B[i];
x[0] = a[0]/7.0;
c[0] = a[0]/3 + 17*b[0] + 3*b[0];
i += blockDim.x * gridDim.x;
a[1] = A[i];
b[1] = B[i];
x[0]*= a[0];
x[1] = a[1]/7.0;
c[1] = a[1]/3 + 17*b[1] + 3*b[1];
x[0]= a[0]*x[0] + x[0]*b[0]*7;
i += blockDim.x * gridDim.x;
a[2] = A[i];
b[2] = B[i];
x[1]*= a[1];
x[2] = a[2]/7.0;
c[2] = a[2]/3 + 17*b[2] + 3*b[2];
x[1]= a[1]*x[1] + x[1]*b[1]*7;
i += blockDim.x * gridDim.x;
if  (i < N) {
a[3] = A[i];
b[3] = B[i];
}
x[2]*= a[2];
x[3] = a[3]/7.0;
c[3] = a[3]/3 + 17*b[3] + 3*b[3];
x[2]= a[2]*x[2] + x[2]*b[2]*7;
x[3]*= a[3];
x[3]= a[3]*x[3] + x[3]*b[3]*7;



i = blockIdx.x * blockDim.x + threadIdx.x;
C[i] += c[0]- x[0];
i += blockDim.x * gridDim.x;
C[i] += c[1]- x[1];
i += blockDim.x * gridDim.x;
C[i] += c[2]- x[2];
i += blockDim.x * gridDim.x;
if  (i < N) C[i] += c[3]- x[3];
}