#include "includes.h"

#define C0  0
#define CZ1 1
#define CX1 2
#define CY1 3
#define CZ2 4
#define CX2 5
#define CY2 6
#define CZ3 7
#define CX3 8
#define CY3 9
#define CZ4 10
#define CX4 11
#define CY4 12

__global__ void prop_gpu(float *p0, float *p1, float *vel, float *coeffs, int _nx, int _ny, int _nz, int _n12){

printf("At the gpu kernel\n");
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
int ii  = row * _nx + col;

if (col >= 4 && col < _nz - 4 && row >= 4 && row < _ny - 4){
for(int z = 4; z < _nz-4; z++){
p0[ii]=vel[ii]*
(
coeffs[0]*p1[ii]
+coeffs[1]*(p1[ii-1]+p1[ii+1])+
+coeffs[2]*(p1[ii-2]+p1[ii+2])+
+coeffs[3]*(p1[ii-3]+p1[ii+3])+
+coeffs[4]*(p1[ii-4]+p1[ii+4])+
+coeffs[5]*(p1[ii-_nx]+p1[ii+_nx])+
+coeffs[6]*(p1[ii-2*_nx]+p1[ii+2*_nx])+
+coeffs[7]*(p1[ii-3*_nx]+p1[ii+3*_nx])+
+coeffs[8]*(p1[ii-4*_nx]+p1[ii+4*_nx])+
+coeffs[9]*(p1[ii-1*_n12]+p1[ii+1*_n12])+
+coeffs[10]*(p1[ii-2*_n12]+p1[ii+2*_n12])+
+coeffs[11]*(p1[ii-3*_n12]+p1[ii+3*_n12])+
+coeffs[12]*(p1[ii-4*_n12]+p1[ii+4*_n12])
)
+p1[ii]+p1[ii]-p0[ii];

ii = ii + _n12;
}

}
}