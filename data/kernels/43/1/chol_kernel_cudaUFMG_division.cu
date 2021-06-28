#include "includes.h"
__global__ void chol_kernel_cudaUFMG_division(float * U, int elem_per_thr) {
// Get a thread identifier
int tx = blockIdx.x * blockDim.x + threadIdx.x;
int ty = blockIdx.y * blockDim.y + threadIdx.y;

int tn = ty * blockDim.x * gridDim.x + tx;


//#define DEBUGDIV

#ifdef DEBUGDIV

int dbg = 0;
if(blockIdx.x == 4){
if(blockIdx.y == 5){
if(threadIdx.x == 2){
if(threadIdx.y == 1){
dbg = 1;
printf("\n\n");
printf("\ntx=%d \nty=%d \ntn=%d", tx, ty, tn);
}
}
}
}

#endif


for(unsigned i=0;i<elem_per_thr;i++){
int iel = tn * elem_per_thr + i;
int xval = iel % MATRIX_SIZE;
int yval = iel / MATRIX_SIZE;

if(xval == yval){
continue;
}


#ifdef     DEBUGDIV
if(dbg == 1){
if(i==37){
printf("\niel=%d \nxval=%d \nyval=%d", iel, xval, yval);
}
}
#endif


// if on the lower diagonal...
if(yval > xval){
xval = MATRIX_SIZE - xval - 1;
yval = MATRIX_SIZE - yval - 1;
}


int iU = xval + yval * MATRIX_SIZE;
int iDiag = yval + yval * MATRIX_SIZE;

#ifdef     DEBUGDIV
if(dbg == 1){
if(i==37){
printf("\nxtrans=%d \nytrans=%d \niU=%d \niDiag=%d", xval, yval, iU, iDiag);
printf("\n\n");
}
}
#endif

U[iU] /= U[iDiag];

}

}