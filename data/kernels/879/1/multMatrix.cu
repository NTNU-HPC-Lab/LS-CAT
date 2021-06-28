#include "includes.h"
__global__ void multMatrix(int *d1_in, int *d2_in, int *d_out, int n, int m, int k){
int indx = threadIdx.x;
int indy = threadIdx.y;
int ind = indy*k+indx;
//printf("%d %d\n",indy,indx);
if(ind<n*k){
d_out[ind] = 0;
for(int i=0;i<m;i++){
d_out[ind] += d1_in[indy*m+i]*d2_in[i*k+indx];
}
}
}