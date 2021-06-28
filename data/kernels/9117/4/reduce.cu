#include "includes.h"
__global__ void reduce( float *a, int size, int c) {
int tid = blockIdx.x;	//Handle the data at the index
int index=c,j=0;//size=b

for(j=index+1;j<size;j++) {
a[((tid+index+1)*size + j)] = (float)(a[((tid+index+1)*size + j)] - (float)a[((tid+index+1)*size+index)] * a[((index*size) + j)]);
}

}