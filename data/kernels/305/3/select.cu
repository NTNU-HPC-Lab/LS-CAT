#include "includes.h"

__device__ int position;			//index of the largest value
__device__ int largest;				//value of the largest value
int lenString = 593;
int maxNumStrings = 1000000;
int threshold = 2;

__device__ void cuda_select(int *db, int size) {
int my_id = blockDim.x * blockIdx.x + threadIdx.x;

if(my_id < size) {
if(db[2 * my_id] > db[2 * my_id + 1])
db[my_id] = db[2 * my_id];
else
db[my_id] = db[2 * my_id + 1];
}
}
__global__ void select(int *db, int size) {
int height = (int)ceil(log2((double)size));
int i = 0;

for(i = 0; i < height; i++) {
size = (int)ceil((double) size/2);
cuda_select(db, size);
}
largest = db[0];
}