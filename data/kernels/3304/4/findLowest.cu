#include "includes.h"
__global__ void findLowest(int numMin, int *array_val, int *cudaResult ) {
int low = threadIdx.x * numMin;
int high = low + numMin -1;
int min = array_val[low];
for (unsigned int i = low; i < high; i++){
if(array_val[i] < min){
min = array_val[i];
}
}
cudaResult[threadIdx.x] = min;
printf("Thread %d returned: %d \n", threadIdx.x, min);
}