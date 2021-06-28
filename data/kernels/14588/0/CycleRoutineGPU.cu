#include "includes.h"




__global__ void CycleRoutineGPU(char *CurrentState , char *NextState , int X , int Dimension){

int sum=0;
int id=blockIdx.x*blockDim.x + threadIdx.x;

if (id<Dimension) {

if(id+X<Dimension ) {
sum += CurrentState[id+X];
}
if(id-X>=0){
sum += CurrentState[id-X];
}
if(id/X == (id+1)/X) {
sum += CurrentState[id+1];
}
if(id/X == (id-1)/X) {
sum += CurrentState[id-1];
}
if(id+X<Dimension && (id+X)/X == (id+X+1)/X) {
sum += CurrentState[id+X+1];
}
if(id+X<Dimension && (id+X)/X == (id+X-1)/X) {
sum += CurrentState[id+X-1];
}
if(id-X>=0 && (id-X)/X == (id-X+1)/X) {
sum += CurrentState[id-X+1];
}
if(id-X>=0 && (id-X)/X == (id-X-1)/X) {
sum += CurrentState[id-X-1];
}


if (sum < 2 || sum > 3)
NextState[id] = 0;
else if (sum == 3)
NextState[id] =  1;
else
NextState[id] = CurrentState[id];

}

__syncthreads();
}