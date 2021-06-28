#include "includes.h"
extern "C"

extern "C"
__global__ void deltasOne(float *inputs, float *outputs, float *weights, float *weightsDeltas, int offsetInputImages, int inputSize){
int gid = blockIdx.x * blockDim.x + threadIdx.x;
float sum=0;
int offsetDeltas = (inputSize+1)*gid;
int offsetInput = inputSize*(gid+offsetInputImages);

for(int imageIndex=0;imageIndex<=inputSize;imageIndex++){
weightsDeltas[offsetDeltas+imageIndex]=0;
}

for(int imageIndex=0;imageIndex<inputSize;imageIndex++){
sum+=inputs[offsetInput+imageIndex]*weights[imageIndex];
}
sum+=weights[inputSize];
if(sum>0) sum=1;
else sum=0;
sum=outputs[offsetInputImages+gid]-sum;
if(sum!=0){
for(int imageIndex=0;imageIndex<inputSize;imageIndex++){
weightsDeltas[offsetDeltas+imageIndex]+=sum*inputs[offsetInput+imageIndex];
}
weightsDeltas[offsetDeltas+inputSize]+=sum;
}

}