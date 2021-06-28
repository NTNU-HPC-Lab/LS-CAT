#include "includes.h"
extern "C"

extern "C"
__global__ void deltasBatch(float *inputs, float *outputs, float *weights, float *weightsDeltas, int noInputs, int inputSize){
int gid = blockIdx.x * blockDim.x + threadIdx.x;
float sum=0;
int offsetDeltas = (inputSize+1)*gid;
int offsetInput = noInputs*inputSize*gid;
int offsetOutputs = noInputs*gid;

for(int imageIndex=0;imageIndex<=inputSize;imageIndex++){
weightsDeltas[offsetDeltas+imageIndex]=0;
}

for (int i=0;i<noInputs;i++){
sum=0;
for(int imageIndex=0;imageIndex<inputSize;imageIndex++){
sum+=inputs[offsetInput+i*inputSize+imageIndex]*weights[imageIndex];
}
sum+=weights[inputSize];
if(sum>0) sum=1;
else sum=0;
sum=outputs[offsetOutputs+i]-sum;
if(sum!=0){
for(int imageIndex=0;imageIndex<inputSize;imageIndex++){
weightsDeltas[offsetDeltas+imageIndex]+=sum*inputs[offsetInput+i*inputSize+imageIndex];
}
weightsDeltas[offsetDeltas+inputSize]+=sum;
}
}
}