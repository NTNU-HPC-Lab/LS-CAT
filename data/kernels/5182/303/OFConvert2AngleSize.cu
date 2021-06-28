#include "includes.h"
__device__ void OFConvertXY2AngleSize (float*of, int id, int imageSize, float& of_size, float& of_angle){
float2 OF_value;

OF_value.x = of[id];
OF_value.y = of[id+imageSize];

of_size  = (float) sqrt( (OF_value.x+OF_value.y) * (OF_value.x+OF_value.y) );  // normalized to be <0,1>
of_angle = (float) atan2(OF_value.x,OF_value.y);  // <-PI;PI>
}
__global__ void OFConvert2AngleSize (float*of, int imageSize){
int id = blockDim.x*blockIdx.y*gridDim.x
+ blockDim.x*blockIdx.x
+ threadIdx.x;

float OF_size;
float OF_angle;

if (id<imageSize){
OFConvertXY2AngleSize(of,id,imageSize,OF_size,OF_angle);

of[id] = OF_angle;
of[id+imageSize] = OF_size;
}
}