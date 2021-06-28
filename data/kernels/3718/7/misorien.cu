#include "includes.h"
const float PI = 3.14159265359;
const float HALFPI = 0.5*PI;
texture<unsigned char, cudaTextureType3D, cudaReadModeElementType> tcExpData;
texture<float, cudaTextureType2D, cudaReadModeElementType> tfG;  // texture to store scattering vectors;
typedef struct {
int iNPixelJ, iNPixelK;
float fPixelJ, fPixelK;
float afCoordOrigin[3];
float afNorm[3];
float afJVector[3];
float afKVector[3];
float fNRot, fAngleStart,fAngleEnd;


} DetInfo;
__device__ void mat3_dot(float* afResult, float* afM0, float* afM1){
/*
* dot product of two 3x3 matrix
*/
for(int i=0;i<3;i++){
for(int j=0;j<3;j++){
afResult[i * 3 + j] = 0;
for(int k=0;k<3;k++){
afResult[i * 3 + j] += afM0[i * 3 + k] * afM1[k * 3 + j];
}
}
}
}
__device__ void mat3_transpose(float* afOut, float* afIn){
/*
* transpose 3x3 matrix
*/
for(int i=0;i<3;i++){
for(int j=0;j<3;j++){
afOut[i * 3 + j] = afIn[j * 3 + i];
}
}
}
__global__ void misorien(float* afMisOrien, float* afM0, float* afM1, float* afSymM){
/*
* calculate the misorientation betwen afM0 and afM1
* afMisOrien: iNM * iNSymM
* afM0: iNM * 9
* afM1: iNM * 9
* afSymM: symmetry matrix, iNSymM * 9
* NSymM: number of symmetry matrix
* call method: <<<(iNM,1),(iNSymM,1,1)>>>
*/
int i = blockIdx.x*blockDim.x + threadIdx.x;
float afTmp0[9];
float afTmp1[9];
float afM1Transpose[9];
float fCosAngle;
mat3_transpose(afM1Transpose, afM1 + blockIdx.x * 9);
mat3_dot(afTmp0, afSymM + threadIdx.x * 9, afM1Transpose);
mat3_dot(afTmp1, afM0 + blockIdx.x * 9, afTmp0);
fCosAngle = 0.5 * (afTmp1[0] + afTmp1[4] + afTmp1[8] - 1);
fCosAngle = min(0.9999999999, fCosAngle);
fCosAngle = max(-0.99999999999, fCosAngle);
afMisOrien[i] = acosf(fCosAngle);
}