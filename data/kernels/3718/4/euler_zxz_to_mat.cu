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
__global__ void euler_zxz_to_mat(float* afEuler, float* afMat, int iNAngle){
int i = blockIdx.x*blockDim.x + threadIdx.x;
if(i<iNAngle){
float s1 = sin(afEuler[i * 3 + 0]);
float s2 = sin(afEuler[i * 3 + 1]);
float s3 = sin(afEuler[i * 3 + 2]);
float c1 = cos(afEuler[i * 3 + 0]);
float c2 = cos(afEuler[i * 3 + 1]);
float c3 = cos(afEuler[i * 3 + 2]);
afMat[i * 9 + 0] = c1 * c3 - c2 * s1 * s3;
afMat[i * 9 + 1] = -c1 * s3 - c3 * c2 * s1;
afMat[i * 9 + 2] = s1 * s2;
afMat[i * 9 + 3] = s1 * c3 + c2 * c1 * s3;
afMat[i * 9 + 4] = c1 * c2 * c3 - s1 * s3;
afMat[i * 9 + 5] = -c1 * s2;
afMat[i * 9 + 6] = s3 * s2;
afMat[i * 9 + 7] = s2 * c3;
afMat[i * 9 + 8] = c2;
}
}