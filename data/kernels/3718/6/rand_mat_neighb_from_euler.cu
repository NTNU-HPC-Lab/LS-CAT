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
__device__ void d_euler_zxz_to_mat(float* afEuler, float* afMat){
float s1 = sin(afEuler[0]);
float s2 = sin(afEuler[1]);
float s3 = sin(afEuler[2]);
float c1 = cos(afEuler[0]);
float c2 = cos(afEuler[1]);
float c3 = cos(afEuler[2]);
afMat[0] = c1 * c3 - c2 * s1 * s3;
afMat[1] = -c1 * s3 - c3 * c2 * s1;
afMat[2] = s1 * s2;
afMat[3] = s1 * c3 + c2 * c1 * s3;
afMat[4] = c1 * c2 * c3 - s1 * s3;
afMat[5] = -c1 * s2;
afMat[6] = s3 * s2;
afMat[7] = s2 * c3;
afMat[8] = c2;
}
__global__ void rand_mat_neighb_from_euler(float* afEulerIn, float* afMatOut, float* afRand, float fBound){
/* generate random matrix according to the input EulerAngle
* afEulerIn: iNEulerIn * 3, !!!!!!!!!! in radian  !!!!!!!!
* afMatOut: iNNeighbour * iNEulerIn * 9
* afRand:   iNNeighbour * iNEulerIn * 3
* fBound: the range for random angle [-fBound,+fBound]
* iNEulerIn: number of Input Euler angles
* iNNeighbour: number of random angle generated for EACH input
* call:: <<(iNNeighbour,1),(iNEulerIn,1,1)>>
* TEST PASSED
*/
//printf("%f||",fBound);
// keep the original input
float afEulerTmp[3];

afEulerTmp[0] = afEulerIn[threadIdx.x * 3 + 0] + (2 * afRand[blockIdx.x * blockDim.x * 3 + threadIdx.x * 3 + 0] - 1) * fBound;
afEulerTmp[2] = afEulerIn[threadIdx.x * 3 + 2] + (2 * afRand[blockIdx.x * blockDim.x * 3 + threadIdx.x * 3 + 2] - 1) * fBound;
float z = cos(afEulerIn[threadIdx.x * 3 + 1]) +
(afRand[blockIdx.x * blockDim.x * 3 + threadIdx.x * 3 + 1] * 2 - 1) * sin(afEulerIn[threadIdx.x * 3 + 1] * fBound);
if(z>1){
z = 1;
}
else if(z<-1){
z = -1;
}
afEulerTmp[1] = acosf(z);

if(blockIdx.x>0){
d_euler_zxz_to_mat(afEulerTmp, afMatOut + blockIdx.x * blockDim.x * 9 + threadIdx.x * 9);
}
else{
// keep the original input
d_euler_zxz_to_mat(afEulerIn + threadIdx.x * 3, afMatOut + blockIdx.x * blockDim.x * 9 + threadIdx.x * 9);
}
}