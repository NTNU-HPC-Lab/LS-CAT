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
__global__ void mat_to_euler_ZXZ(float* afMatIn, float* afEulerOut, int iNAngle){
/*
* transform active rotation matrix to euler angles in ZXZ convention, not right(seems right now)
* afMatIn: iNAngle * 9
* afEulerOut: iNAngle* 3
* TEST PASSED
*/
float threshold = 0.9999999;
int i = blockIdx.x*blockDim.x + threadIdx.x;
if(i<iNAngle){
if(afMatIn[i * 9 + 8] > threshold){
afEulerOut[i * 3 + 0] = 0;
afEulerOut[i * 3 + 1] = 0;
afEulerOut[i * 3 + 2] = atan2(afMatIn[i*9 + 3], afMatIn[i*9 + 0]);           //  atan2(m[1, 0], m[0, 0])
}
else if(afMatIn[i * 9 + 8] < - threshold){
afEulerOut[i * 3 + 0] = 0;
afEulerOut[i * 3 + 1] = PI;
afEulerOut[i * 3 + 2] = atan2(afMatIn[i*9 + 1], afMatIn[i*9 + 0]);           //  atan2(m[0, 1], m[0, 0])
}
else{
afEulerOut[i * 3 + 0] = atan2(afMatIn[i*9 + 2], - afMatIn[i*9 + 5]);          //  atan2(m[0, 2], -m[1, 2])
afEulerOut[i * 3 + 1] = atan2( sqrt(afMatIn[i*9 + 6] * afMatIn[i*9 + 6]
+ afMatIn[i*9 + 7] * afMatIn[i*9 + 7]),
afMatIn[i*9 + 8]);                             //     atan2(np.sqrt(m[2, 0] ** 2 + m[2, 1] ** 2), m[2, 2])
afEulerOut[i * 3 + 2] = atan2( afMatIn[i*9 + 6], afMatIn[i*9 + 7]);           //   atan2(m[2, 0], m[2, 1])
if(afEulerOut[i * 3 + 0] < 0){
afEulerOut[i * 3 + 0] += 2 * PI;
}
if(afEulerOut[i * 3 + 1] < 0){
afEulerOut[i * 3 + 1] += 2 * PI;
}
if(afEulerOut[i * 3 + 2] < 0){
afEulerOut[i * 3 + 2] += 2 * PI;
}
}
}
}