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
__global__ void create_bin_expimages(char* acExpDetImages, const int* aiDetStartIdx, const float* afDetInfo,const int iNDet, const int iNRot, const int* aiDetIndex, const int* aiRotN, const int* aiJExp,const int* aiKExp, int const iNPeak){
/*
* create the image matrix
* acExpDetImages: Sigma_i(iNDet*iNRot*iNJ[i]*iNK[i]) , i for each detector, detectors may have different size
* aiDetStartIdx:   index of Detctor start postition in self.acExpDetImages,
* 					e.g. 3 detectors with size 2048x2048, 180 rotations,
* 			 		aiDetStartIdx = [0,180*2048*2048,2*180*2048*2048]
* afDetInfo: iNDet*19, detector information
* iNDet: number of detectors, e.g. 2 or 3;
* iNRot: number of rotations, e.g. 180,720;
* aiDetIndex: len=iNPeak the index of detector, e.g. 0,1 or 2
* aiRotN: aiJExp: aiKExp: len=iNPeak
* iNPeak number of diffraction peaks
* test ?
*/
int i = blockIdx.x*blockDim.x+threadIdx.x;
if(i<iNPeak){
acExpDetImages[aiDetStartIdx[aiDetIndex[i]]
+ aiRotN[i]*int(afDetInfo[0+19*aiDetIndex[i]])*int(afDetInfo[1+19*aiDetIndex[i]])
+ aiKExp[i]*int(afDetInfo[0+19*aiDetIndex[i]]) + aiJExp[i]] = 1;
}
}