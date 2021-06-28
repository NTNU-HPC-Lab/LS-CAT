#include "includes.h"
__global__ void cudaSBilinearTF_BackWard_kernel( unsigned int outputWidth, unsigned int outputHeight, unsigned int nbChannels, unsigned int batchSize, unsigned int inputWidth, unsigned int inputHeight, const float scaleX, const float scaleY, const float* diffInput, float* diffOutputs)
{

const unsigned int inputOffset
= (blockIdx.z * blockDim.z + threadIdx.z) * nbChannels*inputWidth*inputHeight;

const unsigned int outputOffset
= (blockIdx.z * blockDim.z + threadIdx.z) * nbChannels*outputWidth*outputHeight;
for (unsigned int ch = blockIdx.x; ch < nbChannels; ch += gridDim.x)
{
for (unsigned int oy = threadIdx.y; oy < outputHeight; oy += blockDim.y)
{
const float in_y = oy * scaleY;
const int top_y_index = (int)(floorf(in_y));
//const int bottom_y_index = min((int)(ceilf(in_y)), (int) (inputHeight - 1) ) ;

const int bottom_y_index = (in_y < inputHeight - 1) ? ceilf(in_y) : inputHeight - 1;

const float y_lerp = in_y - top_y_index;
const float inverse_y_lerp = (1.0f - y_lerp);


for (unsigned int ox = threadIdx.x; ox < outputWidth; ox += blockDim.x)
{
const float in_x = ox * scaleX;
const int left_x_index = (int)(floorf(in_x));
//const int right_x_index = min((int)(ceilf(in_x)), (int)(inputWidth - 1));
const int right_x_index = (in_x < inputWidth - 1) ? ceilf(in_x) : inputWidth - 1;


const float x_lerp = in_x - left_x_index;
const float inverse_x_lerp = (1.0f - x_lerp);

const unsigned int inLeftTopIdx = left_x_index + top_y_index*inputWidth + ch*inputWidth*inputHeight + inputOffset;
const unsigned int inRightTopIdx = right_x_index + top_y_index*inputWidth + ch*inputWidth*inputHeight + inputOffset;
const unsigned int inLeftBotIdx = left_x_index + bottom_y_index*inputWidth + ch*inputWidth*inputHeight + inputOffset;
const unsigned int inRightBotIdx = right_x_index + bottom_y_index*inputWidth + ch*inputWidth*inputHeight + inputOffset;

const unsigned int outIdx = ox + oy*outputWidth + ch*outputWidth*outputHeight + outputOffset;
const float outData = diffInput[outIdx];

diffOutputs[inLeftTopIdx]  += outData * inverse_y_lerp * inverse_x_lerp ;
diffOutputs[inRightTopIdx]  += outData * inverse_y_lerp * x_lerp ;
diffOutputs[inLeftBotIdx]  += outData * y_lerp * inverse_x_lerp ;
diffOutputs[inRightBotIdx]  += outData * y_lerp * x_lerp ;

}
}
}
}