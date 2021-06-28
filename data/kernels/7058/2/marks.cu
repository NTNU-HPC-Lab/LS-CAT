#include "includes.h"
__global__ void marks(float * media, int * final){
int thread = blockIdx.x*blockDim.x + threadIdx.x;
final[thread] =	(media[thread] == (int)media[thread]) * (int)media[thread] +
(media[thread] != (int)media[thread] && media[thread] > 4 && media[thread] < 5)* 4 +
(media[thread] != (int)media[thread] && media[thread] > 9)* 9 +
(media[thread] != (int)media[thread] && (media[thread] < 4 || (media[thread] > 5 && media[thread] < 9))) * ((int)media[thread] + 1);
}