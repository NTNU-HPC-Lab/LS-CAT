#include "includes.h"
/***
* File: maxwell_griffin_lab4p2.cu
* Desc: Performs 2 Sobel edge detection operations on a .bmp, once by a
*       serial algorithm, and once by a massively parallel CUDA algorithm.
*/


extern "C"
{
}

#define PIXEL_BLACK (0)
#define PIXEL_WHITE (255)
#define PERCENT_BLACK_THRESHOLD (0.75)

#define CUDA_GRIDS (1)
#define CUDA_BLOCKS_PER_GRID (32)
#define CUDA_THREADS_PER_BLOCK (128)

#define MS_PER_SEC (1000)
#define NS_PER_MS (1000 * 1000)
#define NS_PER_SEC (NS_PER_MS * MS_PER_SEC)

#define LINEARIZE(row, col, dim) \
(((row) * (dim)) + (col))

static struct timespec rtcSerialStart;
static struct timespec rtcSerialEnd;
static struct timespec rtcParallelStart;
static struct timespec rtcParallelEnd;

__device__ int Sobel_Gx[3][3] = {
{ -1, 0, 1 },
{ -2, 0, 2 },
{ -1, 0, 1 }
};

__device__ int Sobel_Gy[3][3] = {
{  1,  2,  1 },
{  0,  0,  0 },
{ -1, -2, -1 }
};

/*
* Display all header information and matrix and CUDA parameters.
*
* @param inputFile -- name of the input image
* @param serialOutputFile -- name of the serial output image
* @param parallelOutputFile -- name of the parallel output image
* @param imageHeight -- in pixels
* @param imageWidth -- in pixels
*/
void DisplayParameters(
char *inputFile,
char *serialOutputFile,
char *cudaOutputFile,
int imageHeight,
int imageWidth)
{
printf("********************************************************************************\n");
printf("lab4p2: serial vs. CUDA Sobel edge detection.\n");
printf("\n");
printf("Input image: %s \t(Height: %d pixels, width: %d pixels)\n", inputFile, imageHeight, imageWidth);
printf("Serial output image: \t%s\n", serialOutputFile);
printf("CUDA output image: \t%s\n", cudaOutputFile);
printf("\n");
printf("CUDA compute structure:\n");
printf("|-- with %d grid\n", CUDA_GRIDS);
printf("    |-- with %d blocks\n", CUDA_BLOCKS_PER_GRID);
printf("        |-- with %d threads per block\n", CUDA_THREADS_PER_BLOCK);
printf("\n");
}

/*
* Display the timing and convergence results to the screen.
*
* @param serialConvergenceThreshold
* @param serialConvergenceThreshold
*/
void DisplayResults(
int serialConvergenceThreshold,
int parallelConvergenceThreshold)
{
printf("Time taken for serial Sobel edge detection: %lf\n",
(LINEARIZE(rtcSerialEnd.tv_sec, rtcSerialEnd.tv_nsec, NS_PER_SEC)
- LINEARIZE(rtcSerialStart.tv_sec, rtcSerialStart.tv_nsec, NS_PER_SEC))
/ ((double)NS_PER_SEC));

printf("Convergence Threshold: %d\n", serialConvergenceThreshold);
printf("\n");

printf("Time taken for CUDA Sobel edge detection: %lf\n",
(LINEARIZE(rtcParallelEnd.tv_sec, rtcParallelEnd.tv_nsec, NS_PER_SEC)
- LINEARIZE(rtcParallelStart.tv_sec, rtcParallelStart.tv_nsec, NS_PER_SEC))
/ ((double)NS_PER_SEC));

printf("Convergence Threshold: %d\n", parallelConvergenceThreshold);
printf("********************************************************************************\n");
}

/*
* Serial algorithm to keep perform a Sobel edge detection on an input pixel
* buffer at different brightness thresholds until a certain percentage of
* pixels in the output pixel buffer are black.
*
* @param input -- input pixel buffer
* @param output -- output pixel buffer
* @param height -- height of pixel image
* @param width -- width of pixel image
* @return -- gradient threshold at which PERCENT_BLACK_THRESHOLD pixels are black
*/
__global__ void CudaSobelEdgeDetection(uint8_t *input, uint8_t *output, int height, int width, int gradientThreshold)
{
int row = 0;
for(int i = 0; row < (height - 1); i++)
{
// Let the blockIdx increment beyond its dimension for cyclic distribution of the test pixels
int blockRow = (i * gridDim.x) + blockIdx.x;

// Calculate the row/col in the image buffer that this thread's stencil's center is on
row = (LINEARIZE(blockRow, threadIdx.x, blockDim.x) / (width - 2)) + 1;
int col = (LINEARIZE(blockRow, threadIdx.x, blockDim.x) % (width - 2)) + 1;

// Calculate Sobel magnitude of gradient directly, instead of using Sobel_Magnitude utility
double Gx = (Sobel_Gx[0][0] * input[LINEARIZE(row - 1, col - 1, width)])
+ (Sobel_Gx[0][2] * input[LINEARIZE(row - 1, col + 1, width)])
+ (Sobel_Gx[1][0] * input[LINEARIZE(row, col - 1, width)])
+ (Sobel_Gx[1][2] * input[LINEARIZE(row, col + 1, width)])
+ (Sobel_Gx[2][0] * input[LINEARIZE(row + 1, col - 1, width)])
+ (Sobel_Gx[2][2] * input[LINEARIZE(row + 1, col + 1, width)]);

double Gy = (Sobel_Gy[0][0] * input[LINEARIZE(row - 1, col - 1, width)])
+ (Sobel_Gy[0][1] * input[LINEARIZE(row - 1, col, width)])
+ (Sobel_Gy[0][2] * input[LINEARIZE(row - 1, col + 1, width)])
+ (Sobel_Gy[2][0] * input[LINEARIZE(row + 1, col - 1, width)])
+ (Sobel_Gy[2][1] * input[LINEARIZE(row + 1, col, width)])
+ (Sobel_Gy[2][2] * input[LINEARIZE(row + 1, col + 1, width)]);

if(((Gx * Gx) + (Gy * Gy)) > (gradientThreshold * gradientThreshold))
{
output[LINEARIZE(row, col, width)] = PIXEL_WHITE;
}
else
{
output[LINEARIZE(row, col, width)] = PIXEL_BLACK;
}
}
}