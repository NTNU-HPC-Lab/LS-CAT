#include "includes.h"
__device__ void gpujpeg_idct_gpu_kernel_inplace(float* V8)
{
//costants which are used more than once
const float koeficient[6] = {0.4142135623f, 0.3535533905f, 0.4619397662f, 0.1989123673f, 0.7071067811f, -2.0f};

V8[2] *= 0.5411961f;
V8[4] *= 0.509795579f;
V8[5] *= 0.601344887f;

V8[1] = (V8[0] - V8[1]) * koeficient[1];
V8[0] = V8[0] * koeficient[4] - V8[1];

V8[3] = V8[2] * koeficient[1] + V8[3] * koeficient[2];
V8[2] = V8[3] * koeficient[0] - V8[2];

V8[6] = V8[5] * koeficient[2] + V8[6] * koeficient[0];
V8[5] = -0.6681786379f * V8[6] + V8[5];

V8[7] = V8[4] * koeficient[3] + V8[7] * 0.49039264f;
V8[4] = V8[7] * koeficient[3] - V8[4];

//instead of float tmp = V8[1]; V8[1] = V8[2] + V8[1]; V8[2] = tmp - V8[2];
//we use this two operations (with a use of a multiply-add instruction)
V8[1] = V8[2] + V8[1];
V8[2] = koeficient[5] * V8[2] + V8[1];

V8[4] = V8[5] + V8[4];
V8[5] = 2.0f * V8[5] - V8[4];

V8[7] = V8[6] + V8[7];
V8[6] = koeficient[5] * V8[6] + V8[7];

V8[0] = V8[3] + V8[0];
V8[3] = koeficient[5] * V8[3] + V8[0];

V8[5] = V8[6] * koeficient[0] + V8[5];
V8[6] = V8[5] * -koeficient[4] + V8[6];
V8[5] = V8[6] * koeficient[0] + V8[5];

V8[3] = V8[3] + V8[4];
V8[4] = koeficient[5] * V8[4] + V8[3];

V8[2] = V8[2] + V8[5];
V8[5] = koeficient[5] * V8[5] + V8[2];

V8[1] = V8[6] + V8[1];
V8[6] = koeficient[5] * V8[6] + V8[1];

V8[0] = V8[0] + V8[7];
V8[7] = koeficient[5] * V8[7] + V8[0];
}
__global__ void gpujpeg_idct_gpu_kernel(int16_t* source, uint8_t* result, int output_stride, uint16_t* quantization_table)
{
//here the grid is assumed to be only in x - it saves a few operations; if a larger
//block count is used (e. g. GPUJPEG_IDCT_BLOCK_Z == 1), it would need to be adjusted,
//the blockIdx.x not to exceed 65535. In the current state this function is good
//enough for a 67.1 MPix picture (8K is 33.1 MPix)

//the first block of picture processed in this thread block
unsigned int picBlockNumber = (blockIdx.x) * GPUJPEG_IDCT_BLOCK_Y * GPUJPEG_IDCT_BLOCK_X
* GPUJPEG_IDCT_BLOCK_Z;

//pointer to the begin of data for this thread block
int16_t* sourcePtr = (int16_t*) (source) + picBlockNumber * 8;

__shared__ float data[GPUJPEG_IDCT_BLOCK_Z][8][GPUJPEG_IDCT_BLOCK_Y][GPUJPEG_IDCT_BLOCK_X + 1];

//variables to be used later more times (only one multiplication here)
unsigned int z64 = threadIdx.z * 64;
unsigned int x8 = threadIdx.x * 8;

//data copying global -> shared, type casting int16_t -> float and dequantization.
//16b reading gives only 50% efectivity but another ways are too complicated
//so this proves to be the fastest way
#pragma unroll
for (int i = 0; i < 8; i++) {
data[threadIdx.z][i][threadIdx.x][threadIdx.y] = sourcePtr[x8
+ threadIdx.y + i * GPUJPEG_IDCT_BLOCK_X * GPUJPEG_IDCT_BLOCK_Y + z64 * 8]
* quantization_table[threadIdx.x * 8 + threadIdx.y];
}

__syncthreads();

float x[8];

//kompilator delal hrozne psi kusy - zbytecne kopirovani konstant do
//registru atp., bylo jednodussi napsat to v assembleru nez snazit se ho
//presvedcit, aby nedelal blbosti; vsechny konstanty se pouzivaji primo
//hodnotou, nestrkaji se zbytecne do registru

//here the data are being processed by columns - each thread processes one column
#if GPUJPEG_IDCT_USE_ASM
GPUJPEG_IDCT_GPU_KERNEL_INPLACE(data[threadIdx.z][threadIdx.x][0][threadIdx.y],
data[threadIdx.z][threadIdx.x][4][threadIdx.y],
data[threadIdx.z][threadIdx.x][6][threadIdx.y],
data[threadIdx.z][threadIdx.x][2][threadIdx.y],
data[threadIdx.z][threadIdx.x][7][threadIdx.y],
data[threadIdx.z][threadIdx.x][5][threadIdx.y],
data[threadIdx.z][threadIdx.x][3][threadIdx.y],
data[threadIdx.z][threadIdx.x][1][threadIdx.y],

data[threadIdx.z][threadIdx.x][0][threadIdx.y],
data[threadIdx.z][threadIdx.x][1][threadIdx.y],
data[threadIdx.z][threadIdx.x][2][threadIdx.y],
data[threadIdx.z][threadIdx.x][3][threadIdx.y],
data[threadIdx.z][threadIdx.x][4][threadIdx.y],
data[threadIdx.z][threadIdx.x][5][threadIdx.y],
data[threadIdx.z][threadIdx.x][6][threadIdx.y],
data[threadIdx.z][threadIdx.x][7][threadIdx.y])
#else
x[0] = data[threadIdx.z][threadIdx.x][0][threadIdx.y];
x[1] = data[threadIdx.z][threadIdx.x][4][threadIdx.y];
x[2] = data[threadIdx.z][threadIdx.x][6][threadIdx.y];
x[3] = data[threadIdx.z][threadIdx.x][2][threadIdx.y];
x[4] = data[threadIdx.z][threadIdx.x][7][threadIdx.y];
x[5] = data[threadIdx.z][threadIdx.x][5][threadIdx.y];
x[6] = data[threadIdx.z][threadIdx.x][3][threadIdx.y];
x[7] = data[threadIdx.z][threadIdx.x][1][threadIdx.y];

gpujpeg_idct_gpu_kernel_inplace(x);

data[threadIdx.z][threadIdx.x][0][threadIdx.y] = x[0];
data[threadIdx.z][threadIdx.x][1][threadIdx.y] = x[1];
data[threadIdx.z][threadIdx.x][2][threadIdx.y] = x[2];
data[threadIdx.z][threadIdx.x][3][threadIdx.y] = x[3];
data[threadIdx.z][threadIdx.x][4][threadIdx.y] = x[4];
data[threadIdx.z][threadIdx.x][5][threadIdx.y] = x[5];
data[threadIdx.z][threadIdx.x][6][threadIdx.y] = x[6];
data[threadIdx.z][threadIdx.x][7][threadIdx.y] = x[7];
#endif
//between data writing and sync it's good to compute something useful
// - the sync will be shorter.

//output pointer (the begin for this thread block)
unsigned int firstByteOfActualBlock = x8 + z64 + picBlockNumber;

//output pointer for this thread + output row shift; each thread writes 1 row of an
//output block (8B), threads [0 - 7] in threadIdx.x write blocks next to each other,
//threads [1 - 7] in threadIdx.y write next rows of a block; threads [0 - 1] in
//threadIdx.z write next 8 blocks
uint8_t* resultPtr = ((uint8_t*) result) + firstByteOfActualBlock
+ (threadIdx.y + ((firstByteOfActualBlock / output_stride) * 7))
* output_stride;

__syncthreads();

#if GPUJPEG_IDCT_USE_ASM
//here the data are being processed by rows - each thread processes one row
GPUJPEG_IDCT_GPU_KERNEL_INPLACE(data[threadIdx.z][threadIdx.x][threadIdx.y][0],
data[threadIdx.z][threadIdx.x][threadIdx.y][4],
data[threadIdx.z][threadIdx.x][threadIdx.y][6],
data[threadIdx.z][threadIdx.x][threadIdx.y][2],
data[threadIdx.z][threadIdx.x][threadIdx.y][7],
data[threadIdx.z][threadIdx.x][threadIdx.y][5],
data[threadIdx.z][threadIdx.x][threadIdx.y][3],
data[threadIdx.z][threadIdx.x][threadIdx.y][1],

x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7])
#else
x[0] = data[threadIdx.z][threadIdx.x][threadIdx.y][0];
x[1] = data[threadIdx.z][threadIdx.x][threadIdx.y][4];
x[2] = data[threadIdx.z][threadIdx.x][threadIdx.y][6];
x[3] = data[threadIdx.z][threadIdx.x][threadIdx.y][2];
x[4] = data[threadIdx.z][threadIdx.x][threadIdx.y][7];
x[5] = data[threadIdx.z][threadIdx.x][threadIdx.y][5];
x[6] = data[threadIdx.z][threadIdx.x][threadIdx.y][3];
x[7] = data[threadIdx.z][threadIdx.x][threadIdx.y][1];

gpujpeg_idct_gpu_kernel_inplace(x);
#endif

//output will be written by 8B (one row) which is the most effective way
uint64_t tempResult;
uint64_t* tempResultP = &tempResult;

#pragma unroll
for (int i = 0; i < 8; i++) {
//this would be faster but will work only for 100% quality otherwise some values overflow 255
//((uint8_t*) tempResultP)[i] = __float2uint_rz(x[i] + ((float) 128.0));

//cast float to uint8_t with saturation (.sat) which cuts values higher than
//255 to 255 and smaller than 0 to 0; cuda can't use a reg smaller than 32b
//(though it can convert to 8b for the saturation purposes and save to 32b reg)
uint32_t save;
asm("cvt.rni.u8.f32.sat	%0, %1;" : "=r"(save) : "f"(x[i] + ((float) 128.0)));
((uint8_t*) tempResultP)[i] = save;
}

//writing result - one row of a picture block by a thread
*((uint64_t*) resultPtr) = tempResult;
}