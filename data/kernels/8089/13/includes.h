#define RADIUS 4
#define NUM_SCALES      5
#define CONVCOL_W      32
#define CONVCOL_H      40
#define CONVCOL_S       8
 __device__ __constant__ float d_Kernel[12*16]; // NOTE: Maximum radius
//new series 
