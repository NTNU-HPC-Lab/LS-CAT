#define MAX_KERNEL_LENGTH 80
 __constant__ float c_Kernel[MAX_KERNEL_LENGTH * 4];
#define   LAYERS_BLOCKDIM_X 8
#define   LAYERS_BLOCKDIM_Y 8
#define   LAYERS_BLOCKDIM_Z 8
#define   LAYERS_RESULT_STEPS 4
#define   LAYERS_HALO_STEPS 2
//new series 
