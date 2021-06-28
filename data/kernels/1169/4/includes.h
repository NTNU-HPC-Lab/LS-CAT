#define MAX_KERNEL_LENGTH 80
 __constant__ float c_Kernel[MAX_KERNEL_LENGTH * 4];
#define   COLUMNS_BLOCKDIM_X 4
#define   COLUMNS_BLOCKDIM_Y 16
#define   COLUMNS_BLOCKDIM_Z 4
#define   COLUMNS_RESULT_STEPS 9
#define   COLUMNS_HALO_STEPS 2
//new series 
