//define statements

#define A h_array[(i) * nCols + (j-1)]
#define B h_array[(i-1) * nCols + (j-1)]
#define C h_array[(i-1) * nCols + (j)]
#define D h_array[(i-1) * nCols + (j+1)]
#define Z h_array[(i) * nCols + (j)]

#define d_A d_array[(i) * nCols + (j-1)]
#define d_B d_array[(i-1) * nCols + (j-1)]
#define d_C d_array[(i-1) * nCols + (j)]
#define d_D d_array[(i-1) * nCols + (j+1)]
#define d_Z d_array[(i) * nCols + (j)]



//CPU declarations
void update_array_cpu(int i);

//GPU declarations
__global__ void update_array_gpu(int, int, int *d_array);

//others
void configure_kernal(long);
