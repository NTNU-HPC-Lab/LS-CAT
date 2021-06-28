#define REDUC_THREADS 256
 __constant__ int d_ncases ;        // Number of cases (needed for using shuffle_index as random sampler)
 __constant__ int d_ntarg ;                 // Number of targets (output neurons)
 __constant__ float *d_targets ;
 __constant__ double *d_output ;
 __constant__ float *d_mse_out ;
//new series 
