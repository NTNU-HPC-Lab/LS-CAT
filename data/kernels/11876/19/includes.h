#define REDUC_THREADS 256
 __constant__ int d_n_inputs_cols ; // Ditto, extended to multiple of 128 bytes
 __constant__ int d_nhid ;          // Number of hidden neurons
 __constant__ float *d_w_grad ;
 __constant__ float *d_prev_grad ;
 __constant__ float *d_len_out ;
 __constant__ float *d_dot_out ;
//new series 
