 __constant__ int d_n_inputs ;      // Number of inputs (size of visible, bottom layer)
 __constant__ int d_n_inputs_cols ; // Ditto, extended to multiple of 128 bytes
 __constant__ int d_nhid_cols ;     // Ditto, extended to multiple of 128 bytes
 __constant__ float *d_w ;
 __constant__ float *d_wtr ;
//new series 
