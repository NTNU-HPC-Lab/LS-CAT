#ifndef NPMMV_CSR_VECTOR_KERNEL_H_
#define NPMMV_CSR_VECTOR_KERNEL_H_

#ifdef __cplusplus
extern "C" { 
#endif
    void internal_spmv_csr_veck_gpu(unsigned int computation_restriction_factor, unsigned int* cum_row_indexes, 
            unsigned int* column_indexes, float* matrix_data, float* in_vector, float* out_vector, 
            unsigned int outerdim);
#ifdef __cplusplus
}
#endif
#endif
