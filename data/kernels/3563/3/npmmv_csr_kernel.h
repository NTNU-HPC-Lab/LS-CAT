#ifndef NPMMV_CSR_KERNEL_H_
#define NPMMV_CSR_KERNEL_H_

#ifdef __cplusplus
extern "C" { 
#endif
    void internal_negative_prob_multiply_csr_matrix_vector_gpu(unsigned int* cum_row_indexes, unsigned int* column_indexes, float* matrix_data, float* in_vector, float* out_vector, unsigned int outerdim);
#ifdef __cplusplus
}
#endif
#endif
