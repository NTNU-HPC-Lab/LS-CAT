#ifndef NPMMV_DENSE_KERNEL_H_
#define NPMMV_DENSE_KERNEL_H_

#ifdef __cplusplus
extern "C" { 
#endif
    void internal_negative_prob_multiply_dense_matrix_vector_gpu(float* matrix, float* in_vector, float* out_vector, unsigned int outerdim, unsigned int innerdim);
#ifdef __cplusplus
}
#endif
#endif
