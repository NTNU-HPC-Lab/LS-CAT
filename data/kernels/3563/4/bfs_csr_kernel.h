#ifndef BFS_CSR_KERNEL_H_
#define BFS_CSR_KERNEL_H_

#ifdef __cplusplus
extern "C" {
#endif
    void internal_breadth_first_search_csr_gpu(unsigned int* cum_row_indexes, unsigned int* column_indexes, 
                                            int* matrix_data, unsigned int* in_infections, 
                                            unsigned int* out_infections, unsigned int rows);
#ifdef __cplusplus
}
#endif
#endif
