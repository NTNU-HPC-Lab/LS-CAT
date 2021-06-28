#ifndef GRAPH_DETERM_WEIGHTS_H_
#define GRAPH_DETERM_WEIGHTS_H_

#ifdef __cplusplus
extern "C" { 
#endif
    void internal_graph_determ_weights(unsigned int* contact_mat_cum_row_indexes, unsigned int* contact_mat_column_indexes, 
            float* contact_mat_values, unsigned int rows, unsigned int values, float* immunities, float* shedding_curve, 
            unsigned int infection_length, float transmission_rate, int* infection_mat_values);
#ifdef __cplusplus
}
#endif
#endif
