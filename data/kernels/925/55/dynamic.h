//##############################################################################
/**
 *  @file    dynamic.h
 *  @author  James R. Schloss (Leios)
 *  @date    1/1/2017
 *  @version 0.1
 *
 *  @brief File for dynamic parameter parsing
 *
 *  @section DESCRIPTION
 *      This file has all functions necessary for dynamic parameter parsing,
 *      including parsing equation strings into abstract syntax trees
 */
 //#############################################################################

#ifndef DYNAMIC_H
#define DYNAMIC_H

#include <string>
#include "../include/ds.h"

/**
* @brief        Parses a provided equation string into an Abstract Syntax Tree
* @ingroup      dynamic, data
* @param        Parameter set
* @param        Equation string
* @param        String value to store in node
* @return       Abstract Syntax Tree representing the provided equation
*/
EqnNode parse_eqn(Grid &par, std::string eqn_string, std::string val_str);

/**
* @brief        finds the total number of elements in an Abstract Syntax Tree
* @ingroup      dynamic, data
* @param        AST
* @param        A count for each node
*/
void find_element_num(EqnNode eqn_tree, int &element_num);

/**
* @brief        Transforms an AST into a GPU array
* @ingroup      dynamic, data
* @param        CPU AST
* @param        GPU AST
* @param        Total number of elements
*/
void tree_to_array(EqnNode eqn, EqnNode_gpu *eqn_array, int &element_num);

/**
* @brief        Allocates space on GPU for AST
* @ingroup      dynamic, data
* @param        CPU AST
* @param        GPU AST
* @param        Element number
*/
void allocate_equation(EqnNode_gpu *eqn_cpu, EqnNode_gpu *eqn_gpu, int n);

/**
* @brief        Parses a provided file into Abstract Syntax Trees
* @ingroup      dynamic, data
* @param        Parameter set
*/
void parse_param_file(Grid &par);

/*----------------------------------------------------------------------------//
* GPU KERNELS
*-----------------------------------------------------------------------------*/

/**
* @brief        Evaluates a GPU AST with provided x, y, z, and t valued
* @ingroup      dynamic, data
* @param        GPUE AST
* @param        x
* @param        y
* @param        z
* @param        t
* @param        element number 
* @return       value for provided node
*/
__device__ double evaluate_eqn_gpu(EqnNode_gpu *eqn, double x, double y,
                                   double z, double time, int element_num);

/**
* @brief        Creates a fields for all values of x, y, and z with AST
* @ingroup      dynamic, data
* @param        Field
* @param        dx
* @param        dy
* @param        dz
* @param        xMax
* @param        yMax
* @param        zMax
* @param        time
* @param        GPUE AST
*/
__global__ void find_field(double *field, double dx, double dy, double dz, 
                           double xMax, double yMax, double zMax,
                           double time, EqnNode_gpu *eqn);

/**
* @brief        Creates a field of all zeros
* @ingroup      dynamic, data
* @param        Field
* @param        gridSize
*/
__global__ void zeros(double *field, int n);

/**
* @brief        Polynomial approximation for J bessel functions
* @ingroup      dynamic, data
* @param        Order of bessel function
* @param        x position
* @param        power of bessel function
* @return       J bessel function approximation
*/
__device__ double poly_j(int v, double x, int n);

#endif
