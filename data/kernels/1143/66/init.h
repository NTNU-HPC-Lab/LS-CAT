///@endcond
//##############################################################################
/**
 *  @file    evolution.h
 *  @author  James Ryan Schloss (leios)
 *  @date    5/31/2016
 *  @version 0.1
 *
 *  @brief function for evolution.
 *
 *  @section DESCRIPTION
 *  These functions and variables are necessary for carrying out the GPUE
 *	simulations. This file will be re-written in an improved form in some
 *	future release.
 */
//##############################################################################

#ifndef INIT_H
#define INIT_H

#include "../include/split_op.h"
#include "../include/kernels.h"
#include "../include/constants.h"
#include "../include/fileIO.h"
#include "../include/tracker.h"
#include "../include/minions.h"
#include "../include/parser.h"
#include "../include/ds.h"
#include "../include/unit_test.h"
#include "../include/operators.h"

#include "../include/lattice.h"
#include "../include/node.h"
#include "../include/edge.h"
#include "../include/manip.h"
#include "../include/vort.h"
#include "../include/evolution.h"
#include <string>
#include <iostream>

 /**
 * @brief       check to make sure we have enough memory for computation
 * @ingroup     data
 * @param       Grid class
 */

void check_memory(Grid &par);

 /**
 * @brief	Initializes data structures
 * @ingroup	data
 * @param	Grid class
 */
int init(Grid &par);

 /**
 * @brief	Sets variables for either real or imaginary time evolution
 * @ingroup	data
 * @param	Grid class
 * @param	ev_type boolean (0 for imaginary-time, 1 for real-time)
 */
void set_variables(Grid &par, bool ev_type);


#endif
