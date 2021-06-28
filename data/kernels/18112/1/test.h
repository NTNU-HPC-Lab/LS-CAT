/**
 * \file test.h
 * \brief A library to test ecm
 */

/**
 * \brief Tests pro_curve_point function using GMP
 * 1. Generates a random curve and a projective point
 * using the function
 *
 * 2. Checks if the coefficients of the curve and the
 * point satisfies the curve equation using GMP
 * @param[in] THRESHOLD total number of tests
 */
void pro_curve_point_gmp_test(int THRESHOLD);

/**
 * \brief Tests aff_curve_point function using GMP
 * 1. Generates a random curve and a affine point
 * using the function
 *
 * 2. Checks if the coefficients of the curve and the
 * point satisfies the curve equation using GMP
 * @param[in] THRESHOLD total number of tests
 */
void aff_curve_point_gmp_test(int THRESHOLD);

/**
 * \brief Tests pro_add function using GMP
 * 1. Generates two random points
 *
 * 2. Computes the addition by implementing the
 * algebraic calculations using GMP
 *
 * 3. Compares the result of the function
 * with the GMP result
 * @param[in] THRESHOLD total number of tests
 */
void pro_add_gmp_test(int THRESHOLD);

/**
 * \brief Tests pro_add function using Magma
 * 1. Generates two random points
 *
 * 2. Computes the addition by implementing the
 * algebraic calculations using Magma
 *
 * 3. Compares the result of the function
 * with the Magma result
 * @param[in] THRESHOLD total number of tests
 */
void pro_add_magma_test(int THRESHOLD);

/**
 * \brief Tests pro_dbl function using Magma
 * 1. Generates a random point
 *
 * 2. Computes the double by implementing the
 * algebraic calculations using Magma
 *
 * 3. Compares the result of the function
 * with the Magma result
 * @param[in] THRESHOLD total number of tests
 */
void pro_dbl_magma_test(int THRESHOLD);

/**
 * \brief Tests pro_ladder function using gmp
 *
 * Due to the commutativity of the point addition, if
 * \f$p1 = l*(k*P)\f$ and \f$p2 = k*(l*P)\f$, then
 * \f$p1 = p2\f$, such that \f$\frac{p1 \rightarrow X}
 * {p1 \rightarrow Z} = \frac{p2 \rightarrow X}
 * {p2 \rightarrow Z}\f$. This function uses this property
 * to test pro_ladder function.
 *
 * 1. Generates a curve \f$c\f$ and a projective point
 * \f$p1\f$ on it using pro_curve_point function
 *
 * 2. Generates random multipliers \f$k\f$ an \f$l\f$
 *
 * 3. Multiplies the point with first \f$k\f$ and
 * then \f$l\f$ such that \f$p3 = l*(k*p1)\f$ using
 * pro_ladder function
 *
 * 4. Multiplies the point with first \f$l\f$ and
 * then \f$k\f$ such that \f$p5 = k*(l*p1)\f$ using
 * pro_ladder function
 *
 * 6. Calculates \f$p3 \rightarrow X * p5 \rightarrow Z\f$
 * using gmp
 *
 * 7. Calculates \f$p5 \rightarrow X * p3 \rightarrow Z\f$
 * using gmp
 *
 * 8. Compares the results of step 5 and 6 using gmp
 * @param[in] THRESHOLD total number of tests
 */
void pro_ladder_gmp_test(int THRESHOLD);

/**
 * \brief Tests pro_ladder function using Magma
 *
 * Due to the commutativity of the point addition, if
 * \f$p1 = l*(k*P)\f$ and \f$p2 = k*(l*P)\f$, then
 * \f$p1 = p2\f$, such that \f$\frac{p1 \rightarrow X}
 * {p1 \rightarrow Z} = \frac{p2 \rightarrow X}
 * {p2 \rightarrow Z}\f$. This function uses this property
 * to test pro_ladder function.
 *
 * 1. Generates a curve \f$c\f$ and a projective point
 * \f$p1\f$ on it using pro_curve_point function
 *
 * 2. Generates random multipliers \f$k\f$ an \f$l\f$
 *
 * 3. Multiplies the point with first \f$k\f$ and
 * then \f$l\f$ such that \f$p3 = l*(k*p1)\f$ using
 * pro_ladder function
 *
 * 4. Multiplies the point with first \f$l\f$ and
 * then \f$k\f$ such that \f$p5 = k*(l*p1)\f$ using
 * pro_ladder function
 *
 * 6. Calculates \f$p3 \rightarrow X * p5 \rightarrow Z\f$
 * using Magma
 *
 * 7. Calculates \f$p5 \rightarrow X * p3 \rightarrow Z\f$
 * using Magma
 *
 * 8. Compares the results of step 5 and 6 using Magma
 * @param[in] THRESHOLD total number of tests
 */
void pro_ladder_magma_test(int THRESHOLD);

/**
 * \brief Tests ecm function
 * 1. Generates a random composite number
 *
 * 2. Calculates a factor of the number
 * using the function
 *
 * 3. Checks whether the factor found
 * actually divides the composite number
 * @param[in] THRESHOLD total number of tests
 */
void ecm_gmp_test(int THRESHOLD);
