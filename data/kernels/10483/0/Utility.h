#ifndef UTILITY_H
#define UTILITY_H

namespace Utility {
    /**
     * Converts the subscripts (rows,columns) used to address an Eigen matrix
     * (only 2D) to single-valued, column-major-based indices, similar to
     * Matlab's sub2ind function.
     *
     * @param X as m by n matrix (Eigen)
     * @param R list of row indices (Eigen)
     * @param C list of column indices (Eigen)
     * @return Y list of indices (Eigen)
     */
    template < typename DerivedX,
            typename DerivedR,
            typename DerivedC,
            typename DerivedY>
    static void sub2ind(
            const Eigen::PlainObjectBase<DerivedX>& X,
            const Eigen::PlainObjectBase<DerivedR>& R,
            const Eigen::PlainObjectBase<DerivedC>& C,
            Eigen::PlainObjectBase<DerivedY>& Y) {

        if (R.size() != C.size()) {
            throw "Utility::sub2ind: Row and column index vectors must be the same length!";
        }

        if (((R.array()<0).sum() > 0) || ((C.array()<0).sum() > 0)) {
            throw "Utility::sub2ind: Matrix subscripts must be >= 0 !";
        }

        int xm = X.rows();
        int xn = X.cols();

        if (((R.array()>=xm).sum() > 0) || ((C.array()>=xn).sum() > 0)) {
            throw "Utility::sub2ind: Index vectors exceed matrix dimensions!";
        }

        Y = R + C*xm;
    }

    /**
     * Converts the single-valued, column-major-based indices used to address
     * an Eigen matrix to subscripts(rows,columns) (only 2D), similar to
     * Matlab's ind2sub function.
     *
     * @param X as m by n matrix (Eigen)
     * @param Y list of indices (Eigen)
     * @return R list of row subscript (Eigen)
     * @return C list of column subscript (Eigen)
     */
    template < typename DerivedX,
            typename DerivedY,
            typename DerivedR,
            typename DerivedC>
    static void ind2sub(
            const Eigen::PlainObjectBase<DerivedX>& X,
            const Eigen::PlainObjectBase<DerivedY>& Y,
            Eigen::PlainObjectBase<DerivedR>& R,
            Eigen::PlainObjectBase<DerivedC>& C) {

        int xm = X.rows();
        int xn = X.cols();

        if ((Y.array() >= xm*xn).sum() > 0) {
            throw "Index vector exceeds matrix size!";
        }

        Eigen::VectorXd tempvec = Y/xm;
        igl::floor(tempvec, C);
        R = Y - C*xm;
    }

    /**
     * Retrieves the corresponding entries at each index in I of matrix X and
     * returns the result to a corresponding vector.
     *
     * @param X as m by n matrix (Eigen)
     * @param I list of indices (Eigen)
     * @return Y list of corresponding values (Eigen)
     */
    template <typename T>
    static void sliceIdx(
            const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& X,
            const Eigen::Matrix<int,Eigen::Dynamic,1>& I,
            Eigen::Matrix<T,Eigen::Dynamic,1>& Y) {

        const T* data = X.data();

        // Loop over indices. The data input matrix is in COLUMN-MAJOR form
        for (int ii = 0; ii < I.size(); ii++) {
            Y(ii) = data[I(ii)];
        }
    }

    /**
     * Assigns to the corresponding entries at each index in I of matrix X and
     * based on an input vectr, Y.
     *
     * @param X as m by n matrix (Eigen)
     * @param I list of indices (Eigen)
     * @return Y list of corresponding values (Eigen)
     */
    template <typename T>
    static void sliceIntoIdx(
            Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& X,
            const Eigen::Matrix<int,Eigen::Dynamic,1>& I,
            const Eigen::Matrix<T,Eigen::Dynamic,1>& Y) {

        T* data = X.data();

        // Loop over indices. The data input matrix is in COLUMN-MAJOR form
        for (int ii = 0; ii < I.size(); ii++) {
            data[I(ii)] = Y(ii);
        }
    }

    /**
     * Assigns to the corresponding entries at each I,J pair of matrix X and
     * based on an input vectr, Y.
     *
     * @param X vector of values to read (Eigen)
     * @param I as corresponding vector of x indices in Y (Eigen)
     * @param J as coresponding vector of y indices in Y (Eigen)
     * @return Y as m by n matrix to update (Eigen)
     */
    template <typename T>
    static void sliceIntoPairs(
            const Eigen::Matrix<T,Eigen::Dynamic,1>& X,
            const Eigen::Matrix<int,Eigen::Dynamic,1>& I,
            const Eigen::Matrix<int,Eigen::Dynamic,1>& J,
            Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& Y) {

        // Loop over pairs
        for (int ii = 0; ii < I.size(); ii++) {
            Y(I(ii),J(ii)) = X(ii);
        }
    }

    /**
     * Assigns to the corresponding entries at each I,J pair of matrix X and
     * based on an input vectr, Y but this time, the inputs are arranged as
     * matrices instead of vectors.
     *
     * @param X as a by c matrix to read (Eigen)
     * @param I as a by c matrix of corresponding x indices in Y (Eigen)
     * @param J as a by c matrix of corresponding y indices in Y (Eigen)
     * @return Y as m by n matrix to update (Eigen)
     */
    template <typename T>
    static void sliceIntoPairs(
            const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& X,
            const Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic>& I,
            const Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic>& J,
            Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& Y) {

        // Loop over pairs
        for (int ii = 0; ii < I.cols(); ii++) {
            for (int jj = 0; jj < I.rows(); jj++) {
                Y(I(jj,ii),J(jj,ii)) = X(jj,ii);
            }
        }
    }

    /**
     * Allows determining where values in a matrix are finite
     */
    template<typename Derived>
    inline bool is_finite(const Eigen::MatrixBase<Derived>& x)
    {
       return ( (x - x).array() == (x - x).array()).all();
    }

    /**
     * Allows determining where values in a matrix are NaN
     */
    template<typename Derived>
    inline bool is_nan(const Eigen::MatrixBase<Derived>& x)
    {
       return ((x.array() == x.array())).all();
    }

    /**
     * Computes the number of times each line segment in XY1 intersects any
     * line segment in XY2. The lines are piecewise linear.
     *
     * @param XY1 as Eigen::MatrixXd&. This is an N1 x 4 matrix where the
     * columns, in order, are: starting x coordinate, starting y coordinate,
     * ending x coordinate, ending y coordinate.
     *
     * @param XY2 as Eigen::MatrixXd&. This is an N2 x 4 matrix where the
     * columns have the same meaning as in XY1. This is the curve that the
     * individual lines in XY1 may intersect.
     *
     * @param gpu as bool. Default = false
     *
     * @return Number of intersections as Eigen::MatrixXi
     */
    Eigen::MatrixXi lineSegmentIntersect(const Eigen::MatrixXd& XY1,
            const Eigen::MatrixXd& XY2, bool gpu = false);

    /**
     * Same as lineSegmentIntersect but with extra information:
     *
     * @param adjMat as Eigen::MatrixXi& : N1xN2 indicator matrix where the
     * entry (i,j) is 1 if line segments XY1(i,:) and XY2(j,:) intersect.
     * @param intMatX as Eigen::MatrixXd& : N1xN2 matrix where the entry (i,j)
     * is the X coordinate of the intersection point between line segments
     * XY1(i,:) and XY2(j,:).
     * @param intMatY as Eigen::MatrixXd& : N1xN2 matrix where the entry (i,j)
     * is the Y coordinate of the intersection point between line segments
     * XY1(i,:) and XY2(j,:).
     * @param intNorDist1to2' as Eigen::MatrixXi& : N1xN2 matrix where the
     * (i,j) entry is the normalized distance from the start point of line
     * segment XY1(i,:) to the intersection point with XY2(j,:).
     * @param intNormdDist2To1 as Eigen::MatrixXi& : N1xN2 matrix where the
     * (i,j) entry is the normalized distance from the start point of line
     * segment XY1(j,:) to the intersection point with XY2(i,:).
     * @param parAdjMat as Eigen::MatrixX<bool,Eigen::Dynamic,Eigen::Dynamic>&
     * : N1xN2 indicator matrix where the (i,j) entry is 1 if line segments
     * XY1(i,:) and XY2(j,:) are parallel.
     * @param coincAdjMat as Eigen::MatrixX<bool,Eigen::Dynamic,Eigen::Dynamic>
     * : N1xN2 indicator matrix where the (i,j) entry is 1 if line segments
     * XY1(i,:) and XY2(j,:) are coincident.
     *
     * N.B. This function will be completed at a later date
     *
     * The original MATLAB code was developed by
     * U. Murat Erdem
     * www.mathworks.com/matlabcentral/fileexchange/27205-fast-line-segment-intersection
     * outStruct = lineSegmentIntersect(patchDistLines,roadSegsVisible);
     */
    void lineSegmentIntersect(
            const Eigen::MatrixXd& XY1, const Eigen::MatrixXd& XY2,
            Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> &adjMat,
            Eigen::MatrixXd& intMatX, Eigen::MatrixXd& intMatY,
            Eigen::MatrixXd& normDist1to2, Eigen::MatrixXd& normDist2to1,
            Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>& parAdjMat,
            Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>& coincAdjMat);

    /**
     * Returns the standard normal inverse given a cumulative probability
     *
     * Code by John D. Cook. Computes the inverse normal distribution using the
     * approximation for statistical functions by Abramawitz and Stegun,
     * Formula 26.2.23:
     * Abramowitz, M., & Stegun, I. A. (1964). Handbook of mathematical
     * functions: with formulas, graphs, and mathematical tables (Vol. 55).
     * Courier Corporation.
     *
     * Code By:
     * John D. Cook
     * www.johndcook.com/blog/normal_cdf_inverse
     * Accessed: October 6, 2016
     *
     * @param p as double
     * @return Standard deviations from the mean as double
     */
    double NormalCDFInverse(double p);

    /**
     * Implements the Abramowitz and Stegun approximation
     *
     * Code by John D. Cook. Computes the inverse normal distribution using the
     * approximation for statistical functions by Abramawitz and Stegun,
     * Formula 26.2.23:
     * Abramowitz, M., & Stegun, I. A. (1964). Handbook of mathematical
     * functions: with formulas, graphs, and mathematical tables (Vol. 55).
     * Courier Corporation.
     *
     * Code By:
     * John D. Cook
     * www.johndcook.com/blog/normal_cdf_inverse
     * Accessed: October 6, 2016
     * @param t as double
     * @return Relevant standard deviation as double
     */
    double RationalApproximation(double t);

    /**
     * Creates orthogonal cutting planes to the straight line joining the start
     * and end points.
     *
     * This routine creates 'n' 2D cutting planes, where 'n' is the number of
     * intersection points defining the road.
     *
     * @param xMin as const double&
     * @param xMax as const double&
     * @param yMin as const double&
     * @param yMax as const double&
     * @param zMin as const double&
     * @param zMax as const double&
     * @param xS as const double&
     * @param yS as const double&
     * @param zS as const double&
     * @param xE as const double&
     * @param yE as const double&
     * @param zE as const double&
     * @param n as const long&
     * @param (output) xO as Eigen::VectorXd&
     * @param (output) yO as Eigen::VectorXd&
     * @param (output) zO as Eigen::VectorXd&
     * @param (output) dU as Eigen::VectorXd&
     * @param (output) dL as Eigen::VectorXd&
     * @param (output) theta as const double&
     */
    void cuttingPlanes(const double& xMin, const double& xMax, const double&
            yMin, const double& yMax, const double& xS, const double& yS,
            const double& zS, const double& xE, const double& yE, const
            double& zE, const long& n, Eigen::VectorXd& xO, Eigen::VectorXd&
            yO, Eigen::VectorXd& zO, Eigen::VectorXd& dU, Eigen::VectorXd& dL,
            double &theta);

    /**
     * Computes local linear kernel smoothing regression with variable window
     * width, producing a vector of inputs (x) mapped to a vector of outputs
     *
     * This routine is adapted from the MATLAB code of Fernando Duarte (Feb 24,
     * 2012)
     *
     * Additional reference for mathematics:
     *
     * 1. Nonparametric econometrics, by Adrian Pagan and Aman Ullah,
     *      Cambridge University Press, 1999, ISBN 0521586119, 9780521586115
     *
     * @param (input) x as const Eigen::VectorXd&
     * @param (input) y as const Eigen::VectorXd&
     * @param (input) ww is the window size as int
     * @param (input) n is the number of x values as int
     * @param (output) rx as Eigen::VectorXd&
     * @param (output) ry as Eigen::VectorXd&
     */
    void ksrlin_vw(const Eigen::VectorXd& x, const Eigen::VectorXd& y, const
            int ww, const int n, Eigen::VectorXd& rx, Eigen::VectorXd& ry);

    /**
     * Interpolates a surrogate function using provided values for the
     * predictors.
     *
     * @param surrogate as Eigen::VectorXd&
     * @param predictors as Eigen::VectorXd&
     * @param dimRes as int
     * @return Value as double
     */
    double interpolateSurrogate(Eigen::VectorXd &surrogate, Eigen::VectorXd&
            predictors, int dimRes);

    /**
     * Returns the name of an input as well as the values
     *
     * @param s as std::string
     * @param del as std::string
     * @return Labels and input values as std::vector<std::string>
     */
    std::vector<std::string> getDictionary(std::string s, std::string del);
}

#endif
