#ifndef REGION_H
#define REGION_H

class Region;
typedef std::shared_ptr<Region> RegionPtr;

/**
 * Class for managing spatial data for the study region
 */
class Region : public std::enable_shared_from_this<Region> {

public:
    // CONSTRUCTORS AND DESTRUCTORS ///////////////////////////////////////////

    /**
     * Constructor I
     *
     * Constructs a %Region object with values passed from main program
     */
    Region(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y, const
            Eigen::MatrixXd& Z, const Eigen::MatrixXd& acCost, const
            Eigen::MatrixXd& ssc, const Eigen::MatrixXd& cc, const
            Eigen::MatrixXi& veg, std::string inputFile = "");

    /**
     * Constructor II
     *
     * Constructs a %Region object using a previously saved data structure
     */
    Region(std::string input);

    /**
     * Constructor III
     *
     * Constructs a %Region object using raw data files. Data is in .csv format:
     * 1. X
     * 2. Y
     * 3. Z
     * 4. Acquisition cost
     * 5. Soil stabilisation cost
     * 6. Clearance cost
     * 7. Habitat (1 = primary/secondary, 2 = marginal, 3 = other, 4 = clear)
     *
     * All dimensions arranged as a matrix of cells:
     * (X1,Y1): {1,2,3,4,5,6,7}, (X2,Y1): {1,2,3,4,5,6,7} ... (XN,Y1): {1,2,3,4,5,6,7}
     * (X1,Y2): {1,2,3,4,5,6,7}, (X2,Y2): {1,2,3,4,5,6,7} ... (XN,Y2): {1,2,3,4,5,6,7}
     * .
     * .
     * .
     * (X1,YM): {1,2,3,4,5,6,7}, (X2,YM): {1,2,3,4,5,6,7} ... (XN,YM): {1,2,3,4,5,6,7}
     */
    Region(std::string rawData, bool rd = true);

    /**
     * Destructor
     */
    ~Region();

    // ACCESSORS //////////////////////////////////////////////////////////////

    /**
     * Returns the matrix of X coordinates
     *
     * @return X matrix as const Eigen::MatrixXd&
     */
    const Eigen::MatrixXd& getX() {
        return this->X;
    }
    /**
     * Sets the matrix of X coordinates
     *
     * @param X as const Eigen::MatrixXd&
     */
    void setX(const Eigen::MatrixXd& X) {
        this->X = X;
    }

    /**
     * Returns the matrix of Y coordinates
     *
     * @return Y matrix as const Eigen::MatrixXd&
     */
    const Eigen::MatrixXd& getY() {
        return this->Y;
    }
    /**
     * Sets the matrix of Y coordinates
     *
     * @param Y as const Eigen::MatrixXd&
     */
    void setY(const Eigen::MatrixXd& Y) {
        this->Y = Y;
    }

    /**
     * Returns the matrix of Z coordinates
     *
     * @return Z matrix as const Eigen::MatrixXd&
     */
    const Eigen::MatrixXd& getZ() {
        return this->Z;
    }
    /**
     * Sets the matrix of Z coordinates
     *
     * @param Z as const Eigen::MatrixXd&
     */
    void setZ(const Eigen::MatrixXd& Z) {
        this->Z = Z;
    }

    /**
     * Returns the matrix of cell indices
     *
     * @return Idx matrix as const Eigen::MatrixXi&
     */
    const Eigen::MatrixXi& getCellIdx() {
        return this->Idx;
    }
    /**
     * Sets the matrix of cell indices
     *
     * @param Z as const Eigen::MatrixXi&
     */
    void setCellIdx(const Eigen::MatrixXi& idx) {
        this->Idx = idx;
    }

    /**
     * Returns the matrix of vegetations
     *
     * @return Vegetation matrix as const Eigen::MatrixXi&
     */
    const Eigen::MatrixXi& getVegetation() {
        return this->veg;
    }
    /**
     * Sets the matrix of vegetations
     *
     * @param beg as const Eigen::MatrixXi&
     */
    void setVegetation(const Eigen::MatrixXi& veg) {
        this->veg = veg;
    }

    /**
     * Returns the matrix of acquisition costs
     *
     * @return Acquisition costs matrix as const Eigen::MatrixXd&
     */
    const Eigen::MatrixXd& getAcquisitionCost() {
        return this->acCost;
    }
    /**
     * Sets the matrix of acquisition costs
     *
     * @param acc as const Eigen::MatrixXd&
     */
    void setAcquisitionCost(const Eigen::MatrixXd& acc) {
        this->acCost = acc;
    }

    /**
     * Returns the matrix of soil stabilisation costs
     *
     * @return Soil stabilisation cost matrix as const Eigen::MatrixXd&
     */
    const Eigen::MatrixXd& getSoilStabilisationCost() {
        return this->soilStabCost;
    }
    /**
     * Sets the matrix of soil stabilisation costs
     *
     * @param ssc as const Eigen::MatrixXd&
     */

    void setSoilStabilisationCost(const Eigen::MatrixXd& ssc) {
        this->soilStabCost = ssc;
    }

    /**
     * Returns the matrix of clearance costs
     *
     * @return Clearance cost matrix as const Eigen::MatrixXd&

    const Eigen::MatrixXd& getClearanceCost() {
        return this->clearCosts;
    }
    /**
     * Sets the matrix of clearance costs
     *
     * @param cc as const Eigen::MatrixXd&

    void setClearanceCost(const Eigen::MatrixXd& cc) {
        this->clearCosts = cc;
    }

    /**
     * Returns the absolute location of the input file
     *
     * @return Input file location as std::string
     */
    std::string getInputFile() {
        return this->inputFile;
    }
    /**
     * Sets the absolute location of the input file
     *
     * @param loc as std::string
     */
    void setInputFile(std::string loc) {
        this->inputFile = loc;
    }

    /**
     * Places a pair of X and Y coordinates onto the region to get the
     * corresponding Z coordinates.
     *
     * @param x as const Eigen::VectorXd&
     * @param y as const Eigen::VectorXd&
     * @param z as Eigen::VectorXd&
     */
    void placeNetwork(const Eigen::VectorXd& x, const Eigen::VectorXd& y,
            Eigen::VectorXd& z);

    /**
     * Places a single X and Y coordinate onto the region to get the
     * corresponding Z coordinate.
     *
     * @param x as double&
     * @param y as double&
     * @param z as double&
     */
    void placeNetwork(double& x, double& y, double& z);


    // STATIC ROUTINES ////////////////////////////////////////////////////////

    // CALCULATION ROUTINES ///////////////////////////////////////////////////

private:
    Eigen::MatrixXd X;              /**< X coordinates matrix (all columns the same) */
    Eigen::MatrixXd Y;              /**< Y coordinates matrix (all rows the same) */
    Eigen::MatrixXd Z;              /**< Z coordinates matrix */
    Eigen::MatrixXi Idx;            /**< Cell indices in column-major format */
    Eigen::MatrixXi veg;            /**< Vegetation at each grid reference */
    Eigen::MatrixXd acCost;         /**< Acquisition cost per sq m */
    Eigen::MatrixXd soilStabCost;   /**< Soil characteristics */
    Eigen::MatrixXd clearCosts;     /**< Clearance costs per sq m */
    std::string inputFile;          /**< Absolute location of input file (built from program) */
};

#endif
