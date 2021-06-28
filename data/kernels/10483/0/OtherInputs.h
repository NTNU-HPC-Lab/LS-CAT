#ifndef OTHERINPUTS_H
#define OTHERINPUTS_H

class OtherInputs;
typedef std::shared_ptr<OtherInputs> OtherInputsPtr;

class OtherInputs : public std::enable_shared_from_this<OtherInputs> {

public:
    // CONSTRUCTORS AND DESTRUCTORS ///////////////////////////////////////////

    /**
     * Constructor I
     */
    OtherInputs();

    /**
     * Constructor II
     *
     * Constructs an %OtherInputs object with default values.
     */
    OtherInputs(std::string& idf, std::string& orf, std::string& itf,
            std::string& erf, double minLat, double maxLat, double minLon,
            double maxLon, unsigned long latPoints, unsigned long lonPoints,
            unsigned long habGridRes, unsigned long noPaths, unsigned long
            dimRes);

    /**
     * Destructor
     */
    ~OtherInputs();

    // ACCESSORS //////////////////////////////////////////////////////////////

    /**
     * Returns the absolute location of the input data file
     *
     * @return Location as std::string
     */
    std::string getInputDataFile() {
        return this->inputDataFile;
    }
    /**
     * Sets the absolute location of the input data file
     *
     * @param loc as std::string
     */
    void setInputDataFile(std::string loc) {
        this->inputDataFile = loc;
    }

    /**
     * Returns the absolute location of the output results file
     *
     * @return Location as std::string
     */
    std::string getOutputResults() {
        return this->outputResultsFile;
    }
    /**
     * Sets the absolute location of the output results file
     *
     * @param loc as std::string
     */
    void setOutputResultsFile(std::string loc) {
        this->outputResultsFile = loc;
    }

    /**
     * Returns the absolute location of the input terrain file
     *
     * @return Location as std::string
     */
    std::string getInputTerrainFile() {
        return this->inputTerrainFile;
    }
    /**
     * Sets the absolute location of the input terrain file
     *
     * @param loc as std::string
     */
    void setInputTerrainFile(std::string loc) {
        this->inputTerrainFile = loc;
    }

    /**
     * Returns the absolute location of the existing roads file
     *
     * @return Location as std::string
     */
    std::string getExistingRoadsFile() {
        return this->existingRoadsFile;
    }
    /**
     * Sets the absolute location of the existing roads file
     *
     * @param loc as std::string
     */
    void setExistingRoadsFile(std::string loc) {
        this->existingRoadsFile = loc;
    }

    /**
     * Returns the minimum latitude
     *
     * @return Minimum latitude as double
     */
    double getMinLat() {
        return this->minLat;
    }
    /**
     * Sets the minimum latitude
     *
     * @param minLat as double
     */
    void setMinLat(double minLat) {
        this->minLat = minLat;
    }

    /**
     * Returns the maximum latitude
     *
     * @return Maximum latitude as double
     */
    double getMaxLat() {
        return this->maxLat;
    }
    /**
     * Sets the maximum latitude
     *
     * @param maxLat as double
     */
    void setMaxLat(double maxLat) {
        this->maxLat = maxLat;
    }

    /**
     * Returns the minimum longitude
     *
     * @return Minimum longitude as double
     */
    double getMinLon() {
        return this->minLon;
    }
    /**
     * Sets the minimum longitude
     *
     * @param minLon as double
     */
    void setMinLon(double minLon) {
        this->minLon = minLon;
    }

    /**
     * Returns the maximum longitude
     *
     * @return Maximum longitude as double
     */
    double getMaxLon() {
        return this->maxLon;
    }
    /**
     * Sets the maximum longitude
     *
     * @param maxLon as double
     */
    void setMaxLon(double maxLon) {
        this->maxLon = maxLon;
    }

    /**
     * Returns the number of latitude grid spacings
     *
     * @return Points as unsigned long
     */
    unsigned long getLatPoints() {
        return this->latPoints;
    }
    /**
     * Sets the number of latitude grid spacings
     *
     * @param points as unsigned long
     */
    void setLatPoints(unsigned long points) {
        this->latPoints = points;
    }

    /**
     * Returns the number of longitude grid spacings
     *
     * @return Points as unsigned long
     */
    unsigned long getLonPoints() {
        return this->lonPoints;
    }
    /**
     * Sets the number of longitude grid spacings
     *
     * @param points as unsigned long
     */
    void setLonPoints(unsigned long points) {
        this->lonPoints = points;
    }

    /**
     * Returns the number of habitat grid spacings in longest dimension
     *
     * @return Points as unsigned long
     */
    unsigned long getHabGridRes() {
        return this->habGridRes;
    }
    /**
     * Sets the number of habitat grid spacings in longest dimension
     *
     * @param hgr as unsigned long
     */
    void setHabGridRes(unsigned long hgr) {
        this->habGridRes = hgr;
    }

    /**
     * Returns the number of simulation paths to use
     *
     * @return Number of paths as unsigned long
     */
    unsigned long getNoPaths() {
        return this->noPaths;
    }
    /**
     * Sets the number of simulation paths to use
     *
     * @param paths as unsigned long
     */
    void setNoPaths(unsigned long paths) {
        this->noPaths = paths;
    }

    /**
     * Returns the resolution of each ROV predictor used in the regressions
     *
     * @return Dimension resolution as unsigned long
     */
    unsigned long getDimRes() {
        return this->dimRes;
    }
    /**
     * Sets the resoultion of each ROV predictor used in the regressions
     *
     * @param dr as unsigned long
     */
    void setDimRes(unsigned long dr) {
        this->dimRes = dr;
    }

    // STATIC ROUTINES ////////////////////////////////////////////////////////

    // CALCULATION ROUTINES ///////////////////////////////////////////////////

private:
    std::string inputDataFile;      /**< Absolute location of input data file */
    std::string outputResultsFile;  /**< Absolute location of output results */
    std::string inputTerrainFile;   /**< Absolute location of terrain file */
    std::string existingRoadsFile;  /**< Absolute location of roads file */
    double minLat;                  /**< Minimum latitude for window */
    double maxLat;                  /**< Maximum latitude for window */
    double minLon;                  /**< Minimum longitude for window */
    double maxLon;                  /**< Maximum longitude for window */
    unsigned long latPoints;        /**< Number of latitude points for grid */
    unsigned long lonPoints;        /**< Number of longitude points for grid */
    unsigned long habGridRes;       /**< Habitat grid resolution in longest dimension */
    unsigned long noPaths;          /**< Number of simulation paths */
    unsigned long dimRes;           /**< Resolution for each dimension in ROV regression */
};

#endif
