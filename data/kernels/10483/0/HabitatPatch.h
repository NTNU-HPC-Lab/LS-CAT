#ifndef HABITATPATCH_H
#define HABITATPATCH_H

class HabitatType;
typedef std::shared_ptr<HabitatType> HabitatTypePtr;

class HabitatPatch;
typedef std::shared_ptr<HabitatPatch> HabitatPatchPtr;

/**
 * Class for managing Habitat patches
 */
class HabitatPatch : public std::enable_shared_from_this<HabitatPatch> {

public:
    // CONSTRUCTORS AND DESTRUCTORS ///////////////////////////////////////////

    /**
     * Constructor I
     *
     * Constructs a %HabitatPatch object with default values
     */
    HabitatPatch();

    /**
     * Constructor II
     *
     * Constructs a %HabitatPatch object with assigned values
     */
    HabitatPatch(HabitatTypePtr typ, double area, double cx, double cy,
            double cap, double gr, double pop, double aar);

    /**
     * Destructor
     */
    ~HabitatPatch();

    // ACCESSORS //////////////////////////////////////////////////////////////

    /**
     * Returns the habitat type
     *
     * @return HabitatType as HabitatTypePtr
     */
    HabitatTypePtr getType() {
        return this->type;
    }
    /**
     * Sets the habitat type
     *
     * @param typ as HabitatTypePtr
     */
    void setType(HabitatTypePtr typ) {
        this->type.reset();
        this->type = typ;
    }

    /**
     * Returns the patch area
     *
     * @return Area as double
     */
    double getArea() {
        return this->area;
    }

    /**
     * Sets the patch area
     *
     * @param area as double
     */
    void setArea(double area) {
        this->area = area;
    }

    /**
     * Returns the x coordinate of the centroid
     *
     * @return Centroid X coordinate as double
     */
    double getCX() {
        return this->centroidX;
    }
    /**
     * Sets the x coordinate of the centroid
     *
     * @param cx as double
     */
    void setCX(double cx) {
        this->centroidX = cx;
    }

    /**
     * Returns the y coordinate of the centroid
     *
     * @return Centroid Y as double
     */
    double getCY() {
        return this->centroidY;
    }
    /**
     * Sets the y coordinate of the centroid
     *
     * @param cy as double
     */
    void setCY(double cy) {
        this->centroidY = cy;
    }

    /**
     * Returns the patch capacity
     *
     * @return Capacity as double
     */
    double getCapacity() {
        return this->capacity;
    }
    /**
     * Sets the patch capacity
     *
     * @param cap as double
     */
    void setCapacity(double cap) {
        this->capacity = cap;
    }

    /**
     * Returns the patch growth rate (%p.a.)
     *
     * @return Growth rate as double
     */
    double getGR() {
        return this->growthRate;
    }
    /**
     * Sets the patch growth rate (%p.a.)
     *
     * @param gr as double
     */
    void setGR(double gr) {
        this->growthRate = gr;
    }

    /**
     * Returns the current patch population
     *
     * @return Population as double
     */
    double getPopulation() {
        return this->population;
    }
    /**
     * Sets the current patch population
     *
     * @param pop as double
     */
    void setPopulation(double pop) {
        this->population = pop;
    }

    /**
     * Returns the current patch animals at risk
     *
     * @return AAR as double
     */
    double getAAR() {
        return this->aar;
    }
    /**
     * Sets the current patch animals at risk
     *
     * @param aar as double
     */
    void setAAR(double aar) {
        this->aar = aar;
    }

    /**
     * Returns the cells occupied by the patch
     *
     * @return Cells as const Eigen::MatrixXi&
     */
    const Eigen::MatrixXi& getCells() {
        return this->cells;
    }
    /**
     * Sets the cells occupied by the patch
     *
     * @param cells as Eigen::MatrixXi&
     */
    void setCells(const Eigen::MatrixXi& cells) {
        this->cells = cells;
    }

    // STATIC ROUTINES ////////////////////////////////////////////////////////

    // CALCULATION ROUTINES ///////////////////////////////////////////////////

private:
    HabitatTypePtr type;    /**< Type of habitat */
    double area;            /**< Area of patch */
    double centroidX;       /**< Longitude/x coord of centroid */
    double centroidY;       /**< Latitude/y coord of centroid */
    double capacity;        /**< Max capacity of patch */
    double growthRate;      /**< Prevailing population growth rate */
    double population;      /**< Prevailing population */
    double aar;             /**< Animals at risk in patch */
    Eigen::MatrixXi cells;  /**< Grid cells occupied by patch */
};

#endif
