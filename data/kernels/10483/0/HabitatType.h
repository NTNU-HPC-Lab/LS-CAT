#ifndef HABITATTYPE_H
#define HABITATTYPE_H

class HabitatType;
typedef std::shared_ptr<HabitatType> HabitatTypePtr;

/**
 * Class for managing habitat types.
 */
class HabitatType : public std::enable_shared_from_this<HabitatType> {

public:

    // ENUMERATIONS ///////////////////////////////////////////////////////////
    typedef enum {
        PRIMARY,
        MARGINAL,
        OTHER,
        CLEAR,
        ROAD
    } habType;

    // CONSTRUCTORS AND DESTRUCTORS ///////////////////////////////////////////

    /**
     * Constructor I
     *
     * Constructs a %HabitatType object with default values.
     */
    HabitatType();

    /**
     * Constructor II
     *
     * Constructs a %HabitatType object with assigned habitat
     */
    HabitatType(HabitatType::habType typ, double maxPop, const Eigen::VectorXi&
            vegetations, double habPrefMean, double habPrefSD,
            double cost);

    /**
     * Destructor
     */
    ~HabitatType();

    // ACCESSORS //////////////////////////////////////////////////////////////

    /**
     * Returns the type of habitat
     *
     * @return Type of habitat as HabitatType::habType
     */
     HabitatType::habType getType() {
             return this->type;
    }
    /**
     * Sets the type of habitat
     *
     * @param type of habitat as HabitatType::habType
     */
    void setType(HabitatType::habType type) {
            this->type = type;
    }

    /**
     * Returns the population per m^2
     *
     * @return Population per m^2 as double
     */
    double getMaxPop() {
            return this->maxPop;
    }
    /**
     * Sets the population per m^2
     *
     * @param pop as double
     */
    void setMaxPop(double pop) {
            this->maxPop = pop;
    }

    /**
     * Return the vegetations corresponding to this habitat
     *
     * @return Vegetations as const Eigen::VectorXi&
     */
    const Eigen::VectorXi& getVegetations() {
        return this->vegetations;
    }
    /**
     * Sets the vegetations corresponding to this habitat
     *
     * @param veg as const Eigen::VectorXi&
     */
    void setVegetations(const Eigen::VectorXi& veg) {
        this->vegetations = veg;
    }

    /**
     * Returns the vector of habitat preference mean with respect to primary
     * habitat.
     * (1 = primary/secondary, all others -ve [primary/secondary, marginal,
     * other,clear/road])
     *
     * @return Habitat preference mean as double
     */
    double getHabPrefMean() {
        return this->habPrefMean;
    }
    /**
     * Sets the vector of habitat preference mean with respect to primary
     * habitat.
     * (1 = primary/secondary, all others -ve [primary/secondary, marginal,
     * other,clear/road])
     *
     * @param hpm as double
     */
    void setHabPrefMean(double hpm) {
        this->habPrefMean = hpm;
    }

    /**
     * Returns the habitat preference standard deviation
     *
     * @return Habitat preference standard deviations as double
     */
    double getHabPrefSD() {
        return this->habPrefSD;
    }
    /**
     * Sets the habitat preference standard deviations
     *
     * @param hpsd as double
     */
    void setHabPrefSD(double hpsd) {
        this->habPrefSD = hpsd;
    }

    /**
     * Returns the cost per m^2 of occupying habitat of this type
     *
     * @return Cost as double
     */
    double getCostPerM2() {
        return this->cost;
    }
    /**
     * Sets the cost per m^2 of occupying habitat of this type
     *
     * @param cost as double
     */
    void setCostPerM2(double cost) {
        this->cost = cost;
    }

    // STATIC ROUTINES ////////////////////////////////////////////////////////
    /**
     * Return all vegetations
     *
     * @return Vegetations as IntVectorPtr
     */
    const Eigen::VectorXi& getAllVegetations() {
        return HabitatType::allVegetations;
    }
    /**
     * Sets all vegetations
     *
     * @param veg as IntVectorPtr
     */
    void setAllVegetations(const Eigen::VectorXi& veg) {
        HabitatType::allVegetations = veg;
    }

    // CALCULATION ROUTINES ///////////////////////////////////////////////////

private:
    HabitatType::habType type;	/**< Habitat type */
    double maxPop;              /**< Maximum population per m^2 */
    double cost;                /**< Cost per m^2 */
    Eigen::VectorXi vegetations;/**< Vegetations in this habitat type */
    double habPrefMean;         /**< Species mean preference for this habitat */
    double habPrefSD;           /**< Habitat preference standard deviation */

public:
    static Eigen::VectorXi allVegetations;  /**< All vegetations in the region */
};

#endif
