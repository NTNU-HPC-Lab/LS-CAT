#ifndef VEHICLE_H
#define VEHICLE_H

class Commodity;
typedef std::shared_ptr<Commodity> CommodityPtr;

class Vehicle;
typedef std::shared_ptr<Vehicle> VehiclePtr;

/**
 * Class for managing vehicle objects
 */
class Vehicle : public std::enable_shared_from_this<Vehicle> {

public:
    // CONSTRUCTORS AND DESTRUCTORS ///////////////////////////////////////////
    /**
     * Constructor I
     */
    Vehicle();

    /**
     * Constructor II
     *
     * Constructs a %Vehicle object
     */
    Vehicle(CommodityPtr fuel, std::string nm, double width, double length,
            double trafficProp, double load, double a, double agr, double av,
            double avsq, double travel);

    /**
     * Destructor
     */
    ~Vehicle();

    // ACCESSORS //////////////////////////////////////////////////////////////

    /**
     * Returns the fuel used
     *
     * @return Fuel as CommodityPtr
     */
    CommodityPtr getFuel() {
        return this->fuel;
    }
    /**
     * Sets the fuel used
     *
     * @param fuel as CommodityPtr
     */
    void setFuel(CommodityPtr fuel) {
        this->fuel.reset();
        this->fuel = fuel;
    }

    /**
     * Returns the name
     *
     * @return Name as std::string
     */
    std::string getName() {
        return this->name;
    }
    /**
     * Sets the name
     *
     * @param nm as std::string
     */
    void setName(std::string nm) {
        this->name = nm;
    }

    /**
     * Returns the average vehicle width
     *
     * @return Average vehicle width as double
     */
    double getWidth() {
        return this->averageWidth;
    }
    /**
     * Sets the average vehicle width
     *
     * @param width as double
     */
    void setWidth(double width) {
        this->averageWidth = width;
    }

    /**
     * Returns the average vehicle length
     *
     * @return Average vehicle Length as double
     */
    double getLength() {
        return this->averageLength;
    }
    /**
     * Sets the average vehicle length
     *
     * @param length as double
     */
    void setLength(double length) {
        this->averageLength = length;
    }

    /**
     * Returns the proportion of traffic represented by this vehicle.
     *
     * @return Proportion as double
     */
    double getProportion() {
        return this->trafficProportion;
    }
    /**
     * Sets the proportion of traffic represented by this vehicle
     *
     * @param prop as double
     */
    void setProportion(double prop) {
        this->trafficProportion = prop;
    }

    /**
     * Returns the maximum carrying load (tonnes)
     *
     * @return Load as double
     */
    double getMaximumLoad() {
        return this->maxLoad;
    }
    /**
     * Sets the maximum carrying load (tonnes)
     *
     * @param load as double
     */
    void setMaximumLoad(double load) {
        this->maxLoad = load;
    }

    /**
     * Returns the a constant from Jong et al 1999
     *
     * @return Constant as double
     */
    double getConstant() {
        return this->aConst;
    }
    /**
     * Sets the a constant from Jong et al 1999
     *
     * @param a as double
     */
    void setConstant(double a) {
        this->aConst = a;
    }

    /**
     * Returns the grade coefficient from Jong et al 1999
     *
     * @return Coefficient as double
     */
    double getGradeCoefficient() {
        return this->agr;
    }
    /**
     * Sets the grade coefficient from Jong et al 1999
     *
     * @param grade as double
     */
    void setGradeCoefficient(double grade) {
        this->agr = grade;
    }

    /**
     * Returns the velocity coefficient from Jong et al 1999
     *
     * @return Velocity coefficient as double
     */
    double getVelocityCoefficient() {
        return this->av;
    }
    /**
     * Sets the velocity coefficient from Jong et al 1999
     *
     * @param vel as double
     */
    void setVelocityCoefficient(double vel) {
        this->av = vel;
    }

    /**
     * Returns the velocity squared coefficient from Jong et al 1999
     *
     * @return Velocity squared coefficient as double
     */
    double getVelocitySquared() {
        return this->avsq;
    }
    /**
     * Sets the velocity squared coefficient from Jong et al 1999
     *
     * @param velsq as double
     */
    void setVelocitySquared(double velsq) {
        this->avsq = velsq;
    }

    /**
     * Returns the hourly travel cost of the vehicle
     *
     * @return Hourly travel cost as double
     */
    double getHourlyCost() {
        return this->travelPerHrCost;
    }
    /**
     * Sets the hourly travel cost of the vehicle
     *
     * @param travel as double
     */
    void setHourlyCost(double travel) {
        this->travelPerHrCost = travel;
    }
    // STATIC ROUTINES ////////////////////////////////////////////////////////

    // CALCULATION ROUTINES ///////////////////////////////////////////////////

private:
    CommodityPtr fuel;          /**< Type of fuel used by the vehicle */
    std::string name;           /**< Vehicle name */
    double averageWidth;        /**< Average vehicle width (m) */
    double averageLength;       /**< Average vehicle length (m) */
    double trafficProportion;   /**< Proportion of traffic */
    double maxLoad;             /**< Maximum carrying capacity (tonne) */
    double aConst;              /**< 'a' constant (see Jong et al 1999) */
    double agr;                 /**< Grade coefficient (see Jong et al 1999) */
    double av;                  /**< Velocity coefficient (see Jong et al 1999) */
    double avsq;                /**< Velocity squared coefficient (see Jong et al 1999) */
    double travelPerHrCost;     /**< Cost per engine hour */
};

#endif
