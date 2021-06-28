#ifndef TRAFFIC_H
#define TRAFFIC_H

class Vehicle;
typedef std::shared_ptr<Vehicle> VehiclePtr;

class Traffic;
typedef std::shared_ptr<Traffic> TrafficPtr;

/**
 * Class for managing traffic
 */
class Traffic : public std::enable_shared_from_this<Traffic> {

public:
    // CONSTRUCTORS AND DESTRUCTORS ///////////////////////////////////////////

    /**
     * Constructor I
     *
     * Constructs a blank %Traffic object
     */
    Traffic();

    /**
     * Constructor II
     *
     * Constructs a %Traffic object with assigned values.
     */
    Traffic(const std::vector<VehiclePtr>& vehicles, double peakProp, double d,
            double peak, double gr);

    /**
     * Destructor
     */
    ~Traffic();

    // ACCESSORS //////////////////////////////////////////////////////////////

    /**
     * Returns the vehicles
     *
     * @return Vehicles as std::vector<VehiclePtr>*
     */
    const std::vector<VehiclePtr>& getVehicles() {
        return this->vehicles;
    }
    /**
     * Sets the vehicles used
     *
     * @param vehicles as const std::vector<VehiclePtr>&
     */
    void setVehicles(const std::vector<VehiclePtr>& vehicles) {
        this->vehicles = vehicles;
    }

    /**
     * Returns the proportion of daily traffic during peak time
     *
     * @return Peak proportion as double
     */
    double getPeakProportion() {
        return this->peakProportion;
    }
    /**
     * Sets the proportion of daily traffic during peak time
     *
     * @param pp as double
     */
    void setPeakProportion(double pp) {
        this->peakProportion = pp;
    }

    /**
     * Returns the directionality of traffic during peak time toward A
     *
     * @return Peak directionality as double
     */
    double getDirectionality() {
        return this->directionality;
    }
    /**
     * Sets the directionality of traffic during peak time toward A
     *
     * @param dir as double
     */
    void setDirectionality(double dir) {
        this->directionality = dir;
    }

    /**
     * Returns the number of peak hours per day
     *
     * @return Peak hours as double
     */
    double getPeakHours() {
        return this->peakHours;
    }
    /**
     * Sets the number of peak hours per day
     *
     * @param hours as double
     */
    void setPeakHours(double hours) {
        this->peakHours = hours;
    }

    /**
     * Returns the annual traffic growth rate (\% p.a.)
     *
     * @return Growth rate as double
     */
    double getGR() {
        return this->growthRate;
    }
    /**
     * Sets the annual traffic growth rate
     *
     * @param gr as double
     */
    void setGR(double gr) {
        this->growthRate = gr;
    }

    // STATIC ROUTINES ////////////////////////////////////////////////////////

    // CALCULATION ROUTINES ///////////////////////////////////////////////////

private:
    std::vector<VehiclePtr> vehicles;	/**< Vehicles in traffic */
    double peakProportion;              /**< Proportion of daily traffic during peak times */
    double directionality;              /**< Peak time directionality toward A */
    double peakHours;                   /**< Peak hours in a day */
    double growthRate;                  /**< Annual traffic growth rate */
};

#endif
