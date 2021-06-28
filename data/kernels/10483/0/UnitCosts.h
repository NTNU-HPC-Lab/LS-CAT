#ifndef UNITCOSTS_H
#define UNITCOSTS_H

class UnitCosts;
typedef std::shared_ptr<UnitCosts> UnitCostsPtr;

/**
 * Class for managing unit costs
 */
class UnitCosts : public std::enable_shared_from_this<UnitCosts> {

public:
    // CONSTRUCTORS AND DESTRUCTORS ///////////////////////////////////////////

    /**
     * Constructor I
     */
    UnitCosts();

    /**
     * Constructor II
     *
     * Constructs a %UnitCosts object with assigned values.
     */
    UnitCosts(double acc, double air, double noise, double water, double oil,
            double land, double chem);
    /**
     * Destructor
     */
    ~UnitCosts();

    // ACCESSORS //////////////////////////////////////////////////////////////

    /**
     * Returns the accident cost per vehicle
     *
     * @return Per vehicle accidence cost as double
     */
    double getAccidentCost() {
        return this->perAccident;
    }
    /**
     * Sets the per vehicle accident cost
     *
     * @param accident as double
     */
    void setAccidentCost(double acc) {
        this->perAccident = acc;
    }

    /**
     * Returns the air pollution cost per vehicle km
     *
     * @return Air pollution cost per vehiclekm as double
     */
    double getAirPollution() {
        return this->airPollution;
    }
    /**
     * Sets the air pollution cost per vehicle km
     *
     * @param airp as double
     */
    void setAirPollution(double airp) {
        this->airPollution = airp;
    }

    /**
     * Returns the noise pollution cost per vehicle km.
     *
     * @return Noise pollution cost as double
     */
    double getNoisePollution() {
        return this->noisePollution;
    }
    /**
     * Sets the noise pollution cost per vehicle km.
     *
     * @param np as double
     */
    void setNoisePollution(double np) {
        this->noisePollution = np;
    }

    /**
     * Returns the water pollution cost per vehicle km.
     *
     * @return Water pollution cost as double
     */
    double getWaterPollution() {
        return this->waterPollution;
    }
    /**
     * Sets the water pollution cost per vehicle km.
     *
     * @param wp cost as double
     */
    void setWaterPollution(double wp) {
        this->waterPollution = wp;
    }

    /**
     * Returns the oil extraction costs per km.
     *
     * @return Oil extraction cost as double
     */
    double getOilExtraction() {
        return this->oilExtractDistUse;
    }
    /**
     * Sets the oil extraction costs per km.
     *
     * @param oil as double
     */
    void setOilExtraction(double oil) {
        this->oilExtractDistUse = oil;
    }

    /**
     * Returns land use costs per vehicle km.
     *
     * @return Land use costs as double
     */
    double getLandUse() {
        return this->landUse;
    }
    /**
     * Sets land use costs per vehicle km.
     *
     * @param landUse as double
     */
    void setLandUse(double landUse) {
        this->landUse = landUse;
    }

    /**
     * Returns the chemical waste disposal cost per vehicle km.
     *
     * @return Chemical waste cost as double
     */
    double getSolidChemWaste() {
        return this->solidChemWaste;
    }
    /**
     * Sets the chemical waste disposal cost per vehicle km.
     *
     * @param chem as double
     */
    void setSolidChemWaste(double chem) {
        this->solidChemWaste = chem;
    }

    // STATIC ROUTINES ////////////////////////////////////////////////////////

    // CALCULATION ROUTINES ///////////////////////////////////////////////////

private:
    double perAccident;         /**< Cost per accident */
    double airPollution;        /**< Air pollution cost (c per km) */
    double noisePollution;      /**< Noise pollution cost (c per km) */
    double waterPollution;      /**< Water pollution cost (c per km) */
    double oilExtractDistUse;   /**< Oil extraction cost (c per km) */
    double landUse;             /**< Land use cost (c per km) */
    double solidChemWaste;      /**< Solid and chemical disposal cost (c per km) */
};

#endif
