#ifndef DESIGNPARAMETERS_H
#define DESIGNPARAMETERS_H

class DesignParameters;
typedef std::shared_ptr<DesignParameters> DesignParametersPtr;

/**
 * Class for managing road design parameters
 */
class DesignParameters : public std::enable_shared_from_this<DesignParameters> {
	
public:
    // CONSTRUCTORS AND DESTRUCTORS ///////////////////////////////////////////

    /**
     * Constructor I
     *
     * Constructs a %DesignParams object with default values.
     */
    DesignParameters();

    /**
     * Constructor II
     *
     * Constructs a %DesignParams object with assigned values.
     */
    DesignParameters(double desVel, double sx, double sy, double ex, double ey,
            double mg, double mse, double rw, double rt, double dr,
            unsigned long ip, double sl, double cr, double fr, double bf,
            double bw, double bh, double tf, double tw, double td, double cpsm,
            double air, double noise, double water, double oil, double land,
            double chem, bool sp);

    /**
     * Destructor
     */
    ~DesignParameters();

    // ACCESSORS //////////////////////////////////////////////////////////////

    /**
     * Returns whether the road includes spiral transitions
     *
     * @return Spiral as bool
     */
    bool getSpiral() {
        return this->spiral;
    }
    /**
     * Sets whether the road includes spiral transitions
     *
     * @param spiral as bool
     */
    void setSpiral(bool spiral) {
        this->spiral = spiral;
    }

    /**
     * Returns the design road velocity
     *
     * @return Velocity as double
     */
    double getDesignVelocity() {
        return this->designVel;
    }
    /**
     * Sets the design road velocity
     *
     * @param vel as double
     */
    void setDesignVelocity(double vel) {
        this->designVel = vel;
    }

    /**
     * Returns the x coordinate of the start point
     *
     * @return Start X as double
     */
    double getStartX() {
        return this->startX;
    }
    /**
     * Sets the x coordinate of the start point
     *
     * @param sx as double
     */
    void setStartX(double sx) {
        this->startX = sx;
    }

    /**
     * Returns the Y coordinate of the start point.
     *
     * @return Start Y as double
     */
    double getStartY() {
        return this->startY;
    }
    /**
     * Sets the Y coordinate of the start point
     *
     * @param sy as double
     */
    void setStartY(double sy) {
        this->startY = sy;
    }

    /**
     * Returns the z coordinate of the start point
     *
     * @return Start Z as double
     */
    double getStartZ() {
        return this->startZ;
    }
    /**
     * Sets the z coordinate of the start point
     *
     * @param sz as double
     */
    void setStartZ(double sz) {
        this->startZ = sz;
    }

    /**
     * Return the X coordinate of the end point
     *
     * @return End X as double
     */
    double getEndX() {
        return this->endX;
    }
    /**
     * Sets the X coordinate of the end point
     *
     * @param ex as double
     */
    void setEndX(double ex) {
        this->endX = ex;
    }

    /**
     * Returns the Y coordinate of the end point
     *
     * @return End Y as double
     */
    double getEndY() {
        return this->endY;
    }
    /**
     * Sets the Y coordinate of the end point
     *
     * @param ey as double
     */
    void setEndY(double ey) {
        this->endY = ey;
    }

    /**
     * Return the Z coordinate of the end point
     *
     * @return End Z as double
     */
    double getEndZ() {
        return this->endZ;
    }
    /**
     * Sets the Z coordinate of the end point
     *
     * @param ez as double
     */
    void setEndZ(double ez) {
        this->endZ = ez;
    }

    /**
     * Returns the maximum grade
     *
     * @return Max grade as double
     */
    double getMaxGrade() {
        return this->maxGrade;
    }
    /**
     * Sets the maximum grade
     *
     * @param mg as double
     */
    void setMaxGrade(double mg) {
        this->maxGrade = mg;
    }

    /**
     * Returns the maximum superelevation
     *
     * @return Max superelevation as double
     */
    double getMaxSE() {
        return this->maxSE;
    }
    /**
     * Sets the maximum superelevation
     *
     * @param mse as double
     */
    void setMaxSE(double mse) {
        this->maxSE = mse;
    }

    /**
     * Returns the design road width
     *
     * @return Design road width as double
     */
    double getRoadWidth() {
        return this->roadWidth;
    }
    /**
     * Sets the design road width
     *
     * @param rw as double
     */
    void setRoadWidth(double rw) {
        this->roadWidth = rw;
    }

    /**
     * Returns the reaction time
     *
     * @return Reaction time as double
     */
    double getReactionTime() {
        return this->reactionTime;
    }
    /**
     * Sets the reaction time
     *
     * @param rt as double
     */
    void setReactionTime(double rt) {
        this->reactionTime = rt;
    }

    /**
     * Returns the deceleration rate
     *
     * @return Deceleration rate as double
     */
    double getDeccelRate() {
        return this->deccelRate;
    }
    /**
     * Sets the deceleration rate
     *
     * @param dr as double
     */
    void setDeccelRate(double dr) {
        this->deccelRate = dr;
    }

    /**
     * Returns the number of design intersection points
     *
     * @return Number of intersection points as unsigned long
     */
    unsigned long getIntersectionPoints() {
        return this->intersectPoints;
    }
    /**
     * Sets the number of design intersection points
     *
     * @param ip as unsigned long
     */
    void setIntersectionPoints(unsigned long ip) {
        this->intersectPoints = ip;
    }

    /**
     * Returns the distance between design stations
     *
     * @return Segment length as double
     */
    double getSegmentLength() {
        return this->segmentLength;
    }
    /**
     * Sets the distance between design stations
     *
     * @param sl as double
     */
    void setSegmentLength(double sl) {
        this->segmentLength = sl;
    }

    /**
     * Returns the angle of repose for a cut
     *
     * @return Angle as double
     */
    double getCutRep() {
        return this->cutRep;
    }
    /**
     * Sets the angle of repose for a cut
     *
     * @param cr as double
     */
    void setCutRep(double cr) {
        this->cutRep = cr;
    }

    /**
     * Returns the angle of repose for a fill
     *
     * @return Angle as double
     */
    double getFillRep() {
        return this->fillRep;
    }
    /**
     * Sets the angle of repose for a fill
     *
     * @param
     */
    void setFillRep(double fr) {
        this->fillRep = fr;
    }

    /**
     * Returns the fixed cost component for a bridge
     *
     * @return Cost as double
     */
    double getBridgeFixed() {
        return this->bridgeFixed;
    }
    /**
     * Sets the fixed cost component for a bridge
     *
     * @param bf as double
     */
    void setBridgeFixed(double bf) {
        this->bridgeFixed = bf;
    }

    /**
     * Returns the width-based component of bridge cost
     *
     * @return Cost as double
     */
    double getBridgeWidth() {
        return this->bridgeWidth;
    }
    /**
     * Sets the width-based component of bridge cost
     *
     * @param bw as double
     */
    void setBridgeWidth(double bw) {
        this->bridgeWidth = bw;
    }

    /**
     * Returns the height-based component of bridge cost
     *
     * @return Cost as double
     */
    double getBridgeHeight() {
        return this->bridgeHeight;
    }
    /**
     * Sets the height-based component of bridge cost
     *
     * @params bh as double
     */
    void setBridgeHeight(double bh) {
        this->bridgeHeight = bh;
    }

    /**
     * Returns the fixed cost component for a tunnel
     *
     * @return Cost as double
     */
    double getTunnelFixed() {
        return this->tunnelFixed;
    }
    /**
     * Sets the fixed cost component for a tunnel
     *
     * @param tf as double
     */
    void setTunnelFixed(double tf) {
        this->tunnelFixed = tf;
    }

    /**
     * Returns the width-based cost component for a tunnel
     *
     * @return Cost as double
     */
    double getTunnelWidth() {
        return this->tunnelWidth;
    }
    /**
     * Sets the width-based cost component for a tunnel
     *
     * @params tw as double
     */
    void setTunnelWidth(double tw) {
        this->tunnelWidth = tw;
    }

    /**
     * Returns the depth-based cost component for a tunnel
     *
     * @return Cost as double
     */
    double getTunnelDepth() {
        return this->tunnelDepth;
    }
    /**
     * Sets the depth-based cost component for a tunnel
     *
     * @params td as double
     */
    void setTunnelDepth(double td) {
        this->tunnelDepth = td;
    }

    /**
     * Returns the per square metre costs
     *
     * @return Cost as double
     */
    double getCostPerSM() {
        return this->costPerSM;
    }
    /**
     * Sets the per square metre costs
     *
     * @params cpsm as double
     */
    void setCostPerSM(double cpsm) {
        this->costPerSM = cpsm;
    }

    /**
     * Returns the air pollution cost per m per vehicle
     *
     * @return Cost as double
     */
    double getAirPollutionCost() {
        return this->airPollution;
    }
    /**
     * Sets the air pollution cost per m per vehicle
     *
     * @params air as double
     */
    void setAirPollutionCost(double air) {
        this->airPollution = air;
    }

    /**
     * Returns the noise pollution cost per m per vehicle
     *
     * @return Cost as double
     */
    double getNoisePollutionCost() {
        return this->noisePollution;
    }
    /**
     * Sets the noise pollution cost per m per vehicle
     *
     * @params noise as double
     */
    void setNoisePollutionCost(double noise) {
        this->noisePollution = noise;
    }

    /**
     * Returns the water pollution cost per m per vehicle
     *
     * @return Cost as double
     */
    double getWaterPollutionCost() {
        return this->waterPollution;
    }
    /**
     * Sets the water pollution cost per m per vehicle
     *
     * @params water as double
     */
    void setWaterPollutionCost(double water) {
        this->waterPollution = water;
    }

    /**
     * Returns the oil extraction cost per m per vehicle
     *
     * @return Cost as double
     */
    double getOilExtractionCost() {
        return this->oilExtraction;
    }
    /**
     * Sets the oil extraction cost per m per vehicle
     *
     * @params oil as double
     */
    void setOilExtractionCost(double oil) {
        this->oilExtraction = oil;
    }

    /**
     * Returns the land use cost per m per vehicle
     *
     * @return Cost as double
     */
    double getLandUseCost() {
        return this->landUse;
    }
    /**
     * Sets the land use cost per m per vehicle
     *
     * @params land as double
     */
    void setLandUseCost(double land) {
        this->landUse = land;
    }

    /**
     * Returns the solid and chemical waste cost per m per vehicle
     *
     * @return Cost as double
     */
    double getSolidChemWasteCost() {
        return this->solidChemWaste;
    }
    /**
     * Sets the soild and chemical waste cost per m per vehicle
     *
     * @params solidchem as double
     */
    void setSolidChemWasteCost(double solidchem) {
        this->solidChemWaste = solidchem;
    }

    // STATIC ROUTINES ////////////////////////////////////////////////////////

    // CALCULATION ROUTINES ///////////////////////////////////////////////////

private:
    bool spiral;                    /**< Road design include spiral transitions */
    double designVel;               /**< Design velocity (m/s) */
    double startX;                  /**< X coordinate of start (m) */
    double startY;                  /**< Y coordinate of start (m) */
    double startZ;                  /**< Z coordinate of start (m) */
    double endX;                    /**< X coordinate of end (m) */
    double endY;                    /**< Y coordinate of end (m) */
    double endZ;                    /**< Z coordinate of end (m) */
    double maxGrade;                /**< Max grade (radians) */
    double maxSE;                   /**< Max superelevation (radians) */
    double roadWidth;               /**< Design road width (m) */
    double reactionTime;            /**< Reaction time (s) */
    double deccelRate;              /**< Deceleration rate (m/s^2) */
    unsigned long intersectPoints;  /**< Road design intersection points */
    double segmentLength;           /**< Distance between design stations (m) */
    double cutRep;                  /**< Angle of repose, cut (radians) */
    double fillRep;                 /**< Angle of repose, fill (radians) */
    double bridgeFixed;             /**< Fixed component of bridge cost ($) */
    double bridgeWidth;             /**< Component of bridge cost from width ($/m) */
    double bridgeHeight;            /**< Component of bridge cost from height ($/m) */
    double tunnelFixed;             /**< Fixed component of tunnel cost ($) */
    double tunnelWidth;             /**< Component of tunnel cost from width ($/m) */
    double tunnelDepth;             /**< Component of tunnel cost from depth ($/m) */
    double costPerSM;               /**< Cost per square meter (pavement etc.) */
    double airPollution;            /**< Air pollution cost ($ per m per vehicle (TO REMOVE) */
    double noisePollution;          /**< Noise pollution cost ($ per m per vehicle (TO REMOVE) */
    double waterPollution;          /**< Water pollution cost ($ per m per vehicle (TO REMOVE) */
    double oilExtraction;           /**< Oil extraction etc. cost ($ per m per vehicle (TO REMOVE) */
    double landUse;                 /**< Land use cost ($ per m per vehicle (TO REMOVE) */
    double solidChemWaste;          /**< Solid and chemical waste cost ($ per m per vehicle (TO REMOVE) */
};

#endif
