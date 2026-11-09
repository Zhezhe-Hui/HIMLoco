#pragma once
#include <cmath>

// Independent of DDS: the bridge calls this at 50Hz even if Python stalls.
struct BridgeSafety {
    bool latched = false;
    double max_vx = 0.20;
    double max_wz = 0.35;
    bool command_valid(double vx, double wz) const {
        return std::isfinite(vx) && std::isfinite(wz) && vx >= 0 && vx <= max_vx && std::abs(wz) <= max_wz;
    }
    bool update(double state_age, double command_age, double roll, double pitch, double height) {
        if (!std::isfinite(state_age) || !std::isfinite(command_age) ||
            state_age > 0.25 || command_age > 0.30 ||
            !std::isfinite(roll) || !std::isfinite(pitch) || !std::isfinite(height) ||
            std::abs(roll) > 0.45 || std::abs(pitch) > 0.45 || height < 0.20 || height > 0.60)
            latched = true;
        return !latched;
    }
};
