#include "bridge_safety.hpp"
#include <cassert>
#include <limits>
int main() {
    BridgeSafety gate;
    assert(gate.command_valid(0.2, -0.35));
    assert(!gate.command_valid(0.21, 0));
    assert(!gate.command_valid(0.30, 0));
    BridgeSafety calibration;
    calibration.max_vx = 0.30; calibration.max_wz = 0;
    assert(calibration.command_valid(0.30, 0));
    assert(!calibration.command_valid(0.301, 0));
    assert(!calibration.command_valid(0.30, 0.01));
    assert(!calibration.update(0.26, 0, 0, 0, 0.31));
    assert(!gate.command_valid(0, std::numeric_limits<double>::quiet_NaN()));
    assert(gate.update(0.01, 0.1, 0.01, 0.02, 0.31));
    assert(!gate.update(0.01, 0.31, 0, 0, 0.31)); // lost Python heartbeat
    assert(!gate.update(0, 0, 0, 0, 0.31)); // cannot resume automatically
    BridgeSafety stale;
    assert(!stale.update(0.26, 0, 0, 0, 0.31));
    BridgeSafety lying;
    assert(!lying.update(0, 0, 0, 0, 0.05));
    BridgeSafety fall;
    assert(!fall.update(0, 0, 0.46, 0, 0.31));
}
