#include "remote_takeover.hpp"
#include <cassert>
#include <limits>
void neutral(RemoteTakeover& g) {g.sample(10,0,0,0,0,0);g.sample(10.3,0,0,0,0,0);g.sample(10.6,0,0,0,0,0);assert(g.arm(10.6));}
int main() {
    RemoteTakeover absent;assert(!absent.arm(10));
    RemoteTakeover held;held.sample(10,1,0,0,0,0);assert(!held.arm(11));
    RemoteTakeover stick;neutral(stick);assert(stick.permits(10.7));
    stick.sample(10.8,0,.3f,0,0,0);assert(!stick.permits(10.8));
    stick.sample(11,0,0,0,0,0);assert(!stick.arm(12));assert(!stick.permits(12));assert(stick.trip==1);
    RemoteTakeover button;neutral(button);button.sample(10.8,4,0,0,0,0);assert(!button.permits(10.8));
    RemoteTakeover lost;neutral(lost);assert(!lost.permits(11.2));
    lost.sample(11.3,0,0,0,0,0);assert(!lost.permits(11.3));
    RemoteTakeover invalid;neutral(invalid);invalid.sample(10.8,0,std::numeric_limits<float>::quiet_NaN(),0,0,0);assert(!invalid.permits(10.8));
    RemoteTakeover noise;neutral(noise);noise.sample(10.8,0,.05f,-.02f,0,0);assert(noise.permits(10.8));
    std::array<std::uint8_t,40> bytes{};
    RemoteTakeover raw;raw.sample_raw(10,bytes);assert(!raw.valid);
    bytes[0]=0x55;bytes[1]=0x51;raw.sample_raw(10.1,bytes);raw.sample_raw(10.4,bytes);raw.sample_raw(10.7,bytes);assert(raw.arm(10.7));
    float ly=.4f;std::memcpy(bytes.data()+20,&ly,4);raw.sample_raw(10.8,bytes);
    assert(raw.axes[1]==ly);assert(!raw.permits(10.8));assert(raw.trip==1);
}
