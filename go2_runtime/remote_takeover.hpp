#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <array>
#include <limits>

// Input timestamps are the actual packet receipt times, never refreshed by polls.
struct RemoteTakeover {
    bool seen=false, valid=false, active=false, armed=false, latched=false;
    double received=0, neutral_since=-1,last_activity=-1;
    std::uint16_t keys=0;
    int trip=0; // 1 operator input, 2 packet loss, 3 malformed input
    float axes[4]={0,0,0,0};
    // Go2 packet layout from the official state_machine/gamepad.hpp.
    // This robot's observed packet header is 0x55 0x51; reject absent/unknown packets.
    void sample_raw(double now,const std::array<std::uint8_t,40>& bytes) {
        if(bytes[0]!=0x55 || bytes[1]!=0x51) {
            sample(now,0,std::numeric_limits<float>::quiet_NaN(),0,0,0);return;
        }
        float lx,ly,rx,ry;
        std::memcpy(&lx,bytes.data()+4,4);std::memcpy(&rx,bytes.data()+8,4);
        std::memcpy(&ry,bytes.data()+12,4);std::memcpy(&ly,bytes.data()+20,4);
        std::uint16_t buttons=std::uint16_t(bytes[2]) | (std::uint16_t(bytes[3])<<8);
        sample(now,buttons,lx,ly,rx,ry);
    }
    void sample(double now, std::uint16_t buttons, float lx, float ly, float rx, float ry) {
        if(seen && (now<received || now-received>.5))neutral_since=-1;
        seen=true; received=now; keys=buttons;
        float input[4]={lx,ly,rx,ry};valid=true;float magnitude=0;
        for(int i=0;i<4;++i) {
            if(!std::isfinite(input[i]) || std::abs(input[i])>1.01f) valid=false;
            axes[i]=std::isfinite(input[i])?input[i]:0;
            magnitude=std::max(magnitude,std::abs(axes[i]));
        }
        active=!valid || keys!=0 || magnitude>0.20f;
        if(active) {if(valid)last_activity=now;neutral_since=-1;if(armed && !latched){latched=true;trip=valid?1:3;}}
        else if(neutral_since<0)neutral_since=now;
    }
    bool fresh(double now) const {return seen && valid && now>=received && now-received<=.5;}
    bool ready(double now) const {return fresh(now) && !active && !latched && neutral_since>=0 && now-neutral_since>=.5;}
    bool arm(double now) {if(!ready(now))return false;armed=true;return true;}
    bool permits(double now) {if(armed && !fresh(now) && !latched){latched=true;trip=2;}return armed && !latched;}
};
