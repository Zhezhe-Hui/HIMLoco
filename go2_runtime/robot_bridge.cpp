// No publishers/clients in default mode. --execute enables bounded velocity only.
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/robot/go2/sport/sport_client.hpp>
#include <unitree/idl/go2/SportModeState_.hpp>
#include <unitree/idl/go2/LowState_.hpp>
#include <chrono>
#include <cmath>
#include <csignal>
#include <iostream>
#include <mutex>
#include <poll.h>
#include <sstream>
#include <thread>
#include <unistd.h>
#include "bridge_safety.hpp"
#include "remote_takeover.hpp"
using Clock = std::chrono::steady_clock;
std::mutex lock;
unitree_go::msg::dds_::SportModeState_ state;
Clock::time_point received;
bool have_state = false;
RemoteTakeover remote;
double seconds() {return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();}
void remote_callback(const void* p) {
    const auto& r=*static_cast<const unitree_go::msg::dds_::LowState_*>(p);
    std::lock_guard<std::mutex> guard(lock);
    remote.sample_raw(seconds(),r.wireless_remote());
}
volatile std::sig_atomic_t running = 1;
void stop(int) { running = 0; }
void callback(const void* p) {
    std::lock_guard<std::mutex> guard(lock);
    state = *static_cast<const unitree_go::msg::dds_::SportModeState_*>(p);
    received = Clock::now(); have_state = true;
}
int main(int argc, char** argv) {
    if (argc < 2 || argc > 3 || (argc == 3 && std::string(argv[2]) != "--execute" && std::string(argv[2]) != "--execute-calibration" && std::string(argv[2]) != "--execute-takeover")) return 2;
    const bool execute = argc == 3;
    const bool calibration = execute && std::string(argv[2]) == "--execute-calibration";
    const bool takeover = execute && std::string(argv[2]) == "--execute-takeover";
    std::signal(SIGINT, stop); std::signal(SIGTERM, stop); std::signal(SIGPIPE, stop);
    unitree::robot::ChannelFactory::Instance()->Init(0, argv[1]);
    unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::SportModeState_> sub("rt/sportmodestate");
    sub.InitChannel(callback, 1);
    unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::LowState_> rcsub("rt/lowstate");
    rcsub.InitChannel(remote_callback,1);
    std::unique_ptr<unitree::robot::go2::SportClient> client;
    if (execute) {
        client.reset(new unitree::robot::go2::SportClient);
        client->SetTimeout(0.1f); client->Init();
    }
    auto command_time = Clock::now(); bool started = false;
    BridgeSafety safety;
    if (calibration || takeover) { safety.max_vx = 0.30; safety.max_wz = 0; }
    auto calibration_start = Clock::now(); bool calibration_started = false;
    int move_result = 0; unsigned long move_calls = 0;
    double sent_vx = 0, sent_wz = 0;
    double vx = 0, wz = 0; std::string input;
    double origin_x=0,origin_y=0;
    while (running) {
        pollfd fd{STDIN_FILENO, POLLIN, 0};
        poll(&fd, 1, 0);
        if (fd.revents & (POLLIN | POLLHUP)) {
            char buf[256]; auto n = read(STDIN_FILENO, buf, sizeof(buf));
            if (n <= 0) break;
            input.append(buf, size_t(n));
            if (input.size() > 4096) break;
            size_t end;
            while ((end = input.find('\n')) != std::string::npos) {
                std::istringstream line(input.substr(0, end)); input.erase(0, end + 1);
                double a, b; std::string extra;
                if (!(line >> a >> b) || (line >> extra) || !safety.command_valid(a, b)) {
                    safety.latched = true; continue;
                }
                if (takeover && !started) {
                    std::lock_guard<std::mutex> guard(lock);
                    if (!have_state || !remote.arm(seconds())) {
                        std::cout<<"{\"bridge_event\":\"remote_not_ready\"}"<<std::endl;
                        running=0;break;
                    }
                    origin_x=state.position()[0];origin_y=state.position()[1];
                }
                vx = a; wz = b; started = true; command_time = Clock::now();
                if (!calibration_started) { calibration_start = command_time; calibration_started = true; }
            }
        }
        if (!running) break;
        double age = 1e9; int mode = -1; float roll = 0, pitch = 0, height = 0;
        bool remote_allowed=true;double displacement=0;
        {
            std::lock_guard<std::mutex> guard(lock);
            if(takeover && started)remote_allowed=remote.permits(seconds());
            if (have_state) {
                displacement=std::hypot(state.position()[0]-origin_x,state.position()[1]-origin_y);
                age = std::chrono::duration<double>(Clock::now() - received).count();
                mode = state.mode(); roll = state.imu_state().rpy()[0]; pitch = state.imu_state().rpy()[1];
                height = state.position()[2];
                // Emit at the existing 50Hz bridge loop, retaining the true DDS age.
                // Never serialize NaN/Inf as invalid JSON during robot startup.
                if (std::isfinite(state.position()[0]) && std::isfinite(state.position()[1]) &&
                    std::isfinite(height) && std::isfinite(roll) && std::isfinite(pitch) &&
                    std::isfinite(state.imu_state().rpy()[2]) &&
                    std::isfinite(state.velocity()[0]) && std::isfinite(state.velocity()[1]) &&
                    std::isfinite(state.yaw_speed())) {
                    const auto& p = state.position(); const auto& r = state.imu_state().rpy();
                    std::cout << "{\"x\":" << p[0] << ",\"y\":" << p[1]
                        << ",\"z\":" << p[2] << ",\"roll\":" << r[0]
                        << ",\"pitch\":" << r[1] << ",\"yaw\":" << r[2]
                        << ",\"mode\":" << mode << ",\"state_age\":" << age
                        << ",\"gait_type\":" << int(state.gait_type())
                        << ",\"vx\":" << state.velocity()[0] << ",\"vy\":" << state.velocity()[1]
                        << ",\"yaw_speed\":" << state.yaw_speed()
                        << ",\"latched\":" << (safety.latched ? "true" : "false")
                        << ",\"bridge_vx\":" << sent_vx << ",\"bridge_wz\":" << sent_wz
                        << ",\"sdk_move_calls\":" << move_calls
                        << ",\"remote_seen\":" << (remote.seen?"true":"false")
                        << ",\"remote_source\":\"lowstate.wireless_remote\""
                        << ",\"remote_activity_age\":" << (remote.last_activity>=0?seconds()-remote.last_activity:1e9)
                        << ",\"remote_valid\":" << (remote.valid?"true":"false")
                        << ",\"remote_age\":" << (remote.seen?seconds()-remote.received:1e9)
                        << ",\"remote_ready\":" << (remote.ready(seconds())?"true":"false")
                        << ",\"remote_active\":" << (remote.active?"true":"false")
                        << ",\"remote_keys\":" << remote.keys
                        << ",\"remote_axes\":[" << remote.axes[0]<<","<<remote.axes[1]<<","<<remote.axes[2]<<","<<remote.axes[3]<<"]"
                        << ",\"sdk_move_result\":" << (move_calls ? std::to_string(move_result) : "null") << "}" << std::endl;
                }
            }
        }
        if (client && started) {
            if(takeover && !remote_allowed) {
                std::lock_guard<std::mutex> guard(lock);
                std::cout<<"{\"bridge_event\":\""<<(remote.trip==1?"remote_takeover":remote.trip==2?"remote_signal_lost":"remote_invalid")<<"\"}"<<std::endl;break;
            }
            if ((calibration || takeover) && calibration_started &&
                std::chrono::duration<double>(Clock::now()-calibration_start).count() >= (takeover?8.0:3.0)) {
                std::cout<<"{\"bridge_event\":\"time_limit\"}"<<std::endl;break;
            }
            if(takeover && displacement>=2.0) {
                std::cout<<"{\"bridge_event\":\"distance_limit\"}"<<std::endl;break;
            }
            if (!safety.update(age, std::chrono::duration<double>(Clock::now()-command_time).count(),
                               roll, pitch, height)) { client->StopMove(); break; }
            if (vx == 0 && wz == 0) { sent_vx = 0; sent_wz = 0; }
            else {
                sent_vx += std::max(-0.005, std::min(0.005, vx-sent_vx));
                sent_wz += std::max(-0.014, std::min(0.014, wz-sent_wz));
            }
            auto result = client->Move(float(sent_vx), 0, float(sent_wz));
            move_result = result; ++move_calls;
            if (result != 0) { std::cerr << "Move failed: " << result << std::endl; break; }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    if (client) client->StopMove();
    return 0;
}
