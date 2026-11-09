// Read-only service/mode query. No movement, service switch, or mode selection.
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>
#include <unitree/robot/go2/robot_state/robot_state_client.hpp>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    unitree::robot::ChannelFactory::Instance()->Init(0, argv[1]);
    unitree::robot::b2::MotionSwitcherClient modes;
    modes.SetTimeout(3.f); modes.Init();
    std::string form, name;
    auto mode_result = modes.CheckMode(form, name);
    std::cout << "CheckMode result=" << mode_result << " form=" << form << " name=" << name << '\n';
    unitree::robot::go2::RobotStateClient services;
    services.SetTimeout(3.f); services.Init();
    std::vector<unitree::robot::go2::ServiceState> list;
    auto service_result = services.ServiceList(list);
    std::cout << "ServiceList result=" << service_result << '\n';
    for (const auto& item : list)
        std::cout << item.name << " status=" << item.status << " protect=" << item.protect << '\n';
    return mode_result || service_result ? 1 : 0;
}
