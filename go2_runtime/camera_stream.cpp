// Read-only D435i capture. Binary stdout is transported through SSH.
#include <librealsense2/rs.hpp>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <csignal>
#include <chrono>
volatile std::sig_atomic_t running = 1;
void stop(int) { running = 0; }
template<class T> void send(const T& v) {
    if (std::fwrite(&v, sizeof(v), 1, stdout) != 1) throw std::runtime_error("stream closed");
}
int main() {
    std::signal(SIGTERM, stop); std::signal(SIGINT, stop);
    try {
        rs2::pipeline pipe; rs2::config cfg;
        cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 15);
        cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_RGB8, 15);
        pipe.start(cfg);
        unsigned long long previous = 0;
        while (running) {
            auto frames = pipe.wait_for_frames(5000);
            auto depth = frames.get_depth_frame(); auto rgb = frames.get_color_frame();
            // A synchronized frameset can reuse the preceding depth frame when
            // only RGB has advanced. Do not publish old depth as a fresh input.
            if (depth.get_frame_number() <= previous) continue;
            previous = depth.get_frame_number();
            auto k = depth.get_profile().as<rs2::video_stream_profile>().get_intrinsics();
            // Explicit little-endian fields, supported on x86_64 and Jetson aarch64.
            send(uint32_t(0x48494d32)); send(uint32_t(depth.get_width()));
            send(uint32_t(depth.get_height())); send(depth.get_frame_number());
            send(double(depth.get_timestamp()));
            send(std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count());
            send(k.fx); send(k.fy); send(k.ppx); send(k.ppy); send(depth.get_units());
            auto n = size_t(depth.get_width()) * depth.get_height();
            if (std::fwrite(depth.get_data(), 2, n, stdout) != n ||
                std::fwrite(rgb.get_data(), 3, n, stdout) != n || std::fflush(stdout))
                break;
        }
        pipe.stop();
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
