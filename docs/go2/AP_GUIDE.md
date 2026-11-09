# GO2盒子独立热点（2026-11-06）

新网卡USB ID为0bda:c811。驱动源码设备表将该ID标为8811CU，使用8821cu驱动。已在盒子5.10.104-tegra/aarch64内核编译、加载，NetworkManager报告AP=yes、2.4/5GHz支持；当前采用2.4GHz频道6方便客户端兼容。

## 连接

2026-11-09更新：这台电脑的“有线连接 1”已增加持久主机路由 `192.168.50.1/32 via 192.168.123.18`，因此有线/无线都可打开 **http://192.168.50.1:8765/**。有线连接存在时走有线；断开电脑网线后，连接GO2-OURS则走无线。盒子eth0仍为192.168.123.18，未将两个接口配置到同一子网。此路由只配置在本电脑；其他电脑用网线时仍访问192.168.123.18，或另加同样路由；手机连接GO2-OURS直接访问192.168.50.1。

本电脑复现配置（已执行，不必重复）：

```bash
nmcli connection modify uuid 2e40e687-e023-3929-b89d-241c09da8653 +ipv4.routes '192.168.50.1/32 192.168.123.18'
nmcli device reapply enx00e04c584b68
```

撤销仅本次路由：把上面`+ipv4.routes`改为`-ipv4.routes`并reapply。不要清空其他路由。配置依赖盒子热点接口保持192.168.50.1（即使经有线访问），若主动关闭AP导致该地址撤销，仍可用有线192.168.123.18作为备用。

1. 机器狗开机，盒子上插好本次网卡，等待盒子启动。
2. 电脑或手机连接 **GO2-OURS**。热点连接密码已在本次对话给出，不写入项目记录。
3. 打开 **http://192.168.50.1:8765/**，免网页登录。
4. SSH命令：`ssh unitree@192.168.50.1`。SSH密码与热点密码是两回事，未修改。

这是盒子发射的热点；机器狗原厂go2热点是另一个网络。新方案不依赖手机热点，网页固定地址不随客户端DHCP地址变化。有线备用仍为192.168.123.18，内部DDS仍走eth0。连接此热点不提供可用公网；网页与SSH局域网开发不需要公网。

## 配置位置与开机行为

- 源码：https://github.com/morrownr/8821cu-20210916 ，commit `bda65aac150d2cde0df9603206eec23a6f3b77c4`。
- 盒子源码：`/home/unitree/himloco-network/8821cu-bda65aac`，编译日志`/home/unitree/himloco-network/build-8821cu.log`。
- 模块：`/lib/modules/5.10.104-tegra/updates/himloco/8821cu.ko`，通过depmod登记USB ID。
- `/etc/modprobe.d/go2-8821cu.conf`：关闭驱动省电及USB自动挂起。
- `/etc/modules-load.d/go2-8821cu.conf`：启动加载8821cu。
- NetworkManager配置`go2-ours-ap`：AP模式、WPA2/CCMP、自动连接、优先级100、192.168.50.1/24、IPv4 shared、无默认路由、IPv6关闭。DHCP由NetworkManager启动dnsmasq提供。
- 网页仍为既有`go2-ours-web.service`，未部署新算法或自动创建运动任务。密码仅由NetworkManager保存。
- 本机保留`setup_box_ap.sh`，需先编译模块再在盒子以root运行，并从标准输入读取热点密码。不要在电脑上运行。
- 当前为匹配内核的手动模块安装，未注册DKMS；升级盒子内核后需为新内核重新编译安装。无须为本次接入升级内核。

## 实际验收与边界

- 新驱动编译/加载成功，AP启动并可扫描。
- 电脑真实连接GO2-OURS，获得192.168.50.123/24；电脑有线接口断开后，SO_BINDTODEVICE固定无线接口访问192.168.50.1:8765成功。
- HTML、RGB/深度预览通过，帧371到379持续增长，约14.99fps；本体输入新鲜、桥写入0，无运动。
- 旧只读状态桥因先前解析错误退出读取，单次connect未重建仍存活的桥；执行只读disconnect/connect后恢复。未修改算法，也未保证该启动时异常不再出现。异常复现可在无任务状态重新连接并检查日志。
- 电脑已恢复XAUAT与原有线连接；盒子AP持续运行。电脑现需自行切到GO2-OURS才可访问新无线地址，不能在校园网直接访问它。
- 热点/模块/网页自启动配置已设置，但本轮未整狗重启，未做手机兼容、多客户端持续负载、距离与实走验收。未触发站立或运动。
- 盒子时钟仍为2025-09-29，实际本次操作日期2026-11-06；编译有时间偏差警告，最终模块编译加载并实际传输通过，实验日志日期仍需校时。
- 证据：`../web_console/data/ap-wireless-verification-20261106.json`。

## 排查和恢复

先接网线后：

```bash
ssh unitree@192.168.123.18
nmcli device status
nmcli connection show --active
ip -4 -br address
systemctl status go2-ours-web.service
journalctl -u NetworkManager -n 40 --no-pager
```

仅停用本项目热点：`sudo nmcli connection down go2-ours-ap`。停用自动启动：`sudo nmcli connection modify go2-ours-ap connection.autoconnect no`。重新启用：`sudo nmcli connection modify go2-ours-ap connection.autoconnect yes`，再`sudo nmcli connection up go2-ours-ap`。不改原厂热点或eth0。

如果要恢复旧手机热点方案，需要插回兼容网卡、让手机开启已保存的2.4GHz热点，并主动启用himloco-phone-wifi；不要在唯一无线SSH通道上盲目切换，优先用网线操作。
