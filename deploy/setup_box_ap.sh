#!/usr/bin/env bash
# Run on the box as root AFTER compiling the pinned driver. Read AP password
# from stdin; do not put credentials in source or operation records.
set -euo pipefail
[[ $EUID -eq 0 ]] || { echo 'Run as root on the box.' >&2; exit 1; }
source_dir=/home/unitree/himloco-network/8821cu-bda65aac
read -r ap_password
[[ ${#ap_password} -ge 8 && ${#ap_password} -le 63 ]]
test -f "$source_dir/8821cu.ko"
module_dir="/lib/modules/$(uname -r)/updates/himloco"
install -d "$module_dir"
install -m 644 "$source_dir/8821cu.ko" "$module_dir/8821cu.ko"
depmod -a
printf '%s\n' 'options 8821cu rtw_power_mgnt=0 rtw_enusbss=0' > /etc/modprobe.d/go2-8821cu.conf
printf '%s\n' '8821cu' > /etc/modules-load.d/go2-8821cu.conf
modprobe 8821cu
nmcli radio wifi on
for attempt in {1..15}; do
  wifi_device=$(nmcli -t -f DEVICE,TYPE device status | awk -F: '$2=="wifi" {print $1; exit}')
  [[ -n "$wifi_device" ]] && break
  sleep 1
done
test -n "$wifi_device"
nmcli -g WIFI-PROPERTIES.AP device show "$wifi_device" | grep -qx yes
if ! nmcli -g NAME connection show | grep -qx go2-ours-ap; then
  nmcli connection add type wifi ifname "$wifi_device" con-name go2-ours-ap ssid GO2-OURS
fi
nmcli connection modify go2-ours-ap connection.interface-name "$wifi_device" \
  connection.autoconnect yes connection.autoconnect-priority 100 \
  802-11-wireless.mode ap 802-11-wireless.band bg 802-11-wireless.channel 6 \
  802-11-wireless.powersave 2 wifi-sec.key-mgmt wpa-psk wifi-sec.proto rsn \
  wifi-sec.pairwise ccmp wifi-sec.group ccmp wifi-sec.psk "$ap_password" \
  ipv4.method shared ipv4.addresses 192.168.50.1/24 ipv4.never-default yes ipv6.method disabled
unset ap_password
nmcli connection up go2-ours-ap
nmcli -f GENERAL.DEVICE,GENERAL.STATE,IP4.ADDRESS device show "$wifi_device"
