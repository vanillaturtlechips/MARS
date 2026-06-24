#!/bin/bash
# MARS — ROS2 Jazzy + Nav2 + colcon 설치 (Ubuntu 24.04 / Noble)
# 사용법: bash deploy/runpod/setup_ros2.sh
set -e

echo "════════════════════════════════════════════"
echo " MARS — ROS2 Jazzy 설치"
echo "════════════════════════════════════════════"

# ── 0. OS 확인 (Jazzy = Noble 전용) ────────────────────────────
. /etc/os-release
if [ "$VERSION_CODENAME" != "noble" ]; then
    echo "[중단] ROS2 Jazzy는 Ubuntu 24.04(noble) 전용. 현재: $VERSION_CODENAME"
    exit 1
fi
echo "▶ OS: $PRETTY_NAME  (noble OK)"
echo "▶ root 디스크:"; df -h / | tail -1

# ── 1. locale ──────────────────────────────────────────────────
echo "[1/4] locale..."
apt-get update -q
apt-get install -y -q locales
locale-gen en_US en_US.UTF-8
update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

# ── 2. ROS2 apt 저장소 ─────────────────────────────────────────
echo "[2/4] ROS2 apt 저장소 등록..."
apt-get install -y -q curl gnupg lsb-release ca-certificates software-properties-common
add-apt-repository universe -y
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu noble main" \
    | tee /etc/apt/sources.list.d/ros2.list > /dev/null
apt-get update -q

# ── 3. ROS2 Jazzy + Nav2 + 빌드도구 ────────────────────────────
echo "[3/4] ros-jazzy-desktop + Nav2 + colcon (수 분 소요)..."
apt-get install -y \
    ros-jazzy-desktop \
    ros-jazzy-navigation2 \
    ros-jazzy-nav2-bringup \
    ros-jazzy-slam-toolbox \
    ros-jazzy-vision-msgs \
    ros-dev-tools \
    python3-colcon-common-extensions \
    python3-rosdep

# ── 4. rosdep init ─────────────────────────────────────────────
echo "[4/4] rosdep..."
rosdep init 2>/dev/null || true
rosdep update || true

# ── 검증 ───────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════"
echo " 검증"
echo "════════════════════════════════════════════"
source /opt/ros/jazzy/setup.bash
echo "ROS_DISTRO = $ROS_DISTRO"
echo "rclpy:   $(python3 -c 'import rclpy; print("OK")' 2>&1)"
echo "nav2 패키지 수: $(ros2 pkg list 2>/dev/null | grep -c nav2)"
echo ""
echo "다음: source /opt/ros/jazzy/setup.bash 를 ~/.bashrc 에 추가."
