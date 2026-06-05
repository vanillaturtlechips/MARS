#!/bin/bash
# MARS Phase 4 — ROS2 Humble + Nav2 + colcon 설치 (Ubuntu 22.04 / Jammy)
# 사용법: bash deploy/runpod/setup_ros2.sh
#
# 이 스크립트는 시스템 레벨(apt)에만 설치한다. RL용 isaac_venv 와 무관.
# Isaac Sim ROS2 bridge 연동(난관)은 별도. 여기서는 "Nav2가 실제로 도는가"까지만.
set -e

echo "════════════════════════════════════════════"
echo " MARS Phase 4 — ROS2 Humble 설치"
echo "════════════════════════════════════════════"

# ── 0. OS 확인 (Humble = Jammy 전용) ───────────────────────────
. /etc/os-release
if [ "$VERSION_CODENAME" != "jammy" ]; then
    echo "[중단] ROS2 Humble은 Ubuntu 22.04(jammy) 전용. 현재: $VERSION_CODENAME"
    exit 1
fi
echo "▶ OS: $PRETTY_NAME  (jammy OK)"

# 디스크 경고 — ros-humble-desktop 은 약 2.5GB (root fs)
echo "▶ root 디스크:"; df -h / | tail -1

# ── 1. locale ──────────────────────────────────────────────────
echo "[1/5] locale..."
apt-get update -q
apt-get install -y -q locales
locale-gen en_US en_US.UTF-8
update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

# ── 2. ROS2 apt 저장소 ─────────────────────────────────────────
echo "[2/5] ROS2 apt 저장소 등록..."
apt-get install -y -q software-properties-common curl gnupg lsb-release
add-apt-repository universe -y
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu jammy main" \
    | tee /etc/apt/sources.list.d/ros2.list > /dev/null
apt-get update -q

# ── 3. ROS2 Humble + Nav2 + 빌드도구 ───────────────────────────
# desktop = RViz 포함(데모 시각화에 필요). 디스크 부족 시 ros-humble-ros-base 로 교체.
echo "[3/5] ros-humble-desktop + Nav2 + colcon (수 분 소요)..."
apt-get install -y \
    ros-humble-desktop \
    ros-humble-navigation2 \
    ros-humble-nav2-bringup \
    ros-dev-tools \
    python3-colcon-common-extensions \
    python3-rosdep

# Nav2 동작 스모크테스트용 TB3 sim (Isaac Sim과 무관하게 "Nav2가 도는가" 검증용)
echo "[4/5] TurtleBot3 sim (Nav2 검증용)..."
apt-get install -y \
    ros-humble-turtlebot3 \
    ros-humble-turtlebot3-gazebo \
    ros-humble-nav2-minimal-tb3-sim || \
    echo "  (TB3 패키지 일부 없음 — 검증 단계에서 다시 시도)"

# ── 5. rosdep init ─────────────────────────────────────────────
echo "[5/5] rosdep..."
rosdep init 2>/dev/null || true
rosdep update || true

# ── 검증 ───────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════"
echo " 검증"
echo "════════════════════════════════════════════"
source /opt/ros/humble/setup.bash
echo "ROS_DISTRO = $ROS_DISTRO"
echo "rclpy:   $(python3 -c 'import rclpy; print("OK")' 2>&1)"
echo "colcon:  $(colcon version-check 2>/dev/null | head -1 || colcon --help >/dev/null 2>&1 && echo OK)"
echo "nav2 패키지 수: $(ros2 pkg list 2>/dev/null | grep -c nav2)"
echo ""
echo "다음: source /opt/ros/humble/setup.bash 를 ~/.bashrc 에 추가하거나 매 셸에서 실행."
echo "그다음 Step 2 — mars_msgs colcon 빌드."
