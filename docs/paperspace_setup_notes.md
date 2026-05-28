# Paperspace GPU 머신 셋업 노트

## 머신 재생성 후 즉시 실행할 것

새 머신 켜자마자 아래 명령어로 커널/드라이버 자동 업데이트를 막아야 한다.
재부팅 시 커널이 업데이트되면 NVIDIA 커널 모듈이 깨져서 Isaac Sim GUI가 블랙스크린이 된다.

```bash
# 커널 업데이트 방지
sudo apt-mark hold linux-image-generic linux-headers-generic

# NVIDIA 드라이버 업데이트 방지
sudo apt-mark hold nvidia-driver-* libnvidia-* xserver-xorg-video-nvidia-*

# 자동 업데이트 비활성화
sudo systemctl disable unattended-upgrades
sudo systemctl stop unattended-upgrades
```

## 머신 재생성 절차

1. 구 머신에서 체크포인트 GitHub에 푸시
2. 새 머신 생성 (Paperspace Desktop GPU)
3. 위 hold 명령어 실행
4. 환경 설치:

```bash
apt-get update && apt install -y git
git clone https://github.com/vanillaturtlechips/MARS.git /workspace/MARS
bash /workspace/MARS/deploy/desktop/setup.sh
```

## Isaac Sim GUI 실행 시 주의

- `--headless` 없이 실행하면 GUI 뜸
- `--num_envs 16` 정도로 낮춰서 실행 (GUI는 환경 수 적게)

```bash
source /workspace/isaac_venv/bin/activate
cd /workspace/MARS
python training/single_robot/train_manipulation.py \
    --num_envs 16 \
    --resume_ckpt moodel_900_backup.pt
```

## 교훈

- 재부팅 전에 반드시 hold 걸어둘 것
- Isaac Sim RTX renderer는 NVIDIA 드라이버 535.129 이상 필요
- 커널 업데이트 후 NVIDIA 모듈이 새 커널용으로 재빌드 안 되면 GUI 불가
