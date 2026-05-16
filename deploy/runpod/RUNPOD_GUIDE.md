# RunPod 설정 가이드

## Pod 생성 시 필수 포트 설정

**반드시 Pod 생성할 때 아래 포트를 추가해야 함. 생성 후엔 추가 불가.**

| 포트 | 용도 |
|------|------|
| 8211 | Isaac Sim WebRTC 스트리밍 (GUI) |
| 6006 | TensorBoard |

RunPod 대시보드 → Pod 생성 → "Expose HTTP Ports" 또는 "TCP Ports" 에 추가.

---

## Isaac Sim GUI 스트리밍 (livestream)

Pod에 포트 8211 추가된 상태에서:

**RunPod 터미널:**
```bash
source /workspace/isaac_venv/bin/activate
cd /workspace/MARS
python training/multi_robot/train_ippo.py \
  --checkpoint /workspace/MARS/logs/warehouse_ippo/model_400.pt \
  --livestream 1
```

**로컬 터미널 (SSH 터널):**
```bash
ssh -L 8211:localhost:8211 root@{POD_IP} -p {PORT} -i ~/.ssh/id_ed25519
```

**브라우저:**
```
http://localhost:8211
```

`Simulation App Startup Complete` 로그 뜨면 접속.

---

## TensorBoard

**RunPod 터미널:**
```bash
tensorboard --logdir /workspace/MARS/logs --port 6006 --bind_all
```

**로컬 터미널 (SSH 터널):**
```bash
ssh -L 6006:localhost:6006 root@{POD_IP} -p {PORT} -i ~/.ssh/id_ed25519
```

**브라우저:**
```
http://localhost:6006
```

---

## 훈련 재시작 (체크포인트에서 이어받기)

```bash
# 체크포인트 확인
find /workspace/MARS/logs -name "model_*.pt"

# 이어받아 훈련
python training/multi_robot/train_ippo.py \
  --headless --num_envs 256 \
  --checkpoint /workspace/MARS/logs/warehouse_ippo/model_400.pt
```

---

## 주의사항

- `--livestream 2` 는 브라우저 접속 안 됨. 반드시 `--livestream 1` 사용.
- Isaac Sim 훈련 중엔 GUI 동시 실행 불가 (GPU 메모리 부족).
- 훈련 멈추고 → GUI 확인 → 다시 훈련 순서로.
