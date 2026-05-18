---
name: feedback-rsl-rl-runpod
description: rsl_rl ModuleNotFoundError on RunPod — two known fixes
metadata:
  type: feedback
---

`ModuleNotFoundError: No module named 'rsl_rl'` 는 RunPod 새 인스턴스마다 반복되는 뻔한 문제.

**Why:** Isaac Sim AppLauncher가 시작 시 `sys.path`를 덮어쓰기 때문에, venv에 `rsl-rl-lib` PyPI 패키지가 설치돼 있어도 못 찾는 경우가 있음. pip install이 system pip으로 들어간 경우도 있음.

**How to apply:** 발생하면 아래 두 방법 중 하나. setup.sh에서는 방법 1(소스 클론)로 고정할 것.

방법 1 — 소스 직접 설치 (가장 확실):
```bash
cd /workspace
git clone https://github.com/isaac-sim/rsl_rl.git
cd rsl_rl
pip install -e .
```

방법 2 — Isaac Lab 스크립트 활용:
```bash
cd /workspace/IsaacLab
./isaaclab.sh --extra rsl_rl
```

setup.sh에서는 `pip install "rsl-rl-lib==3.1.2"` 대신 소스 클론 방식으로 변경 필요.
