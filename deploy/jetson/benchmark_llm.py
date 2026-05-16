"""Jetson LLM 벤치마킹 — tokens/sec, JSON 성공률, 첫 토큰 레이턴시.

실행:
  python3 benchmark_llm.py
  python3 benchmark_llm.py --n 50  # 반복 횟수 줄이기
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
import urllib.error

OLLAMA_URL = "http://localhost:11434/api/generate"

MODELS = [
    "llama3.2:3b-instruct-q4_K_M",
    "qwen2.5:3b-instruct-q4_K_M",
]

# 창고 로봇 명령 생성 프롬프트 — 실제 사용 패턴 반영
PROMPT = """You are a warehouse robot controller.
Generate a robot command in valid JSON format only. No explanation.

Task: Robot A must pick up box from shelf B3 and deliver to gate G1.

Required JSON format:
{
  "robot_id": "A",
  "action": "pick_and_place",
  "pickup": {"shelf": "B3", "x": -2.0, "y": 2.5},
  "dropoff": {"gate": "G1", "x": 4.0, "y": 0.0},
  "priority": 1
}"""


def call_ollama(model: str, prompt: str, timeout: int = 60) -> dict:
    payload = json.dumps({
        "model":  model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 200},
    }).encode()

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        elapsed = time.perf_counter() - t0
        result  = json.loads(resp.read())
    result["_elapsed"] = elapsed
    return result


def first_token_latency(model: str, prompt: str, timeout: int = 30) -> float:
    """스트리밍으로 첫 번째 토큰까지의 레이턴시 측정."""
    payload = json.dumps({
        "model":  model,
        "prompt": prompt,
        "stream": True,
        "options": {"temperature": 0.1, "num_predict": 5},
    }).encode()

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        resp.readline()   # 첫 번째 청크 (= 첫 토큰)
        latency = time.perf_counter() - t0
    return latency * 1000   # ms


def try_parse_json(text: str) -> bool:
    """응답에서 JSON 추출 시도."""
    text = text.strip()
    # 코드 블록 제거
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip().lstrip("json").strip()
            try:
                json.loads(part)
                return True
            except json.JSONDecodeError:
                pass
    # 직접 파싱
    try:
        json.loads(text)
        return True
    except json.JSONDecodeError:
        pass
    # { } 블록 추출 시도
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start != -1 and end > start:
        try:
            json.loads(text[start:end])
            return True
        except json.JSONDecodeError:
            pass
    return False


def benchmark_model(model: str, n: int) -> dict:
    print(f"\n{'='*50}")
    print(f"모델: {model}")
    print(f"{'='*50}")

    # 워밍업
    print("워밍업 중...")
    try:
        call_ollama(model, "Hello", timeout=120)
    except Exception as e:
        print(f"[오류] 워밍업 실패: {e}")
        return {}

    # 첫 토큰 레이턴시 (5회 평균)
    print("첫 토큰 레이턴시 측정 (5회)...")
    ftl_list = []
    for _ in range(5):
        try:
            ftl = first_token_latency(model, PROMPT)
            ftl_list.append(ftl)
        except Exception:
            pass

    # 본 벤치마킹
    print(f"본 측정 {n}회...")
    tokens_per_sec_list = []
    json_success = 0
    errors = 0

    for i in range(n):
        try:
            result = call_ollama(model, PROMPT)
            response_text = result.get("response", "")
            eval_count    = result.get("eval_count", 0)
            eval_duration = result.get("eval_duration", 1)   # ns

            tps = eval_count / (eval_duration / 1e9) if eval_duration > 0 else 0
            tokens_per_sec_list.append(tps)

            if try_parse_json(response_text):
                json_success += 1

            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{n} — 현재 {tps:.1f} tok/s")

        except Exception as e:
            errors += 1
            print(f"  [{i+1}] 오류: {e}")

    if not tokens_per_sec_list:
        print("측정 실패")
        return {}

    avg_tps  = sum(tokens_per_sec_list) / len(tokens_per_sec_list)
    avg_ftl  = sum(ftl_list) / len(ftl_list) if ftl_list else -1
    json_rate = json_success / n * 100

    print(f"\n--- {model} 결과 ---")
    print(f"tokens/sec:      {avg_tps:.1f} (목표: > 15)")
    print(f"첫 토큰 레이턴시: {avg_ftl:.0f} ms (목표: < 500)")
    print(f"JSON 성공률:     {json_rate:.1f}% ({json_success}/{n}, 목표: > 95%)")
    print(f"오류 횟수:        {errors}")

    return {
        "model":       model,
        "tokens_per_sec": avg_tps,
        "first_token_latency_ms": avg_ftl,
        "json_success_rate": json_rate,
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",      type=int, default=30, help="반복 횟수")
    parser.add_argument("--models", nargs="+", default=MODELS)
    args = parser.parse_args()

    print(f"=== Jetson LLM 벤치마킹 ({args.n}회) ===\n")
    print("ollama 서버 확인 중...")
    try:
        urllib.request.urlopen("http://localhost:11434", timeout=5)
        print("ollama 서버 정상\n")
    except Exception:
        print("ollama 서버 응답 없음. 'ollama serve' 실행 후 재시도")
        return

    results = []
    for model in args.models:
        r = benchmark_model(model, args.n)
        if r:
            results.append(r)

    if len(results) < 2:
        return

    print(f"\n{'='*50}")
    print("=== 최종 비교 ===")
    print(f"{'='*50}")
    print(f"{'항목':<25} {'llama3.2:3b':>15} {'qwen2.5:3b':>15}")
    print(f"{'-'*55}")

    metrics = [
        ("tokens/sec",          "tokens_per_sec",          "{:.1f}"),
        ("첫 토큰 레이턴시(ms)", "first_token_latency_ms",  "{:.0f}"),
        ("JSON 성공률(%)",       "json_success_rate",        "{:.1f}"),
    ]
    for label, key, fmt in metrics:
        vals = [fmt.format(r.get(key, 0)) for r in results]
        print(f"{label:<25} {vals[0]:>15} {vals[1]:>15}")

    print(f"\n선택 기준: tokens/sec > 15 AND JSON 성공률 > 95%")

    # 결과 저장
    with open("benchmark_result.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"결과 저장: benchmark_result.json")


if __name__ == "__main__":
    main()
