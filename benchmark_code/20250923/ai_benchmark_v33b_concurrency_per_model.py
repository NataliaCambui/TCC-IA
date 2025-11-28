
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ai_benchmark_v33b_concurrency_per_model.py

Changelog vs v33:
- Semáforos de concorrência por provedor (OpenAI, Gemini, Claude) e para o Executor Service.
- Registro por requisição de: número de retries, número de 429 recebidos, status final.
- Agregação de métricas p50/p95/p99 + médias e também totais de 429/retries por provedor.
- HTTP pooling via requests.Session compartilhada.
- Backoff exponencial com jitter.
- Timeout configurável por CLI.
- Script tolerante à ausência de SDKs (p.ex., google-generativeai).
"""

from __future__ import annotations
import os, sys, time, json, math, random, logging, argparse, statistics as stats
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------- CLI ----------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Benchmark com controle de concorrência por provedor e métricas de retries/429.")
    p.add_argument("--num_runs", type=int, default=2, help="Número de execuções completas do conjunto de testes.")
    p.add_argument("--output_dir", type=str, default="benchmark_reports", help="Diretório de saída para relatórios.")
    p.add_argument("--openai_model", type=str, default=os.getenv("OPENAI_MODEL_NAME","gpt-4o-mini"))
    p.add_argument("--gemini_model", type=str, default=os.getenv("GEMINI_MODEL_NAME","gemini-1.5-pro"))
    p.add_argument("--claude_model", type=str, default=os.getenv("CLAUDE_MODEL_NAME","claude-3-5-sonnet"))
    p.add_argument("--executor_url", type=str, default=os.getenv("EXECUTOR_SERVICE_URL","https://executor-service-1029404216382.europe-southwest1.run.app/execute"))
    p.add_argument("--requests_timeout", type=float, default=float(os.getenv("REQUESTS_TIMEOUT", "60")), help="Timeout por requisição (s).")
    # Concorrência global e específica por provedor
    p.add_argument("--max_concurrency", type=int, default=int(os.getenv("MAX_CONCURRENCY","8")), help="Concorrência global máxima.")
    p.add_argument("--max_concurrency_openai", type=int, default=int(os.getenv("MAX_CONCURRENCY_OPENAI","0")), help="Se >0, sobrescreve o limite global para OpenAI.")
    p.add_argument("--max_concurrency_gemini", type=int, default=int(os.getenv("MAX_CONCURRENCY_GEMINI","0")), help="Se >0, sobrescreve o limite global para Gemini.")
    p.add_argument("--max_concurrency_claude", type=int, default=int(os.getenv("MAX_CONCURRENCY_CLAUDE","0")), help="Se >0, sobrescreve o limite global para Claude.")
    p.add_argument("--max_concurrency_executor", type=int, default=int(os.getenv("MAX_CONCURRENCY_EXECUTOR","0")), help="Se >0, sobrescreve o limite global para Executor Service.")
    p.add_argument("-v","--verbose", action="store_true", help="Logs detalhados.")
    return p

# ---------- Infra de HTTP com retries ----------

@dataclass
class HttpCallResult:
    ok: bool
    status: int
    elapsed_ms: float
    data: Optional[dict]
    error: Optional[str]
    retries: int
    hits_429: int
    url: str
    provider: str

class HttpClient:
    def __init__(self, timeout: float, max_workers: int, 
                 max_conn_pool: int = 100, 
                 log: Optional[logging.Logger] = None):
        self.timeout = timeout
        self.session = requests.Session()
        adapter = HTTPAdapter(pool_connections=max_conn_pool, pool_maxsize=max_conn_pool, max_retries=Retry(connect=0, read=0, redirect=0, status=0))
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.log = log or logging.getLogger("HttpClient")
        # Semáforos
        self.sem_global = threading.Semaphore(max_workers)
        self.sem_provider: Dict[str, threading.Semaphore] = {}

    def set_provider_limit(self, provider: str, limit: int | None):
        if limit and limit > 0:
            self.sem_provider[provider] = threading.Semaphore(limit)

    def _enter_sems(self, provider: str):
        self.sem_global.acquire()
        prov_sem = self.sem_provider.get(provider, None)
        if prov_sem:
            prov_sem.acquire()

    def _exit_sems(self, provider: str):
        prov_sem = self.sem_provider.get(provider, None)
        if prov_sem:
            prov_sem.release()
        self.sem_global.release()

    def post_json_with_backoff(self, url: str, payload: dict, headers: dict, provider: str, max_attempts: int = 5) -> HttpCallResult:
        hits_429 = 0
        last_err = None
        t0_all = time.time()
        status = 0
        data = None
        for attempt in range(max_attempts):
            self._enter_sems(provider)
            try:
                t0 = time.time()
                resp = self.session.post(url, json=payload, headers=headers, timeout=self.timeout)
                elapsed_ms = (time.time() - t0) * 1000.0
                status = resp.status_code
                if status == 429:
                    hits_429 += 1
                    raise RuntimeError(f"HTTP 429 (rate limited)")
                if status >= 500:
                    raise RuntimeError(f"HTTP {status} (server error)")
                resp.raise_for_status()
                try:
                    data = resp.json()
                except Exception as je:
                    # JSON inválido → trata como erro transitório
                    raise RuntimeError(f"Invalid JSON: {je}") from je
                return HttpCallResult(True, status, (time.time()-t0_all)*1000.0, data, None, attempt, hits_429, url, provider)
            except Exception as e:
                last_err = str(e)
                # Backoff com jitter
                delay = min(2 ** attempt, 16) + random.random()
                if self.log:
                    self.log.debug(f"[{provider}] POST {url} attempt={attempt+1}/{max_attempts} err={e} sleep={delay:.2f}s")
                time.sleep(delay)
            finally:
                self._exit_sems(provider)
        return HttpCallResult(False, status, (time.time()-t0_all)*1000.0, data, last_err, max_attempts-1, hits_429, url, provider)

# ---------- Helpers ----------

def percentile(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    k = (len(values)-1) * (p/100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return values[int(k)]
    d0 = values[f] * (c - k)
    d1 = values[c] * (k - f)
    return d0 + d1

def ensure_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)

# ---------- Benchmark tasks (exemplos simples) ----------
# Observação: Mantemos exemplos mínimos para acionar OpenAI / Executor Service.
# Gemini/Claude são acionados se SDKs/chaves existirem.

def build_openai_payload(model: str, prompt: str) -> Tuple[str, dict, dict]:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY','')}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role":"user","content": prompt}],
        "temperature": 0.2,
        "max_tokens": 64,
    }
    return url, headers, payload

def run_openai_call(http: HttpClient, model: str, prompt: str) -> Dict[str, Any]:
    url, headers, payload = build_openai_payload(model, prompt)
    res = http.post_json_with_backoff(url, payload, headers, provider="openai")
    out = {
        "provider":"openai",
        "model": model,
        "ok": res.ok,
        "status": res.status,
        "elapsed_ms": res.elapsed_ms,
        "retries": res.retries,
        "hits_429": res.hits_429,
        "error": res.error,
    }
    return out

def run_executor_code(http: HttpClient, executor_url: str, code: str, func_name: str, args: list) -> Dict[str, Any]:
    headers = {"Content-Type":"application/json"}
    payload = {"code_string": code, "func_name": func_name, "args": args}
    res = http.post_json_with_backoff(executor_url, payload, headers, provider="executor")
    out = {
        "provider":"executor",
        "ok": res.ok,
        "status": res.status,
        "elapsed_ms": res.elapsed_ms,
        "retries": res.retries,
        "hits_429": res.hits_429,
        "error": res.error,
    }
    if res.ok and isinstance(res.data, dict):
        out.update({
            "success": res.data.get("success"),
            "stdout": res.data.get("stdout"),
            "output": res.data.get("output"),
        })
    return out

# (Opcional) Gemini via SDK se disponível
def run_gemini_call(http: HttpClient, model: str, prompt: str) -> Dict[str, Any]:
    try:
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY","")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY ausente")
        genai.configure(api_key=api_key)
        t0 = time.time()
        # Não temos semáforo automático dentro do SDK; usamos global ao redor da chamada
        http._enter_sems("gemini")
        try:
            model_obj = genai.GenerativeModel(model)
            resp = model_obj.generate_content(prompt)
        finally:
            http._exit_sems("gemini")
        elapsed_ms = (time.time()-t0)*1000.0
        return {"provider":"gemini","model":model,"ok":True,"status":200,"elapsed_ms":elapsed_ms,"retries":0,"hits_429":0,"error":None}
    except Exception as e:
        return {"provider":"gemini","model":model,"ok":False,"status":0,"elapsed_ms":0.0,"retries":0,"hits_429":0,"error":str(e)}

# ---------- Runner ----------

def run_benchmark(args) -> Dict[str, Any]:
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("benchmark")

    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    http = HttpClient(timeout=args.requests_timeout, max_workers=args.max_concurrency, log=log)
    # Limites específicos
    if args.max_concurrency_openai > 0:
        http.set_provider_limit("openai", args.max_concurrency_openai)
    if args.max_concurrency_gemini > 0:
        http.set_provider_limit("gemini", args.max_concurrency_gemini)
    if args.max_concurrency_claude > 0:
        http.set_provider_limit("claude", args.max_concurrency_claude)
    if args.max_concurrency_executor > 0:
        http.set_provider_limit("executor", args.max_concurrency_executor)

    log.info(f"Executor URL: {args.executor_url}")
    log.info(f"Modelos: openai={args.openai_model} gemini={args.gemini_model} claude={args.claude_model}")
    log.info(f"Concorrência global={args.max_concurrency} (prov-specific: openai={args.max_concurrency_openai} gemini={args.max_concurrency_gemini} claude={args.max_concurrency_claude} executor={args.max_concurrency_executor})")

    results: List[Dict[str,Any]] = []

    # Exemplos de tarefas para cada run:
    # 1) OpenAI prompt curto
    prompt = "Return the length of the word 'benchmark'."
    # 2) Executor: função simples add(a,b)
    code = "def add(a,b):\n    return a+b\n"

    with ThreadPoolExecutor(max_workers=args.max_concurrency) as pool:
        futures = []
        for r in range(args.num_runs):
            futures.append(pool.submit(run_openai_call, http, args.openai_model, prompt))
            futures.append(pool.submit(run_executor_code, http, args.executor_url, code, "add", [2, 40]))
            # Gemini opcional
            futures.append(pool.submit(run_gemini_call, http, args.gemini_model, "Say 'ok'"))
        for f in as_completed(futures):
            results.append(f.result())

    # --- Agregação por provedor ---
    by_provider: Dict[str, List[Dict[str,Any]]] = {}
    for row in results:
        by_provider.setdefault(row["provider"], []).append(row)

    summary = {"providers":{}}
    for prov, rows in by_provider.items():
        lat_ok = sorted([r["elapsed_ms"] for r in rows if r.get("ok")])
        p50 = percentile(lat_ok, 50) if lat_ok else float("nan")
        p95 = percentile(lat_ok, 95) if lat_ok else float("nan")
        p99 = percentile(lat_ok, 99) if lat_ok else float("nan")
        mean = stats.mean(lat_ok) if lat_ok else float("nan")
        total_429 = sum(r.get("hits_429",0) for r in rows)
        total_retries = sum(r.get("retries",0) for r in rows)
        ok_rate = sum(1 for r in rows if r.get("ok"))/len(rows)
        summary["providers"][prov] = {
            "count": len(rows),
            "ok_rate": ok_rate,
            "p50_ms": p50,
            "p95_ms": p95,
            "p99_ms": p99,
            "mean_ms": mean,
            "total_429": total_429,
            "total_retries": total_retries,
        }

    # Persistir JSON com detalhe e resumo
    detail_path = out_dir / f"detail_{int(time.time())}.json"
    with detail_path.open("w", encoding="utf-8") as f:
        json.dump({"results": results}, f, ensure_ascii=False, indent=2)

    summary_path = out_dir / f"summary_{int(time.time())}.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log.info(f"Gravou detalhes em {detail_path}")
    log.info(f"Gravou resumo em {summary_path}")

    return {"detail_path": str(detail_path), "summary_path": str(summary_path), "summary": summary}

def main():
    args = build_argparser().parse_args()
    _ = run_benchmark(args)

if __name__ == "__main__":
    main()
