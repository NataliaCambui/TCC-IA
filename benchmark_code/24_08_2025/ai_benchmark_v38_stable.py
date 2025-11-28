# -*- coding: utf-8 -*-
import requests
import time
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
from dotenv import load_dotenv
import google.generativeai as genai
import google.api_core.exceptions
import subprocess
import tempfile
import platform
import traceback
import sys
import multiprocessing
import queue
from collections import OrderedDict, deque, defaultdict
import builtins
import logging
import argparse
from abc import ABC, abstractmethod
from functools import lru_cache
from string import ascii_lowercase, digits

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

try:
    from importlib import metadata as importlib_metadata
except ImportError:
    try:
        import pkg_resources  # type: ignore
    except ImportError:
        pkg_resources = None

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# --- Classes de Provedor de IA Abstraídas ---
class AIProvider(ABC):
    def __init__(self, model_name: str, api_key: str | None):
        self.model_name = model_name
        self.api_key = api_key
        if not self.api_key:
            raise ValueError("Chave API não fornecida.")

    @abstractmethod
    def query(self, prompt: str) -> dict:
        pass

class ClaudeProvider(AIProvider):
    def query(self, prompt: str) -> dict:
        start_time = time.time()
        content, last_exception = "", None
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        data = {"model": self.model_name, "max_tokens": 4000, "messages": [{"role": "user", "content": prompt}]}
        try:
            response = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=data, timeout=120)
            response.raise_for_status()
            result = response.json()
            if "content" in result and result["content"]:
                content = result["content"][0].get("text", "")
        except Exception as e:
            last_exception = e
            logging.error(f"Erro ao consultar Claude: {e}", exc_info=False)
        return {"content": content, "execution_time": time.time() - start_time, "last_error": str(last_exception) if last_exception else None}

class OpenAIProvider(AIProvider):
    def query(self, prompt: str) -> dict:
        start_time = time.time()
        content, last_exception = "", None
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        data = {"model": self.model_name, "messages": [{"role": "user", "content": prompt}], "max_tokens": 4000}
        try:
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data, timeout=120)
            response.raise_for_status()
            result = response.json()
            if "choices" in result and result["choices"]:
                content = result["choices"][0].get("message", {}).get("content", "")
        except Exception as e:
            last_exception = e
            logging.error(f"Erro ao consultar ChatGPT: {e}", exc_info=False)
        return {"content": content, "execution_time": time.time() - start_time, "last_error": str(last_exception) if last_exception else None}

class GeminiProvider(AIProvider):
    def __init__(self, model_name: str, api_key: str | None):
        super().__init__(model_name, api_key)
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
        except Exception as e:
            raise ConnectionError(f"Falha ao configurar Gemini: {e}") from e

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=5, max=60), retry=retry_if_exception_type(google.api_core.exceptions.ResourceExhausted), reraise=True)
    def _call_gemini_api_with_retry(self, prompt_str: str):
        safety_settings = [{"category": c, "threshold": "BLOCK_NONE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
        return self.model.generate_content(prompt_str, safety_settings=safety_settings)

    def query(self, prompt: str) -> dict:
        start_time = time.time()
        content, last_exception = "", None
        try:
            response = self._call_gemini_api_with_retry(prompt)
            content = response.text
        except Exception as e:
            last_exception = e
            logging.error(f"Erro ao consultar Gemini: {e}", exc_info=False)
        return {"content": content, "execution_time": time.time() - start_time, "last_error": str(last_exception) if last_exception else None}


class AIBenchmark:
    STATUS_PASS = "Pass"
    STATUS_FAIL = "Fail"
    STATUS_API_ERROR = "API Error"
    STATUS_NO_CODE = "No Code/No Function"
    STATUS_ERROR = "Error"
    CHALLENGE_LRU_CACHE = "LRU Cache"
    CHALLENGE_TWO_SUM = "Two Sum"

    def __init__(
        self,
        num_runs: int = 1,
        output_dir: str = "benchmark_reports",
        claude_model_name: str | None = None,
        openai_model_name: str | None = None,
        gemini_model_name: str | None = None,
        executor_service_url: str | None = None,
    ):
        self.num_runs = max(1, num_runs)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        logging.info(f"Número de execuções por desafio/IA: {self.num_runs}")
        logging.info(f"Diretório de saída: '{self.output_dir}/'")

        self.results = []
        self.detailed_test_results = []

        # --- ARMAZENAR CONFIGURAÇÕES, NÃO OBJETOS ---
        self.provider_configs = {}
        claude_key = os.getenv("CLAUDE_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        gemini_key = os.getenv("GEMINI_API_KEY")

        if claude_key:
            self.provider_configs["Claude"] = {
                "class": ClaudeProvider,
                "model": claude_model_name or os.getenv("CLAUDE_MODEL_NAME", "claude-3-opus-20240229"),
                "api_key": claude_key
            }
        if openai_key:
            self.provider_configs["ChatGPT"] = {
                "class": OpenAIProvider,
                "model": openai_model_name or os.getenv("OPENAI_MODEL_NAME", "gpt-4-turbo"),
                "api_key": openai_key
            }
        if gemini_key:
            self.provider_configs["Gemini"] = {
                "class": GeminiProvider,
                "model": gemini_model_name or os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-pro-latest"),
                "api_key": gemini_key
            }
        
        default_executor_url = "https://executor-service-1029404216382.europe-southwest1.run.app/execute"
        self.executor_url = executor_service_url or os.getenv("EXECUTOR_SERVICE_URL", default_executor_url)

        self.challenge_details = {
            "Fibonacci Sequence": "Testa a geração de sequências numéricas básicas.",
            "Binary Search": "Testa a implementação de algoritmos de busca eficientes.",
            "Merge Sort": "Testa a implementação de algoritmos de ordenação recursivos.",
            self.CHALLENGE_LRU_CACHE: "Testa o design de estruturas de dados customizadas.",
            self.CHALLENGE_TWO_SUM: "Testa o uso eficiente de dicionários para busca rápida."
        }
        
        self.code_challenges = [
            {
                "name": "Fibonacci Sequence", "description": "Generate the first n numbers in the Fibonacci sequence.",
                "difficulty": "Easy", "function_signature": "fibonacci(n: int) -> list[int]",
                "test_cases": [{"input_args": [10], "expected_output": [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]}, {"input_args": [1], "expected_output": [0]}]
            },
            {
                "name": "Binary Search", "description": "Find the index of a target value in a sorted list.",
                "difficulty": "Medium", "function_signature": "binary_search(arr: list[int], target: int) -> int",
                "test_cases": [{"input_args": [[2, 5, 8, 12, 16], 8], "expected_output": 2}, {"input_args": [[2, 5, 8, 12, 16], 3], "expected_output": -1}]
            },
            {
                "name": "Merge Sort", "description": "Implement merge sort.",
                "difficulty": "Medium", "function_signature": "merge_sort(arr: list[int]) -> list[int]",
                "test_cases": [{"input_args": [[3, 1, 4, 1, 5, 9]], "expected_output": [1, 1, 3, 4, 5, 9]}]
            },
            {
                "name": self.CHALLENGE_LRU_CACHE, "description": "Implement a Least Recently Used (LRU) cache.",
                "difficulty": "Hard", "function_signature": "class LRUCache",
                "test_cases": [{"sequence": [{"op": "__init__", "args": [2]}, {"op": "put", "args": [1, 1]}, {"op": "put", "args": [2, 2]}, {"op": "get", "args": [1], "expected": 1}, {"op": "put", "args": [3, 3]}, {"op": "get", "args": [2], "expected": -1}]}]
            },
            {
                "name": self.CHALLENGE_TWO_SUM, "description": "Find indices of two numbers that add up to a target.",
                "difficulty": "Easy", "function_signature": "two_sum(nums: list[int], target: int) -> list[int]",
                "test_cases": [{"input_args": [[2, 7, 11, 15], 9], "expected_output": [0, 1]}]
            },
        ]
        self._check_flake8()

    def _check_flake8(self):
        try:
            subprocess.run([sys.executable, "-m", "flake8", "--version"], check=True, capture_output=True, timeout=5)
            logging.info("Verificação Flake8: OK")
            self.flake8_available = True
        except Exception:
            logging.warning("Flake8 não encontrado.")
            self.flake8_available = False

    def extract_code(self, content: str) -> str:
        if not content: return ""
        match = re.search(r"```python\n(.*?)\n```", content, re.DOTALL)
        return match.group(1).strip() if match else content.strip()

    def analyze_code_quality_flake8(self, code: str) -> int:
        if not self.flake8_available or not code: return -1
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".py", delete=False) as tf:
            tf.write(code)
            path = tf.name
        try:
            res = subprocess.run([sys.executable, "-m", "flake8", "--count", path], capture_output=True, text=True)
            return int(res.stdout.strip().splitlines()[-1]) if res.returncode in [0, 1] and res.stdout else -1
        finally:
            os.remove(path)

    def _lru_worker(self, code, sequence, queue):
        try:
            namespace = {"OrderedDict": OrderedDict}
            exec(code, {"__builtins__": builtins}, namespace)
            LruClass = next(v for v in namespace.values() if isinstance(v, type))
            instance = None
            for step in sequence:
                op, args, expected = step["op"], step.get("args", []), step.get("expected")
                if op == "__init__": instance = LruClass(*args)
                elif op == "put": getattr(instance, op)(*args)
                elif op == "get":
                    if str(getattr(instance, op)(*args)) != str(expected):
                        raise AssertionError("LRU Get Mismatch")
            queue.put({"success": True, "error": None})
        except Exception as e:
            queue.put({"success": False, "error": str(e)})

    def _execute_lru_locally(self, code, sequence) -> dict:
        ctx = multiprocessing.get_context('spawn')
        q = ctx.Queue()
        p = ctx.Process(target=self._lru_worker, args=(code, sequence, q))
        try:
            p.start()
            res = q.get(timeout=10)
        except queue.Empty:
            res = {"success": False, "error": "Timeout"}
        finally:
            if p.is_alive():
                p.terminate()
                p.join(2)
                if p.is_alive():
                    p.kill()
            p.close()
            q.close()
        return res

    def evaluate_solution(self, ai_name, challenge, response, run_idx):
        content, api_time, api_error = response.get("content", ""), response.get("execution_time", 0), response.get("last_error")
        if api_error:
            self._handle_evaluation_error(ai_name, challenge, run_idx, self.STATUS_API_ERROR, api_error)
            return
        code = self.extract_code(content)
        if not code:
            self._handle_evaluation_error(ai_name, challenge, run_idx, self.STATUS_NO_CODE, "Code not extracted")
            return
        
        n_errors, first_err = 0, ""
        if challenge["name"] == self.CHALLENGE_LRU_CACHE:
            res = self._execute_lru_locally(code, challenge["test_cases"][0]["sequence"])
            if not res["success"]:
                n_errors, first_err = 1, res["error"]
        else:
            func_name = self._detect_function_name(code, challenge)
            if not func_name:
                self._handle_evaluation_error(ai_name, challenge, run_idx, self.STATUS_NO_CODE, "Function name not found")
                return
            
            for tc in challenge["test_cases"]:
                res = self._call_executor_service(code, func_name, tc["input_args"])
                if not res["success"] or not np.array_equal(res["output"], tc["expected_output"]):
                    n_errors += 1
                    if not first_err: first_err = res.get("error", "Output mismatch")

        self.results.append({
            "run_index": run_idx, "ai": ai_name, "challenge": challenge["name"],
            "api_execution_time": api_time, "flake8_issues": self.analyze_code_quality_flake8(code),
            "num_execution_errors": n_errors, "first_execution_error": first_err
        })

    def _handle_evaluation_error(self, ai_name, challenge, run_idx, status, error_message):
        logging.warning(f"{error_message} for {ai_name} on {challenge['name']}")
        self.results.append({
            "run_index": run_idx, "ai": ai_name, "challenge": challenge["name"],
            "api_execution_time": 0, "flake8_issues": -1,
            "num_execution_errors": len(challenge["test_cases"]), "first_execution_error": error_message
        })

    def _detect_function_name(self, code, challenge):
        match = re.search(r"^\s*(?:def|class)\s+(\w+)", code, re.MULTILINE)
        return match.group(1) if match else None
        
    def _call_executor_service(self, code, func_name, args):
        imports = (
            "import re\nfrom collections import OrderedDict, deque, defaultdict\n"
            "from functools import lru_cache\nfrom string import ascii_lowercase, digits\n"
            "from typing import List, Dict, Any\nimport numpy as np\n\n"
        )
        payload = {"code_string": imports + code, "function_name": func_name, "test_input_args": args}
        try:
            resp = requests.post(self.executor_url, json=payload, timeout=60)
            return resp.json() if resp.ok else {"success": False, "error": f"Executor error {resp.status_code}"}
        except requests.RequestException as e:
            return {"success": False, "error": str(e)}

    def run_benchmark(self):
        if not self.provider_configs:
            logging.critical("Nenhuma API configurada.")
            return

        logging.info(f"Iniciando Benchmark para: {', '.join(self.provider_configs.keys())} ({self.num_runs} execução(ões))")
        for run_idx in range(self.num_runs):
            logging.info(f"\n{'='*20} INICIANDO EXECUÇÃO {run_idx + 1}/{self.num_runs} {'='*20}")
            for chal in self.code_challenges:
                logging.info(f"\n--- [Run {run_idx+1}] Desafio: {chal['name']} ---")
                logging.info(f"    Propósito: {self.challenge_details.get(chal['name'], 'N/A')}")
                
                for ai_name, config in self.provider_configs.items():
                    logging.info(f"  Consultando {ai_name}...")
                    provider = config["class"](config["model"], config["api_key"])
                    
                    prompt = (f"Implement the Python function/class for: {chal['description']}.\n"
                              f"Signature: {chal['function_signature']}\n"
                              "Provide ONLY the raw Python code.")
                    
                    response = provider.query(prompt)
                    self.evaluate_solution(ai_name, chal, response, run_idx)
                    time.sleep(1)
        
        self.generate_report()

    def generate_report(self):
        if not self.results:
            logging.error("Nenhum resultado para gerar relatório.")
            return
        logging.info("\n--- Gerando Relatórios Finais ---")
        df = pd.DataFrame(self.results)
        
        # Análise Agregada
        summary = df.groupby('ai').agg(
            avg_api_time=('api_execution_time', 'mean'),
            avg_flake8=('flake8_issues', lambda x: x[x!=-1].mean()),
            total_errors=('num_execution_errors', 'sum'),
            success_rate=('num_execution_errors', lambda x: 1 - (x > 0).sum() / len(x))
        ).reset_index()
        
        print("\n===== SUMÁRIO GERAL DO BENCHMARK =====")
        try:
            from tabulate import tabulate
            print(tabulate(summary, headers='keys', tablefmt='grid', showindex=False, floatfmt=".2f"))
        except ImportError:
            print(summary.to_string(index=False))

if __name__ == "__main__":
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(description="Executa um benchmark de programação de IA.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-n", "--num_runs", type=int, default=3, help="Número de execuções por desafio.")
    parser.add_argument("-o", "--output_dir", type=str, default="benchmark_reports", help="Diretório de saída.")
    parser.add_argument("--claude_model", type=str, help="Nome do modelo Claude.")
    parser.add_argument("--openai_model", type=str, help="Nome do modelo OpenAI.")
    parser.add_argument("--gemini_model", type=str, help="Nome do modelo Gemini.")
    parser.add_argument("--executor_url", type=str, help="URL do serviço de execução.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Ativa logging detalhado.")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        benchmark = AIBenchmark(
            num_runs=args.num_runs,
            output_dir=args.output_dir,
            claude_model_name=args.claude_model,
            openai_model_name=args.openai_model,
            gemini_model_name=args.gemini_model,
            executor_service_url=args.executor_url
        )
        benchmark.run_benchmark()
        logging.info(f"\nBenchmark concluído. Relatórios em: '{benchmark.output_dir}'")
    except Exception as e:
        logging.critical("\nERRO INESPERADO DURANTE A EXECUÇÃO:", exc_info=True)