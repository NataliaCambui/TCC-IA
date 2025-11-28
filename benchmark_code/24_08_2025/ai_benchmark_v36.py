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
    """
    Executa um benchmark de programação comparando modelos de IA
    (Claude, ChatGPT, Gemini) em desafios de código, avaliando a qualidade,
    corretude e desempenho.
    """
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

        claude_model = claude_model_name or os.getenv("CLAUDE_MODEL_NAME", "claude-3-opus-20240229")
        openai_model = openai_model_name or os.getenv("OPENAI_MODEL_NAME", "gpt-4-turbo")
        gemini_model = gemini_model_name or os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-pro-latest")
        default_executor_url = "https://executor-service-1029404216382.europe-southwest1.run.app/execute"
        self.executor_url = executor_service_url or os.getenv("EXECUTOR_SERVICE_URL", default_executor_url)

        self.providers = {}
        api_keys = {
            "Claude": (os.getenv("CLAUDE_API_KEY"), ClaudeProvider, claude_model),
            "ChatGPT": (os.getenv("OPENAI_API_KEY"), OpenAIProvider, openai_model),
            "Gemini": (os.getenv("GEMINI_API_KEY"), GeminiProvider, gemini_model),
        }
        for name, (key, provider_class, model) in api_keys.items():
            if key:
                try:
                    self.providers[name] = provider_class(model, key)
                    logging.info(f"Provedor '{name}' (modelo: {model}) configurado com sucesso.")
                except Exception as e:
                    logging.error(f"Falha ao inicializar provedor '{name}': {e}")
            else:
                logging.warning(f"Chave API para '{name}' não encontrada. O provedor será pulado.")

        self.code_challenges = [
             {
                "name": "Fibonacci Sequence", "difficulty": "Easy", "function_signature": "fibonacci(n: int) -> list[int]",
                "test_cases": [{"input_args": [10], "expected_output": [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]}, {"input_args": [1], "expected_output": [0]}, {"input_args": [0], "expected_output": []}, {"input_args": [2], "expected_output": [0, 1]}]
            },
            {
                "name": "Binary Search", "difficulty": "Medium", "function_signature": "binary_search(arr: list[int], target: int) -> int",
                "test_cases": [{"input_args": [[2, 5, 7, 8, 11, 12], 13], "expected_output": -1}, {"input_args": [[2, 5, 7, 8, 11, 12], 8], "expected_output": 3}]
            },
            {
                "name": self.CHALLENGE_LRU_CACHE, "difficulty": "Hard", "function_signature": "class LRUCache:\n    def __init__(self, capacity: int): ...",
                "test_cases": [{"description": "Simple put/get sequence", "sequence": [{"op": "__init__", "args": [2]}, {"op": "put", "args": [1, 1]}, {"op": "put", "args": [2, 2]}, {"op": "get", "args": [1], "expected": 1}, {"op": "put", "args": [3, 3]}, {"op": "get", "args": [2], "expected": -1}, {"op": "put", "args": [4, 4]}, {"op": "get", "args": [1], "expected": -1}, {"op": "get", "args": [3], "expected": 3}, {"op": "get", "args": [4], "expected": 4}]}]
            },
            {
                "name": self.CHALLENGE_TWO_SUM, "difficulty": "Easy", "function_signature": "two_sum(nums: list[int], target: int) -> list[int]",
                "test_cases": [{"input_args": [[2, 7, 11, 15], 9], "expected_output": [0, 1]}, {"input_args": [[3, 2, 4], 6], "expected_output": [1, 2]}]
            },
            # Adicionar outros desafios aqui se necessário
        ]
        self._check_flake8()

    def _check_flake8(self):
        try:
            subprocess.run([sys.executable, "-m", "flake8", "--version"], check=True, capture_output=True, timeout=5)
            logging.info("Verificação Flake8: OK")
            self.flake8_available = True
        except Exception:
            logging.warning("Flake8 não encontrado. Análise de qualidade será pulada.")
            self.flake8_available = False

    def extract_code(self, content: str) -> str:
        if not content: return ""
        m_python = re.search(r"```python\n(.*?)\n```", content, re.DOTALL)
        if m_python: return m_python.group(1).strip()
        m_generic = re.search(r"```(?:.*\n)?(.*?)\n```", content, re.DOTALL)
        if m_generic: return m_generic.group(1).strip()
        return content.strip()

    def analyze_code_quality_flake8(self, code_string: str) -> int:
        if not self.flake8_available or not code_string: return -1
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".py", delete=False, encoding="utf-8") as tf:
            tf.write(code_string)
            temp_file_path = tf.name
        try:
            result = subprocess.run([sys.executable, "-m", "flake8", "--count", temp_file_path], capture_output=True, text=True, timeout=15)
            if result.returncode in [0, 1] and result.stdout:
                return int(result.stdout.strip().splitlines()[-1])
            return 0 if result.returncode == 0 else -1
        except Exception: return -1
        finally:
            if os.path.exists(temp_file_path): os.remove(temp_file_path)

    def _lru_worker(self, code_string: str, test_sequence: list, result_queue: multiprocessing.Queue):
        results, error_message, final_success = [], None, True
        try:
            safe_globals = {"__builtins__": builtins, "OrderedDict": OrderedDict}
            local_namespace = {}
            exec(code_string, safe_globals, local_namespace)
            LruClass = next(v for k, v in local_namespace.items() if isinstance(v, type))
            lru_instance = None
            for op_data in test_sequence:
                op, args, expected = op_data["op"], op_data.get("args", []), op_data.get("expected")
                step_res = {"op": op, "args": args, "exp": expected, "ok": False, "err": None}
                try:
                    if op == "__init__": lru_instance = LruClass(*args)
                    elif op == "put": getattr(lru_instance, op)(*args)
                    elif op == "get":
                        output = getattr(lru_instance, op)(*args)
                        if str(output) != str(expected):
                            step_res["err"] = f"Output:'{output}' != Expected:'{expected}'"
                            final_success = False
                    step_res["ok"] = step_res["err"] is None
                except Exception as e_step:
                    step_res["err"] = f"{type(e_step).__name__}: {e_step}"
                    final_success = False
                results.append(step_res)
        except Exception as e_main:
            error_message = f"Erro fatal no worker LRU: {e_main}"
            traceback.print_exc()
            final_success = False
        result_queue.put({"success": final_success, "results": results, "error": error_message})

    def _execute_lru_locally(self, code_string: str, test_sequence: list) -> dict:
        ctx = multiprocessing.get_context('spawn')
        result_queue = ctx.Queue()
        timeout_seconds = 10
        process = ctx.Process(target=self._lru_worker, args=(code_string, test_sequence, result_queue))
        return_value = {"success": False, "results": [], "error": "Unknown execution error"}
        try:
            process.start()
            try:
                return_value = result_queue.get(timeout=timeout_seconds)
            except queue.Empty:
                return_value["error"] = f"Timeout ({timeout_seconds}s) atingido."
                logging.warning(f"Timeout para processo LRU. Tentando terminar...")
                if process.is_alive():
                    process.terminate()
                    process.join(2)
                    if process.is_alive():
                        process.kill()
                        process.join()
        except Exception as e_p:
            logging.error(f"Falha ao gerir processo worker LRU: {e_p}", exc_info=True)
            return_value["error"] = f"Process management error: {e_p}"
            if process.is_alive():
                process.kill()
        finally:
            if process.is_alive():
                process.kill()
            process.join()
            process.close()
            try:
                result_queue.close()
                result_queue.join_thread()
            except (OSError, ValueError):
                 pass
        return return_value

    def evaluate_solution(self, ai_name: str, challenge: dict, response: dict, run_idx: int):
        content, api_time, api_error = response.get("content", ""), response.get("execution_time", 0), response.get("last_error")
        if not content and api_error:
            self._handle_evaluation_error(ai_name, challenge, run_idx, self.STATUS_API_ERROR, api_error)
            return
        code = self.extract_code(content)
        lines = len(code.splitlines())
        flake8 = self.analyze_code_quality_flake8(code)
        if not code:
            self._handle_evaluation_error(ai_name, challenge, run_idx, self.STATUS_NO_CODE, "Code not extracted")
            return
        n_errors, first_err, raw_results = 0, "", []
        if challenge["name"] == self.CHALLENGE_LRU_CACHE:
            n_errors, first_err, raw_results = self._evaluate_lru_challenge(ai_name, challenge, code, run_idx)
        else:
            func_name = self._detect_function_name(code, challenge)
            if not func_name:
                self._handle_evaluation_error(ai_name, challenge, run_idx, self.STATUS_NO_CODE, "Function name not found")
                return
            n_errors, first_err, raw_results = self._evaluate_standard_challenge(ai_name, challenge, code, func_name, run_idx)
        run_summary = {"run_index": run_idx, "ai": ai_name, "challenge": challenge["name"], "difficulty": challenge["difficulty"], "api_execution_time": api_time, "lines_of_code": lines, "flake8_issues": flake8, "num_execution_errors": n_errors, "first_execution_error": first_err, "code": code}
        self.results.append(run_summary)

    def _handle_evaluation_error(self, ai_name: str, challenge: dict, run_idx: int, status: str, error_message: str):
        logging.warning(f"{error_message} para {ai_name} em {challenge['name']} (Run {run_idx+1}).")
        num_tests = len(challenge.get("test_cases", [{}]))
        self.results.append({"run_index": run_idx, "ai": ai_name, "challenge": challenge["name"], "difficulty": challenge["difficulty"], "api_execution_time": 0, "lines_of_code": 0, "flake8_issues": -1, "num_execution_errors": num_tests, "first_execution_error": error_message, "code": ""})
        for i in range(num_tests):
            self.detailed_test_results.append({"run_index": run_idx, "ai": ai_name, "challenge": challenge["name"], "test_case_index": i, "status": status, "error_message": error_message})

    def _evaluate_lru_challenge(self, ai_name: str, challenge: dict, code: str, run_idx: int) -> tuple[int, str, list]:
        logging.info(f"Executando LRU Cache de {ai_name} localmente (Run {run_idx+1})...")
        sequence = challenge.get("test_cases", [{}])[0].get("sequence")
        if not sequence:
            error = "Sequência de teste LRU inválida."
            self._handle_evaluation_error(ai_name, challenge, run_idx, self.STATUS_ERROR, error)
            return 1, error, []
        lru_res = self._execute_lru_locally(code, sequence)
        ok = lru_res.get("success", False)
        n_errors = 0 if ok else 1
        first_err = lru_res.get("error", "") if not ok else ""
        self.detailed_test_results.append({"run_index": run_idx, "ai": ai_name, "challenge": challenge["name"], "status": self.STATUS_PASS if ok else self.STATUS_FAIL, "error_message": str(first_err)[:250]})
        logging.info(f"Resultado LRU local de {ai_name} (Run {run_idx+1}): {'Sucesso' if ok else 'Falha'}")
        return n_errors, first_err, lru_res.get("results", [])

    def _evaluate_standard_challenge(self, ai_name: str, challenge: dict, code: str, func_name: str, run_idx: int) -> tuple[int, str, list]:
        logging.info(f"Executando '{challenge['name']}' de {ai_name} via Executor (Run {run_idx+1})...")
        raw, details, errors, first_e = self._run_test_cases_via_executor(ai_name, challenge, code, func_name, run_idx)
        self.detailed_test_results.extend(details)
        return errors, first_e, raw

    def _detect_function_name(self, code: str, challenge: dict) -> str | None:
        sig_match = re.search(r"^\s*(?:def|class)\s+(\w+)", challenge.get("function_signature", ""), re.MULTILINE)
        if sig_match and re.search(rf"^\s*(?:def|class)\s+{re.escape(sig_match.group(1))}\b", code, re.MULTILINE):
            return sig_match.group(1)
        first_def_match = re.search(r"^\s*(?:def|class)\s+(\w+)", code, re.MULTILINE)
        if first_def_match:
            return first_def_match.group(1)
        return None

    def _run_test_cases_via_executor(self, ai_name: str, challenge: dict, code: str, func_name: str, run_idx: int) -> tuple[list, list, int, str]:
        raw_results, details, n_fails_errors, first_err_msg = [], [], 0, ""
        for i, tc in enumerate(challenge.get("test_cases", [])):
            exec_res = self._call_executor_service(code, func_name, tc["input_args"])
            raw_results.append(exec_res)
            detail = {"run_index": run_idx, "ai": ai_name, "challenge": challenge["name"], "test_case_index": i}
            if exec_res.get("success"):
                actual, expected = exec_res.get("output"), tc["expected_output"]
                are_equal = (sorted(actual) == sorted(expected)) if (challenge["name"] == self.CHALLENGE_TWO_SUM and isinstance(actual, list)) else (np.array_equal(actual, expected))
                detail["status"] = self.STATUS_PASS if are_equal else self.STATUS_FAIL
                if not are_equal:
                    n_fails_errors += 1
                    if not first_err_msg: first_err_msg = f"Output != Expected (Case {i})"
            else:
                detail["status"] = self.STATUS_ERROR
                detail["error_message"] = exec_res.get("error", "Execution error")
                n_fails_errors += 1
                if not first_err_msg: first_err_msg = str(detail["error_message"])[:250]
            details.append(detail)
        return raw_results, details, n_fails_errors, first_err_msg

    def _call_executor_service(self, code_string: str, function_name: str, input_args: list) -> dict:
        imports = (
            "from typing import List, Dict, Tuple, Set, Optional, Any\n"
            "import re\n"
            "from collections import OrderedDict, deque, defaultdict\n"
            "from functools import lru_cache\n"
            "from string import ascii_lowercase, digits\n\n"
        )
        code_to_exec = imports + code_string
        payload = {"code_string": code_to_exec, "function_name": function_name, "test_input_args": input_args}
        try:
            resp = requests.post(self.executor_url, headers={"Content-Type": "application/json"}, json=payload, timeout=60)
            return resp.json() if resp.status_code == 200 else {"success": False, "error": f"Executor returned status {resp.status_code}"}
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Executor request error: {e}"}

    def run_benchmark(self):
        if not self.providers:
            logging.critical("ERRO: Nenhuma API configurada. Saindo.")
            return
        logging.info(f"Iniciando Benchmark para: {', '.join(self.providers.keys())} ({self.num_runs} execução(ões))")
        for run_idx in range(self.num_runs):
            logging.info(f"\n{'='*20} INICIANDO EXECUÇÃO {run_idx + 1}/{self.num_runs} {'='*20}")
            for chal in self.code_challenges:
                sig = chal.get("function_signature", "function(...)")
                prompt = (f"Implement the following Python {'class' if 'class' in sig else 'function'}.\n"
                          f"Description: {chal['description']}\nSignature:\n{sig}\n"
                          f"Provide ONLY the raw Python code definition. "
                          f"Include any necessary standard library imports (like `from collections import deque`).")
                logging.info(f"\n--- [Run {run_idx+1}] Desafio: {chal['name']} ---")
                for ai_name, provider in self.providers.items():
                    logging.info(f"  Consultando {ai_name}...")
                    response = provider.query(prompt)
                    self.evaluate_solution(ai_name, chal, response, run_idx)
                    time.sleep(1)
        self.generate_report()

    def generate_report(self):
        if not self.results:
            logging.error("Nenhum resultado para gerar relatório.")
            return
        logging.info("\n--- Gerando Relatórios Finais ---")
        # Aqui entra a sua lógica completa de geração de relatórios (DataFrames, CSVs, gráficos)
        # Exemplo simplificado para mostrar que funciona:
        df = pd.DataFrame(self.results)
        summary = df.groupby('ai').agg(
            avg_api_time=('api_execution_time', 'mean'),
            avg_flake8=('flake8_issues', lambda x: x[x!=-1].mean()),
            total_errors=('num_execution_errors', 'sum')
        ).reset_index()
        print("\n===== SUMÁRIO DO BENCHMARK =====")
        try:
            from tabulate import tabulate
            print(tabulate(summary, headers='keys', tablefmt='grid', showindex=False, floatfmt=".2f"))
        except ImportError:
            print(summary.to_string(index=False))
        # Para restaurar a funcionalidade completa, copie os métodos:
        # _prepare_dataframe, _calculate_statistics, _display_summary_table,
        # create_visualizations, e os métodos _plot_* da sua versão v32 para aqui.

# --- Ponto de Entrada ---
if __name__ == "__main__":
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(description="Executa um benchmark de programação de IA.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-n", "--num_runs", type=int, default=3, help="Número de execuções por desafio.")
    parser.add_argument("-o", "--output_dir", type=str, default="benchmark_reports", help="Diretório de saída.")
    parser.add_argument("--claude_model", type=str, help="Nome do modelo Claude.")
    parser.add_argument("--openai_model", type=str, help="Nome do modelo OpenAI.")
    parser.add_argument("--gemini_model", type=str, help="Nome do modelo Gemini.")
    parser.add_argument("--executor_url", type=str, help="URL do serviço de execução de código.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Ativa logging nível DEBUG.")
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