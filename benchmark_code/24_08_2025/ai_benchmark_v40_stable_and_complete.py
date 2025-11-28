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
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

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

    def __init__(self, num_runs: int = 3, output_dir: str = "benchmark_reports", claude_model_name: str = None, openai_model_name: str = None, gemini_model_name: str = None, executor_service_url: str = None):
        self.num_runs = max(1, num_runs)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.results = []
        self.detailed_test_results = []
        
        self.provider_configs = {}
        claude_key, openai_key, gemini_key = os.getenv("CLAUDE_API_KEY"), os.getenv("OPENAI_API_KEY"), os.getenv("GEMINI_API_KEY")
        
        if claude_key: self.provider_configs["Claude"] = {"class": ClaudeProvider, "model": claude_model_name or "claude-3-opus-20240229", "api_key": claude_key}
        if openai_key: self.provider_configs["ChatGPT"] = {"class": OpenAIProvider, "model": openai_model_name or "gpt-4-turbo", "api_key": openai_key}
        if gemini_key: self.provider_configs["Gemini"] = {"class": GeminiProvider, "model": gemini_model_name or "gemini-1.5-pro-latest", "api_key": gemini_key}

        self.executor_url = executor_service_url or os.getenv("EXECUTOR_SERVICE_URL", "https://executor-service-1029404216382.europe-southwest1.run.app/execute")
        
        self.challenge_details = {
            "Fibonacci Sequence": "Gera os N primeiros números da sequência de Fibonacci (0, 1, 1, 2, 3,...). \nTécnica comum: Iteração com duas variáveis ou recursão (risco de ineficiência).\nFinalidade: Avaliar compreensão de sequências e algoritmos básicos.",
            "Binary Search": "Procura o índice de um `target` numa lista `arr` *ordenada* de inteiros. Retorna -1 se não encontrado.\nTécnica: Divisão e conquista, comparando com o elemento do meio e descartando metade da lista.\nFinalidade: Avaliar algoritmos de busca eficientes (O(log n)) em dados ordenados.",
            "Merge Sort": "Ordena uma lista de inteiros em ordem crescente usando o algoritmo Merge Sort. Retorna uma *nova* lista ordenada.\nTécnica: Divisão e conquista recursiva (divide, ordena metades, junta metades ordenadas - merge).\nFinalidade: Avaliar implementação de algoritmos de ordenação recursivos e eficientes (O(n log n)).",
            "Count Islands in Grid": "Conta o número de 'ilhas' (grupos de 1s conectados horizontal/verticalmente) numa grelha 2D. 0 representa água.\nTécnica: Travessia de grafo (busca em profundidade - DFS ou busca em largura - BFS) para marcar ilhas visitadas.\nFinalidade: Testar algoritmos de travessia de grafos/matrizes.",
            "Palindrome Check": "Verifica se uma string `s` é um palíndromo (lê-se igual de trás para frente), ignorando caixa e caracteres não alfanuméricos.\nTécnica: Filtrar/normalizar a string e comparar com a sua versão invertida, ou usar dois ponteiros.\nFinalidade: Avaliar manipulação de strings e lógica de comparação.",
            "Valid Parentheses": "Determina se uma string com '()', '{}', '[]' é válida (parênteses abertos são fechados na ordem correta pelo tipo correspondente).\nTécnica: Uso de uma pilha (stack) para guardar parênteses de abertura e verificar correspondência ao encontrar fechos.\nFinalidade: Testar o uso correto de pilhas para validação de sequências.",
            self.CHALLENGE_LRU_CACHE: "Implementa uma cache 'Least Recently Used' com capacidade fixa. Suporta `get(key)` (retorna valor ou -1, marca como usado) e `put(key, value)` (insere/atualiza, remove o item menos usado se exceder capacidade).\nTécnica: Combinação de dicionário (hash map) para acesso O(1) e lista duplamente ligada (ou `OrderedDict`) para manter ordem de uso e permitir remoção O(1).\nFinalidade: Avaliar design de estruturas de dados customizadas.",
            self.CHALLENGE_TWO_SUM: "Dada uma lista `nums` e um `target`, retorna os *índices* de dois números que somam `target`. Assume solução única e não repetida.\nTécnica: Uso de um dicionário (hash map) para armazenar números vistos e verificar rapidamente se o 'complemento' (`target - num`) existe.\nFinalidade: Avaliar uso eficiente de dicionários para busca rápida (O(n)).",
            "Longest Substring Without Repeating Characters": "Encontra o comprimento da maior substring dentro de `s` que não contém caracteres repetidos.\nTécnica: Janela deslizante (sliding window) com um conjunto (set) ou dicionário para rastrear caracteres na janela atual e ajustar o início quando uma repetição é encontrada.\nFinalidade: Testar a técnica de janela deslizante.",
            "Minimum Edit Distance": "Calcula o número mínimo de operações (inserção, deleção, substituição) para transformar `word1` em `word2`.\nTécnica: Programação Dinâmica (algoritmo de Levenshtein), usando uma matriz 2D para armazenar distâncias parciais.\nFinalidade: Avaliar aplicação de programação dinâmica em problemas de strings.",
        }

        self.code_challenges = [
            {"name": "Fibonacci Sequence", "description": "Generate the first n numbers in the Fibonacci sequence.", "difficulty": "Easy", "function_signature": "fibonacci(n: int) -> list[int]", "test_cases": [{"input_args": [10], "expected_output": [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]}, {"input_args": [1], "expected_output": [0]}, {"input_args": [0], "expected_output": []}]},
            {"name": "Binary Search", "description": "Find the index of a target value in a sorted list.", "difficulty": "Medium", "function_signature": "binary_search(arr: list[int], target: int) -> int", "test_cases": [{"input_args": [[2, 5, 7, 8, 11, 12], 13], "expected_output": -1}, {"input_args": [[2, 5, 7, 8, 11, 12], 8], "expected_output": 3}]},
            {"name": "Merge Sort", "description": "Implement the merge sort algorithm.", "difficulty": "Medium", "function_signature": "merge_sort(arr: list[int]) -> list[int]", "test_cases": [{"input_args": [[3, 1, 4, 1, 5, 9, 2, 6]], "expected_output": [1, 1, 2, 3, 4, 5, 6, 9]}, {"input_args": [[]], "expected_output": []}]},
            {"name": "Count Islands in Grid", "description": "Count the number of islands in a 2D grid.", "difficulty": "Hard", "function_signature": "count_islands(grid: list[list[int]]) -> int", "test_cases": [{"input_args": [[[[1,1,0,0,0], [1,1,0,0,0], [0,0,1,0,0], [0,0,0,1,1]]]], "expected_output": 3}, {"input_args": [[[]]], "expected_output": 0}]},
            {"name": "Palindrome Check", "description": "Check if a given string is a palindrome.", "difficulty": "Easy", "function_signature": "is_palindrome(s: str) -> bool", "test_cases": [{"input_args": ["A man, a plan, a canal: Panama"], "expected_output": True}, {"input_args": ["race a car"], "expected_output": False}]},
            {"name": "Valid Parentheses", "description": "Determine if the input string is valid (brackets must close in the correct order).", "difficulty": "Medium", "function_signature": "is_valid_parentheses(s: str) -> bool", "test_cases": [{"input_args": ["()[]{}D"], "expected_output": True}, {"input_args": ["([)]"], "expected_output": False}]},
            {"name": self.CHALLENGE_LRU_CACHE, "description": "Implement a Least Recently Used (LRU) cache.", "difficulty": "Hard", "function_signature": "class LRUCache", "test_cases": [{"sequence": [{"op": "__init__", "args": [2]}, {"op": "put", "args": [1, 1]}, {"op": "put", "args": [2, 2]}, {"op": "get", "args": [1], "expected": 1}, {"op": "put", "args": [3, 3]}, {"op": "get", "args": [2], "expected": -1}]}]},
            {"name": self.CHALLENGE_TWO_SUM, "description": "Return indices of the two numbers such that they add up to target.", "difficulty": "Easy", "function_signature": "two_sum(nums: list[int], target: int) -> list[int]", "test_cases": [{"input_args": [[2, 7, 11, 15], 9], "expected_output": [0, 1]}, {"input_args": [[3, 2, 4], 6], "expected_output": [1, 2]}]},
            {"name": "Longest Substring Without Repeating Characters", "description": "Find the length of the longest substring without repeating characters.", "difficulty": "Medium", "function_signature": "length_of_longest_substring(s: str) -> int", "test_cases": [{"input_args": ["abcabcbb"], "expected_output": 3}, {"input_args": ["bbbbb"], "expected_output": 1}]},
            {"name": "Minimum Edit Distance", "description": "Return the minimum number of operations to convert word1 to word2.", "difficulty": "Hard", "function_signature": "min_distance(word1: str, word2: str) -> int", "test_cases": [{"input_args": ["horse", "ros"], "expected_output": 3}, {"input_args": ["intention", "execution"], "expected_output": 5}]}
        ]
        self._check_flake8()

    def _check_flake8(self):
        try:
            subprocess.run([sys.executable, "-m", "flake8", "--version"], check=True, capture_output=True, timeout=5)
            self.flake8_available = True
        except Exception:
            self.flake8_available = False

    def extract_code(self, content: str) -> str:
        if not content: return ""
        match = re.search(r"```python\n(.*?)\n```", content, re.DOTALL)
        return match.group(1).strip() if match else content.strip()

    def analyze_code_quality_flake8(self, code: str) -> int:
        if not self.flake8_available or not code: return -1
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".py", delete=False) as tf:
            path = tf.name
            tf.write(code)
        try:
            res = subprocess.run([sys.executable, "-m", "flake8", "--count", path], capture_output=True, text=True)
            return int(res.stdout.strip().splitlines()[-1]) if res.returncode in [0, 1] and res.stdout else -1
        finally:
            os.remove(path)

    def _lru_worker(self, code, sequence, result_queue):
        try:
            namespace = {"OrderedDict": OrderedDict, "__builtins__": builtins}
            exec(code, namespace)
            LruClass = next(v for v in namespace.values() if isinstance(v, type))
            instance = None
            for step in sequence:
                op, args, expected = step["op"], step.get("args", []), step.get("expected")
                if op == "__init__": instance = LruClass(*args)
                elif op == "put": getattr(instance, op)(*args)
                elif op == "get":
                    output = getattr(instance, op)(*args)
                    if str(output) != str(expected):
                        raise AssertionError(f"LRU Get Mismatch: Expected {expected}, got {output}")
            result_queue.put({"success": True, "error": None})
        except Exception as e:
            result_queue.put({"success": False, "error": str(e)})

    def _execute_lru_locally(self, code, sequence) -> dict:
        ctx = multiprocessing.get_context('spawn')
        q = ctx.Queue()
        p = ctx.Process(target=self._lru_worker, args=(code, sequence, q))
        res = {"success": False, "error": "Unknown Error"}
        try:
            p.start()
            res = q.get(timeout=10)
        except queue.Empty:
            res = {"success": False, "error": "Timeout"}
        except Exception as e:
            res = {"success": False, "error": f"Process management error: {e}"}
        finally:
            if p.is_alive():
                p.terminate()
                p.join(2)
                if p.is_alive(): p.kill()
            p.close()
            q.close()
        return res

    def evaluate_solution(self, ai_name, challenge, response, run_idx):
        content, api_time, api_error = response.get("content", ""), response.get("execution_time", 0), response.get("last_error")
        if api_error: return self._handle_evaluation_error(ai_name, challenge, run_idx, self.STATUS_API_ERROR, api_error)
        
        code = self.extract_code(content)
        if not code: return self._handle_evaluation_error(ai_name, challenge, run_idx, self.STATUS_NO_CODE, "Code not extracted")
        
        n_errors, first_err = 0, ""
        test_cases = challenge.get("test_cases", [])
        
        if challenge["name"] == self.CHALLENGE_LRU_CACHE:
            res = self._execute_lru_locally(code, test_cases[0]["sequence"])
            if not res["success"]: n_errors, first_err = 1, res["error"]
        else:
            func_name = self._detect_function_name(code, challenge)
            if not func_name: return self._handle_evaluation_error(ai_name, challenge, run_idx, self.STATUS_NO_CODE, "Function name not found")
            
            for i, tc in enumerate(test_cases):
                res = self._call_executor_service(code, func_name, tc["input_args"])
                is_correct = res["success"] and np.array_equal(res["output"], tc["expected_output"])
                if not is_correct:
                    n_errors += 1
                    if not first_err: first_err = res.get("error", "Output mismatch")

        self.results.append({"run_index": run_idx, "ai": ai_name, "challenge": challenge["name"], "difficulty": challenge["difficulty"], "api_execution_time": api_time, "flake8_issues": self.analyze_code_quality_flake8(code), "num_execution_errors": n_errors, "first_execution_error": first_err, "code": code})

    def _handle_evaluation_error(self, ai_name, challenge, run_idx, status, error_message):
        logging.warning(f"{status}: {error_message} for {ai_name} on {challenge['name']}")
        self.results.append({"run_index": run_idx, "ai": ai_name, "challenge": challenge["name"], "difficulty": challenge["difficulty"], "api_execution_time": 0, "flake8_issues": -1, "num_execution_errors": len(challenge.get("test_cases", [{}])), "first_execution_error": error_message, "code": ""})

    def _detect_function_name(self, code, challenge):
        match = re.search(r"^\s*(?:def|class)\s+(\w+)", code, re.MULTILINE)
        return match.group(1) if match else None

    def _call_executor_service(self, code, func_name, args):
        imports = ("from typing import List, Dict, Tuple, Set, Optional, Any\nimport re\n"
                   "from collections import OrderedDict, deque, defaultdict\n"
                   "from functools import lru_cache\nfrom string import ascii_lowercase, digits\n"
                   "import numpy as np\n\n")
        payload = {"code_string": imports + code, "function_name": func_name, "test_input_args": args}
        try:
            resp = requests.post(self.executor_url, json=payload, timeout=60)
            return resp.json() if resp.ok else {"success": False, "error": f"Executor error {resp.status_code}"}
        except requests.RequestException as e:
            return {"success": False, "error": str(e)}

    def run_benchmark(self):
        if not self.provider_configs: return logging.critical("Nenhuma API configurada.")
        
        logging.info(f"Iniciando Benchmark para: {', '.join(self.provider_configs.keys())} ({self.num_runs} execução(ões))")
        for run_idx in range(self.num_runs):
            logging.info(f"\n{'='*20} INICIANDO EXECUÇÃO {run_idx + 1}/{self.num_runs} {'='*20}")
            for chal in self.code_challenges:
                logging.info(f"\n--- [Run {run_idx+1}] Desafio: {chal['name']} ---")
                logging.info(f"    Propósito: {self.challenge_details.get(chal['name'], 'N/A')}")
                
                for ai_name, config in self.provider_configs.items():
                    logging.info(f"  Consultando {ai_name}...")
                    try:
                        provider = config["class"](config["model"], config["api_key"])
                        prompt = (f"Implement the Python function/class for: {chal['description']}.\n"
                                  f"Signature: {chal['function_signature']}\n"
                                  "Provide ONLY the raw Python code.")
                        response = provider.query(prompt)
                        self.evaluate_solution(ai_name, chal, response, run_idx)
                    except Exception as e:
                        logging.error(f"Falha crítica ao processar {ai_name} para {chal['name']}: {e}")
                        self._handle_evaluation_error(ai_name, chal, run_idx, self.STATUS_ERROR, str(e))
                    time.sleep(1)
        self.generate_report()

    def generate_report(self):
        if not self.results: return logging.error("Nenhum resultado para gerar relatório.")
        
        logging.info("\n--- Gerando Relatórios Finais ---")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        df = pd.DataFrame(self.results)
        
        # Salvar resultados completos
        df.to_csv(os.path.join(self.output_dir, f"benchmark_results_full_{ts}.csv"), index=False, encoding='utf-8-sig')
        
        # Análise Agregada
        agg_data = df.groupby('ai').agg(
            avg_api_time=('api_execution_time', 'mean'),
            avg_flake8=('flake8_issues', lambda x: x[x != -1].mean()),
            total_errors=('num_execution_errors', 'sum'),
            success_rate=('num_execution_errors', lambda x: 1 - (x > 0).sum() / len(x))
        ).reset_index()
        
        agg_data.to_csv(os.path.join(self.output_dir, f"benchmark_summary_{ts}.csv"), index=False, encoding='utf-8-sig')
        
        print("\n===== SUMÁRIO GERAL DO BENCHMARK =====")
        try:
            from tabulate import tabulate
            print(tabulate(agg_data, headers='keys', tablefmt='grid', showindex=False, floatfmt=".2f"))
        except ImportError:
            print(agg_data.to_string(index=False))

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