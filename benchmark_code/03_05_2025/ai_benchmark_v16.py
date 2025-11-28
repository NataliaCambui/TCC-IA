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
from collections import OrderedDict
import builtins # Import necessário

# Carrega variáveis de ambiente
load_dotenv()

class AIBenchmark:
    """
    Executa benchmark de programação comparando IAs (Claude, ChatGPT, Gemini).
    Avalia qualidade, corretude (exec externa/local) e desempenho.
    """
    def __init__(self):
        """Inicializa o benchmark."""
        self.claude_api_key = os.getenv('CLAUDE_API_KEY')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        self.results = []
        self.detailed_test_results = []
        self.output_dir = "benchmark_reports"
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Diretório de saída: '{self.output_dir}/'")
        self._configure_gemini_api()
        self.claude_model = "claude-3-opus-20240229"
        self.openai_model = "gpt-4-turbo"
        self.gemini_model = "gemini-1.5-pro-latest" if self.gemini_api_key else None
        self.executor_url = "https://executor-service-1029404216382.europe-southwest1.run.app/execute"
        self.code_challenges = [
             { "name": "Fibonacci Sequence", "description": "Generate the first n numbers in the Fibonacci sequence, starting with 0.", "difficulty": "Easy", "function_signature": "fibonacci(n: int) -> list[int]", "test_cases": [ {"input_args": [10], "expected_output": [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]}, {"input_args": [1], "expected_output": [0]}, {"input_args": [0], "expected_output": []}, {"input_args": [2], "expected_output": [0, 1]} ] },
             { "name": "Binary Search", "description": "Find the index of a target value in a sorted list of integers. Return -1 if the target is not found.", "difficulty": "Medium", "function_signature": "binary_search(arr: list[int], target: int) -> int", "test_cases": [ {"input_args": [[2, 5, 7, 8, 11, 12], 13], "expected_output": -1}, {"input_args": [[2, 5, 7, 8, 11, 12], 8], "expected_output": 3}, {"input_args": [[1, 3, 5], 1], "expected_output": 0}, {"input_args": [[1, 3, 5], 5], "expected_output": 2}, {"input_args": [[], 5], "expected_output": -1} ] },
             { "name": "Merge Sort", "description": "Implement the merge sort algorithm to sort a list of integers in ascending order. Return a new sorted list.", "difficulty": "Medium", "function_signature": "merge_sort(arr: list[int]) -> list[int]", "test_cases": [ {"input_args": [[3, 1, 4, 1, 5, 9, 2, 6]], "expected_output": [1, 1, 2, 3, 4, 5, 6, 9]}, {"input_args": [[5, 4, 3, 2, 1]], "expected_output": [1, 2, 3, 4, 5]}, {"input_args": [[1]], "expected_output": [1]}, {"input_args": [[]], "expected_output": []} ] },
             { "name": "Count Islands in Grid", "description": "Count the number of islands in a 2D grid (list of lists) where 1 is land and 0 is water. An island is formed by connecting adjacent lands horizontally or vertically.", "difficulty": "Hard", "function_signature": "count_islands(grid: list[list[int]]) -> int", "test_cases": [ {"input_args": [[ [1, 1, 1, 1, 0], [1, 1, 0, 1, 0], [1, 1, 0, 0, 0], [0, 0, 0, 0, 0] ]], "expected_output": 1}, {"input_args": [[ [1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 1] ]], "expected_output": 3}, {"input_args": [[]], "expected_output": 0}, {"input_args": [[ [0, 0, 0], [0, 0, 0] ]], "expected_output": 0} ] },
             { "name": "Palindrome Check", "description": "Check if a given string is a palindrome, ignoring case and non-alphanumeric characters. Return True if it is, False otherwise.", "difficulty": "Easy", "function_signature": "is_palindrome(s: str) -> bool", "test_cases": [ {"input_args": ["A man, a plan, a canal: Panama"], "expected_output": True}, {"input_args": ["race a car"], "expected_output": False}, {"input_args": [" "], "expected_output": True}, {"input_args": [""], "expected_output": True}, {"input_args": ["Was it a car or a cat I saw?"], "expected_output": True}, {"input_args": ["tab a cat"], "expected_output": False} ] },
             { "name": "Valid Parentheses", "description": "Given a string containing just '(', ')', '{', '}', '[' and ']', determine if the input string is valid (brackets must close in the correct order).", "difficulty": "Medium", "function_signature": "is_valid_parentheses(s: str) -> bool", "test_cases": [ {"input_args": ["()"], "expected_output": True}, {"input_args": ["()[]{}"], "expected_output": True}, {"input_args": ["(]"], "expected_output": False}, {"input_args": ["([)]"], "expected_output": False}, {"input_args": ["{[]}"], "expected_output": True}, {"input_args": [""], "expected_output": True}, {"input_args": ["["], "expected_output": False}, {"input_args": ["]"], "expected_output": False} ] },
             { "name": "LRU Cache", "description": "Implement a Least Recently Used (LRU) cache with a given capacity. Support get(key) and put(key, value) operations. get returns value or -1. put inserts/updates; evicts LRU item if capacity exceeded.", "difficulty": "Hard", "function_signature": "class LRUCache:\n    def __init__(self, capacity: int):\n        pass\n    def get(self, key: int) -> int:\n        pass\n    def put(self, key: int, value: int) -> None:\n        pass", "test_cases": [ {"description": "Simple put/get sequence", "sequence": [ {"op": "__init__", "args": [2]}, {"op": "put", "args": [1, 1]}, {"op": "put", "args": [2, 2]}, {"op": "get", "args": [1], "expected": 1}, {"op": "put", "args": [3, 3]}, {"op": "get", "args": [2], "expected": -1}, {"op": "put", "args": [4, 4]}, {"op": "get", "args": [1], "expected": -1}, {"op": "get", "args": [3], "expected": 3}, {"op": "get", "args": [4], "expected": 4} ] }, ] }
        ]
        self._check_flake8()
        self.environment_info = self._get_environment_info()

    def _configure_gemini_api(self):
        """Configura a API Gemini."""
        if self.gemini_api_key:
            try: genai.configure(api_key=self.gemini_api_key); print("Chave API Gemini configurada.")
            except Exception as e: print(f"Erro configurar API Gemini: {e}. AI pulada."); self.gemini_api_key = None
        else: print("AVISO: Chave GEMINI_API_KEY não encontrada. Gemini AI pulada.")

    def _get_environment_info(self):
        """Coleta informações do ambiente."""
        info = {"Timestamp": datetime.now().isoformat(), "OS": f"{platform.system()} {platform.release()}", "Python Version": sys.version.split()[0], "Models": {"Claude": self.claude_model if self.claude_api_key else "N/A", "ChatGPT": self.openai_model if self.openai_api_key else "N/A", "Gemini": self.gemini_model if self.gemini_api_key else "N/A"}, "Executor Service URL": self.executor_url, "Local LRU Cache Execution": True, "Libraries": {}}
        libs = ['pandas', 'numpy', 'matplotlib', 'requests', 'python-dotenv', 'google-generativeai', 'google-api-core']
        try:
            for lib in libs: info["Libraries"][lib] = importlib_metadata.version(lib)
        except (NameError, importlib_metadata.PackageNotFoundError):
            try: # Fallback para pkg_resources ou Not Found
                 import pkg_resources
                 for lib in libs:
                      try: info["Libraries"][lib] = pkg_resources.get_distribution(lib).version
                      except pkg_resources.DistributionNotFound: info["Libraries"][lib] = "Not Found"
                      except Exception: info["Libraries"][lib] = "Error getting version"
            except ImportError: # Se nem pkg_resources existe
                 for lib in libs: info["Libraries"][lib] = "Unknown (importlib/pkg_resources missing)"
        return info

    def _check_flake8(self):
        """Verifica disponibilidade do flake8."""
        try: subprocess.run([sys.executable, '-m', 'flake8', '--version'], check=True, capture_output=True, timeout=5); print("Verificação Flake8: OK"); self.flake8_available = True
        except Exception: print(f"AVISO: flake8 não encontrado via '{sys.executable} -m flake8'."); self.flake8_available = False

    def query_claude(self, prompt: str) -> dict:
        """Consulta API Claude."""
        start_time = time.time(); content = ""; execution_time = 0
        if not self.claude_api_key: print("Pulando Claude: Chave API N/A."); return {"content": "", "execution_time": 0}
        headers = {"x-api-key": self.claude_api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}; data = {"model": self.claude_model, "max_tokens": 4000, "messages": [{"role": "user", "content": prompt}]}
        try:
            response = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=data, timeout=120); response.raise_for_status(); result = response.json()
            if ("content" in result and isinstance(result["content"], list) and len(result["content"]) > 0): first_content = result["content"][0]; content = first_content.get("text", "") if isinstance(first_content, dict) else ""
            else: print(f"Resposta Claude inesperada: {result}")
        except requests.exceptions.RequestException as e:
            error_message = f"Erro req Claude: {e}";
            if e.response is not None: error_message += f" | Status: {e.response.status_code}"; try: error_message += f" | Body: {e.response.json()}"; except json.JSONDecodeError: error_message += f" | Body: {e.response.text[:100]}..."
            print(error_message)
        except Exception as e: print(f"Erro inesperado Claude: {type(e).__name__} - {e}")
        execution_time = time.time() - start_time; return {"content": content or "", "execution_time": execution_time}

    def query_chatgpt(self, prompt: str) -> dict:
        """Consulta API ChatGPT."""
        start_time = time.time(); content = ""; execution_time = 0
        if not self.openai_api_key: print("Pulando ChatGPT: Chave API N/A."); return {"content": "", "execution_time": 0}
        headers = {"Authorization": f"Bearer {self.openai_api_key}", "Content-Type": "application/json"}; data = {"model": self.openai_model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 4000}
        try:
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data, timeout=120); response.raise_for_status(); result = response.json()
            if ("choices" in result and isinstance(result["choices"], list) and len(result["choices"]) > 0):
                first_choice = result["choices"][0]
                if (isinstance(first_choice, dict) and "message" in first_choice and isinstance(first_choice["message"], dict)): content = first_choice["message"].get("content", "")
                else: print(f"Resposta ChatGPT inesperada (message): {result}")
            else: print(f"Resposta ChatGPT inesperada (choices): {result}")
        except requests.exceptions.RequestException as e:
            error_message = f"Erro req ChatGPT: {e}";
            if e.response is not None: error_message += f" | Status: {e.response.status_code}"; try: error_details = e.response.json(); error_msg = error_details.get('error', {}).get('message', ''); error_message += f" | API Error: {error_msg}" if error_msg else f" | Body: {error_details}"; except json.JSONDecodeError: error_message += f" | Body: {e.response.text[:100]}..."
            print(error_message)
        except Exception as e: print(f"Erro inesperado ChatGPT: {type(e).__name__} - {e}")
        execution_time = time.time() - start_time; return {"content": content or "", "execution_time": execution_time}

    def query_gemini(self, prompt: str) -> dict:
        """Consulta API Gemini."""
        start_time = time.time(); content = ""; execution_time = 0
        if not self.gemini_api_key or not self.gemini_model: print("Pulando Gemini: Chave/Modelo N/A."); return {"content": "", "execution_time": 0}
        try:
            model = genai.GenerativeModel(self.gemini_model); safety_settings=[{"category": c, "threshold": "BLOCK_NONE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
            response = model.generate_content(prompt, safety_settings=safety_settings); execution_time = time.time() - start_time
            try: content = response.text
            except ValueError: print(f"Resposta Gemini bloqueada?"); try: print(f"  Feedback: {response.prompt_feedback}"); if hasattr(response, 'candidates') and response.candidates: print(f"  Reason: {response.candidates[0].finish_reason}"); except Exception as fe: print(f"  Erro feedback: {fe}")
        except google.api_core.exceptions.ResourceExhausted as e: print(f"Erro Gemini (Rate Limit): {e}");
        except Exception as e: print(f"Erro inesperado Gemini: {type(e).__name__} - {e}")
        if execution_time == 0: execution_time = time.time() - start_time # Garante tempo mesmo em erro
        return {"content": content or "", "execution_time": execution_time}

    def extract_code(self, content: str) -> str:
        """Extrai bloco de código Python."""
        if not content: return ""; code_blocks_python = re.findall(r'```python\n(.*?)\n```', content, re.DOTALL);
        if code_blocks_python: return code_blocks_python[0].strip(); code_blocks_generic = re.findall(r'```(?:.*?\n)?(.*?)\n```', content, re.DOTALL);
        if code_blocks_generic: return code_blocks_generic[0].strip(); lines = content.split('\n'); code_lines = []; start_found = False
        for line in lines: stripped = line.strip();
            if not start_found and (line.startswith(' ') or line.startswith('\t') or stripped.startswith('def ') or stripped.startswith('class ')): start_found = True
            if start_found: code_lines.append(line)
        if code_lines: return '\n'.join(code_lines).strip();
        if len(content.splitlines()) < 3 and ("def " not in content and "class " not in content): return ""
        else: return content.strip() # Fallback final retorna tudo

    def count_code_lines(self, code: str) -> int:
        """Conta linhas de código reais."""
        if not code: return 0; return len([ln for ln in code.splitlines() if ln.strip() and not ln.strip().startswith('#')])

    def measure_response_completeness(self, code: str, description: str) -> int:
        """Estima completude da resposta (0-100)."""
        score = 0; lines = self.count_code_lines(code);
        if lines > 0:
            score += 50;
            if "def " in code or "class " in code: score += 10
            if "return " in code or "yield " in code: score += 10
            elif "island" in description.lower() and ("dfs(" in code or "bfs(" in code): score += 5
            desc_l = description.lower(); code_l = code.lower(); kw_bonus = 0;
            if "fibonacci" in desc_l and ("fibonacci" in code_l or "fib(" in code_l or "yield" in code): kw_bonus = 10
            elif "binary search" in desc_l and ("mid =" in code or "mid=" in code or "low" in code or "high" in code): kw_bonus = 10
            elif "merge sort" in desc_l and ("merge(" in code_l or ("left" in code and "right" in code and "mid" in code)): kw_bonus = 10
            elif "island" in desc_l and ("dfs(" in code_l or "bfs(" in code_l or "visit" in code_l or ("grid[" in code and "==" in code)): kw_bonus = 10
            elif "palindrome" in desc_l and ("[::-1]" in code or ".isalnum()" in code_l or ".lower()" in code_l): kw_bonus = 10
            elif "parentheses" in desc_l and ("stack" in code_l or "append(" in code or "pop(" in code): kw_bonus = 10
            elif "lru cache" in desc_l and ("dict" in code_l or "capacity" in code_l or "OrderedDict" in code): kw_bonus = 10
            score += kw_bonus; struct_bonus = 0;
            if "if" in code or "for" in code or "while" in code: struct_bonus = 10
            if struct_bonus > 0 and ("if not" in code or "if len(" in code or "== 0" in code or "is None" in code): struct_bonus += 5
            score += min(struct_bonus, 15)
        return min(score, 100)

    def analyze_code_quality_flake8(self, code_string: str) -> int:
        """Analisa código com Flake8."""
        if not self.flake8_available or not code_string or self.count_code_lines(code_string) == 0: return -1; tfp = None
        try:
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.py', delete=False, encoding='utf-8') as tf: tf.write(code_string); tfp = tf.name
            exe = sys.executable; res = subprocess.run([exe, '-m', 'flake8', '--count', tfp], capture_output=True, text=True, timeout=15, check=False)
            if res.returncode in [0, 1] and res.stdout: try: return int(res.stdout.strip().splitlines()[-1]); except: return len(res.stdout.strip().splitlines()) # Fallback
            elif res.stderr: print(f"Erro flake8: {res.stderr}"); return -1
            else: return 0
        except subprocess.TimeoutExpired: print("Erro: Flake8 timeout."); return -1
        except Exception as e: print(f"Erro inesperado flake8: {e}"); return -1
        finally:
            if tfp and os.path.exists(tfp): try: os.remove(tfp); except Exception as er: print(f"Aviso: Falha remover temp file: {er}")

    # --- Métodos Execução Local LRU Cache ---

    def _lru_worker(self, code_string: str, test_sequence: list, result_queue: multiprocessing.Queue):
        """Função worker para executar LRU Cache localmente com restrições."""
        results = []; error_message = None; lru_instance = None; final_success = True
        try:
            lines = code_string.splitlines()
            filtered_lines = [ln for ln in lines if not (ln.strip().startswith("from collections import OrderedDict") or ln.strip().startswith("import collections") or ln.strip().startswith("import OrderedDict"))]
            processed_code_string = "\n".join(filtered_lines)

            safe_builtins = {
                'int': int, 'str': str, 'list': list, 'dict': dict, 'tuple': tuple,
                'len': len, 'range': range, 'print': print, 'True': True, 'False': False,
                'None': None, 'object': object, 'Exception': Exception, 'RuntimeError': RuntimeError,
                'ValueError': ValueError, 'NameError': NameError, 'AttributeError': AttributeError,
                'TypeError': TypeError, 'StopIteration': StopIteration,
                '__build_class__': builtins.__build_class__, # Correção
                'getattr': getattr, 'isinstance': isinstance,
                '__name__': '__exec_sandbox__' # <--- CORREÇÃO: Adicionado aqui
            }
            safe_globals = {"__builtins__": safe_builtins, "OrderedDict": OrderedDict}

            local_namespace = {}; exec(processed_code_string, safe_globals, local_namespace)

            class_name = None
            for name, obj in local_namespace.items():
                if isinstance(obj, type) and name != 'OrderedDict': class_name = name; break
            if not class_name:
                match = re.search(r"^\s*class\s+(\w+)", processed_code_string, re.MULTILINE)
                if match: class_name = match.group(1)
                else: raise NameError("Nome classe LRUCache não encontrado.")
            LruClass = local_namespace.get(class_name)
            if not LruClass: raise NameError(f"Classe '{class_name}' não definida.")

            op_total_time = 0.0
            for op_data in test_sequence:
                op_start_time = time.perf_counter()
                operation = op_data["op"]; args = op_data.get("args", []); expected = op_data.get("expected")
                step_res = {"op": operation, "args": args, "exp": expected, "out": None, "ok": False, "err": None, "t_ms": -1.0} # Nomes curtos
                try:
                    if operation == "__init__": lru_instance = LruClass(*args); step_res["ok"] = True; step_res["out"] = "OK"
                    elif lru_instance is None: raise RuntimeError("LRUCache não inicializada.")
                    elif operation in ["put", "get"]:
                         method = getattr(lru_instance, operation)
                         if operation == "put": method(*args); step_res["ok"] = True; step_res["out"] = "OK"
                         elif operation == "get":
                              output = method(*args); step_res["out"] = output
                              if str(output) == str(expected): step_res["ok"] = True
                              else: step_res["err"] = f"Out:'{output}'({type(output)})!=Exp:'{expected}'({type(expected)})"; final_success = False
                    else: raise ValueError(f"Operação LRU inválida: {operation}")
                except Exception as e_step: step_res["err"] = f"{type(e_step).__name__}: {str(e_step).splitlines()[0]}"; final_success = False
                op_end_time = time.perf_counter(); step_res["t_ms"] = (op_end_time - op_start_time) * 1000
                op_total_time += (op_end_time - op_start_time); results.append(step_res)
                # if not step_res["ok"]: break # Opcional: parar no erro

            # Simplificando nome das chaves no log
            print(f"    LRU Worker done. Total step time: {op_total_time*1000:.2f} ms")

        except Exception as e_main:
            error_message = f"Erro worker LRU: {type(e_main).__name__} - {str(e_main).splitlines()[0]}"
            print(f"    ERRO WORKER LRU: {error_message}"); traceback.print_exc(); final_success = False

        try: result_queue.put({"success": final_success, "results": results, "error": error_message})
        except Exception as q_err: print(f"    ERRO fila worker LRU: {q_err}")

    def _execute_lru_locally(self, code_string: str, test_sequence: list) -> dict:
        """Gere o processo para execução local do LRU Cache."""
        ctx = multiprocessing.get_context(None); result_queue = ctx.Queue(); timeout_seconds = 10
        process = ctx.Process(target=self._lru_worker, args=(code_string, test_sequence, result_queue))
        return_value = {"success": False, "results": [], "error": "Unknown execution error"}; terminated = False
        try:
            start_time = time.time(); process.start(); process.join(timeout=timeout_seconds)
            if process.is_alive():
                print(f"  AVISO: Timeout ({timeout_seconds}s) processo LRU. Terminando."); process.terminate(); process.join(1)
                if process.is_alive(): print("  AVISO: Forçando kill processo LRU."); process.kill(); process.join()
                return_value = {"success": False, "results": [], "error": f"Timeout ({timeout_seconds}s)"}; terminated = True
            elapsed_time = time.time() - start_time
            if not terminated:
                exit_code = process.exitcode
                if exit_code == 0:
                    try: result = result_queue.get(timeout=1); print(f"  Exec LRU local OK ({elapsed_time:.2f}s). Worker success: {result.get('success')}"); return_value = result
                    except queue.Empty: print("  ERRO: Processo LRU OK, mas fila vazia."); return_value = {"success": False, "results": [], "error": "Worker OK but no result"}
                    except Exception as e_q: print(f"  ERRO: Falha obter resultado fila LRU: {e_q}"); return_value = {"success": False, "results": [], "error": f"Queue error: {e_q}"}
                else: print(f"  ERRO: Processo LRU terminou c/ exitcode: {exit_code}"); return_value = {"success": False, "results": [], "error": f"Worker exitcode {exit_code}"}
        except Exception as e_p: print(f"  ERRO: Falha gerir processo LRU: {e_p}"); return_value = {"success": False, "results": [], "error": f"Process mgmt error: {e_p}"};
             if process and process.is_alive(): try: process.kill(); process.join(); except: pass
        finally:
            try: result_queue.close(); result_queue.join_thread();
                 if process: process.close()
            except Exception as e_c: print(f"  Aviso: Erro limpar recursos proc LRU: {e_c}")
        return return_value

    # --- Fim métodos execução local ---

    def evaluate_solution(self, ai_name: str, challenge: dict, response: dict) -> dict:
        """Avalia a solução da IA (exec local p/ LRU, externa p/ outros)."""
        content = response.get("content", ""); api_execution_time = response.get("execution_time", 0)
        code = self.extract_code(content); lines = self.count_code_lines(code)
        completeness = self.measure_response_completeness(code, challenge["description"])
        flake8_issues = self.analyze_code_quality_flake8(code)

        exec_results_raw = []; details_curr = []; n_errors = 0; first_err = ""
        can_exec = bool(code); func_name = None

        # Passar o dict 'challenge' para _detect_function_name
        if can_exec and challenge["name"] != "LRU Cache":
            func_name = self._detect_function_name(code, challenge) # <--- Passa 'challenge'
            if not func_name: can_exec = False

        if can_exec:
            if challenge["name"] == "LRU Cache":
                print(f"  Executando LRU Cache de {ai_name} localmente...")
                test_cases = challenge.get("test_cases", [])
                if test_cases and isinstance(test_cases[0], dict) and "sequence" in test_cases[0]:
                    lru_res = self._execute_lru_locally(code, test_cases[0]["sequence"])
                    exec_results_raw = lru_res.get("results", []) # Passos detalhados
                    ok = lru_res.get("success", False); n_errors = 0 if ok else 1
                    first_err = lru_res.get("error", "")
                    if not first_err and not ok:
                        for step in exec_results_raw:
                            if step.get("err"): first_err = f"Step {step['op']}{step['args']}: {step['err']}"; break
                    if not first_err and not ok: first_err = "LRU exec falhou (erro passo desconhecido)"
                    t_ok = [s['t_ms'] for s in exec_results_raw if s.get('t_ms',-1)>=0]; avg_t = np.mean(t_ok) if t_ok else -1.0
                    details_curr.append({ # Entrada única para CSV
                        "ai": ai_name, "challenge": challenge["name"], "difficulty": challenge["difficulty"],
                        "test_case_index": 0, "input_args_str": json.dumps(test_cases[0]["sequence"]),
                        "expected_output_str": "Ver passos", "actual_output_str": "Ver passos",
                        "status": "Pass" if ok else "Fail" if exec_results_raw else "Error",
                        "execution_time_ms": avg_t, "error_message": str(first_err)[:250], "stdout": None #<-- CORREÇÃO str()
                    })
                    print(f"  Resultado LRU local: {'Sucesso' if ok else 'Falha/Erro'}")
                else:
                    print("  AVISO: Casos de teste LRU inválidos."); n_errors = 1; first_err = "LRU test case inválido."
                    details_curr.append({"ai": ai_name, "challenge": challenge["name"], "difficulty": challenge["difficulty"], "test_case_index": 0, "input_args_str": "N/A", "expected_output_str": "N/A", "actual_output_str": None, "status": "Error", "execution_time_ms": -1, "error_message": first_err, "stdout": None })
            else: # Outros desafios (func_name é válido)
                print(f"  Executando '{challenge['name']}' de {ai_name} via Executor Service...")
                raw, details, errors, first_e = self._run_test_cases(ai_name, challenge, code, func_name)
                exec_results_raw = raw; n_errors = errors; first_err = first_e; details_curr.extend(details)
        else: # Sem código ou nome de função
            err_msg = "Código não extraído" if not code else "Nome func/classe não detectado"
            print(f"  AVISO: {err_msg} para '{challenge['name']}' por {ai_name}. Pulando execução.")
            n_tests = len(challenge.get("test_cases", [])) if challenge["name"] != "LRU Cache" else 1
            n_errors = n_tests; first_err = err_msg
            for i in range(n_tests):
                 details_curr.append({"ai": ai_name, "challenge": challenge["name"], "difficulty": challenge["difficulty"], "test_case_index": i, "input_args_str": None, "expected_output_str": None, "actual_output_str": None, "status": "No Code", "execution_time_ms": -1, "error_message": err_msg, "stdout": None })

        # Adiciona detalhes gerados nesta chamada à lista principal
        self.detailed_test_results.extend(details_curr)

        return { # Retorna dicionário agregado
            "ai": ai_name, "challenge": challenge["name"], "difficulty": challenge["difficulty"],
            "api_execution_time": api_execution_time, "lines_of_code": lines,
            "completeness": completeness, "flake8_issues": flake8_issues,
            "num_execution_errors": n_errors, "first_execution_error": first_err,
            "code": code, "full_response": content, "execution_results_raw": exec_results_raw
        }

    def _detect_function_name(self, code: str, challenge: dict) -> str | None:
        """Detecta nome da função/classe, priorizando assinatura."""
        if not code: return None; expected_name = None
        signature = challenge.get("function_signature", "")
        sig_match = re.match(r"^\s*(?:def|class)\s+(\w+)", signature)
        if sig_match:
            expected_name = sig_match.group(1)
            # --- CORREÇÃO: Regex menos restrito (sem ^) ---
            pattern = rf"\s*(?:def|class)\s+{re.escape(expected_name)}\b"
            exact_match = re.search(pattern, code, re.MULTILINE)
            if exact_match:
                 # Não imprime aqui para reduzir ruído, só no fallback
                 # print(f"  Nome esperado '{expected_name}' encontrado.")
                 return expected_name
            # else: print(f"  AVISO: Nome esperado '{expected_name}' não encontrado.") # Log opcional

        # Fallback: primeira definição encontrada
        first_match = re.search(r"^\s*(?:def|class)\s+(\w+)", code, re.MULTILINE)
        if first_match:
            found_name = first_match.group(1)
            if not expected_name or found_name != expected_name:
                 print(f"  AVISO: Usando 1ª definição '{found_name}' (Esperado: '{expected_name or 'N/A'}').")
            return found_name
        return None # Não encontrou nada

    def _run_test_cases(self, ai_name: str, challenge: dict, code: str,
                        actual_function_name: str) -> tuple[list, list, int, str]:
        """Executa casos de teste via Executor Service."""
        exec_results_raw = []; details = []; n_fails_errors = 0; first_err = ""
        test_cases = challenge.get("test_cases", [])
        if not test_cases: print(f"  Aviso: Sem casos teste p/ '{challenge['name']}'."); return [], [], 0, ""
        for i, tc in enumerate(test_cases):
            args = tc.get("input_args"); exp_out = tc.get("expected_output")
            if args is None: print(f"  Aviso: Caso {i+1} '{challenge['name']}' s/ 'input_args'."); continue
            exec_res = self._call_executor_service(code, actual_function_name, args); exec_results_raw.append(exec_res)
            t_det = {"ai": ai_name, "challenge": challenge["name"], "difficulty": challenge["difficulty"], "test_case_index": i, "input_args_str": json.dumps(args), "expected_output_str": json.dumps(exp_out), "actual_output_str": None, "status": "Unknown", "execution_time_ms": exec_res.get("execution_time_ms", -1), "error_message": None, "stdout": exec_res.get("stdout")}
            case_err = None
            if exec_res.get("success") is True:
                act_out = exec_res.get("output"); t_det["actual_output_str"] = json.dumps(act_out)
                try:
                    eq = np.array_equal(np.array(act_out, dtype=object), np.array(exp_out, dtype=object)); t_det["status"] = "Pass" if eq else "Fail"
                    if not eq: case_err = f"Caso {i} Falhou: Output != Expected"; n_fails_errors += 1
                except Exception as comp_e: t_det["status"] = "Fail (Comp Error)"; t_det["error_message"] = f"Erro comparar: {comp_e}"; case_err = t_det["error_message"]; n_fails_errors += 1
            elif exec_res.get("success") is False: t_det["status"] = "Error"; t_det["error_message"] = exec_res.get("error", "Erro execução desconhecido"); case_err = str(t_det["error_message"])[:250]; n_fails_errors += 1
            else: t_det["status"] = "Unknown"; t_det["error_message"] = exec_res.get("error", "Executor sem status sucesso"); case_err = str(t_det["error_message"])[:250]; n_fails_errors += 1
            if case_err and not first_err: first_err = case_err
            details.append(t_det)
        if n_fails_errors > 0 and not first_err: first_err = "Um ou mais casos falharam na comparação."
        return exec_results_raw, details, n_fails_errors, first_err

    def _call_executor_service(self, code_string: str, function_name: str, input_args: list) -> dict:
        """Chama o serviço executor externo."""
        imports = ("from typing import List, Dict, Tuple, Set, Optional, Any\n""import re\n""from collections import OrderedDict, deque\n\n")
        code_to_exec = imports + code_string; payload = {"code_string": code_to_exec, "function_name": function_name, "test_input_args": input_args}
        headers = {"Content-Type": "application/json"}; default_err = {"success": False, "output": None, "stdout": None, "error": "Falha chamada executor", "execution_time_ms": -1}
        try:
            resp = requests.post(self.executor_url, headers=headers, json=payload, timeout=60)
            if resp.status_code == 200:
                try: return resp.json()
                except json.JSONDecodeError as json_e: print(f"  Erro JSON executor: {json_e}"); default_err["error"] = f"Executor JSON decode error: {json_e}"; return default_err
            else: err_txt = resp.text[:200]; print(f"  Erro executor status {resp.status_code}: {err_txt}..."); default_err["error"] = f"Executor status {resp.status_code}: {err_txt}"; return default_err
        except requests.exceptions.Timeout: print(f"  Erro Timeout executor ({self.executor_url})."); default_err["error"] = "Executor timeout"; return default_err
        except requests.exceptions.RequestException as req_e: print(f"  Erro conexão executor: {req_e}"); default_err["error"] = f"Executor connection error: {req_e}"; return default_err
        except Exception as e: print(f"  Erro inesperado executor: {type(e).__name__} - {e}"); default_err["error"] = f"Executor unexpected error: {type(e).__name__}"; return default_err

    def run_benchmark(self):
        """Executa o ciclo completo do benchmark."""
        can_claude = bool(self.claude_api_key); can_openai = bool(self.openai_api_key); can_gemini = bool(self.gemini_api_key and self.gemini_model)
        enabled_ais = [n for n, can in [("Claude", can_claude), ("ChatGPT", can_openai), ("Gemini", can_gemini)] if can]
        if not enabled_ais: print("ERRO: Nenhuma API configurada."); return
        print(f"Iniciando Benchmark para: {', '.join(enabled_ais)}"); print(f"  Claude: {self.claude_model if can_claude else 'N/A'} | ChatGPT: {self.openai_model if can_openai else 'N/A'} | Gemini: {self.gemini_model if can_gemini else 'N/A'}"); print(f"  LRU Exec: Local"); print("="*42)
        ch_details = { "Fibonacci Sequence": "Téc: Iterativa/Recursiva. Fin: Algoritmos básicos, sequências, casos base.", "Binary Search": "Téc: Dividir/Conquistar. Fin: Busca binária, índices, condições parada.", "Merge Sort": "Téc: Dividir/Conquistar (recursão/merge). Fin: Ordenação recursiva, divisão, intercalação.", "Count Islands in Grid": "Téc: Travessia Grafo (DFS/BFS). Fin: Modelar grafo, DFS/BFS, matriz 2D, visitados.", "Palindrome Check": "Téc: Manipulação Strings. Fin: Processar strings, ignorar chars/case, simetria.", "Valid Parentheses": "Téc: Pilha (Stack). Fin: Rastrear/Validar ordem parênteses.", "LRU Cache": "Téc: Estrutura Dados Custom (dict/OrderedDict). Fin: Estruturas complexas, capacidade, get/put O(1), despejo LRU." }

        for chal in self.code_challenges:
            sig = chal.get("function_signature", f"{chal['name'].lower().replace(' ', '_')}(...)")
            if "class LRUCache" in sig: prompt = (f"Implement Python class: {chal['description']}. Signature:\n{sig}\nProvide ONLY Python code for class and imports (like `from collections import OrderedDict`). No explanations.")
            else: prompt = ( f"Write Python function: `{sig}`. Problem: {chal['description']}. Provide ONLY Python code for function and imports. No explanations.")
            print(f"\n--- Desafio: {chal['name']} ({chal['difficulty']}) ---"); print(f"  {ch_details.get(chal['name'], 'N/A.')}")
            resps = {};
            if can_claude: print("Consultando Claude AI..."); resps["Claude"] = self.query_claude(prompt); time.sleep(1)
            if can_openai: print("Consultando ChatGPT..."); resps["ChatGPT"] = self.query_chatgpt(prompt); time.sleep(1)
            if can_gemini: print("Consultando Gemini AI..."); resps["Gemini"] = self.query_gemini(prompt); print("Aguardando 31s API Gemini..."); time.sleep(31)
            summary = []
            for ai in enabled_ais:
                resp_data = resps.get(ai, {"content": "", "execution_time": 0});
                result = self.evaluate_solution(ai, chal, resp_data); self.results.append(result)
                n_err_fail = result.get('num_execution_errors', 0);
                det = [d for d in self.detailed_test_results if d['ai'] == ai and d['challenge'] == chal['name']]
                n_att = len([d for d in det if d['status'] not in ['No Code']]); n_pass = len([d for d in det if d['status'] == 'Pass'])
                f8 = (str(result['flake8_issues']) if result['flake8_issues'] != -1 else 'N/A')
                if n_att == 0: status = "Not Run/No Code"
                elif chal['name'] == "LRU Cache": status = det[0]['status'] if det else "Error"
                else: status = f"{n_pass}/{n_att} passed"
                summary.append(f"  {ai:<10}: Exec: {status}. Flake8: {f8}. API Time: {result['api_execution_time']:.2f}s.")
            print(f"--- Resumo Desafio {chal['name']} ---"); [print(s) for s in summary]; print(f"--- Desafio {chal['name']} concluído ---")
        if not self.results: print("\nNenhum resultado coletado."); return
        self.generate_report()

    def _prepare_dataframe(self) -> pd.DataFrame | None:
        if not self.results: print("Não há resultados p/ DF."); return None
        try: df = pd.DataFrame(self.results); agg = self._calculate_aggregated_test_metrics(); df['avg_exec_success_rate']=df.apply(lambda r: agg.get((r['ai'], r['challenge']), {}).get('avg_exec_success_rate',0.0), axis=1); df['avg_exec_time_ms']=df.apply(lambda r: agg.get((r['ai'], r['challenge']), {}).get('avg_exec_time_ms',np.nan), axis=1); df['total_tests_passed']=df.apply(lambda r: agg.get((r['ai'], r['challenge']), {}).get('total_tests_passed',0), axis=1); df['total_tests_attempted']=df.apply(lambda r: agg.get((r['ai'], r['challenge']), {}).get('total_tests_attempted',0), axis=1); df=df.drop(['execution_results_raw'], axis=1, errors='ignore'); self._normalize_dataframe_types(df); return df
        except Exception as e: print(f"Erro criar/processar DF: {e}"); traceback.print_exc(); return None

    def _calculate_aggregated_test_metrics(self) -> dict:
        agg = {}; keys = set((d['ai'], d['challenge']) for d in self.detailed_test_results)
        for k in keys:
            ai, chal = k; dets = [d for d in self.detailed_test_results if d['ai']==ai and d['challenge']==chal]
            if not dets: continue; passed=[d for d in dets if d['status']=='Pass']; attempted=[d for d in dets if d['status']!='No Code']
            times=[d['execution_time_ms'] for d in dets if d.get('execution_time_ms',-1)>=0]
            n_att=len(attempted); n_pass=len(passed); rate=n_pass/n_att if n_att>0 else 0.0; mean_t=np.mean(times) if times else np.nan
            agg[k]={'avg_exec_success_rate': rate,'avg_exec_time_ms':mean_t,'total_tests_passed':n_pass,'total_tests_attempted':n_att}
        return agg

    def _normalize_dataframe_types(self, df: pd.DataFrame):
        cols = ['api_execution_time','lines_of_code','completeness','flake8_issues','num_execution_errors','avg_exec_success_rate','avg_exec_time_ms','total_tests_passed','total_tests_attempted']
        for c in cols:
            if c in df.columns: df[c]=pd.to_numeric(df[c],errors='coerce');
                if c=='flake8_issues': df.loc[:,c]=df[c].fillna(-1).astype(int)
                elif c=='num_execution_errors': df.loc[:,c]=df[c].fillna(0).astype(int)
                elif c=='avg_exec_success_rate': df.loc[:,c]=df[c].fillna(0.0)
                elif c=='avg_exec_time_ms': df.loc[:,c]=df[c].fillna(-1.0)
                elif c in ['total_tests_passed','total_tests_attempted']: df.loc[:,c]=df[c].fillna(0).astype(int)
                else: df.loc[:,c]=df[c].fillna(0.0)
        if 'first_execution_error' in df.columns: df['first_execution_error']=df['first_execution_error'].fillna('').astype(str)

    def _save_environment_info(self, ts: str):
        p = os.path.join(self.output_dir, f"environment_info_{ts}.txt"); try:
            with open(p,"w",encoding='utf-8') as f: [f.write(f"{k}: {v}\n") if not isinstance(v,dict) else f.write(f"{k}:\n") or [f.write(f"  {sk}: {sv}\n") for sk,sv in v.items()] for k,v in self.environment_info.items()]
            print(f"- Info Ambiente salvo: {p}")
        except Exception as e: print(f"Erro salvar info amb: {e}")

    def _save_full_results_json(self, ts: str):
        p = os.path.join(self.output_dir, f"benchmark_results_full_{ts}.json"); try:
            class NpEnc(json.JSONEncoder):
                def default(self, o):
                    if isinstance(o,np.integer): return int(o);
                    if isinstance(o,np.floating): return float(o);
                    if isinstance(o,np.ndarray): return o.tolist();
                    if isinstance(o,datetime): return o.isoformat();
                    if isinstance(o,np.bool_): return bool(o);
                    if isinstance(o,bytes): return o.decode('utf-8','replace');
                    return super(NpEnc,self).default(o)
            with open(p,"w",encoding='utf-8') as f: json.dump(self.results,f,indent=2,ensure_ascii=False,cls=NpEnc)
            print(f"- Resultados JSON salvo: {p}")
        except Exception as e: print(f"Erro salvar JSON: {e}")

    def _save_detailed_tests_csv(self, ts: str):
        p = os.path.join(self.output_dir, f"benchmark_test_details_{ts}.csv");
        if self.detailed_test_results:
            try: df=pd.DataFrame(self.detailed_test_results); cols=['ai','challenge','difficulty','test_case_index','status','execution_time_ms','error_message','stdout','input_args_str','expected_output_str','actual_output_str'];
                for c in cols:
                    if c not in df.columns: df[c]=pd.NA
                df[cols].to_csv(p, index=False, encoding='utf-8'); print(f"- Detalhes CSV salvo: {p}")
            except Exception as e: print(f"Erro salvar CSV detalhes: {e}")
        else: print("- Nenhum detalhe teste p/ salvar CSV.")

    def _create_csv_header(self) -> str:
        hdr = ["# --- Environment Info ---"]; [hdr.append(f"# {k}:") or [hdr.append(f"#   {sk}: {sv}") for sk,sv in v.items()] if isinstance(v,dict) else hdr.append(f"# {k}: {v}") for k,v in self.environment_info.items()]; hdr.append("# --- Benchmark Data ---"); return "\n".join(hdr)+"\n"

    def _save_summary_csv(self, df: pd.DataFrame, ts: str):
        p = os.path.join(self.output_dir, f"benchmark_summary_{ts}.csv"); try:
            cols = [c for c in df.columns if c not in ['code','full_response']]; summary=df[cols].copy(); hdr=self._create_csv_header();
            with open(p,"w",encoding='utf-8',newline='') as f: f.write(hdr); summary.to_csv(f,index=False,encoding='utf-8',float_format='%.4f',lineterminator='\n')
            print(f"- Resumo CSV salvo: {p}")
        except Exception as e: print(f"Erro salvar resumo CSV: {e}")

    def _calculate_statistics(self, df: pd.DataFrame) -> pd.DataFrame | None:
         try:
             m=[("Avg API Time (s)","api_execution_time","mean"),("Avg LoC","lines_of_code","mean"),("Avg Completeness (%)","completeness","mean"),("Avg Flake8 Issues","flake8_issues","mean_valid"),("Total Passed","total_tests_passed","sum"),("Total Attempted","total_tests_attempted","sum"),("Overall Success (%)","avg_exec_success_rate","mean_percentage"),("Avg Exec Time (ms)","avg_exec_time_ms","mean_valid"),("Total Errors/Fails","num_execution_errors","sum"),("Example Error","first_execution_error","first_non_empty")]
             ais=sorted(df['ai'].unique()); data={"Metric":[i[0] for i in m]};
             for ai in ais:
                 df_ai=df[df["ai"]==ai]; c_data=[]
                 for _,col,agg in m:
                     if col not in df_ai.columns: c_data.append(np.nan); continue
                     if agg=="mean": v=df_ai[col].mean()
                     elif agg=="sum": v=df_ai[col].sum()
                     elif agg=="mean_valid": d=df_ai[df_ai[col].notna()&(df_ai[col]!=-1)][col]; v=d.mean() if not d.empty else np.nan
                     elif agg=="mean_percentage": v=df_ai[col].mean()*100
                     elif agg=="first_non_empty": errs=df_ai[df_ai[col].fillna('').astype(str)!=''][col]; v=errs.iloc[0] if not errs.empty else ""
                     else: v=np.nan
                     c_data.append(v)
                 data[ai]=c_data
             return pd.DataFrame(data)
         except Exception as e: print(f"Erro calcular stats: {e}"); return None

    def _save_statistics_csv(self, stats_df: pd.DataFrame, ts: str):
        p = os.path.join(self.output_dir, f"benchmark_stats_{ts}.csv"); try:
            hdr=self._create_csv_header();
            with open(p,"w",encoding='utf-8',newline='') as f: f.write(hdr); stats_df.to_csv(f,index=False,encoding='utf-8',float_format='%.2f',lineterminator='\n')
            print(f"- Estatísticas CSV salvo: {p}")
        except Exception as e: print(f"Erro salvar stats CSV: {e}")

    def _display_summary_table(self, stats_df: pd.DataFrame):
        print("\n===== SUMÁRIO DO BENCHMARK (ESTATÍSTICAS) =====");
        if stats_df is None or stats_df.empty: print("Não foi possível gerar sumário."); return
        disp = stats_df.copy()
        for col in disp.columns:
            if col=='Metric': continue; fmt_col=[];
            if disp.empty: continue
            for i in disp.index:
                m=disp.loc[i,'Metric']; v=disp.loc[i,col]; fmt="N/A"
                if not pd.isna(v):
                    if "Flake8" in m: fmt=f"{v:.1f}" if v!=-1 else "N/A"
                    elif "Time (ms)" in m: fmt=f"{v:.1f}" if v!=-1 else "N/A"
                    elif "(%)" in m: fmt=f"{v:.1f}%"
                    elif "Total " in m: fmt=f"{int(v)}"
                    elif "Time (s)" in m: fmt=f"{v:.2f}"
                    elif "Example Error" in m: s=str(v); fmt=s[:60]+('...'if len(s)>60 else '')
                    else: fmt=f"{v:.1f}"
                fmt_col.append(fmt)
            if fmt_col: disp[col] = fmt_col
        try: from tabulate import tabulate; print(tabulate(disp, headers='keys', tablefmt='grid', showindex=False))
        except ImportError: print("(Instale 'tabulate' p/ tabela bonita)"); print(disp.to_string(index=False))
        except Exception as e: print(f"Erro imprimir tabela: {e}"); print(disp.to_string(index=False))

    def generate_report(self):
        """Gera todos os relatórios e visualizações."""
        if not self.results: print("Não há resultados p/ relatório."); return
        print("\n--- Gerando Relatórios ---"); ts=datetime.now().strftime("%Y%m%d_%H%M%S"); print(f"Timestamp: {ts}"); print(f"Diretório: '{self.output_dir}/'")
        self._save_environment_info(ts); df=self._prepare_dataframe()
        if df is None: print("Falha preparar DF. Abortando."); return
        self._save_full_results_json(ts); self._save_detailed_tests_csv(ts); self._save_summary_csv(df, ts)
        stats=self._calculate_statistics(df)
        if stats is not None: self._save_statistics_csv(stats, ts)
        else: print("- Não possível calcular/salvar stats.")
        try: print("\n--- Gerando Visualizações ---"); self.create_visualizations(df, ts, self.output_dir); print(f"- Visualizações salvas em '{self.output_dir}/'")
        except Exception as e: print(f"Erro gerar visualizações: {e}")
        if stats is not None: self._display_summary_table(stats)

    def _plot_bar_charts(self, df_plot, ai_list, ai_color_map, challenge_order, ts, out_dir):
        """Gera gráficos de barras."""
        print("Gerando Gráficos de Barras..."); metrics = {"1_api_time":("API Time","s","api_execution_time"),"2_loc":("LoC","Linhas","lines_of_code"),"3_complete":("Completude","(%)","completeness"),"4_flake8":("Flake8 Issues","Issues","flake8_issues"),"7_errors":("Exec Errors/Fails","Num","num_execution_errors"),}
        if 'avg_exec_success_rate' in df_plot.columns and df_plot['avg_exec_success_rate'].notna().any(): metrics["5_success"]=("Exec Success Rate","(%)","avg_exec_success_rate")
        if 'avg_exec_time_ms' in df_plot.columns and (df_plot['avg_exec_time_ms']!=-1).any(): metrics["6_exec_time"]=("Exec Time","(ms)","avg_exec_time_ms")
        metrics = dict(sorted(metrics.items()))
        for k, (t, y, c) in metrics.items():
            if c not in df_plot.columns: print(f"Skip bar '{t}': Coluna N/A."); continue
            data = df_plot.copy();
            if c=='flake8_issues': data=data[data[c]!=-1]
            elif c=='avg_exec_time_ms': data=data[data[c].notna()&(data[c]!=-1)]
            elif c=='num_execution_errors': data=data[data[c].notna()]
            elif c=='avg_exec_success_rate': data[c]=data[c]*100
            if data.empty or data[c].isnull().all(): print(f"Skip bar '{t}': Sem dados."); continue
            try:
                plt.figure(); piv=data.pivot_table(index='challenge',columns='ai',values=c,aggfunc='mean'); piv=piv.reindex(challenge_order,axis=0).reindex(ai_list,axis=1);
                fill = 0.0 if c not in ['flake8_issues','avg_exec_time_ms','num_execution_errors'] else np.nan; piv.fillna(fill,inplace=True)
                if piv.isnull().all().all(): print(f"Skip bar '{t}': Dados NaN."); plt.close(); continue
                n_ai=len(ai_list); ax=piv.plot(kind='bar',figsize=(max(12,n_ai*2),7),color=[ai_color_map.get(ai,'grey') for ai in piv.columns],width=0.8)
                plt.title(f'{t} por Desafio'); plt.ylabel(y); plt.xlabel('Desafio'); plt.xticks(rotation=45,ha='right',fontsize=9); plt.yticks(fontsize=9)
                if c in ['completeness','avg_exec_success_rate']: plt.ylim(0,105)
                elif c in ['flake8_issues','num_execution_errors'] and not piv.isnull().all().all(): mv=piv.max(skipna=True).max(skipna=True); plt.ylim(bottom=0,top=max(1,mv*1.1) if not pd.isna(mv) else 1)
                elif c=='avg_exec_time_ms' and not piv.isnull().all().all(): plt.ylim(bottom=0)
                plt.legend(title='IA',bbox_to_anchor=(1.02,1),loc='upper left'); plt.tight_layout(rect=[0,0,0.88,1]); p=os.path.join(out_dir,f"{k}_{ts}.png"); plt.savefig(p); plt.close()
            except Exception as e: print(f"Erro gráfico barras '{t}': {e}"); plt.close()

    def _plot_box_plots(self, df_plot, ai_list, ai_color_map, ts, out_dir):
        """Gera box plots."""
        print("Gerando Box Plots..."); metrics = {"api_time": {"t":"Distribuição API Time","y":"Tempo(s)","v":None},"loc": {"t":"Distribuição LoC","y":"Linhas","v":(lambda d,c: d[c]>0)},"complete": {"t":"Distribuição Completude","y":"Score(%)","v":None}}
        if 'flake8_issues' in df_plot.columns and (df_plot['flake8_issues']!=-1).any(): metrics["flake8"]={"t":"Distribuição Flake8 Issues","y":"Issues","v":(lambda d,c: d[c]!=-1)}
        if 'avg_exec_time_ms' in df_plot.columns and (df_plot['avg_exec_time_ms']!=-1).any(): metrics["exec_time"]={"t":"Distribuição Exec Time","y":"Tempo(ms)","v":(lambda d,c: d[c].notna()&(d[c]!=-1))}
        if 'num_execution_errors' in df_plot.columns and df_plot['num_execution_errors'].notna().any(): metrics["errors"]={"t":"Distribuição Erros/Falhas Exec","y":"Num","v":(lambda d,c: d[c].notna())}
        cnt = 7
        for k, cfg in metrics.items():
            cnt += 1;
            if k not in df_plot.columns: print(f"  Skip box '{cfg['t']}': Coluna N/A."); continue
            data=[]; lbls=[]; tmp=df_plot.copy()
            if cfg.get("v"): try: tmp=tmp[cfg["v"](tmp,k)]; except Exception as e: print(f" Aviso box cond '{k}': {e}.")
            if tmp.empty or tmp[k].isnull().all(): print(f"  Skip box '{cfg['t']}': Sem dados."); continue
            for ai in ai_list:
                ai_data=tmp[tmp['ai']==ai][k].dropna()
                if not ai_data.empty: data.append(ai_data.values); lbls.append(ai)
            if not data: print(f"  Skip box '{cfg['t']}': Sem dados por IA."); continue
            try:
                plt.figure(figsize=(max(8,len(lbls)*1.2),6)); bp=plt.boxplot(data,tick_labels=lbls,patch_artist=True,showfliers=True)
                plt.title(cfg["t"],fontsize=14); plt.ylabel(cfg["y"],fontsize=10); plt.xlabel("IA",fontsize=10); plt.xticks(rotation=30,ha='right',fontsize=9); plt.yticks(fontsize=9); plt.grid(axis='y',linestyle='--',alpha=0.7)
                colors=[ai_color_map.get(lbl,'grey') for lbl in lbls];
                for patch, color in zip(bp['boxes'],colors): patch.set_facecolor(color); patch.set_alpha(0.6)
                plt.tight_layout(); fname=f"{cnt}_boxplot_{k}_{ts}.png"; p=os.path.join(out_dir,fname); plt.savefig(p); plt.close()
            except Exception as e: print(f"  Erro box plot '{cfg['t']}': {e}"); plt.close()
        print("Geração Box Plots concluída.")

    def _plot_radar_chart(self, df, ai_list, ai_color_map, ts, out_dir):
        """Gera gráfico de radar."""
        n_ai=len(ai_list); if n_ai<2: print("Radar pulado (< 2 IAs)."); return
        print(f"Gerando radar p/ {n_ai} IAs..."); cols=['api_execution_time','lines_of_code','completeness']
        try:
            if not all(c in df.columns for c in cols): print(f"Skip Radar: Faltando colunas."); return
            means={};
            for ai in ai_list: df_ai=df[df["ai"]==ai]; m={}; for c in cols: m[c]=df_ai[c].mean() if not df_ai[c].isnull().all() else 0; means[ai]=m
            times=[m.get(cols[0],0) for m in means.values()]; locs=[m.get(cols[1],0) for m in means.values()]; comps=[m.get(cols[2],0) for m in means.values()]
            max_t=max(times+[1e-6]); t_norm={ai:100*(1-means[ai].get(cols[0],0)/max_t) if max_t>0 else 0 for ai in ai_list}
            v_locs=[l for l in locs if l>0]; min_l=min(v_locs) if v_locs else 0; max_l=max(locs+[1e-6]); l_norm={}
            for ai in ai_list:
                l=means[ai].get(cols[1],0)
                if l<=0: l_norm[ai]=0
                elif min_l>0 and max_l>min_l: nv=100*(1-(l-min_l)/(max_l-min_l)); l_norm[ai]=max(0,min(100,nv))
                else: l_norm[ai]=100 if l>0 else 0
                if l==min_l and min_l>0: l_norm[ai]=100
            c_norm={ai: means[ai].get(cols[2],0) for ai in ai_list}
            cats=['Veloc API\n(Norm)','Conc LoC\n(Norm)','Completude\n(%)']; N=len(cats); ang=[n/N*2*np.pi for n in range(N)]; ang+=ang[:1]
            fig=plt.figure(figsize=(9,9)); ax=fig.add_subplot(111,polar=True)
            for ai in ai_list: vals=[t_norm.get(ai,0),l_norm.get(ai,0),c_norm.get(ai,0)]; pv=vals+vals[:1]; clr=ai_color_map.get(ai,'grey'); ax.plot(ang,pv,'o-',linewidth=2,label=ai,color=clr); ax.fill(ang,pv,clr,alpha=0.2)
            ax.set_xticks(ang[:-1]); ax.set_xticklabels(cats); ax.set_yticks(np.arange(0,101,20)); ax.set_ylim(0,100)
            plt.legend(loc='upper right',bbox_to_anchor=(1.35,1.15)); plt.title('Comparação Geral (Norm)',size=14,y=1.18); plt.tight_layout()
            cnt=15; p=os.path.join(out_dir,f"{cnt}_radar_chart_{ts}.png"); plt.savefig(p); plt.close()
        except Exception as e: print(f"Erro radar: {e}"); plt.close()

    def create_visualizations(self, df, ts, out_dir):
        """Cria e salva visualizações."""
        plt.style.use('ggplot'); ais=sorted(df['ai'].unique()); n_ai=len(ais)
        colors=plt.cm.tab10(np.linspace(0,1,n_ai)) if n_ai<=10 else plt.cm.viridis(np.linspace(0,1,n_ai)); cmap={ai:colors[i] for i,ai in enumerate(ais)}
        req_cols=['challenge','ai','api_execution_time','lines_of_code','completeness','flake8_issues','difficulty','num_execution_errors'] + [c for c in ['avg_exec_success_rate','avg_exec_time_ms'] if c in df.columns]
        if df.empty or not all(c in df.columns for c in req_cols): print(f"AVISO: Faltando colunas p/ gráficos. Pulando."); return
        dtype=pd.CategoricalDtype(['Easy','Medium','Hard'],ordered=True);
        try: df['diff_cat']=df['difficulty'].astype(dtype); order=df.sort_values('diff_cat')['challenge'].unique()
        except: print("AVISO: Erro ordenar dificuldade."); order=sorted(df['challenge'].unique())
        df_plot=df.copy(); self._plot_bar_charts(df_plot,ais,cmap,order,ts,out_dir); self._plot_box_plots(df_plot,ais,cmap,ts,out_dir); self._plot_radar_chart(df,ais,cmap,ts,out_dir)

# --- Ponto de Entrada ---
if __name__ == "__main__":
    print("="*59); print(" AI Programming Benchmark: Claude vs ChatGPT vs Gemini"); print("      (Incluindo Análise Flake8 e Execução Externa/Local)"); print("="*59)
    multiprocessing.freeze_support()
    if not os.path.exists(".env"): print("AVISO: .env não encontrado.")
    try:
        benchmark = AIBenchmark()
        if benchmark.claude_api_key or benchmark.openai_api_key or benchmark.gemini_api_key: benchmark.run_benchmark(); print("\nBenchmark concluído."); print(f"Relatórios salvos em: '{benchmark.output_dir}'")
        else: print("\nERRO FATAL: Nenhuma API configurada.")
    except Exception as e: print("\nERRO INESPERADO:"); print(f"Tipo: {type(e).__name__}"); print(f"Erro: {e}"); print("Traceback:"); traceback.print_exc()
    print("="*59)