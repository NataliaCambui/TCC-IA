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

# Verifica se é Python 3.8+ para importlib.metadata
try:
    from importlib import metadata as importlib_metadata
except ImportError:
    import pkg_resources # type: ignore

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
            try:
                genai.configure(api_key=self.gemini_api_key)
                print("Chave API Gemini configurada.")
            except Exception as e:
                print(f"Erro configurar API Gemini: {e}. AI pulada.")
                self.gemini_api_key = None
        else:
            print("AVISO: Chave GEMINI_API_KEY não encontrada. Gemini AI pulada.")

    def _get_environment_info(self):
        """Coleta informações do ambiente."""
        info = {
            "Timestamp": datetime.now().isoformat(),
            "OS": f"{platform.system()} {platform.release()}",
            "Python Version": sys.version.split()[0],
            "Models": {
                "Claude": self.claude_model if self.claude_api_key else "N/A",
                "ChatGPT": self.openai_model if self.openai_api_key else "N/A",
                "Gemini": self.gemini_model if self.gemini_api_key else "N/A"
            },
            "Executor Service URL": self.executor_url,
            "Local LRU Cache Execution": True,
            "Libraries": {}
        }
        libs = [
            'pandas', 'numpy', 'matplotlib', 'requests', 'python-dotenv',
            'google-generativeai', 'google-api-core'
        ]
        try: # Tenta importlib.metadata primeiro (Python 3.8+)
            import importlib.metadata as importlib_metadata
            for lib in libs:
                try: info["Libraries"][lib] = importlib_metadata.version(lib)
                except importlib_metadata.PackageNotFoundError: info["Libraries"][lib] = "Not Found"
        except ImportError: # Fallback para pkg_resources
            try:
                import pkg_resources
                for lib in libs:
                    try: info["Libraries"][lib] = pkg_resources.get_distribution(lib).version
                    except pkg_resources.DistributionNotFound: info["Libraries"][lib] = "Not Found"
                    except Exception: info["Libraries"][lib] = "Error getting version"
            except ImportError: # Se nenhum funcionar
                 for lib in libs: info["Libraries"][lib] = "Unknown (No pkg info)"
        return info

    def _check_flake8(self):
        """Verifica disponibilidade do flake8."""
        try:
            subprocess.run([sys.executable, '-m', 'flake8', '--version'],
                           check=True, capture_output=True, timeout=5)
            print("Verificação Flake8: OK")
            self.flake8_available = True
        except Exception:
            print(f"AVISO: flake8 não encontrado via '{sys.executable} -m flake8'.")
            self.flake8_available = False

    def query_claude(self, prompt: str) -> dict:
        """Consulta API Claude."""
        start_time = time.time(); content = ""; execution_time = 0
        if not self.claude_api_key:
            print("Pulando Claude: Chave API N/A.")
            return {"content": "", "execution_time": 0} # Retorna tempo 0
        headers = {
            "x-api-key": self.claude_api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        data = {"model": self.claude_model, "max_tokens": 4000, "messages": [{"role": "user", "content": prompt}]}
        try:
            response = requests.post("https://api.anthropic.com/v1/messages",
                                     headers=headers, json=data, timeout=120)
            response.raise_for_status()
            result = response.json()
            if ("content" in result and isinstance(result["content"], list) and
                    len(result["content"]) > 0):
                first_content = result["content"][0]
                content = first_content.get("text", "") if isinstance(first_content, dict) else ""
            else:
                print(f"Resposta Claude inesperada: {result}")
        except requests.exceptions.RequestException as e:
            error_message = f"Erro req Claude: {e}"
            if e.response is not None:
                error_message += f" | Status: {e.response.status_code}"
                try: error_message += f" | Body: {e.response.json()}"
                except json.JSONDecodeError: error_message += f" | Body: {e.response.text[:100]}..."
            print(error_message)
        except Exception as e:
            print(f"Erro inesperado Claude: {type(e).__name__} - {e}")
        execution_time = time.time() - start_time
        return {"content": content or "", "execution_time": execution_time}

    def query_chatgpt(self, prompt: str) -> dict:
        """Consulta API ChatGPT."""
        start_time = time.time(); content = ""; execution_time = 0
        if not self.openai_api_key:
            print("Pulando ChatGPT: Chave API N/A.")
            return {"content": "", "execution_time": 0}
        headers = {"Authorization": f"Bearer {self.openai_api_key}", "Content-Type": "application/json"}
        data = {"model": self.openai_model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 4000}
        try:
            response = requests.post("https://api.openai.com/v1/chat/completions",
                                     headers=headers, json=data, timeout=120)
            response.raise_for_status()
            result = response.json()
            if ("choices" in result and isinstance(result["choices"], list) and
                    len(result["choices"]) > 0):
                first_choice = result["choices"][0]
                if (isinstance(first_choice, dict) and "message" in first_choice and
                        isinstance(first_choice["message"], dict)):
                    content = first_choice["message"].get("content", "")
                else:
                    print(f"Resposta ChatGPT inesperada (message): {result}")
            else:
                print(f"Resposta ChatGPT inesperada (choices): {result}")
        except requests.exceptions.RequestException as e:
            error_message = f"Erro req ChatGPT: {e}"
            if e.response is not None:
                error_message += f" | Status: {e.response.status_code}"
                try:
                    error_details = e.response.json()
                    error_msg = error_details.get('error', {}).get('message', '')
                    error_message += f" | API Error: {error_msg}" if error_msg else f" | Body: {error_details}"
                except json.JSONDecodeError:
                    error_message += f" | Body: {e.response.text[:100]}..."
            print(error_message)
        except Exception as e:
            print(f"Erro inesperado ChatGPT: {type(e).__name__} - {e}")
        execution_time = time.time() - start_time
        return {"content": content or "", "execution_time": execution_time}

    def query_gemini(self, prompt: str) -> dict:
        """Consulta API Gemini."""
        start_time = time.time(); content = ""; execution_time = 0
        if not self.gemini_api_key or not self.gemini_model:
            print("Pulando Gemini: Chave/Modelo N/A.")
            return {"content": "", "execution_time": 0}
        try:
            model = genai.GenerativeModel(self.gemini_model)
            safety_settings=[
                {"category": c, "threshold": "BLOCK_NONE"} for c in [
                    "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
                    "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"
                ]
            ]
            response = model.generate_content(prompt, safety_settings=safety_settings)
            execution_time = time.time() - start_time # Calcula tempo mesmo se houver erro abaixo
            try:
                content = response.text
            except ValueError:
                # Resposta bloqueada ou sem texto
                print("Resposta Gemini bloqueada ou sem texto.")
                try:
                    print(f"  Feedback: {response.prompt_feedback}")
                    if hasattr(response, 'candidates') and response.candidates:
                        print(f"  Reason: {response.candidates[0].finish_reason}")
                except Exception as fe:
                    print(f"  Erro ao obter feedback Gemini: {fe}")
        except google.api_core.exceptions.ResourceExhausted as e:
            print(f"Erro Gemini (Rate Limit): {e}")
            execution_time = time.time() - start_time # Garante que tempo é medido
        except Exception as e:
            print(f"Erro inesperado Gemini: {type(e).__name__} - {e}")
            if execution_time == 0: execution_time = time.time() - start_time
        return {"content": content or "", "execution_time": execution_time}

    def extract_code(self, content: str) -> str:
        """Extrai bloco de código Python."""
        if not content: return ""
        # Tenta ```python ... ```
        m = re.search(r'```python\n(.*?)\n```', content, re.DOTALL)
        if m: return m.group(1).strip()
        # Tenta ``` ... ```
        m = re.search(r'```(?:.*\n)?(.*?)\n```', content, re.DOTALL)
        if m: return m.group(1).strip()
        # Fallback por indentação/keywords
        lines = content.splitlines(); code_lines = []; start = False
        for line in lines:
            s_line = line.strip()
            if not start and (line.startswith((' ','\t')) or s_line.startswith(('def ','class '))): start = True
            if start: code_lines.append(line)
        if code_lines: return '\n'.join(code_lines).strip()
        # Fallback final
        if len(lines) < 3 and not ("def " in content or "class " in content): return ""
        return content.strip()

    def count_code_lines(self, code: str) -> int:
        """Conta linhas de código reais."""
        if not code: return 0
        return len([ln for ln in code.splitlines() if ln.strip() and not ln.strip().startswith('#')])

    def measure_response_completeness(self, code: str, description: str) -> int:
        """Estima completude da resposta (0-100)."""
        score = 0; lines = self.count_code_lines(code)
        if lines > 0:
            score += 50
            if "def " in code or "class " in code: score += 10
            if "return " in code or "yield " in code: score += 10
            elif "island" in description.lower() and ("dfs(" in code or "bfs(" in code): score += 5
            desc_l = description.lower(); code_l = code.lower(); kw_bonus = 0
            if "fibonacci" in desc_l and ("fibonacci" in code_l or "fib(" in code_l or "yield" in code): kw_bonus = 10
            elif "binary search" in desc_l and ("mid =" in code or "mid=" in code or "low" in code or "high" in code): kw_bonus = 10
            elif "merge sort" in desc_l and ("merge(" in code_l or ("left" in code and "right" in code and "mid" in code)): kw_bonus = 10
            elif "island" in desc_l and ("dfs(" in code_l or "bfs(" in code_l or "visit" in code_l or ("grid[" in code and "==" in code)): kw_bonus = 10
            elif "palindrome" in desc_l and ("[::-1]" in code or ".isalnum()" in code_l or ".lower()" in code_l): kw_bonus = 10
            elif "parentheses" in desc_l and ("stack" in code_l or "append(" in code or "pop(" in code): kw_bonus = 10
            elif "lru cache" in desc_l and ("dict" in code_l or "capacity" in code_l or "OrderedDict" in code): kw_bonus = 10
            score += kw_bonus; struct_bonus = 0
            if "if" in code or "for" in code or "while" in code: struct_bonus = 10
            if struct_bonus > 0 and ("if not" in code or "if len(" in code or "== 0" in code or "is None" in code): struct_bonus += 5
            score += min(struct_bonus, 15)
        return min(score, 100)

    def analyze_code_quality_flake8(self, code_string: str) -> int:
        """Analisa código com Flake8."""
        if not self.flake8_available or not code_string or self.count_code_lines(code_string) == 0: return -1
        tfp = None
        try:
            # Usar delete=False porque o flake8 precisa do caminho após o 'with' fechar
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.py', delete=False, encoding='utf-8') as tf:
                tf.write(code_string)
                tfp = tf.name # Guarda o caminho
            # Executa flake8
            exe = sys.executable
            res = subprocess.run([exe, '-m', 'flake8', '--count', tfp],
                                 capture_output=True, text=True, timeout=15, check=False)
            # Interpreta resultado
            if res.returncode in [0, 1] and res.stdout:
                try: return int(res.stdout.strip().splitlines()[-1])
                except (IndexError, ValueError): return len(res.stdout.strip().splitlines()) # Fallback
            elif res.stderr: print(f"Erro flake8: {res.stderr}"); return -1
            else: return 0 # Sem output e exit code 0 = sem erros
        except subprocess.TimeoutExpired: print("Erro: Flake8 timeout."); return -1
        except Exception as e: print(f"Erro inesperado flake8: {e}"); return -1
        finally:
            # Garante remoção do ficheiro temporário
            if tfp and os.path.exists(tfp):
                try: os.remove(tfp)
                except Exception as er: print(f"Aviso: Falha remover temp file {tfp}: {er}")

    # --- Métodos Execução Local LRU Cache ---

    def _lru_worker(self, code_string: str, test_sequence: list, result_queue: multiprocessing.Queue):
        """Função worker para executar LRU Cache localmente com restrições."""
        results = []; error_message = None; lru_instance = None; final_success = True
        try:
            # 1. Pré-processar código: Remover imports
            lines = code_string.splitlines()
            filtered_lines = [
                ln for ln in lines if not (
                    ln.strip().startswith("from collections import OrderedDict") or
                    ln.strip().startswith("import collections") or
                    ln.strip().startswith("import OrderedDict")
                )
            ]
            processed_code_string = "\n".join(filtered_lines)

            # 2. Preparar ambiente restrito
            safe_builtins = {
                'int': int, 'str': str, 'list': list, 'dict': dict, 'tuple': tuple,
                'len': len, 'range': range, 'print': print, 'True': True, 'False': False,
                'None': None, 'object': object, 'Exception': Exception, 'RuntimeError': RuntimeError,
                'ValueError': ValueError, 'NameError': NameError, 'AttributeError': AttributeError,
                'TypeError': TypeError, 'StopIteration': StopIteration, 'KeyError': KeyError, # Permitir KeyError para dicts
                '__build_class__': builtins.__build_class__,
                'getattr': getattr, 'isinstance': isinstance,
                '__name__': '__exec_sandbox__' # <--- CORREÇÃO v16
            }
            safe_globals = {
                "__builtins__": safe_builtins,
                "OrderedDict": OrderedDict # Passa explicitamente
            }

            # 3. Executa o código da classe
            local_namespace = {}
            exec(processed_code_string, safe_globals, local_namespace)

            # 4. Encontrar o nome da classe
            class_name = None
            for name, obj in local_namespace.items():
                # Procura por um tipo (classe) que não seja OrderedDict
                if isinstance(obj, type) and name != 'OrderedDict':
                    class_name = name
                    break
            # Fallback se não estiver no namespace local (pode acontecer)
            if not class_name:
                match = re.search(r"^\s*class\s+(\w+)", processed_code_string, re.MULTILINE)
                if match: class_name = match.group(1)
                else: raise NameError("Nome classe LRUCache não encontrado no código.")

            LruClass = local_namespace.get(class_name)
            if not LruClass:
                # Tenta obter do globals se não estiver no locals (menos comum com exec)
                LruClass = safe_globals.get(class_name)
                if not LruClass:
                     raise NameError(f"Classe '{class_name}' não definida após exec.")

            # 5. Executar sequência de testes
            op_total_time = 0.0
            for op_data in test_sequence:
                op_start_time = time.perf_counter()
                operation = op_data["op"]
                args = op_data.get("args", [])
                expected = op_data.get("expected") # None para 'put'

                # Nomes curtos para dicionário de resultados
                step_res = {"op": operation, "args": args, "exp": expected,
                            "out": None, "ok": False, "err": None, "t_ms": -1.0}
                try:
                    if operation == "__init__":
                        lru_instance = LruClass(*args)
                        step_res["ok"] = True; step_res["out"] = "OK"
                    elif lru_instance is None:
                        raise RuntimeError("LRUCache não inicializada.")
                    elif operation in ["put", "get"]:
                         # Tenta obter o método de forma segura
                         if not hasattr(lru_instance, operation):
                              raise AttributeError(f"Instância LRUCache não tem método '{operation}'")
                         method_to_call = getattr(lru_instance, operation)

                         if operation == "put":
                              method_to_call(*args)
                              step_res["ok"] = True; step_res["out"] = "OK"
                         elif operation == "get":
                              output = method_to_call(*args)
                              step_res["out"] = output
                              # Comparação como string para evitar problemas de tipo (ex: int vs float)
                              if str(output) == str(expected):
                                   step_res["ok"] = True
                              else:
                                   step_res["err"] = f"Out:'{output}'({type(output)})!=Exp:'{expected}'({type(expected)})"
                                   final_success = False # Teste geral falha
                    else:
                        raise ValueError(f"Operação LRU inválida: {operation}")

                except Exception as e_step:
                    # Captura erro no passo, regista e marca falha geral
                    step_error_msg = f"{type(e_step).__name__}: {str(e_step).splitlines()[0]}"
                    step_res["err"] = step_error_msg
                    print(f"      Erro passo LRU ({operation}): {step_error_msg}")
                    final_success = False

                op_end_time = time.perf_counter()
                step_res["t_ms"] = (op_end_time - op_start_time) * 1000
                op_total_time += (op_end_time - op_start_time)
                results.append(step_res) # Adiciona resultado do passo

                # Descomentar para parar a sequência no primeiro erro
                # if not step_res["ok"]:
                #     break

            print(f"    Sequência LRU worker concluída. Tempo passos: {op_total_time*1000:.2f} ms")

        except Exception as e_main:
            # Captura erro fora da sequência (ex: na definição da classe)
            error_message = f"Erro worker LRU: {type(e_main).__name__} - {str(e_main).splitlines()[0]}"
            print(f"    ERRO WORKER LRU: {error_message}")
            traceback.print_exc() # Log completo do erro no worker
            final_success = False

        # Envia resultado para processo pai
        try:
            result_queue.put({"success": final_success, "results": results, "error": error_message})
        except Exception as q_err:
             print(f"    ERRO fila worker LRU: {q_err}")

    def _execute_lru_locally(self, code_string: str, test_sequence: list) -> dict:
        """Gere o processo para execução local do LRU Cache."""
        ctx = multiprocessing.get_context(None) # Contexto padrão (spawn no Win)
        result_queue = ctx.Queue()
        timeout_seconds = 10 # Timeout

        process = ctx.Process(target=self._lru_worker, args=(code_string, test_sequence, result_queue))
        return_value = {"success": False, "results": [], "error": "Unknown execution error"}
        terminated = False # Flag para indicar se o processo foi terminado por timeout

        try:
            start_time = time.time()
            process.start()
            process.join(timeout=timeout_seconds)

            # Verifica se o processo ainda está vivo (excedeu timeout)
            if process.is_alive():
                print(f"  AVISO: Timeout ({timeout_seconds}s) processo LRU. Terminando.")
                process.terminate(); process.join(1) # Tenta terminar
                if process.is_alive(): process.kill(); process.join() # Força
                return_value = {"success": False, "results": [], "error": f"Timeout ({timeout_seconds}s)"}
                terminated = True

            elapsed_time = time.time() - start_time

            # Se não foi terminado por timeout, tenta obter resultado
            if not terminated:
                exit_code = process.exitcode
                if exit_code == 0: # Processo terminou normalmente
                    try:
                        # Pega resultado da fila (pode ter tido erro interno reportado)
                        result = result_queue.get(timeout=1)
                        print(f"  Exec LRU local OK ({elapsed_time:.2f}s). Worker success: {result.get('success')}")
                        return_value = result
                    except queue.Empty:
                        print("  ERRO: Processo LRU OK, mas fila vazia (worker falhou?).")
                        return_value = {"success": False, "results": [], "error": "Worker OK but no result"}
                    except Exception as e_q:
                        print(f"  ERRO: Falha obter resultado fila LRU: {e_q}")
                        return_value = {"success": False, "results": [], "error": f"Queue error: {e_q}"}
                else: # Processo terminou com erro
                     print(f"  ERRO: Processo LRU terminou c/ exitcode: {exit_code}")
                     # Tenta obter algo da fila (pode ter msg de erro do worker)
                     try:
                          err_result = result_queue.get_nowait()
                          return_value = err_result # Usa o erro reportado pelo worker se houver
                          if not return_value.get("error"): # Se worker não reportou erro, usa exit code
                               return_value["error"] = f"Worker process exited with code {exit_code}"
                               return_value["success"] = False
                     except queue.Empty:
                          # Sem erro explícito do worker, reporta apenas o exit code
                          return_value = {"success": False, "results": [], "error": f"Worker process exited with code {exit_code}"}
                     except Exception as e_q_final:
                           return_value = {"success": False, "results": [], "error": f"Worker exit {exit_code}, queue error: {e_q_final}"}

        except Exception as e_p: # Erro ao gerir o processo
             print(f"  ERRO: Falha gerir processo LRU: {e_p}")
             return_value = {"success": False, "results": [], "error": f"Process mgmt error: {e_p}"}
             if process and process.is_alive():
                  try: process.kill(); process.join()
                  except: pass # Ignora erros ao matar
        finally:
            # Limpeza de recursos
            try: result_queue.close(); result_queue.join_thread()
            except Exception as e_qc: print(f"  Aviso: Erro limpar fila LRU: {e_qc}")
            try:
                 if process: process.close()
            except Exception as e_pc: print(f"  Aviso: Erro limpar processo LRU: {e_pc}")

        return return_value

    # --- Fim métodos execução local ---

    def evaluate_solution(self, ai_name: str, challenge: dict, response: dict) -> dict:
        """Avalia a solução da IA (exec local p/ LRU, externa p/ outros)."""
        content = response.get("content", ""); api_time = response.get("execution_time", 0)
        code = self.extract_code(content); lines = self.count_code_lines(code)
        completeness = self.measure_response_completeness(code, challenge["description"])
        flake8 = self.analyze_code_quality_flake8(code)

        raw_results = []; details = []; n_errors = 0; first_err = ""
        can_exec = bool(code); func_name = None

        # --- CORREÇÃO: Passar 'challenge' dict para _detect_function_name ---
        if can_exec and challenge["name"] != "LRU Cache":
            func_name = self._detect_function_name(code, challenge)
            if not func_name: can_exec = False # Não executa sem nome de função

        if can_exec:
            if challenge["name"] == "LRU Cache":
                print(f"  Executando LRU Cache de {ai_name} localmente...")
                tc = challenge.get("test_cases", [])
                if tc and isinstance(tc[0], dict) and "sequence" in tc[0]:
                    lru_res = self._execute_lru_locally(code, tc[0]["sequence"])
                    raw_results = lru_res.get("results", []) # Passos detalhados
                    ok = lru_res.get("success", False); n_errors = 0 if ok else 1
                    first_err = lru_res.get("error", "") # Erro geral worker
                    if not first_err and not ok: # Procura erro no 1o passo falhado
                        for step in raw_results:
                            if step.get("err"): first_err = f"Step {step['op']}{step['args']}: {step['err']}"; break
                    if not first_err and not ok: first_err = "LRU exec falhou (passo?)"

                    # Calcula tempo médio dos passos executados
                    t_ok = [s['t_ms'] for s in raw_results if s.get('t_ms',-1)>=0]; avg_t = np.mean(t_ok) if t_ok else -1.0
                    # Adiciona 1 entrada ao details
                    details.append({
                        "ai": ai_name, "challenge": challenge["name"], "difficulty": challenge["difficulty"],
                        "test_case_index": 0, "input_args_str": json.dumps(tc[0]["sequence"]),
                        "expected_output_str": "Ver Passos", "actual_output_str": "Ver Passos",
                        "status": "Pass" if ok else "Fail" if raw_results else "Error", # Status agregado
                        "execution_time_ms": avg_t,
                        # --- CORREÇÃO: Garante que first_err é string antes do slice ---
                        "error_message": str(first_err)[:250],
                        "stdout": None
                    })
                    print(f"  Resultado LRU local: {'Sucesso' if ok else 'Falha/Erro'}")
                else:
                    print("  AVISO: Casos teste LRU inválidos."); n_errors = 1; first_err = "Caso teste LRU inválido."
                    details.append({"ai": ai_name, "challenge": challenge["name"], "difficulty": challenge["difficulty"],"test_case_index": 0, "input_args_str": "N/A", "expected_output_str": "N/A", "actual_output_str": None, "status": "Error", "execution_time_ms": -1, "error_message": first_err, "stdout": None })
            else: # Outros desafios (func_name é válido)
                print(f"  Executando '{challenge['name']}' de {ai_name} via Executor Service...")
                raw, det, errs, first_e = self._run_test_cases(ai_name, challenge, code, func_name)
                raw_results = raw; n_errors = errs; first_err = first_e; details.extend(det)
        else: # Sem código ou nome função
            err_msg = "Código não extraído" if not code else "Nome func/classe N/A"
            print(f"  AVISO: {err_msg} p/ '{challenge['name']}' por {ai_name}. Pulando exec.")
            n_tests = len(challenge.get("test_cases",[])) if challenge['name']!='LRU Cache' else 1
            n_errors = n_tests; first_err = err_msg
            for i in range(n_tests):
                 details.append({"ai": ai_name, "challenge": challenge["name"], "difficulty": challenge["difficulty"], "test_case_index": i, "input_args_str": None, "expected_output_str": None, "actual_output_str": None, "status": "No Code", "execution_time_ms": -1, "error_message": err_msg, "stdout": None })

        # Adiciona detalhes à lista principal da classe
        self.detailed_test_results.extend(details)

        return { # Retorna dicionário agregado p/ self.results
            "ai": ai_name, "challenge": challenge["name"], "difficulty": challenge["difficulty"],
            "api_execution_time": api_time, "lines_of_code": lines,
            "completeness": completeness, "flake8_issues": flake8,
            "num_execution_errors": n_errors, "first_execution_error": first_err,
            "code": code, "full_response": content, "execution_results_raw": raw_results
        }

    def _detect_function_name(self, code: str, challenge: dict) -> str | None:
        """Detecta nome da função/classe, priorizando assinatura."""
        if not code: return None
        expected_name = None
        signature = challenge.get("function_signature", "")
        sig_match = re.match(r"^\s*(?:def|class)\s+(\w+)", signature)
        if sig_match:
            expected_name = sig_match.group(1)
            # --- CORREÇÃO v16: Regex menos restrito (sem ^) ---
            pattern = rf"\s*(?:def|class)\s+{re.escape(expected_name)}\b"
            exact_match = re.search(pattern, code, re.MULTILINE)
            if exact_match:
                 # print(f"  Nome esperado '{expected_name}' encontrado.") # Opcional: Log
                 return expected_name
            # else: print(f"  AVISO: Nome esperado '{expected_name}' não encontrado.") # Opcional: Log

        # Fallback: primeira definição encontrada
        first_match = re.search(r"^\s*(?:def|class)\s+(\w+)", code, re.MULTILINE)
        if first_match:
            found_name = first_match.group(1)
            if not expected_name or found_name != expected_name:
                 print(f"  AVISO: Usando 1ª definição '{found_name}' (Esperado: '{expected_name or 'N/A'}').")
            return found_name
        return None

    def _run_test_cases(self, ai_name: str, challenge: dict, code: str,
                        actual_function_name: str) -> tuple[list, list, int, str]:
        """Executa casos de teste via Executor Service."""
        raw_results = []; details = []; n_fails_errors = 0; first_err = ""
        test_cases = challenge.get("test_cases", [])
        if not test_cases: print(f"  Aviso: Sem casos teste p/ '{challenge['name']}'."); return [], [], 0, ""
        for i, tc in enumerate(test_cases):
            args = tc.get("input_args"); exp_out = tc.get("expected_output")
            if args is None: print(f"  Aviso: Caso {i+1} '{challenge['name']}' s/ 'input_args'."); continue
            exec_res = self._call_executor_service(code, actual_function_name, args); raw_results.append(exec_res)
            t_det = {"ai": ai_name, "challenge": challenge["name"], "difficulty": challenge["difficulty"], "test_case_index": i, "input_args_str": json.dumps(args), "expected_output_str": json.dumps(exp_out), "actual_output_str": None, "status": "Unknown", "execution_time_ms": exec_res.get("execution_time_ms", -1), "error_message": None, "stdout": exec_res.get("stdout")}
            case_err = None
            if exec_res.get("success") is True:
                act_out = exec_res.get("output"); t_det["actual_output_str"] = json.dumps(act_out)
                try: eq = np.array_equal(np.array(act_out, dtype=object), np.array(exp_out, dtype=object)); t_det["status"] = "Pass" if eq else "Fail"
                    if not eq: case_err = f"Caso {i} Falhou: Output != Expected"; n_fails_errors += 1
                except Exception as comp_e: t_det["status"] = "Fail (Comp Error)"; t_det["error_message"] = f"Erro comparar: {comp_e}"; case_err = t_det["error_message"]; n_fails_errors += 1
            elif exec_res.get("success") is False: t_det["status"] = "Error"; t_det["error_message"] = exec_res.get("error", "Erro exec desconhecido"); case_err = str(t_det["error_message"])[:250]; n_fails_errors += 1
            else: t_det["status"] = "Unknown"; t_det["error_message"] = exec_res.get("error", "Executor sem status sucesso"); case_err = str(t_det["error_message"])[:250]; n_fails_errors += 1
            if case_err and not first_err: first_err = case_err
            details.append(t_det)
        if n_fails_errors > 0 and not first_err: first_err = "Um ou mais casos falharam."
        return raw_results, details, n_fails_errors, first_err

    def _call_executor_service(self, code_string: str, function_name: str, input_args: list) -> dict:
        """Chama o serviço executor externo."""
        imports = ("from typing import List, Dict, Tuple, Set, Optional, Any\n""import re\n""from collections import OrderedDict, deque\n\n"); code_to_exec = imports + code_string
        payload = {"code_string": code_to_exec, "function_name": function_name, "test_input_args": input_args}; headers = {"Content-Type": "application/json"}; default_err = {"success": False, "output": None, "stdout": None, "error": "Falha chamada executor", "execution_time_ms": -1}
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
        can_claude=bool(self.claude_api_key); can_openai=bool(self.openai_api_key); can_gemini=bool(self.gemini_api_key and self.gemini_model)
        enabled_ais=[n for n,can in [("Claude",can_claude),("ChatGPT",can_openai),("Gemini",can_gemini)] if can]
        if not enabled_ais: print("ERRO: Nenhuma API config."); return
        print(f"Iniciando Benchmark para: {', '.join(enabled_ais)}"); print(f"  Claude:{self.claude_model if can_claude else 'N/A'}|ChatGPT:{self.openai_model if can_openai else 'N/A'}|Gemini:{self.gemini_model if can_gemini else 'N/A'}"); print(f"  LRU Exec: Local"); print("="*42)
        ch_details = { "Fibonacci Sequence": "Téc: Iterativa/Recursiva. Fin: Algoritmos básicos.", "Binary Search": "Téc: Dividir/Conquistar. Fin: Busca binária.", "Merge Sort": "Téc: Dividir/Conquistar. Fin: Ordenação recursiva.", "Count Islands in Grid": "Téc: Travessia Grafo (DFS/BFS). Fin: Modelar grafo, DFS/BFS.", "Palindrome Check": "Téc: Manipulação Strings. Fin: Processar strings.", "Valid Parentheses": "Téc: Pilha (Stack). Fin: Validar ordem parênteses.", "LRU Cache": "Téc: Estrutura Dados Custom. Fin: Gerenciar cache LRU." }
        for chal in self.code_challenges:
            sig=chal.get("function_signature",f"{chal['name'].lower().replace(' ','_')}(...)")
            prompt = (f"Implement Python {'class' if 'class' in sig else 'function'}: {chal['description']}. Signature:\n{sig}\nProvide ONLY Python code for def/class and necessary imports (like `from collections import OrderedDict`). No explanations.")
            print(f"\n--- Desafio: {chal['name']} ({chal['difficulty']}) ---"); print(f"  {ch_details.get(chal['name'],'N/A.')}")
            resps={};
            if can_claude: print("Consultando Claude AI..."); resps["Claude"]=self.query_claude(prompt); time.sleep(1)
            if can_openai: print("Consultando ChatGPT..."); resps["ChatGPT"]=self.query_chatgpt(prompt); time.sleep(1)
            if can_gemini: print("Consultando Gemini AI..."); resps["Gemini"]=self.query_gemini(prompt); print("Aguardando 31s API Gemini..."); time.sleep(31)
            summary=[]
            for ai in enabled_ais:
                resp_data=resps.get(ai,{"content":"","execution_time":0});
                result=self.evaluate_solution(ai, chal, resp_data); self.results.append(result)
                n_err_fail=result.get('num_execution_errors',0);
                det=[d for d in self.detailed_test_results if d['ai']==ai and d['challenge']==chal['name']]
                n_att=len([d for d in det if d['status']!='No Code']); n_pass=len([d for d in det if d['status']=='Pass'])
                f8=(str(result['flake8_issues']) if result['flake8_issues']!=-1 else 'N/A')
                if n_att==0: status="Not Run/No Code"
                elif chal['name']=="LRU Cache": status=det[0]['status'] if det else "Error"
                else: status=f"{n_pass}/{n_att} passed"
                summary.append(f"  {ai:<10}: Exec: {status}. Flake8: {f8}. API Time: {result['api_execution_time']:.2f}s.")
            print(f"--- Resumo Desafio {chal['name']} ---"); [print(s) for s in summary]; print(f"--- Desafio {chal['name']} concluído ---")
        if not self.results: print("\nNenhum resultado coletado."); return
        self.generate_report()

    def _prepare_dataframe(self) -> pd.DataFrame | None:
        """Prepara DataFrame principal."""
        if not self.results: print("Não há resultados p/ DF."); return None
        try:
             df = pd.DataFrame(self.results)
             # Calcula métricas agregadas DEPOIS de processar todos os desafios
             agg = self._calculate_aggregated_test_metrics()
             df['avg_exec_success_rate']=df.apply(lambda r: agg.get((r['ai'], r['challenge']), {}).get('avg_exec_success_rate',0.0), axis=1)
             df['avg_exec_time_ms']=df.apply(lambda r: agg.get((r['ai'], r['challenge']), {}).get('avg_exec_time_ms',np.nan), axis=1)
             df['total_tests_passed']=df.apply(lambda r: agg.get((r['ai'], r['challenge']), {}).get('total_tests_passed',0), axis=1)
             df['total_tests_attempted']=df.apply(lambda r: agg.get((r['ai'], r['challenge']), {}).get('total_tests_attempted',0), axis=1)
             df = df.drop(['execution_results_raw'], axis=1, errors='ignore')
             self._normalize_dataframe_types(df)
             return df
        except Exception as e: print(f"Erro criar/processar DF: {e}"); traceback.print_exc(); return None

    def _calculate_aggregated_test_metrics(self) -> dict:
        """Calcula métricas agregadas dos detalhes de teste."""
        agg = {}; keys = set((d['ai'], d['challenge']) for d in self.detailed_test_results)
        for k in keys:
            ai, chal = k; dets = [d for d in self.detailed_test_results if d['ai']==ai and d['challenge']==chal]
            if not dets: continue
            passed=[d for d in dets if d['status']=='Pass']; attempted=[d for d in dets if d['status']!='No Code']
            times=[d['execution_time_ms'] for d in dets if d.get('execution_time_ms',-1)>=0] # Inclui tempos de falha
            n_att=len(attempted); n_pass=len(passed); rate=n_pass/n_att if n_att>0 else 0.0; mean_t=np.mean(times) if times else np.nan
            agg[k]={'avg_exec_success_rate': rate,'avg_exec_time_ms':mean_t,'total_tests_passed':n_pass,'total_tests_attempted':n_att}
        return agg

    def _normalize_dataframe_types(self, df: pd.DataFrame):
        """Normaliza tipos de colunas numéricas."""
        cols = ['api_execution_time','lines_of_code','completeness','flake8_issues','num_execution_errors','avg_exec_success_rate','avg_exec_time_ms','total_tests_passed','total_tests_attempted']
        for c in cols:
            if c in df.columns:
                df[c]=pd.to_numeric(df[c],errors='coerce')
                fill_val = 0.0 # Default fill
                dtype = None # Default type
                if c=='flake8_issues': fill_val=-1; dtype=int
                elif c in ['num_execution_errors','total_tests_passed','total_tests_attempted']: fill_val=0; dtype=int
                elif c=='avg_exec_time_ms': fill_val=-1.0
                df.loc[:,c] = df[c].fillna(fill_val)
                if dtype: df.loc[:, c] = df[c].astype(dtype)
        if 'first_execution_error' in df.columns: df['first_execution_error']=df['first_execution_error'].fillna('').astype(str)

    def _save_environment_info(self, ts: str):
        """Salva info do ambiente."""
        p=os.path.join(self.output_dir, f"environment_info_{ts}.txt")
        try:
            with open(p,"w",encoding='utf-8') as f:
                for k,v in self.environment_info.items():
                    if isinstance(v,dict): f.write(f"{k}:\n"); [f.write(f"  {sk}:{sv}\n") for sk,sv in v.items()]
                    else: f.write(f"{k}: {v}\n")
            print(f"- Info Ambiente salvo: {p}")
        except Exception as e: print(f"Erro salvar info amb: {e}")

    def _save_full_results_json(self, ts: str):
        """Salva resultados completos em JSON."""
        p=os.path.join(self.output_dir, f"benchmark_results_full_{ts}.json")
        try:
            class NpEnc(json.JSONEncoder):
                def default(self, o):
                    if isinstance(o,(np.integer)): return int(o)
                    if isinstance(o,(np.floating)): return float(o)
                    if isinstance(o,np.ndarray): return o.tolist()
                    if isinstance(o,(datetime)): return o.isoformat()
                    if isinstance(o,np.bool_): return bool(o)
                    if isinstance(o,bytes): return o.decode('utf-8','replace')
                    return super(NpEnc,self).default(o)
            with open(p,"w",encoding='utf-8') as f: json.dump(self.results,f,indent=2,ensure_ascii=False,cls=NpEnc)
            print(f"- Resultados JSON salvo: {p}")
        except Exception as e: print(f"Erro salvar JSON: {e}")

    def _save_detailed_tests_csv(self, ts: str):
        """Salva detalhes dos testes em CSV."""
        p=os.path.join(self.output_dir, f"benchmark_test_details_{ts}.csv")
        if self.detailed_test_results:
            try:
                df=pd.DataFrame(self.detailed_test_results)
                cols=['ai','challenge','difficulty','test_case_index','status','execution_time_ms','error_message','stdout','input_args_str','expected_output_str','actual_output_str']
                for c in cols:
                    if c not in df.columns: df[c]=pd.NA # Adiciona coluna se faltar
                df[cols].to_csv(p,index=False,encoding='utf-8') # Seleciona e salva
                print(f"- Detalhes CSV salvo: {p}")
            except Exception as e: print(f"Erro salvar CSV detalhes: {e}")
        else: print("- Nenhum detalhe teste p/ salvar CSV.")

    def _create_csv_header(self) -> str:
        """Cria cabeçalho para CSVs."""
        hdr=["# --- Environment Info ---"]
        for k,v in self.environment_info.items():
            if isinstance(v,dict): hdr.append(f"# {k}:"); [hdr.append(f"#   {sk}: {sv}") for sk,sv in v.items()]
            else: hdr.append(f"# {k}: {v}")
        hdr.append("# --- Benchmark Data ---"); return "\n".join(hdr)+"\n"

    def _save_summary_csv(self, df: pd.DataFrame, ts: str):
        """Salva resumo em CSV."""
        p=os.path.join(self.output_dir, f"benchmark_summary_{ts}.csv")
        try:
            cols=[c for c in df.columns if c not in ['code','full_response']]; summary=df[cols].copy()
            hdr=self._create_csv_header();
            with open(p,"w",encoding='utf-8',newline='') as f: f.write(hdr); summary.to_csv(f,index=False,encoding='utf-8',float_format='%.4f',lineterminator='\n')
            print(f"- Resumo CSV salvo: {p}")
        except Exception as e: print(f"Erro salvar resumo CSV: {e}")

    def _calculate_statistics(self, df: pd.DataFrame) -> pd.DataFrame | None:
         """Calcula estatísticas agregadas."""
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
        """Salva estatísticas em CSV."""
        p = os.path.join(self.output_dir, f"benchmark_stats_{ts}.csv")
        try:
            hdr=self._create_csv_header();
            with open(p,"w",encoding='utf-8',newline='') as f: f.write(hdr); stats_df.to_csv(f,index=False,encoding='utf-8',float_format='%.2f',lineterminator='\n')
            print(f"- Estatísticas CSV salvo: {p}")
        except Exception as e: print(f"Erro salvar stats CSV: {e}")

    def _display_summary_table(self, stats_df: pd.DataFrame):
        """Mostra tabela de resumo no console."""
        print("\n===== SUMÁRIO DO BENCHMARK (ESTATÍSTICAS) =====");
        if stats_df is None or stats_df.empty: print("Não foi possível gerar sumário."); return
        disp=stats_df.copy()
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
            if fmt_col: disp[col]=fmt_col
        try: from tabulate import tabulate; print(tabulate(disp,headers='keys',tablefmt='grid',showindex=False))
        except ImportError: print("(Instale 'tabulate' p/ tabela bonita)"); print(disp.to_string(index=False))
        except Exception as e: print(f"Erro imprimir tabela: {e}"); print(disp.to_string(index=False))

    def generate_report(self):
        """Gera todos os relatórios e visualizações."""
        if not self.results: print("Não há resultados p/ relatório."); return
        print("\n--- Gerando Relatórios ---"); ts=datetime.now().strftime("%Y%m%d_%H%M%S"); print(f"Timestamp: {ts}"); print(f"Diretório: '{self.output_dir}/'")
        self._save_environment_info(ts); df=self._prepare_dataframe()
        if df is None: print("Falha preparar DF. Abortando."); return
        self._save_full_results_json(ts); self._save_detailed_tests_csv(ts); self._save_summary_csv(df,ts)
        stats=self._calculate_statistics(df)
        if stats is not None: self._save_statistics_csv(stats,ts)
        else: print("- Não possível calcular/salvar stats.")
        try: print("\n--- Gerando Visualizações ---"); self.create_visualizations(df,ts,self.output_dir); print(f"- Visualizações salvas em '{self.output_dir}/'")
        except Exception as e: print(f"Erro gerar visualizações: {e}")
        if stats is not None: self._display_summary_table(stats)

    def _plot_bar_charts(self, df_plot, ai_list, cmap, challenge_order, ts, out_dir):
        """Gera gráficos de barras."""
        print("Gerando Gráficos de Barras..."); metrics={"1_api_t":("API Time","s","api_execution_time"),"2_loc":("LoC","Linhas","lines_of_code"),"3_comp":("Completude","(%)","completeness"),"4_f8":("Flake8 Iss","Iss","flake8_issues"),"7_err":("Exec Err/Fail","Num","num_execution_errors"),}
        if 'avg_exec_success_rate' in df_plot.columns and df_plot['avg_exec_success_rate'].notna().any(): metrics["5_ok"]=("Exec Success","(%)","avg_exec_success_rate")
        if 'avg_exec_time_ms' in df_plot.columns and (df_plot['avg_exec_time_ms']!=-1).any(): metrics["6_exec_t"]=("Exec Time","(ms)","avg_exec_time_ms")
        metrics=dict(sorted(metrics.items()))
        for k,(t,y,c) in metrics.items():
            if c not in df_plot.columns: print(f"Skip bar '{t}': N/A."); continue
            data=df_plot.copy();
            if c=='flake8_issues': data=data[data[c]!=-1]
            elif c=='avg_exec_time_ms': data=data[data[c].notna()&(data[c]!=-1)]
            elif c=='num_execution_errors': data=data[data[c].notna()]
            elif c=='avg_exec_success_rate': data[c]=data[c]*100
            if data.empty or data[c].isnull().all(): print(f"Skip bar '{t}': Sem dados."); continue
            try:
                plt.figure(); piv=data.pivot_table(index='challenge',columns='ai',values=c,aggfunc='mean'); piv=piv.reindex(challenge_order,axis=0).reindex(ai_list,axis=1);
                fill=0.0 if c not in ['flake8_issues','avg_exec_time_ms','num_execution_errors'] else np.nan; piv.fillna(fill,inplace=True)
                if piv.isnull().all().all(): print(f"Skip bar '{t}': NaN."); plt.close(); continue
                n=len(ai_list); ax=piv.plot(kind='bar',figsize=(max(12,n*2),7),color=[cmap.get(ai,'grey') for ai in piv.columns],width=0.8)
                plt.title(f'{t} por Desafio'); plt.ylabel(y); plt.xlabel('Desafio'); plt.xticks(rotation=45,ha='right',fontsize=9); plt.yticks(fontsize=9)
                if c in ['completeness','avg_exec_success_rate']: plt.ylim(0,105)
                elif c in ['flake8_issues','num_execution_errors'] and not piv.isnull().all().all(): mv=piv.max(skipna=True).max(skipna=True); plt.ylim(bottom=0,top=max(1,mv*1.1) if not pd.isna(mv) else 1)
                elif c=='avg_exec_time_ms' and not piv.isnull().all().all(): plt.ylim(bottom=0)
                plt.legend(title='IA',bbox_to_anchor=(1.02,1),loc='upper left'); plt.tight_layout(rect=[0,0,0.88,1]); p=os.path.join(out_dir,f"{k}_{ts}.png"); plt.savefig(p); plt.close()
            except Exception as e: print(f"Erro bar '{t}': {e}"); plt.close()

    def _plot_box_plots(self, df_plot, ai_list, cmap, ts, out_dir):
        """Gera box plots."""
        print("Gerando Box Plots..."); metrics={"api_t":{"t":"Distribuição API Time","y":"Tempo(s)","v":None},"loc":{"t":"Distribuição LoC","y":"Linhas","v":(lambda d,c: d[c]>0)},"comp":{"t":"Distribuição Completude","y":"Score(%)","v":None}}
        if 'flake8_issues' in df_plot.columns and (df_plot['flake8_issues']!=-1).any(): metrics["f8"]={"t":"Distribuição Flake8 Issues","y":"Issues","v":(lambda d,c: d[c]!=-1)}
        if 'avg_exec_time_ms' in df_plot.columns and (df_plot['avg_exec_time_ms']!=-1).any(): metrics["exec_t"]={"t":"Distribuição Exec Time","y":"Tempo(ms)","v":(lambda d,c: d[c].notna()&(d[c]!=-1))}
        if 'num_execution_errors' in df_plot.columns and df_plot['num_execution_errors'].notna().any(): metrics["err"]={"t":"Distribuição Erros/Falhas Exec","y":"Num","v":(lambda d,c: d[c].notna())}
        cnt=7
        for k,cfg in metrics.items():
            cnt+=1;
            if k not in df_plot.columns: print(f"  Skip box '{cfg['t']}': N/A."); continue
            data=[]; lbls=[]; tmp=df_plot.copy()
            if cfg.get("v"): try: tmp=tmp[cfg["v"](tmp,k)]; except Exception as e: print(f" Aviso box cond '{k}': {e}.")
            if tmp.empty or tmp[k].isnull().all(): print(f"  Skip box '{cfg['t']}': Sem dados."); continue
            for ai in ai_list:
                ai_d=tmp[tmp['ai']==ai][k].dropna()
                if not ai_d.empty: data.append(ai_d.values); lbls.append(ai)
            if not data: print(f"  Skip box '{cfg['t']}': Sem dados por IA."); continue
            try:
                plt.figure(figsize=(max(8,len(lbls)*1.2),6)); bp=plt.boxplot(data,tick_labels=lbls,patch_artist=True,showfliers=True)
                plt.title(cfg["t"],fontsize=14); plt.ylabel(cfg["y"],fontsize=10); plt.xlabel("IA",fontsize=10); plt.xticks(rotation=30,ha='right',fontsize=9); plt.yticks(fontsize=9); plt.grid(axis='y',linestyle='--',alpha=0.7)
                colors=[cmap.get(lbl,'grey') for lbl in lbls];
                for patch, color in zip(bp['boxes'],colors): patch.set_facecolor(color); patch.set_alpha(0.6)
                plt.tight_layout(); fname=f"{cnt}_boxplot_{k}_{ts}.png"; p=os.path.join(out_dir,fname); plt.savefig(p); plt.close()
            except Exception as e: print(f"  Erro box plot '{cfg['t']}': {e}"); plt.close()
        print("Geração Box Plots concluída.")

    def _plot_radar_chart(self, df, ai_list, cmap, ts, out_dir):
        """Gera gráfico de radar."""
        n=len(ai_list); if n<2: print("Radar pulado (< 2 IAs)."); return
        print(f"Gerando radar p/ {n} IAs..."); cols=['api_execution_time','lines_of_code','completeness']
        try:
            if not all(c in df.columns for c in cols): print(f"Skip Radar: Faltando cols."); return
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
            cats=['Veloc API(Norm)','Conc LoC(Norm)','Completude(%)']; N=len(cats); ang=[n/N*2*np.pi for n in range(N)]; ang+=ang[:1]
            fig=plt.figure(figsize=(9,9)); ax=fig.add_subplot(111,polar=True)
            for ai in ai_list: vals=[t_norm.get(ai,0),l_norm.get(ai,0),c_norm.get(ai,0)]; pv=vals+vals[:1]; clr=cmap.get(ai,'grey'); ax.plot(ang,pv,'o-',linewidth=2,label=ai,color=clr); ax.fill(ang,pv,clr,alpha=0.2)
            ax.set_xticks(ang[:-1]); ax.set_xticklabels(cats); ax.set_yticks(np.arange(0,101,20)); ax.set_ylim(0,100)
            plt.legend(loc='upper right',bbox_to_anchor=(1.35,1.15)); plt.title('Comparação Geral (Norm)',size=14,y=1.18); plt.tight_layout()
            cnt=15; p=os.path.join(out_dir,f"{cnt}_radar_chart_{ts}.png"); plt.savefig(p); plt.close()
        except Exception as e: print(f"Erro radar: {e}"); plt.close()

    def create_visualizations(self, df, ts, out_dir):
        """Cria e salva visualizações."""
        plt.style.use('ggplot'); ais=sorted(df['ai'].unique()); n=len(ais)
        colors=plt.cm.tab10(np.linspace(0,1,n)) if n<=10 else plt.cm.viridis(np.linspace(0,1,n)); cmap={ai:colors[i] for i,ai in enumerate(ais)}
        req=['challenge','ai','api_execution_time','lines_of_code','completeness','flake8_issues','difficulty','num_execution_errors'] + [c for c in ['avg_exec_success_rate','avg_exec_time_ms'] if c in df.columns]
        if df.empty or not all(c in df.columns for c in req): print(f"AVISO: Faltando cols p/ gráficos. Pulando."); return
        dtype=pd.CategoricalDtype(['Easy','Medium','Hard'],ordered=True);
        try: df['diff_cat']=df['difficulty'].astype(dtype); order=df.sort_values('diff_cat')['challenge'].unique()
        except: print("AVISO: Erro ordenar dificuldade."); order=sorted(df['challenge'].unique())
        df_p=df.copy(); self._plot_bar_charts(df_p,ais,cmap,order,ts,out_dir); self._plot_box_plots(df_p,ais,cmap,ts,out_dir); self._plot_radar_chart(df,ais,cmap,ts,out_dir)

# --- Ponto de Entrada ---
if __name__ == "__main__":
    print("="*59); print(" AI Programming Benchmark: Claude vs ChatGPT vs Gemini"); print("      (Incluindo Análise Flake8 e Execução Externa/Local)"); print("="*59)
    # Necessário para 'spawn' no Windows se script for congelado
    multiprocessing.freeze_support()
    if not os.path.exists(".env"): print("AVISO: .env não encontrado.")
    try:
        benchmark = AIBenchmark()
        # Verifica se pelo menos uma API está funcional
        if benchmark.claude_api_key or benchmark.openai_api_key or benchmark.gemini_api_key:
            benchmark.run_benchmark()
            print("\nBenchmark concluído."); print(f"Relatórios salvos em: '{benchmark.output_dir}'")
        else: print("\nERRO FATAL: Nenhuma API configurada.")
    except Exception as e:
        print("\nERRO INESPERADO:"); print(f"Tipo: {type(e).__name__}"); print(f"Erro: {e}"); print("Traceback:"); traceback.print_exc()
    print("="*59)