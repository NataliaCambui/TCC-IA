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
import builtins # <--- Import necessário para __build_class__

try:
    # Python 3.8+
    from importlib import metadata as importlib_metadata
except ImportError:
    # Fallback para versões anteriores
    import pkg_resources # type: ignore

# Carrega variáveis de ambiente do .env
load_dotenv()

class AIBenchmark:
    """
    Classe para executar um benchmark de programação comparando modelos de IA
    (Claude, ChatGPT, Gemini) em desafios de código, avaliando a qualidade,
    corretude (via execução externa ou local para LRU) e desempenho.
    """
    def __init__(self):
        """Inicializa o benchmark, carrega chaves de API e configura o ambiente."""
        self.claude_api_key = os.getenv('CLAUDE_API_KEY')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')

        self.results = []
        self.detailed_test_results = [] # Lista principal para detalhes
        self.output_dir = "benchmark_reports"
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Diretório de saída: '{self.output_dir}/'")

        self._configure_gemini_api()

        self.claude_model = "claude-3-opus-20240229"
        self.openai_model = "gpt-4-turbo"
        self.gemini_model = "gemini-1.5-pro-latest" if self.gemini_api_key else None
        self.executor_url = "https://executor-service-1029404216382.europe-southwest1.run.app/execute"

        # --- LISTA DE DESAFIOS COM CASOS DE TESTE ---
        self.code_challenges = [
             { "name": "Fibonacci Sequence", "description": "Generate the first n numbers in the Fibonacci sequence, starting with 0.", "difficulty": "Easy", "function_signature": "fibonacci(n: int) -> list[int]", "test_cases": [ {"input_args": [10], "expected_output": [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]}, {"input_args": [1], "expected_output": [0]}, {"input_args": [0], "expected_output": []}, {"input_args": [2], "expected_output": [0, 1]} ] },
             { "name": "Binary Search", "description": "Find the index of a target value in a sorted list of integers. Return -1 if the target is not found.", "difficulty": "Medium", "function_signature": "binary_search(arr: list[int], target: int) -> int", "test_cases": [ {"input_args": [[2, 5, 7, 8, 11, 12], 13], "expected_output": -1}, {"input_args": [[2, 5, 7, 8, 11, 12], 8], "expected_output": 3}, {"input_args": [[1, 3, 5], 1], "expected_output": 0}, {"input_args": [[1, 3, 5], 5], "expected_output": 2}, {"input_args": [[], 5], "expected_output": -1} ] },
             { "name": "Merge Sort", "description": "Implement the merge sort algorithm to sort a list of integers in ascending order. Return a new sorted list.", "difficulty": "Medium", "function_signature": "merge_sort(arr: list[int]) -> list[int]", "test_cases": [ {"input_args": [[3, 1, 4, 1, 5, 9, 2, 6]], "expected_output": [1, 1, 2, 3, 4, 5, 6, 9]}, {"input_args": [[5, 4, 3, 2, 1]], "expected_output": [1, 2, 3, 4, 5]}, {"input_args": [[1]], "expected_output": [1]}, {"input_args": [[]], "expected_output": []} ] },
             { "name": "Count Islands in Grid", "description": "Count the number of islands in a 2D grid (list of lists) where 1 is land and 0 is water. An island is formed by connecting adjacent lands horizontally or vertically.", "difficulty": "Hard", "function_signature": "count_islands(grid: list[list[int]]) -> int", "test_cases": [ {"input_args": [[ [1, 1, 1, 1, 0], [1, 1, 0, 1, 0], [1, 1, 0, 0, 0], [0, 0, 0, 0, 0] ]], "expected_output": 1}, {"input_args": [[ [1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 1] ]], "expected_output": 3}, {"input_args": [[]], "expected_output": 0}, {"input_args": [[ [0, 0, 0], [0, 0, 0] ]], "expected_output": 0} ] },
             { "name": "Palindrome Check", "description": "Check if a given string is a palindrome, ignoring case and non-alphanumeric characters. Return True if it is, False otherwise.", "difficulty": "Easy", "function_signature": "is_palindrome(s: str) -> bool", "test_cases": [ {"input_args": ["A man, a plan, a canal: Panama"], "expected_output": True}, {"input_args": ["race a car"], "expected_output": False}, {"input_args": [" "], "expected_output": True}, {"input_args": [""], "expected_output": True}, {"input_args": ["Was it a car or a cat I saw?"], "expected_output": True}, {"input_args": ["tab a cat"], "expected_output": False} ] },
             { "name": "Valid Parentheses", "description": "Given a string containing just '(', ')', '{', '}', '[' and ']', determine if the input string is valid (brackets must close in the correct order).", "difficulty": "Medium", "function_signature": "is_valid_parentheses(s: str) -> bool", "test_cases": [ {"input_args": ["()"], "expected_output": True}, {"input_args": ["()[]{}"], "expected_output": True}, {"input_args": ["(]"], "expected_output": False}, {"input_args": ["([)]"], "expected_output": False}, {"input_args": ["{[]}"], "expected_output": True}, {"input_args": [""], "expected_output": True}, {"input_args": ["["], "expected_output": False}, {"input_args": ["]"], "expected_output": False} ] },
             { "name": "LRU Cache", "description": "Implement a Least Recently Used (LRU) cache with a given capacity. Support get(key) and put(key, value) operations. get returns value or -1. put inserts/updates; evicts LRU item if capacity exceeded.", "difficulty": "Hard", "function_signature": "class LRUCache:\n    def __init__(self, capacity: int):\n        pass\n    def get(self, key: int) -> int:\n        pass\n    def put(self, key: int, value: int) -> None:\n        pass", "test_cases": [ {"description": "Simple put/get sequence", "sequence": [ {"op": "__init__", "args": [2]}, {"op": "put", "args": [1, 1]}, {"op": "put", "args": [2, 2]}, {"op": "get", "args": [1], "expected": 1}, {"op": "put", "args": [3, 3]}, {"op": "get", "args": [2], "expected": -1}, {"op": "put", "args": [4, 4]}, {"op": "get", "args": [1], "expected": -1}, {"op": "get", "args": [3], "expected": 3}, {"op": "get", "args": [4], "expected": 4} ] }, ] }
        ]
        # --- FIM DA LISTA DE DESAFIOS ---

        self._check_flake8()
        self.environment_info = self._get_environment_info()

    def _configure_gemini_api(self):
        """Configura a API Gemini se a chave estiver disponível."""
        if self.gemini_api_key:
            try:
                genai.configure(api_key=self.gemini_api_key)
                print("Chave API Gemini configurada com sucesso.")
            except Exception as e:
                print(f"Erro ao configurar a API da Gemini: {e}. Gemini AI será pulada.")
                self.gemini_api_key = None
        else:
            print("AVISO: Chave API Gemini não encontrada. Gemini AI será pulada.")

    def _get_environment_info(self):
        """Coleta informações sobre o ambiente de execução."""
        info = {
            "Timestamp": datetime.now().isoformat(),
            "OS": f"{platform.system()} {platform.release()}",
            "Python Version": sys.version.split()[0],
            "Models": {
                "Claude": self.claude_model if self.claude_api_key else "N/A",
                "ChatGPT": self.openai_model if self.openai_api_key else "N/A",
                "Gemini": self.gemini_model if self.gemini_api_key else "N/A",
            },
            "Executor Service URL": self.executor_url,
            "Local LRU Cache Execution": True,
            "Libraries": {}
        }
        libs_to_check = [
            'pandas', 'numpy', 'matplotlib', 'requests',
            'python-dotenv', 'google-generativeai', 'google-api-core'
        ]
        try:
            for lib in libs_to_check:
                try:
                    info["Libraries"][lib] = importlib_metadata.version(lib)
                except importlib_metadata.PackageNotFoundError:
                    info["Libraries"][lib] = "Not Found"
        except NameError:
            for lib in libs_to_check:
                try:
                    info["Libraries"][lib] = pkg_resources.get_distribution(lib).version
                except pkg_resources.DistributionNotFound:
                     info["Libraries"][lib] = "Not Found"
                except Exception:
                     info["Libraries"][lib] = "Error getting version"
        return info

    def _check_flake8(self):
        """Verifica se o comando flake8 está disponível no sistema."""
        try:
            subprocess.run(
                [sys.executable, '-m', 'flake8', '--version'],
                check=True, capture_output=True, timeout=5
            )
            print("Verificação Flake8: OK")
            self.flake8_available = True
        except Exception:
            print(f"AVISO: 'flake8' não encontrado via '{sys.executable} -m flake8'. Análise de qualidade pulada.")
            self.flake8_available = False

    def query_claude(self, prompt: str) -> dict:
        """Envia um prompt para a API do Claude e retorna a resposta."""
        start_time = time.time()
        content = ""
        execution_time = 0
        if not self.claude_api_key:
            print("Pulando Claude: Chave API não fornecida.")
            execution_time = time.time() - start_time
            return {"content": "", "execution_time": execution_time}
        headers = {
            "x-api-key": self.claude_api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        data = {
            "model": self.claude_model,
            "max_tokens": 4000,
            "messages": [{"role": "user", "content": prompt}]
        }
        try:
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers, json=data, timeout=120
            )
            response.raise_for_status()
            result = response.json()
            if ("content" in result and isinstance(result["content"], list) and
                    len(result["content"]) > 0):
                first_content = result["content"][0]
                content = first_content.get("text", "") if isinstance(first_content, dict) else ""
            else:
                print(f"Resposta Claude: Estrutura 'content' inesperada. Resp: {result}")
            execution_time = time.time() - start_time
            return {"content": content or "", "execution_time": execution_time}
        except requests.exceptions.RequestException as e:
            error_message = f"Erro consulta Claude (RequestException): {e}"
            if e.response is not None:
                error_message += f" | Status: {e.response.status_code}"
                try:
                    error_message += f" | Body: {e.response.json()}"
                except json.JSONDecodeError:
                    error_message += f" | Body: {e.response.text}"
            print(error_message)
            execution_time = time.time() - start_time
            return {"content": "", "execution_time": execution_time}
        except Exception as e:
            print(f"Erro inesperado consulta Claude: {type(e).__name__} - {e}")
            execution_time = time.time() - start_time
            return {"content": "", "execution_time": execution_time}

    def query_chatgpt(self, prompt: str) -> dict:
        """Envia um prompt para a API do OpenAI (ChatGPT) e retorna a resposta."""
        start_time = time.time()
        content = ""
        execution_time = 0
        if not self.openai_api_key:
            print("Pulando ChatGPT: Chave API não fornecida.")
            execution_time = time.time() - start_time
            return {"content": "", "execution_time": execution_time}
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.openai_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4000
        }
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers, json=data, timeout=120
            )
            response.raise_for_status()
            result = response.json()
            if ("choices" in result and isinstance(result["choices"], list) and
                    len(result["choices"]) > 0):
                first_choice = result["choices"][0]
                if (isinstance(first_choice, dict) and "message" in first_choice and
                        isinstance(first_choice["message"], dict)):
                    content = first_choice["message"].get("content", "")
                else:
                    print(f"Resposta ChatGPT: Estrutura 'message' inesperada. Resp: {result}")
            else:
                print(f"Resposta ChatGPT: Estrutura 'choices' inesperada. Resp: {result}")
            execution_time = time.time() - start_time
            return {"content": content or "", "execution_time": execution_time}
        except requests.exceptions.RequestException as e:
            error_message = f"Erro consulta ChatGPT (RequestException): {e}"
            if e.response is not None:
                error_message += f" | Status: {e.response.status_code}"
                try:
                    error_details = e.response.json()
                    error_msg = error_details.get('error', {}).get('message', '')
                    error_message += f" | API Error: {error_msg}" if error_msg else f" | Body: {error_details}"
                except json.JSONDecodeError:
                    error_message += f" | Body: {e.response.text}"
            print(error_message)
            execution_time = time.time() - start_time
            return {"content": "", "execution_time": execution_time}
        except Exception as e:
            print(f"Erro inesperado consulta ChatGPT: {type(e).__name__} - {e}")
            execution_time = time.time() - start_time
            return {"content": "", "execution_time": execution_time}

    def query_gemini(self, prompt: str) -> dict:
        """Envia um prompt para a API do Gemini e retorna a resposta."""
        start_time = time.time()
        content = ""
        execution_time = 0
        if not self.gemini_api_key or not self.gemini_model:
            print("Pulando Gemini: Chave API não configurada ou modelo inválido.")
            execution_time = time.time() - start_time
            return {"content": "", "execution_time": execution_time}
        try:
            model = genai.GenerativeModel(self.gemini_model)
            safety_settings=[
                {"category": c, "threshold": "BLOCK_NONE"}
                for c in [
                    "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
                    "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"
                ]
            ]
            response = model.generate_content(prompt, safety_settings=safety_settings)
            execution_time = time.time() - start_time
            try:
                content = response.text
            except ValueError:
                print("Resposta Gemini não contém texto (possivelmente bloqueada).")
                try:
                    print(f"  Prompt Feedback: {response.prompt_feedback}")
                    if hasattr(response, 'candidates') and response.candidates:
                        print(f"  Finish Reason: {response.candidates[0].finish_reason}")
                except Exception as feedback_error:
                    print(f"  Erro ao acessar feedback Gemini: {feedback_error}")
            return {"content": content or "", "execution_time": execution_time}
        except google.api_core.exceptions.ResourceExhausted as e:
            print(f"Erro consulta Gemini (ResourceExhausted - Rate Limit): {e}\n>> Rate Limit. Tente mais tarde.")
            execution_time = time.time() - start_time
            return {"content": "", "execution_time": execution_time}
        except Exception as e:
            print(f"Erro inesperado consulta Gemini: {type(e).__name__} - {e}")
            execution_time = time.time() - start_time
            return {"content": "", "execution_time": execution_time}

    def extract_code(self, content: str) -> str:
        """Extrai o primeiro bloco de código Python de uma string de texto."""
        if not content:
            return ""
        code_blocks_python = re.findall(r'```python\n(.*?)\n```', content, re.DOTALL)
        if code_blocks_python:
            return code_blocks_python[0].strip()
        code_blocks_generic = re.findall(r'```(?:.*?\n)?(.*?)\n```', content, re.DOTALL)
        if code_blocks_generic:
            return code_blocks_generic[0].strip()
        lines = content.split('\n')
        code_lines = []
        start_found = False
        for line in lines:
            stripped = line.strip()
            if not start_found and (line.startswith(' ') or line.startswith('\t') or
                                     stripped.startswith('def ') or stripped.startswith('class ')):
                start_found = True
            if start_found:
                code_lines.append(line)
        if code_lines:
            return '\n'.join(code_lines).strip()
        if len(content.splitlines()) < 3 and ("def " not in content and "class " not in content):
            #print("AVISO: extract_code retornando vazio.") # Reduzindo verbosidade
            return ""
        else:
            #print("AVISO: extract_code usando fallback final.") # Reduzindo verbosidade
            return content.strip()

    def count_code_lines(self, code: str) -> int:
        """Conta as linhas de código 'reais', ignorando vazias e comentários."""
        if not code:
            return 0
        return len([line for line in code.splitlines()
                    if line.strip() and not line.strip().startswith('#')])

    def measure_response_completeness(self, code: str, description: str) -> int:
        """Estima a 'completude' da resposta da IA (score 0-100)."""
        completeness_score = 0
        lines_count = self.count_code_lines(code)
        if lines_count > 0:
            completeness_score += 50
            if "def " in code or "class " in code:
                completeness_score += 10
            if "return " in code or "yield " in code:
                completeness_score += 10
            elif "island" in description.lower() and ("dfs(" in code or "bfs(" in code):
                 completeness_score += 5
            desc_lower = description.lower()
            code_lower = code.lower()
            keyword_bonus = 0
            if "fibonacci" in desc_lower and ("fibonacci" in code_lower or "fib(" in code_lower or "yield" in code): keyword_bonus = 10
            elif "binary search" in desc_lower and ("mid =" in code or "mid=" in code or "low" in code or "high" in code): keyword_bonus = 10
            elif "merge sort" in desc_lower and ("merge(" in code_lower or ("left" in code and "right" in code and "mid" in code)): keyword_bonus = 10
            elif "island" in desc_lower and ("dfs(" in code_lower or "bfs(" in code_lower or "visit" in code_lower or ("grid[" in code and "==" in code)): keyword_bonus = 10
            elif "palindrome" in desc_lower and ("[::-1]" in code or ".isalnum()" in code_lower or ".lower()" in code_lower): keyword_bonus = 10
            elif "parentheses" in desc_lower and ("stack" in code_lower or "append(" in code or "pop(" in code): keyword_bonus = 10
            elif "lru cache" in desc_lower and ("dict" in code_lower or "capacity" in code_lower or "OrderedDict" in code): keyword_bonus = 10
            completeness_score += keyword_bonus
            structure_bonus = 0
            if "if" in code or "for" in code or "while" in code:
                structure_bonus = 10
            if structure_bonus > 0 and ("if not" in code or "if len(" in code or
                                        "== 0" in code or "is None" in code):
                structure_bonus += 5
            completeness_score += min(structure_bonus, 15)
        return min(completeness_score, 100)

    def analyze_code_quality_flake8(self, code_string: str) -> int:
        """Analisa código Python com Flake8 e retorna o número de problemas (-1 se erro)."""
        if not self.flake8_available or not code_string or self.count_code_lines(code_string) == 0:
            return -1
        temp_filepath = None
        try:
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.py', delete=False,
                                             encoding='utf-8') as temp_file:
                temp_file.write(code_string)
                temp_filepath = temp_file.name
            python_exe = sys.executable
            result = subprocess.run(
                [python_exe, '-m', 'flake8', '--count', temp_filepath],
                capture_output=True, text=True, timeout=15, check=False
            )
            if result.returncode in [0, 1] and result.stdout:
                try:
                    count = int(result.stdout.strip().splitlines()[-1])
                    return count
                except (IndexError, ValueError):
                    return len(result.stdout.strip().splitlines()) # Fallback
            elif result.stderr:
                print(f"Erro na execução do flake8: {result.stderr}")
                return -1
            else:
                return 0
        except subprocess.TimeoutExpired:
            print("Erro: Análise com Flake8 excedeu o tempo limite (15s).")
            return -1
        except Exception as e:
            print(f"Erro inesperado ao executar Flake8: {e}")
            return -1
        finally:
            if temp_filepath and os.path.exists(temp_filepath):
                try:
                    os.remove(temp_filepath)
                except Exception as e_rem:
                    print(f"Aviso: Não foi possível remover o arquivo temporário Flake8: {e_rem}")

    # --- Métodos para Execução Local Segura do LRU Cache (CORRIGIDOS) ---

    def _lru_worker(self, code_string: str, test_sequence: list, result_queue: multiprocessing.Queue):
        """
        Função alvo para o processo filho. Executa o código LRU com restrições.
        AVISO: A restrição de builtins é uma mitigação, não garantia de segurança.
        """
        results = []
        error_message = None
        lru_instance = None
        final_success = True

        try:
            # 1. Pré-processar código: Remover imports de collections/OrderedDict
            lines = code_string.splitlines()
            filtered_lines = [
                line for line in lines
                if not (line.strip().startswith("from collections import OrderedDict") or
                        line.strip().startswith("import collections") or
                        line.strip().startswith("import OrderedDict"))
            ]
            processed_code_string = "\n".join(filtered_lines)

            # 2. Preparar ambiente restrito (Permitindo __build_class__ e getattr)
            safe_builtins = {
                'int': int, 'str': str, 'list': list, 'dict': dict, 'tuple': tuple,
                'len': len, 'range': range, 'print': print,
                'True': True, 'False': False, 'None': None, 'object': object,
                'Exception': Exception, 'RuntimeError': RuntimeError, 'ValueError': ValueError,
                'NameError': NameError, 'AttributeError': AttributeError, 'TypeError': TypeError,
                'StopIteration': StopIteration,
                '__build_class__': builtins.__build_class__, # <--- CORREÇÃO
                'getattr': getattr,                         # <--- CORREÇÃO
                'isinstance': isinstance
            }
            safe_globals = {
                "__builtins__": safe_builtins,
                "OrderedDict": OrderedDict, # Passa explicitamente
                 # --- CORREÇÃO: Adicionar __name__ ---
                "__name__": "__exec_sandbox__"
                # --- FIM DA CORREÇÃO ---
            }

            # 3. Executa o código da classe (sem o import)
            local_namespace = {}
            exec(processed_code_string, safe_globals, local_namespace) # Usa o código processado

            # 4. Encontrar o nome da classe
            class_name = None
            for name, obj in local_namespace.items():
                if isinstance(obj, type) and name != 'OrderedDict':
                    class_name = name
                    break
            if not class_name:
                match = re.search(r"^\s*class\s+(\w+)", processed_code_string, re.MULTILINE)
                if match:
                    class_name = match.group(1)
                else:
                    raise NameError("Não foi possível encontrar nome da classe LRUCache no código.")

            LruClass = local_namespace.get(class_name)
            if not LruClass:
                raise NameError(f"Classe '{class_name}' não encontrada no namespace.")

            # 5. Executar sequência de testes
            op_total_time = 0.0
            for op_data in test_sequence:
                op_start_time = time.perf_counter()
                operation = op_data["op"]
                args = op_data.get("args", [])
                expected = op_data.get("expected")

                step_result = {
                    "operation": operation, "args": args, "expected": expected,
                    "output": None, "success": False, "error": None, "exec_time_ms": -1.0
                }
                try:
                    if operation == "__init__":
                        lru_instance = LruClass(*args)
                        step_result["success"] = True
                        step_result["output"] = "Instance created"
                    elif lru_instance is None:
                        raise RuntimeError("LRUCache não inicializada.")
                    elif operation in ["put", "get"]:
                         method_to_call = getattr(lru_instance, operation) # Usa getattr
                         if operation == "put":
                              method_to_call(*args)
                              step_result["success"] = True
                              step_result["output"] = "put executed"
                         elif operation == "get":
                              output = method_to_call(*args)
                              step_result["output"] = output
                              # Comparação mais robusta (converte ambos para string)
                              if str(output) == str(expected):
                                   step_result["success"] = True
                              else:
                                   step_result["error"] = f"Output '{output}' ({type(output)}) != Expected '{expected}' ({type(expected)})"
                                   final_success = False
                    else:
                        raise ValueError(f"Operação LRU desconhecida: {operation}")

                except Exception as e_step:
                    step_error_msg = f"{type(e_step).__name__}: {str(e_step).splitlines()[0]}"
                    step_result["error"] = step_error_msg
                    print(f"      Erro no passo LRU ({operation}): {step_error_msg}")
                    final_success = False # Falha no teste geral

                op_end_time = time.perf_counter()
                step_result["exec_time_ms"] = (op_end_time - op_start_time) * 1000
                op_total_time += (op_end_time - op_start_time)
                results.append(step_result)

                # Descomentar para parar no primeiro erro
                # if not step_result["success"]:
                #     break

            print(f"    Sequência LRU concluída no worker. Tempo total passos: {op_total_time*1000:.2f} ms")

        except Exception as e_main:
            error_message = f"Erro principal no worker LRU: {type(e_main).__name__} - {str(e_main).splitlines()[0]}"
            print(f"    ERRO GERAL NO WORKER LRU: {error_message}")
            traceback.print_exc() # Imprime traceback completo no log do worker
            final_success = False

        # Coloca o resultado final na fila
        try:
            result_queue.put({
                "success": final_success,
                "results": results,
                "error": error_message
            })
        except Exception as q_err:
            # Pode acontecer se a fila já estiver fechada pelo processo pai (raro)
             print(f"    ERRO ao colocar resultado na fila no worker LRU: {q_err}")


    def _execute_lru_locally(self, code_string: str, test_sequence: list) -> dict:
        """
        Tenta executar a sequência LRU localmente em processo separado.
        """
        ctx = multiprocessing.get_context(None) # Usa contexto padrão (spawn no Win)
        result_queue = ctx.Queue()
        timeout_seconds = 10 # Timeout mais curto para testes locais

        process = ctx.Process(
            target=self._lru_worker,
            args=(code_string, test_sequence, result_queue)
        )

        return_value = {"success": False, "results": [], "error": "Unknown execution error"}
        process_terminated = False

        try:
            start_time = time.time()
            process.start()
            process.join(timeout=timeout_seconds)

            if process.is_alive():
                print(f"  AVISO: Processo LRU excedeu timeout ({timeout_seconds}s). Terminando.")
                process.terminate()
                process.join(1)
                if process.is_alive():
                    print("  AVISO: Forçando kill processo LRU.")
                    process.kill()
                    process.join()
                return_value = {"success": False, "results": [], "error": f"Execution timed out after {timeout_seconds}s"}
                process_terminated = True

            elapsed_time = time.time() - start_time

            if not process_terminated:
                exit_code = process.exitcode
                # Exit code 0 significa que o processo terminou sem crashar, mas pode ter tido erros internos
                if exit_code == 0:
                    try:
                        # Tenta obter o resultado da fila (o worker pode ter terminado com erro interno)
                        result = result_queue.get(timeout=1)
                        print(f"  Execução local LRU concluída em {elapsed_time:.2f}s. Sucesso worker: {result.get('success')}")
                        return_value = result
                    except queue.Empty:
                        print("  ERRO: Processo LRU terminou (exitcode 0) mas fila vazia (erro interno no worker?).")
                        return_value = {"success": False, "results": [], "error": "Worker process finished successfully but queue was empty"}
                    except Exception as e_queue:
                        print(f"  ERRO: Falha obter resultado fila LRU: {e_queue}")
                        return_value = {"success": False, "results": [], "error": f"Failed to retrieve result from worker queue: {e_queue}"}
                else:
                     # Processo crashou (exit code != 0)
                     print(f"  ERRO: Processo LRU terminou com exitcode: {exit_code}")
                     return_value = {"success": False, "results": [], "error": f"Worker process exited with code {exit_code}"}

        except Exception as e_proc:
             print(f"  ERRO: Falha gerir processo LRU: {e_proc}")
             return_value = {"success": False, "results": [], "error": f"Failed to manage worker process: {e_proc}"}
             if process and process.is_alive():
                  try:
                      process.kill()
                      process.join()
                  except Exception: pass
        finally:
            # Limpeza robusta de recursos de multiprocessing
            try:
                # Tenta fechar a fila e esperar pelo thread alimentador
                result_queue.close()
                result_queue.join_thread()
            except (ValueError, OSError, EOFError) as e_q:
                 print(f"  Aviso: Erro menor ao limpar fila LRU: {e_q}") # Erros comuns se processo morreu
            except Exception as e_q_other:
                 print(f"  Aviso: Erro inesperado ao limpar fila LRU: {e_q_other}")

            try:
                 if process:
                     # Garante que o processo seja juntado se ainda existir
                     if process.is_alive():
                          process.join(0.1) # Tenta juntar rapidamente
                     if process.is_alive():
                          process.kill() # Mata se ainda estiver vivo
                          process.join()
                     process.close() # Fecha o objeto Process
            except Exception as e_p_clean:
                 print(f"  Aviso: Erro ao limpar processo LRU: {e_p_clean}")

        return return_value

    # --- Fim dos métodos de execução local ---

    def evaluate_solution(self, ai_name: str, challenge: dict, response: dict) -> dict:
        """
        Avalia a solução de uma IA. Usa execução local para LRU Cache.
        """
        content = response.get("content", "") if isinstance(response, dict) else ""
        api_execution_time = response.get("execution_time", 0) if isinstance(response, dict) else 0

        code = self.extract_code(content)
        lines_of_code = self.count_code_lines(code)
        completeness = self.measure_response_completeness(code, challenge["description"])
        flake8_issues = self.analyze_code_quality_flake8(code)

        execution_results_list = []
        processed_test_details_current = [] # Detalhes formatados *apenas* para este desafio/AI
        num_execution_errors = 0
        first_execution_error = ""

        can_attempt_execution = bool(code)
        actual_function_name = None
        if can_attempt_execution and challenge["name"] != "LRU Cache":
            actual_function_name = self._detect_function_name(code, challenge['name'])
            if not actual_function_name:
                can_attempt_execution = False # Não executa se não achar nome da função

        if can_attempt_execution:
            if challenge["name"] == "LRU Cache":
                print(f"  Tentando executar LRU Cache de {ai_name} localmente...")
                test_cases = challenge.get("test_cases", [])
                if test_cases and isinstance(test_cases[0], dict) and "sequence" in test_cases[0]:
                    lru_exec_result = self._execute_lru_locally(code, test_cases[0]["sequence"])
                    execution_results_list = lru_exec_result.get("results", [])
                    overall_success = lru_exec_result.get("success", False)
                    num_execution_errors = 0 if overall_success else 1
                    first_execution_error = lru_exec_result.get("error", "") # Erro geral do worker
                    if not first_execution_error and not overall_success:
                        for step in execution_results_list:
                            if step.get("error"):
                                first_execution_error = f"Step {step['operation']}{step['args']}: {step['error']}"
                                break
                    if not first_execution_error and not overall_success:
                         first_execution_error = "LRU Cache execution failed (unknown step error)"

                    avg_step_time = np.mean([s['exec_time_ms'] for s in execution_results_list if s.get('exec_time_ms',-1)>=0]) if execution_results_list else -1.0
                    # Adiciona entrada única para processed_test_details_current
                    processed_test_details_current.append({
                        "ai": ai_name, "challenge": challenge["name"], "difficulty": challenge["difficulty"],
                        "test_case_index": 0, "input_args_str": json.dumps(test_cases[0]["sequence"]),
                        "expected_output_str": "See sequence steps", "actual_output_str": "See sequence steps results",
                        "status": "Pass" if overall_success else "Fail" if execution_results_list else "Error",
                        "execution_time_ms": avg_step_time, "error_message": first_execution_error[:250], "stdout": None
                    })
                    print(f"  Resultado execução local LRU: {'Sucesso' if overall_success else 'Falha/Erro'}")
                else:
                    print("  AVISO: Casos de teste LRU formato inválido."); num_execution_errors = 1; first_execution_error = "LRU test case sequence not found."
                    processed_test_details_current.append({ "ai": ai_name, "challenge": challenge["name"], "difficulty": challenge["difficulty"], "test_case_index": 0, "input_args_str": "N/A", "expected_output_str": "N/A", "actual_output_str": None, "status": "Error", "execution_time_ms": -1, "error_message": first_execution_error, "stdout": None })

            else: # Outros desafios (actual_function_name deve ser válido aqui)
                print(f"  Tentando executar código de {ai_name} para '{challenge['name']}' via Executor Service...")
                results_ext, details_ext, errors_ext, first_err_ext = \
                    self._run_test_cases(ai_name, challenge, code, actual_function_name)
                execution_results_list = results_ext
                num_execution_errors = errors_ext
                first_execution_error = first_err_ext
                processed_test_details_current.extend(details_ext) # Adiciona detalhes formatados

        else: # Caso sem código ou nome de função
            error_msg = "No code extracted" if not code else "Function/Class name not detected"
            print(f"  AVISO: {error_msg} para '{challenge['name']}' por {ai_name}. Pulando execução.")
            num_tests = len(challenge.get("test_cases", [])) if challenge["name"] != "LRU Cache" else 1
            num_execution_errors = num_tests # Considera todos como erro
            first_execution_error = error_msg
            for i in range(num_tests):
                 processed_test_details_current.append({ # Adiciona placeholder de erro
                     "ai": ai_name, "challenge": challenge["name"], "difficulty": challenge["difficulty"],
                     "test_case_index": i, "input_args_str": None, "expected_output_str": None,
                     "actual_output_str": None, "status": "No Code", "execution_time_ms": -1,
                     "error_message": error_msg, "stdout": None
                 })

        # --- CORREÇÃO: Adicionar resultados processados (processed_test_details_current) à lista global ---
        self.detailed_test_results.extend(processed_test_details_current)
        # --- FIM DA CORREÇÃO ---

        # Retorna o dicionário agregado para self.results
        return {
            "ai": ai_name, "challenge": challenge["name"], "difficulty": challenge["difficulty"],
            "api_execution_time": api_execution_time, "lines_of_code": lines_of_code,
            "completeness": completeness, "flake8_issues": flake8_issues,
            "num_execution_errors": num_execution_errors, "first_execution_error": first_execution_error,
            "code": code, "full_response": content,
            "execution_results_raw": execution_results_list
        }

    def _detect_function_name(self, code: str, challenge_name: str) -> str | None:
        """Detecta o nome da função ou classe principal no código."""
        if not code: return None
        match = re.search(r"^\s*(?:def|class)\s+(\w+)", code, re.MULTILINE)
        if match:
            name = match.group(1)
            # Só imprime se não for LRU, para reduzir ruído
            if challenge_name != "LRU Cache":
                print(f"  Nome da função/classe detectado no código: '{name}'")
            return name
        else:
            return None # Não imprime aviso aqui

    def _run_test_cases(self, ai_name: str, challenge: dict, code: str,
                        actual_function_name: str) -> tuple[list, list, int, str]:
        """Executa casos de teste para funções usando o Executor Service externo."""
        execution_results_list = []
        processed_test_details = []
        num_failures_or_errors = 0
        first_error_message = ""
        test_cases = challenge.get("test_cases", [])

        if not test_cases:
            print(f"  Aviso: Nenhum caso de teste definido para '{challenge['name']}'.")
            return [], [], 0, ""

        for i, test_case in enumerate(test_cases):
            input_args = test_case.get("input_args")
            expected_output = test_case.get("expected_output")
            if input_args is None:
                print(f"  Aviso: Caso de teste {i+1} '{challenge['name']}' s/ 'input_args'. Pulando.")
                continue

            exec_result = self._call_executor_service(code, actual_function_name, input_args)
            execution_results_list.append(exec_result)

            test_detail = {
                "ai": ai_name, "challenge": challenge["name"], "difficulty": challenge["difficulty"],
                "test_case_index": i, "input_args_str": json.dumps(input_args),
                "expected_output_str": json.dumps(expected_output), "actual_output_str": None,
                "status": "Unknown", "execution_time_ms": exec_result.get("execution_time_ms", -1),
                "error_message": None, "stdout": exec_result.get("stdout")
            }
            current_case_error = None
            if exec_result.get("success") is True:
                actual_output = exec_result.get("output")
                test_detail["actual_output_str"] = json.dumps(actual_output)
                try:
                    # Comparação usando numpy (pode falhar para tipos complexos)
                    are_equal = np.array_equal(np.array(actual_output, dtype=object),
                                               np.array(expected_output, dtype=object))
                    test_detail["status"] = "Pass" if are_equal else "Fail"
                    if not are_equal:
                         current_case_error = f"Test Case {i} Failed: Output != Expected"
                         num_failures_or_errors += 1
                except Exception as compare_err:
                    test_detail["status"] = "Fail (Comparison Error)"
                    test_detail["error_message"] = f"Error comparing outputs: {compare_err}"
                    current_case_error = test_detail["error_message"]
                    num_failures_or_errors += 1
            elif exec_result.get("success") is False:
                test_detail["status"] = "Error"
                test_detail["error_message"] = exec_result.get("error", "Unknown execution error")
                current_case_error = str(test_detail["error_message"])[:250] # Limita tamanho
                num_failures_or_errors += 1
            else:
                 test_detail["status"] = "Unknown"
                 test_detail["error_message"] = exec_result.get("error", "Executor did not return success status")
                 current_case_error = str(test_detail["error_message"])[:250] # Limita tamanho
                 num_failures_or_errors += 1

            # Guarda a primeira mensagem de erro/falha encontrada
            if current_case_error and not first_error_message:
                first_error_message = current_case_error

            processed_test_details.append(test_detail)

        # Se não houve erro explícito, mas houve falha, define a msg genérica
        if num_failures_or_errors > 0 and not first_error_message:
             first_error_message = "One or more test cases failed comparison."

        return execution_results_list, processed_test_details, num_failures_or_errors, first_error_message

    def _call_executor_service(self, code_string: str, function_name: str, input_args: list) -> dict:
        """Chama o serviço externo para executar uma função."""
        imports_to_add = (
            "from typing import List, Dict, Tuple, Set, Optional, Any\n"
            "import re\n"
            "from collections import OrderedDict, deque\n\n"
        )
        full_code_to_execute = imports_to_add + code_string
        payload = {
            "code_string": full_code_to_execute,
            "function_name": function_name,
            "test_input_args": input_args
        }
        headers = {"Content-Type": "application/json"}
        default_error_result = {
            "success": False, "output": None, "stdout": None,
            "error": "Executor service call failed", "execution_time_ms": -1
        }
        try:
            response = requests.post(self.executor_url, headers=headers, json=payload, timeout=60)
            if response.status_code == 200:
                try:
                    return response.json()
                except json.JSONDecodeError as json_err:
                    print(f"  Erro: Falha JSON executor. Status: {response.status_code}, Erro: {json_err}")
                    default_error_result["error"] = f"Executor JSON decode error: {json_err}"
                    return default_error_result
            else:
                error_text = response.text[:200]
                print(f"  Erro: Chamada executor falhou. Status: {response.status_code}, Resp: {error_text}...")
                default_error_result["error"] = f"Executor status {response.status_code}: {error_text}"
                return default_error_result
        except requests.exceptions.Timeout:
            print(f"  Erro: Timeout executor service ({self.executor_url}).")
            default_error_result["error"] = "Executor service call timed out"
            return default_error_result
        except requests.exceptions.RequestException as req_err:
            print(f"  Erro: Conexão/rede executor service: {req_err}")
            default_error_result["error"] = f"Executor service connection error: {req_err}"
            return default_error_result
        except Exception as e:
            print(f"  Erro inesperado executor service: {type(e).__name__} - {e}")
            default_error_result["error"] = f"Unexpected error calling executor: {type(e).__name__}"
            return default_error_result

    def run_benchmark(self):
        """Executa o ciclo completo do benchmark."""
        can_run_claude = bool(self.claude_api_key)
        can_run_openai = bool(self.openai_api_key)
        can_run_gemini = bool(self.gemini_api_key and self.gemini_model)
        enabled_ais = [name for name, can_run in [
                       ("Claude", can_run_claude),
                       ("ChatGPT", can_run_openai),
                       ("Gemini", can_run_gemini)] if can_run]

        if not enabled_ais:
            print("ERRO: Nenhuma chave API configurada.")
            return

        print(f"Iniciando Benchmark para: {', '.join(enabled_ais)}")
        print(f"  Claude Model : {self.claude_model if can_run_claude else 'N/A'}")
        print(f"  ChatGPT Model: {self.openai_model if can_run_openai else 'N/A'}")
        print(f"  Gemini Model : {self.gemini_model if can_run_gemini else 'N/A'}")
        print(f"  LRU Cache Execution: Local Process")
        print("==========================================")

        challenge_details = {
           "Fibonacci Sequence": "Técnica: Lógica iterativa/recursiva.\n  Finalidade: Implementar algoritmos básicos, lidar com sequências e casos base.",
           "Binary Search": "Técnica: Busca eficiente (dividir/conquistar).\n  Finalidade: Implementar busca binária, lidar com índices, condições de parada.",
           "Merge Sort": "Técnica: Ordenação dividir/conquistar (recursão/merge).\n  Finalidade: Implementar ordenação recursiva, lógica de divisão e intercalação.",
           "Count Islands in Grid": "Técnica: Travessia grafo (Grid 2D) DFS/BFS.\n  Finalidade: Modelar grafo, implementar DFS/BFS, matrizes 2D, marcar visitados.",
           "Palindrome Check": "Técnica: Manipulação strings (limpeza/inversão/comparação).\n  Finalidade: Processar strings, ignorar caracteres/case, verificar simetria.",
           "Valid Parentheses": "Técnica: Pilha (Stack) para validação de ordem.\n  Finalidade: Usar pilhas para rastrear e validar ordem de fechamento.",
           "LRU Cache": "Técnica: Estrutura dados customizada (cache) LRU (dict/OrderedDict).\n  Finalidade: Projetar estruturas complexas, gerenciar capacidade, lógica get/put O(1), política despejo."
        }

        for challenge in self.code_challenges:
            signature = challenge.get("function_signature", f"{challenge['name'].lower().replace(' ', '_')}(...)")
            # Ajusta prompt para pedir import de OrderedDict explicitamente para LRU
            if "class LRUCache" in signature:
                 prompt = (
                     f"Implement Python class: {challenge['description']}. Signature:\n{signature}\n"
                     f"Provide ONLY Python code for class and imports (like `from collections import OrderedDict`). "
                     f"No explanations, examples, markdown."
                 )
            else:
                 prompt = (
                     f"Write Python function: `{signature}`. Problem: {challenge['description']}. "
                     f"Provide ONLY Python code for function and imports. "
                     f"No explanations, examples, markdown."
                 )

            print(f"\n--- Desafio: {challenge['name']} (Dificuldade: {challenge['difficulty']}) ---")
            description_detail = challenge_details.get(challenge['name'], "N/A.")
            print(f"  {description_detail}")

            responses = {}
            if can_run_claude:
                print("Consultando Claude AI...")
                responses["Claude"] = self.query_claude(prompt)
                time.sleep(1)
            if can_run_openai:
                print("Consultando ChatGPT...")
                responses["ChatGPT"] = self.query_chatgpt(prompt)
                time.sleep(1)
            if can_run_gemini:
                print("Consultando Gemini AI...")
                responses["Gemini"] = self.query_gemini(prompt)
                print("Aguardando 31s API Gemini...")
                time.sleep(31)

            challenge_summary_lines = []
            for ai_name in enabled_ais:
                response_data = responses.get(ai_name, {"content": "", "execution_time": 0})
                # Chama evaluate_solution (agora executa LRU localmente e guarda resultados)
                result = self.evaluate_solution(ai_name, challenge, response_data)
                self.results.append(result) # Guarda resultado agregado

                # Calcula resumo para print imediato
                tests_attempted = 0
                tests_passed = 0
                tests_error_fails = result.get('num_execution_errors', 0) # Conta erros/falhas

                # Filtra detalhes relevantes DEPOIS de evaluate_solution ter adicionado
                related_details = [d for d in self.detailed_test_results
                                   if d['ai'] == ai_name and d['challenge'] == challenge['name']]

                tests_attempted = len([d for d in related_details if d['status'] not in ['No Code']])
                tests_passed = len([d for d in related_details if d['status'] == 'Pass'])

                flake8_str = (str(result['flake8_issues'])
                              if result['flake8_issues'] != -1 else 'N/A')

                # Formata status da execução
                if tests_attempted == 0:
                     exec_status = "Not Run/No Code"
                elif challenge['name'] == "LRU Cache":
                     # Para LRU, usa o status agregado do processed_test_details (que é 1 entrada)
                     exec_status = related_details[0]['status'] if related_details else "Error"
                else:
                     exec_status = f"{tests_passed}/{tests_attempted} passed"

                summary_line = (
                    f"  {ai_name:<10}: Exec: {exec_status}. "
                    f"Flake8: {flake8_str}. "
                    f"API Time: {result['api_execution_time']:.2f}s."
                )
                challenge_summary_lines.append(summary_line)

            print(f"--- Resumo Desafio {challenge['name']} ---")
            for line in challenge_summary_lines:
                print(line)
            print(f"--- Desafio {challenge['name']} concluído ---")

        if not self.results:
            print("\nNenhum resultado coletado.")
            return
        self.generate_report()


    # --- Métodos de Geração de Relatório e Visualização (sem alterações) ---

    def _prepare_dataframe(self) -> pd.DataFrame | None:
        """Prepara o DataFrame principal com resultados e métricas agregadas."""
        if not self.results:
            print("Não há resultados p/ DataFrame.")
            return None
        try:
            df = pd.DataFrame(self.results)
            agg_metrics = self._calculate_aggregated_test_metrics()
            # Adiciona métricas agregadas ao DataFrame principal
            df['avg_exec_success_rate'] = df.apply(lambda r: agg_metrics.get((r['ai'], r['challenge']), {}).get('avg_exec_success_rate', 0.0), axis=1)
            df['avg_exec_time_ms'] = df.apply(lambda r: agg_metrics.get((r['ai'], r['challenge']), {}).get('avg_exec_time_ms', np.nan), axis=1)
            df['total_tests_passed'] = df.apply(lambda r: agg_metrics.get((r['ai'], r['challenge']), {}).get('total_tests_passed', 0), axis=1)
            df['total_tests_attempted'] = df.apply(lambda r: agg_metrics.get((r['ai'], r['challenge']), {}).get('total_tests_attempted', 0), axis=1)
            df = df.drop(['execution_results_raw'], axis=1, errors='ignore')
            self._normalize_dataframe_types(df)
            return df
        except Exception as e:
            print(f"Erro crítico ao criar/processar DataFrame: {e}\nTraceback: {traceback.format_exc()}")
            return None

    def _calculate_aggregated_test_metrics(self) -> dict:
        """Calcula métricas agregadas por (IA, Desafio) a partir dos detalhes de teste."""
        agg_metrics = {}
        # Usa a lista `self.detailed_test_results` que agora deve estar correta
        all_keys = set((d['ai'], d['challenge']) for d in self.detailed_test_results)

        for key in all_keys:
            ai_name, challenge_name = key
            details = [d for d in self.detailed_test_results
                       if d['ai'] == ai_name and d['challenge'] == challenge_name]

            if not details: continue # Deve ter detalhes se a chave existe

            passed_runs = [d for d in details if d['status'] == 'Pass']
            attempted_runs = [d for d in details if d['status'] not in ['No Code']]
            valid_times = [d['execution_time_ms'] for d in details if d.get('execution_time_ms', -1) >= 0]

            num_attempted = len(attempted_runs)
            num_passed = len(passed_runs)
            success_rate = num_passed / num_attempted if num_attempted > 0 else 0.0
            mean_time = np.mean(valid_times) if valid_times else np.nan

            agg_metrics[key] = {
                'avg_exec_success_rate': success_rate,
                'avg_exec_time_ms': mean_time,
                'total_tests_passed': num_passed,
                'total_tests_attempted': num_attempted # Agora deve ser > 0 se executado
            }
        return agg_metrics

    def _normalize_dataframe_types(self, df: pd.DataFrame):
        """Converte colunas para tipos numéricos e trata NaNs inplace."""
        num_cols = [
            'api_execution_time', 'lines_of_code', 'completeness', 'flake8_issues',
            'num_execution_errors', 'avg_exec_success_rate', 'avg_exec_time_ms',
            'total_tests_passed', 'total_tests_attempted'
        ]
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if col == 'flake8_issues': df.loc[:, col] = df[col].fillna(-1).astype(int)
                elif col == 'num_execution_errors': df.loc[:, col] = df[col].fillna(0).astype(int)
                elif col == 'avg_exec_success_rate': df.loc[:, col] = df[col].fillna(0.0)
                elif col == 'avg_exec_time_ms': df.loc[:, col] = df[col].fillna(-1.0)
                elif col in ['total_tests_passed', 'total_tests_attempted']: df.loc[:, col] = df[col].fillna(0).astype(int)
                else: df.loc[:, col] = df[col].fillna(0.0)
        if 'first_execution_error' in df.columns:
             df['first_execution_error'] = df['first_execution_error'].fillna('').astype(str)

    def _save_environment_info(self, ts: str):
        """Salva as informações do ambiente num arquivo TXT."""
        p = os.path.join(self.output_dir, f"environment_info_{ts}.txt")
        try:
            with open(p, "w", encoding='utf-8') as f:
                for k, v in self.environment_info.items():
                    if isinstance(v, dict):
                        f.write(f"{k}:\n"); [f.write(f"  {sk}: {sv}\n") for sk, sv in v.items()]
                    else: f.write(f"{k}: {v}\n")
            print(f"- Info Ambiente salvo: {p}")
        except Exception as e: print(f"Erro salvar info ambiente: {e}")

    def _save_full_results_json(self, ts: str):
        """Salva a lista completa de resultados brutos em JSON."""
        p = os.path.join(self.output_dir, f"benchmark_results_full_{ts}.json")
        try:
            class NpEncoder(json.JSONEncoder):
                def default(self, o):
                    if isinstance(o, np.integer): return int(o)
                    if isinstance(o, np.floating): return float(o)
                    if isinstance(o, np.ndarray): return o.tolist()
                    if isinstance(o, (datetime,)): return o.isoformat()
                    if isinstance(o, np.bool_): return bool(o)
                    if isinstance(o, bytes): return o.decode('utf-8', errors='replace')
                    return super(NpEncoder, self).default(o)
            with open(p, "w", encoding='utf-8') as f: json.dump(self.results, f, indent=2, ensure_ascii=False, cls=NpEncoder)
            print(f"- Resultados JSON salvo: {p}")
        except Exception as e: print(f"Erro salvar JSON: {e}")

    def _save_detailed_tests_csv(self, ts: str):
        """Salva os detalhes de cada caso de teste (ou sequência LRU) em CSV."""
        p = os.path.join(self.output_dir, f"benchmark_test_details_{ts}.csv")
        if self.detailed_test_results: # Verifica se a lista principal tem dados
            try:
                df = pd.DataFrame(self.detailed_test_results)
                cols = ['ai', 'challenge', 'difficulty', 'test_case_index', 'status',
                        'execution_time_ms', 'error_message', 'stdout', 'input_args_str',
                        'expected_output_str', 'actual_output_str']
                for c in cols: # Garante colunas
                    if c not in df.columns: df[c] = pd.NA
                df = df[cols]
                df.to_csv(p, index=False, encoding='utf-8')
                print(f"- Detalhes CSV salvo: {p}") # Mensagem de sucesso
            except Exception as e:
                print(f"Erro salvar CSV detalhes: {e}\nTraceback: {traceback.format_exc()}")
        else:
            # Mensagem corrigida
            print("- Nenhum detalhe de teste foi registrado para salvar em CSV.")

    def _create_csv_header(self) -> str:
        """Cria um cabeçalho de comentário com informações do ambiente para CSVs."""
        env_comments = ["# --- Environment Info ---"]
        for k, v in self.environment_info.items():
            if isinstance(v, dict):
                env_comments.append(f"# {k}:"); [env_comments.append(f"#   {sk}: {sv}") for sk, sv in v.items()]
            else: env_comments.append(f"# {k}: {v}")
        env_comments.append("# --- Benchmark Data ---")
        return "\n".join(env_comments) + "\n"

    def _save_summary_csv(self, df: pd.DataFrame, ts: str):
        """Salva o resumo do benchmark (DataFrame principal) em CSV."""
        p = os.path.join(self.output_dir, f"benchmark_summary_{ts}.csv")
        try:
            drop_cols = ['code', 'full_response']; summary_cols = [c for c in df.columns if c not in drop_cols]; summary = df[summary_cols].copy()
            hdr = self._create_csv_header()
            with open(p, "w", encoding='utf-8', newline='') as f:
                f.write(hdr)
                summary.to_csv(f, index=False, encoding='utf-8', float_format='%.4f', lineterminator='\n')
            print(f"- Resumo CSV salvo: {p}")
        except Exception as e: print(f"Erro salvar resumo CSV: {e}\nTraceback: {traceback.format_exc()}")

    def _calculate_statistics(self, df: pd.DataFrame) -> pd.DataFrame | None:
         """Calcula estatísticas agregadas por IA."""
         # Agora deve ter dados corretos de testes passados/tentados
         try:
             metrics = [
                 ("Avg API Time (s)", "api_execution_time", "mean"),
                 ("Avg LoC", "lines_of_code", "mean"),
                 ("Avg Completeness (%)", "completeness", "mean"),
                 ("Avg Flake8 Issues", "flake8_issues", "mean_valid"),
                 ("Total Passed", "total_tests_passed", "sum"),
                 ("Total Attempted", "total_tests_attempted", "sum"),
                 ("Overall Success (%)", "avg_exec_success_rate", "mean_percentage"),
                 ("Avg Exec Time (ms)", "avg_exec_time_ms", "mean_valid"),
                 ("Total Errors/Fails", "num_execution_errors", "sum"),
                 ("Example Error", "first_execution_error", "first_non_empty")
             ]
             ais = sorted(df['ai'].unique())
             data = {"Metric": [m[0] for m in metrics]}
             for ai in ais:
                 df_ai = df[df["ai"] == ai]
                 col_data = []
                 for _, col, agg in metrics:
                     if col not in df_ai.columns: col_data.append(np.nan); continue
                     if agg == "mean": v = df_ai[col].mean()
                     elif agg == "sum": v = df_ai[col].sum()
                     elif agg == "mean_valid":
                         d = df_ai[df_ai[col].notna() & (df_ai[col] != -1)][col]
                         v = d.mean() if not d.empty else np.nan
                     elif agg == "mean_percentage": v = df_ai[col].mean() * 100
                     elif agg == "first_non_empty":
                         errs = df_ai[df_ai[col].fillna('').astype(str) != ''][col]
                         v = errs.iloc[0] if not errs.empty else ""
                     else: v = np.nan
                     col_data.append(v)
                 data[ai] = col_data
             return pd.DataFrame(data)
         except Exception as e:
             print(f"Erro calcular estatísticas: {e}\nTraceback: {traceback.format_exc()}")
             return None

    def _save_statistics_csv(self, stats_df: pd.DataFrame, ts: str):
        """Salva as estatísticas calculadas em CSV."""
        p = os.path.join(self.output_dir, f"benchmark_stats_{ts}.csv")
        try:
            hdr = self._create_csv_header()
            with open(p, "w", encoding='utf-8', newline='') as f:
                f.write(hdr)
                stats_df.to_csv(f, index=False, encoding='utf-8', float_format='%.2f', lineterminator='\n')
            print(f"- Estatísticas CSV salvo: {p}")
        except Exception as e:
            print(f"Erro salvar estatísticas CSV: {e}\nTraceback: {traceback.format_exc()}")

    def _display_summary_table(self, stats_df: pd.DataFrame):
        """Exibe uma tabela de resumo das estatísticas no console."""
        print("\n===== SUMÁRIO DO BENCHMARK (ESTATÍSTICAS) =====")
        if stats_df is None or stats_df.empty:
            print("Não foi possível gerar sumário.")
            return
        disp = stats_df.copy()
        for col in disp.columns:
            if col == 'Metric': continue
            fmt_col = []
            if disp.empty: continue
            for i in disp.index:
                m = disp.loc[i, 'Metric']; v = disp.loc[i, col]; fmt_v = "N/A"
                if not pd.isna(v):
                    if "Flake8" in m: fmt_v=f"{v:.1f}" if v!=-1 else "N/A"
                    elif "Time (ms)" in m: fmt_v=f"{v:.1f}" if v!=-1 else "N/A"
                    elif "(%)" in m: fmt_v=f"{v:.1f}%"
                    elif "Total " in m: fmt_v=f"{int(v)}"
                    elif "Time (s)" in m: fmt_v=f"{v:.2f}"
                    elif "Example Error" in m: s=str(v); fmt_v=s[:60]+('...' if len(s)>60 else '')
                    else: fmt_v=f"{v:.1f}"
                fmt_col.append(fmt_v)
            if fmt_col: disp[col] = fmt_col
        try:
            from tabulate import tabulate
            print(tabulate(disp, headers='keys', tablefmt='grid', showindex=False))
        except ImportError:
            print("(Instale 'tabulate' p/ tabela bonita)")
            print(disp.to_string(index=False))
        except Exception as e:
            print(f"Erro imprimir tabela: {e}")
            print(disp.to_string(index=False)) # Fallback

    def generate_report(self):
        """Gera todos os relatórios e visualizações."""
        if not self.results: print("Não há resultados p/ relatório."); return
        print("\n--- Gerando Relatórios ---"); ts = datetime.now().strftime("%Y%m%d_%H%M%S"); print(f"Timestamp: {ts}"); print(f"Diretório: '{self.output_dir}/'")
        self._save_environment_info(ts)
        df = self._prepare_dataframe()
        if df is None: print("Falha preparar DataFrame. Abortando relatórios."); return
        self._save_full_results_json(ts)
        self._save_detailed_tests_csv(ts) # Deve funcionar agora
        self._save_summary_csv(df, ts)
        stats_df = self._calculate_statistics(df) # Deve ter dados de testes agora
        if stats_df is not None: self._save_statistics_csv(stats_df, ts)
        else: print("- Não foi possível calcular/salvar estatísticas.")
        try:
            print("\n--- Gerando Visualizações ---")
            self.create_visualizations(df, ts, self.output_dir) # Deve ter dados de execução agora
            print(f"- Visualizações salvas em '{self.output_dir}/'")
        except Exception as e: print(f"Erro gerar visualizações: {e}\nTraceback: {traceback.format_exc()}")
        if stats_df is not None: self._display_summary_table(stats_df) # Deve mostrar stats corretos

    def _plot_bar_charts(self, df_plot: pd.DataFrame, ai_list: list, ai_color_map: dict,
                         challenge_order: list, timestamp: str, output_dir: str):
        """Gera gráficos de barras para diversas métricas."""
        print("Gerando Gráficos de Barras...")
        metrics_to_plot = {
            "1_api_execution_time": ("Tempo Médio Resposta API", "Tempo (s)", "api_execution_time"),
            "2_lines_of_code": ("Média LoC", "Linhas", "lines_of_code"),
            "3_completeness": ("Completude Média", "Score (%)", "completeness"),
            "4_flake8_issues": ("Média Problemas Flake8", "Problemas", "flake8_issues"),
            "7_num_execution_errors": ("Média Erros/Falhas Exec", "Num Erros/Falhas", "num_execution_errors"),
        }
        # Adiciona gráficos de execução SE as colunas existirem e tiverem dados
        if 'avg_exec_success_rate' in df_plot.columns and df_plot['avg_exec_success_rate'].notna().any():
            metrics_to_plot["5_avg_exec_success_rate"] = ("Taxa Média Sucesso Exec", "Sucesso (%)", "avg_exec_success_rate")
        if 'avg_exec_time_ms' in df_plot.columns and df_plot['avg_exec_time_ms'].notna().any():
             # Apenas plota se houver tempos válidos (diferentes de -1 e não NaN)
             if (df_plot['avg_exec_time_ms'] != -1).any():
                  metrics_to_plot["6_avg_exec_time_ms"] = ("Tempo Médio Execução", "Tempo (ms)", "avg_exec_time_ms")

        metrics_to_plot = dict(sorted(metrics_to_plot.items()))

        for metric_key, (title, ylabel, col_name) in metrics_to_plot.items():
            if col_name not in df_plot.columns:
                 print(f"Pulando gráfico '{title}': Coluna {col_name} N/A.")
                 continue

            plot_data = df_plot.copy()
            # Filtra dados inválidos específicos da métrica
            if col_name == 'flake8_issues': plot_data = plot_data[plot_data[col_name] != -1]
            elif col_name == 'avg_exec_time_ms': plot_data = plot_data[plot_data[col_name].notna() & (plot_data[col_name] != -1)]
            elif col_name == 'num_execution_errors': plot_data = plot_data[plot_data[col_name].notna()]
            elif col_name == 'avg_exec_success_rate': plot_data[col_name] = plot_data[col_name] * 100

            if plot_data.empty or plot_data[col_name].isnull().all():
                print(f"Pulando gráfico '{title}': Sem dados válidos.")
                continue

            try:
                plt.figure()
                pivot = plot_data.pivot_table(index='challenge', columns='ai', values=col_name, aggfunc='mean')
                pivot = pivot.reindex(challenge_order, axis=0).reindex(ai_list, axis=1)
                fill_value = 0.0
                # Para tempos/erros, NaN é mais apropriado que 0 para média
                if col_name in ['flake8_issues', 'avg_exec_time_ms', 'num_execution_errors']:
                    fill_value = np.nan
                pivot.fillna(fill_value, inplace=True)

                # Se TUDO for NaN, não plota
                if pivot.isnull().all().all():
                    print(f"Pulando gráfico '{title}': Dados NaN.")
                    plt.close(); continue

                num_ais = len(ai_list)
                ax = pivot.plot(kind='bar', figsize=(max(12, num_ais * 2), 7), color=[ai_color_map.get(ai, 'grey') for ai in pivot.columns], width=0.8)
                plt.title(f'{title} por Desafio'); plt.ylabel(ylabel); plt.xlabel('Desafio (Ordenado por Dificuldade)')
                plt.xticks(rotation=45, ha='right', fontsize=9); plt.yticks(fontsize=9)

                # Ajusta eixos Y
                if col_name in ['completeness', 'avg_exec_success_rate']: plt.ylim(0, 105)
                elif col_name in ['flake8_issues', 'num_execution_errors'] and not pivot.isnull().all().all():
                    max_val = pivot.max(skipna=True).max(skipna=True) # Ignora NaN no max
                    plt.ylim(bottom=0, top = max(1, max_val * 1.1) if not pd.isna(max_val) else 1)
                elif col_name == 'avg_exec_time_ms' and not pivot.isnull().all().all():
                    # Apenas define limite inferior 0 para tempo
                    plt.ylim(bottom=0)

                plt.legend(title='IA', bbox_to_anchor=(1.02, 1), loc='upper left'); plt.tight_layout(rect=[0, 0, 0.88, 1])
                path = os.path.join(output_dir, f"{metric_key}_{timestamp}.png"); plt.savefig(path); plt.close()
            except Exception as e:
                print(f"Erro gerar gráfico barras '{title}': {e}")
                plt.close()

    def _plot_box_plots(self, df_plot: pd.DataFrame, ai_list: list, ai_color_map: dict,
                        timestamp: str, output_dir: str):
        """Gera box plots para comparar a distribuição das métricas entre IAs."""
        print("Gerando Box Plots...")
        metrics_for_boxplot = {
            "api_execution_time": {"title": "Distribuição Tempo Resposta API", "ylabel": "Tempo (s)", "valid": None},
            "lines_of_code": {"title": "Distribuição LoC", "ylabel": "Linhas", "valid": (lambda d,c: d[c]>0)},
            "completeness": {"title": "Distribuição Completude", "ylabel": "Score (%)", "valid": None}
        }
        # Adiciona box plots de execução SE as colunas existirem e tiverem dados
        if 'flake8_issues' in df_plot.columns and (df_plot['flake8_issues'] != -1).any():
             metrics_for_boxplot["flake8_issues"] = {"title": "Distribuição Problemas Flake8", "ylabel": "Problemas", "valid": (lambda d,c: d[c]!=-1)}
        if 'avg_exec_time_ms' in df_plot.columns and df_plot['avg_exec_time_ms'].notna().any() and (df_plot['avg_exec_time_ms'] != -1).any():
             metrics_for_boxplot["avg_exec_time_ms"] = {"title": "Distribuição Tempo Médio Execução", "ylabel": "Tempo (ms)", "valid": (lambda d,c: d[c].notna() & (d[c]!=-1))}
        if 'num_execution_errors' in df_plot.columns and df_plot['num_execution_errors'].notna().any():
             metrics_for_boxplot["num_execution_errors"] = {"title": "Distribuição Erros/Falhas Execução", "ylabel": "Num Erros/Falhas", "valid": (lambda d,c: d[c].notna())}

        plot_counter_boxplot = 7 # Reinicia contador (aproximado)

        for metric_key_boxplot, config in metrics_for_boxplot.items():
            plot_counter_boxplot += 1
            if metric_key_boxplot not in df_plot.columns:
                print(f"  Pulando Box Plot '{config['title']}': coluna N/A.")
                continue

            data_for_current_metric = []
            labels_for_current_metric = []
            temp_df = df_plot.copy()
            # Aplica filtro de validade
            if config.get("valid"):
                try:
                    condition_func = config["valid"]
                    temp_df = temp_df[condition_func(temp_df, metric_key_boxplot)]
                except Exception as e: print(f"  Aviso: Falha condição boxplot '{metric_key_boxplot}': {e}.")

            if temp_df.empty or temp_df[metric_key_boxplot].isnull().all():
                print(f"  Pulando Box Plot '{config['title']}': Sem dados válidos.")
                continue

            for ai_name in ai_list:
                ai_data = temp_df[temp_df['ai'] == ai_name][metric_key_boxplot].dropna()
                if not ai_data.empty:
                    data_for_current_metric.append(ai_data.values)
                    labels_for_current_metric.append(ai_name)

            if not data_for_current_metric:
                print(f"  Pulando Box Plot '{config['title']}': Nenhum dado por IA.")
                continue

            try:
                plt.figure(figsize=(max(8, len(labels_for_current_metric) * 1.2), 6))
                # Usa tick_labels em vez de labels (Matplotlib 3.9+)
                box_plot = plt.boxplot(data_for_current_metric,
                                       tick_labels=labels_for_current_metric,
                                       patch_artist=True, showfliers=True)
                plt.title(config["title"], fontsize=14); plt.ylabel(config["ylabel"], fontsize=10); plt.xlabel("IA", fontsize=10)
                plt.xticks(rotation=30, ha='right', fontsize=9); plt.yticks(fontsize=9); plt.grid(axis='y', linestyle='--', alpha=0.7)
                colors_to_use = [ai_color_map.get(label, 'grey') for label in labels_for_current_metric];
                for patch, color in zip(box_plot['boxes'], colors_to_use): patch.set_facecolor(color); patch.set_alpha(0.6)
                plt.tight_layout()
                clean_metric_name = metric_key_boxplot.replace('_ms', '')
                plot_filename = f"{plot_counter_boxplot}_boxplot_{clean_metric_name}_{timestamp}.png"
                path = os.path.join(output_dir, plot_filename); plt.savefig(path); plt.close()
            except Exception as e: print(f"  Erro ao gerar Box Plot '{config['title']}': {e}"); plt.close()
        print("Geração de Box Plots concluída.")

    def _plot_radar_chart(self, df: pd.DataFrame, ai_list: list, ai_color_map: dict,
                          timestamp: str, output_dir: str):
        """Gera um gráfico de radar comparando métricas normalizadas entre IAs."""
        num_ais = len(ai_list)
        if num_ais < 2: print("Gráfico Radar pulado (requer >= 2 IAs)."); return
        print(f"Gerando gráfico Radar para {num_ais} IAs (3 métricas base)...")
        try:
            numeric_cols_radar = ['api_execution_time', 'lines_of_code', 'completeness']
            if not all(col in df.columns for col in numeric_cols_radar):
                missing = [col for col in numeric_cols_radar if col not in df.columns]
                print(f"Pulando Radar: Faltando colunas {missing}"); return
            means_data = {}
            for ai in ai_list:
                df_ai_radar = df[df["ai"] == ai]; means = {}
                for col in numeric_cols_radar: means[col] = df_ai_radar[col].mean() if not df_ai_radar[col].isnull().all() else 0
                means_data[ai] = means
            all_times = [m.get("api_execution_time", 0) for m in means_data.values()]; all_locs = [m.get("lines_of_code", 0) for m in means_data.values()]; all_comps = [m.get("completeness", 0) for m in means_data.values()]
            max_time = max(all_times + [1e-6]); time_norm = {ai: 100 * (1 - means_data[ai].get("api_execution_time", 0) / max_time) if max_time > 0 else 0 for ai in ai_list}
            valid_locs = [loc for loc in all_locs if loc > 0]; min_loc_valid = min(valid_locs) if valid_locs else 0; max_loc_valid = max(all_locs + [1e-6]); lines_norm = {}
            for ai in ai_list:
                loc = means_data[ai].get("lines_of_code", 0)
                if loc <= 0: lines_norm[ai] = 0
                elif min_loc_valid > 0 and max_loc_valid > min_loc_valid: norm_val = 100 * (1 - (loc - min_loc_valid) / (max_loc_valid - min_loc_valid)); lines_norm[ai] = max(0, min(100, norm_val))
                else: lines_norm[ai] = 100 if loc > 0 else 0
                if loc == min_loc_valid and min_loc_valid > 0: lines_norm[ai] = 100
            comp_norm = {ai: means_data[ai].get("completeness", 0) for ai in ai_list}
            categories = ['Velocidade API\n(Norm)', 'Concisão LoC\n(Norm)', 'Completude\n(%)']; N = len(categories)
            angles = [n / float(N) * 2 * np.pi for n in range(N)]; angles += angles[:1]
            fig = plt.figure(figsize=(9, 9)); ax = fig.add_subplot(111, polar=True)
            for ai_name in ai_list:
                 values = [time_norm.get(ai_name, 0), lines_norm.get(ai_name, 0), comp_norm.get(ai_name, 0)]
                 plot_values = values + values[:1]
                 color = ai_color_map.get(ai_name, 'grey'); ax.plot(angles, plot_values, 'o-', linewidth=2, label=ai_name, color=color); ax.fill(angles, plot_values, color, alpha=0.2)
            ax.set_xticks(angles[:-1]); ax.set_xticklabels(categories); ax.set_yticks(np.arange(0, 101, 20)); ax.set_ylim(0, 100)
            plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15)); plt.title('Comparação Geral (3 Métricas Normalizadas)', size=14, y=1.18); plt.tight_layout()
            plot_counter_radar = 15 # Contador aproximado
            path = os.path.join(output_dir, f"{plot_counter_radar}_radar_chart_{timestamp}.png"); plt.savefig(path); plt.close()
        except Exception as e: print(f"Erro gerar gráfico Radar: {e}"); plt.close()

    def create_visualizations(self, df: pd.DataFrame, timestamp: str, output_dir: str):
        """Cria e salva visualizações (gráficos de barras, box plots, radar chart)."""
        plt.style.use('ggplot')
        ai_list = sorted(df['ai'].unique())
        num_ais = len(ai_list)
        if num_ais <= 10: colors = plt.cm.tab10(np.linspace(0, 1, num_ais))
        else: colors = plt.cm.viridis(np.linspace(0, 1, num_ais))
        ai_color_map = {ai: colors[i] for i, ai in enumerate(ai_list)}
        base_required_cols = ['challenge', 'ai', 'api_execution_time', 'lines_of_code', 'completeness', 'flake8_issues', 'difficulty', 'num_execution_errors']
        exec_cols = [col for col in ['avg_exec_success_rate', 'avg_exec_time_ms'] if col in df.columns]
        required_cols = base_required_cols + exec_cols
        if df.empty or not all(col in df.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in df.columns]
            print(f"AVISO: DataFrame vazio/faltando colunas p/ gráficos. Faltando: {missing_cols}. Pulando.")
            return
        difficulty_order = pd.CategoricalDtype(['Easy', 'Medium', 'Hard'], ordered=True)
        try: # Adiciona try/except para erro raro de categoria desconhecida
             df['difficulty_cat'] = df['difficulty'].astype(difficulty_order)
             challenge_order = df.sort_values('difficulty_cat')['challenge'].unique()
        except ValueError as e_cat:
             print(f"AVISO: Erro ao ordenar por dificuldade ({e_cat}). Usando ordem alfabética.")
             challenge_order = sorted(df['challenge'].unique())
        df_plot = df.copy()
        self._plot_bar_charts(df_plot, ai_list, ai_color_map, challenge_order, timestamp, output_dir)
        self._plot_box_plots(df_plot, ai_list, ai_color_map, timestamp, output_dir)
        self._plot_radar_chart(df, ai_list, ai_color_map, timestamp, output_dir)


# --- Ponto de Entrada Principal ---
if __name__ == "__main__":
    print("===========================================================")
    print(" AI Programming Benchmark: Claude vs ChatGPT vs Gemini")
    print("      (Incluindo Análise Flake8 e Execução Externa/Local)")
    print("===========================================================")

    multiprocessing.freeze_support() # Para compatibilidade (Windows)

    if not os.path.exists(".env"):
        print("AVISO: Arquivo .env não encontrado. Chaves API devem ser variáveis de ambiente.")

    try:
        benchmark = AIBenchmark()
        if benchmark.claude_api_key or benchmark.openai_api_key or benchmark.gemini_api_key:
            benchmark.run_benchmark()
            print("\nBenchmark concluído.")
            print(f"Relatórios salvos em: '{benchmark.output_dir}'")
        else:
            print("\nERRO FATAL: Nenhuma chave API config/encontrada.")

    except Exception as main_e:
        print("\nERRO INESPERADO DURANTE A EXECUÇÃO DO BENCHMARK:")
        print(f"Tipo: {type(main_e).__name__}"); print(f"Erro: {main_e}"); print("Traceback:"); traceback.print_exc()

    print("===========================================================")