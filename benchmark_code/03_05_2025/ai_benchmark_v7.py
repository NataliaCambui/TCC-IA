
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

# Carrega variáveis de ambiente do .env
load_dotenv()

# --- Configuração Global da API Gemini ---
# (código mantido igual)
gemini_api_key_global = None
try:
    gemini_api_key_loaded = os.getenv('GEMINI_API_KEY')
    if gemini_api_key_loaded:
        genai.configure(api_key=gemini_api_key_loaded)
        gemini_api_key_global = gemini_api_key_loaded
        print("Chave API Gemini configurada com sucesso.")
    else:
        print("AVISO: Chave de API da Gemini (GEMINI_API_KEY) não encontrada. Gemini AI será pulada.")
except Exception as e:
    print(f"Erro ao configurar a API da Gemini: {e}. Gemini AI será pulada.")

class AIBenchmark:
    def __init__(self):
        """Inicializa o benchmark."""
        self.claude_api_key = os.getenv('CLAUDE_API_KEY')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.gemini_api_key = gemini_api_key_global
        self.results = []
        self.output_dir = "benchmark_reports"
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Diretório de saída: '{self.output_dir}/'")

        # --- LISTA DE DESAFIOS COM CASOS DE TESTE ---
        # (Cole aqui a sua lista completa de self.code_challenges)
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

    def _check_flake8(self):
        try:
            subprocess.run(['flake8', '--version'], check=True, capture_output=True, timeout=5)
            print("Verificação Flake8: OK"); self.flake8_available = True
        except Exception:
            print("AVISO: Comando 'flake8' não encontrado/executável. Análise de qualidade será pulada."); self.flake8_available = False

    def query_claude(self, prompt):
        start_time = time.time(); execution_time = 0; content = ""
        if not self.claude_api_key: print("Pulando Claude: Chave API não fornecida."); execution_time = time.time() - start_time; return {"content": "", "execution_time": execution_time}
        headers = {"x-api-key": self.claude_api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        data = {"model": "claude-3-opus-20240229", "max_tokens": 4000, "messages": [{"role": "user", "content": prompt}]}
        try:
            response = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=data, timeout=120); response.raise_for_status()
            execution_time = time.time() - start_time; result = response.json()
            if "content" in result and isinstance(result["content"], list) and len(result["content"]) > 0: first_content = result["content"][0]; content = first_content.get("text", "") if isinstance(first_content, dict) else ""
            else: print(f"Resposta Claude: Estrutura 'content' inesperada. Resp: {result}")
            return {"content": content or "", "execution_time": execution_time}
        except requests.exceptions.RequestException as e:
            execution_time = time.time() - start_time
            error_message = f"Erro consulta Claude (RequestException): {e}"
            if e.response is not None:
                error_message += f" | Status: {e.response.status_code}"
                try:
                    error_message += f" | Body: {e.response.json()}"
                except json.JSONDecodeError:
                    error_message += f" | Body: {e.response.text}"
            print(error_message)
            return {"content": "", "execution_time": execution_time}
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"Erro inesperado consulta Claude: {type(e).__name__} - {e}")
            return {"content": "", "execution_time": execution_time}

    def query_chatgpt(self, prompt):
        start_time = time.time(); execution_time = 0; content = ""
        if not self.openai_api_key: print("Pulando ChatGPT: Chave API não fornecida."); execution_time = time.time() - start_time; return {"content": "", "execution_time": execution_time}
        headers = {"Authorization": f"Bearer {self.openai_api_key}", "Content-Type": "application/json"}
        data = {"model": "gpt-4-turbo", "messages": [{"role": "user", "content": prompt}], "max_tokens": 4000}
        try:
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data, timeout=120); response.raise_for_status()
            execution_time = time.time() - start_time; result = response.json()
            if "choices" in result and isinstance(result["choices"], list) and len(result["choices"]) > 0:
                first_choice = result["choices"][0]
                if isinstance(first_choice, dict) and "message" in first_choice and isinstance(first_choice["message"], dict): content = first_choice["message"].get("content", "")
                else: print(f"Resposta ChatGPT: Estrutura 'message' inesperada. Resp: {result}")
            else: print(f"Resposta ChatGPT: Estrutura 'choices' inesperada. Resp: {result}")
            return {"content": content or "", "execution_time": execution_time}
        except requests.exceptions.RequestException as e:
            execution_time = time.time() - start_time
            error_message = f"Erro consulta ChatGPT (RequestException): {e}"
            if e.response is not None:
                error_message += f" | Status: {e.response.status_code}"
                try:
                    error_details = e.response.json()
                    error_msg = error_details.get('error', {}).get('message', '')
                    if error_msg:
                        error_message += f" | API Error: {error_msg}"
                    else:
                        error_message += f" | Body: {error_details}"
                except json.JSONDecodeError:
                    error_message += f" | Body: {e.response.text}"
            print(error_message)
            return {"content": "", "execution_time": execution_time}
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"Erro inesperado consulta ChatGPT: {type(e).__name__} - {e}")
            return {"content": "", "execution_time": execution_time}

    def query_gemini(self, prompt):
        start_time = time.time(); execution_time = 0; content = ""
        if not self.gemini_api_key: print("Pulando Gemini: Chave API não configurada."); execution_time = time.time() - start_time; return {"content": "", "execution_time": execution_time}
        try:
            model = genai.GenerativeModel('gemini-1.5-pro-latest')
            safety_settings = [{"category": c, "threshold": "BLOCK_NONE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
            response = model.generate_content(prompt, safety_settings=safety_settings)
            execution_time = time.time() - start_time
            try:
                content = response.text
            except ValueError:
                print(f"Resposta Gemini não contém texto (possivelmente bloqueada).")
                try:
                    print(f"   Prompt Feedback: {response.prompt_feedback}")
                    if hasattr(response, 'candidates') and response.candidates:
                        # Print movido para linha seguinte e indentado
                        print(f"   Finish Reason: {response.candidates[0].finish_reason}")
                except Exception as feedback_error:
                    # Print movido para linha seguinte e indentado
                    print(f"   Erro ao acessar feedback Gemini: {feedback_error}")
            return {"content": content or "", "execution_time": execution_time}
        except google.api_core.exceptions.ResourceExhausted as e:
            execution_time = time.time() - start_time; print(f"Erro consulta Gemini (ResourceExhausted - Rate Limit): {e}"); print("                 >> Atingido o limite de requisições da API Gemini. Tente novamente mais tarde.")
            return {"content": "", "execution_time": execution_time}
        except Exception as e:
            execution_time = time.time() - start_time; print(f"Erro inesperado consulta Gemini: {type(e).__name__} - {e}");
            return {"content": "", "execution_time": execution_time}

    # --- Funções de Avaliação (extract_code, count_code_lines, etc.) ---
    def extract_code(self, content):
        if not content: return ""
        code_blocks_python = re.findall(r'```python\n(.*?)\n```', content, re.DOTALL);
        if code_blocks_python: return code_blocks_python[0].strip()
        code_blocks_generic = re.findall(r'```(?:.*?\n)?(.*?)\n```', content, re.DOTALL);
        if code_blocks_generic: return code_blocks_generic[0].strip()
        lines = content.split('\n'); code_lines = []; start_found = False
        for line in lines:
            stripped = line.strip()
            if not start_found and (line.startswith(' ') or line.startswith('\t') or stripped.startswith('def ') or stripped.startswith('class ')): start_found = True
            if start_found: code_lines.append(line)
        if code_lines: return '\n'.join(code_lines).strip()
        if len(content.splitlines()) < 3 and ("def " not in content and "class " not in content): return ""
        else: return content.strip()

    def count_code_lines(self, code):
        if not code: return 0
        return len([line for line in code.splitlines() if line.strip() and not line.strip().startswith('#')])

    def measure_response_completeness(self, code, description):
        completeness_score = 0; lines_count = self.count_code_lines(code)
        if lines_count > 0:
            completeness_score += 50
            if "def " in code or "class " in code: completeness_score += 10
            if "return " in code or "yield " in code: completeness_score += 10
            elif "island" in description.lower() and ("dfs(" in code or "bfs(" in code): completeness_score += 5
            desc_lower = description.lower(); code_lower = code.lower(); keyword_bonus = 0
            if "fibonacci" in desc_lower and ("fibonacci" in code_lower or "fib(" in code_lower or "yield" in code): keyword_bonus = 10
            elif "binary search" in desc_lower and ("mid =" in code or "mid=" in code or "low" in code or "high" in code): keyword_bonus = 10
            elif "merge sort" in desc_lower and ("merge(" in code_lower or ("left" in code and "right" in code and "mid" in code)): keyword_bonus = 10
            elif "island" in desc_lower and ("dfs(" in code_lower or "bfs(" in code_lower or "visit" in code_lower or ("grid[" in code and "==" in code)): keyword_bonus = 10
            elif "palindrome" in desc_lower and ("[::-1]" in code or ".isalnum()" in code_lower or ".lower()" in code_lower): keyword_bonus = 10
            elif "parentheses" in desc_lower and ("stack" in code_lower or "append(" in code or "pop(" in code): keyword_bonus = 10
            elif "lru cache" in desc_lower and ("dict" in code_lower or "capacity" in code_lower or "OrderedDict" in code): keyword_bonus = 10
            completeness_score += keyword_bonus; structure_bonus = 0
            if "if" in code or "for" in code or "while" in code: structure_bonus = 10
            if structure_bonus > 0 and ("if not" in code or "if len(" in code or "== 0" in code or "is None" in code): structure_bonus += 5
            completeness_score += min(structure_bonus, 15)
        return min(completeness_score, 100)

    def analyze_code_quality_flake8(self, code_string):
        if not self.flake8_available or not code_string or self.count_code_lines(code_string) == 0: return -1
        try:
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.py', delete=False, encoding='utf-8') as temp_file: temp_file.write(code_string); temp_filepath = temp_file.name
            python_exe = 'python' if platform.system() == 'Windows' else 'python3'
            result = subprocess.run([python_exe, '-m', 'flake8', '--count', temp_filepath], capture_output=True, text=True, timeout=15, check=False)
            if result.returncode in [0, 1] and result.stdout:
                try: count = int(result.stdout.strip().splitlines()[-1]); return count
                except (IndexError, ValueError): return len(result.stdout.strip().splitlines())
            elif result.stderr: print(f"Erro na execução do flake8: {result.stderr}"); return -1
            else: return 0
        except subprocess.TimeoutExpired: print("Erro: Análise com Flake8 excedeu o tempo limite (15s)."); return -1
        except Exception as e: print(f"Erro inesperado ao executar Flake8: {e}"); return -1
        finally:
            if 'temp_filepath' in locals() and os.path.exists(temp_filepath):
                try: os.remove(temp_filepath)
                except Exception as e_rem: print(f"Aviso: Não foi possível remover o arquivo temporário Flake8: {e_rem}")

    def evaluate_solution(self, ai_name, challenge, response):
        content = response.get("content", "") if isinstance(response, dict) else ""; api_execution_time = response.get("execution_time", 0) if isinstance(response, dict) else 0
        code = self.extract_code(content); lines_of_code = self.count_code_lines(code); completeness = self.measure_response_completeness(code, challenge["description"]); flake8_issues = self.analyze_code_quality_flake8(code)
        execution_results_list = []; actual_function_name = None
        if code:
            match = re.search(r"^\s*(?:def|class)\s+(\w+)", code, re.MULTILINE)
            if match: actual_function_name = match.group(1); print(f"   Nome da função/classe detectado no código: '{actual_function_name}'")
            else: print(f"   AVISO: Não foi possível detectar nome da função/classe no código extraído para '{challenge['name']}'. Pulando execução.")
        else: print(f"   AVISO: Nenhum código extraído para '{challenge['name']}'. Pulando execução.")

        if code and actual_function_name and challenge["name"] != "LRU Cache":
            print(f"   Tentando executar código de {ai_name} para '{challenge['name']}' (função: {actual_function_name}) via Executor Service...")
            test_cases = challenge.get("test_cases", [])
            if not test_cases: print(f"   Aviso: Nenhum caso de teste definido para '{challenge['name']}'. Pulando execução.")
            else:
                for i, test_case in enumerate(test_cases):
                    input_args = test_case.get("input_args"); expected_output = test_case.get("expected_output")
                    if input_args is None: print(f"   Aviso: Caso de teste {i+1} para '{challenge['name']}' não tem 'input_args'. Pulando."); continue
                    exec_result = self._call_executor_service(code, actual_function_name, input_args)
                    exec_result["test_case_index"] = i; exec_result["input_args"] = input_args; exec_result["expected_output"] = expected_output
                    execution_results_list.append(exec_result)
        elif challenge["name"] == "LRU Cache":
            execution_results_list.append({ "success": None, "output": None, "stdout": None, "error": "LRU Cache class execution not directly supported by current executor", "execution_time_ms": -1, "test_case_index": 0, "input_args": None, "expected_output": None })

        return { "ai": ai_name, "challenge": challenge["name"], "difficulty": challenge["difficulty"], "api_execution_time": api_execution_time, "lines_of_code": lines_of_code, "completeness": completeness, "flake8_issues": flake8_issues, "code": code, "full_response": content, "execution_results": execution_results_list }

    def _call_executor_service(self, code_string, function_name, input_args):
        executor_url = "https://executor-service-1029404216382.europe-southwest1.run.app/execute"
        imports_to_add = (
            "from typing import List, Dict, Tuple, Set, Optional, Any\n"
            "import re\n"
            "from collections import OrderedDict, deque\n\n"
        )
        full_code_to_execute = imports_to_add + code_string
        payload = { "code_string": full_code_to_execute, "function_name": function_name, "test_input_args": input_args }
        headers = {"Content-Type": "application/json"}
        default_error_result = { "success": False, "output": None, "stdout": None, "error": "Executor service call failed", "execution_time_ms": -1 }
        try:
            response = requests.post(executor_url, headers=headers, json=payload, timeout=60)
            if response.status_code == 200:
                try: return response.json()
                except json.JSONDecodeError as json_err: print(f"   Erro: Falha ao decodificar JSON da resposta do executor. Status: {response.status_code}, Erro: {json_err}"); default_error_result["error"] = f"Executor response JSON decode error: {json_err}"; return default_error_result
            else: print(f"   Erro: Chamada ao executor falhou. Status: {response.status_code}, Resposta: {response.text[:200]}..."); default_error_result["error"] = f"Executor service returned status {response.status_code}"; return default_error_result
        except requests.exceptions.Timeout: print(f"   Erro: Timeout ao chamar o executor service ({executor_url})."); default_error_result["error"] = "Executor service call timed out"; return default_error_result
        except requests.exceptions.RequestException as req_err: print(f"   Erro: Problema de conexão/rede ao chamar o executor service: {req_err}"); default_error_result["error"] = f"Executor service connection error: {req_err}"; return default_error_result
        except Exception as e: print(f"   Erro inesperado ao chamar o executor service: {type(e).__name__} - {e}"); default_error_result["error"] = f"Unexpected error calling executor: {type(e).__name__}"; return default_error_result

    def run_benchmark(self):
        can_run_claude = bool(self.claude_api_key); can_run_openai = bool(self.openai_api_key); can_run_gemini = bool(self.gemini_api_key)
        enabled_ais = [name for name, can_run in [("Claude", can_run_claude), ("ChatGPT", can_run_openai), ("Gemini", can_run_gemini)] if can_run]
        if not enabled_ais: print("ERRO: Nenhuma chave de API válida configurada."); return
        print(f"Benchmark para: {', '.join(enabled_ais)}"); print("==========================================")

        for challenge in self.code_challenges:
            signature = challenge.get("function_signature", f"{challenge['name'].lower().replace(' ', '_')}(...)")
            if "class LRUCache" in signature: prompt = ( f"Please implement the following Python class based on this description: {challenge['description']}. The class and method signatures should be:\n{signature}\nProvide ONLY the complete Python code for the class definition and necessary imports. Do not include any explanations, example usage, markdown formatting (like ```python), or introductory sentences.")
            else: prompt = ( f"Please write a single, complete Python function that implements the following signature: `{signature}`. The function should solve this problem: {challenge['description']}. Provide ONLY the Python code for the function definition and necessary imports. Do not include any explanations, example usage, markdown formatting (like ```python), or introductory sentences.")
            print(f"\n--- Desafio: {challenge['name']} (Dificuldade: {challenge['difficulty']}) ---"); responses = {}
            if can_run_claude: print("Consultando Claude AI..."); responses["Claude"] = self.query_claude(prompt); time.sleep(1)
            if can_run_openai: print("Consultando ChatGPT..."); responses["ChatGPT"] = self.query_chatgpt(prompt); time.sleep(1)
            if can_run_gemini: print("Consultando Gemini AI..."); responses["Gemini"] = self.query_gemini(prompt); print("Aguardando 31s para o limite da API Gemini..."); time.sleep(31)
            for ai_name in enabled_ais:
                response_data = responses.get(ai_name)
                if response_data is None: response_data = {"content": "", "execution_time": 0}
                result = self.evaluate_solution(ai_name, challenge, response_data)
                self.results.append(result)
            print(f"--- Desafio {challenge['name']} concluído ---")
        if not self.results: print("\nNenhum resultado coletado."); return
        self.generate_report()

    # --- Funções generate_report e create_visualizations ---
    # (Cole aqui as suas funções completas)
    def generate_report(self):
         if not self.results: print("Não há resultados para gerar o relatório."); return
         try:
             df = pd.DataFrame(self.results)
             avg_exec_success = []; avg_exec_time_ms = []; total_tests_passed = []; total_tests_attempted = []
             for row_results in df['execution_results']:
                 if not isinstance(row_results, list) or not row_results:
                     avg_exec_success.append(np.nan); avg_exec_time_ms.append(np.nan)
                     total_tests_passed.append(0); total_tests_attempted.append(0); continue
                 passed_runs = [r for r in row_results if r.get('success') == True and np.array_equal(np.array(r.get('output')), np.array(r.get('expected_output')))]
                 valid_times = [r.get('execution_time_ms') for r in row_results if r.get('execution_time_ms', -1) >= 0 and r.get('success') is not None]
                 num_attempted = len([r for r in row_results if r.get('success') is not None])
                 num_passed = len(passed_runs)
                 total_tests_passed.append(num_passed); total_tests_attempted.append(num_attempted)
                 success_rate = num_passed / num_attempted if num_attempted > 0 else 0; avg_exec_success.append(success_rate)
                 mean_time = np.mean(valid_times) if valid_times else np.nan; avg_exec_time_ms.append(mean_time)

             df['avg_exec_success_rate'] = avg_exec_success; df['avg_exec_time_ms'] = avg_exec_time_ms
             df['total_tests_passed'] = total_tests_passed; df['total_tests_attempted'] = total_tests_attempted
             df = df.drop('execution_results', axis=1)
         except Exception as e: print(f"Erro crítico ao criar/processar DataFrame: {e}\nResultados brutos: {self.results}"); return

         expected_numeric_cols = ['api_execution_time', 'lines_of_code', 'completeness', 'flake8_issues', 'avg_exec_success_rate', 'avg_exec_time_ms', 'total_tests_passed', 'total_tests_attempted']
         for col in expected_numeric_cols:
             if col in df.columns:
                 df[col] = pd.to_numeric(df[col], errors='coerce')
                 if col == 'flake8_issues': df.loc[:, col] = df[col].fillna(-1).astype(int)
                 elif col == 'avg_exec_success_rate': df.loc[:, col] = df[col].fillna(0.0)
                 elif col == 'avg_exec_time_ms': df.loc[:, col] = df[col].fillna(-1.0)
                 elif col in ['total_tests_passed', 'total_tests_attempted']: df.loc[:, col] = df[col].fillna(0).astype(int)
                 else: df.loc[:, col] = df[col].fillna(0.0)
         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S"); print(f"\nGerando relatórios em '{self.output_dir}/' com timestamp {timestamp}")

         json_path = os.path.join(self.output_dir, f"benchmark_results_full_{timestamp}.json")
         try:
             class NpEncoder(json.JSONEncoder):
                 def default(self, obj):
                     if isinstance(obj, np.integer): return int(obj)
                     if isinstance(obj, np.floating): return float(obj)
                     if isinstance(obj, np.ndarray): return obj.tolist()
                     if isinstance(obj, (datetime,)): return obj.isoformat()
                     if isinstance(obj, np.bool_): return bool(obj)
                     return super(NpEncoder, self).default(obj)
             with open(json_path, "w", encoding='utf-8') as f: json.dump(self.results, f, indent=2, ensure_ascii=False, cls=NpEncoder)
             print(f"- Resultados detalhados (JSON completo): {json_path}")
         except Exception as e: print(f"Erro ao salvar JSON completo: {e}")

         summary_path = os.path.join(self.output_dir, f"benchmark_summary_{timestamp}.csv")
         try: cols_to_drop_summary = ['code', 'full_response']; summary_cols = [col for col in df.columns if col not in cols_to_drop_summary]; summary = df[summary_cols]; summary.to_csv(summary_path, index=False, encoding='utf-8', float_format='%.4f'); print(f"- Resumo salvo (CSV): {summary_path}")
         except Exception as e: print(f"Erro ao salvar resumo CSV: {e}")

         stats_path = os.path.join(self.output_dir, f"benchmark_stats_{timestamp}.csv")
         stats = None
         try:
             stats_metrics = [ ("Average API Time (s)", "api_execution_time"), ("Average Lines of Code", "lines_of_code"), ("Average Completeness (%)", "completeness"), ("Average Flake8 Issues", "flake8_issues"), ("Average Execution Success (%)", "avg_exec_success_rate"), ("Average Execution Time (ms)", "avg_exec_time_ms"), ("Total Tests Passed", "total_tests_passed"), ("Total Tests Attempted", "total_tests_attempted") ]
             ai_list = sorted(df['ai'].unique()); stats_data = {"Metric": [m[0] for m in stats_metrics]}
             for ai_name in ai_list:
                 df_ai = df[df["ai"] == ai_name]; ai_column_data = []
                 for metric_name, col_key in stats_metrics:
                     if col_key not in df_ai.columns: ai_column_data.append(np.nan); continue
                     if col_key == 'flake8_issues': valid_data = df_ai[df_ai[col_key] != -1][col_key]; avg_val = valid_data.mean() if not valid_data.empty else -1
                     elif col_key == 'avg_exec_time_ms': valid_data = df_ai[df_ai[col_key] != -1][col_key]; avg_val = valid_data.mean() if not valid_data.empty else -1
                     elif col_key == 'avg_exec_success_rate': avg_val = df_ai[col_key].mean() * 100
                     elif col_key in ['total_tests_passed', 'total_tests_attempted']: avg_val = df_ai[col_key].sum()
                     else: avg_val = df_ai[col_key].mean()
                     ai_column_data.append(avg_val)
                 stats_data[ai_name] = ai_column_data
             stats = pd.DataFrame(stats_data); stats.to_csv(stats_path, index=False, encoding='utf-8', float_format='%.2f'); print(f"- Estatísticas salvas (CSV): {stats_path}")
         except Exception as e: print(f"Erro ao calcular/salvar estatísticas: {e}")

         try: self.create_visualizations(df, timestamp, self.output_dir); print(f"- Visualizações salvas como PNG em '{self.output_dir}/'")
         except Exception as e: print(f"Erro ao gerar visualizações: {e}")

         print("\n===== SUMÁRIO DO BENCHMARK =====");
         if stats is not None:
             stats_display = stats.copy()
             for col in stats_display.columns:
                 if col != 'Metric':
                     for i in stats_display.index:
                         metric_name = stats_display.loc[i, 'Metric']; value = stats_display.loc[i, col]
                         if pd.isna(value): stats_display.loc[i, col] = "N/A"
                         elif "Flake8" in metric_name or "Time (ms)" in metric_name: stats_display.loc[i, col] = f"{value:.1f}" if value != -1 else "N/A"
                         elif "Success (%)" in metric_name or "Completeness (%)" in metric_name: stats_display.loc[i, col] = f"{value:.1f}%"
                         elif "Total Tests" in metric_name: stats_display.loc[i, col] = f"{int(value)}"
                         elif "Time (s)" in metric_name: stats_display.loc[i, col] = f"{value:.2f}"
                         else: stats_display.loc[i, col] = f"{value:.1f}"
             print(stats_display.to_string(index=False))
         else: print("Não foi possível gerar o sumário de estatísticas.")


    def create_visualizations(self, df, timestamp, output_dir):
         plt.style.use('ggplot'); ai_list = sorted(df['ai'].unique()); num_ais = len(ai_list)
         if num_ais <= 10: colors = plt.cm.tab10(np.linspace(0, 1, num_ais))
         else: colors = plt.cm.viridis(np.linspace(0, 1, num_ais))
         ai_color_map = {ai: colors[i] for i, ai in enumerate(ai_list)}
         base_required_cols = ['challenge', 'ai', 'api_execution_time', 'lines_of_code', 'completeness', 'flake8_issues', 'difficulty']
         exec_cols = [col for col in ['avg_exec_success_rate', 'avg_exec_time_ms'] if col in df.columns]; required_cols = base_required_cols + exec_cols
         if df.empty or not all(col in df.columns for col in required_cols): print("AVISO: DataFrame vazio ou faltando colunas para gerar gráficos."); return
         difficulty_order = pd.CategoricalDtype(['Easy', 'Medium', 'Hard'], ordered=True); df['difficulty_cat'] = df['difficulty'].astype(difficulty_order)
         challenge_order = df.sort_values('difficulty_cat')['challenge'].unique(); df_plot = df.copy()
         metrics_to_plot = { "1_api_execution_time": ("Tempo Médio de Resposta API", "Tempo (s)", "api_execution_time"), "2_lines_of_code": ("Média Linhas de Código (LoC)", "Número de Linhas", "lines_of_code"), "3_completeness": ("Completude Média da Solução", "Score (%)", "completeness"), "4_flake8_issues": ("Média Problemas Flake8", "Número de Problemas", "flake8_issues") }
         if 'avg_exec_success_rate' in exec_cols: metrics_to_plot["5_avg_exec_success_rate"] = ("Taxa Média de Sucesso na Execução", "Sucesso (%)", "avg_exec_success_rate")
         if 'avg_exec_time_ms' in exec_cols: metrics_to_plot["6_avg_exec_time_ms"] = ("Tempo Médio de Execução (Executor)", "Tempo (ms)", "avg_exec_time_ms")

         for metric_key, (title, ylabel, col_name) in metrics_to_plot.items():
             if col_name == 'flake8_issues': plot_data = df_plot[df_plot[col_name] != -1]
             elif col_name == 'avg_exec_time_ms': plot_data = df_plot[df_plot[col_name] != -1]
             # <<< CORREÇÃO DE INDENTAÇÃO/ESTRUTURA ABAIXO >>>
             elif col_name == 'avg_exec_success_rate':
                 # Usar indentação consistente (4 espaços)
                 plot_data = df_plot.copy() # Instrução separada
                 if col_name in plot_data.columns:
                     plot_data[col_name] = plot_data[col_name] * 100
                 else:
                     continue # Pula se a coluna não existir
             # <<< FIM DA CORREÇÃO DE INDENTAÇÃO >>>
             else: plot_data = df_plot
             if plot_data.empty or col_name not in plot_data.columns or plot_data[col_name].isnull().all(): print(f"Pulando gráfico de barras '{title}': Nenhum dado válido encontrado."); continue
             try:
                 plt.figure(); pivot = plot_data.pivot_table(index='challenge', columns='ai', values=col_name, aggfunc='mean')
                 pivot = pivot.reindex(challenge_order, axis=0).reindex(ai_list, axis=1); fill_value = 0.0;
                 if col_name in ['flake8_issues', 'avg_exec_time_ms']: fill_value = np.nan
                 pivot.fillna(fill_value, inplace=True); ax = pivot.plot(kind='bar', figsize=(max(12, num_ais * 2.5), 7), color=[ai_color_map.get(ai, 'grey') for ai in pivot.columns], width=0.8)
                 plt.title(f'{title} por Desafio'); plt.ylabel(ylabel); plt.xlabel('Desafio (Ordenado por Dificuldade)'); plt.xticks(rotation=45, ha='right', fontsize=9); plt.yticks(fontsize=9)
                 if col_name in ['completeness', 'avg_exec_success_rate']: plt.ylim(0, 105)
                 if col_name == 'flake8_issues' and not pivot.isnull().all().all(): max_val = pivot.max().max(); plt.ylim(bottom=-0.1 if max_val <= 1 else 0)
                 if col_name == 'avg_exec_time_ms' and not pivot.isnull().all().all(): plt.ylim(bottom=0)
                 plt.legend(title='IA', bbox_to_anchor=(1.02, 1), loc='upper left'); plt.tight_layout(rect=[0, 0, 0.88, 1])
                 path = os.path.join(output_dir, f"{metric_key}_{timestamp}.png"); plt.savefig(path); plt.close()
             except Exception as e: print(f"Erro ao gerar gráfico de barras '{title}': {e}"); plt.close()

         print("Gerando Box Plots...")
         metrics_for_boxplot = { "api_execution_time": {"title": "Distribuição do Tempo de Resposta API por IA", "ylabel": "Tempo (s)", "valid_condition": None}, "lines_of_code": {"title": "Distribuição de Linhas de Código (LoC) por IA", "ylabel": "Número de Linhas", "valid_condition": (lambda df, col: df[col] > 0)}, "completeness": {"title": "Distribuição da Completude da Solução por IA", "ylabel": "Score (%)", "valid_condition": None} }
         if 'flake8_issues' in df_plot.columns: metrics_for_boxplot["flake8_issues"] = {"title": "Distribuição de Problemas Flake8 por IA", "ylabel": "Número de Problemas", "valid_condition": (lambda df, col: df[col] != -1)}
         if 'avg_exec_time_ms' in df_plot.columns: metrics_for_boxplot["avg_exec_time_ms"] = {"title": "Distribuição do Tempo Médio de Execução (Executor) por IA", "ylabel": "Tempo (ms)", "valid_condition": (lambda df, col: df[col] != -1)}
         plot_counter_boxplot = len(metrics_to_plot)
         for metric_key_boxplot, config in metrics_for_boxplot.items():
             plot_counter_boxplot += 1
             if metric_key_boxplot not in df_plot.columns: print(f"  Pulando Box Plot para '{metric_key_boxplot}': coluna não encontrada."); continue
             data_for_current_metric = []; labels_for_current_metric = []; temp_df = df_plot.copy()
             if config.get("valid_condition"):
                 try: condition_func = config["valid_condition"]; temp_df = temp_df[condition_func(temp_df, metric_key_boxplot)]
                 except Exception as e: print(f"  Aviso: Falha ao aplicar condição para boxplot de '{metric_key_boxplot}': {e}.")
             if temp_df.empty or temp_df[metric_key_boxplot].isnull().all(): print(f"  Pulando Box Plot para '{metric_key_boxplot}': Sem dados válidos."); continue
             for ai_name in ai_list:
                 ai_data = temp_df[temp_df['ai'] == ai_name][metric_key_boxplot].dropna()
                 if not ai_data.empty: data_for_current_metric.append(ai_data.values); labels_for_current_metric.append(ai_name)
             if not data_for_current_metric: print(f"  Pulando Box Plot para '{metric_key_boxplot}': Nenhum dado encontrado."); continue
             try:
                 plt.figure(figsize=(max(8, len(labels_for_current_metric) * 1.2), 6)); box_plot = plt.boxplot(data_for_current_metric, labels=labels_for_current_metric, patch_artist=True, showfliers=True)
                 plt.title(config["title"], fontsize=14); plt.ylabel(config["ylabel"], fontsize=10); plt.xlabel("IA", fontsize=10); plt.xticks(rotation=30, ha='right', fontsize=9); plt.yticks(fontsize=9); plt.grid(axis='y', linestyle='--', alpha=0.7)
                 colors_to_use = [ai_color_map.get(label, 'grey') for label in labels_for_current_metric]
                 for patch, color in zip(box_plot['boxes'], colors_to_use): patch.set_facecolor(color); patch.set_alpha(0.6)
                 plt.tight_layout(); clean_metric_name = metric_key_boxplot.replace('_ms', ''); plot_filename = f"{plot_counter_boxplot}_boxplot_{clean_metric_name}_{timestamp}.png"; path = os.path.join(output_dir, plot_filename); plt.savefig(path); plt.close()
             except Exception as e: print(f"  Erro ao gerar Box Plot para '{metric_key_boxplot}': {e}"); plt.close()
         print("Geração de Box Plots concluída.")

         if num_ais >= 2:
             print(f"Gerando gráfico Radar para {num_ais} IAs (3 métricas base)...")
             try:
                 numeric_cols_radar = ['api_execution_time', 'lines_of_code', 'completeness']
                 means_data = {}
                 for ai in ai_list: df_ai_radar = df[df["ai"] == ai]; means = {}; 
                 for col in numeric_cols_radar: means[col] = df_ai_radar[col].mean() if col in df_ai_radar.columns else 0; means_data[ai] = means
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
                     values = [time_norm.get(ai_name, 0), lines_norm.get(ai_name, 0), comp_norm.get(ai_name, 0)]; plot_values = values + values[:1]
                     color = ai_color_map.get(ai_name, 'grey'); ax.plot(angles, plot_values, 'o-', linewidth=2, label=ai_name, color=color); ax.fill(angles, plot_values, color, alpha=0.2)
                 ax.set_xticks(angles[:-1]); ax.set_xticklabels(categories); ax.set_yticks(np.arange(0, 101, 20)); ax.set_ylim(0, 100)
                 plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15)); plt.title('Comparação Geral (3 Métricas Normalizadas)', size=14, y=1.18); plt.tight_layout()
                 plot_counter_radar = plot_counter_boxplot + 1; path = os.path.join(output_dir, f"{plot_counter_radar}_radar_chart_{timestamp}.png"); plt.savefig(path); plt.close()
             except Exception as e: print(f"Erro ao gerar gráfico Radar: {e}"); plt.close()
         else: print("Gráfico Radar pulado (requer >= 2 IAs).")


# --- Ponto de Entrada Principal ---
if __name__ == "__main__":
    print("===========================================================")
    print(" AI Programming Benchmark: Claude vs ChatGPT vs Gemini")
    print("               (Incluindo Análise Flake8 e Execução)")
    print("===========================================================")
    if not os.path.exists(".env"): print("AVISO: Arquivo .env não encontrado. Chaves API devem ser variáveis de ambiente.")

    benchmark = AIBenchmark()
    if benchmark.claude_api_key or benchmark.openai_api_key or benchmark.gemini_api_key:
        benchmark.run_benchmark()
        print("\nBenchmark concluído.")
        print(f"Relatórios salvos em: '{benchmark.output_dir}'")
    else:
        print("\nERRO FATAL: Nenhuma chave de API configurada (via .env ou variáveis de ambiente). Benchmark não pode ser executado.")

    print("===========================================================")
