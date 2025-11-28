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
import traceback # <- Adicionado
import sys
import multiprocessing # <- Já existia, mas confirmar
import queue         # <- Adicionado
from collections import OrderedDict # <- Já existia (importado globalmente)

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
        self.gemini_api_key = os.getenv('GEMINI_API_KEY') # Carrega a chave Gemini

        self.results = []
        self.detailed_test_results = []
        self.output_dir = "benchmark_reports"
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Diretório de saída: '{self.output_dir}/'")

        # Configura a API Gemini aqui, se a chave existir
        self._configure_gemini_api()

        self.claude_model = "claude-3-opus-20240229"
        self.openai_model = "gpt-4-turbo"
        # Usa o modelo configurado ou None se a configuração falhou
        self.gemini_model = "gemini-1.5-pro-latest" if self.gemini_api_key else None
        # URL do executor externo (usado para desafios que não são LRU Cache)
        self.executor_url = "https://executor-service-1029404216382.europe-southwest1.run.app/execute"

        # --- LISTA DE DESAFIOS COM CASOS DE TESTE ---
        self.code_challenges = [
             { "name": "Fibonacci Sequence", "description": "Generate the first n numbers in the Fibonacci sequence, starting with 0.", "difficulty": "Easy", "function_signature": "fibonacci(n: int) -> list[int]", "test_cases": [ {"input_args": [10], "expected_output": [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]}, {"input_args": [1], "expected_output": [0]}, {"input_args": [0], "expected_output": []}, {"input_args": [2], "expected_output": [0, 1]} ] },
             { "name": "Binary Search", "description": "Find the index of a target value in a sorted list of integers. Return -1 if the target is not found.", "difficulty": "Medium", "function_signature": "binary_search(arr: list[int], target: int) -> int", "test_cases": [ {"input_args": [[2, 5, 7, 8, 11, 12], 13], "expected_output": -1}, {"input_args": [[2, 5, 7, 8, 11, 12], 8], "expected_output": 3}, {"input_args": [[1, 3, 5], 1], "expected_output": 0}, {"input_args": [[1, 3, 5], 5], "expected_output": 2}, {"input_args": [[], 5], "expected_output": -1} ] },
             { "name": "Merge Sort", "description": "Implement the merge sort algorithm to sort a list of integers in ascending order. Return a new sorted list.", "difficulty": "Medium", "function_signature": "merge_sort(arr: list[int]) -> list[int]", "test_cases": [ {"input_args": [[3, 1, 4, 1, 5, 9, 2, 6]], "expected_output": [1, 1, 2, 3, 4, 5, 6, 9]}, {"input_args": [[5, 4, 3, 2, 1]], "expected_output": [1, 2, 3, 4, 5]}, {"input_args": [[1]], "expected_output": [1]}, {"input_args": [[]], "expected_output": []} ] },
             { "name": "Count Islands in Grid", "description": "Count the number of islands in a 2D grid (list of lists) where 1 is land and 0 is water. An island is formed by connecting adjacent lands horizontally or vertically.", "difficulty": "Hard", "function_signature": "count_islands(grid: list[list[int]]) -> int", "test_cases": [ {"input_args": [[ [1, 1, 1, 1, 0], [1, 1, 0, 1, 0], [1, 1, 0, 0, 0], [0, 0, 0, 0, 0] ]], "expected_output": 1}, {"input_args": [[ [1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 1] ]], "expected_output": 3}, {"input_args": [[]], "expected_output": 0}, {"input_args": [[ [0, 0, 0], [0, 0, 0] ]], "expected_output": 0} ] },
             { "name": "Palindrome Check", "description": "Check if a given string is a palindrome, ignoring case and non-alphanumeric characters. Return True if it is, False otherwise.", "difficulty": "Easy", "function_signature": "is_palindrome(s: str) -> bool", "test_cases": [ {"input_args": ["A man, a plan, a canal: Panama"], "expected_output": True}, {"input_args": ["race a car"], "expected_output": False}, {"input_args": [" "], "expected_output": True}, {"input_args": [""], "expected_output": True}, {"input_args": ["Was it a car or a cat I saw?"], "expected_output": True}, {"input_args": ["tab a cat"], "expected_output": False} ] },
             { "name": "Valid Parentheses", "description": "Given a string containing just '(', ')', '{', '}', '[' and ']', determine if the input string is valid (brackets must close in the correct order).", "difficulty": "Medium", "function_signature": "is_valid_parentheses(s: str) -> bool", "test_cases": [ {"input_args": ["()"], "expected_output": True}, {"input_args": ["()[]{}"], "expected_output": True}, {"input_args": ["(]"], "expected_output": False}, {"input_args": ["([)]"], "expected_output": False}, {"input_args": ["{[]}"], "expected_output": True}, {"input_args": [""], "expected_output": True}, {"input_args": ["["], "expected_output": False}, {"input_args": ["]"], "expected_output": False} ] },
             # LRU Cache agora usa execução local, a definição do teste permanece a mesma
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
                self.gemini_api_key = None # Garante que a chave seja None se a configuração falhar
        else:
            print("AVISO: Chave API Gemini não encontrada. Gemini AI será pulada.")

    def _get_environment_info(self):
        """Coleta informações sobre o ambiente de execução."""
        # (Sem alterações nesta função)
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
            "Local LRU Cache Execution": True, # Indica que LRU é executado localmente
            "Libraries": {}
        }
        libs_to_check = ['pandas', 'numpy', 'matplotlib', 'requests',
                         'python-dotenv', 'google-generativeai', 'google-api-core']
        try:
            for lib in libs_to_check:
                try: info["Libraries"][lib] = importlib_metadata.version(lib)
                except importlib_metadata.PackageNotFoundError: info["Libraries"][lib] = "Not Found"
        except NameError:
            for lib in libs_to_check:
                try: info["Libraries"][lib] = pkg_resources.get_distribution(lib).version
                except pkg_resources.DistributionNotFound: info["Libraries"][lib] = "Not Found"
                except Exception: info["Libraries"][lib] = "Error getting version"
        return info

    def _check_flake8(self):
        """Verifica se o comando flake8 está disponível no sistema."""
        # (Sem alterações nesta função - já usava sys.executable)
        try:
            subprocess.run(
                [sys.executable, '-m', 'flake8', '--version'],
                check=True, capture_output=True, timeout=5
            )
            print("Verificação Flake8: OK")
            self.flake8_available = True
        except Exception:
            print("AVISO: Comando 'flake8' não encontrado/executável via "
                  f"'{sys.executable} -m flake8'. Análise de qualidade será pulada.")
            self.flake8_available = False

    def query_claude(self, prompt: str) -> dict:
        """Envia um prompt para a API do Claude e retorna a resposta."""
        # (Sem alterações nesta função)
        start_time = time.time()
        content = ""
        if not self.claude_api_key:
            print("Pulando Claude: Chave API não fornecida.")
            execution_time = time.time() - start_time
            return {"content": "", "execution_time": execution_time}
        headers = {"x-api-key": self.claude_api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        data = {"model": self.claude_model, "max_tokens": 4000, "messages": [{"role": "user", "content": prompt}]}
        try:
            response = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=data, timeout=120)
            response.raise_for_status()
            result = response.json()
            if ("content" in result and isinstance(result["content"], list) and len(result["content"]) > 0):
                first_content = result["content"][0]; content = first_content.get("text", "") if isinstance(first_content, dict) else ""
            else: print(f"Resposta Claude: Estrutura 'content' inesperada. Resp: {result}")
            execution_time = time.time() - start_time
            return {"content": content or "", "execution_time": execution_time}
        except requests.exceptions.RequestException as e:
            error_message = f"Erro consulta Claude (RequestException): {e}"
            if e.response is not None: error_message += f" | Status: {e.response.status_code}" ; try: error_message += f" | Body: {e.response.json()}" ; except json.JSONDecodeError: error_message += f" | Body: {e.response.text}"
            print(error_message); execution_time = time.time() - start_time; return {"content": "", "execution_time": execution_time}
        except Exception as e: print(f"Erro inesperado consulta Claude: {type(e).__name__} - {e}"); execution_time = time.time() - start_time; return {"content": "", "execution_time": execution_time}

    def query_chatgpt(self, prompt: str) -> dict:
        """Envia um prompt para a API do OpenAI (ChatGPT) e retorna a resposta."""
        # (Sem alterações nesta função)
        start_time = time.time(); content = ""
        if not self.openai_api_key: print("Pulando ChatGPT: Chave API não fornecida."); execution_time = time.time() - start_time; return {"content": "", "execution_time": execution_time}
        headers = {"Authorization": f"Bearer {self.openai_api_key}", "Content-Type": "application/json"}
        data = {"model": self.openai_model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 4000}
        try:
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data, timeout=120); response.raise_for_status()
            result = response.json()
            if ("choices" in result and isinstance(result["choices"], list) and len(result["choices"]) > 0):
                first_choice = result["choices"][0]
                if (isinstance(first_choice, dict) and "message" in first_choice and isinstance(first_choice["message"], dict)): content = first_choice["message"].get("content", "")
                else: print(f"Resposta ChatGPT: Estrutura 'message' inesperada. Resp: {result}")
            else: print(f"Resposta ChatGPT: Estrutura 'choices' inesperada. Resp: {result}")
            execution_time = time.time() - start_time; return {"content": content or "", "execution_time": execution_time}
        except requests.exceptions.RequestException as e:
            error_message = f"Erro consulta ChatGPT (RequestException): {e}"
            if e.response is not None: error_message += f" | Status: {e.response.status_code}"; try: error_details = e.response.json(); error_msg = error_details.get('error', {}).get('message', ''); error_message += f" | API Error: {error_msg}" if error_msg else f" | Body: {error_details}"; except json.JSONDecodeError: error_message += f" | Body: {e.response.text}"
            print(error_message); execution_time = time.time() - start_time; return {"content": "", "execution_time": execution_time}
        except Exception as e: print(f"Erro inesperado consulta ChatGPT: {type(e).__name__} - {e}"); execution_time = time.time() - start_time; return {"content": "", "execution_time": execution_time}

    def query_gemini(self, prompt: str) -> dict:
        """Envia um prompt para a API do Gemini e retorna a resposta."""
        # (Sem alterações nesta função)
        start_time = time.time(); content = ""
        if not self.gemini_api_key or not self.gemini_model: print("Pulando Gemini: Chave API não configurada ou modelo inválido."); execution_time = time.time() - start_time; return {"content": "", "execution_time": execution_time}
        try:
            model = genai.GenerativeModel(self.gemini_model)
            safety_settings=[{"category": c, "threshold": "BLOCK_NONE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
            response = model.generate_content(prompt, safety_settings=safety_settings); execution_time = time.time() - start_time
            try: content = response.text
            except ValueError: print(f"Resposta Gemini não contém texto (possivelmente bloqueada)."); try: print(f"  Prompt Feedback: {response.prompt_feedback}"); if hasattr(response, 'candidates') and response.candidates: print(f"  Finish Reason: {response.candidates[0].finish_reason}"); except Exception as feedback_error: print(f"  Erro ao acessar feedback Gemini: {feedback_error}")
            return {"content": content or "", "execution_time": execution_time}
        except google.api_core.exceptions.ResourceExhausted as e: print(f"Erro consulta Gemini (ResourceExhausted - Rate Limit): {e}\n                >> Atingido o limite de requisições da API Gemini. Tente novamente mais tarde."); execution_time = time.time() - start_time; return {"content": "", "execution_time": execution_time}
        except Exception as e: print(f"Erro inesperado consulta Gemini: {type(e).__name__} - {e}"); execution_time = time.time() - start_time; return {"content": "", "execution_time": execution_time}

    def extract_code(self, content: str) -> str:
        """Extrai o primeiro bloco de código Python de uma string de texto."""
        # (Sem alterações nesta função)
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
        if len(content.splitlines()) < 3 and ("def " not in content and "class " not in content): print("AVISO: extract_code retornando vazio (conteúdo curto sem def/class)."); return ""
        else: print("AVISO: extract_code usando fallback final (retornando conteúdo original)."); return content.strip()

    def count_code_lines(self, code: str) -> int:
        """Conta as linhas de código 'reais', ignorando vazias e comentários."""
        # (Sem alterações nesta função)
        if not code: return 0
        return len([line for line in code.splitlines() if line.strip() and not line.strip().startswith('#')])

    def measure_response_completeness(self, code: str, description: str) -> int:
        """Estima a 'completude' da resposta da IA (score 0-100)."""
        # (Sem alterações nesta função)
        completeness_score = 0; lines_count = self.count_code_lines(code)
        if lines_count > 0:
            completeness_score += 50;
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

    def analyze_code_quality_flake8(self, code_string: str) -> int:
        """Analisa código Python com Flake8 e retorna o número de problemas (-1 se erro)."""
        # (Sem alterações nesta função)
        if not self.flake8_available or not code_string or self.count_code_lines(code_string) == 0: return -1
        temp_filepath = None
        try:
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.py', delete=False, encoding='utf-8') as temp_file: temp_file.write(code_string); temp_filepath = temp_file.name
            python_exe = sys.executable
            result = subprocess.run([python_exe, '-m', 'flake8', '--count', temp_filepath], capture_output=True, text=True, timeout=15, check=False)
            if result.returncode in [0, 1] and result.stdout: try: count = int(result.stdout.strip().splitlines()[-1]); return count; except (IndexError, ValueError): return len(result.stdout.strip().splitlines()) # Fallback
            elif result.stderr: print(f"Erro na execução do flake8: {result.stderr}"); return -1
            else: return 0
        except subprocess.TimeoutExpired: print("Erro: Análise com Flake8 excedeu o tempo limite (15s)."); return -1
        except Exception as e: print(f"Erro inesperado ao executar Flake8: {e}"); return -1
        finally:
            if temp_filepath and os.path.exists(temp_filepath):
                try: os.remove(temp_filepath)
                except Exception as e_rem: print(f"Aviso: Não foi possível remover o arquivo temporário Flake8: {e_rem}")

    # --- Métodos para Execução Local Segura do LRU Cache ---

    def _lru_worker(self, code_string: str, test_sequence: list, result_queue: multiprocessing.Queue):
        """
        Função alvo para o processo filho. Executa o código LRU com restrições.
        AVISO: A restrição de builtins é uma mitigação, não garantia de segurança.
        NÃO CHAME DIRETAMENTE. Use _execute_lru_locally.
        """
        # Importa OrderedDict dentro do worker se não for passado por globals
        # from collections import OrderedDict # -> Já importado globalmente e passado
        results = []
        error_message = None
        lru_instance = None
        final_success = True # Assume sucesso até que um passo falhe

        try:
            # 1. Preparar ambiente restrito (MELHOR ESFORÇO - NÃO É INFALÍVEL)
            #    Permite apenas built-ins e módulos considerados seguros/necessários.
            safe_builtins = {
                'int': int, 'str': str, 'list': list, 'dict': dict, 'tuple': tuple,
                'len': len, 'range': range, 'print': print, # print para debug, pode ser removido
                'True': True, 'False': False, 'None': None, 'object': object,
                'Exception': Exception, 'RuntimeError': RuntimeError, 'ValueError': ValueError,
                'NameError': NameError, 'AttributeError': AttributeError, 'TypeError': TypeError,
                'StopIteration': StopIteration, # Necessário por algumas operações internas
                # Adicionar outros built-ins seguros essenciais, se necessário
            }
            safe_globals = {
                "__builtins__": safe_builtins,
                 # Passa explicitamente OrderedDict em vez de importar 'collections'
                 "OrderedDict": OrderedDict
            }

            # 2. Executa o código da classe para defini-la dentro do ambiente seguro
            #    Precisamos adicionar a definição da classe ao safe_globals APÓS exec
            local_namespace = {} # Executa no namespace local para capturar a classe
            exec(code_string, safe_globals, local_namespace)

            # 3. Encontrar o nome da classe (assumindo que é LRUCache ou similar)
            class_name = None
            for name, obj in local_namespace.items():
                 # Verifica se é uma classe definida no código executado
                 if isinstance(obj, type) and name != 'OrderedDict':
                      class_name = name
                      break
            if not class_name:
                 # Tenta uma heurística se a execução não colocou no local_namespace
                 match = re.search(r"^\s*class\s+(\w+)", code_string, re.MULTILINE)
                 if match:
                      class_name = match.group(1)
                 else:
                      raise NameError("Não foi possível encontrar o nome da classe LRUCache no código fornecido.")

            # Tenta obter a classe do namespace onde foi definida
            LruClass = local_namespace.get(class_name)
            if not LruClass:
                 raise NameError(f"Classe '{class_name}' não encontrada no namespace após exec.")


            # 4. Executar sequência de testes
            op_total_time = 0.0
            for op_data in test_sequence:
                op_start_time = time.perf_counter()
                operation = op_data["op"]
                args = op_data.get("args", [])
                expected = op_data.get("expected") # Pode ser None para 'put'

                step_result = {"operation": operation, "args": args, "expected": expected,
                               "output": None, "success": False, "error": None, "exec_time_ms": -1.0}

                try:
                    if operation == "__init__":
                        lru_instance = LruClass(*args)
                        step_result["success"] = True
                        step_result["output"] = "Instance created" # Indica sucesso
                    elif lru_instance is None:
                        raise RuntimeError("LRUCache não foi inicializada antes de outras operações.")
                    elif operation == "put":
                        # Put não retorna valor, apenas executa
                        getattr(lru_instance, operation)(*args)
                        step_result["success"] = True # Assume sucesso se não houver exceção
                        step_result["output"] = "put executed"
                    elif operation == "get":
                        output = getattr(lru_instance, operation)(*args)
                        step_result["output"] = output
                        # Compara output com expected (cuidado com tipos)
                        if str(output) == str(expected): # Comparação simples como string
                            step_result["success"] = True
                        else:
                            step_result["error"] = f"Output '{output}' ({type(output)}) != Expected '{expected}' ({type(expected)})"
                            final_success = False # Falha no teste geral
                    else:
                        raise ValueError(f"Operação LRU desconhecida: {operation}")

                except Exception as e_step:
                    step_error_msg = f"{type(e_step).__name__}: {str(e_step).splitlines()[0]}" # Primeira linha do erro
                    step_result["error"] = step_error_msg
                    print(f"      Erro no passo LRU ({operation}): {step_error_msg}") # Log no worker
                    # traceback.print_exc() # Descomentar para ver traceback completo no log do worker
                    final_success = False # Falha no teste geral

                op_end_time = time.perf_counter()
                step_result["exec_time_ms"] = (op_end_time - op_start_time) * 1000
                op_total_time += (op_end_time - op_start_time)
                results.append(step_result)

                # Opcional: Parar no primeiro erro encontrado
                # if not step_result["success"]:
                #     break

            print(f"    Sequência LRU concluída no worker. Tempo total passos: {op_total_time*1000:.2f} ms")

        except Exception as e_main:
            # Captura erros na configuração do ambiente ou fora da sequência
            error_message = f"Erro principal no worker LRU: {type(e_main).__name__} - {str(e_main).splitlines()[0]}"
            print(f"    ERRO GERAL NO WORKER LRU: {error_message}")
            traceback.print_exc() # Imprime traceback completo no log do worker
            final_success = False

        # Coloca o resultado final na fila para o processo pai
        result_queue.put({
            "success": final_success, # Indica se TODOS os passos relevantes passaram
            "results": results,     # Lista com o resultado de cada passo da sequência
            "error": error_message  # Erro geral ocorrido fora da sequência de passos
        })


    def _execute_lru_locally(self, code_string: str, test_sequence: list) -> dict:
        """
        Tenta executar a sequência de teste do LRU Cache localmente em um processo
        separado com restrições e timeout. Gere o processo e recolhe o resultado.

        Args:
            code_string: O código da classe LRUCache gerado pela IA.
            test_sequence: A lista de operações de teste.

        Returns:
            Dicionário com resultados: {'success': bool, 'results': list, 'error': str|None}
        """
        # Fila para obter o resultado do processo filho
        # Usar contexto para garantir que recursos de MP sejam limpos
        ctx = multiprocessing.get_context('fork') # ou 'spawn' se 'fork' causar problemas
        result_queue = ctx.Queue()
        timeout_seconds = 10 # Timeout para a execução completa da sequência

        process = ctx.Process(
            target=self._lru_worker, # Método da instância como alvo
            args=(code_string, test_sequence, result_queue)
        )

        return_value = {"success": False, "results": [], "error": "Unknown execution error"}
        process_terminated = False

        try:
            start_time = time.time()
            process.start()
            process.join(timeout=timeout_seconds) # Espera pelo processo com timeout

            if process.is_alive():
                print(f"  AVISO: Processo de execução local do LRU excedeu timeout ({timeout_seconds}s). Terminando.")
                process.terminate() # Tenta terminar educadamente
                process.join(1) # Espera um pouco pela terminação
                if process.is_alive():
                    print("  AVISO: Forçando terminação (kill) do processo LRU.")
                    process.kill() # Força terminação (último recurso)
                    process.join()
                return_value = {"success": False, "results": [], "error": f"Execution timed out after {timeout_seconds}s"}
                process_terminated = True

            elapsed_time = time.time() - start_time

            if not process_terminated:
                exit_code = process.exitcode
                if exit_code != 0:
                    # Processo terminou com erro antes de colocar na fila?
                     print(f"  ERRO: Processo LRU terminou com exitcode não-zero: {exit_code}")
                     return_value = {"success": False, "results": [], "error": f"Worker process exited with code {exit_code}"}
                else:
                    # Processo terminou com sucesso (exitcode 0), tenta obter resultado
                    try:
                        result = result_queue.get(timeout=1) # Pequeno timeout para obter da fila
                        print(f"  Execução local LRU concluída em {elapsed_time:.2f}s. Sucesso worker: {result.get('success')}")
                        return_value = result
                    except queue.Empty:
                        # Processo terminou OK mas não colocou nada na fila (erro interno no worker?)
                        print("  ERRO: Processo LRU terminou (exitcode 0) mas não retornou resultado na fila.")
                        return_value = {"success": False, "results": [], "error": "Worker process finished but queue was empty"}
                    except Exception as e_queue:
                        print(f"  ERRO: Falha ao obter resultado da fila do processo LRU: {e_queue}")
                        return_value = {"success": False, "results": [], "error": f"Failed to retrieve result from worker queue: {e_queue}"}

        except Exception as e_proc:
             # Erro ao gerir o processo em si
             print(f"  ERRO: Falha ao gerir o processo de execução local LRU: {e_proc}")
             return_value = {"success": False, "results": [], "error": f"Failed to manage worker process: {e_proc}"}
             # Garante que o processo morra se ainda estiver vivo
             if process and process.is_alive():
                  try: process.kill(); process.join()
                  except Exception: pass # Ignora erros ao tentar matar
        finally:
            # Limpa a fila e o processo
            try:
                 result_queue.close()
                 result_queue.join_thread()
                 if process:
                     process.close() # Fecha o objeto Process
            except Exception as e_clean:
                 print(f"  Aviso: Erro ao limpar recursos do processo LRU: {e_clean}")

        return return_value

    # --- Fim dos métodos de execução local ---

    def evaluate_solution(self, ai_name: str, challenge: dict, response: dict) -> dict:
        """
        Avalia a solução de uma IA para um desafio específico.
        Usa execução local para LRU Cache e serviço externo para os outros.
        """
        content = response.get("content", "") if isinstance(response, dict) else ""
        api_execution_time = response.get("execution_time", 0) if isinstance(response, dict) else 0

        code = self.extract_code(content)
        lines_of_code = self.count_code_lines(code)
        completeness = self.measure_response_completeness(code, challenge["description"])
        flake8_issues = self.analyze_code_quality_flake8(code)

        execution_results_list = [] # Lista de resultados brutos (do executor ou dos passos do LRU)
        processed_test_details = [] # Lista formatada para o CSV detalhado
        num_execution_errors = 0    # Número de *test cases* que falharam/erraram
        first_execution_error = ""  # Primeira mensagem de erro encontrada

        actual_function_name = self._detect_function_name(code, challenge['name'])

        if code and (actual_function_name or challenge["name"] == "LRU Cache"): # Precisa do código para LRU
            # --- Lógica de Execução ---
            if challenge["name"] == "LRU Cache":
                print(f"  Tentando executar LRU Cache de {ai_name} localmente...")
                test_cases = challenge.get("test_cases", [])
                # Garante que há casos de teste e que o primeiro tem a sequência
                if test_cases and isinstance(test_cases[0], dict) and "sequence" in test_cases[0]:
                    lru_exec_result = self._execute_lru_locally(code, test_cases[0]["sequence"])

                    # Guarda os resultados detalhados dos passos
                    execution_results_list = lru_exec_result.get("results", [])

                    # Determina o resultado geral para o sumário
                    overall_success = lru_exec_result.get("success", False)
                    num_execution_errors = 0 if overall_success else 1
                    first_execution_error = lru_exec_result.get("error", "") # Erro geral do worker
                    if not first_execution_error and not overall_success:
                        # Se não houve erro geral, procura o erro no primeiro passo que falhou
                        for step in execution_results_list:
                            if step.get("error"):
                                first_execution_error = f"Step {step['operation']}{step['args']}: {step['error']}"
                                break
                    if not first_execution_error and not overall_success:
                         first_execution_error = "LRU Cache execution failed (unknown step error)"

                    # Cria uma única entrada para o detailed_test_results representando a sequência toda
                    avg_step_time = -1.0
                    valid_step_times = [s['exec_time_ms'] for s in execution_results_list if s.get('exec_time_ms', -1) >= 0]
                    if valid_step_times:
                         avg_step_time = np.mean(valid_step_times)

                    processed_test_details.append({
                        "ai": ai_name, "challenge": challenge["name"], "difficulty": challenge["difficulty"],
                        "test_case_index": 0, # Índice único para a sequência
                        "input_args_str": json.dumps(test_cases[0]["sequence"]), # Input é a sequência
                        "expected_output_str": "See sequence steps", # Output esperado está nos passos
                        "actual_output_str": "See sequence steps results", # Output real está nos passos
                        "status": "Pass" if overall_success else "Fail" if execution_results_list else "Error",
                        "execution_time_ms": avg_step_time, # Média do tempo dos passos válidos
                        "error_message": first_execution_error[:250], # Limita tamanho da msg de erro
                        "stdout": None # stdout não é capturado na execução local aqui
                    })
                    print(f"  Resultado execução local LRU: {'Sucesso' if overall_success else 'Falha/Erro'}")

                else:
                    print("  AVISO: Casos de teste para LRU Cache não encontrados ou formato inválido.")
                    num_execution_errors = 1
                    first_execution_error = "LRU test case sequence not found."
                    # Adiciona entrada de erro ao detailed_test_results
                    processed_test_details.append({
                        "ai": ai_name, "challenge": challenge["name"], "difficulty": challenge["difficulty"],
                        "test_case_index": 0, "input_args_str": "N/A", "expected_output_str": "N/A",
                        "actual_output_str": None, "status": "Error", "execution_time_ms": -1,
                        "error_message": first_execution_error, "stdout": None
                    })

            # --- Fim da Lógica LRU Cache ---
            elif actual_function_name: # Executa outros desafios via serviço externo
                print(f"  Tentando executar código de {ai_name} para '{challenge['name']}' "
                      f"(função: {actual_function_name}) via Executor Service...")
                # Usa a função _run_test_cases que chama _call_executor_service
                execution_results_list, processed_details_external, num_errors_external, first_error_external = \
                    self._run_test_cases(ai_name, challenge, code, actual_function_name)
                num_execution_errors = num_errors_external
                first_execution_error = first_error_external
                processed_test_details.extend(processed_details_external) # Adiciona resultados formatados

            # --- Fim da Lógica de outros desafios ---
        else: # Caso não haja código ou nome de função detectado (exceto LRU)
            print(f"  AVISO: Nenhum código válido ou nome de função detectado para "
                  f"'{challenge['name']}' por {ai_name}. Pulando execução.")
            num_tests = len(challenge.get("test_cases", [])) if challenge["name"] != "LRU Cache" else 1
            for i in range(num_tests):
                 processed_test_details.append({
                     "ai": ai_name, "challenge": challenge["name"], "difficulty": challenge["difficulty"],
                     "test_case_index": i, "input_args_str": None, "expected_output_str": None,
                     "actual_output_str": None, "status": "No Code", "execution_time_ms": -1,
                     "error_message": "No code extracted or function name detected", "stdout": None
                 })
            num_execution_errors = 1 # Considera como erro para fins de sumário
            first_execution_error = "No code extracted or function name detected"

        # Adiciona os detalhes formatados à lista global
        self.detailed_test_results.extend(processed_test_details)

        # Retorna o dicionário de resultados agregados para este AI/Desafio
        return {
            "ai": ai_name,
            "challenge": challenge["name"],
            "difficulty": challenge["difficulty"],
            "api_execution_time": api_execution_time,
            "lines_of_code": lines_of_code,
            "completeness": completeness,
            "flake8_issues": flake8_issues,
            "num_execution_errors": num_execution_errors, # Baseado no resultado da execução (local ou externa)
            "first_execution_error": first_execution_error, # Primeira mensagem de erro encontrada
            "code": code, # Código extraído
            "full_response": content, # Resposta completa da API
            # Guarda os resultados brutos (lista de dicts do executor ou lista de dicts dos passos do LRU)
            "execution_results_raw": execution_results_list
        }


    def _detect_function_name(self, code: str, challenge_name: str) -> str | None:
        """Detecta o nome da função ou classe principal no código."""
        # (Sem alterações nesta função)
        if not code: return None
        match = re.search(r"^\s*(?:def|class)\s+(\w+)", code, re.MULTILINE)
        if match: name = match.group(1); print(f"  Nome da função/classe detectado no código: '{name}'"); return name
        else: print(f"  AVISO: Não foi possível detectar nome da função/classe no código extraído para '{challenge_name}'."); return None

    def _run_test_cases(self, ai_name: str, challenge: dict, code: str,
                        actual_function_name: str) -> tuple[list, list, int, str]:
        """Executa casos de teste para funções usando o Executor Service externo."""
        # (Sem alterações significativas nesta função - apenas formatação/clareza)
        execution_results_list = []
        processed_test_details = []
        num_execution_errors = 0
        first_execution_error = ""
        test_cases = challenge.get("test_cases", [])

        if not test_cases: print(f"  Aviso: Nenhum caso de teste definido para '{challenge['name']}'."); return [], [], 0, ""

        for i, test_case in enumerate(test_cases):
            input_args = test_case.get("input_args")
            expected_output = test_case.get("expected_output")
            if input_args is None: print(f"  Aviso: Caso de teste {i+1} para '{challenge['name']}' não tem 'input_args'. Pulando."); continue

            exec_result = self._call_executor_service(code, actual_function_name, input_args)
            execution_results_list.append(exec_result) # Guarda resultado bruto

            test_detail = {
                "ai": ai_name, "challenge": challenge["name"], "difficulty": challenge["difficulty"],
                "test_case_index": i, "input_args_str": json.dumps(input_args),
                "expected_output_str": json.dumps(expected_output), "actual_output_str": None,
                "status": "Unknown", "execution_time_ms": exec_result.get("execution_time_ms", -1),
                "error_message": None, "stdout": exec_result.get("stdout")
            }

            if exec_result.get("success") is True:
                actual_output = exec_result.get("output")
                test_detail["actual_output_str"] = json.dumps(actual_output)
                try:
                    are_equal = np.array_equal(np.array(actual_output, dtype=object), np.array(expected_output, dtype=object))
                    test_detail["status"] = "Pass" if are_equal else "Fail"
                    if not are_equal and not first_execution_error: # Regista o primeiro erro de falha
                         first_execution_error = f"Test Case {i} Failed: Output != Expected"
                except Exception as compare_err:
                    test_detail["status"] = "Fail (Comparison Error)"
                    test_detail["error_message"] = f"Error comparing outputs: {compare_err}"
                    if not first_execution_error: first_execution_error = test_detail["error_message"]
            elif exec_result.get("success") is False:
                test_detail["status"] = "Error"
                test_detail["error_message"] = exec_result.get("error", "Unknown execution error")
                num_execution_errors += 1
                if not first_execution_error and test_detail["error_message"]: first_execution_error = str(test_detail["error_message"])[:250] # Limita tamanho
            else: # Caso inesperado
                 test_detail["status"] = "Unknown"
                 test_detail["error_message"] = exec_result.get("error", "Executor did not return success status")
                 num_execution_errors += 1
                 if not first_execution_error and test_detail["error_message"]: first_execution_error = str(test_detail["error_message"])[:250] # Limita tamanho

            processed_test_details.append(test_detail)

        # Ajusta num_execution_errors para contar falhas também como "erros" do ponto de vista do sumário
        num_failures_or_errors = len([d for d in processed_test_details if d['status'] not in ['Pass', 'No Code']])
        # Se não houve erro explícito, mas houve falha, atualiza a mensagem
        if num_failures_or_errors > 0 and not first_execution_error:
             first_execution_error = "One or more test cases failed comparison."

        return execution_results_list, processed_test_details, num_failures_or_errors, first_execution_error

    def _call_executor_service(self, code_string: str, function_name: str,
                               input_args: list) -> dict:
        """Chama o serviço externo para executar uma função."""
        # (Sem alterações nesta função)
        imports_to_add = ("from typing import List, Dict, Tuple, Set, Optional, Any\n""import re\n""from collections import OrderedDict, deque\n\n")
        full_code_to_execute = imports_to_add + code_string
        payload = { "code_string": full_code_to_execute, "function_name": function_name, "test_input_args": input_args }
        headers = {"Content-Type": "application/json"}
        default_error_result = { "success": False, "output": None, "stdout": None, "error": "Executor service call failed", "execution_time_ms": -1 }
        try:
            response = requests.post(self.executor_url, headers=headers, json=payload, timeout=60)
            if response.status_code == 200:
                try: return response.json()
                except json.JSONDecodeError as json_err: print(f"  Erro: Falha ao decodificar JSON da resposta do executor. Status: {response.status_code}, Erro: {json_err}"); default_error_result["error"] = f"Executor response JSON decode error: {json_err}"; return default_error_result
            else: error_text = response.text[:200]; print(f"  Erro: Chamada ao executor falhou. Status: {response.status_code}, Resposta: {error_text}..."); default_error_result["error"] = f"Executor service returned status {response.status_code}: {error_text}"; return default_error_result
        except requests.exceptions.Timeout: print(f"  Erro: Timeout ao chamar o executor service ({self.executor_url})."); default_error_result["error"] = "Executor service call timed out"; return default_error_result
        except requests.exceptions.RequestException as req_err: print(f"  Erro: Problema de conexão/rede ao chamar o executor service: {req_err}"); default_error_result["error"] = f"Executor service connection error: {req_err}"; return default_error_result
        except Exception as e: print(f"  Erro inesperado ao chamar o executor service: {type(e).__name__} - {e}"); default_error_result["error"] = f"Unexpected error calling executor: {type(e).__name__}"; return default_error_result

    def run_benchmark(self):
        """Executa o ciclo completo do benchmark para todas as IAs e desafios."""
        # (Sem alterações significativas, apenas logs)
        can_run_claude = bool(self.claude_api_key)
        can_run_openai = bool(self.openai_api_key)
        can_run_gemini = bool(self.gemini_api_key and self.gemini_model)
        enabled_ais = [name for name, can_run in [("Claude", can_run_claude), ("ChatGPT", can_run_openai), ("Gemini", can_run_gemini)] if can_run]

        if not enabled_ais: print("ERRO: Nenhuma chave de API válida configurada. Benchmark não pode ser executado."); return

        print(f"Iniciando Benchmark para: {', '.join(enabled_ais)}")
        print(f"  Claude Model : {self.claude_model if can_run_claude else 'N/A (No Key)'}")
        print(f"  ChatGPT Model: {self.openai_model if can_run_openai else 'N/A (No Key)'}")
        print(f"  Gemini Model : {self.gemini_model if can_run_gemini else 'N/A (No Key/Config Error)'}")
        print(f"  LRU Cache Execution: Local Process")
        print("==========================================")

        challenge_details = { # (Mantido igual)
           "Fibonacci Sequence": "Técnica: Requer lógica iterativa ou recursiva simples.\n  Finalidade: Avaliar a capacidade de implementar algoritmos básicos, lidar com sequências e casos base (n=0, n=1).",
           "Binary Search": "Técnica: Busca eficiente em lista ordenada (dividir para conquistar).\n  Finalidade: Avaliar a capacidade de implementar busca binária, lidar com índices, condições de parada e casos onde o elemento não é encontrado.",
           "Merge Sort": "Técnica: Ordenação eficiente usando \"dividir para conquistar\" (recursão e merge).\n  Finalidade: Avaliar a capacidade de implementar algoritmos de ordenação recursivos, especificamente a lógica de divisão e a de intercalação (merge) de sub-listas ordenadas.",
           "Count Islands in Grid": "Técnica: Travessia de grafo (Grid 2D) usando Busca em Profundidade (DFS) ou Busca em Largura (BFS).\n  Finalidade: Avaliar a capacidade de modelar um problema como travessia de grafo, implementar DFS/BFS, lidar com matrizes 2D e evitar recontagem (marcando visitados).",
           "Palindrome Check": "Técnica: Manipulação de strings (limpeza, inversão, comparação).\n  Finalidade: Avaliar a capacidade de processar strings, ignorar caracteres não alfanuméricos e maiúsculas/minúsculas, e verificar simetria.",
           "Valid Parentheses": "Técnica: Uso de estrutura de dados Pilha (Stack) para validação de ordem.\n  Finalidade: Avaliar a capacidade de usar pilhas para rastrear parênteses/colchetes/chaves abertos e validar o fechamento correto e na ordem apropriada.",
           "LRU Cache": "Técnica: Implementação de estrutura de dados customizada (cache) com política de remoção LRU, frequentemente usando dicionários e/ou listas ligadas (ou OrderedDict).\n  Finalidade: Avaliar a capacidade de projetar e implementar estruturas de dados mais complexas, gerenciar capacidade, implementar lógica de get e put com eficiência de tempo (idealmente O(1)), e lidar com a política de despejo LRU."
        }

        for challenge in self.code_challenges:
            signature = challenge.get("function_signature", f"{challenge['name'].lower().replace(' ', '_')}(...)")
            if "class LRUCache" in signature: prompt = (f"Please implement the following Python class based on this description: {challenge['description']}. The class and method signatures should be:\n{signature}\nProvide ONLY the complete Python code for the class definition and necessary imports (like `from collections import OrderedDict`). Do not include any explanations, example usage, markdown formatting (like ```python), or introductory sentences.") # Reforça pedido de import
            else: prompt = ( f"Please write a single, complete Python function that implements the following signature: `{signature}`. The function should solve this problem: {challenge['description']}. Provide ONLY the Python code for the function definition and necessary imports. Do not include any explanations, example usage, markdown formatting (like ```python), or introductory sentences.")

            print(f"\n--- Desafio: {challenge['name']} (Dificuldade: {challenge['difficulty']}) ---");
            description_detail = challenge_details.get(challenge['name'], "Descrição detalhada não encontrada.")
            print(f"  {description_detail}")
            responses = {}
            if can_run_claude: print("Consultando Claude AI..."); responses["Claude"] = self.query_claude(prompt); time.sleep(1)
            if can_run_openai: print("Consultando ChatGPT..."); responses["ChatGPT"] = self.query_chatgpt(prompt); time.sleep(1)
            if can_run_gemini: print("Consultando Gemini AI..."); responses["Gemini"] = self.query_gemini(prompt); print("Aguardando 31s para o limite da API Gemini..."); time.sleep(31)

            challenge_summary_lines = []
            for ai_name in enabled_ais:
                response_data = responses.get(ai_name);
                if response_data is None: response_data = {"content": "", "execution_time": 0}
                # Chama evaluate_solution, que agora lida com LRU localmente
                result = self.evaluate_solution(ai_name, challenge, response_data)
                self.results.append(result)

                # Calcula resumo de testes para o print imediato (lógica ajustada ligeiramente)
                tests_attempted = 0; tests_passed = 0; tests_error = result.get('num_execution_errors', 0)
                related_details = [d for d in self.detailed_test_results if d['ai'] == ai_name and d['challenge'] == challenge['name']]
                # Conta tentativas reais (Pass, Fail, Error - exclui No Code)
                tests_attempted = len([d for d in related_details if d['status'] not in ['No Code']])
                tests_passed = len([d for d in related_details if d['status'] == 'Pass'])
                # tests_error já vem de evaluate_solution (conta falhas/erros)

                flake8_str = (str(result['flake8_issues']) if result['flake8_issues'] != -1 else 'N/A')
                exec_status = f"{tests_passed}/{tests_attempted} passed" if tests_attempted > 0 else "Not Run/No Code"
                summary_line = (f"  {ai_name:<10}: Exec: {exec_status} ({tests_error} errors/fails). "
                                f"Flake8: {flake8_str}. API Time: {result['api_execution_time']:.2f}s.")
                challenge_summary_lines.append(summary_line)

            print(f"--- Resumo Desafio {challenge['name']} ---")
            for line in challenge_summary_lines: print(line)
            print(f"--- Desafio {challenge['name']} concluído ---")

        if not self.results: print("\nNenhum resultado coletado. Encerrando."); return
        self.generate_report()

    # --- Métodos Auxiliares para Geração de Relatórios ---
    # (_prepare_dataframe, _calculate_aggregated_test_metrics, etc. permanecem os mesmos)
    # ... (todas as funções auxiliares de relatório _prepare_dataframe até _display_summary_table) ...
    # NOTE: Nenhuma alteração necessária nestas funções, pois elas operam sobre
    # os dados já processados e agregados por evaluate_solution e detailed_test_results.
    # A forma como as métricas de LRU Cache (num_errors, first_error) são definidas
    # em evaluate_solution agora reflete o resultado da execução local.

    def _prepare_dataframe(self) -> pd.DataFrame | None:
        """Prepara o DataFrame principal com resultados e métricas agregadas."""
        # (Sem alterações nesta função)
        if not self.results: print("Não há resultados para preparar o DataFrame."); return None
        try:
            df = pd.DataFrame(self.results); agg_metrics = self._calculate_aggregated_test_metrics()
            df['avg_exec_success_rate'] = df.apply(lambda row: agg_metrics.get((row['ai'], row['challenge']), {}).get('avg_exec_success_rate', 0.0), axis=1)
            df['avg_exec_time_ms'] = df.apply(lambda row: agg_metrics.get((row['ai'], row['challenge']), {}).get('avg_exec_time_ms', np.nan), axis=1)
            df['total_tests_passed'] = df.apply(lambda row: agg_metrics.get((row['ai'], row['challenge']), {}).get('total_tests_passed', 0), axis=1)
            df['total_tests_attempted'] = df.apply(lambda row: agg_metrics.get((row['ai'], row['challenge']), {}).get('total_tests_attempted', 0), axis=1)
            df = df.drop(['execution_results_raw'], axis=1, errors='ignore')
            self._normalize_dataframe_types(df)
            return df
        except Exception as e: print(f"Erro crítico ao criar/processar DataFrame principal: {e}\nTraceback: {traceback.format_exc()}\nResultados brutos: {self.results}"); return None

    def _calculate_aggregated_test_metrics(self) -> dict:
        """Calcula métricas agregadas por (IA, Desafio) a partir dos detalhes de teste."""
        # (Sem alterações nesta função - processa o que está em detailed_test_results)
        agg_metrics = {}; result_keys = set((row['ai'], row['challenge']) for row in self.results)
        for key in result_keys:
            ai_name, challenge_name = key; details = [d for d in self.detailed_test_results if d['ai'] == ai_name and d['challenge'] == challenge_name]
            if not details: agg_metrics[key] = {'avg_exec_success_rate': 0.0, 'avg_exec_time_ms': np.nan, 'total_tests_passed': 0, 'total_tests_attempted': 0}; continue
            passed_runs = [d for d in details if d['status'] == 'Pass']
            # Usa o tempo médio já calculado para LRU ou tempos válidos dos outros
            if challenge_name == "LRU Cache":
                 valid_times = [d['execution_time_ms'] for d in details if d.get('execution_time_ms', -1) >= 0] # Tempo médio dos passos
            else:
                 valid_times = [d['execution_time_ms'] for d in details if d.get('execution_time_ms', -1) >= 0 and d['status'] not in ['Unknown', 'Not Executed', 'No Code', 'Error']]
            # Tentativas = todos os que não são 'No Code'
            attempted_runs = [d for d in details if d['status'] not in ['No Code']]
            num_attempted = len(attempted_runs); num_passed = len(passed_runs)
            success_rate = num_passed / num_attempted if num_attempted > 0 else 0.0
            mean_time = np.mean(valid_times) if valid_times else np.nan
            agg_metrics[key] = {'avg_exec_success_rate': success_rate, 'avg_exec_time_ms': mean_time, 'total_tests_passed': num_passed, 'total_tests_attempted': num_attempted}
        return agg_metrics

    def _normalize_dataframe_types(self, df: pd.DataFrame):
        """Converte colunas para tipos numéricos e trata NaNs inplace."""
        # (Sem alterações nesta função)
        expected_numeric_cols = ['api_execution_time', 'lines_of_code', 'completeness', 'flake8_issues', 'num_execution_errors', 'avg_exec_success_rate', 'avg_exec_time_ms', 'total_tests_passed', 'total_tests_attempted']
        for col in expected_numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if col == 'flake8_issues': df.loc[:, col] = df[col].fillna(-1).astype(int)
                elif col == 'num_execution_errors': df.loc[:, col] = df[col].fillna(0).astype(int)
                elif col == 'avg_exec_success_rate': df.loc[:, col] = df[col].fillna(0.0)
                elif col == 'avg_exec_time_ms': df.loc[:, col] = df[col].fillna(-1.0) # Usa -1 para indicar NaN/Não aplicável
                elif col in ['total_tests_passed', 'total_tests_attempted']: df.loc[:, col] = df[col].fillna(0).astype(int)
                else: df.loc[:, col] = df[col].fillna(0.0)
        if 'first_execution_error' in df.columns: df['first_execution_error'] = df['first_execution_error'].fillna('').astype(str)

    def _save_environment_info(self, timestamp: str):
        """Salva as informações do ambiente num arquivo TXT."""
        # (Sem alterações nesta função)
        env_info_path = os.path.join(self.output_dir, f"environment_info_{timestamp}.txt")
        try:
            with open(env_info_path, "w", encoding='utf-8') as f:
                for key, value in self.environment_info.items():
                    if isinstance(value, dict): f.write(f"{key}:\n"); [f.write(f"  {sk}: {sv}\n") for sk, sv in value.items()]
                    else: f.write(f"{key}: {value}\n")
            print(f"- Informações do Ambiente salvas: {env_info_path}")
        except Exception as e: print(f"Erro ao salvar info do ambiente: {e}")

    def _save_full_results_json(self, timestamp: str):
        """Salva a lista completa de resultados brutos em JSON."""
        # (Sem alterações nesta função - usa self.results que foi populado por evaluate_solution)
        json_path = os.path.join(self.output_dir, f"benchmark_results_full_{timestamp}.json")
        try:
            class NpEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer): return int(obj);
                    if isinstance(obj, np.floating): return float(obj);
                    if isinstance(obj, np.ndarray): return obj.tolist();
                    if isinstance(obj, (datetime,)): return obj.isoformat();
                    if isinstance(obj, np.bool_): return bool(obj);
                    if isinstance(obj, bytes): return obj.decode('utf-8', errors='replace');
                    return super(NpEncoder, self).default(obj)
            with open(json_path, "w", encoding='utf-8') as f: json.dump(self.results, f, indent=2, ensure_ascii=False, cls=NpEncoder)
            print(f"- Resultados detalhados (JSON completo): {json_path}")
        except Exception as e: print(f"Erro ao salvar JSON completo: {e}")

    def _save_detailed_tests_csv(self, timestamp: str):
        """Salva os detalhes de cada caso de teste (ou sequência LRU) em CSV."""
        # (Sem alterações nesta função - usa self.detailed_test_results populado por evaluate_solution)
        detailed_csv_path = os.path.join(self.output_dir, f"benchmark_test_details_{timestamp}.csv")
        if self.detailed_test_results:
            try:
                df_detailed = pd.DataFrame(self.detailed_test_results); cols_order = ['ai', 'challenge', 'difficulty', 'test_case_index', 'status', 'execution_time_ms', 'error_message', 'stdout', 'input_args_str', 'expected_output_str', 'actual_output_str']
                for col in cols_order: # Garante que colunas existam
                     if col not in df_detailed.columns: df_detailed[col] = pd.NA
                df_detailed = df_detailed[cols_order]; df_detailed.to_csv(detailed_csv_path, index=False, encoding='utf-8'); print(f"- Detalhes por caso de teste (CSV): {detailed_csv_path}")
            except Exception as e: print(f"Erro ao salvar CSV detalhado por teste: {e}\nTraceback: {traceback.format_exc()}")
        else: print("- Nenhum detalhe por caso de teste para salvar.")

    def _create_csv_header(self) -> str:
        """Cria um cabeçalho de comentário com informações do ambiente para CSVs."""
        # (Sem alterações nesta função)
        env_comments = ["# --- Environment Info ---"];
        for key, value in self.environment_info.items():
            if isinstance(value, dict): env_comments.append(f"# {key}:"); [env_comments.append(f"#   {sk}: {sv}") for sk, sv in value.items()]
            else: env_comments.append(f"# {key}: {value}")
        env_comments.append("# --- Benchmark Data ---"); return "\n".join(env_comments) + "\n"

    def _save_summary_csv(self, df: pd.DataFrame, timestamp: str):
        """Salva o resumo do benchmark (DataFrame principal) em CSV."""
        # (Sem alterações nesta função)
        summary_path = os.path.join(self.output_dir, f"benchmark_summary_{timestamp}.csv")
        try:
            cols_to_drop_summary = ['code', 'full_response']; summary_cols = [col for col in df.columns if col not in cols_to_drop_summary]; summary = df[summary_cols].copy()
            env_header = self._create_csv_header()
            with open(summary_path, "w", encoding='utf-8', newline='') as f: f.write(env_header); summary.to_csv(f, index=False, encoding='utf-8', float_format='%.4f', lineterminator='\n')
            print(f"- Resumo salvo (CSV): {summary_path}")
        except Exception as e: print(f"Erro ao salvar resumo CSV: {e}\nTraceback: {traceback.format_exc()}")

    def _calculate_statistics(self, df: pd.DataFrame) -> pd.DataFrame | None:
         """Calcula estatísticas agregadas por IA."""
         # (Sem alterações nesta função)
         try:
             stats_metrics = [("Average API Time (s)", "api_execution_time", "mean"), ("Average Lines of Code", "lines_of_code", "mean"), ("Average Completeness (%)", "completeness", "mean"), ("Average Flake8 Issues", "flake8_issues", "mean_valid"), ("Total Tests Passed", "total_tests_passed", "sum"), ("Total Tests Attempted", "total_tests_attempted", "sum"), ("Overall Execution Success (%)", "avg_exec_success_rate", "mean_percentage"), ("Average Execution Time (ms)", "avg_exec_time_ms", "mean_valid"), ("Total Execution Errors/Fails", "num_execution_errors", "sum"), ("Example First Error", "first_execution_error", "first_non_empty")] # Renomeado métrica de erro
             ai_list = sorted(df['ai'].unique()); stats_data = {"Metric": [m[0] for m in stats_metrics]}
             for ai_name in ai_list:
                 df_ai = df[df["ai"] == ai_name]; ai_column_data = []
                 for metric_name, col_key, agg_type in stats_metrics:
                     if col_key not in df_ai.columns: ai_column_data.append(np.nan); continue
                     if agg_type == "mean": avg_val = df_ai[col_key].mean()
                     elif agg_type == "sum": avg_val = df_ai[col_key].sum()
                     elif agg_type == "mean_valid": valid_data = df_ai[df_ai[col_key].notna() & (df_ai[col_key] != -1)][col_key]; avg_val = valid_data.mean() if not valid_data.empty else np.nan
                     elif agg_type == "mean_percentage": avg_val = df_ai[col_key].mean() * 100
                     elif agg_type == "first_non_empty": non_empty_errors = df_ai[df_ai[col_key].fillna('').astype(str) != ''][col_key]; avg_val = non_empty_errors.iloc[0] if not non_empty_errors.empty else "";
                     else: avg_val = np.nan
                     ai_column_data.append(avg_val)
                 stats_data[ai_name] = ai_column_data
             return pd.DataFrame(stats_data)
         except Exception as e: print(f"Erro ao calcular estatísticas: {e}\nTraceback: {traceback.format_exc()}"); return None

    def _save_statistics_csv(self, stats_df: pd.DataFrame, timestamp: str):
        """Salva as estatísticas calculadas em CSV."""
        # (Sem alterações nesta função)
        stats_path = os.path.join(self.output_dir, f"benchmark_stats_{timestamp}.csv");
        try:
            env_header_stats = self._create_csv_header()
            with open(stats_path, "w", encoding='utf-8', newline='') as f: f.write(env_header_stats); stats_df.to_csv(f, index=False, encoding='utf-8', float_format='%.2f', lineterminator='\n')
            print(f"- Estatísticas salvas (CSV): {stats_path}")
        except Exception as e: print(f"Erro ao salvar estatísticas CSV: {e}\nTraceback: {traceback.format_exc()}")

    def _display_summary_table(self, stats_df: pd.DataFrame):
        """Exibe uma tabela de resumo das estatísticas no console."""
        # (Sem alterações nesta função)
        print("\n===== SUMÁRIO DO BENCHMARK (ESTATÍSTICAS) =====")
        if stats_df is None or stats_df.empty: print("Não foi possível gerar o sumário de estatísticas."); return
        stats_display = stats_df.copy()
        for col in stats_display.columns:
            if col == 'Metric': continue
            formatted_col = [];
            if stats_display.empty: continue
            for i in stats_display.index:
                metric_name = stats_display.loc[i, 'Metric']; value = stats_display.loc[i, col]; formatted_val = "N/A" # Default
                if not pd.isna(value):
                    if "Flake8" in metric_name: formatted_val = f"{value:.1f}" if value != -1 else "N/A"
                    elif "Time (ms)" in metric_name: formatted_val = f"{value:.1f}" if value != -1 else "N/A"
                    elif "Success (%)" in metric_name or "Completeness (%)" in metric_name: formatted_val = f"{value:.1f}%"
                    elif "Total Tests" in metric_name or "Total Execution Errors/Fails" in metric_name: formatted_val = f"{int(value)}" # Ajustado nome métrica
                    elif "Time (s)" in metric_name: formatted_val = f"{value:.2f}"
                    elif "Example First Error" in metric_name: val_str = str(value); formatted_val = val_str[:60] + ('...' if len(val_str) > 60 else '')
                    else: formatted_val = f"{value:.1f}"
                formatted_col.append(formatted_val)
            if formatted_col: stats_display[col] = formatted_col
        try: from tabulate import tabulate; print(tabulate(stats_display, headers='keys', tablefmt='grid', showindex=False))
        except ImportError: print("(Instale 'tabulate' para uma tabela mais bonita: pip install tabulate)"); print(stats_display.to_string(index=False))
        except Exception as e_print: print(f"Erro ao imprimir tabela final: {e_print}"); print(stats_display.to_string(index=False)) # Fallback

    def generate_report(self):
        """Gera todos os relatórios e visualizações."""
        # (Sem alterações nesta função - chama os auxiliares)
        if not self.results: print("Não há resultados para gerar o relatório."); return
        print("\n--- Gerando Relatórios ---"); timestamp = datetime.now().strftime("%Y%m%d_%H%M%S"); print(f"Timestamp dos relatórios: {timestamp}"); print(f"Diretório de saída: '{self.output_dir}/'")
        self._save_environment_info(timestamp)
        df = self._prepare_dataframe()
        if df is None: print("Falha ao preparar o DataFrame. Geração de relatórios subsequentes abortada."); return
        self._save_full_results_json(timestamp)
        self._save_detailed_tests_csv(timestamp)
        self._save_summary_csv(df, timestamp)
        stats_df = self._calculate_statistics(df)
        if stats_df is not None: self._save_statistics_csv(stats_df, timestamp)
        else: print("- Não foi possível calcular ou salvar as estatísticas.")
        try: print("\n--- Gerando Visualizações ---"); self.create_visualizations(df, timestamp, self.output_dir); print(f"- Visualizações salvas como PNG em '{self.output_dir}/'")
        except Exception as e: print(f"Erro ao gerar visualizações: {e}\nTraceback: {traceback.format_exc()}")
        if stats_df is not None: self._display_summary_table(stats_df)

    # --- Métodos Auxiliares para Visualizações ---
    # (_plot_bar_charts, _plot_box_plots, _plot_radar_chart permanecem os mesmos)
    # NOTE: Nenhuma alteração necessária nestas funções. Elas usam os dados
    # do DataFrame principal, que já reflete os resultados (incluindo LRU).

    def _plot_bar_charts(self, df_plot: pd.DataFrame, ai_list: list, ai_color_map: dict, challenge_order: list, timestamp: str, output_dir: str):
        """Gera gráficos de barras para diversas métricas."""
        # (Sem alterações nesta função)
        print("Gerando Gráficos de Barras..."); metrics_to_plot = {"1_api_execution_time": ("Tempo Médio de Resposta API", "Tempo (s)", "api_execution_time"), "2_lines_of_code": ("Média Linhas de Código (LoC)", "Número de Linhas", "lines_of_code"), "3_completeness": ("Completude Média da Solução", "Score (%)", "completeness"), "4_flake8_issues": ("Média Problemas Flake8", "Número de Problemas", "flake8_issues"), "7_num_execution_errors": ("Média Erros/Falhas Execução", "Número de Erros/Falhas", "num_execution_errors"),} # Nome métrica ajustado
        if 'avg_exec_success_rate' in df_plot.columns: metrics_to_plot["5_avg_exec_success_rate"] = ("Taxa Média de Sucesso na Execução", "Sucesso (%)", "avg_exec_success_rate")
        if 'avg_exec_time_ms' in df_plot.columns: metrics_to_plot["6_avg_exec_time_ms"] = ("Tempo Médio de Execução", "Tempo (ms)", "avg_exec_time_ms") # Label Y ajustado
        metrics_to_plot = dict(sorted(metrics_to_plot.items()))
        for metric_key, (title, ylabel, col_name) in metrics_to_plot.items():
            if col_name not in df_plot.columns: print(f"Pulando gráfico de barras '{title}': Coluna {col_name} não encontrada."); continue
            plot_data = df_plot.copy()
            if col_name == 'flake8_issues': plot_data = plot_data[plot_data[col_name] != -1]
            elif col_name == 'avg_exec_time_ms': plot_data = plot_data[plot_data[col_name].notna() & (plot_data[col_name] != -1)]
            elif col_name == 'num_execution_errors': plot_data = plot_data[plot_data[col_name].notna()]
            elif col_name == 'avg_exec_success_rate': plot_data[col_name] = plot_data[col_name] * 100
            if plot_data.empty or plot_data[col_name].isnull().all(): print(f"Pulando gráfico de barras '{title}': Nenhum dado válido encontrado."); continue
            try:
                plt.figure(); pivot = plot_data.pivot_table(index='challenge', columns='ai', values=col_name, aggfunc='mean'); pivot = pivot.reindex(challenge_order, axis=0).reindex(ai_list, axis=1);
                fill_value = 0.0;
                if col_name in ['flake8_issues', 'avg_exec_time_ms', 'num_execution_errors']: fill_value = np.nan
                pivot.fillna(fill_value, inplace=True);
                if pivot.isnull().all().all(): print(f"Pulando gráfico de barras '{title}': Nenhum dado válido (tudo NaN)."); plt.close(); continue
                num_ais = len(ai_list); ax = pivot.plot(kind='bar', figsize=(max(12, num_ais * 2), 7), color=[ai_color_map.get(ai, 'grey') for ai in pivot.columns], width=0.8)
                plt.title(f'{title} por Desafio'); plt.ylabel(ylabel); plt.xlabel('Desafio (Ordenado por Dificuldade)'); plt.xticks(rotation=45, ha='right', fontsize=9); plt.yticks(fontsize=9)
                if col_name in ['completeness', 'avg_exec_success_rate']: plt.ylim(0, 105)
                elif col_name in ['flake8_issues', 'num_execution_errors'] and not pivot.isnull().all().all(): max_val = pivot.max().max(); plt.ylim(bottom=0, top = max(1, max_val * 1.1) if not pd.isna(max_val) else 1)
                elif col_name == 'avg_exec_time_ms' and not pivot.isnull().all().all(): plt.ylim(bottom=0)
                plt.legend(title='IA', bbox_to_anchor=(1.02, 1), loc='upper left'); plt.tight_layout(rect=[0, 0, 0.88, 1]); path = os.path.join(output_dir, f"{metric_key}_{timestamp}.png"); plt.savefig(path); plt.close()
            except Exception as e: print(f"Erro ao gerar gráfico de barras '{title}': {e}\nTraceback: {traceback.format_exc()}"); plt.close()

    def _plot_box_plots(self, df_plot: pd.DataFrame, ai_list: list, ai_color_map: dict, timestamp: str, output_dir: str):
        """Gera box plots para comparar a distribuição das métricas entre IAs."""
        # (Sem alterações nesta função)
        print("Gerando Box Plots..."); metrics_for_boxplot = {"api_execution_time": {"title": "Distribuição Tempo Resposta API por IA", "ylabel": "Tempo (s)", "valid_condition": None}, "lines_of_code": {"title": "Distribuição Linhas de Código (LoC) por IA", "ylabel": "Número de Linhas", "valid_condition": (lambda df, col: df[col] > 0)}, "completeness": {"title": "Distribuição Completude da Solução por IA", "ylabel": "Score (%)", "valid_condition": None}}
        if 'flake8_issues' in df_plot.columns: metrics_for_boxplot["flake8_issues"] = {"title": "Distribuição Problemas Flake8 por IA", "ylabel": "Número de Problemas", "valid_condition": (lambda df, col: df[col] != -1)}
        if 'avg_exec_time_ms' in df_plot.columns: metrics_for_boxplot["avg_exec_time_ms"] = {"title": "Distribuição Tempo Médio Execução por IA", "ylabel": "Tempo (ms)", "valid_condition": (lambda df, col: df[col].notna() & (df[col] != -1))} # Título ajustado
        if 'num_execution_errors' in df_plot.columns: metrics_for_boxplot["num_execution_errors"] = {"title": "Distribuição Erros/Falhas Execução por IA", "ylabel": "Número de Erros/Falhas", "valid_condition": (lambda df, col: df[col].notna())} # Título ajustado
        plot_counter_boxplot = 7
        for metric_key_boxplot, config in metrics_for_boxplot.items():
            plot_counter_boxplot += 1;
            if metric_key_boxplot not in df_plot.columns: print(f"  Pulando Box Plot para '{metric_key_boxplot}': coluna não encontrada."); continue
            data_for_current_metric = []; labels_for_current_metric = []; temp_df = df_plot.copy()
            if config.get("valid_condition"): try: condition_func = config["valid_condition"]; temp_df = temp_df[condition_func(temp_df, metric_key_boxplot)]; except Exception as e: print(f"  Aviso: Falha ao aplicar condição para boxplot de '{metric_key_boxplot}': {e}.")
            if temp_df.empty or temp_df[metric_key_boxplot].isnull().all(): print(f"  Pulando Box Plot para '{metric_key_boxplot}': Sem dados válidos."); continue
            for ai_name in ai_list:
                ai_data = temp_df[temp_df['ai'] == ai_name][metric_key_boxplot].dropna()
                if not ai_data.empty: data_for_current_metric.append(ai_data.values); labels_for_current_metric.append(ai_name)
            if not data_for_current_metric: print(f"  Pulando Box Plot para '{metric_key_boxplot}': Nenhum dado encontrado por IA."); continue
            try:
                plt.figure(figsize=(max(8, len(labels_for_current_metric) * 1.2), 6)); box_plot = plt.boxplot(data_for_current_metric, labels=labels_for_current_metric, patch_artist=True, showfliers=True)
                plt.title(config["title"], fontsize=14); plt.ylabel(config["ylabel"], fontsize=10); plt.xlabel("IA", fontsize=10); plt.xticks(rotation=30, ha='right', fontsize=9); plt.yticks(fontsize=9); plt.grid(axis='y', linestyle='--', alpha=0.7)
                colors_to_use = [ai_color_map.get(label, 'grey') for label in labels_for_current_metric];
                for patch, color in zip(box_plot['boxes'], colors_to_use): patch.set_facecolor(color); patch.set_alpha(0.6)
                plt.tight_layout(); clean_metric_name = metric_key_boxplot.replace('_ms', ''); plot_filename = f"{plot_counter_boxplot}_boxplot_{clean_metric_name}_{timestamp}.png"; path = os.path.join(output_dir, plot_filename); plt.savefig(path); plt.close()
            except Exception as e: print(f"  Erro ao gerar Box Plot para '{metric_key_boxplot}': {e}"); plt.close()
        print("Geração de Box Plots concluída.")

    def _plot_radar_chart(self, df: pd.DataFrame, ai_list: list, ai_color_map: dict, timestamp: str, output_dir: str):
        """Gera um gráfico de radar comparando métricas normalizadas entre IAs."""
        # (Sem alterações nesta função)
        num_ais = len(ai_list);
        if num_ais < 2: print("Gráfico Radar pulado (requer >= 2 IAs)."); return
        print(f"Gerando gráfico Radar para {num_ais} IAs (3 métricas base)...")
        try:
            numeric_cols_radar = ['api_execution_time', 'lines_of_code', 'completeness'];
            if not all(col in df.columns for col in numeric_cols_radar): missing = [col for col in numeric_cols_radar if col not in df.columns]; print(f"Pulando Radar: Faltando colunas {missing}"); return
            means_data = {}
            for ai in ai_list: df_ai_radar = df[df["ai"] == ai]; means = {}; for col in numeric_cols_radar: means[col] = df_ai_radar[col].mean() if not df_ai_radar[col].isnull().all() else 0; means_data[ai] = means
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
            plot_counter_radar = 15 # Contador aproximado
            path = os.path.join(output_dir, f"{plot_counter_radar}_radar_chart_{timestamp}.png"); plt.savefig(path); plt.close()
        except Exception as e: print(f"Erro ao gerar gráfico Radar: {e}\nTraceback: {traceback.format_exc()}"); plt.close()

    def create_visualizations(self, df: pd.DataFrame, timestamp: str, output_dir: str):
        """Cria e salva visualizações (gráficos de barras, box plots, radar chart)."""
        # (Sem alterações nesta função - chama os auxiliares)
        plt.style.use('ggplot'); ai_list = sorted(df['ai'].unique()); num_ais = len(ai_list)
        if num_ais <= 10: colors = plt.cm.tab10(np.linspace(0, 1, num_ais))
        else: colors = plt.cm.viridis(np.linspace(0, 1, num_ais))
        ai_color_map = {ai: colors[i] for i, ai in enumerate(ai_list)}
        base_required_cols = ['challenge', 'ai', 'api_execution_time', 'lines_of_code', 'completeness', 'flake8_issues', 'difficulty', 'num_execution_errors']
        exec_cols = [col for col in ['avg_exec_success_rate', 'avg_exec_time_ms'] if col in df.columns]; required_cols = base_required_cols + exec_cols
        if df.empty or not all(col in df.columns for col in required_cols): missing_cols = [col for col in required_cols if col not in df.columns]; print(f"AVISO: DataFrame vazio ou faltando colunas para gerar gráficos. Faltando: {missing_cols}. Pulando visualizações."); return
        difficulty_order = pd.CategoricalDtype(['Easy', 'Medium', 'Hard'], ordered=True); df['difficulty_cat'] = df['difficulty'].astype(difficulty_order)
        challenge_order = df.sort_values('difficulty_cat')['challenge'].unique(); df_plot = df.copy()
        self._plot_bar_charts(df_plot, ai_list, ai_color_map, challenge_order, timestamp, output_dir)
        self._plot_box_plots(df_plot, ai_list, ai_color_map, timestamp, output_dir)
        self._plot_radar_chart(df, ai_list, ai_color_map, timestamp, output_dir)


# --- Ponto de Entrada Principal ---
if __name__ == "__main__":
    print("===========================================================")
    print(" AI Programming Benchmark: Claude vs ChatGPT vs Gemini")
    print("      (Incluindo Análise Flake8 e Execução Externa/Local)")
    print("===========================================================")

    # Necessário para multiprocessing em alguns ambientes (ex: Windows 'spawn')
    # Se estiver a usar 'fork' (default Linux/macOS), isto não é estritamente necessário
    # mas não prejudica.
    multiprocessing.freeze_support()

    if not os.path.exists(".env"):
        print("AVISO: Arquivo .env não encontrado. Chaves API devem ser variáveis de ambiente.")

    try:
        benchmark = AIBenchmark()
        if benchmark.claude_api_key or benchmark.openai_api_key or benchmark.gemini_api_key:
            benchmark.run_benchmark()
            print("\nBenchmark concluído.")
            print(f"Relatórios salvos em: '{benchmark.output_dir}'")
        else:
            print("\nERRO FATAL: Nenhuma chave de API válida configurada ou encontrada. "
                  "Benchmark não pode ser executado.")

    except Exception as main_e:
        print("\nERRO INESPERADO DURANTE A EXECUÇÃO DO BENCHMARK:")
        print(f"Tipo: {type(main_e).__name__}"); print(f"Erro: {main_e}"); print("Traceback:"); traceback.print_exc()

    print("===========================================================")