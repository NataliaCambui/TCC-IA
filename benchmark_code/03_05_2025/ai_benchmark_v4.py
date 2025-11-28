

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
import google.api_core.exceptions # <-- Importar exceções específicas do Google
import subprocess
import tempfile
import platform
import traceback # <-- Importar para debug de erros inesperados

# Carrega variáveis de ambiente do .env
load_dotenv()

# --- Configuração Global da API Gemini ---
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
# -----------------------------------------

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
        self.code_challenges = [
            {
                "name": "Fibonacci Sequence",
                "description": "Generate the first n numbers in the Fibonacci sequence, starting with 0.",
                "difficulty": "Easy",
                "function_signature": "fibonacci(n: int) -> list[int]",
                "test_cases": [
                    {"input_args": [10], "expected_output": [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]},
                    {"input_args": [1], "expected_output": [0]},
                    {"input_args": [0], "expected_output": []},
                    {"input_args": [2], "expected_output": [0, 1]}
                ]
            },
            {
                "name": "Binary Search",
                "description": "Find the index of a target value in a sorted list of integers. Return -1 if the target is not found.",
                "difficulty": "Medium",
                "function_signature": "binary_search(arr: list[int], target: int) -> int",
                "test_cases": [
                    {"input_args": [[2, 5, 7, 8, 11, 12], 13], "expected_output": -1},
                    {"input_args": [[2, 5, 7, 8, 11, 12], 8], "expected_output": 3},
                    {"input_args": [[1, 3, 5], 1], "expected_output": 0},
                    {"input_args": [[1, 3, 5], 5], "expected_output": 2},
                    {"input_args": [[], 5], "expected_output": -1} # Teste com lista vazia
                ]
            },
            {
                "name": "Merge Sort",
                "description": "Implement the merge sort algorithm to sort a list of integers in ascending order. Return a new sorted list.",
                "difficulty": "Medium",
                "function_signature": "merge_sort(arr: list[int]) -> list[int]",
                "test_cases": [
                    {"input_args": [[3, 1, 4, 1, 5, 9, 2, 6]], "expected_output": [1, 1, 2, 3, 4, 5, 6, 9]},
                    {"input_args": [[5, 4, 3, 2, 1]], "expected_output": [1, 2, 3, 4, 5]},
                    {"input_args": [[1]], "expected_output": [1]},
                    {"input_args": [[]], "expected_output": []}
                ]
            },
            {
                "name": "Count Islands in Grid",
                "description": "Count the number of islands in a 2D grid (list of lists) where 1 is land and 0 is water. An island is formed by connecting adjacent lands horizontally or vertically.",
                "difficulty": "Hard",
                "function_signature": "count_islands(grid: list[list[int]]) -> int",
                "test_cases": [
                    {"input_args": [[ # Grid 1
                        [1, 1, 1, 1, 0],
                        [1, 1, 0, 1, 0],
                        [1, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0]
                    ]], "expected_output": 1},
                    {"input_args": [[ # Grid 2
                        [1, 1, 0, 0, 0],
                        [1, 1, 0, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 0, 1, 1]
                    ]], "expected_output": 3},
                    {"input_args": [[ # Grid 3 (vazio)
                    ]], "expected_output": 0},
                     {"input_args": [[ # Grid 4 (tudo agua)
                        [0, 0, 0],
                        [0, 0, 0]
                    ]], "expected_output": 0}
                ]
             },
             {
                "name": "Palindrome Check",
                "description": "Check if a given string is a palindrome, ignoring case and non-alphanumeric characters. Return True if it is, False otherwise.",
                "difficulty": "Easy",
                "function_signature": "is_palindrome(s: str) -> bool",
                "test_cases": [
                    {"input_args": ["A man, a plan, a canal: Panama"], "expected_output": True},
                    {"input_args": ["race a car"], "expected_output": False},
                    {"input_args": [" "], "expected_output": True}, # Considerar espaço como palíndromo aqui
                    {"input_args": [""], "expected_output": True},
                    {"input_args": ["Was it a car or a cat I saw?"], "expected_output": True},
                    {"input_args": ["tab a cat"], "expected_output": False}
                ]
            },
            {
                "name": "Valid Parentheses",
                "description": "Given a string containing just '(', ')', '{', '}', '[' and ']', determine if the input string is valid (brackets must close in the correct order).",
                "difficulty": "Medium",
                "function_signature": "is_valid_parentheses(s: str) -> bool",
                "test_cases": [
                    {"input_args": ["()"], "expected_output": True},
                    {"input_args": ["()[]{}"], "expected_output": True},
                    {"input_args": ["(]"], "expected_output": False},
                    {"input_args": ["([)]"], "expected_output": False},
                    {"input_args": ["{[]}"], "expected_output": True},
                    {"input_args": [""], "expected_output": True},
                    {"input_args": ["["], "expected_output": False},
                    {"input_args": ["]"], "expected_output": False}
                ]
             },
             {
                "name": "LRU Cache",
                "description": "Implement a Least Recently Used (LRU) cache with a given capacity. Support get(key) and put(key, value) operations. get returns value or -1. put inserts/updates; evicts LRU item if capacity exceeded.",
                "difficulty": "Hard",
                "function_signature": "class LRUCache:\n    def __init__(self, capacity: int):\n        pass\n    def get(self, key: int) -> int:\n        pass\n    def put(self, key: int, value: int) -> None:\n        pass",
                # ADENDO: Testar classes via executor atual é complexo.
                # Estes 'test_cases' descrevem sequências, mas precisaríamos
                # adaptar o executor OU criar uma função wrapper para enviar.
                # Por ora, servem como documentação dos testes desejados.
                "test_cases": [
                    {
                        "description": "Simple put/get sequence",
                        "sequence": [
                            {"op": "__init__", "args": [2]}, # capacity = 2
                            {"op": "put", "args": [1, 1]},
                            {"op": "put", "args": [2, 2]},
                            {"op": "get", "args": [1], "expected": 1},
                            {"op": "put", "args": [3, 3]}, # evicts key 2
                            {"op": "get", "args": [2], "expected": -1},
                            {"op": "put", "args": [4, 4]}, # evicts key 1
                            {"op": "get", "args": [1], "expected": -1},
                            {"op": "get", "args": [3], "expected": 3},
                            {"op": "get", "args": [4], "expected": 4}
                        ]
                    },
                    # Poderíamos adicionar mais sequências aqui
                ]
             }
        ]
        # --- FIM DA LISTA DE DESAFIOS ---

        self._check_flake8()

    def _check_flake8(self):
        """Verifica se flake8 está acessível."""
        try:
             subprocess.run(['flake8', '--version'], check=True, capture_output=True, timeout=5)
             print("Verificação Flake8: OK"); self.flake8_available = True
        except Exception:
             print("AVISO: Comando 'flake8' não encontrado/executável. Análise de qualidade será pulada."); self.flake8_available = False

    def query_claude(self, prompt):
        """Envia uma consulta para a API Claude com tratamento de erro revisado."""
        start_time = time.time()
        execution_time = 0 # Inicializa aqui
        content = ""

        if not self.claude_api_key:
             print("Pulando Claude: Chave API não fornecida.")
             # Calcula tempo de execução mesmo pulando (será próximo de zero)
             execution_time = time.time() - start_time
             return {"content": "", "execution_time": execution_time}

        headers = {"x-api-key": self.claude_api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        data = {"model": "claude-3-opus-20240229", "max_tokens": 4000, "messages": [{"role": "user", "content": prompt}]}

        try:
            response = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=data, timeout=120)
            response.raise_for_status()
            execution_time = time.time() - start_time # Recalcula se sucesso
            result = response.json()
            # Validação da resposta... (igual antes)
            if "content" in result and isinstance(result["content"], list) and len(result["content"]) > 0:
                 first_content = result["content"][0]; content = first_content.get("text", "") if isinstance(first_content, dict) else ""
            else: print(f"Resposta Claude: Estrutura 'content' inesperada. Resp: {result}")
            return {"content": content or "", "execution_time": execution_time} # Retorna daqui em caso de sucesso

        except requests.exceptions.RequestException as e:
            execution_time = time.time() - start_time # Calcula tempo no erro
            error_message = f"Erro consulta Claude (RequestException): {e}"
            if e.response is not None:
                error_message += f" | Status: {e.response.status_code}"
                try: error_message += f" | Body: {e.response.json()}"
                except json.JSONDecodeError: error_message += f" | Body: {e.response.text}"
            print(error_message)
            # Retorna dicionário de erro diretamente do except
            return {"content": "", "execution_time": execution_time}

        except Exception as e:
            execution_time = time.time() - start_time # Calcula tempo no erro
            print(f"Erro inesperado consulta Claude: {type(e).__name__} - {e}")
            # traceback.print_exc() # Descomente para ver o stack trace completo em erros inesperados
            # Retorna dicionário de erro diretamente do except
            return {"content": "", "execution_time": execution_time}

    def query_chatgpt(self, prompt):
        """Envia uma consulta para a API ChatGPT (OpenAI) com tratamento de erro revisado."""
        start_time = time.time()
        execution_time = 0 # Inicializa aqui
        content = ""

        if not self.openai_api_key:
             print("Pulando ChatGPT: Chave API não fornecida.")
             execution_time = time.time() - start_time
             return {"content": "", "execution_time": execution_time}

        headers = {"Authorization": f"Bearer {self.openai_api_key}", "Content-Type": "application/json"}
        data = {"model": "gpt-4-turbo", "messages": [{"role": "user", "content": prompt}], "max_tokens": 4000}

        try:
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data, timeout=120)
            response.raise_for_status()
            execution_time = time.time() - start_time # Recalcula se sucesso
            result = response.json()
            # Validação da resposta... (igual antes)
            if "choices" in result and isinstance(result["choices"], list) and len(result["choices"]) > 0:
                first_choice = result["choices"][0]
                if isinstance(first_choice, dict) and "message" in first_choice and isinstance(first_choice["message"], dict):
                    content = first_choice["message"].get("content", "")
                else: print(f"Resposta ChatGPT: Estrutura 'message' inesperada. Resp: {result}")
            else: print(f"Resposta ChatGPT: Estrutura 'choices' inesperada. Resp: {result}")
            return {"content": content or "", "execution_time": execution_time} # Retorna daqui em caso de sucesso

        except requests.exceptions.RequestException as e:
            execution_time = time.time() - start_time # Calcula tempo no erro
            error_message = f"Erro consulta ChatGPT (RequestException): {e}"
            if e.response is not None:
                error_message += f" | Status: {e.response.status_code}"
                try:
                    error_details = e.response.json(); error_msg = error_details.get('error', {}).get('message', '')
                    error_message += f" | API Error: {error_msg}" if error_msg else f" | Body: {error_details}"
                except json.JSONDecodeError: error_message += f" | Body: {e.response.text}"
            print(error_message)
             # Retorna dicionário de erro diretamente do except
            return {"content": "", "execution_time": execution_time}

        except Exception as e:
            execution_time = time.time() - start_time # Calcula tempo no erro
            print(f"Erro inesperado consulta ChatGPT: {type(e).__name__} - {e}")
            # traceback.print_exc() # Descomente para debug
             # Retorna dicionário de erro diretamente do except
            return {"content": "", "execution_time": execution_time}


    def query_gemini(self, prompt):
        """Envia uma consulta para a API Gemini com tratamento de erro revisado e rate limit."""
        start_time = time.time()
        execution_time = 0 # Inicializa aqui
        content = ""

        if not self.gemini_api_key:
            print("Pulando Gemini: Chave API não configurada.")
            execution_time = time.time() - start_time
            return {"content": "", "execution_time": execution_time}

        try:
            model = genai.GenerativeModel('gemini-1.5-pro-latest')
            safety_settings = [{"category": c, "threshold": "BLOCK_NONE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
            # Adicionado timeout à chamada generate_content se suportado pela biblioteca/método (verificar documentação)
            # Exemplo: response = model.generate_content(prompt, safety_settings=safety_settings, request_options={'timeout': 120})
            # Se não suportado diretamente, a lógica de timeout do request (se houver) ou outra forma deve ser usada.
            # Por enquanto, não adicionamos timeout explícito aqui, assumindo que a biblioteca gerencia.
            response = model.generate_content(prompt, safety_settings=safety_settings)

            execution_time = time.time() - start_time # Recalcula se sucesso
            try:
                content = response.text
            except ValueError: # Captura erro se 'text' não acessível (ex: bloqueio)
                 print(f"Resposta Gemini não contém texto (possivelmente bloqueada).")
                 try:
                     print(f"   Prompt Feedback: {response.prompt_feedback}")
                     if hasattr(response, 'candidates') and response.candidates: print(f"   Finish Reason: {response.candidates[0].finish_reason}")
                 except Exception as feedback_error: print(f"   Erro ao acessar feedback Gemini: {feedback_error}")
            return {"content": content or "", "execution_time": execution_time} # Retorna daqui em caso de sucesso

        # --- Tratamento específico para Rate Limit ---
        except google.api_core.exceptions.ResourceExhausted as e:
            execution_time = time.time() - start_time # Calcula tempo no erro
            print(f"Erro consulta Gemini (ResourceExhausted - Rate Limit): {e}")
            print("       >> Atingido o limite de requisições da API Gemini (provavelmente o limite da camada gratuita). Tente novamente mais tarde ou verifique seu plano.")
            # Retorna dicionário de erro diretamente do except
            return {"content": "", "execution_time": execution_time}
        # ---------------------------------------------

        except Exception as e:
            execution_time = time.time() - start_time # Calcula tempo no erro
            print(f"Erro inesperado consulta Gemini: {type(e).__name__} - {e}")
            # traceback.print_exc() # Descomente para debug
            # Retorna dicionário de erro diretamente do except
            return {"content": "", "execution_time": execution_time}

    # --- Funções de Avaliação (extract_code, count_code_lines, measure_response_completeness, analyze_code_quality_flake8, evaluate_solution) ---
    def extract_code(self, content):
        """Extrai blocos de código Python de uma string de texto."""
        if not content: return ""
        # Primeiro tenta encontrar blocos ```python ... ```
        code_blocks_python = re.findall(r'```python\n(.*?)\n```', content, re.DOTALL)
        if code_blocks_python: return code_blocks_python[0].strip()
        # Se não encontrar, tenta blocos genéricos ``` ... ```
        code_blocks_generic = re.findall(r'```(?:.*?\n)?(.*?)\n```', content, re.DOTALL)
        if code_blocks_generic: return code_blocks_generic[0].strip()
        # Se ainda não encontrar, tenta extrair por indentação ou palavras-chave 'def'/'class'
        lines = content.split('\n'); code_lines = []; start_found = False
        for line in lines:
            stripped = line.strip()
            # Considera início do código se for indentado ou começar com def/class
            if not start_found and (line.startswith(' ') or line.startswith('\t') or stripped.startswith('def ') or stripped.startswith('class ')):
                start_found = True
            if start_found:
                code_lines.append(line)
        if code_lines:
            return '\n'.join(code_lines).strip()
        # Último recurso: se for curto e não tiver def/class, provavelmente não é código. Senão, retorna tudo.
        if len(content.splitlines()) < 3 and ("def " not in content and "class " not in content):
             return "" # Assume que não é código se for muito curto e sem keywords
        else:
             return content.strip() # Pode ser código sem markdown

    def count_code_lines(self, code):
        """Conta linhas de código não vazias e que não são comentários."""
        if not code: return 0
        return len([line for line in code.splitlines() if line.strip() and not line.strip().startswith('#')])

    def measure_response_completeness(self, code, description):
        """Avalia heuristicamente a completude da resposta baseada em keywords e estrutura."""
        completeness_score = 0
        lines_count = self.count_code_lines(code)

        if lines_count > 0:
            completeness_score += 50 # Base score for having any code

            # Presence of function or class definition
            if "def " in code or "class " in code:
                completeness_score += 10

            # Presence of return or yield (indication of output)
            if "return " in code or "yield " in code:
                completeness_score += 10
            # Specific check for island count which might not use return directly
            elif "island" in description.lower() and ("dfs(" in code or "bfs(" in code):
                 completeness_score += 5 # Bonus if it seems to implement traversal

            # Keyword matching bonus based on description
            desc_lower = description.lower()
            code_lower = code.lower()
            keyword_bonus = 0
            if "fibonacci" in desc_lower and ("fibonacci" in code_lower or "fib(" in code_lower or "yield" in code):
                keyword_bonus = 10
            elif "binary search" in desc_lower and ("mid =" in code or "mid=" in code or "low" in code or "high" in code):
                keyword_bonus = 10
            elif "merge sort" in desc_lower and ("merge(" in code_lower or ("left" in code and "right" in code and "mid" in code)):
                 keyword_bonus = 10
            elif "island" in desc_lower and ("dfs(" in code_lower or "bfs(" in code_lower or "visit" in code_lower or ("grid[" in code and "==" in code)):
                 keyword_bonus = 10
            elif "palindrome" in desc_lower and ("[::-1]" in code or ".isalnum()" in code_lower or ".lower()" in code_lower):
                 keyword_bonus = 10
            elif "parentheses" in desc_lower and ("stack" in code_lower or "append(" in code or "pop(" in code):
                 keyword_bonus = 10
            elif "lru cache" in desc_lower and ("dict" in code_lower or "capacity" in code_lower or "OrderedDict" in code):
                 keyword_bonus = 10
            completeness_score += keyword_bonus

            # Basic control structure bonus
            structure_bonus = 0
            if "if" in code or "for" in code or "while" in code:
                structure_bonus = 10
            # Bonus for edge case handling checks
            if structure_bonus > 0 and ("if not" in code or "if len(" in code or "== 0" in code or "is None" in code):
                 structure_bonus += 5
            completeness_score += min(structure_bonus, 15) # Cap structure bonus

        return min(completeness_score, 100) # Ensure score is between 0 and 100

    def analyze_code_quality_flake8(self, code_string):
        """Analisa a qualidade do código usando Flake8."""
        if not self.flake8_available or not code_string or self.count_code_lines(code_string) == 0:
            #print("Flake8 pulado (não disponível, código vazio ou só comentários)")
            return -1 # Retorna -1 para indicar que a análise não foi feita ou não aplicável

        try:
            # Usar NamedTemporaryFile para garantir que o arquivo seja criado e tenha um nome
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.py', delete=False, encoding='utf-8') as temp_file:
                temp_file.write(code_string)
                temp_filepath = temp_file.name # Guarda o caminho do arquivo temporário

            # Determina o executável Python a ser usado
            python_exe = 'python' if platform.system() == 'Windows' else 'python3'

            # Executa flake8 como módulo para evitar problemas de path
            result = subprocess.run(
                [python_exe, '-m', 'flake8', '--count', temp_filepath],
                capture_output=True, text=True, timeout=15, check=False # Não falha se flake8 retornar != 0
            )

            # Processa a saída
            if result.returncode in [0, 1] and result.stdout: # Sucesso ou encontrou issues
                # A saída do --count é o número de erros na última linha
                try:
                    # Pega a última linha da saída e converte para int
                    count = int(result.stdout.strip().splitlines()[-1])
                    return count
                except (IndexError, ValueError):
                    # Se não conseguir extrair o número, conta as linhas de erro como fallback (menos preciso)
                    return len(result.stdout.strip().splitlines())
            elif result.stderr:
                 # Se houve erro na execução do flake8 em si
                 print(f"Erro na execução do flake8: {result.stderr}")
                 return -1 # Indica erro na análise
            else:
                 # Se não houve stdout e nem stderr (raro, mas possível)
                 return 0 # Assume 0 issues

        except subprocess.TimeoutExpired:
            print("Erro: Análise com Flake8 excedeu o tempo limite (15s).")
            return -1 # Indica timeout
        except Exception as e:
            print(f"Erro inesperado ao executar Flake8: {e}")
            return -1 # Indica erro genérico
        finally:
            # Garante que o arquivo temporário seja removido
            if 'temp_filepath' in locals() and os.path.exists(temp_filepath):
                try:
                    os.remove(temp_filepath)
                except Exception as e_rem:
                    print(f"Aviso: Não foi possível remover o arquivo temporário Flake8: {e_rem}")


    # --- Função evaluate_solution MODIFICADA para incluir execução ---
    def evaluate_solution(self, ai_name, challenge, response):
        """Avalia uma única solução de IA, incluindo execução via executor service."""
        content = response.get("content", "") if isinstance(response, dict) else ""
        api_execution_time = response.get("execution_time", 0) if isinstance(response, dict) else 0 # Renomeado de 'execution_time'

        code = self.extract_code(content)
        lines_of_code = self.count_code_lines(code)
        completeness = self.measure_response_completeness(code, challenge["description"])
        flake8_issues = self.analyze_code_quality_flake8(code) # Retorna -1 se não aplicável/erro

        # --- Lógica de Execução Adicionada ---
        execution_results_list = [] # Armazena resultados de múltiplos casos de teste
        function_name_to_call = None

        # Tenta extrair o nome da função/classe da assinatura (simplificado)
        signature = challenge.get("function_signature", "")
        if signature:
            match_func = re.match(r"def\s+(\w+)\s*\(", signature)
            match_class = re.match(r"class\s+(\w+):", signature)
            if match_func:
                function_name_to_call = match_func.group(1)
            elif match_class:
                 function_name_to_call = challenge["name"] # Usar nome do desafio para classes (requer adaptação no executor ou wrapper)
                 print(f"   Aviso: Teste de execução para classe '{challenge['name']}' não suportado diretamente pelo executor atual. Pulando execução remota.")
            else:
                 # Tenta pegar pelo nome do desafio se assinatura for simples
                 function_name_to_call = challenge["name"].lower().replace(' ', '_').replace('-', '_')


        if code and function_name_to_call and challenge["name"] != "LRU Cache": # Só executa se tiver código, nome da função e não for LRU Cache
            print(f"   Tentando executar código de {ai_name} para '{challenge['name']}' via Executor Service...")
            test_cases = challenge.get("test_cases", [])
            if not test_cases:
                 print(f"   Aviso: Nenhum caso de teste definido para '{challenge['name']}'. Pulando execução.")
            else:
                for i, test_case in enumerate(test_cases):
                    input_args = test_case.get("input_args")
                    expected_output = test_case.get("expected_output") # Guardamos para comparar depois, se quisermos

                    if input_args is None:
                        print(f"   Aviso: Caso de teste {i+1} para '{challenge['name']}' não tem 'input_args'. Pulando.")
                        continue

                    # Chama o serviço executor
                    exec_result = self._call_executor_service(code, function_name_to_call, input_args)

                    # Adiciona informações do caso de teste ao resultado da execução
                    exec_result["test_case_index"] = i
                    exec_result["input_args"] = input_args
                    exec_result["expected_output"] = expected_output # Adiciona o esperado para referência

                    execution_results_list.append(exec_result)

        elif challenge["name"] == "LRU Cache":
             # Adiciona um placeholder indicando que não foi executado
             execution_results_list.append({
                 "success": None, "output": None, "stdout": None,
                 "error": "LRU Cache class execution not directly supported by current executor",
                 "execution_time_ms": -1, "test_case_index": 0, "input_args": None, "expected_output": None
             })
        # --- Fim da Lógica de Execução ---

        # Retorna o dicionário de resultados incluindo a lista de execuções
        return {
            "ai": ai_name,
            "challenge": challenge["name"],
            "difficulty": challenge["difficulty"],
            "api_execution_time": api_execution_time,
            "lines_of_code": lines_of_code,
            "completeness": completeness,
            "flake8_issues": flake8_issues,
            "code": code, # Código extraído
            "full_response": content, # Resposta completa da API
            "execution_results": execution_results_list # Lista com resultados de cada caso de teste executado
        }
    
    # --- Nova Função para Chamar o Executor Service ---
    def _call_executor_service(self, code_string, function_name, input_args):
        """Envia código e argumentos para o executor service e retorna a resposta."""
        executor_url = "https://executor-service-1029404216382.europe-southwest1.run.app/execute" # Sua URL
        payload = {
            "code_string": code_string,
            "function_name": function_name,
            "test_input_args": input_args # input_args já deve ser uma lista
        }
        headers = {"Content-Type": "application/json"}
        default_error_result = {
            "success": False, "output": None, "stdout": None,
            "error": "Executor service call failed", "execution_time_ms": -1 # -1 indica falha na chamada
        }

        try:
            # Definir um timeout razoável (ex: 60 segundos)
            response = requests.post(executor_url, headers=headers, json=payload, timeout=60)

            # Verifica se a resposta foi bem-sucedida (status code 2xx)
            if response.status_code == 200:
                try:
                    # Tenta decodificar o JSON da resposta
                    return response.json()
                except json.JSONDecodeError as json_err:
                    print(f"   Erro: Falha ao decodificar JSON da resposta do executor. Status: {response.status_code}, Erro: {json_err}")
                    default_error_result["error"] = f"Executor response JSON decode error: {json_err}"
                    return default_error_result
            else:
                # Se o status code não for 200 (ex: 4xx, 5xx)
                print(f"   Erro: Chamada ao executor falhou. Status: {response.status_code}, Resposta: {response.text[:200]}...") # Mostra início da resposta
                default_error_result["error"] = f"Executor service returned status {response.status_code}"
                return default_error_result

        except requests.exceptions.Timeout:
            print(f"   Erro: Timeout ao chamar o executor service ({executor_url}).")
            default_error_result["error"] = "Executor service call timed out"
            return default_error_result
        except requests.exceptions.RequestException as req_err:
            print(f"   Erro: Problema de conexão/rede ao chamar o executor service: {req_err}")
            default_error_result["error"] = f"Executor service connection error: {req_err}"
            return default_error_result
        except Exception as e:
            # Captura outros erros inesperados durante a chamada
            print(f"   Erro inesperado ao chamar o executor service: {type(e).__name__} - {e}")
            # traceback.print_exc() # Descomente para debug detalhado
            default_error_result["error"] = f"Unexpected error calling executor: {type(e).__name__}"
            return default_error_result


    # --- Função Principal de Execução ---
    def run_benchmark(self):
        """Executa o benchmark para todas as IAs configuradas."""
        can_run_claude = bool(self.claude_api_key)
        can_run_openai = bool(self.openai_api_key)
        can_run_gemini = bool(self.gemini_api_key)
        enabled_ais = [name for name, can_run in [("Claude", can_run_claude), ("ChatGPT", can_run_openai), ("Gemini", can_run_gemini)] if can_run]

        if not enabled_ais:
            print("ERRO: Nenhuma chave de API válida configurada. Verifique o arquivo .env")
            return

        print(f"Benchmark para: {', '.join(enabled_ais)}")
        print("==========================================")

        for challenge in self.code_challenges:
            signature = challenge.get("function_signature", f"{challenge['name'].lower().replace(' ', '_')}(...)")
            # Ajusta o prompt para ser mais direto sobre a saída esperada (apenas código)
            if "class LRUCache" in signature:
                prompt = (
                    f"Please implement the following Python class based on this description: {challenge['description']}. "
                    f"The class and method signatures should be:\n{signature}\n"
                    f"Provide ONLY the complete Python code for the class definition and necessary imports. "
                    f"Do not include any explanations, example usage, markdown formatting (like ```python), or introductory sentences."
                )
            else:
                prompt = (
                    f"Please write a single, complete Python function that implements the following signature: `{signature}`. "
                    f"The function should solve this problem: {challenge['description']}. "
                    f"Provide ONLY the Python code for the function definition and necessary imports. "
                    f"Do not include any explanations, example usage, markdown formatting (like ```python), or introductory sentences."
                )

            print(f"\n--- Desafio: {challenge['name']} (Dificuldade: {challenge['difficulty']}) ---")
            responses = {}

            # Chama as APIs habilitadas com delays ajustados
            if can_run_claude:
                print("Consultando Claude AI...")
                responses["Claude"] = self.query_claude(prompt)
                time.sleep(1) # Delay para evitar rate limits potenciais

            if can_run_openai:
                print("Consultando ChatGPT...")
                responses["ChatGPT"] = self.query_chatgpt(prompt)
                time.sleep(1) # Delay

            if can_run_gemini:
                print("Consultando Gemini AI...")
                responses["Gemini"] = self.query_gemini(prompt)
                # Aumentado para 31 segundos devido a limites mais estritos da API gratuita (1 req/min)
                # Se você tiver um plano pago, pode reduzir este valor.
                print("Aguardando 31s para o limite da API Gemini...")
                time.sleep(31)

            # Avalia resultados
            for ai_name in enabled_ais:
                 response_data = responses.get(ai_name)
                 # Cria placeholder se a query falhou
                 if response_data is None:
                     response_data = {"content": "", "execution_time": 0} # Tempo API é 0 pois falhou antes

                 result = self.evaluate_solution(ai_name, challenge, response_data)
                 self.results.append(result)

                 # --- Chamada ao Executor Service (a ser adicionada aqui) ---
                 # if result["code"]: # Só tenta executar se houver código
                 #     print(f"   Executando código de {ai_name} para '{challenge['name']}' via Executor Service...")
                 #     for test_case in challenge.get("test_cases", []):
                 #         # Chamar função (a ser criada) para enviar para o executor
                 #         # Ex: exec_data = self.execute_code_via_service(result["code"], ...)
                 #         # Atualizar result["execution_result"] com exec_data
                 #         pass # Placeholder
                 # ---------------------------------------------------------


            print(f"--- Desafio {challenge['name']} concluído ---")

        if not self.results:
            print("\nNenhum resultado coletado.")
            return

        self.generate_report()

    # --- Funções de Geração de Relatório e Visualização ---
    def generate_report(self):
        """Gera um relatório abrangente do benchmark com correção do fillna."""
        if not self.results:
            print("Não há resultados para gerar o relatório.")
            return

        try:
            df = pd.DataFrame(self.results)
            # Separar a coluna aninhada 'execution_result' se ela existir e contiver dados
            if 'execution_result' in df.columns:
                try:
                    exec_results_df = pd.json_normalize(df['execution_result']).add_prefix('exec_')
                    df = pd.concat([df.drop(['execution_result'], axis=1), exec_results_df], axis=1)
                except Exception as e_norm:
                    print(f"Aviso: Não foi possível normalizar 'execution_result': {e_norm}")
                    # Se a normalização falhar, remove a coluna para evitar erros posteriores
                    if 'execution_result' in df.columns:
                       df = df.drop(['execution_result'], axis=1)

        except Exception as e:
            print(f"Erro crítico ao criar DataFrame inicial: {e}\nResultados brutos: {self.results}")
            return

        # Correção Pandas Warning: Usar .loc para fillna
        # Define colunas numéricas esperadas (incluindo as potencialmente extraídas de execution_result)
        expected_numeric_cols = ['api_execution_time', 'lines_of_code', 'completeness', 'flake8_issues',
                                 'exec_success', 'exec_execution_time_ms'] # Adicionar outras se normalizadas

        for col in expected_numeric_cols:
            if col in df.columns: # Verifica se a coluna existe no DataFrame
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if col == 'flake8_issues':
                     df.loc[:, col] = df[col].fillna(-1)
                # Tratar booleano 'exec_success' separadamente se necessário
                elif col == 'exec_success':
                     df.loc[:, col] = df[col].fillna(False).astype(bool) # Assume False se não executado/falhou
                elif col == 'exec_execution_time_ms':
                     df.loc[:, col] = df[col].fillna(-1) # Usar -1 para tempo não executado
                else: # execution_time, lines_of_code, completeness
                     df.loc[:, col] = df[col].fillna(0)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"\nGerando relatórios em '{self.output_dir}/' com timestamp {timestamp}")

        # Salvar JSON com todos os dados brutos (incluindo código e respostas completas)
        json_path = os.path.join(self.output_dir, f"benchmark_results_full_{timestamp}.json")
        try:
            # Converter DataFrame de volta para dicionário para salvar tudo
            # É mais fácil salvar a lista original 'self.results' que já tem tudo
            with open(json_path, "w", encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            print(f"- Resultados detalhados (JSON completo): {json_path}")
        except Exception as e:
            print(f"Erro ao salvar JSON completo: {e}")

        # Salvar Resumo CSV (sem código e resposta completa)
        summary_path = os.path.join(self.output_dir, f"benchmark_summary_{timestamp}.csv")
        try:
             # Seleciona colunas para o resumo (exclui as colunas grandes)
             cols_to_drop = ['code', 'full_response'] # Adicionar outras se não desejadas no resumo
             summary_cols = [col for col in df.columns if col not in cols_to_drop]
             summary = df[summary_cols]
             summary.to_csv(summary_path, index=False, encoding='utf-8')
             print(f"- Resumo salvo (CSV): {summary_path}")
        except Exception as e:
            print(f"Erro ao salvar resumo CSV: {e}")

        # Calcular e Salvar Estatísticas Agregadas
        stats_path = os.path.join(self.output_dir, f"benchmark_stats_{timestamp}.csv")
        stats = None
        try:
            stats_metrics = [
                "Average API Time (s)",
                "Average Lines of Code",
                "Average Completeness (%)",
                "Average Flake8 Issues",
                # Adicionar métricas de execução se disponíveis
                #"Average Execution Success (%)",
                #"Average Execution Time (ms)"
            ]
            # Incluir métricas de execução se as colunas existirem após normalização
            if 'exec_success' in df.columns: stats_metrics.append("Average Execution Success (%)")
            if 'exec_execution_time_ms' in df.columns: stats_metrics.append("Average Execution Time (ms)")

            stats_data = {"Metric": stats_metrics}
            ai_list = sorted(df['ai'].unique())

            for ai_name in ai_list:
                df_ai = df[df["ai"] == ai_name]
                # Flake8 (ignora -1)
                df_ai_valid_flake8 = df_ai[df_ai["flake8_issues"] != -1]
                avg_flake8 = df_ai_valid_flake8["flake8_issues"].mean() if not df_ai_valid_flake8.empty else -1
                # Outras métricas estáticas
                avg_api_time = df_ai["api_execution_time"].mean()
                avg_loc = df_ai["lines_of_code"].mean()
                avg_comp = df_ai["completeness"].mean()

                ai_column_data = [avg_api_time, avg_loc, avg_comp, avg_flake8]

                # Métricas de execução (se disponíveis)
                if 'exec_success' in df.columns:
                    avg_exec_success = df_ai["exec_success"].mean() * 100 # Em porcentagem
                    ai_column_data.append(avg_exec_success)
                if 'exec_execution_time_ms' in df.columns:
                    df_ai_valid_exec_time = df_ai[df_ai["exec_execution_time_ms"] != -1] # Ignora não executado
                    avg_exec_time = df_ai_valid_exec_time["exec_execution_time_ms"].mean() if not df_ai_valid_exec_time.empty else -1
                    ai_column_data.append(avg_exec_time)

                stats_data[ai_name] = ai_column_data

            stats = pd.DataFrame(stats_data)
            stats.to_csv(stats_path, index=False, encoding='utf-8')
            print(f"- Estatísticas salvas (CSV): {stats_path}")
        except Exception as e:
            print(f"Erro ao calcular/salvar estatísticas: {e}")
            # traceback.print_exc() # Para debug

        # Gerar Visualizações
        try:
            self.create_visualizations(df, timestamp, self.output_dir) # Passa o DataFrame processado
            print(f"- Visualizações salvas como PNG em '{self.output_dir}/'")
        except Exception as e:
            print(f"Erro ao gerar visualizações: {e}")
            # traceback.print_exc() # Para debug

        # Imprimir Sumário no Console
        print("\n===== SUMÁRIO DO BENCHMARK =====");
        if stats is not None:
             stats_display = stats.copy()
             # Formatação melhorada
             for col in stats_display.columns:
                 if col != 'Metric':
                     for i in stats_display.index:
                         metric_name = stats_display.loc[i, 'Metric']
                         value = stats_display.loc[i, col]
                         if pd.isna(value):
                             stats_display.loc[i, col] = "N/A"
                         elif "Flake8" in metric_name:
                             stats_display.loc[i, col] = f"{value:.1f}" if value != -1 else "N/A (No Code/Error)"
                         elif "Time (ms)" in metric_name:
                              stats_display.loc[i, col] = f"{value:.1f}" if value != -1 else "N/A (Not Run)"
                         elif "Success (%)" in metric_name:
                              stats_display.loc[i, col] = f"{value:.1f}%"
                         elif "Time (s)" in metric_name:
                              stats_display.loc[i, col] = f"{value:.2f}"
                         else: # LOC, Completeness
                              stats_display.loc[i, col] = f"{value:.1f}"

             print(stats_display.to_string(index=False))
        else:
            print("Não foi possível gerar o sumário de estatísticas.")


    def create_visualizations(self, df, timestamp, output_dir):
        """Gera visualizações dos resultados do benchmark."""
        plt.style.use('ggplot')
        ai_list = sorted(df['ai'].unique())
        num_ais = len(ai_list)

        # Define mapa de cores
        if num_ais <= 10:
            colors = plt.cm.tab10(np.linspace(0, 1, num_ais))
        else: # Mais de 10 IAs, usa um colormap diferente
            colors = plt.cm.viridis(np.linspace(0, 1, num_ais))
        ai_color_map = {ai: colors[i] for i, ai in enumerate(ai_list)}

        # Verifica colunas necessárias (incluindo as de execução se existirem)
        base_required_cols = ['challenge', 'ai', 'api_execution_time', 'lines_of_code', 'completeness', 'flake8_issues', 'difficulty']
        exec_cols = [col for col in ['exec_success', 'exec_execution_time_ms'] if col in df.columns]
        required_cols = base_required_cols + exec_cols

        if df.empty or not all(col in df.columns for col in required_cols):
            print("AVISO: DataFrame vazio ou faltando colunas para gerar gráficos.")
            return

        # Ordena desafios por dificuldade
        difficulty_order = pd.CategoricalDtype(['Easy', 'Medium', 'Hard'], ordered=True)
        df['difficulty_cat'] = df['difficulty'].astype(difficulty_order)
        # Obtém a ordem dos desafios baseada na dificuldade do DataFrame
        challenge_order = df.sort_values('difficulty_cat')['challenge'].unique()

        # DataFrame para plotagem (cópia)
        df_plot = df.copy()

        # Métricas e títulos para gráficos de barras
        metrics_to_plot = {
            "1_api_execution_time": ("Tempo Médio de Resposta API", "Tempo (s)"),
            "2_lines_of_code": ("Média Linhas de Código (LoC)", "Número de Linhas"),
            "3_completeness": ("Completude Média da Solução", "Score (%)"),
            "4_flake8_issues": ("Média Problemas Flake8", "Número de Problemas"),
            # Adiciona gráficos de execução se os dados existirem
        }
        if 'exec_success' in exec_cols:
            metrics_to_plot["5_exec_success"] = ("Taxa Média de Sucesso na Execução", "Sucesso (%)")
        if 'exec_execution_time_ms' in exec_cols:
             metrics_to_plot["6_exec_execution_time_ms"] = ("Tempo Médio de Execução (Executor)", "Tempo (ms)")


        # Gera gráficos de barras
        for metric_key, (title, ylabel) in metrics_to_plot.items():
            col_name = metric_key.split('_', 1)[1]

            # Prepara dados específicos (ignora flake8=-1, exec_time=-1)
            if col_name == 'flake8_issues':
                plot_data = df_plot[df_plot['flake8_issues'] != -1]
            elif col_name == 'exec_execution_time_ms':
                plot_data = df_plot[df_plot['exec_execution_time_ms'] != -1]
            elif col_name == 'exec_success':
                 # Converte True/False para 100/0 para média
                 plot_data = df_plot.copy()
                 if 'exec_success' in plot_data.columns:
                      plot_data[col_name] = plot_data[col_name].astype(float) * 100
                 else: continue # Pula se a coluna não existir
            else:
                plot_data = df_plot

            if plot_data.empty:
                print(f"Pulando gráfico '{title}': Nenhum dado válido encontrado.")
                continue

            try:
                plt.figure()
                pivot = plot_data.pivot_table(index='challenge', columns='ai', values=col_name, aggfunc='mean')

                # Garante a ordem correta dos desafios e IAs
                pivot = pivot.reindex(challenge_order, axis=0).reindex(ai_list, axis=1)

                # Preenche NaNs para plotagem (mas usa NaN para flake8/exec_time para não plotar barra)
                fill_value = 0
                if col_name in ['flake8_issues', 'exec_execution_time_ms']:
                     fill_value = np.nan
                pivot.fillna(fill_value, inplace=True)

                # Plotagem
                ax = pivot.plot(kind='bar', figsize=(max(12, num_ais * 2.5), 7),
                                color=[ai_color_map.get(ai, 'grey') for ai in pivot.columns],
                                width=0.8) # Ajusta largura das barras

                plt.title(f'{title} por Desafio'); plt.ylabel(ylabel); plt.xlabel('Desafio (Ordenado por Dificuldade)');
                plt.xticks(rotation=45, ha='right', fontsize=9) # Ajusta tamanho da fonte
                plt.yticks(fontsize=9)

                # Ajustes de limite Y específicos
                if col_name == 'completeness' or col_name == 'exec_success': plt.ylim(0, 105)
                if col_name == 'flake8_issues' and not pivot.isnull().all().all(): # Só ajusta se houver dados
                    max_val = pivot.max().max()
                    if max_val <= 1: plt.ylim(bottom=-0.1) # Melhora visualização para poucos erros
                if col_name == 'exec_execution_time_ms' and not pivot.isnull().all().all():
                     plt.ylim(bottom=0) # Garante que tempo não seja negativo

                plt.legend(title='IA', bbox_to_anchor=(1.02, 1), loc='upper left');
                plt.tight_layout(rect=[0, 0, 0.88, 1]) # Ajusta rect para caber a legenda

                path = os.path.join(output_dir, f"{metric_key}_{timestamp}.png")
                plt.savefig(path)
                plt.close()
            except Exception as e:
                print(f"Erro ao gerar gráfico de barras '{title}': {e}")
                # traceback.print_exc() # Para debug

        # Gráfico Radar (se aplicável e se houver dados de execução)
        # Normaliza 3 métricas chave: Velocidade API (inversa), Conciseness (inversa LoC), Completude
        # Se houver dados de execução, podemos pensar em outro radar ou substituir métricas
        if num_ais >= 2:
            print(f"Gerando gráfico Radar para {num_ais} IAs (3 métricas base)...")
            try:
                # Usamos as métricas base por enquanto
                numeric_cols_radar = ['api_execution_time', 'lines_of_code', 'completeness']
                means_data = {ai: df[df["ai"] == ai][numeric_cols_radar].mean() for ai in ai_list}

                # Normalização (igual antes, usando .get para segurança)
                all_times = [m.get("api_execution_time", 0) for m in means_data.values()]
                all_locs = [m.get("lines_of_code", 0) for m in means_data.values()]
                all_comps = [m.get("completeness", 0) for m in means_data.values()]

                # Tempo API (Menor é melhor -> Inverter para Score)
                max_time = max(all_times + [1e-6]) # Evita divisão por zero
                time_norm = {ai: 100 * (1 - means_data[ai].get("api_execution_time", 0) / max_time) if max_time > 0 else 0 for ai in ai_list}

                # LoC (Menor é melhor -> Inverter para Score, com cuidado para LoC=0)
                valid_locs = [loc for loc in all_locs if loc > 0]
                min_loc_valid = min(valid_locs) if valid_locs else 0
                max_loc_valid = max(all_locs + [1e-6]) # Evita divisão por zero
                lines_norm = {}
                for ai in ai_list:
                    loc = means_data[ai].get("lines_of_code", 0)
                    if loc <= 0: # Se não gerou código, score é 0
                        lines_norm[ai] = 0
                    elif min_loc_valid > 0 and max_loc_valid > min_loc_valid: # Normalização invertida
                        # (loc - min) / (max - min) -> dá 0 pro melhor, 1 pro pior
                        # 1 - (...) -> dá 1 pro melhor, 0 pro pior
                        norm_val = 100 * (1 - (loc - min_loc_valid) / (max_loc_valid - min_loc_valid))
                        lines_norm[ai] = max(0, min(100, norm_val)) # Garante 0-100
                    else: # Caso onde todos > 0 mas min == max (ou só tem 1 AI com código)
                        lines_norm[ai] = 100 if loc > 0 else 0 # Melhor possível se gerou código
                    # Garantia extra se por acaso loc == min_loc_valid (deve ser 100)
                    if loc == min_loc_valid and min_loc_valid > 0: lines_norm[ai] = 100

                # Completude (Maior é melhor - já está em 0-100)
                comp_norm = {ai: means_data[ai].get("completeness", 0) for ai in ai_list}


                categories = ['Velocidade API\n(Norm)', 'Conciseness LoC\n(Norm)', 'Completude\n(%)']
                N = len(categories)
                angles = [n / float(N) * 2 * np.pi for n in range(N)]
                angles += angles[:1] # Fecha o círculo

                fig = plt.figure(figsize=(9, 9))
                ax = fig.add_subplot(111, polar=True)

                for ai_name in ai_list:
                    # Usa .get(key, 0) para evitar erro se alguma média falhar
                    values = [time_norm.get(ai_name, 0), lines_norm.get(ai_name, 0), comp_norm.get(ai_name, 0)]
                    plot_values = values + values[:1] # Fecha o círculo
                    color = ai_color_map.get(ai_name, 'grey')
                    ax.plot(angles, plot_values, 'o-', linewidth=2, label=ai_name, color=color)
                    ax.fill(angles, plot_values, color, alpha=0.2)

                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(categories)
                ax.set_yticks(np.arange(0, 101, 20)) # Escala de 0 a 100
                ax.set_ylim(0, 100)

                plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15))
                plt.title('Comparação Geral (3 Métricas Normalizadas)', size=14, y=1.18)
                plt.tight_layout()
                path = os.path.join(output_dir, f"5_radar_chart_{timestamp}.png")
                plt.savefig(path)
                plt.close()
            except Exception as e:
                print(f"Erro ao gerar gráfico Radar: {e}")
                # traceback.print_exc() # Para debug
        else:
            print("Gráfico Radar pulado (requer >= 2 IAs).")


# --- Ponto de Entrada Principal ---
if __name__ == "__main__":
    print("===========================================================")
    print(" AI Programming Benchmark: Claude vs ChatGPT vs Gemini")
    print("             (Incluindo Análise de Qualidade Flake8)")
    print("===========================================================")
    # Verifica se o .env existe, apenas informativo
    if not os.path.exists(".env"):
        print("AVISO: Arquivo .env não encontrado. Chaves API devem ser variáveis de ambiente.")

    benchmark = AIBenchmark()
    # Verifica se alguma AI pode rodar antes de iniciar
    if benchmark.claude_api_key or benchmark.openai_api_key or benchmark.gemini_api_key:
         benchmark.run_benchmark()
         print("\nBenchmark concluído.")
         print(f"Relatórios salvos em: '{benchmark.output_dir}'")
    else:
         print("\nERRO FATAL: Nenhuma chave de API configurada (via .env ou variáveis de ambiente). Benchmark não pode ser executado.")

    print("===========================================================")
