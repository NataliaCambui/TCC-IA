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
    corretude (via execução) e desempenho.
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
        self.executor_url = "https://executor-service-1029404216382.europe-southwest1.run.app/execute" # ATENÇÃO: Risco de segurança inerente

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
                self.gemini_api_key = None # Garante que a chave seja None se a configuração falhar
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
            "Libraries": {}
        }
        libs_to_check = ['pandas', 'numpy', 'matplotlib', 'requests',
                         'python-dotenv', 'google-generativeai', 'google-api-core']
        try:
            # Tenta usar importlib.metadata (Python 3.8+)
            for lib in libs_to_check:
                try:
                    info["Libraries"][lib] = importlib_metadata.version(lib)
                except importlib_metadata.PackageNotFoundError:
                    info["Libraries"][lib] = "Not Found"
        except NameError: # Fallback para pkg_resources
            for lib in libs_to_check:
                try:
                    info["Libraries"][lib] = pkg_resources.get_distribution(lib).version
                # Pylance pode reportar um erro aqui devido à complexidade da exceção,
                # mas o código deve funcionar se pkg_resources estiver instalado.
                except pkg_resources.DistributionNotFound:
                     info["Libraries"][lib] = "Not Found"
                # Captura exceção genérica para outros erros ao obter versão
                except Exception:
                     info["Libraries"][lib] = "Error getting version"
        return info

    def _check_flake8(self):
        """Verifica se o comando flake8 está disponível no sistema."""
        try:
            # Usa sys.executable para garantir que está a usar o python correto
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
        """
        Envia um prompt para a API do Claude e retorna a resposta e o tempo de execução.

        Args:
            prompt: O prompt a ser enviado.

        Returns:
            Um dicionário com 'content' (str) e 'execution_time' (float).
        """
        start_time = time.time()
        content = ""
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
                    # Pylance pode reportar erro aqui, mas a sintaxe f-string é válida.
                    error_message += f" | Body: {e.response.json()}"
                except json.JSONDecodeError:
                     # Pylance pode reportar erro aqui, mas a sintaxe f-string é válida.
                    error_message += f" | Body: {e.response.text}"
            print(error_message)
            execution_time = time.time() - start_time
            return {"content": "", "execution_time": execution_time}
        except Exception as e:
            print(f"Erro inesperado consulta Claude: {type(e).__name__} - {e}")
            execution_time = time.time() - start_time
            return {"content": "", "execution_time": execution_time}

    def query_chatgpt(self, prompt: str) -> dict:
        """
        Envia um prompt para a API do OpenAI (ChatGPT) e retorna a resposta e o tempo de execução.

        Args:
            prompt: O prompt a ser enviado.

        Returns:
            Um dicionário com 'content' (str) e 'execution_time' (float).
        """
        start_time = time.time()
        content = ""
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
        """
        Envia um prompt para a API do Gemini e retorna a resposta e o tempo de execução.

        Args:
            prompt: O prompt a ser enviado.

        Returns:
            Um dicionário com 'content' (str) e 'execution_time' (float).
        """
        start_time = time.time()
        content = ""
        # Verifica se a API foi configurada com sucesso no __init__
        if not self.gemini_api_key or not self.gemini_model:
            print("Pulando Gemini: Chave API não configurada ou modelo inválido.")
            execution_time = time.time() - start_time
            return {"content": "", "execution_time": execution_time}

        try:
            model = genai.GenerativeModel(self.gemini_model)
            # Configurações de segurança para tentar evitar bloqueios
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
                # A resposta pode não ter 'text' se for bloqueada
                print("Resposta Gemini não contém texto (possivelmente bloqueada).")
                try:
                    print(f"  Prompt Feedback: {response.prompt_feedback}")
                    if hasattr(response, 'candidates') and response.candidates:
                        print(f"  Finish Reason: {response.candidates[0].finish_reason}")
                except Exception as feedback_error:
                    print(f"  Erro ao acessar feedback Gemini: {feedback_error}")
            return {"content": content or "", "execution_time": execution_time}

        except google.api_core.exceptions.ResourceExhausted as e:
            print(f"Erro consulta Gemini (ResourceExhausted - Rate Limit): {e}\n"
                  "                >> Atingido o limite de requisições da API Gemini."
                  " Tente novamente mais tarde.")
            execution_time = time.time() - start_time
            return {"content": "", "execution_time": execution_time}
        except Exception as e:
            print(f"Erro inesperado consulta Gemini: {type(e).__name__} - {e}")
            execution_time = time.time() - start_time
            return {"content": "", "execution_time": execution_time}

    def extract_code(self, content: str) -> str:
        """
        Extrai o primeiro bloco de código Python de uma string de texto.
        Prioriza blocos ```python, depois ``` genéricos, e por fim tenta
        identificar código baseado em indentação e palavras-chave 'def'/'class'.

        Args:
            content: A string contendo potencialmente código.

        Returns:
            O bloco de código extraído (sem os marcadores ```) ou uma string vazia.
        """
        if not content:
            return ""

        # 1. Procura por ```python ... ```
        code_blocks_python = re.findall(r'```python\n(.*?)\n```', content, re.DOTALL)
        if code_blocks_python:
            return code_blocks_python[0].strip()

        # 2. Procura por ``` ... ``` genérico
        code_blocks_generic = re.findall(r'```(?:.*?\n)?(.*?)\n```', content, re.DOTALL)
        if code_blocks_generic:
            return code_blocks_generic[0].strip()

        # 3. Fallback: Tenta encontrar código por indentação/palavras-chave
        lines = content.split('\n')
        code_lines = []
        start_found = False
        for line in lines:
            stripped = line.strip()
            # Começa a capturar se a linha for indentada ou começar com def/class
            if not start_found and (line.startswith(' ') or line.startswith('\t') or
                                     stripped.startswith('def ') or stripped.startswith('class ')):
                start_found = True
            if start_found:
                code_lines.append(line)

        if code_lines:
            return '\n'.join(code_lines).strip()

        # 4. Último fallback: Se for curto e não parecer código, retorna vazio. Senão, retorna tudo.
        if len(content.splitlines()) < 3 and ("def " not in content and "class " not in content):
            # Considera que respostas muito curtas sem def/class não são código útil
            print("AVISO: extract_code retornando vazio (conteúdo curto sem def/class).")
            return ""
        else:
            # Pode ser código sem markdown ou um erro da IA, retorna o conteúdo
            print("AVISO: extract_code usando fallback final (retornando conteúdo original).")
            return content.strip()

    def count_code_lines(self, code: str) -> int:
        """Conta as linhas de código 'reais', ignorando vazias e comentários."""
        if not code:
            return 0
        return len([line for line in code.splitlines()
                    if line.strip() and not line.strip().startswith('#')])

    def measure_response_completeness(self, code: str, description: str) -> int:
        """
        Estima a 'completude' da resposta da IA baseada na presença de código,
        estruturas esperadas e palavras-chave relacionadas à descrição.
        Retorna um score de 0 a 100.
        """
        completeness_score = 0
        lines_count = self.count_code_lines(code)

        if lines_count > 0:
            completeness_score += 50 # Pontuação base por ter código

            # Bónus por estruturas fundamentais
            if "def " in code or "class " in code:
                completeness_score += 10
            if "return " in code or "yield " in code:
                completeness_score += 10
            # Caso específico para 'Count Islands' onde pode não haver return explícito
            elif "island" in description.lower() and ("dfs(" in code or "bfs(" in code):
                 completeness_score += 5 # Bónus menor

            # Bónus por palavras-chave/lógica específica do desafio
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

            # Bónus por estruturas de controle/lógica
            structure_bonus = 0
            if "if" in code or "for" in code or "while" in code:
                structure_bonus = 10
            # Bónus extra por tratamento de casos base/vazios
            if structure_bonus > 0 and ("if not" in code or "if len(" in code or
                                        "== 0" in code or "is None" in code):
                structure_bonus += 5
            completeness_score += min(structure_bonus, 15) # Limita o bónus de estrutura

        return min(completeness_score, 100) # Garante que o score não passe de 100

    def analyze_code_quality_flake8(self, code_string: str) -> int:
        """
        Analisa uma string de código Python usando Flake8 e retorna o número de problemas.

        Args:
            code_string: O código Python a ser analisado.

        Returns:
            O número de problemas reportados pelo Flake8, ou -1 se o Flake8
            não estiver disponível, o código for vazio, ou ocorrer um erro.
        """
        if not self.flake8_available or not code_string or self.count_code_lines(code_string) == 0:
            return -1

        temp_filepath = None # Inicializa para garantir que existe no finally
        try:
            # Cria um arquivo temporário para o flake8 analisar
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.py', delete=False,
                                             encoding='utf-8') as temp_file:
                temp_file.write(code_string)
                temp_filepath = temp_file.name # Guarda o caminho

            # Usa sys.executable para garantir o python correto
            python_exe = sys.executable
            # Executa o flake8 como módulo
            result = subprocess.run(
                [python_exe, '-m', 'flake8', '--count', temp_filepath],
                capture_output=True, text=True, timeout=15, check=False # check=False para pegar o output mesmo com erros
            )

            # Flake8 retorna 0 se não há erros, 1 se há erros.
            # Queremos o contador no stdout.
            if result.returncode in [0, 1] and result.stdout:
                try:
                    # O contador costuma ser a última linha do stdout
                    count = int(result.stdout.strip().splitlines()[-1])
                    return count
                except (IndexError, ValueError):
                    # Fallback: Se não conseguir parsear o número, conta as linhas de output como aproximação
                    return len(result.stdout.strip().splitlines())
            elif result.stderr:
                # Erro na execução do flake8 em si
                print(f"Erro na execução do flake8: {result.stderr}")
                return -1
            else:
                # Sem erros e sem output (código perfeito ou erro inesperado)
                return 0 # Assume 0 problemas se o exit code foi 0

        except subprocess.TimeoutExpired:
            print("Erro: Análise com Flake8 excedeu o tempo limite (15s).")
            return -1
        except Exception as e:
            print(f"Erro inesperado ao executar Flake8: {e}")
            return -1
        finally:
            # Garante a remoção do arquivo temporário
            if temp_filepath and os.path.exists(temp_filepath):
                try:
                    os.remove(temp_filepath)
                except Exception as e_rem:
                    print(f"Aviso: Não foi possível remover o arquivo temporário Flake8: {e_rem}")

    def evaluate_solution(self, ai_name: str, challenge: dict, response: dict) -> dict:
        """
        Avalia a solução de uma IA para um desafio específico.
        Extrai o código, analisa qualidade (Flake8), mede completude,
        e tenta executar os casos de teste usando o Executor Service.

        Args:
            ai_name: Nome da IA (e.g., "Claude").
            challenge: Dicionário contendo os detalhes do desafio.
            response: Dicionário da resposta da API da IA {'content': str, 'execution_time': float}.

        Returns:
            Um dicionário com os resultados da avaliação.
        """
        content = response.get("content", "") if isinstance(response, dict) else ""
        api_execution_time = response.get("execution_time", 0) if isinstance(response, dict) else 0

        code = self.extract_code(content)
        lines_of_code = self.count_code_lines(code)
        completeness = self.measure_response_completeness(code, challenge["description"])
        flake8_issues = self.analyze_code_quality_flake8(code)

        execution_results_list = []
        processed_test_details = []
        num_execution_errors = 0
        first_execution_error = ""
        actual_function_name = self._detect_function_name(code, challenge['name'])

        if code and actual_function_name:
            if challenge["name"] == "LRU Cache":
                # Lida com o caso especial LRU Cache (não executável pelo serviço atual)
                error_msg = "LRU Cache class execution not directly supported by current executor"
                execution_results_list.append({
                    "success": False, "output": None, "stdout": None, "error": error_msg,
                    "execution_time_ms": -1, "test_case_index": 0,
                    "input_args": None, "expected_output": None
                })
                num_execution_errors = 1
                first_execution_error = error_msg
                processed_test_details.append({
                    "ai": ai_name, "challenge": challenge["name"], "difficulty": challenge["difficulty"],
                    "test_case_index": 0, "input_args_str": "N/A (LRU)",
                    "expected_output_str": "N/A (LRU)", "actual_output_str": None,
                    "status": "Not Executed", "execution_time_ms": -1,
                    "error_message": error_msg, "stdout": None
                })
            else:
                # Executa os testes para desafios baseados em funções
                print(f"  Tentando executar código de {ai_name} para '{challenge['name']}' "
                      f"(função: {actual_function_name}) via Executor Service...")
                execution_results_list, processed_test_details, num_execution_errors, first_execution_error = \
                    self._run_test_cases(ai_name, challenge, code, actual_function_name)
        else:
            print(f"  AVISO: Nenhum código válido ou nome de função detectado para "
                  f"'{challenge['name']}' por {ai_name}. Pulando execução.")
            # Adiciona um resultado vazio/não executado se não houver código
            num_tests = len(challenge.get("test_cases", [])) if challenge["name"] != "LRU Cache" else 1
            for i in range(num_tests):
                 processed_test_details.append({
                     "ai": ai_name, "challenge": challenge["name"], "difficulty": challenge["difficulty"],
                     "test_case_index": i, "input_args_str": None, "expected_output_str": None,
                     "actual_output_str": None, "status": "No Code", "execution_time_ms": -1,
                     "error_message": "No code extracted or function name detected", "stdout": None
                 })

        self.detailed_test_results.extend(processed_test_details)

        return {
            "ai": ai_name,
            "challenge": challenge["name"],
            "difficulty": challenge["difficulty"],
            "api_execution_time": api_execution_time,
            "lines_of_code": lines_of_code,
            "completeness": completeness,
            "flake8_issues": flake8_issues,
            "num_execution_errors": num_execution_errors,
            "first_execution_error": first_execution_error,
            "code": code, # Código extraído
            "full_response": content, # Resposta completa da API
            "execution_results_raw": execution_results_list # Resultados brutos do executor
        }

    def _detect_function_name(self, code: str, challenge_name: str) -> str | None:
        """Detecta o nome da função ou classe principal no código."""
        if not code:
            return None
        match = re.search(r"^\s*(?:def|class)\s+(\w+)", code, re.MULTILINE)
        if match:
            name = match.group(1)
            print(f"  Nome da função/classe detectado no código: '{name}'")
            return name
        else:
            print(f"  AVISO: Não foi possível detectar nome da função/classe no código "
                  f"extraído para '{challenge_name}'.")
            return None

    def _run_test_cases(self, ai_name: str, challenge: dict, code: str,
                        actual_function_name: str) -> tuple[list, list, int, str]:
        """Executa os casos de teste para um desafio usando o Executor Service."""
        execution_results_list = []
        processed_test_details = []
        num_execution_errors = 0
        first_execution_error = ""
        test_cases = challenge.get("test_cases", [])

        if not test_cases:
            print(f"  Aviso: Nenhum caso de teste definido para '{challenge['name']}'.")
            return [], [], 0, ""

        for i, test_case in enumerate(test_cases):
            input_args = test_case.get("input_args")
            expected_output = test_case.get("expected_output")
            if input_args is None:
                print(f"  Aviso: Caso de teste {i+1} para '{challenge['name']}' "
                      "não tem 'input_args'. Pulando.")
                continue

            exec_result = self._call_executor_service(code, actual_function_name, input_args)
            execution_results_list.append(exec_result) # Guarda resultado bruto

            # Prepara detalhes para o relatório CSV detalhado
            test_detail = {
                "ai": ai_name, "challenge": challenge["name"], "difficulty": challenge["difficulty"],
                "test_case_index": i, "input_args_str": json.dumps(input_args),
                "expected_output_str": json.dumps(expected_output),
                "actual_output_str": None, "status": "Unknown", # Status inicial
                "execution_time_ms": exec_result.get("execution_time_ms", -1),
                "error_message": None, "stdout": exec_result.get("stdout")
            }

            if exec_result.get("success") is True:
                actual_output = exec_result.get("output")
                test_detail["actual_output_str"] = json.dumps(actual_output)
                try:
                    # Comparação: Usa np.array_equal para listas/arrays simples.
                    # Pode precisar de ajuste para tipos mais complexos ou floats.
                    are_equal = np.array_equal(
                        np.array(actual_output, dtype=object),
                        np.array(expected_output, dtype=object)
                    )
                    test_detail["status"] = "Pass" if are_equal else "Fail"
                except Exception as compare_err:
                    test_detail["status"] = "Fail (Comparison Error)"
                    test_detail["error_message"] = f"Error comparing outputs: {compare_err}"
            elif exec_result.get("success") is False:
                test_detail["status"] = "Error"
                test_detail["error_message"] = exec_result.get("error", "Unknown execution error")
                num_execution_errors += 1
                if not first_execution_error and test_detail["error_message"]:
                    first_execution_error = str(test_detail["error_message"])[:200] # Limita tamanho
            else:
                # Caso inesperado onde 'success' não é True nem False
                 test_detail["status"] = "Unknown"
                 test_detail["error_message"] = exec_result.get("error", "Executor did not return success status")
                 num_execution_errors += 1
                 if not first_execution_error and test_detail["error_message"]:
                     first_execution_error = str(test_detail["error_message"])[:200] # Limita tamanho

            processed_test_details.append(test_detail)

        return execution_results_list, processed_test_details, num_execution_errors, first_execution_error

    def _call_executor_service(self, code_string: str, function_name: str,
                               input_args: list) -> dict:
        """
        Chama o serviço externo para executar o código com os argumentos fornecidos.

        ATENÇÃO: Executar código (especialmente de IAs) num serviço externo
        apresenta riscos de segurança. Certifique-se que o serviço executor
        está devidamente isolado (sandboxed).

        Args:
            code_string: O código Python a executar (geralmente uma função).
            function_name: O nome da função a ser chamada.
            input_args: Uma lista de argumentos para passar à função.

        Returns:
            Um dicionário com o resultado da execução ou detalhes do erro.
        """
        # Adiciona imports comuns que as IAs podem omitir
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
                    print(f"  Erro: Falha ao decodificar JSON da resposta do executor. "
                          f"Status: {response.status_code}, Erro: {json_err}")
                    default_error_result["error"] = f"Executor response JSON decode error: {json_err}"
                    return default_error_result
            else:
                error_text = response.text[:200] # Limita o tamanho do texto do erro
                print(f"  Erro: Chamada ao executor falhou. Status: {response.status_code}, "
                      f"Resposta: {error_text}...")
                default_error_result["error"] = (f"Executor service returned status "
                                                 f"{response.status_code}: {error_text}")
                return default_error_result

        except requests.exceptions.Timeout:
            print(f"  Erro: Timeout ao chamar o executor service ({self.executor_url}).")
            default_error_result["error"] = "Executor service call timed out"
            return default_error_result
        except requests.exceptions.RequestException as req_err:
            print(f"  Erro: Problema de conexão/rede ao chamar o executor service: {req_err}")
            default_error_result["error"] = f"Executor service connection error: {req_err}"
            return default_error_result
        except Exception as e:
            print(f"  Erro inesperado ao chamar o executor service: {type(e).__name__} - {e}")
            default_error_result["error"] = f"Unexpected error calling executor: {type(e).__name__}"
            return default_error_result

    def run_benchmark(self):
        """Executa o ciclo completo do benchmark para todas as IAs e desafios."""
        can_run_claude = bool(self.claude_api_key)
        can_run_openai = bool(self.openai_api_key)
        # Verifica se a chave Gemini ainda é válida após a tentativa de configuração
        can_run_gemini = bool(self.gemini_api_key and self.gemini_model)
        enabled_ais = [name for name, can_run in [
                       ("Claude", can_run_claude),
                       ("ChatGPT", can_run_openai),
                       ("Gemini", can_run_gemini)] if can_run]

        if not enabled_ais:
            print("ERRO: Nenhuma chave de API válida configurada. Benchmark não pode ser executado.")
            return

        print(f"Iniciando Benchmark para: {', '.join(enabled_ais)}")
        print(f"  Claude Model : {self.claude_model if can_run_claude else 'N/A (No Key)'}")
        print(f"  ChatGPT Model: {self.openai_model if can_run_openai else 'N/A (No Key)'}")
        print(f"  Gemini Model : {self.gemini_model if can_run_gemini else 'N/A (No Key/Config Error)'}")
        print("==========================================")

        # Descrições detalhadas dos desafios (mantido igual)
        challenge_details = {
           "Fibonacci Sequence": "Técnica: Requer lógica iterativa ou recursiva simples.\n  Finalidade: Avaliar a capacidade de implementar algoritmos básicos, lidar com sequências e casos base (n=0, n=1).",
           "Binary Search": "Técnica: Busca eficiente em lista ordenada (dividir para conquistar).\n  Finalidade: Avaliar a capacidade de implementar busca binária, lidar com índices, condições de parada e casos onde o elemento não é encontrado.",
           "Merge Sort": "Técnica: Ordenação eficiente usando \"dividir para conquistar\" (recursão e merge).\n  Finalidade: Avaliar a capacidade de implementar algoritmos de ordenação recursivos, especificamente a lógica de divisão e a de intercalação (merge) de sub-listas ordenadas.",
           "Count Islands in Grid": "Técnica: Travessia de grafo (Grid 2D) usando Busca em Profundidade (DFS) ou Busca em Largura (BFS).\n  Finalidade: Avaliar a capacidade de modelar um problema como travessia de grafo, implementar DFS/BFS, lidar com matrizes 2D e evitar recontagem (marcando visitados).",
           "Palindrome Check": "Técnica: Manipulação de strings (limpeza, inversão, comparação).\n  Finalidade: Avaliar a capacidade de processar strings, ignorar caracteres não alfanuméricos e maiúsculas/minúsculas, e verificar simetria.",
           "Valid Parentheses": "Técnica: Uso de estrutura de dados Pilha (Stack) para validação de ordem.\n  Finalidade: Avaliar a capacidade de usar pilhas para rastrear parênteses/colchetes/chaves abertos e validar o fechamento correto e na ordem apropriada.",
           "LRU Cache": "Técnica: Implementação de estrutura de dados customizada (cache) com política de remoção LRU, frequentemente usando dicionários e/ou listas ligadas (ou OrderedDict).\n  Finalidade: Avaliar a capacidade de projetar e implementar estruturas de dados mais complexas, gerenciar capacidade, implementar lógica de get e put com eficiência de tempo (idealmente O(1)), e lidar com a política de despejo LRU."
        }

        for challenge in self.code_challenges:
            signature = challenge.get("function_signature",
                                      f"{challenge['name'].lower().replace(' ', '_')}(...)")
            # Cria o prompt específico para classe ou função
            if "class LRUCache" in signature:
                 prompt = (
                     f"Please implement the following Python class based on this description: {challenge['description']}. "
                     f"The class and method signatures should be:\n{signature}\n"
                     f"Provide ONLY the complete Python code for the class definition and necessary imports. "
                     f"Do not include any explanations, example usage, markdown formatting (like ```python), "
                     f"or introductory sentences."
                 )
            else:
                 prompt = (
                     f"Please write a single, complete Python function that implements the following signature: `{signature}`. "
                     f"The function should solve this problem: {challenge['description']}. "
                     f"Provide ONLY the Python code for the function definition and necessary imports. "
                     f"Do not include any explanations, example usage, markdown formatting (like ```python), "
                     f"or introductory sentences."
                 )

            print(f"\n--- Desafio: {challenge['name']} (Dificuldade: {challenge['difficulty']}) ---")
            description_detail = challenge_details.get(challenge['name'],
                                                       "Descrição detalhada não encontrada.")
            print(f"  {description_detail}")

            responses = {}
            # Consulta as IAs habilitadas
            if can_run_claude:
                print("Consultando Claude AI...")
                responses["Claude"] = self.query_claude(prompt)
                time.sleep(1) # Pequena pausa entre APIs
            if can_run_openai:
                print("Consultando ChatGPT...")
                responses["ChatGPT"] = self.query_chatgpt(prompt)
                time.sleep(1) # Pequena pausa entre APIs
            if can_run_gemini:
                print("Consultando Gemini AI...")
                responses["Gemini"] = self.query_gemini(prompt)
                # Pausa para o limite da API Gemini (considerar alternativas como backoff)
                print("Aguardando 31s para o limite da API Gemini...")
                time.sleep(31)

            challenge_summary_lines = []
            # Avalia a resposta de cada IA
            for ai_name in enabled_ais:
                response_data = responses.get(ai_name)
                # Garante que há um dicionário de resposta mesmo que a consulta falhe
                if response_data is None:
                    response_data = {"content": "", "execution_time": 0}

                result = self.evaluate_solution(ai_name, challenge, response_data)
                self.results.append(result) # Adiciona resultado agregado

                # Calcula resumo de testes para o print imediato
                tests_attempted = 0
                tests_passed = 0
                tests_error = result.get('num_execution_errors', 0)
                # Filtra os detalhes de teste relevantes para este AI/desafio
                related_details = [d for d in self.detailed_test_results
                                   if d['ai'] == ai_name and d['challenge'] == challenge['name']]
                tests_attempted = len([d for d in related_details if d['status'] not in ['No Code', 'Not Executed']]) # Conta tentativas reais
                tests_passed = len([d for d in related_details if d['status'] == 'Pass'])

                flake8_str = (str(result['flake8_issues'])
                              if result['flake8_issues'] != -1 else 'N/A')
                summary_line = (f"  {ai_name:<10}: Tests: {tests_passed}/{tests_attempted} passed "
                                f"({tests_error} errors). Flake8: {flake8_str}. "
                                f"API Time: {result['api_execution_time']:.2f}s.")
                challenge_summary_lines.append(summary_line)

            # Imprime o resumo do desafio atual
            print(f"--- Resumo Desafio {challenge['name']} ---")
            for line in challenge_summary_lines:
                print(line)
            print(f"--- Desafio {challenge['name']} concluído ---")

        if not self.results:
            print("\nNenhum resultado coletado. Encerrando.")
            return

        # Gera os relatórios finais
        self.generate_report()

    # --- Métodos Auxiliares para Geração de Relatórios ---

    def _prepare_dataframe(self) -> pd.DataFrame | None:
        """Prepara o DataFrame principal com resultados e métricas agregadas."""
        if not self.results:
            print("Não há resultados para preparar o DataFrame.")
            return None
        try:
            df = pd.DataFrame(self.results)
            agg_metrics = self._calculate_aggregated_test_metrics()

            # Adiciona métricas agregadas ao DataFrame principal
            df['avg_exec_success_rate'] = df.apply(
                lambda row: agg_metrics.get((row['ai'], row['challenge']), {})
                                     .get('avg_exec_success_rate', 0.0), axis=1
            )
            df['avg_exec_time_ms'] = df.apply(
                lambda row: agg_metrics.get((row['ai'], row['challenge']), {})
                                     .get('avg_exec_time_ms', np.nan), axis=1
            )
            df['total_tests_passed'] = df.apply(
                lambda row: agg_metrics.get((row['ai'], row['challenge']), {})
                                     .get('total_tests_passed', 0), axis=1
            )
            df['total_tests_attempted'] = df.apply(
                lambda row: agg_metrics.get((row['ai'], row['challenge']), {})
                                       .get('total_tests_attempted', 0), axis=1
            )

            # Remove colunas potencialmente grandes ou redundantes
            df = df.drop(['execution_results_raw'], axis=1, errors='ignore')

            # Garante tipos numéricos e trata NaNs
            self._normalize_dataframe_types(df)

            return df

        except Exception as e:
            print(f"Erro crítico ao criar/processar DataFrame principal: {e}\n"
                  f"Traceback: {traceback.format_exc()}\nResultados brutos: {self.results}")
            return None

    def _calculate_aggregated_test_metrics(self) -> dict:
        """Calcula métricas agregadas por (IA, Desafio) a partir dos detalhes de teste."""
        agg_metrics = {}
        # Cria chaves únicas para agregação
        result_keys = set((row['ai'], row['challenge']) for row in self.results)

        for key in result_keys:
            ai_name, challenge_name = key
            # Filtra detalhes de teste para a combinação atual
            details = [d for d in self.detailed_test_results
                       if d['ai'] == ai_name and d['challenge'] == challenge_name]

            if not details:
                agg_metrics[key] = {'avg_exec_success_rate': 0.0, 'avg_exec_time_ms': np.nan,
                                    'total_tests_passed': 0, 'total_tests_attempted': 0}
                continue

            passed_runs = [d for d in details if d['status'] == 'Pass']
            # Considera apenas tempos de execuções que não falharam completamente ou não foram executadas
            valid_times = [d['execution_time_ms'] for d in details
                           if d.get('execution_time_ms', -1) >= 0 and
                           d['status'] not in ['Unknown', 'Not Executed', 'No Code', 'Error']] # Foco no tempo de sucesso/falha
            # Testes que foram realmente tentados (tiveram Pass, Fail ou Error de execução)
            attempted_runs = [d for d in details if d['status'] in ['Pass', 'Fail', 'Error']]
            num_attempted = len(attempted_runs)
            num_passed = len(passed_runs)

            success_rate = num_passed / num_attempted if num_attempted > 0 else 0.0
            mean_time = np.mean(valid_times) if valid_times else np.nan

            agg_metrics[key] = {
                'avg_exec_success_rate': success_rate,
                'avg_exec_time_ms': mean_time,
                'total_tests_passed': num_passed,
                'total_tests_attempted': num_attempted
            }
        return agg_metrics

    def _normalize_dataframe_types(self, df: pd.DataFrame):
        """Converte colunas para tipos numéricos e trata NaNs inplace."""
        expected_numeric_cols = [
            'api_execution_time', 'lines_of_code', 'completeness', 'flake8_issues',
            'num_execution_errors', 'avg_exec_success_rate', 'avg_exec_time_ms',
            'total_tests_passed', 'total_tests_attempted'
        ]
        for col in expected_numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce') # Converte para numérico, erros viram NaN
                # Tratamento específico de NaNs por coluna
                if col == 'flake8_issues':
                    df.loc[:, col] = df[col].fillna(-1).astype(int)
                elif col == 'num_execution_errors':
                    df.loc[:, col] = df[col].fillna(0).astype(int)
                elif col == 'avg_exec_success_rate':
                    df.loc[:, col] = df[col].fillna(0.0)
                elif col == 'avg_exec_time_ms':
                    df.loc[:, col] = df[col].fillna(-1.0) # Usa -1 para indicar NaN/Não aplicável
                elif col in ['total_tests_passed', 'total_tests_attempted']:
                    df.loc[:, col] = df[col].fillna(0).astype(int)
                else: # Outras colunas numéricas (api_time, loc, completeness)
                    df.loc[:, col] = df[col].fillna(0.0)
        if 'first_execution_error' in df.columns:
             df['first_execution_error'] = df['first_execution_error'].fillna('').astype(str)

    def _save_environment_info(self, timestamp: str):
        """Salva as informações do ambiente num arquivo TXT."""
        env_info_path = os.path.join(self.output_dir, f"environment_info_{timestamp}.txt")
        try:
            with open(env_info_path, "w", encoding='utf-8') as f:
                for key, value in self.environment_info.items():
                    if isinstance(value, dict):
                        f.write(f"{key}:\n")
                        for sk, sv in value.items():
                            f.write(f"  {sk}: {sv}\n")
                    else:
                        f.write(f"{key}: {value}\n")
            print(f"- Informações do Ambiente salvas: {env_info_path}")
        except Exception as e:
            print(f"Erro ao salvar info do ambiente: {e}")

    def _save_full_results_json(self, timestamp: str):
        """Salva a lista completa de resultados brutos em JSON."""
        json_path = os.path.join(self.output_dir, f"benchmark_results_full_{timestamp}.json")
        try:
            # Encoder customizado para lidar com tipos numpy e datetime
            class NpEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer): return int(obj)
                    if isinstance(obj, np.floating): return float(obj)
                    if isinstance(obj, np.ndarray): return obj.tolist()
                    if isinstance(obj, (datetime,)): return obj.isoformat()
                    if isinstance(obj, np.bool_): return bool(obj)
                    if isinstance(obj, bytes): return obj.decode('utf-8', errors='replace')
                    return super(NpEncoder, self).default(obj)

            with open(json_path, "w", encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False, cls=NpEncoder)
            print(f"- Resultados detalhados (JSON completo): {json_path}")
        except Exception as e:
            print(f"Erro ao salvar JSON completo: {e}")

    def _save_detailed_tests_csv(self, timestamp: str):
        """Salva os detalhes de cada caso de teste em CSV."""
        detailed_csv_path = os.path.join(self.output_dir,
                                         f"benchmark_test_details_{timestamp}.csv")
        if self.detailed_test_results:
            try:
                df_detailed = pd.DataFrame(self.detailed_test_results)
                # Define a ordem desejada das colunas
                cols_order = [
                    'ai', 'challenge', 'difficulty', 'test_case_index', 'status',
                    'execution_time_ms', 'error_message', 'stdout', 'input_args_str',
                    'expected_output_str', 'actual_output_str'
                ]
                # Garante que todas as colunas existam, preenchendo com NaN se faltar
                for col in cols_order:
                    if col not in df_detailed.columns:
                        df_detailed[col] = pd.NA
                df_detailed = df_detailed[cols_order] # Reordena
                df_detailed.to_csv(detailed_csv_path, index=False, encoding='utf-8')
                print(f"- Detalhes por caso de teste (CSV): {detailed_csv_path}")
            except Exception as e:
                print(f"Erro ao salvar CSV detalhado por teste: {e}\n"
                      f"Traceback: {traceback.format_exc()}")
        else:
            print("- Nenhum detalhe por caso de teste para salvar.")

    def _create_csv_header(self) -> str:
        """Cria um cabeçalho de comentário com informações do ambiente para CSVs."""
        env_comments = ["# --- Environment Info ---"]
        for key, value in self.environment_info.items():
            if isinstance(value, dict):
                env_comments.append(f"# {key}:")
                for sk, sv in value.items():
                     env_comments.append(f"#   {sk}: {sv}")
            else:
                env_comments.append(f"# {key}: {value}")
        env_comments.append("# --- Benchmark Data ---")
        return "\n".join(env_comments) + "\n"

    def _save_summary_csv(self, df: pd.DataFrame, timestamp: str):
        """Salva o resumo do benchmark (DataFrame principal) em CSV."""
        summary_path = os.path.join(self.output_dir, f"benchmark_summary_{timestamp}.csv")
        try:
            # Seleciona colunas para o resumo (exclui código e resposta completa)
            cols_to_drop_summary = ['code', 'full_response']
            summary_cols = [col for col in df.columns if col not in cols_to_drop_summary]
            summary = df[summary_cols].copy()

            env_header = self._create_csv_header()

            with open(summary_path, "w", encoding='utf-8', newline='') as f:
                f.write(env_header)
                # Usa lineterminator='\n' para consistência entre OS
                summary.to_csv(f, index=False, encoding='utf-8',
                               float_format='%.4f', lineterminator='\n')
            print(f"- Resumo salvo (CSV): {summary_path}")
        except Exception as e:
            print(f"Erro ao salvar resumo CSV: {e}\nTraceback: {traceback.format_exc()}")

    def _calculate_statistics(self, df: pd.DataFrame) -> pd.DataFrame | None:
         """Calcula estatísticas agregadas por IA."""
         try:
             stats_metrics = [
                 ("Average API Time (s)", "api_execution_time", "mean"),
                 ("Average Lines of Code", "lines_of_code", "mean"),
                 ("Average Completeness (%)", "completeness", "mean"),
                 ("Average Flake8 Issues", "flake8_issues", "mean_valid"), # Ignora -1
                 ("Total Tests Passed", "total_tests_passed", "sum"),
                 ("Total Tests Attempted", "total_tests_attempted", "sum"),
                 ("Overall Execution Success (%)", "avg_exec_success_rate", "mean_percentage"),
                 ("Average Execution Time (ms)", "avg_exec_time_ms", "mean_valid"), # Ignora -1/NaN
                 ("Total Execution Errors", "num_execution_errors", "sum"),
                 ("Example First Error", "first_execution_error", "first_non_empty")
             ]
             ai_list = sorted(df['ai'].unique())
             stats_data = {"Metric": [m[0] for m in stats_metrics]} # Coluna de Métricas

             for ai_name in ai_list:
                 df_ai = df[df["ai"] == ai_name]
                 ai_column_data = []
                 for metric_name, col_key, agg_type in stats_metrics:
                     if col_key not in df_ai.columns:
                         ai_column_data.append(np.nan) # Coluna não existe
                         continue

                     if agg_type == "mean":
                         avg_val = df_ai[col_key].mean()
                     elif agg_type == "sum":
                         avg_val = df_ai[col_key].sum()
                     elif agg_type == "mean_valid": # Média ignorando -1 ou NaN
                         valid_data = df_ai[df_ai[col_key].notna() & (df_ai[col_key] != -1)][col_key]
                         avg_val = valid_data.mean() if not valid_data.empty else np.nan
                     elif agg_type == "mean_percentage": # Média convertida para percentagem
                         avg_val = df_ai[col_key].mean() * 100
                     elif agg_type == "first_non_empty": # Primeiro erro não vazio
                         non_empty_errors = df_ai[df_ai[col_key].fillna('').astype(str) != ''][col_key]
                         avg_val = non_empty_errors.iloc[0] if not non_empty_errors.empty else ""
                     else:
                         avg_val = np.nan # Tipo de agregação não conhecido

                     ai_column_data.append(avg_val)
                 stats_data[ai_name] = ai_column_data # Adiciona coluna da IA

             return pd.DataFrame(stats_data)

         except Exception as e:
             print(f"Erro ao calcular estatísticas: {e}\nTraceback: {traceback.format_exc()}")
             return None

    def _save_statistics_csv(self, stats_df: pd.DataFrame, timestamp: str):
        """Salva as estatísticas calculadas em CSV."""
        stats_path = os.path.join(self.output_dir, f"benchmark_stats_{timestamp}.csv")
        try:
            env_header_stats = self._create_csv_header()
            with open(stats_path, "w", encoding='utf-8', newline='') as f:
                f.write(env_header_stats)
                stats_df.to_csv(f, index=False, encoding='utf-8',
                                float_format='%.2f', lineterminator='\n')
            print(f"- Estatísticas salvas (CSV): {stats_path}")
        except Exception as e:
            print(f"Erro ao salvar estatísticas CSV: {e}\nTraceback: {traceback.format_exc()}")

    def _display_summary_table(self, stats_df: pd.DataFrame):
        """Exibe uma tabela de resumo das estatísticas no console."""
        print("\n===== SUMÁRIO DO BENCHMARK (ESTATÍSTICAS) =====")
        if stats_df is None or stats_df.empty:
            print("Não foi possível gerar o sumário de estatísticas.")
            return

        # Formata os valores para exibição
        stats_display = stats_df.copy()
        for col in stats_display.columns:
            if col == 'Metric': continue # Ignora a coluna de métricas

            formatted_col = []
            if stats_display.empty: continue # Pula se não houver linhas

            for i in stats_display.index:
                metric_name = stats_display.loc[i, 'Metric']
                value = stats_display.loc[i, col]
                formatted_val = "N/A" # Default

                if not pd.isna(value):
                    if "Flake8" in metric_name: formatted_val = f"{value:.1f}" if value != -1 else "N/A"
                    elif "Time (ms)" in metric_name: formatted_val = f"{value:.1f}" if value != -1 else "N/A"
                    elif "Success (%)" in metric_name or "Completeness (%)" in metric_name: formatted_val = f"{value:.1f}%"
                    elif "Total Tests" in metric_name or "Total Execution Errors" in metric_name: formatted_val = f"{int(value)}"
                    elif "Time (s)" in metric_name: formatted_val = f"{value:.2f}"
                    elif "Example First Error" in metric_name:
                         val_str = str(value)
                         formatted_val = val_str[:60] + ('...' if len(val_str) > 60 else '')
                    else: formatted_val = f"{value:.1f}" # Default para outros números

                formatted_col.append(formatted_val)

            if formatted_col: # Garante que a lista não está vazia
                stats_display[col] = formatted_col

        # Tenta imprimir com tabulate, se disponível
        try:
            from tabulate import tabulate
            print(tabulate(stats_display, headers='keys', tablefmt='grid', showindex=False))
        except ImportError:
            print("(Instale 'tabulate' para uma tabela mais bonita: pip install tabulate)")
            print(stats_display.to_string(index=False))
        except Exception as e_print:
            print(f"Erro ao imprimir tabela final: {e_print}")
            print(stats_display.to_string(index=False)) # Fallback para to_string

    def generate_report(self):
        """
        Gera todos os relatórios (ambiente, JSON, CSVs) e visualizações
        a partir dos resultados coletados.
        """
        if not self.results:
            print("Não há resultados para gerar o relatório.")
            return

        print("\n--- Gerando Relatórios ---")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"Timestamp dos relatórios: {timestamp}")
        print(f"Diretório de saída: '{self.output_dir}/'")

        # 1. Salva Informações do Ambiente
        self._save_environment_info(timestamp)

        # 2. Prepara DataFrame Principal
        df = self._prepare_dataframe()
        if df is None:
            print("Falha ao preparar o DataFrame. Geração de relatórios subsequentes abortada.")
            return

        # 3. Salva Resultados Completos em JSON
        self._save_full_results_json(timestamp)

        # 4. Salva Detalhes dos Testes em CSV
        self._save_detailed_tests_csv(timestamp)

        # 5. Salva Resumo em CSV
        self._save_summary_csv(df, timestamp)

        # 6. Calcula e Salva Estatísticas em CSV
        stats_df = self._calculate_statistics(df)
        if stats_df is not None:
            self._save_statistics_csv(stats_df, timestamp)
        else:
            print("- Não foi possível calcular ou salvar as estatísticas.")

        # 7. Gera Visualizações
        try:
            print("\n--- Gerando Visualizações ---")
            self.create_visualizations(df, timestamp, self.output_dir)
            print(f"- Visualizações salvas como PNG em '{self.output_dir}/'")
        except Exception as e:
            print(f"Erro ao gerar visualizações: {e}\nTraceback: {traceback.format_exc()}")

        # 8. Exibe Sumário no Console (usando as estatísticas calculadas)
        if stats_df is not None:
             self._display_summary_table(stats_df)

    # --- Métodos Auxiliares para Visualizações ---

    def _plot_bar_charts(self, df_plot: pd.DataFrame, ai_list: list, ai_color_map: dict,
                         challenge_order: list, timestamp: str, output_dir: str):
        """Gera gráficos de barras para diversas métricas."""
        print("Gerando Gráficos de Barras...")
        # Métricas para plotar em barras (Chave -> (Título, Label Y, Nome da Coluna))
        metrics_to_plot = {
            "1_api_execution_time": ("Tempo Médio de Resposta API", "Tempo (s)", "api_execution_time"),
            "2_lines_of_code": ("Média Linhas de Código (LoC)", "Número de Linhas", "lines_of_code"),
            "3_completeness": ("Completude Média da Solução", "Score (%)", "completeness"),
            "4_flake8_issues": ("Média Problemas Flake8", "Número de Problemas", "flake8_issues"),
            "7_num_execution_errors": ("Média Erros de Execução", "Número de Erros", "num_execution_errors"),
        }
        # Adiciona métricas de execução se existirem
        if 'avg_exec_success_rate' in df_plot.columns:
             metrics_to_plot["5_avg_exec_success_rate"] = ("Taxa Média de Sucesso na Execução", "Sucesso (%)", "avg_exec_success_rate")
        if 'avg_exec_time_ms' in df_plot.columns:
             metrics_to_plot["6_avg_exec_time_ms"] = ("Tempo Médio de Execução (Executor)", "Tempo (ms)", "avg_exec_time_ms")
        metrics_to_plot = dict(sorted(metrics_to_plot.items())) # Ordena pela chave numérica

        for metric_key, (title, ylabel, col_name) in metrics_to_plot.items():
            if col_name not in df_plot.columns:
                 print(f"Pulando gráfico de barras '{title}': Coluna {col_name} não encontrada.")
                 continue

            plot_data = df_plot.copy()
            # Filtra dados inválidos específicos da métrica
            if col_name == 'flake8_issues': plot_data = plot_data[plot_data[col_name] != -1]
            elif col_name == 'avg_exec_time_ms': plot_data = plot_data[plot_data[col_name].notna() & (plot_data[col_name] != -1)]
            elif col_name == 'num_execution_errors': plot_data = plot_data[plot_data[col_name].notna()]
            # Converte taxa de sucesso para percentual para plotagem
            elif col_name == 'avg_exec_success_rate': plot_data[col_name] = plot_data[col_name] * 100

            # Verifica se há dados válidos para plotar
            if plot_data.empty or plot_data[col_name].isnull().all():
                print(f"Pulando gráfico de barras '{title}': Nenhum dado válido encontrado.")
                continue

            try:
                plt.figure()
                # Cria tabela pivot para plotagem
                pivot = plot_data.pivot_table(index='challenge', columns='ai',
                                              values=col_name, aggfunc='mean')
                # Reordena linhas (desafios) e colunas (AIs)
                pivot = pivot.reindex(challenge_order, axis=0).reindex(ai_list, axis=1)

                # Define valor para preencher NaNs (0 para scores, NaN para tempos/erros)
                fill_value = 0.0
                if col_name in ['flake8_issues', 'avg_exec_time_ms', 'num_execution_errors']:
                    fill_value = np.nan # Não preenche NaNs para não distorcer médias
                pivot.fillna(fill_value, inplace=True)

                # Pula se todos os valores forem NaN após o fill
                if pivot.isnull().all().all():
                    print(f"Pulando gráfico de barras '{title}': Nenhum dado válido (tudo NaN).")
                    plt.close()
                    continue

                # Plota o gráfico de barras
                num_ais = len(ai_list)
                ax = pivot.plot(kind='bar', figsize=(max(12, num_ais * 2), 7),
                                color=[ai_color_map.get(ai, 'grey') for ai in pivot.columns],
                                width=0.8)

                plt.title(f'{title} por Desafio')
                plt.ylabel(ylabel)
                plt.xlabel('Desafio (Ordenado por Dificuldade)')
                plt.xticks(rotation=45, ha='right', fontsize=9)
                plt.yticks(fontsize=9)

                # Ajusta limites do eixo Y conforme a métrica
                if col_name in ['completeness', 'avg_exec_success_rate']: plt.ylim(0, 105)
                # Para flake8/erros, ajusta o limite superior para incluir o máximo + margem
                elif col_name in ['flake8_issues', 'num_execution_errors'] and not pivot.isnull().all().all():
                    max_val = pivot.max().max() # Máximo valor em todo o pivot
                    plt.ylim(bottom=0, top = max(1, max_val * 1.1) if not pd.isna(max_val) else 1)
                elif col_name == 'avg_exec_time_ms' and not pivot.isnull().all().all():
                    plt.ylim(bottom=0) # Começa em 0 para tempos

                plt.legend(title='IA', bbox_to_anchor=(1.02, 1), loc='upper left')
                plt.tight_layout(rect=[0, 0, 0.88, 1]) # Ajusta layout para caber a legenda
                path = os.path.join(output_dir, f"{metric_key}_{timestamp}.png")
                plt.savefig(path)
                plt.close() # Fecha a figura para liberar memória
            except Exception as e:
                print(f"Erro ao gerar gráfico de barras '{title}': {e}\n"
                      f"Traceback: {traceback.format_exc()}")
                plt.close()

    def _plot_box_plots(self, df_plot: pd.DataFrame, ai_list: list, ai_color_map: dict,
                        timestamp: str, output_dir: str):
        """Gera box plots para comparar a distribuição das métricas entre IAs."""
        print("Gerando Box Plots...")
        # Métricas para Box Plot e suas condições de validade
        metrics_for_boxplot = {
            "api_execution_time": {"title": "Distribuição Tempo Resposta API por IA", "ylabel": "Tempo (s)", "valid_condition": None},
            "lines_of_code": {"title": "Distribuição Linhas de Código (LoC) por IA", "ylabel": "Número de Linhas", "valid_condition": (lambda df, col: df[col] > 0)},
            "completeness": {"title": "Distribuição Completude da Solução por IA", "ylabel": "Score (%)", "valid_condition": None}
        }
        if 'flake8_issues' in df_plot.columns:
             metrics_for_boxplot["flake8_issues"] = {"title": "Distribuição Problemas Flake8 por IA", "ylabel": "Número de Problemas", "valid_condition": (lambda df, col: df[col] != -1)}
        if 'avg_exec_time_ms' in df_plot.columns:
             metrics_for_boxplot["avg_exec_time_ms"] = {"title": "Distribuição Tempo Médio Execução (Executor) por IA", "ylabel": "Tempo (ms)", "valid_condition": (lambda df, col: df[col].notna() & (df[col] != -1))}
        if 'num_execution_errors' in df_plot.columns:
             metrics_for_boxplot["num_execution_errors"] = {"title": "Distribuição Erros de Execução por IA", "ylabel": "Número de Erros", "valid_condition": (lambda df, col: df[col].notna())}

        plot_counter_boxplot = 7 # Contador para nomes de arquivos

        for metric_key_boxplot, config in metrics_for_boxplot.items():
            plot_counter_boxplot += 1
            if metric_key_boxplot not in df_plot.columns:
                print(f"  Pulando Box Plot para '{metric_key_boxplot}': coluna não encontrada.")
                continue

            # Prepara dados para o boxplot
            data_for_current_metric = []
            labels_for_current_metric = []
            temp_df = df_plot.copy()

            # Aplica condição de validade se existir
            if config.get("valid_condition"):
                try:
                    condition_func = config["valid_condition"]
                    temp_df = temp_df[condition_func(temp_df, metric_key_boxplot)]
                except Exception as e:
                    print(f"  Aviso: Falha ao aplicar condição para boxplot "
                          f"de '{metric_key_boxplot}': {e}.")

            if temp_df.empty or temp_df[metric_key_boxplot].isnull().all():
                print(f"  Pulando Box Plot para '{metric_key_boxplot}': Sem dados válidos.")
                continue

            # Coleta dados por IA
            for ai_name in ai_list:
                ai_data = temp_df[temp_df['ai'] == ai_name][metric_key_boxplot].dropna()
                if not ai_data.empty:
                    data_for_current_metric.append(ai_data.values)
                    labels_for_current_metric.append(ai_name)

            if not data_for_current_metric:
                print(f"  Pulando Box Plot para '{metric_key_boxplot}': Nenhum dado por IA.")
                continue

            # Gera o Box Plot
            try:
                plt.figure(figsize=(max(8, len(labels_for_current_metric) * 1.2), 6))
                box_plot = plt.boxplot(data_for_current_metric,
                                       labels=labels_for_current_metric,
                                       patch_artist=True, # Para preencher cores
                                       showfliers=True) # Mostra outliers

                plt.title(config["title"], fontsize=14)
                plt.ylabel(config["ylabel"], fontsize=10)
                plt.xlabel("IA", fontsize=10)
                plt.xticks(rotation=30, ha='right', fontsize=9)
                plt.yticks(fontsize=9)
                plt.grid(axis='y', linestyle='--', alpha=0.7)

                # Aplica cores do mapa
                colors_to_use = [ai_color_map.get(label, 'grey')
                                 for label in labels_for_current_metric]
                for patch, color in zip(box_plot['boxes'], colors_to_use):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.6) # Leve transparência

                plt.tight_layout()
                clean_metric_name = metric_key_boxplot.replace('_ms', '') # Nome mais limpo
                plot_filename = (f"{plot_counter_boxplot}_boxplot_{clean_metric_name}_"
                                 f"{timestamp}.png")
                path = os.path.join(output_dir, plot_filename)
                plt.savefig(path)
                plt.close()
            except Exception as e:
                print(f"  Erro ao gerar Box Plot para '{metric_key_boxplot}': {e}")
                plt.close()
        print("Geração de Box Plots concluída.")

    def _plot_radar_chart(self, df: pd.DataFrame, ai_list: list, ai_color_map: dict,
                          timestamp: str, output_dir: str):
        """Gera um gráfico de radar comparando métricas normalizadas entre IAs."""
        num_ais = len(ai_list)
        if num_ais < 2:
            print("Gráfico Radar pulado (requer >= 2 IAs).")
            return

        print(f"Gerando gráfico Radar para {num_ais} IAs (3 métricas base)...")
        try:
            # Métricas base para o radar
            numeric_cols_radar = ['api_execution_time', 'lines_of_code', 'completeness']
            # Verifica se todas as colunas necessárias existem
            if not all(col in df.columns for col in numeric_cols_radar):
                missing = [col for col in numeric_cols_radar if col not in df.columns]
                print(f"Pulando Radar: Faltando colunas {missing}")
                return

            # Calcula médias por IA
            means_data = {}
            for ai in ai_list:
                df_ai_radar = df[df["ai"] == ai]
                means = {}
                for col in numeric_cols_radar:
                    # Calcula média apenas se a coluna existe e tem valores não nulos
                    means[col] = df_ai_radar[col].mean() if not df_ai_radar[col].isnull().all() else 0
                means_data[ai] = means

            # Normaliza as métricas (0-100, onde maior é melhor)
            # 1. Tempo API (Invertido: menor tempo -> maior score)
            all_times = [m.get("api_execution_time", 0) for m in means_data.values()]
            max_time = max(all_times + [1e-6]) # Evita divisão por zero
            time_norm = {ai: 100 * (1 - means_data[ai].get("api_execution_time", 0) / max_time)
                         if max_time > 0 else 0 for ai in ai_list}

            # 2. Linhas de Código (Invertido e Normalizado: menos linhas -> maior score)
            all_locs = [m.get("lines_of_code", 0) for m in means_data.values()]
            valid_locs = [loc for loc in all_locs if loc > 0]
            min_loc_valid = min(valid_locs) if valid_locs else 0
            max_loc_valid = max(all_locs + [1e-6]) # Evita divisão por zero
            lines_norm = {}
            for ai in ai_list:
                loc = means_data[ai].get("lines_of_code", 0)
                if loc <= 0: lines_norm[ai] = 0 # Código inválido ou não gerado
                # Normaliza entre o min e max válidos
                elif min_loc_valid > 0 and max_loc_valid > min_loc_valid:
                    norm_val = 100 * (1 - (loc - min_loc_valid) / (max_loc_valid - min_loc_valid))
                    # Garante que o score fique entre 0 e 100
                    lines_norm[ai] = max(0, min(100, norm_val))
                # Se todos os LoCs válidos forem iguais (ou só houver 1)
                else: lines_norm[ai] = 100 if loc > 0 else 0
                # Garante que o mínimo absoluto (se for > 0) tenha score 100
                if loc == min_loc_valid and min_loc_valid > 0: lines_norm[ai] = 100

            # 3. Completude (Já é 0-100)
            comp_norm = {ai: means_data[ai].get("completeness", 0) for ai in ai_list}

            # Prepara dados para o gráfico de radar
            categories = ['Velocidade API\n(Norm)', 'Concisão LoC\n(Norm)', 'Completude\n(%)']
            N = len(categories)
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1] # Fecha o círculo

            fig = plt.figure(figsize=(9, 9))
            ax = fig.add_subplot(111, polar=True)

            # Plota os dados para cada IA
            for ai_name in ai_list:
                 values = [time_norm.get(ai_name, 0),
                           lines_norm.get(ai_name, 0),
                           comp_norm.get(ai_name, 0)]
                 plot_values = values + values[:1] # Fecha o polígono
                 color = ai_color_map.get(ai_name, 'grey')
                 ax.plot(angles, plot_values, 'o-', linewidth=2, label=ai_name, color=color)
                 ax.fill(angles, plot_values, color, alpha=0.2) # Preenche a área

            # Configurações do gráfico
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            ax.set_yticks(np.arange(0, 101, 20)) # Escala de 0 a 100
            ax.set_ylim(0, 100)

            plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15)) # Legenda fora
            plt.title('Comparação Geral (3 Métricas Normalizadas)', size=14, y=1.18)
            plt.tight_layout()

            plot_counter_radar = 15 # Aproximação do contador de plots
            path = os.path.join(output_dir, f"{plot_counter_radar}_radar_chart_{timestamp}.png")
            plt.savefig(path)
            plt.close()
        except Exception as e:
            print(f"Erro ao gerar gráfico Radar: {e}\nTraceback: {traceback.format_exc()}")
            plt.close()

    def create_visualizations(self, df: pd.DataFrame, timestamp: str, output_dir: str):
        """
        Cria e salva visualizações (gráficos de barras, box plots, radar chart)
        a partir do DataFrame de resultados.
        """
        plt.style.use('ggplot') # Estilo dos gráficos
        ai_list = sorted(df['ai'].unique())
        num_ais = len(ai_list)

        # Define mapa de cores para as IAs
        if num_ais <= 10: colors = plt.cm.tab10(np.linspace(0, 1, num_ais))
        else: colors = plt.cm.viridis(np.linspace(0, 1, num_ais)) # Mais cores se necessário
        ai_color_map = {ai: colors[i] for i, ai in enumerate(ai_list)}

        # Verifica colunas essenciais para os gráficos
        base_required_cols = ['challenge', 'ai', 'api_execution_time', 'lines_of_code',
                              'completeness', 'flake8_issues', 'difficulty',
                              'num_execution_errors']
        exec_cols = [col for col in ['avg_exec_success_rate', 'avg_exec_time_ms'] if col in df.columns]
        required_cols = base_required_cols + exec_cols

        if df.empty or not all(col in df.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in df.columns]
            print(f"AVISO: DataFrame vazio ou faltando colunas para gerar gráficos. "
                  f"Faltando: {missing_cols}. Pulando visualizações.")
            return

        # Prepara DataFrame para plotagem (ordena por dificuldade)
        difficulty_order = pd.CategoricalDtype(['Easy', 'Medium', 'Hard'], ordered=True)
        df['difficulty_cat'] = df['difficulty'].astype(difficulty_order)
        # Obtém ordem dos desafios baseada na dificuldade
        challenge_order = df.sort_values('difficulty_cat')['challenge'].unique()
        df_plot = df.copy() # Usa cópia para não alterar o original

        # Gera os diferentes tipos de gráficos
        self._plot_bar_charts(df_plot, ai_list, ai_color_map, challenge_order,
                              timestamp, output_dir)
        self._plot_box_plots(df_plot, ai_list, ai_color_map, timestamp, output_dir)
        self._plot_radar_chart(df, ai_list, ai_color_map, timestamp, output_dir)


# --- Ponto de Entrada Principal ---
if __name__ == "__main__":
    print("===========================================================")
    print(" AI Programming Benchmark: Claude vs ChatGPT vs Gemini")
    print("                 (Incluindo Análise Flake8 e Execução)")
    print("===========================================================")

    if not os.path.exists(".env"):
        print("AVISO: Arquivo .env não encontrado. Chaves API devem ser variáveis de ambiente.")

    try:
        benchmark = AIBenchmark()
        # Verifica se pelo menos uma API está funcional após __init__
        if benchmark.claude_api_key or benchmark.openai_api_key or benchmark.gemini_api_key:
            benchmark.run_benchmark()
            print("\nBenchmark concluído.")
            print(f"Relatórios salvos em: '{benchmark.output_dir}'")
        else:
            # Se nenhuma chave funcionou ou foi fornecida
            print("\nERRO FATAL: Nenhuma chave de API válida configurada ou encontrada. "
                  "Benchmark não pode ser executado.")

    except Exception as main_e:
        # Captura erros inesperados durante a inicialização ou execução
        print("\nERRO INESPERADO DURANTE A EXECUÇÃO DO BENCHMARK:")
        print(f"Tipo: {type(main_e).__name__}")
        print(f"Erro: {main_e}")
        print("Traceback:")
        traceback.print_exc()

    print("===========================================================")