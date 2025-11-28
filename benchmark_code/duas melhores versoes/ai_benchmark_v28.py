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
    try:
        import pkg_resources # type: ignore
    except ImportError:
        pkg_resources = None

# Carrega variáveis de ambiente do .env
load_dotenv()

class AIBenchmark:
    """
    Executa um benchmark de programação comparando modelos de IA
    (Claude, ChatGPT, Gemini) em desafios de código, avaliando a qualidade,
    corretude (via execução externa ou local para LRU) e desempenho.

    Permite configuração de chaves API, nomes de modelo e URL do executor
    via variáveis de ambiente (ficheiro .env). Gera relatórios detalhados
    e visualizações dos resultados.

    Atributos:
        claude_api_key (str | None): Chave API para Anthropic Claude.
        openai_api_key (str | None): Chave API para OpenAI ChatGPT.
        gemini_api_key (str | None): Chave API para Google Gemini.
        claude_model (str): Nome do modelo Claude a ser usado.
        openai_model (str): Nome do modelo OpenAI a ser usado.
        gemini_model (str | None): Nome do modelo Gemini a ser usado.
        executor_url (str): URL do serviço externo para execução de código.
        code_challenges (list): Lista de dicionários definindo os desafios.
        results (list): Lista para armazenar resultados agregados por desafio/IA.
        detailed_test_results (list): Lista para resultados detalhados por caso de teste.
        output_dir (str): Diretório para salvar relatórios.
        flake8_available (bool): Indica se o Flake8 está disponível.
        environment_info (dict): Informações sobre o ambiente de execução.
    """
    def __init__(self):
        """
        Inicializa o benchmark. Carrega configurações, define desafios,
        verifica ferramentas (flake8) e coleta informações do ambiente.
        """
        self.claude_api_key = os.getenv('CLAUDE_API_KEY')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')

        self.results = []
        self.detailed_test_results = []
        self.output_dir = "benchmark_reports"
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Diretório de saída: '{self.output_dir}/'")

        # --- Carrega Configurações (com fallbacks) ---
        self._configure_gemini_api() # Configura ou anula self.gemini_api_key

        # Nomes dos modelos via .env ou fallback para defaults
        self.claude_model = os.getenv('CLAUDE_MODEL_NAME', "claude-3-opus-20240229")
        self.openai_model = os.getenv('OPENAI_MODEL_NAME', "gpt-4-turbo")
        # Gemini model só é definido se a chave/configuração for válida
        self.gemini_model = os.getenv('GEMINI_MODEL_NAME', "gemini-1.5-pro-latest") if self.gemini_api_key else None

        # URL do Executor via .env ou fallback
        default_executor_url = "https://executor-service-1029404216382.europe-southwest1.run.app/execute"
        self.executor_url = os.getenv('EXECUTOR_SERVICE_URL', default_executor_url)
        # --- Fim Configurações ---

        # Definição dos desafios
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
        self.environment_info = self._get_environment_info() # Coleta info do ambiente

    def _configure_gemini_api(self):
        """Configura a API Gemini se a chave API estiver disponível."""
        if self.gemini_api_key:
            try:
                genai.configure(api_key=self.gemini_api_key)
                print("Chave API Gemini configurada.")
            except Exception as e:
                print(f"Erro configurar API Gemini: {e}. AI pulada.")
                self.gemini_api_key = None # Anula a chave se configuração falhar
        else:
            print("AVISO: Chave GEMINI_API_KEY não encontrada. Gemini AI pulada.")

    def _get_environment_info(self):
        """Coleta e retorna informações sobre o ambiente de execução."""
        # Usa os nomes de modelo e URL definidos no __init__
        info = {
            "Timestamp": datetime.now().isoformat(),
            "OS": f"{platform.system()} {platform.release()}",
            "Python Version": sys.version.split()[0],
            "Models": {
                "Claude": self.claude_model if self.claude_api_key else "N/A",
                "ChatGPT": self.openai_model if self.openai_api_key else "N/A",
                "Gemini": self.gemini_model if self.gemini_api_key else "N/A (Config Failed or No Key)"
            },
            "Executor Service URL": self.executor_url,
            "Local LRU Cache Execution": True,
            "Libraries": {}
        }
        libs_to_check = [
            'pandas', 'numpy', 'matplotlib', 'requests', 'python-dotenv',
            'google-generativeai', 'google-api-core'
        ]
        lib_versions = {}
        try: # Tenta importlib.metadata
            import importlib.metadata as importlib_metadata
            for lib in libs_to_check:
                try: lib_versions[lib] = importlib_metadata.version(lib)
                except importlib_metadata.PackageNotFoundError: lib_versions[lib] = "Not Found"
        except ImportError: # Fallback pkg_resources
            if pkg_resources:
                for lib in libs_to_check:
                    try: lib_versions[lib] = pkg_resources.get_distribution(lib).version
                    except pkg_resources.DistributionNotFound: lib_versions[lib] = "Not Found"
                    except Exception as e: lib_versions[lib] = f"Error ({type(e).__name__})"
            else:
                 for lib in libs_to_check: lib_versions[lib] = "Unknown"
        info["Libraries"] = lib_versions
        return info

    def _check_flake8(self):
        """Verifica disponibilidade do flake8."""
        try:
            subprocess.run([sys.executable, '-m', 'flake8', '--version'],
                           check=True, capture_output=True, timeout=5)
            print("Verificação Flake8: OK")
            self.flake8_available = True
        except Exception:
            print(f"AVISO: flake8 não encontrado via '{sys.executable} -m flake8'. Análise pulada.")
            self.flake8_available = False

    # --- query_claude CORRIGIDA ---
    def query_claude(self, prompt: str) -> dict:
        """
        Envia um prompt para a API do Claude e retorna a resposta.

        Args:
            prompt (str): O prompt a ser enviado para a API.

        Returns:
            dict: Um dicionário contendo 'content' (a resposta da API) e
                  'execution_time' (o tempo de chamada da API em segundos).
        """
        start_time = time.time()
        content = ""
        execution_time = 0
        if not self.claude_api_key:
            print("Pulando Claude: Chave API N/A.")
            return {"content": "", "execution_time": 0}

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
                headers=headers,
                json=data,
                timeout=120
            )
            response.raise_for_status() # Levanta HTTPError para 4xx/5xx
            result = response.json()
            if ("content" in result and
                    isinstance(result["content"], list) and
                    len(result["content"]) > 0):
                first_content = result["content"][0]
                content = first_content.get("text", "") if isinstance(first_content, dict) else ""
            else:
                print(f"Resposta Claude inesperada: {result}")

        except requests.exceptions.RequestException as e:
            err_msg = f"Erro req Claude: {e}"
            status_code_val = "N/A" # Valor padrão

            # Tenta obter o status code como inteiro
            if e.response is not None:
                try:
                    # Tenta converter para int, caso e.response.status_code retorne algo não numérico
                    status_code_val = int(e.response.status_code)
                except (ValueError, TypeError):
                    # Se falhar a conversão, usa o valor original (pode ser string)
                    status_code_val = e.response.status_code
                    print(f"  AVISO: Claude status code não é um inteiro ({type(status_code_val).__name__}): {status_code_val}")

            # Determina o tipo de erro COM BASE no status code numérico (se disponível)
            err_type = "(Unknown Error Type)" # Default
            if isinstance(status_code_val, int): # Só compara se for inteiro
                 if 400 <= status_code_val < 500: err_type = "(Client Error)"
                 elif status_code_val >= 500: err_type = "(Server Error)"
            elif status_code_val != "N/A": # Se não for inteiro mas não for N/A, é algo estranho
                 err_type = f"(Non-int Status: {status_code_val})"
            else: # Se for "N/A" (sem resposta ou erro antes)
                 err_type = "(Network/Timeout/Other Error)"

            # Constrói a mensagem de erro detalhada
            if e.response is not None:
                 err_msg += f" | Status:{status_code_val}{err_type}"
                 try:
                     # Tenta obter detalhes do JSON da resposta de erro
                     err_details = e.response.json()
                     api_error_type = err_details.get('error', {}).get('type', 'N/A')
                     api_error_message = err_details.get('error', {}).get('message', 'N/A')
                     err_msg += f" | API Err Type: {api_error_type} | API Err Msg: {api_error_message}"
                 except json.JSONDecodeError:
                     # Se não for JSON, pega o texto cru (limitado)
                     err_msg += f" | Body (non-JSON):{e.response.text[:150]}..."
                 except Exception as json_e: # Captura outros erros de processamento JSON
                      err_msg += f" | Error parsing error response: {json_e}"
            else:
                 err_msg += f" | {err_type}" # Adiciona tipo de erro se não houver resposta

            print(err_msg) # Imprime a mensagem de erro detalhada

        except Exception as e:
            print(f"Erro inesperado Claude: {type(e).__name__}-{e}")

        execution_time = time.time() - start_time
        return {"content": content or "", "execution_time": execution_time}
    # --- FIM DA CORREÇÃO query_claude ---


    def query_chatgpt(self, prompt: str) -> dict:
        """
        Envia um prompt para a API do OpenAI (ChatGPT) e retorna a resposta.

        Args:
            prompt (str): O prompt a ser enviado para a API.

        Returns:
            dict: Um dicionário contendo 'content' (a resposta da API) e
                  'execution_time' (o tempo de chamada da API em segundos).
        """
        start_time = time.time(); content = ""; execution_time = 0
        if not self.openai_api_key: print("Pulando ChatGPT: Chave API N/A."); return {"content": "", "execution_time": 0}
        # Usa self.openai_model definido no __init__
        headers={"Authorization":f"Bearer {self.openai_api_key}","Content-Type":"application/json"}
        data={"model":self.openai_model,"messages":[{"role":"user","content":prompt}],"max_tokens":4000}
        try:
            response=requests.post("https://api.openai.com/v1/chat/completions",headers=headers,json=data,timeout=120); response.raise_for_status(); result=response.json()
            if ("choices" in result and isinstance(result["choices"], list) and len(result["choices"]) > 0):
                choice=result["choices"][0]
                if (isinstance(choice, dict) and "message" in choice and isinstance(choice["message"], dict)): content=choice["message"].get("content", "")
                else: print(f"Resposta ChatGPT inesperada (message): {result}")
            else: print(f"Resposta ChatGPT inesperada (choices): {result}")
        except requests.exceptions.RequestException as e:
            err_msg=f"Erro req ChatGPT: {e}"
            status_code_val = "N/A"
            if e.response is not None:
                try:
                    status_code_val = int(e.response.status_code)
                except (ValueError, TypeError):
                    status_code_val = e.response.status_code
                    print(f"  AVISO: ChatGPT status code não é um inteiro ({type(status_code_val).__name__}): {status_code_val}")

            err_type = "(Unknown Error Type)"
            if isinstance(status_code_val, int):
                 if 400 <= status_code_val < 500: err_type = "(Client Error)"
                 elif status_code_val >= 500: err_type = "(Server Error)"
            elif status_code_val != "N/A":
                 err_type = f"(Non-int Status: {status_code_val})"
            else:
                 err_type = "(Network/Timeout/Other Error)"

            if e.response is not None:
                err_msg+=f" | Status:{status_code_val}{err_type}";
                try:
                    details=e.response.json(); api_err=details.get('error',{}).get('message',''); err_msg+=f" | API Err:{api_err}" if api_err else f" | Body:{details}";
                except json.JSONDecodeError:
                    err_msg+=f" | Body (non-JSON):{e.response.text[:100]}..."
                except Exception as json_e:
                     err_msg += f" | Error parsing error response: {json_e}"
            else:
                err_msg += f" | {err_type}"
            print(err_msg)
        except Exception as e: print(f"Erro inesperado ChatGPT: {type(e).__name__}-{e}")
        execution_time=time.time()-start_time; return {"content":content or "","execution_time":execution_time}

    def query_gemini(self, prompt: str) -> dict:
        """
        Envia um prompt para a API do Gemini e retorna a resposta.
        Inclui um sleep fixo para lidar com rate limiting (considerar backoff).

        Args:
            prompt (str): O prompt a ser enviado para a API.

        Returns:
            dict: Um dicionário contendo 'content' (a resposta da API) e
                  'execution_time' (o tempo de chamada da API em segundos).
        """
        # Nota: Para produção, usar bibliotecas como 'tenacity' para implementar
        #       exponential backoff em vez de time.sleep() fixo.

        start_time = time.time(); content = ""; execution_time = 0
        if not self.gemini_api_key or not self.gemini_model: print("Pulando Gemini: Chave/Modelo N/A."); return {"content": "", "execution_time": 0}
        try:
            # Usa self.gemini_model definido no __init__
            model=genai.GenerativeModel(self.gemini_model)
            safety=[{"category":c,"threshold":"BLOCK_NONE"} for c in ["HARM_CATEGORY_HARASSMENT","HARM_CATEGORY_HATE_SPEECH","HARM_CATEGORY_SEXUALLY_EXPLICIT","HARM_CATEGORY_DANGEROUS_CONTENT"]]
            response=model.generate_content(prompt,safety_settings=safety)
            execution_time=time.time()-start_time
            try: content=response.text
            except ValueError:
                 print("Resposta Gemini bloqueada ou sem texto.")
                 if hasattr(response, 'prompt_feedback'): print(f"  Feedback: {response.prompt_feedback}")
                 if hasattr(response, 'candidates') and response.candidates: print(f"  Reason: {response.candidates[0].finish_reason}")
        except google.api_core.exceptions.ResourceExhausted as e: print(f"Erro Gemini (Rate Limit): {e}"); execution_time=time.time()-start_time
        except Exception as e: print(f"Erro inesperado Gemini: {type(e).__name__}-{e}");
        if execution_time==0: execution_time=time.time()-start_time # Ensure execution_time is measured even if error occurs early
        return {"content":content or "","execution_time":execution_time}

    def extract_code(self, content: str) -> str:
        """
        Extrai o primeiro bloco de código Python de uma string de texto.
        Prioriza ```python, depois ``` genérico, e por fim tenta por indentação.

        Args:
            content (str): String contendo potencialmente o código.

        Returns:
            str: O bloco de código extraído ou string vazia.
        """
        if not content: return ""
        # Prioriza ```python ... ```
        m=re.search(r'```python\n(.*?)\n```',content,re.DOTALL);
        if m: return m.group(1).strip()
        # Tenta ``` ... ``` genérico
        m=re.search(r'```(?:.*\n)?(.*?)\n```',content,re.DOTALL);
        if m: return m.group(1).strip()
        # Fallback: tenta detectar por indentação ou keywords comuns
        lines=content.splitlines(); code=[]; start=False
        for ln in lines:
            s_ln=ln.strip()
            # Começa a capturar se linha tem indentação ou começa com def/class
            if not start and (ln.startswith((' ','\t')) or s_ln.startswith(('def ','class '))): start=True
            if start: code.append(ln)
        if code: return '\n'.join(code).strip()
        # Fallback final: se poucas linhas e sem def/class, retorna vazio, senão retorna tudo
        if len(lines)<3 and not ("def " in content or "class " in content): return ""
        return content.strip()

    def count_code_lines(self, code: str) -> int:
        """Conta linhas de código reais (não vazias, não comentários)."""
        if not code: return 0
        return len([ln for ln in code.splitlines() if ln.strip() and not ln.strip().startswith('#')])

    def measure_response_completeness(self, code: str, description: str) -> int:
        """
        Estima a 'completude' da resposta da IA (score 0-100) baseado
        na presença de código, estruturas e palavras-chave.
        """
        score=0; lines=self.count_code_lines(code)
        if lines > 0:
            score+=50
            if "def " in code or "class " in code: score+=10
            if "return " in code or "yield " in code: score+=10
            # Bonus específico por desafio
            dl=description.lower(); cl=code.lower(); bonus=0
            if "fibonacci" in dl and ("fibonacci" in cl or "fib(" in cl or "yield" in code): bonus=10
            elif "binary search" in dl and ("mid =" in code or "mid=" in code or "low" in code or "high" in code): bonus=10
            elif "merge sort" in dl and ("merge(" in cl or ("left" in code and "right" in code and "mid" in code)): bonus=10
            elif "island" in dl and ("dfs(" in cl or "bfs(" in cl or "visit" in cl or ("grid[" in code and "==" in code)): bonus=10
            elif "palindrome" in dl and ("[::-1]" in code or ".isalnum()" in cl or ".lower()" in cl): bonus=10
            elif "parentheses" in dl and ("stack" in cl or "append(" in code or "pop(" in code): bonus=10
            elif "lru cache" in dl and ("dict" in cl or "capacity" in cl or "ordereddict" in code.lower()): bonus=10 # check lower for OrderedDict
            score+=bonus;
            # Bonus por estruturas de controle
            struct=0
            if "if" in code or "for" in code or "while" in code: struct=10
            # Bonus extra por tratamento de edge cases comuns
            if struct>0 and ("if not" in code or "if len(" in code or "== 0" in code or "is None" in code or "else:" in code): struct+=5
            score+=min(struct,15)
        return min(score,100)

    def analyze_code_quality_flake8(self, code_string: str) -> int:
        """
        Analisa qualidade do código com Flake8.

        Args:
            code_string (str): O código Python.

        Returns:
            int: Número de problemas Flake8, ou -1 se erro/indisponível.
        """
        if not self.flake8_available or not code_string or self.count_code_lines(code_string)==0: return -1
        tfp=None
        try:
            # Usa NamedTemporaryFile para criar ficheiro temporário
            with tempfile.NamedTemporaryFile(mode='w+',suffix='.py',delete=False,encoding='utf-8') as tf:
                tf.write(code_string); tfp=tf.name
            # Executa flake8 como módulo python
            exe=sys.executable; res=subprocess.run([exe,'-m','flake8','--count',tfp],capture_output=True,text=True,timeout=15,check=False)
            # Processa resultado
            if res.returncode in [0,1] and res.stdout: # Flake8 retorna 1 se encontrar erros, 0 se não
                try: return int(res.stdout.strip().splitlines()[-1]) # Tenta pegar o contador no final
                except: return len(res.stdout.strip().splitlines()) # Fallback: conta linhas de output
            elif res.stderr: print(f"Erro flake8: {res.stderr}"); return -1
            else: return 0 # Sem output e sem erro = 0 issues
        except subprocess.TimeoutExpired: print("Erro: Flake8 timeout."); return -1
        except Exception as e: print(f"Erro inesperado flake8: {e}"); return -1
        finally:
            # Garante remoção do ficheiro temporário
            if tfp and os.path.exists(tfp):
                try: os.remove(tfp)
                except Exception as er: print(f"Aviso: Falha remover temp {tfp}: {er}")

    # --- Métodos Execução Local LRU Cache ---

    def _lru_worker(self, code_string: str, test_sequence: list, result_queue: multiprocessing.Queue):
        """
        Função worker para executar LRU Cache localmente com restrições.
        Executa a sequência de operações e coloca o resultado na fila.
        AVISO: Restrições de segurança são limitadas.

        Args:
            code_string (str): Código da classe LRUCache gerado pela IA.
            test_sequence (list): Lista de operações a executar.
            result_queue (multiprocessing.Queue): Fila para retornar resultados.
        """
        results = []; error_message = None; lru_instance = None; final_success = True
        try:
            # 1. Pré-processar código: Remover imports de collections/OrderedDict
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
                'TypeError': TypeError, 'StopIteration': StopIteration, 'KeyError': KeyError,
                '__build_class__': builtins.__build_class__, # Necessário para definir classes
                'getattr': getattr, 'isinstance': isinstance,
                '__name__': '__exec_sandbox__' # Identifica o ambiente
            }
            # Globals incluem builtins restritos e OrderedDict injetado
            safe_globals = {
                "__builtins__": safe_builtins,
                "OrderedDict": OrderedDict, # Permite que o código use OrderedDict se quiser
                "__name__": "__exec_sandbox__"
            }

            # 3. Executa o código da classe dentro do ambiente restrito
            local_namespace = {}
            exec(processed_code_string, safe_globals, local_namespace)

            # 4. Encontrar o nome da classe (assume que é LRUCache ou a primeira classe definida)
            class_name = None
            for name, obj in local_namespace.items():
                 # Procura por um tipo 'type' que não seja OrderedDict
                if isinstance(obj, type) and name != 'OrderedDict': class_name = name; break
            if not class_name: # Fallback: Tenta encontrar via Regex
                match = re.search(r"^\s*class\s+(\w+)", processed_code_string, re.MULTILINE)
                if match: class_name = match.group(1)
                else: raise NameError("Nome classe LRUCache não encontrado no código processado.")

            LruClass = local_namespace.get(class_name) # Tenta obter do namespace local
            if not LruClass: raise NameError(f"Classe '{class_name}' não definida no namespace local.")

            # 5. Executar sequência de testes
            op_total_time = 0.0
            for op_data in test_sequence:
                op_start_time = time.perf_counter()
                operation = op_data["op"]; args = op_data.get("args", []); expected = op_data.get("expected")
                step_res = {"op": operation, "args": args, "exp": expected, "out": None, "ok": False, "err": None, "t_ms": -1.0}
                try:
                    if operation == "__init__": lru_instance = LruClass(*args); step_res["ok"] = True; step_res["out"] = "OK"
                    elif lru_instance is None: raise RuntimeError("LRUCache não inicializada antes de chamar outros métodos.")
                    elif operation in ["put", "get"]:
                         if not hasattr(lru_instance, operation): raise AttributeError(f"Instância da classe '{class_name}' não tem método '{operation}'")
                         method = getattr(lru_instance, operation)
                         if operation == "put": method(*args); step_res["ok"] = True; step_res["out"] = "OK"
                         elif operation == "get":
                              output = method(*args); step_res["out"] = output
                              # Comparação como string para simplificar (pode não ser ideal para todos os tipos)
                              if str(output) == str(expected): step_res["ok"] = True
                              else: step_res["err"] = f"Output:'{output}' != Expected:'{expected}'"; final_success = False
                    else: raise ValueError(f"Operação LRU inválida na sequência de teste: {operation}")
                except Exception as e_step:
                    step_res["err"] = f"{type(e_step).__name__}: {str(e_step).splitlines()[0]}"; final_success = False
                    # print(f"      Erro passo LRU ({operation}): {step_res['err']}") # Debug opcional
                op_end_time = time.perf_counter(); step_res["t_ms"] = (op_end_time - op_start_time) * 1000
                op_total_time += (op_end_time - op_start_time); results.append(step_res)
                # Opcional: parar no primeiro erro
                # if not step_res["ok"] and operation != "__init__":
                #    print(f"      Parando sequência LRU devido a erro no passo: {step_res['err']}")
                #    break

            # print(f"    LRU Worker done. Total step time: {op_total_time*1000:.2f} ms") # Debug opcional

        except Exception as e_main:
            error_message = f"Erro worker LRU: {type(e_main).__name__} - {str(e_main).splitlines()[0]}"
            print(f"    ERRO FATAL NO WORKER LRU: {error_message}"); traceback.print_exc(); final_success = False

        # Coloca resultado (sucesso geral, lista de passos, erro) na fila
        try: result_queue.put({"success": final_success, "results": results, "error": error_message})
        except Exception as q_err: print(f"    ERRO ao colocar resultado na fila do worker LRU: {q_err}")


    def _execute_lru_locally(self, code_string: str, test_sequence: list) -> dict:
        """
        Gere o processo para execução local do LRU Cache, com timeout e
        recolha de resultados via Queue.

        Args:
            code_string (str): O código da classe.
            test_sequence (list): Lista de operações a testar.

        Returns:
            dict: Dicionário com {'success': bool, 'results': list, 'error': str|None}.
                  'results' contém detalhes de cada passo da sequência.
        """
        ctx = multiprocessing.get_context(None); result_queue = ctx.Queue(); timeout_seconds = 10
        process = ctx.Process(target=self._lru_worker, args=(code_string, test_sequence, result_queue))
        return_value = {"success": False, "results": [], "error": "Unknown execution error"}; terminated = False
        try:
            start_time = time.time(); process.start(); process.join(timeout=timeout_seconds)
            elapsed_time = time.time() - start_time
            if process.is_alive():
                print(f"  AVISO: Timeout ({timeout_seconds}s) atingido para processo LRU. Terminando...");
                process.terminate(); process.join(1) # Tenta terminar graciosamente
                if process.is_alive(): print("  AVISO: Terminate falhou, forçando kill processo LRU."); process.kill(); process.join() # Força se necessário
                return_value = {"success": False, "results": [], "error": f"Timeout ({timeout_seconds}s)"}; terminated = True

            if not terminated:
                exit_code = process.exitcode
                if exit_code == 0:
                    try:
                        result = result_queue.get(timeout=1);
                        # print(f"  Exec LRU local OK ({elapsed_time:.2f}s). Worker success: {result.get('success')}") # Debug opcional
                        return_value = result
                    except queue.Empty:
                         print("  ERRO: Processo LRU terminou OK, mas fila de resultados vazia."); return_value = {"success": False, "results": [], "error": "Worker finished OK but result queue empty"}
                    except Exception as e_q: print(f"  ERRO: Falha obter resultado da fila LRU: {e_q}"); return_value = {"success": False, "results": [], "error": f"Queue communication error: {e_q}"}
                else: # Processo terminou com erro
                    print(f"  ERRO: Processo worker LRU terminou inesperadamente com exitcode: {exit_code}")
                    # Tenta obter uma mensagem de erro da fila (o worker pode ter colocado antes de crashar)
                    try:
                        err_result = result_queue.get_nowait()
                        return_value = err_result
                        # Garante que erro está preenchido e success é False
                        if not return_value.get("error"): return_value["error"] = f"Worker process exited with code {exit_code}"
                        return_value["success"] = False
                    except queue.Empty: return_value = {"success": False, "results": [], "error": f"Worker process exited with code {exit_code} (no error message in queue)"}
                    except Exception as e_qf: return_value = {"success": False, "results": [], "error": f"Worker exit {exit_code}, queue fail getting error: {e_qf}"}
        except Exception as e_p:
             print(f"  ERRO: Falha ao gerir processo worker LRU: {e_p}"); return_value = {"success": False, "results": [], "error": f"Process management error: {e_p}"}
             if process and process.is_alive(): # Tenta matar se ainda vivo
                 try: process.kill(); process.join();
                 except: pass # Ignora erros ao tentar matar
        finally:
            # Limpeza da fila e processo
            try: result_queue.close(); result_queue.join_thread()
            except Exception as e_qc: print(f"  Aviso: Erro ao limpar fila LRU: {e_qc}")
            try:
                if process: process.close() # Libera recursos associados ao processo
            except Exception as e_pc: print(f"  Aviso: Erro ao limpar objeto Process LRU: {e_pc}")
        return return_value

    # --- Fim métodos execução local ---

    def evaluate_solution(self, ai_name: str, challenge: dict, response: dict) -> dict:
        """
        Avalia a solução de uma IA para um desafio.

        Extrai código, analisa com Flake8, mede completude, e executa testes
        (localmente para LRU Cache, externamente para os outros). Adiciona
        resultados detalhados à lista `self.detailed_test_results`.

        Args:
            ai_name (str): Nome da IA (e.g., "Claude").
            challenge (dict): Dicionário com detalhes do desafio.
            response (dict): Resposta da API {'content': str, 'execution_time': float}.

        Returns:
            dict: Dicionário com resultados agregados da avaliação para este AI/Desafio.
                  Usado para preencher `self.results`.
        """
        content = response.get("content", ""); api_time = response.get("execution_time", 0)
        code = self.extract_code(content); lines = self.count_code_lines(code)
        completeness = self.measure_response_completeness(code, challenge["description"])
        flake8 = self.analyze_code_quality_flake8(code)

        raw_results = []; details_curr = []; n_errors = 0; first_err = ""
        can_exec = bool(code); func_name = None

        # Tenta detetar nome da função/classe (exceto para LRU onde o nome é fixo)
        if can_exec and challenge["name"] != "LRU Cache":
            func_name = self._detect_function_name(code, challenge) # Passa challenge dict
            if not func_name:
                can_exec = False # Não pode executar sem nome

        # Proceder com execução se houver código e nome (ou se for LRU)
        if can_exec:
            if challenge["name"] == "LRU Cache":
                print(f"  Executando LRU Cache de {ai_name} localmente...")
                tc = challenge.get("test_cases", [])
                if tc and isinstance(tc[0], dict) and "sequence" in tc[0]:
                    # Executa localmente
                    lru_res = self._execute_lru_locally(code, tc[0]["sequence"])
                    raw_results = lru_res.get("results", []) # Lista de passos
                    ok = lru_res.get("success", False); n_errors = 0 if ok else 1
                    first_err = lru_res.get("error", "") # Erro geral do worker ou timeout
                    # Se não houve erro geral mas falhou, pega erro do primeiro passo falhado
                    if not first_err and not ok:
                        for step in raw_results:
                            if step.get("err"): first_err = f"Step {step['op']}{step['args']}: {step['err']}"; break
                    if not first_err and not ok: first_err = "LRU execution failed (unknown step error)" # Fallback
                    # Calcula tempo médio DOS PASSOS BEM SUCEDIDOS
                    t_ok=[s['t_ms'] for s in raw_results if s.get('t_ms',-1)>=0 and s.get('ok') is True]
                    avg_t=np.mean(t_ok) if t_ok else -1.0 # Média dos passos OK
                    # Adiciona UM registo agregado aos detalhes
                    details_curr.append({
                        "ai": ai_name, "challenge": challenge["name"], "difficulty": challenge["difficulty"],
                        "test_case_index": 0, "input_args_str": json.dumps(tc[0]["sequence"]),
                        "expected_output_str": "Sequence Result", "actual_output_str": f"Success: {ok}",
                        "status": "Pass" if ok else ("Fail" if raw_results else "Error"), # Status baseado no sucesso geral
                        "execution_time_ms": avg_t, # Média do tempo dos passos ok
                        "error_message": str(first_err)[:250], # Garante string e limita tamanho
                        "stdout": None # Stdout não capturado na execução local
                    })
                    print(f"  Resultado LRU local: {'Sucesso' if ok else 'Falha/Erro'}")
                else: # Caso de teste LRU mal formatado
                    print("  AVISO: Casos teste LRU inválidos."); n_errors = 1; first_err = "Invalid LRU test case format."
                    details_curr.append({"ai": ai_name, "challenge": challenge["name"], "difficulty": challenge["difficulty"],"test_case_index": 0, "input_args_str": "N/A", "expected_output_str": "N/A", "actual_output_str": None, "status": "Error", "execution_time_ms": -1, "error_message": first_err, "stdout": None })
            else: # Outros desafios (func_name deve ser válido aqui)
                print(f"  Executando '{challenge['name']}' de {ai_name} via Executor Service...")
                # Chama a função para executar testes externos
                raw, details, errors, first_e = self._run_test_cases_fixed(ai_name, challenge, code, func_name)
                raw_results = raw; n_errors = errors; first_err = first_e; details_curr.extend(details)
        else: # Sem código ou nome função (e não é LRU)
            err_msg = "Code not extracted" if not code else "Function/Class name not found"
            print(f"  AVISO: {err_msg} para '{challenge['name']}' por {ai_name}. Pulando execução.")
            n_tests = len(challenge.get("test_cases",[])) # Número de testes que seriam corridos
            n_errors = n_tests; first_err = err_msg # Marca todos como erro
            for i in range(n_tests):
                 details_curr.append({"ai": ai_name, "challenge": challenge["name"], "difficulty": challenge["difficulty"], "test_case_index": i, "input_args_str": None, "expected_output_str": None, "actual_output_str": None, "status": "No Code", "execution_time_ms": -1, "error_message": err_msg, "stdout": None })

        # Adiciona detalhes gerados (seja de LRU, externo ou erro 'No Code') à lista principal
        self.detailed_test_results.extend(details_curr)

        # Retorna dicionário agregado para self.results
        return {
            "ai": ai_name, "challenge": challenge["name"], "difficulty": challenge["difficulty"],
            "api_execution_time": api_time, "lines_of_code": lines,
            "completeness": completeness, "flake8_issues": flake8,
            "num_execution_errors": n_errors, # Número total de erros/falhas nos casos de teste (ou 1 para LRU se falhar, ou n_tests se 'No Code')
            "first_execution_error": first_err, # Primeiro erro encontrado
            "code": code, # Código extraído
            "full_response": content, # Resposta completa da API
            "execution_results_raw": raw_results # Resultados brutos da execução (lista de dicts)
        }

    def _detect_function_name(self, code: str, challenge: dict) -> str | None:
        """Detecta nome da função/classe, priorizando assinatura e nome esperado."""
        if not code: return None
        expected_name = None
        signature = challenge.get("function_signature", "")
        # Tenta extrair nome da assinatura (def nome(... ou class nome:)
        sig_match = re.match(r"^\s*(?:def|class)\s+(\w+)", signature)

        if sig_match:
            expected_name = sig_match.group(1)
            # 1. Tenta encontrar definição exata no início da linha
            simple_def = f"def {expected_name}("
            simple_class = f"class {expected_name}:"
            simple_class_paren = f"class {expected_name}(" # Considera herança ou parênteses vazios
            # Usa re.MULTILINE para ^ funcionar em cada linha
            if re.search(rf"^\s*{re.escape(simple_def)}", code, re.MULTILINE) or \
               re.search(rf"^\s*{re.escape(simple_class)}", code, re.MULTILINE) or \
               re.search(rf"^\s*{re.escape(simple_class_paren)}", code, re.MULTILINE):
                return expected_name

            # 2. Tenta encontrar definição com regex menos restrita (não necessariamente no início da linha)
            #    Usa \b para garantir que é a palavra completa (evita match parcial)
            pattern = rf"\s*(?:def|class)\s+{re.escape(expected_name)}\b"
            if re.search(pattern, code, re.MULTILINE):
                return expected_name

            # 3. Se não encontrou definição, verifica se o NOME existe no código
            #    Isso ajuda se a IA usou o nome mas não na definição padrão.
            if re.search(rf"\b{re.escape(expected_name)}\b", code):
                print(f"  AVISO: Nome esperado '{expected_name}' encontrado, mas não como definição padrão. Usando '{expected_name}'.")
                return expected_name

        # 4. Fallback: pega a primeira definição (def ou class) encontrada no código
        first_match = re.search(r"^\s*(?:def|class)\s+(\w+)", code, re.MULTILINE)
        if first_match:
            found_name = first_match.group(1)
            # Só avisa se for diferente do esperado (se havia um esperado)
            if expected_name and found_name != expected_name:
                 print(f"  AVISO: Usando 1ª definição encontrada '{found_name}' (Esperado: '{expected_name}').")
            # Se não havia nome esperado, não imprime aviso ao pegar o primeiro
            elif not expected_name:
                 print(f"  AVISO: Usando 1ª definição encontrada '{found_name}'.")
            return found_name

        # Se NADA foi encontrado
        print(f"  ERRO: Nenhuma definição de função ou classe encontrada para {challenge['name']}.")
        return None

    def _run_test_cases_fixed(self, ai_name: str, challenge: dict, code: str,
                            actual_function_name: str) -> tuple[list, list, int, str]:
        """Executa casos de teste via Executor Service."""
        raw_results = []; details = []; n_fails_errors = 0; first_err = ""
        test_cases = challenge.get("test_cases", [])
        if not test_cases: print(f"  Aviso: Sem casos de teste para '{challenge['name']}'."); return [], [], 0, ""

        for i, tc in enumerate(test_cases):
            args = tc.get("input_args"); exp_out = tc.get("expected_output")
            if args is None: print(f"  Aviso: Caso {i+1} '{challenge['name']}' sem 'input_args'. Pulando."); continue

            # Chama o serviço executor para este caso de teste
            exec_res = self._call_executor_service(code, actual_function_name, args)
            raw_results.append(exec_res) # Guarda resultado bruto

            # Prepara dicionário detalhado para este caso
            t_det = {
                "ai": ai_name, "challenge": challenge["name"], "difficulty": challenge["difficulty"],
                "test_case_index": i, "input_args_str": json.dumps(args),
                "expected_output_str": json.dumps(exp_out), "actual_output_str": None,
                "status": "Unknown", # Default
                "execution_time_ms": exec_res.get("execution_time_ms", -1),
                "error_message": None, "stdout": exec_res.get("stdout")
            }
            case_err = None # Erro específico deste caso

            # Avalia resultado da execução
            if exec_res.get("success") is True:
                act_out = exec_res.get("output")
                t_det["actual_output_str"] = json.dumps(act_out)
                try:
                    # Comparação usando numpy.array_equal para lidar com listas/arrays aninhados
                    # Nota: Pode ter limitações com floats imprecisos ou ordem de dicionários.
                    # Convertendo para array de 'object' para comparação mais genérica.
                    are_equal = np.array_equal(np.array(act_out, dtype=object), np.array(exp_out, dtype=object))
                    if are_equal:
                        t_det["status"] = "Pass"
                    else:
                        t_det["status"] = "Fail"
                        case_err = f"Output != Expected (Case {i})"
                        n_fails_errors += 1 # Incrementa para falha de comparação
                except Exception as comp_e:
                    # Erro durante a comparação em si
                    print(f"  AVISO: Erro ao comparar outputs {ai_name}/{challenge['name']}/Caso {i}: {comp_e}")
                    t_det["status"] = "Fail (Comparison Error)"
                    t_det["error_message"] = f"Error comparing outputs: {comp_e}"
                    case_err = t_det["error_message"]
                    n_fails_errors += 1 # Incrementa para erro de comparação

            elif exec_res.get("success") is False:
                # Erro reportado pelo executor service
                t_det["status"] = "Error"
                t_det["error_message"] = exec_res.get("error", "Execution error reported by service")
                case_err = str(t_det["error_message"])[:250] # Limita tamanho
                n_fails_errors += 1
            else:
                # Resposta inesperada do executor (sem campo 'success')
                t_det["status"] = "Unknown Executor Status"
                t_det["error_message"] = exec_res.get("error", "Unexpected response from executor service")
                case_err = str(t_det["error_message"])[:250]
                n_fails_errors += 1

            # Guarda o primeiro erro/falha específico encontrado
            if case_err and not first_err:
                first_err = case_err

            details.append(t_det) # Adiciona detalhes deste caso de teste

        # Mensagem genérica se houve falhas mas nenhuma específica foi capturada
        if n_fails_errors > 0 and not first_err:
            first_err = "One or more test cases failed or errored."

        return raw_results, details, n_fails_errors, first_err


    def _call_executor_service(self, code_string: str, function_name: str, input_args: list) -> dict:
        """Chama o serviço executor externo."""
        # Adiciona imports comuns que podem ser necessários pelo código gerado
        imports = (
            "from typing import List, Dict, Tuple, Set, Optional, Any\n"
            "import re\n"
            "from collections import OrderedDict, deque\n\n"
        )
        code_to_exec = imports + code_string
        payload = {"code_string": code_to_exec, "function_name": function_name, "test_input_args": input_args}
        headers = {"Content-Type": "application/json"}
        default_err = {"success": False, "output": None, "stdout": None, "error": "Failed to call executor service", "execution_time_ms": -1}

        try:
            resp = requests.post(self.executor_url, headers=headers, json=payload, timeout=60) # Timeout de 60s
            if resp.status_code == 200:
                try: return resp.json() # Retorna resposta JSON se sucesso
                except json.JSONDecodeError as json_e: print(f"  Erro JSON decode executor: {json_e}"); default_err["error"] = f"Executor JSON decode error: {json_e}"; return default_err
            else: # Erro HTTP (e.g., 4xx, 5xx)
                err_txt = resp.text[:200] # Pega início do texto do erro
                print(f"  Erro executor status {resp.status_code}: {err_txt}..."); default_err["error"] = f"Executor status {resp.status_code}: {err_txt}"; return default_err
        except requests.exceptions.Timeout: print(f"  Erro Timeout ao chamar executor ({self.executor_url})."); default_err["error"] = "Executor service request timeout"; return default_err
        except requests.exceptions.RequestException as req_e: print(f"  Erro conexão/rede com executor: {req_e}"); default_err["error"] = f"Executor connection error: {req_e}"; return default_err
        except Exception as e: print(f"  Erro inesperado ao chamar executor: {type(e).__name__} - {e}"); default_err["error"] = f"Unexpected error calling executor: {type(e).__name__}"; return default_err

    def run_benchmark(self):
        """Executa o ciclo completo do benchmark."""
        can_claude=bool(self.claude_api_key); can_openai=bool(self.openai_api_key); can_gemini=bool(self.gemini_api_key and self.gemini_model)
        enabled_ais=[n for n,can in [("Claude",can_claude),("ChatGPT",can_openai),("Gemini",can_gemini)] if can]
        if not enabled_ais: print("ERRO: Nenhuma API configurada ou disponível. Saindo."); return

        print(f"Iniciando Benchmark para: {', '.join(enabled_ais)}")
        print(f"  Claude Model: {self.claude_model if can_claude else 'N/A'}")
        print(f"  ChatGPT Model: {self.openai_model if can_openai else 'N/A'}")
        print(f"  Gemini Model: {self.gemini_model or 'N/A'}")
        print(f"  LRU Cache Execution: Local")
        print(f"  Executor Service URL: {self.executor_url}")
        print("="*42)

        # Dicionário com descrições técnicas dos desafios
        ch_details = {
            "Fibonacci Sequence": "Técnica: Geração de sequência numérica onde cada número é a soma dos dois anteriores (pode ser implementado iterativamente ou recursivamente).\n  Finalidade: Avaliar a capacidade de implementar algoritmos fundamentais e lidar com casos base.",
            "Binary Search":      "Técnica: Algoritmo eficiente de busca (Dividir e Conquistar) que opera sobre listas ordenadas, dividindo o espaço de busca pela metade a cada passo.\n  Finalidade: Testar a implementação de algoritmos de busca e lógica de comparação em estruturas ordenadas.",
            "Merge Sort":         "Técnica: Algoritmo de ordenação eficiente (Dividir e Conquistar) que divide a lista recursivamente e depois combina (merge) as sub-listas ordenadas.\n  Finalidade: Avaliar a implementação de algoritmos de ordenação recursivos e a lógica de combinação.",
            "Count Islands in Grid": "Técnica: Travessia de grafo (usando Busca em Profundidade - DFS ou Busca em Largura - BFS) para explorar conexões numa matriz 2D (grid).\n  Finalidade: Testar a capacidade de modelar problemas como grafos e aplicar algoritmos de travessia para contagem.",
            "Palindrome Check":   "Técnica: Manipulação de strings para verificar se uma sequência de caracteres é a mesma lida de trás para frente, ignorando não alfanuméricos e caixa.\n  Finalidade: Avaliar o processamento e limpeza de strings, e a lógica de comparação.",
            "Valid Parentheses":  "Técnica: Uso de estrutura de dados tipo Pilha (Stack) para verificar se os parênteses/chaves/colchetes numa string abrem e fecham na ordem correta.\n  Finalidade: Testar o uso de pilhas para validação de sequências estruturadas.",
            "LRU Cache":          "Técnica: Implementação de uma estrutura de dados customizada (Cache com política 'Least Recently Used' - LRU) usando dicionários (ou OrderedDict) para acesso rápido e manutenção da ordem de uso.\n  Finalidade: Avaliar a capacidade de criar estruturas de dados complexas com políticas específicas de gerenciamento."
        }

        for chal in self.code_challenges:
            sig=chal.get("function_signature",f"{chal['name'].lower().replace(' ','_')}(...)")
            # Prompt mais direto pedindo apenas código
            prompt = (
                f"Implement the following Python {'class' if 'class' in sig else 'function'}.\n"
                f"Description: {chal['description']}\n"
                f"Signature:\n{sig}\n"
                f"Provide ONLY the Python code definition for the function/class and any necessary standard library imports (like `from collections import OrderedDict` if needed). "
                f"Do not include any explanations, comments within the code, examples, or markdown formatting like ```python."
            )

            print(f"\n--- Desafio: {chal['name']} ({chal['difficulty']}) ---")
            print(f"  {ch_details.get(chal['name'],'Explicação técnica N/A.')}") # Mostra detalhes técnicos

            resps={} # Armazena respostas das APIs
            if can_claude: print("Consultando Claude AI..."); resps["Claude"]=self.query_claude(prompt); time.sleep(1)
            if can_openai: print("Consultando ChatGPT..."); resps["ChatGPT"]=self.query_chatgpt(prompt); time.sleep(1)
            if can_gemini: print("Consultando Gemini AI..."); resps["Gemini"]=self.query_gemini(prompt); print("Aguardando rate limit API Gemini (31s)..."); time.sleep(31) # Espera após Gemini

            summary=[] # Resumo dos resultados deste desafio
            for ai in enabled_ais:
                resp_data=resps.get(ai,{"content":"","execution_time":0}) # Pega resposta ou default
                if not resp_data.get("content"): # Se a API falhou ou retornou vazio
                    print(f"  AVISO: Sem resposta válida de {ai} para {chal['name']}. Pulando avaliação.")
                    # Adiciona um resultado de falha para esta IA/desafio
                    fail_result = {
                         "ai": ai, "challenge": chal["name"], "difficulty": chal["difficulty"],
                         "api_execution_time": resp_data.get("execution_time", 0), "lines_of_code": 0,
                         "completeness": 0, "flake8_issues": -1,
                         "num_execution_errors": len(chal.get("test_cases", [])), # Marca todos os testes como erro
                         "first_execution_error": "API Error or No Content",
                         "code": "", "full_response": "", "execution_results_raw": []
                    }
                    self.results.append(fail_result)
                    # Adiciona detalhes de erro para cada caso de teste
                    for i, tc in enumerate(chal.get("test_cases", [])):
                         self.detailed_test_results.append({
                             "ai": ai, "challenge": chal["name"], "difficulty": chal["difficulty"],
                             "test_case_index": i if chal['name'] != "LRU Cache" else 0, # Index 0 para LRU agregado
                             "input_args_str": json.dumps(tc.get("input_args")) if chal['name'] != "LRU Cache" else json.dumps(tc.get("sequence")),
                             "expected_output_str": json.dumps(tc.get("expected_output")) if chal['name'] != "LRU Cache" else "Sequence Result",
                             "actual_output_str": None, "status": "API Error", "execution_time_ms": -1,
                             "error_message": "API Error or No Content", "stdout": None
                         })
                    summary.append(f"  {ai:<10}: API ERROR/No Content. Flake8: N/A. API Time: {fail_result['api_execution_time']:.2f}s.")
                    continue # Pula para próxima IA

                # Avalia a solução se a API respondeu
                result=self.evaluate_solution(ai, chal, resp_data); self.results.append(result)

                # Prepara linha de resumo
                n_err_fail=result.get('num_execution_errors',0); # Num erros/falhas na execução
                # Pega detalhes específicos deste AI/Challenge
                det=[d for d in self.detailed_test_results if d['ai']==ai and d['challenge']==chal['name']]
                # Calcula status agregado da execução
                n_att=len([d for d in det if d['status'] not in ['No Code', 'API Error']]) # Tentativas válidas
                n_pass=len([d for d in det if d['status']=='Pass']) # Passaram
                f8=(str(result['flake8_issues']) if result['flake8_issues']!=-1 else 'N/A') # Issues Flake8

                if n_att == 0: status="Not Run (No Code/API Error)"
                elif chal['name']=="LRU Cache": status=det[0]['status'] if det else "Error Evaluating" # Status do teste LRU único
                else: status=f"{n_pass}/{n_att} passed" # Status para múltiplos casos

                summary.append(f"  {ai:<10}: Exec: {status}. Flake8: {f8}. API Time: {result['api_execution_time']:.2f}s.")

            print(f"--- Resumo Desafio {chal['name']} ---")
            for s_line in summary: print(s_line)
            print(f"--- Desafio {chal['name']} concluído ---")

        if not self.results: print("\nNenhum resultado coletado durante o benchmark."); return
        self.generate_report() # Gera relatórios no final

    # --- Métodos de Geração de Relatório ---

    def _prepare_dataframe(self) -> pd.DataFrame | None:
        """Prepara DataFrame principal a partir de self.results."""
        if not self.results: print("Não há resultados (self.results) para gerar DataFrame."); return None
        try:
             df = pd.DataFrame(self.results)
             # Calcula métricas agregadas dos testes detalhados
             agg_metrics = self._calculate_aggregated_test_metrics()
             # Mapeia as métricas agregadas de volta para o DataFrame principal
             df['avg_exec_success_rate']=df.apply(lambda r: agg_metrics.get((r['ai'], r['challenge']), {}).get('avg_exec_success_rate',0.0), axis=1)
             df['avg_exec_time_ms']=df.apply(lambda r: agg_metrics.get((r['ai'], r['challenge']), {}).get('avg_exec_time_ms',np.nan), axis=1)
             df['total_tests_passed']=df.apply(lambda r: agg_metrics.get((r['ai'], r['challenge']), {}).get('total_tests_passed',0), axis=1)
             df['total_tests_attempted']=df.apply(lambda r: agg_metrics.get((r['ai'], r['challenge']), {}).get('total_tests_attempted',0), axis=1)
             # Remove colunas grandes/complexas para o sumário
             df_clean = df.drop(['execution_results_raw', 'code', 'full_response'], axis=1, errors='ignore')
             # Normaliza tipos de dados
             self._normalize_dataframe_types(df_clean)
             return df_clean
        except Exception as e: print(f"Erro ao criar/processar DataFrame principal: {e}"); traceback.print_exc(); return None

    def _calculate_aggregated_test_metrics(self) -> dict:
        """Calcula métricas agregadas a partir de self.detailed_test_results."""
        agg = {}
        # Cria chaves únicas (ai, challenge) a partir dos resultados detalhados
        keys = set((d['ai'], d['challenge']) for d in self.detailed_test_results)

        for k in keys:
            ai, chal = k
            # Filtra detalhes para esta combinação AI/Challenge
            dets = [d for d in self.detailed_test_results if d['ai']==ai and d['challenge']==chal]
            if not dets: continue

            # Filtra casos que passaram e casos que foram tentados (exclui 'No Code', 'API Error')
            passed=[d for d in dets if d['status']=='Pass']
            attempted=[d for d in dets if d['status'] not in ['No Code', 'API Error']]
            # Filtra tempos de execução válidos (>= 0)
            times=[d['execution_time_ms'] for d in dets if d.get('execution_time_ms',-1)>=0]

            n_att=len(attempted); n_pass=len(passed)
            # Calcula taxa de sucesso e tempo médio
            rate=n_pass/n_att if n_att>0 else 0.0
            mean_t=np.mean(times) if times else np.nan # Média dos tempos válidos

            agg[k]={'avg_exec_success_rate': rate,'avg_exec_time_ms':mean_t,'total_tests_passed':n_pass,'total_tests_attempted':n_att}
        return agg

    def _normalize_dataframe_types(self, df: pd.DataFrame):
        """Normaliza tipos de colunas numéricas e trata NaNs."""
        numeric_cols = [
            'api_execution_time','lines_of_code','completeness','flake8_issues',
            'num_execution_errors','avg_exec_success_rate','avg_exec_time_ms',
            'total_tests_passed','total_tests_attempted'
        ]
        for c in numeric_cols:
            if c in df.columns:
                df[c]=pd.to_numeric(df[c],errors='coerce') # Converte para numérico, erros viram NaN
                fill_value=0.0 # Default fill
                target_dtype=float # Default type

                if c=='flake8_issues': fill_value=-1; target_dtype=int
                elif c in ['lines_of_code','completeness','num_execution_errors','total_tests_passed','total_tests_attempted']:
                     fill_value=0; target_dtype=int
                elif c=='avg_exec_time_ms': fill_value=-1.0 # Usa -1 para indicar que não rodou ou falhou

                # Preenche NaNs APÓS conversão para numérico
                df.loc[:,c]=df[c].fillna(fill_value)

                # Tenta converter para Inteiro se apropriado (após preencher NaN)
                if target_dtype == int:
                    try:
                        # Usa astype(int) que funciona bem se não houver mais NaNs
                        df.loc[:,c]=df[c].astype(int)
                    except ValueError as e:
                         # Se ainda houver algo não conversível (raro após fillna), avisa
                         print(f"Aviso: Falha converter coluna '{c}' para int após fillna: {e}")
                         df.loc[:,c]=df[c].astype(float) # Mantém como float
            else:
                 print(f"Aviso: Coluna numérica esperada '{c}' não encontrada no DataFrame para normalização.")

        # Garante que coluna de erro é string e preenche NaNs
        if 'first_execution_error' in df.columns:
            df['first_execution_error']=df['first_execution_error'].fillna('').astype(str)
        elif 'first_execution_error' not in df.columns and 'num_execution_errors' in df.columns:
             # Adiciona coluna se não existir (caso nenhuma falha ocorra)
             df['first_execution_error'] = ''


    def _save_environment_info(self, ts: str):
        """Salva info do ambiente em TXT."""
        p = os.path.join(self.output_dir, f"environment_info_{ts}.txt")
        try:
            with open(p, "w", encoding='utf-8') as f:
                for k, v in self.environment_info.items():
                    if isinstance(v, dict):
                        f.write(f"{k}:\n")
                        for sk, sv in v.items(): f.write(f"  {sk}: {sv}\n")
                    else: f.write(f"{k}: {v}\n")
            print(f"- Info Ambiente salvo: {p}")
        except Exception as e: print(f"Erro ao salvar info do ambiente: {e}")

    def _save_full_results_json(self, ts: str):
        """Salva resultados completos (raw) em JSON."""
        p=os.path.join(self.output_dir, f"benchmark_results_full_{ts}.json")
        try:
            # Encoder customizado para lidar com tipos numpy e datetime
            class NpEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer): return int(obj)
                    elif isinstance(obj, np.floating): return float(obj)
                    elif isinstance(obj, np.ndarray): return obj.tolist()
                    elif isinstance(obj, datetime): return obj.isoformat()
                    elif isinstance(obj, np.bool_): return bool(obj)
                    elif isinstance(obj, bytes): return obj.decode('utf-8','replace')
                    return super(NpEncoder, self).default(obj)

            with open(p,"w",encoding='utf-8') as f:
                 json.dump(self.results,f,indent=2,ensure_ascii=False,cls=NpEncoder)
            print(f"- Resultados JSON (completos) salvo: {p}")
        except Exception as e: print(f"Erro ao salvar JSON completo: {e}")

    def _save_detailed_tests_csv(self, ts: str):
        """Salva detalhes dos casos de teste individuais em CSV."""
        p=os.path.join(self.output_dir, f"benchmark_test_details_{ts}.csv")
        if self.detailed_test_results:
            try:
                df=pd.DataFrame(self.detailed_test_results)
                # Define colunas e ordem desejada
                cols=['ai','challenge','difficulty','test_case_index','status','execution_time_ms','error_message','stdout','input_args_str','expected_output_str','actual_output_str']
                # Adiciona colunas faltantes se necessário (caso raro)
                for c in cols:
                    if c not in df.columns: df[c]=pd.NA
                # Salva CSV com colunas na ordem definida
                df[cols].to_csv(p,index=False,encoding='utf-8', float_format='%.3f')
                print(f"- Detalhes dos Testes CSV salvo: {p}")
            except Exception as e: print(f"Erro ao salvar CSV de detalhes dos testes: {e}")
        else: print("- Nenhum resultado detalhado de teste para salvar em CSV.")

    def _create_csv_header(self) -> str:
        """Cria cabeçalho com info do ambiente para CSVs."""
        hdr=["# --- Environment Info ---"]
        for k,v in self.environment_info.items():
            if isinstance(v,dict): hdr.append(f"# {k}:"); [hdr.append(f"#   {sk}: {sv}") for sk,sv in v.items()]
            else: hdr.append(f"# {k}: {v}")
        hdr.append("# --- Benchmark Data ---"); return "\n".join(hdr)+"\n"

    def _save_summary_csv(self, df: pd.DataFrame, ts: str):
        """Salva resumo agregado (sem código/raw) em CSV."""
        p=os.path.join(self.output_dir, f"benchmark_summary_{ts}.csv")
        if df is None: print("Erro: DataFrame de resumo é None. Pulando salvamento CSV."); return
        try:
            # Seleciona colunas relevantes para o sumário
            summary_cols = [c for c in df.columns if c not in ['code','full_response', 'execution_results_raw']] # Exclui colunas grandes
            summary_df = df[summary_cols].copy()
            hdr=self._create_csv_header(); # Pega cabeçalho formatado
            # Salva com cabeçalho e formatação
            with open(p,"w",encoding='utf-8',newline='') as f:
                 f.write(hdr); summary_df.to_csv(f,index=False,encoding='utf-8',float_format='%.4f',lineterminator='\n')
            print(f"- Resumo Agregado CSV salvo: {p}")
        except Exception as e: print(f"Erro ao salvar resumo CSV: {e}")

    def _calculate_statistics(self, df: pd.DataFrame) -> pd.DataFrame | None:
        """Calcula estatísticas agregadas por IA."""
        if df is None: print("Erro: DataFrame de entrada para estatísticas é None."); return None
        try:
            # Define métricas a calcular: (Nome Exibição, Coluna Original, Tipo Agregação)
            metrics_def = [
                ("Avg API Time (s)","api_execution_time","mean"),
                ("Avg LoC","lines_of_code","mean"),
                ("Avg Completeness (%)","completeness","mean"),
                ("Avg Flake8 Issues","flake8_issues","mean_valid"), # Média ignorando -1
                ("Total Passed","total_tests_passed","sum"),
                ("Total Attempted","total_tests_attempted","sum"),
                ("Overall Success (%)","avg_exec_success_rate","mean_percentage"), # Média das taxas * 100
                ("Avg Exec Time (ms)","avg_exec_time_ms","mean_valid"), # Média ignorando -1 / NaN
                ("Total Errors/Fails","num_execution_errors","sum"),
                ("Example Error","first_execution_error","first_non_empty") # Pega primeiro erro não vazio
            ]

            ais=sorted(df['ai'].unique())
            stats_data={"Metric":[m[0] for m in metrics_def]} # Coluna com nomes das métricas

            for ai in ais:
                df_ai=df[df["ai"]==ai]; col_data=[] # Filtra dados para a IA atual
                for _,col_key,agg_type in metrics_def:
                    value = np.nan # Valor padrão
                    if col_key in df_ai.columns:
                        series = df_ai[col_key]
                        if agg_type=="mean": value = series.mean()
                        elif agg_type=="sum": value = series.sum()
                        elif agg_type=="mean_valid":
                             # Calcula média apenas de valores válidos (não NaN e não -1)
                             valid_data = series[series.notna() & (series != -1)]; value = valid_data.mean() if not valid_data.empty else np.nan
                        elif agg_type=="mean_percentage":
                             # Calcula média e multiplica por 100
                             value = series.mean() * 100 if series.notna().any() else 0.0
                        elif agg_type=="first_non_empty":
                             # Pega o primeiro valor não vazio/não nulo na série
                             errors = series[series.fillna('').astype(str)!='']; value = errors.iloc[0] if not errors.empty else ""
                    col_data.append(value) # Adiciona valor calculado para esta IA/métrica
                stats_data[ai] = col_data # Adiciona coluna de resultados para esta IA

            return pd.DataFrame(stats_data) # Retorna DataFrame de estatísticas
        except Exception as e: print(f"Erro ao calcular estatísticas: {e}"); traceback.print_exc(); return None

    def _save_statistics_csv(self, stats_df: pd.DataFrame, ts: str):
        """Salva estatísticas agregadas em CSV."""
        if stats_df is None: print("Erro: DataFrame de estatísticas é None. Pulando salvamento CSV."); return
        p = os.path.join(self.output_dir, f"benchmark_stats_{ts}.csv")
        try:
            hdr=self._create_csv_header();
            with open(p,"w",encoding='utf-8',newline='') as f:
                 f.write(hdr); stats_df.to_csv(f,index=False,encoding='utf-8',float_format='%.2f',lineterminator='\n')
            print(f"- Estatísticas Agregadas CSV salvo: {p}")
        except Exception as e: print(f"Erro ao salvar estatísticas CSV: {e}")

    def _display_summary_table(self, stats_df: pd.DataFrame):
        """Mostra tabela de resumo formatada no console."""
        print("\n===== SUMÁRIO DO BENCHMARK (ESTATÍSTICAS) =====")
        if stats_df is None or stats_df.empty: print("Não foi possível gerar sumário (DataFrame vazio ou None)."); return

        disp=stats_df.copy()
        # Formata colunas numéricas para exibição
        for col_name in disp.columns:
            if col_name=='Metric': continue # Ignora coluna de métricas
            formatted_values=[]
            for index, row in disp.iterrows():
                metric_name = row['Metric']; raw_value = row[col_name]; fmt_value = "N/A" # Default
                if not pd.isna(raw_value):
                    try: # Formatação específica por métrica
                        if "Flake8" in metric_name: fmt_value=f"{raw_value:.1f}" if raw_value != -1 else "N/A"
                        elif "Time (ms)" in metric_name: fmt_value=f"{raw_value:.1f}" if raw_value != -1 else "N/A"
                        elif "(%)" in metric_name: fmt_value=f"{raw_value:.1f}%"
                        elif "Total " in metric_name: fmt_value=f"{int(raw_value)}"
                        elif "Time (s)" in metric_name: fmt_value=f"{raw_value:.2f}"
                        elif "Example Error" in metric_name: s=str(raw_value); fmt_value=s[:60]+('...'if len(s)>60 else '') # Limita tamanho
                        else: fmt_value=f"{raw_value:.1f}" # Default para LoC, etc.
                    except (ValueError, TypeError): fmt_value = str(raw_value) # Fallback se formatação falhar
                formatted_values.append(fmt_value)
            disp[col_name]=formatted_values # Atualiza coluna com valores formatados

        # Imprime usando tabulate se disponível, senão usa to_string
        try:
            from tabulate import tabulate; print(tabulate(disp,headers='keys',tablefmt='grid',showindex=False))
        except ImportError: print("(Instale 'tabulate' para tabela formatada)"); print(disp.to_string(index=False))
        except Exception as e: print(f"Erro ao imprimir tabela com tabulate: {e}"); print(disp.to_string(index=False)) # Fallback

    def generate_report(self):
        """Gera todos os relatórios e visualizações."""
        if not self.results: print("Não há resultados para gerar relatório."); return
        print("\n--- Gerando Relatórios ---"); ts=datetime.now().strftime("%Y%m%d_%H%M%S"); print(f"Timestamp: {ts}"); print(f"Diretório: '{self.output_dir}/'")
        self._save_environment_info(ts) # Salva info do ambiente

        df=self._prepare_dataframe() # Prepara DF agregado principal
        if df is None: print("Falha ao preparar DataFrame principal. Abortando geração de relatórios."); return

        # Salva os diferentes arquivos de dados
        self._save_full_results_json(ts); self._save_detailed_tests_csv(ts); self._save_summary_csv(df,ts)

        # Calcula e salva estatísticas
        stats=self._calculate_statistics(df)
        if stats is not None:
             self._save_statistics_csv(stats,ts)
             # Gera visualizações APENAS se stats foram calculados (implica df era válido)
             try:
                 print("\n--- Gerando Visualizações ---")
                 self.create_visualizations(df, ts, self.output_dir) # Passa o df agregado
                 print(f"- Visualizações salvas em '{self.output_dir}/'")
             except Exception as e: print(f"Erro ao gerar visualizações: {e}")
             # Exibe a tabela de sumário no console
             self._display_summary_table(stats)
        else:
             print("- Não foi possível calcular/salvar estatísticas.")
             print("AVISO: Tabela de sumário e visualizações podem não ter sido geradas devido a erro no cálculo de estatísticas.")


    # --- Funções de Plotagem ---

    def _plot_bar_charts(self, df_plot, ai_list, cmap, challenge_order, ts, out_dir):
        """Gera gráficos de barras comparativos por desafio."""
        print("Gerando Gráficos de Barras...")
        metrics_def = {
            "1_api_t":("API Time","s","api_execution_time"),
            "2_loc":("LoC","Linhas","lines_of_code"),
            "3_comp":("Completude","(%)","completeness"),
            "4_f8":("Flake8 Issues","Issues","flake8_issues"),
            "7_err":("Exec Errors/Fails","Num Errors/Fails","num_execution_errors") # Renomeado ylabel
        }
        # Adiciona métricas opcionais se existirem e tiverem dados válidos
        if 'avg_exec_success_rate' in df_plot.columns and df_plot['avg_exec_success_rate'].notna().any():
             metrics_def["5_ok"]=("Exec Success Rate","Success Rate (%)","avg_exec_success_rate")
        if 'avg_exec_time_ms' in df_plot.columns and (df_plot['avg_exec_time_ms'] != -1).any():
             metrics_def["6_exec_t"]=("Avg Exec Time (Steps)","Time (ms)","avg_exec_time_ms")
        metrics_def = dict(sorted(metrics_def.items())) # Ordena por chave numérica

        for key, (title, ylabel, col_name) in metrics_def.items():
            if col_name not in df_plot.columns: print(f"  Skip bar '{title}': Coluna '{col_name}' não encontrada."); continue

            data=df_plot.copy()
            # Aplica filtros específicos da métrica antes de pivotar
            if col_name=='flake8_issues': data=data[data[col_name] != -1].copy()
            elif col_name=='avg_exec_time_ms': data=data[data[col_name].notna() & (data[col_name] != -1)].copy()
            elif col_name=='num_execution_errors': data=data[data[col_name].notna()].copy()
            elif col_name=='avg_exec_success_rate': data.loc[:,col_name]=data[col_name]*100 # Converte para percentagem

            if data.empty or data[col_name].isnull().all(): print(f"  Skip bar '{title}': Sem dados válidos após filtro."); continue

            try:
                plt.figure() # Cria nova figura para cada gráfico
                # Cria tabela pivot para plotagem (desafio vs AI)
                pivot_table=data.pivot_table(index='challenge',columns='ai',values=col_name,aggfunc='mean')
                # Reordena linhas (desafios) e colunas (AIs)
                pivot_table=pivot_table.reindex(challenge_order,axis=0).reindex(ai_list,axis=1)
                # Preenche NaNs que podem surgir do reindex
                fill_val=0.0 if col_name not in ['flake8_issues','avg_exec_time_ms','num_execution_errors'] else np.nan
                pivot_table.fillna(value=fill_val, inplace=True)

                # Pula se todos os valores forem NaN após pivot/fillna
                if pivot_table.isnull().all().all(): print(f"  Skip bar '{title}': Tabela pivot apenas com NaN."); plt.close(); continue

                num_ais=len(ai_list)
                ax=pivot_table.plot(kind='bar',figsize=(max(12,num_ais*2),7),
                                     color=[cmap.get(ai,'grey') for ai in pivot_table.columns],width=0.8)
                ax.set_title(f'{title} por Desafio'); ax.set_ylabel(ylabel); ax.set_xlabel('Desafio')
                ax.tick_params(axis='x', rotation=45, labelsize=9)
                ax.tick_params(axis='y', labelsize=9); plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")

                # Ajusta limites do eixo Y conforme a métrica
                if col_name in ['completeness','avg_exec_success_rate']: ax.set_ylim(0, 105)
                elif col_name in ['flake8_issues','num_execution_errors'] and not pivot_table.isnull().all().all():
                    max_val=pivot_table.max(skipna=True).max(skipna=True)
                    ax.set_ylim(bottom=0,top=max(1,max_val*1.1) if not pd.isna(max_val) else 1) # Garante Y>=1
                elif col_name=='avg_exec_time_ms' and not pivot_table.isnull().all().all(): ax.set_ylim(bottom=0) # Tempo começa em 0

                ax.legend(title='IA', bbox_to_anchor=(1.02, 1), loc='upper left')
                plt.tight_layout(rect=[0, 0, 0.88, 1]) # Ajusta layout para legenda externa
                path=os.path.join(out_dir, f"{key}_{ts}.png"); plt.savefig(path); plt.close() # Salva e fecha figura
            except Exception as e: print(f"  Erro ao gerar gráfico de barras para '{title}': {e}"); plt.close() # Fecha figura em caso de erro

    def _plot_lru_times_chart(self, detailed_df, ai_list, cmap, ts, out_dir):
        """Gera gráfico de barras para o tempo médio de operação do LRU Cache."""
        print("Gerando Gráfico de Tempo Médio - LRU Cache...")
        challenge_name = "LRU Cache"

        # 1. Filtrar dados apenas para LRU Cache e onde o tempo é válido
        lru_data = detailed_df[
            (detailed_df['challenge'] == challenge_name) &
            (detailed_df['execution_time_ms'].notna()) &
            (detailed_df['execution_time_ms'] >= 0) # Considera tempos válidos
        ].copy()

        if lru_data.empty:
            print(f"  Skip LRU Times chart: Sem dados válidos para '{challenge_name}'.")
            return

        # 2. Pega os tempos médios (já agregados por IA/Execução)
        avg_times_per_ai = lru_data.set_index('ai')['execution_time_ms']
        # Reordena para garantir consistência com outras legendas e remove NaNs
        avg_times_per_ai = avg_times_per_ai.reindex(ai_list).dropna()

        if avg_times_per_ai.empty:
            print(f"  Skip LRU Times chart: Sem dados válidos após reindexar/dropna por IA.")
            return

        # 3. Gerar o gráfico de barras
        try:
            fig, ax = plt.subplots(figsize=(max(8, len(avg_times_per_ai) * 1.5), 6))
            colors_for_plot = [cmap.get(ai, 'grey') for ai in avg_times_per_ai.index]
            avg_times_per_ai.plot(kind='bar', ax=ax, color=colors_for_plot, width=0.6)

            ax.set_title(f'Tempo Médio por Operação - {challenge_name}')
            ax.set_ylabel('Tempo Médio (ms)')
            ax.set_xlabel('Inteligência Artificial')
            ax.tick_params(axis='x', rotation=0, labelsize=10)
            ax.tick_params(axis='y', labelsize=9)
            ax.grid(axis='y', linestyle='--', alpha=0.7)

            # Adiciona os valores no topo das barras
            for container in ax.containers:
                ax.bar_label(container, fmt='%.3f', label_type='edge', padding=3, fontsize=8) # Aumentei precisão

            ax.set_ylim(bottom=0, top=ax.get_ylim()[1] * 1.1) # Ajusta limite Y
            plt.tight_layout()
            plot_prefix = "8_lru_exec_time"
            path = os.path.join(out_dir, f"{plot_prefix}_{ts}.png")
            plt.savefig(path); plt.close(fig)
            print(f"- Gráfico Tempos LRU salvo: {path}")

        except Exception as e:
            print(f"  Erro ao gerar gráfico de tempos LRU: {e}")
            plt.close() # Fecha a figura em caso de erro

    def _plot_box_plots(self, df_plot, ai_list, cmap, ts, out_dir):
        """Gera box plots para distribuição das métricas."""
        print("Gerando Box Plots...")

        # Definição das métricas para boxplot (chave = nome coluna no df_plot)
        metrics_def = {
            "api_execution_time":{"t":"Distribuição API Time","y":"Tempo(s)","v":None}, # Sem filtro extra
            "lines_of_code":{"t":"Distribuição LoC","y":"Linhas","v":(lambda d,c: d[c]>0)}, # Filtro: Apenas LoC > 0
            "completeness":{"t":"Distribuição Completude","y":"Score (%)","v":None}, # Sem filtro extra
        }
        # Adiciona métricas opcionais se existirem e tiverem dados válidos
        if 'flake8_issues' in df_plot.columns and (df_plot['flake8_issues'] != -1).any():
             metrics_def["flake8_issues"]={"t":"Distribuição Flake8 Issues","y":"Issues","v":(lambda d,c: d[c]!=-1)}
        if 'avg_exec_time_ms' in df_plot.columns and (df_plot['avg_exec_time_ms'].notna() & (df_plot['avg_exec_time_ms']!=-1)).any():
             metrics_def["avg_exec_time_ms"]={"t":"Distribuição Exec Time (Steps)","y":"Tempo (ms)","v":(lambda d,c: d[c].notna()&(d[c]!=-1))}
        if 'num_execution_errors' in df_plot.columns and df_plot['num_execution_errors'].notna().any():
             # Apenas plota se houver algum erro > 0 para ser interessante
             if (df_plot['num_execution_errors'] > 0).any():
                  metrics_def["num_execution_errors"]={"t":"Distribuição Erros/Falhas Exec","y":"Num Errors/Fails","v":(lambda d,c: d[c].notna() & (d[c] > 0))} # Filtro: Apenas onde erro > 0
             else:
                  print("  Skip box 'Distribuição Erros/Falhas Exec': Nenhum erro registrado.")


        plot_counter = 8 # Ajustado contador inicial

        if not metrics_def:
            print("Nenhuma métrica válida definida para box plots.")
            return

        # Itera sobre as métricas definidas
        for col_name, config in metrics_def.items():
            plot_counter += 1
            if col_name not in df_plot.columns:
                print(f"  Skip box '{config['t']}': Coluna '{col_name}' N/A no df_plot.")
                continue

            plot_data = []; plot_labels = []; temp_df = df_plot.copy()

            # Aplica filtro de validade, se houver
            filter_func = config.get("v")
            if filter_func:
                try:
                    filter_condition = filter_func(temp_df, col_name)
                    temp_df = temp_df[filter_condition]
                except Exception as e:
                     print(f"  AVISO: Erro ao aplicar filtro para '{col_name}': {e}. Pulando filtro.")

            # Verifica se há dados após o filtro
            if temp_df.empty or temp_df[col_name].isnull().all():
                print(f"  Skip box '{config['t']}': Sem dados válidos após filtro para coluna '{col_name}'.")
                continue

            # Coleta dados por IA (após o filtro)
            valid_ai_data_found = False
            for ai in ai_list:
                ai_data = temp_df[temp_df['ai']==ai][col_name].dropna()
                if not ai_data.empty:
                    # Boxplot precisa de pelo menos 2 pontos para ser informativo, idealmente mais.
                    # Adicionaremos mesmo com 1 ponto, mas o plot pode não ser útil.
                    plot_data.append(ai_data.values)
                    plot_labels.append(ai)
                    valid_ai_data_found = True
                    # if ai_data.nunique() <= 1: # Opcional: Avisar se não há variação
                    #     print(f"    AVISO: Dados para IA '{ai}' ({col_name}) têm variação mínima/nula.")

            # Pula se nenhuma IA teve dados válidos
            if not valid_ai_data_found:
                print(f"  Skip box '{config['t']}': Sem dados válidos por IA após filtro para '{col_name}'.")
                continue

            # Gera o plot
            try:
                fig, ax = plt.subplots(figsize=(max(8,len(plot_labels)*1.2),6))
                bp = ax.boxplot(plot_data, tick_labels=plot_labels, patch_artist=True, showfliers=True) # showfliers=True mostra outliers
                ax.set_title(config["t"], fontsize=14); ax.set_ylabel(config["y"], fontsize=10); ax.set_xlabel("IA", fontsize=10)
                ax.tick_params(axis='x', rotation=30, labelsize=9); ax.tick_params(axis='y', labelsize=9)
                plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor"); ax.grid(axis='y', linestyle='--', alpha=0.7)
                colors=[cmap.get(lbl,'grey') for lbl in plot_labels]
                for patch, color in zip(bp['boxes'],colors): patch.set_facecolor(color); patch.set_alpha(0.6)
                plt.tight_layout()
                fname=f"{plot_counter}_boxplot_{col_name}_{ts}.png"; p=os.path.join(out_dir,fname); plt.savefig(p); plt.close(fig)
                print(f"  Gráfico box plot salvo: {p}")
            except Exception as e:
                print(f"  Erro ao gerar box plot para '{config['t']}': {e}")
                if 'fig' in locals() and plt.fignum_exists(fig.number): plt.close(fig) # Tenta fechar figura em caso de erro

        print("Geração Box Plots concluída.")


    def _plot_radar_chart(self, df, ai_list, cmap, ts, out_dir):
        """Gera gráfico de radar comparativo (Normalizado)."""
        n_ais = len(ai_list)
        if n_ais < 2: print("Radar chart pulado (< 2 IAs para comparar)."); return
        print(f"Gerando gráfico de radar para {n_ais} IAs...")

          # <<< ADICIONE ESTA LINHA >>>
        plot_counter = 14 # Define um número inicial para nome do arquivo (pode ajustar se necessário)
        # <<< FIM DA LINHA ADICIONADA >>>

        # Métricas para o radar (usar colunas do df agregado)
        # Normalizaremos para que 'MAIOR seja MELHOR' em todos os eixos (0-100)
        radar_cols_def = {
             'api_execution_time': {'label': 'Veloc API (Norm)', 'lower_is_better': True},
             'lines_of_code': {'label': 'Conc LoC (Norm)', 'lower_is_better': True},
             'completeness': {'label': 'Completude (%)', 'lower_is_better': False},
             'flake8_issues': {'label': 'Qualid Flake8 (Norm)', 'lower_is_better': True, 'ignore_val': -1},
             'avg_exec_success_rate': {'label': 'Taxa Sucesso Exec (%)', 'lower_is_better': False},
             # 'avg_exec_time_ms': {'label': 'Veloc Exec (Norm)', 'lower_is_better': True, 'ignore_val': -1.0} # Opcional
        }

        # Filtra colunas que realmente existem no DataFrame
        valid_radar_cols = {k: v for k, v in radar_cols_def.items() if k in df.columns}
        if len(valid_radar_cols) < 3: print(f"Radar chart pulado (< 3 métricas válidas encontradas no DF: {list(valid_radar_cols.keys())})."); return

        categories = [v['label'] for v in valid_radar_cols.values()]
        n_cats = len(categories)

        # Calcula médias por IA para as colunas válidas
        means = df.groupby('ai')[list(valid_radar_cols.keys())].mean()

        # Normaliza os dados (0-100, maior é melhor)
        norm_data = pd.DataFrame(index=means.index)
        for col, config in valid_radar_cols.items():
            series = means[col].copy()
            # Ignora valores específicos (como -1 para flake8/tempo) antes de normalizar
            ignore_val = config.get('ignore_val')
            if ignore_val is not None:
                 series = series[series != ignore_val]

            min_val = series.min(); max_val = series.max()
            range_val = max_val - min_val if max_val > min_val else 1 # Evita divisão por zero

            if config['lower_is_better']:
                 # Inverte: 100 * (max - x) / range
                 norm_series = 100 * (max_val - means[col]) / range_val
            else:
                 # Normal: 100 * (x - min) / range
                 norm_series = 100 * (means[col] - min_val) / range_val

            # Trata caso onde todos os valores eram iguais (range_val=1) ou ignorados
            if max_val == min_val: norm_series = 100 if not config['lower_is_better'] else 0 # 100 se maior=melhor, 0 se menor=melhor
            if ignore_val is not None: norm_series = norm_series.reindex(means.index).fillna(0 if config['lower_is_better'] else 100) # Preenche ignorados com pior/melhor valor

            norm_data[config['label']] = norm_series.clip(0, 100) # Garante que está entre 0 e 100

        # Prepara para plotagem
        angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
        angles += angles[:1] # Fecha o círculo

        fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_yticks(np.arange(0, 101, 20)) # Escala 0-100
        ax.set_ylim(0, 100)

        # Plota dados para cada IA
        for idx, ai in enumerate(norm_data.index):
            if ai not in ai_list: continue # Garante que só plota IAs habilitadas
            values = norm_data.loc[ai].values.flatten().tolist()
            values += values[:1] # Fecha o polígono
            color = cmap.get(ai, plt.cm.tab10(idx / n_ais)) # Cor do mapa ou default
            ax.plot(angles, values, 'o-', linewidth=2, label=ai, color=color)
            ax.fill(angles, values, color, alpha=0.2)

        plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15))
        plt.title('Comparação Geral Normalizada (Maior = Melhor)', size=14, y=1.18)
        plt.tight_layout()
        path = os.path.join(out_dir, f"{plot_counter + 1}_radar_chart_{ts}.png"); plt.savefig(path); plt.close(fig)
        print(f"  Gráfico Radar salvo: {path}")


    def create_visualizations(self, df, ts, out_dir):
        """Cria e salva visualizações baseadas no DataFrame agregado."""
        if df is None or df.empty: print("AVISO: DataFrame vazio para gráficos. Pulando visualizações."); return

        plt.style.use('ggplot') # Define estilo dos gráficos
        ais=sorted(df['ai'].unique()); n_ais=len(ais)
        if n_ais == 0: print("AVISO: Nenhuma AI encontrada nos dados para plotar."); return

        # Define mapa de cores para as IAs
        colors=plt.cm.viridis(np.linspace(0,1,n_ais)) if n_ais>10 else plt.cm.tab10(np.linspace(0,1,n_ais))
        cmap={ai:colors[i] for i,ai in enumerate(ais)}

        # Verifica se colunas essenciais existem
        required_cols = ['challenge', 'ai', 'difficulty'] # Mínimo necessário
        if not all(c in df.columns for c in required_cols):
            missing = [c for c in required_cols if c not in df.columns]
            print(f"AVISO: Faltando colunas essenciais para gráficos: {missing}. Pulando visualizações.")
            return

        # Ordena desafios por dificuldade para plots mais lógicos
        challenge_order=sorted(df['challenge'].unique()) # Fallback alfabético
        if 'difficulty' in df.columns:
            try:
                # Define ordem categórica e ordena DataFrame por ela
                difficulty_order = pd.CategoricalDtype(['Easy','Medium','Hard'],ordered=True)
                df['diff_cat']=df['difficulty'].astype(difficulty_order)
                # Pega ordem única dos desafios baseada na dificuldade
                challenge_order=df.sort_values('diff_cat')['challenge'].unique()
            except Exception as e_cat: print(f"AVISO: Erro ao ordenar desafios por dificuldade ({e_cat}). Usando ordem alfabética.")

        # --- Prepara DataFrame Detalhado para Plot LRU ---
        detailed_df = None
        if self.detailed_test_results: # Verifica se há resultados detalhados
            try:
                detailed_df = pd.DataFrame(self.detailed_test_results)
                # Garante que 'execution_time_ms' é numérico
                detailed_df['execution_time_ms'] = pd.to_numeric(detailed_df['execution_time_ms'], errors='coerce')
            except Exception as e:
                print(f"AVISO: Erro ao preparar DataFrame detalhado para plot LRU: {e}")
                detailed_df = None # Anula se houver erro
        else:
             print("AVISO: Sem dados detalhados (`detailed_test_results`) para plot LRU.")
        # --- Fim Preparação Detalhada ---

        df_plot=df.copy() # Usa cópia do DF agregado para plots gerais

        # Chama funções de plotagem individuais
        self._plot_bar_charts(df_plot, ais, cmap, challenge_order, ts, out_dir)
        self._plot_box_plots(df_plot, ais, cmap, ts, out_dir) # Passa df_plot agregado
        self._plot_radar_chart(df_plot, ais, cmap, ts, out_dir) # Passa df_plot agregado

        # --- Chama a função de plot LRU se houver dados detalhados ---
        if detailed_df is not None:
             self._plot_lru_times_chart(detailed_df, ais, cmap, ts, out_dir) # Passa df detalhado
        # --- Fim Chamada Plot LRU ---


# --- Ponto de Entrada ---
if __name__ == "__main__":
    print("="*59)
    print(" AI Programming Benchmark: Claude vs ChatGPT vs Gemini")
    print("       (Incluindo Análise Flake8 e Execução Externa/Local)")
    print("="*59)

    # Necessário para 'spawn' no Windows se script for congelado com PyInstaller
    multiprocessing.freeze_support()

    if not os.path.exists(".env"):
        print("AVISO: Ficheiro .env não encontrado. Chaves API podem não ser carregadas.")

    try:
        benchmark = AIBenchmark()
        # Verifica se pelo menos uma API está funcional antes de rodar
        if benchmark.claude_api_key or benchmark.openai_api_key or benchmark.gemini_api_key:
            benchmark.run_benchmark()
            print("\nBenchmark concluído.")
            print(f"Relatórios salvos em: '{benchmark.output_dir}'")
        else:
            # Mensagem de erro se nenhuma API estiver configurada
            print("\nERRO FATAL: Nenhuma chave API encontrada ou configurada corretamente.")
            print("Verifique o seu ficheiro .env e as variáveis CLAUDE_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY.")
    except Exception as e:
        # Captura exceções não esperadas no nível principal
        print("\nERRO INESPERADO DURANTE A EXECUÇÃO DO BENCHMARK:")
        print(f"Tipo: {type(e).__name__}")
        print(f"Erro: {e}")
        print("Traceback:")
        traceback.print_exc() # Mostra traceback completo para depuração

    print("="*59)