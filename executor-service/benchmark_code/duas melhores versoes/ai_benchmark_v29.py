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
import builtins  # Import necessário

# --- PASSO 1: Importar Tenacity ---
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

# Verifica se é Python 3.8+ para importlib.metadata
try:
    from importlib import metadata as importlib_metadata
except ImportError:
    try:
        import pkg_resources  # type: ignore
    except ImportError:
        pkg_resources = None

# Carrega variáveis de ambiente do .env
load_dotenv()


class AIBenchmark:
    """
    Executa um benchmark de programação comparando modelos de IA
    (Claude, ChatGPT, Gemini) em desafios de código, avaliando a qualidade,
    corretude (via execução externa ou local para LRU) e desempenho.
    Permite múltiplas execuções para resultados mais robustos.

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
        num_runs (int): Número de vezes que cada desafio/IA é executado.
        code_challenges (list): Lista de dicionários definindo os desafios.
        results (list): Lista para armazenar resultados agregados (TODAS AS RUNS).
        detailed_test_results (list): Lista para resultados detalhados (TODAS AS RUNS).
        output_dir (str): Diretório para salvar relatórios.
        flake8_available (bool): Indica se o Flake8 está disponível.
        environment_info (dict): Informações sobre o ambiente de execução.
    """

    # --- PASSO 3: Modificar __init__ para aceitar num_runs ---
    def __init__(self, num_runs: int = 1):  # Adiciona parâmetro com default 1
        """
        Inicializa o benchmark. Carrega configurações, define desafios,
        verifica ferramentas (flake8) e coleta informações do ambiente.
        """
        self.claude_api_key = os.getenv("CLAUDE_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")

        self.num_runs = max(1, num_runs)  # Garante pelo menos 1 execução
        print(f"Número de execuções por desafio/IA: {self.num_runs}")

        self.results = []  # Acumulará resultados de TODAS as runs
        self.detailed_test_results = []  # Acumulará detalhes de TODAS as runs
        self.output_dir = "benchmark_reports"
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Diretório de saída: '{self.output_dir}/'")

        # --- Carrega Configurações (com fallbacks) ---
        self._configure_gemini_api()

        # Nomes dos modelos via .env ou fallback para defaults
        self.claude_model = os.getenv("CLAUDE_MODEL_NAME", "claude-3-opus-20240229")
        self.openai_model = os.getenv("OPENAI_MODEL_NAME", "gpt-4-turbo")
        self.gemini_model = (
            os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-pro-latest")
            if self.gemini_api_key
            else None
        )

        # URL do Executor via .env ou fallback
        default_executor_url = (
            "https://executor-service-1029404216382.europe-southwest1.run.app/execute"
        )
        self.executor_url = os.getenv("EXECUTOR_SERVICE_URL", default_executor_url)

        # --- PASSO 2: Adicionar Novos Desafios ---
        self.code_challenges = [
            {
                "name": "Fibonacci Sequence",
                "description": "Generate the first n numbers in the Fibonacci sequence, starting with 0.",
                "difficulty": "Easy",
                "function_signature": "fibonacci(n: int) -> list[int]",
                "test_cases": [
                    {
                        "input_args": [10],
                        "expected_output": [0, 1, 1, 2, 3, 5, 8, 13, 21, 34],
                    },
                    {"input_args": [1], "expected_output": [0]},
                    {"input_args": [0], "expected_output": []},
                    {"input_args": [2], "expected_output": [0, 1]},
                ],
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
                    {"input_args": [[], 5], "expected_output": -1},
                ],
            },
            {
                "name": "Merge Sort",
                "description": "Implement the merge sort algorithm to sort a list of integers in ascending order. Return a new sorted list.",
                "difficulty": "Medium",
                "function_signature": "merge_sort(arr: list[int]) -> list[int]",
                "test_cases": [
                    {
                        "input_args": [[3, 1, 4, 1, 5, 9, 2, 6]],
                        "expected_output": [1, 1, 2, 3, 4, 5, 6, 9],
                    },
                    {
                        "input_args": [[5, 4, 3, 2, 1]],
                        "expected_output": [1, 2, 3, 4, 5],
                    },
                    {"input_args": [[1]], "expected_output": [1]},
                    {"input_args": [[]], "expected_output": []},
                ],
            },
            {
                "name": "Count Islands in Grid",
                "description": "Count the number of islands in a 2D grid (list of lists) where 1 is land and 0 is water. An island is formed by connecting adjacent lands horizontally or vertically.",
                "difficulty": "Hard",
                "function_signature": "count_islands(grid: list[list[int]]) -> int",
                "test_cases": [
                    {
                        "input_args": [
                            [
                                [1, 1, 1, 1, 0],
                                [1, 1, 0, 1, 0],
                                [1, 1, 0, 0, 0],
                                [0, 0, 0, 0, 0],
                            ]
                        ],
                        "expected_output": 1,
                    },
                    {
                        "input_args": [
                            [
                                [1, 1, 0, 0, 0],
                                [1, 1, 0, 0, 0],
                                [0, 0, 1, 0, 0],
                                [0, 0, 0, 1, 1],
                            ]
                        ],
                        "expected_output": 3,
                    },
                    {"input_args": [[]], "expected_output": 0},
                    {"input_args": [[[0, 0, 0], [0, 0, 0]]], "expected_output": 0},
                ],
            },
            {
                "name": "Palindrome Check",
                "description": "Check if a given string is a palindrome, ignoring case and non-alphanumeric characters. Return True if it is, False otherwise.",
                "difficulty": "Easy",
                "function_signature": "is_palindrome(s: str) -> bool",
                "test_cases": [
                    {
                        "input_args": ["A man, a plan, a canal: Panama"],
                        "expected_output": True,
                    },
                    {"input_args": ["race a car"], "expected_output": False},
                    {"input_args": [" "], "expected_output": True},
                    {"input_args": [""], "expected_output": True},
                    {
                        "input_args": ["Was it a car or a cat I saw?"],
                        "expected_output": True,
                    },
                    {"input_args": ["tab a cat"], "expected_output": False},
                ],
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
                    {"input_args": ["]"], "expected_output": False},
                ],
            },
            {
                "name": "LRU Cache",
                "description": "Implement a Least Recently Used (LRU) cache with a given capacity. Support get(key) and put(key, value) operations. get returns value or -1. put inserts/updates; evicts LRU item if capacity exceeded.",
                "difficulty": "Hard",
                "function_signature": "class LRUCache:\n    def __init__(self, capacity: int):\n        pass\n    def get(self, key: int) -> int:\n        pass\n    def put(self, key: int, value: int) -> None:\n        pass",
                "test_cases": [
                    {
                        "description": "Simple put/get sequence",
                        "sequence": [
                            {"op": "__init__", "args": [2]},
                            {"op": "put", "args": [1, 1]},
                            {"op": "put", "args": [2, 2]},
                            {"op": "get", "args": [1], "expected": 1},
                            {"op": "put", "args": [3, 3]},
                            {"op": "get", "args": [2], "expected": -1},
                            {"op": "put", "args": [4, 4]},
                            {"op": "get", "args": [1], "expected": -1},
                            {"op": "get", "args": [3], "expected": 3},
                            {"op": "get", "args": [4], "expected": 4},
                        ],
                    },
                ],
            },
            # --- NOVOS DESAFIOS ---
            {
                "name": "Two Sum",
                "description": "Given a list of integers `nums` and an integer `target`, return indices of the two numbers such that they add up to `target`. You may assume that each input would have exactly one solution, and you may not use the same element twice.",
                "difficulty": "Easy",
                "function_signature": "two_sum(nums: list[int], target: int) -> list[int]",
                "test_cases": [
                    {
                        "input_args": [[2, 7, 11, 15], 9],
                        "expected_output": [0, 1],
                    },  # Ordem pode variar
                    {"input_args": [[3, 2, 4], 6], "expected_output": [1, 2]},
                    {"input_args": [[3, 3], 6], "expected_output": [0, 1]},
                    {"input_args": [[-1, -3, 5, 9], 4], "expected_output": [0, 2]},
                ],
            },
            {
                "name": "Longest Substring Without Repeating Characters",
                "description": "Given a string `s`, find the length of the longest substring without repeating characters.",
                "difficulty": "Medium",
                "function_signature": "length_of_longest_substring(s: str) -> int",
                "test_cases": [
                    {"input_args": ["abcabcbb"], "expected_output": 3},
                    {"input_args": ["bbbbb"], "expected_output": 1},
                    {"input_args": ["pwwkew"], "expected_output": 3},
                    {"input_args": [""], "expected_output": 0},
                    {"input_args": [" "], "expected_output": 1},
                    {"input_args": ["dvdf"], "expected_output": 3},
                ],
            },
            {
                "name": "Minimum Edit Distance",
                "description": "Given two strings `word1` and `word2`, return the minimum number of operations (insert, delete, substitute a character) required to convert `word1` to `word2`.",
                "difficulty": "Hard",
                "function_signature": "min_distance(word1: str, word2: str) -> int",
                "test_cases": [
                    {"input_args": ["horse", "ros"], "expected_output": 3},
                    {"input_args": ["intention", "execution"], "expected_output": 5},
                    {"input_args": ["", ""], "expected_output": 0},
                    {"input_args": ["a", "b"], "expected_output": 1},
                    {
                        "input_args": ["zoologicoarchaeologist", "zoogeologist"],
                        "expected_output": 10,
                    },
                ],
            },
            # --- FIM NOVOS DESAFIOS ---
        ]  # Fim da lista self.code_challenges

        self._check_flake8()
        self.environment_info = self._get_environment_info()  # Coleta info do ambiente

    def _configure_gemini_api(self):
        """Configura a API Gemini se a chave API estiver disponível."""
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
        """Coleta e retorna informações sobre o ambiente de execução."""
        info = {
            "Timestamp": datetime.now().isoformat(),
            "OS": f"{platform.system()} {platform.release()}",
            "Python Version": sys.version.split()[0],
            "Num Runs": self.num_runs,  # Adiciona número de runs
            "Models": {
                "Claude": self.claude_model if self.claude_api_key else "N/A",
                "ChatGPT": self.openai_model if self.openai_api_key else "N/A",
                "Gemini": (
                    self.gemini_model
                    if self.gemini_api_key
                    else "N/A (Config Failed or No Key)"
                ),
            },
            "Executor Service URL": self.executor_url,
            "Local LRU Cache Execution": True,
            "Libraries": {},
        }
        libs_to_check = [
            "pandas",
            "numpy",
            "matplotlib",
            "requests",
            "python-dotenv",
            "google-generativeai",
            "google-api-core",
            "tenacity",  # Adiciona tenacity
        ]
        lib_versions = {}
        try:
            import importlib.metadata as importlib_metadata

            for lib in libs_to_check:
                try:
                    lib_versions[lib] = importlib_metadata.version(lib)
                except importlib_metadata.PackageNotFoundError:
                    lib_versions[lib] = "Not Found"
        except ImportError:
            if pkg_resources:
                for lib in libs_to_check:
                    try:
                        lib_versions[lib] = pkg_resources.get_distribution(lib).version
                    except pkg_resources.DistributionNotFound:
                        lib_versions[lib] = "Not Found"
                    except Exception as e:
                        lib_versions[lib] = f"Error ({type(e).__name__})"
            else:
                for lib in libs_to_check:
                    lib_versions[lib] = "Unknown"
        info["Libraries"] = lib_versions
        return info

    def _check_flake8(self):
        """Verifica disponibilidade do flake8."""
        try:
            subprocess.run(
                [sys.executable, "-m", "flake8", "--version"],
                check=True,
                capture_output=True,
                timeout=5,
            )
            print("Verificação Flake8: OK")
            self.flake8_available = True
        except Exception:
            print(
                f"AVISO: flake8 não encontrado via '{sys.executable} -m flake8'. Análise pulada."
            )
            self.flake8_available = False

    def query_claude(self, prompt: str) -> dict:
        """Envia um prompt para a API do Claude e retorna a resposta (com tratamento de erro)."""
        start_time = time.time()
        content = ""
        execution_time = 0
        last_exception = None
        if not self.claude_api_key:
            print("Pulando Claude: Chave API N/A.")
            return {
                "content": "",
                "execution_time": 0,
                "last_error": "API Key Not Available",
            }

        headers = {
            "x-api-key": self.claude_api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        data = {
            "model": self.claude_model,
            "max_tokens": 4000,
            "messages": [{"role": "user", "content": prompt}],
        }

        try:
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=data,
                timeout=120,
            )
            response.raise_for_status()
            result = response.json()
            if (
                "content" in result
                and isinstance(result["content"], list)
                and len(result["content"]) > 0
            ):
                first_content = result["content"][0]
                content = (
                    first_content.get("text", "")
                    if isinstance(first_content, dict)
                    else ""
                )
            else:
                print(f"Resposta Claude inesperada: {result}")
                last_exception = ValueError(
                    f"Unexpected Claude Response Format: {result}"
                )

        except requests.exceptions.RequestException as e:
            last_exception = e
            err_msg = f"Erro req Claude: {e}"
            status_code_val = "N/A"
            if e.response is not None:
                try:
                    status_code_val = int(e.response.status_code)
                except (ValueError, TypeError):
                    status_code_val = e.response.status_code
                    print(
                        f"  AVISO: Claude status code não é um inteiro ({type(status_code_val).__name__}): {status_code_val}"
                    )

            err_type = "(Unknown Error Type)"
            if isinstance(status_code_val, int):
                if 400 <= status_code_val < 500:
                    err_type = "(Client Error)"
                elif status_code_val >= 500:
                    err_type = "(Server Error)"
            elif status_code_val != "N/A":
                err_type = f"(Non-int Status: {status_code_val})"
            else:
                err_type = "(Network/Timeout/Other Error)"

            if e.response is not None:
                err_msg += f" | Status:{status_code_val}{err_type}"
                try:
                    err_details = e.response.json()
                    api_error_type = err_details.get("error", {}).get("type", "N/A")
                    api_error_message = err_details.get("error", {}).get(
                        "message", "N/A"
                    )
                    err_msg += f" | API Err Type: {api_error_type} | API Err Msg: {api_error_message}"
                except json.JSONDecodeError:
                    err_msg += f" | Body (non-JSON):{e.response.text[:150]}..."
                except Exception as json_e:
                    err_msg += f" | Error parsing error response: {json_e}"
            else:
                err_msg += f" | {err_type}"
            print(err_msg)

        except Exception as e:
            last_exception = e
            print(f"Erro inesperado Claude: {type(e).__name__}-{e}")

        execution_time = time.time() - start_time
        return {
            "content": content or "",
            "execution_time": execution_time,
            "last_error": str(last_exception) if last_exception else None,
        }

    def query_chatgpt(self, prompt: str) -> dict:
        """Envia um prompt para a API do OpenAI (ChatGPT) e retorna a resposta."""
        start_time = time.time()
        content = ""
        execution_time = 0
        last_exception = None
        if not self.openai_api_key:
            print("Pulando ChatGPT: Chave API N/A.")
            return {
                "content": "",
                "execution_time": 0,
                "last_error": "API Key Not Available",
            }

        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": self.openai_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4000,
        }
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=120,
            )
            response.raise_for_status()
            result = response.json()
            if (
                "choices" in result
                and isinstance(result["choices"], list)
                and len(result["choices"]) > 0
            ):
                choice = result["choices"][0]
                if (
                    isinstance(choice, dict)
                    and "message" in choice
                    and isinstance(choice["message"], dict)
                ):
                    content = choice["message"].get("content", "")
                else:
                    print(f"Resposta ChatGPT inesperada (message): {result}")
                    last_exception = ValueError(
                        "Unexpected ChatGPT Response Format (message)"
                    )
            else:
                print(f"Resposta ChatGPT inesperada (choices): {result}")
                last_exception = ValueError(
                    "Unexpected ChatGPT Response Format (choices)"
                )
        except requests.exceptions.RequestException as e:
            last_exception = e
            err_msg = f"Erro req ChatGPT: {e}"
            status_code_val = "N/A"
            if e.response is not None:
                try:
                    status_code_val = int(e.response.status_code)
                except (ValueError, TypeError):
                    status_code_val = e.response.status_code
                    print(
                        f"  AVISO: ChatGPT status code não é um inteiro ({type(status_code_val).__name__}): {status_code_val}"
                    )

            err_type = "(Unknown Error Type)"
            if isinstance(status_code_val, int):
                if 400 <= status_code_val < 500:
                    err_type = "(Client Error)"
                elif status_code_val >= 500:
                    err_type = "(Server Error)"
            elif status_code_val != "N/A":
                err_type = f"(Non-int Status: {status_code_val})"
            else:
                err_type = "(Network/Timeout/Other Error)"

            if e.response is not None:
                err_msg += f" | Status:{status_code_val}{err_type}"
                try:
                    details = e.response.json()
                    api_err = details.get("error", {}).get("message", "")
                    err_msg += (
                        f" | API Err:{api_err}" if api_err else f" | Body:{details}"
                    )
                except json.JSONDecodeError:
                    err_msg += f" | Body (non-JSON):{e.response.text[:100]}..."
                except Exception as json_e:
                    err_msg += f" | Error parsing error response: {json_e}"
            else:
                err_msg += f" | {err_type}"
            print(err_msg)
        except Exception as e:
            last_exception = e
            print(f"Erro inesperado ChatGPT: {type(e).__name__}-{e}")

        execution_time = time.time() - start_time
        return {
            "content": content or "",
            "execution_time": execution_time,
            "last_error": str(last_exception) if last_exception else None,
        }

    # --- PASSO 1: query_gemini com Tenacity ---
    # --- query_gemini CORRIGIDA (SEM LOG DE RETRY PROBLEMÁTICO) ---
    def query_gemini(self, prompt: str) -> dict:
        """
        Envia um prompt para a API do Gemini e retorna a resposta.
        Usa Tenacity para retentativas com backoff exponencial em caso de rate limit.
        """
        start_time = time.time()
        content = ""
        final_response = None
        last_exception = None

        if not self.gemini_api_key or not self.gemini_model:
            print("Pulando Gemini: Chave/Modelo N/A.")
            return {
                "content": "",
                "execution_time": 0,
                "last_error": "API Key/Model Not Available",
            }

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=5, max=60),
            retry=retry_if_exception_type(google.api_core.exceptions.ResourceExhausted),
            reraise=True,
        )
        def _call_gemini_api_with_retry(
            prompt_str: str,
        ) -> genai.types.GenerateContentResponse:
            """Função interna que faz a chamada real à API."""
            model = genai.GenerativeModel(self.gemini_model)
            safety_settings = [
                {"category": c, "threshold": "BLOCK_NONE"}
                for c in [
                    "HARM_CATEGORY_HARASSMENT",
                    "HARM_CATEGORY_HATE_SPEECH",
                    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "HARM_CATEGORY_DANGEROUS_CONTENT",
                ]
            ]
            # Log removido para evitar AttributeError
            response = model.generate_content(
                prompt_str, safety_settings=safety_settings
            )
            return response

        try:
            final_response = _call_gemini_api_with_retry(prompt)
            try:
                content = final_response.text
            except ValueError:
                err_msg = "Resposta Gemini bloqueada ou sem texto."
                if hasattr(final_response, "prompt_feedback"):
                    err_msg += f" Feedback: {final_response.prompt_feedback}"
                if hasattr(final_response, "candidates") and final_response.candidates:
                    err_msg += f" Reason: {final_response.candidates[0].finish_reason}"
                print(err_msg)
                last_exception = ValueError(err_msg)
            except Exception as e_text:
                err_msg = f"Erro ao extrair texto da resposta Gemini: {e_text}"
                print(err_msg)
                last_exception = ValueError(err_msg)

        except google.api_core.exceptions.ResourceExhausted as e_rate_limit:
            err_msg = (
                f"Erro Gemini (Rate Limit) após múltiplas tentativas: {e_rate_limit}"
            )
            print(err_msg)
            last_exception = e_rate_limit
        except Exception as e_unexpected:
            err_msg = f"Erro inesperado Gemini: {type(e_unexpected).__name__} - {e_unexpected}"
            print(err_msg)
            last_exception = e_unexpected

        execution_time = time.time() - start_time

        return {
            "content": content or "",
            "execution_time": execution_time,
            "last_error": str(last_exception) if last_exception else None,
        }
        # --- FIM query_gemini CORRIGIDA ---

        try:
            # Chama a função com retentativas
            final_response = _call_gemini_api_with_retry(prompt)
            try:
                # Tenta extrair o texto da resposta final
                content = final_response.text
            except ValueError:
                # Lida com resposta bloqueada ou sem texto
                err_msg = "Resposta Gemini bloqueada ou sem texto."
                if hasattr(final_response, "prompt_feedback"):
                    err_msg += f" Feedback: {final_response.prompt_feedback}"
                if hasattr(final_response, "candidates") and final_response.candidates:
                    err_msg += f" Reason: {final_response.candidates[0].finish_reason}"
                print(err_msg)
                last_exception = ValueError(err_msg)
            except Exception as e_text:
                err_msg = f"Erro ao extrair texto da resposta Gemini: {e_text}"
                print(err_msg)
                last_exception = ValueError(err_msg)

        except google.api_core.exceptions.ResourceExhausted as e_rate_limit:
            # Captura erro de rate limit se todas as tentativas falharem
            err_msg = (
                f"Erro Gemini (Rate Limit) após múltiplas tentativas: {e_rate_limit}"
            )
            print(err_msg)
            last_exception = e_rate_limit
        except Exception as e_unexpected:
            # Captura outros erros inesperados
            err_msg = f"Erro inesperado Gemini: {type(e_unexpected).__name__} - {e_unexpected}"
            print(err_msg)
            last_exception = e_unexpected

        execution_time = (
            time.time() - start_time
        )  # Tempo total, incluindo esperas e retentativas

        return {
            "content": content or "",
            "execution_time": execution_time,
            "last_error": str(last_exception) if last_exception else None,
        }

    # --- Fim da modificação query_gemini ---

    def extract_code(self, content: str) -> str:
        """Extrai o primeiro bloco de código Python de uma string de texto."""
        if not content:
            return ""
        m = re.search(r"```python\n(.*?)\n```", content, re.DOTALL)
        if m:
            return m.group(1).strip()
        m = re.search(r"```(?:.*\n)?(.*?)\n```", content, re.DOTALL)
        if m:
            return m.group(1).strip()
        lines = content.splitlines()
        code = []
        start = False
        for ln in lines:
            s_ln = ln.strip()
            if not start and (
                ln.startswith((" ", "\t")) or s_ln.startswith(("def ", "class "))
            ):
                start = True
            if start:
                code.append(ln)
        if code:
            return "\n".join(code).strip()
        if len(lines) < 3 and not ("def " in content or "class " in content):
            return ""
        # Fallback final: Se nada foi extraído e o conteúdo parece código (tem def/class ou indentação)
        if (
            "def " in content
            or "class " in content
            or any(l.startswith((" ", "\t")) for l in lines)
        ):
            return content.strip()  # Retorna tudo como código
        return ""  # Retorna vazio se não parecer código

    def count_code_lines(self, code: str) -> int:
        """Conta linhas de código reais (não vazias, não comentários)."""
        if not code:
            return 0
        return len(
            [
                ln
                for ln in code.splitlines()
                if ln.strip() and not ln.strip().startswith("#")
            ]
        )

    # --- PASSO 2: Atualizar measure_response_completeness ---
    def measure_response_completeness(self, code: str, description: str) -> int:
        """Estima a 'completude' da resposta da IA (score 0-100)."""
        score = 0
        lines = self.count_code_lines(code)
        if lines > 0:
            score += 50
            if "def " in code or "class " in code:
                score += 10
            if "return " in code or "yield " in code:
                score += 10
            dl = description.lower()
            cl = code.lower()
            bonus = 0
            if "fibonacci" in dl and (
                "fibonacci" in cl or "fib(" in cl or "yield" in code
            ):
                bonus = 10
            elif "binary search" in dl and (
                "mid =" in code or "mid=" in code or "low" in code or "high" in code
            ):
                bonus = 10
            elif "merge sort" in dl and (
                "merge(" in cl or ("left" in code and "right" in code and "mid" in code)
            ):
                bonus = 10
            elif "island" in dl and (
                "dfs(" in cl
                or "bfs(" in cl
                or "visit" in cl
                or ("grid[" in code and "==" in code)
            ):
                bonus = 10
            elif "palindrome" in dl and (
                "[::-1]" in code or ".isalnum()" in cl or ".lower()" in cl
            ):
                bonus = 10
            elif "parentheses" in dl and (
                "stack" in cl or "append(" in code or "pop(" in code
            ):
                bonus = 10
            elif "lru cache" in dl and (
                "dict" in cl or "capacity" in cl or "ordereddict" in code.lower()
            ):
                bonus = 10
            elif "two sum" in dl and (
                "dict" in cl
                or "hashmap" in cl
                or "target -" in code
                or "complement" in cl
            ):
                bonus = 10
            elif "longest substring" in dl and (
                "set(" in code
                or "dict(" in code
                or "window" in cl
                or "sliding" in cl
                or "left" in cl
                and "right" in cl
            ):
                bonus = 10
            elif "edit distance" in dl and (
                "dp[" in code or "matrix" in cl or "min(" in code and "+ 1" in code
            ):
                bonus = 10
            score += bonus
            struct = 0
            if "if" in code or "for" in code or "while" in code:
                struct = 10
            if struct > 0 and (
                "if not" in code
                or "if len(" in code
                or "== 0" in code
                or "is None" in code
                or "else:" in code
            ):
                struct += 5
            score += min(struct, 15)
        return min(score, 100)

    # --- FIM DA ATUALIZAÇÃO ---

    def analyze_code_quality_flake8(self, code_string: str) -> int:
        """Analisa qualidade do código com Flake8."""
        if (
            not self.flake8_available
            or not code_string
            or self.count_code_lines(code_string) == 0
        ):
            return -1
        tfp = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w+", suffix=".py", delete=False, encoding="utf-8"
            ) as tf:
                tf.write(code_string)
                tfp = tf.name
            exe = sys.executable
            res = subprocess.run(
                [exe, "-m", "flake8", "--count", tfp],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            if res.returncode in [0, 1] and res.stdout:
                try:
                    return int(res.stdout.strip().splitlines()[-1])
                except:
                    return len(res.stdout.strip().splitlines())
            elif res.stderr:
                print(f"Erro flake8: {res.stderr}")
                return -1
            else:
                return 0
        except subprocess.TimeoutExpired:
            print("Erro: Flake8 timeout.")
            return -1
        except Exception as e:
            print(f"Erro inesperado flake8: {e}")
            return -1
        finally:
            if tfp and os.path.exists(tfp):
                try:
                    os.remove(tfp)
                except Exception as er:
                    print(f"Aviso: Falha remover temp {tfp}: {er}")

    # --- Métodos Execução Local LRU Cache (sem alterações) ---
    def _lru_worker(
        self, code_string: str, test_sequence: list, result_queue: multiprocessing.Queue
    ):
        # (Código _lru_worker como estava antes)
        results = []
        error_message = None
        lru_instance = None
        final_success = True
        try:
            lines = code_string.splitlines()
            filtered_lines = [
                ln
                for ln in lines
                if not (
                    ln.strip().startswith("from collections import OrderedDict")
                    or ln.strip().startswith("import collections")
                    or ln.strip().startswith("import OrderedDict")
                )
            ]
            processed_code_string = "\n".join(filtered_lines)
            safe_builtins = {
                "int": int,
                "str": str,
                "list": list,
                "dict": dict,
                "tuple": tuple,
                "len": len,
                "range": range,
                "print": print,
                "True": True,
                "False": False,
                "None": None,
                "object": object,
                "Exception": Exception,
                "RuntimeError": RuntimeError,
                "ValueError": ValueError,
                "NameError": NameError,
                "AttributeError": AttributeError,
                "TypeError": TypeError,
                "StopIteration": StopIteration,
                "KeyError": KeyError,
                "__build_class__": builtins.__build_class__,
                "getattr": getattr,
                "isinstance": isinstance,
                "__name__": "__exec_sandbox__",
            }
            safe_globals = {
                "__builtins__": safe_builtins,
                "OrderedDict": OrderedDict,
                "__name__": "__exec_sandbox__",
            }
            local_namespace = {}
            exec(processed_code_string, safe_globals, local_namespace)
            class_name = None
            for name, obj in local_namespace.items():
                if isinstance(obj, type) and name != "OrderedDict":
                    class_name = name
                    break
            if not class_name:
                match = re.search(
                    r"^\s*class\s+(\w+)", processed_code_string, re.MULTILINE
                )
                if match:
                    class_name = match.group(1)
                else:
                    raise NameError(
                        "Nome classe LRUCache não encontrado no código processado."
                    )
            LruClass = local_namespace.get(class_name)
            if not LruClass:
                raise NameError(
                    f"Classe '{class_name}' não definida no namespace local."
                )
            op_total_time = 0.0
            for op_data in test_sequence:
                op_start_time = time.perf_counter()
                operation = op_data["op"]
                args = op_data.get("args", [])
                expected = op_data.get("expected")
                step_res = {
                    "op": operation,
                    "args": args,
                    "exp": expected,
                    "out": None,
                    "ok": False,
                    "err": None,
                    "t_ms": -1.0,
                }
                try:
                    if operation == "__init__":
                        lru_instance = LruClass(*args)
                        step_res["ok"] = True
                        step_res["out"] = "OK"
                    elif lru_instance is None:
                        raise RuntimeError(
                            "LRUCache não inicializada antes de chamar outros métodos."
                        )
                    elif operation in ["put", "get"]:
                        if not hasattr(lru_instance, operation):
                            raise AttributeError(
                                f"Instância da classe '{class_name}' não tem método '{operation}'"
                            )
                        method = getattr(lru_instance, operation)
                        if operation == "put":
                            method(*args)
                            step_res["ok"] = True
                            step_res["out"] = "OK"
                        elif operation == "get":
                            output = method(*args)
                            step_res["out"] = output
                            if str(output) == str(expected):
                                step_res["ok"] = True
                            else:
                                step_res["err"] = (
                                    f"Output:'{output}' != Expected:'{expected}'"
                                )
                                final_success = False
                    else:
                        raise ValueError(
                            f"Operação LRU inválida na sequência de teste: {operation}"
                        )
                except Exception as e_step:
                    step_res["err"] = (
                        f"{type(e_step).__name__}: {str(e_step).splitlines()[0]}"
                    )
                    final_success = False
                op_end_time = time.perf_counter()
                step_res["t_ms"] = (op_end_time - op_start_time) * 1000
                op_total_time += op_end_time - op_start_time
                results.append(step_res)
        except Exception as e_main:
            error_message = f"Erro worker LRU: {type(e_main).__name__} - {str(e_main).splitlines()[0]}"
            print(f"    ERRO FATAL NO WORKER LRU: {error_message}")
            traceback.print_exc()
            final_success = False
        try:
            result_queue.put(
                {"success": final_success, "results": results, "error": error_message}
            )
        except Exception as q_err:
            print(f"    ERRO ao colocar resultado na fila do worker LRU: {q_err}")

    def _execute_lru_locally(self, code_string: str, test_sequence: list) -> dict:
        """Gere o processo para execução local do LRU Cache, com timeout."""
        # (Código _execute_lru_locally como estava antes)
        ctx = multiprocessing.get_context(None)
        result_queue = ctx.Queue()
        timeout_seconds = 10
        process = ctx.Process(
            target=self._lru_worker, args=(code_string, test_sequence, result_queue)
        )
        return_value = {
            "success": False,
            "results": [],
            "error": "Unknown execution error",
        }
        terminated = False
        try:
            start_time = time.time()
            process.start()
            process.join(timeout=timeout_seconds)
            elapsed_time = time.time() - start_time
            if process.is_alive():
                print(
                    f"  AVISO: Timeout ({timeout_seconds}s) atingido para processo LRU. Terminando..."
                )
                process.terminate()
                process.join(1)
                if process.is_alive():
                    print("  AVISO: Terminate falhou, forçando kill processo LRU.")
                    process.kill()
                    process.join()
                return_value = {
                    "success": False,
                    "results": [],
                    "error": f"Timeout ({timeout_seconds}s)",
                }
                terminated = True
            if not terminated:
                exit_code = process.exitcode
                if exit_code == 0:
                    try:
                        result = result_queue.get(timeout=1)
                        return_value = result
                    except queue.Empty:
                        print(
                            "  ERRO: Processo LRU terminou OK, mas fila de resultados vazia."
                        )
                        return_value = {
                            "success": False,
                            "results": [],
                            "error": "Worker finished OK but result queue empty",
                        }
                    except Exception as e_q:
                        print(f"  ERRO: Falha obter resultado da fila LRU: {e_q}")
                        return_value = {
                            "success": False,
                            "results": [],
                            "error": f"Queue communication error: {e_q}",
                        }
                else:
                    print(
                        f"  ERRO: Processo worker LRU terminou inesperadamente com exitcode: {exit_code}"
                    )
                    try:
                        err_result = result_queue.get_nowait()
                        return_value = err_result
                    except:
                        pass
                if not return_value.get("error"):
                    return_value["error"] = (
                        f"Worker process exited with code {exit_code}"
                    )
                    return_value["success"] = False
        except queue.Empty:
            return_value = {
                "success": False,
                "results": [],
                "error": f"Worker process exited with code {exit_code} (no error message in queue)",
            }
        except Exception as e_qf:
            return_value = {
                "success": False,
                "results": [],
                "error": f"Worker exit {exit_code}, queue fail getting error: {e_qf}",
            }
        except Exception as e_p:
            print(f"  ERRO: Falha ao gerir processo worker LRU: {e_p}")
            return_value = {
                "success": False,
                "results": [],
                "error": f"Process management error: {e_p}",
            }
            if process and process.is_alive():
                try:
                    process.kill()
                    process.join()
                except:
                    pass
        finally:
            try:
                result_queue.close()
                result_queue.join_thread()
            except Exception as e_qc:
                print(f"  Aviso: Erro ao limpar fila LRU: {e_qc}")
            try:
                if process:
                    process.close()
            except Exception as e_pc:
                print(f"  Aviso: Erro ao limpar objeto Process LRU: {e_pc}")
        return return_value

    # --- Fim métodos execução local ---

    # --- PASSO 3: Modificar evaluate_solution para aceitar e usar run_idx ---
    def evaluate_solution(
        self, ai_name: str, challenge: dict, response: dict, run_idx: int
    ) -> dict:
        """
        Avalia a solução de uma IA para um desafio E UMA EXECUÇÃO ESPECÍFICA.

        Extrai código, analisa com Flake8, mede completude, e executa testes.
        Adiciona resultados detalhados com 'run_idx' à lista GERAL `self.detailed_test_results`.

        Args:
            ai_name (str): Nome da IA.
            challenge (dict): Dicionário do desafio.
            response (dict): Resposta da API {'content': str, 'execution_time': float, 'last_error': str|None}.
            run_idx (int): Índice da execução atual (0 a num_runs-1).

        Returns:
            dict: Dicionário com resultados agregados para este AI/Desafio/Execução.
        """

        content = response.get("content", "")
        api_time = response.get("execution_time", 0)
        api_error = response.get("last_error")  # Captura erro da API, se houver

        # Se não houve conteúdo E houve erro na API, retorna falha direto
        if not content and api_error:
            print(
                f"  AVISO [Run {run_idx+1}]: Sem conteúdo de {ai_name} para {challenge['name']} devido a erro API. Avaliação pulada."
            )
            fail_result = {
                "run_index": run_idx,
                "ai": ai_name,
                "challenge": challenge["name"],
                "difficulty": challenge["difficulty"],
                "api_execution_time": api_time,
                "lines_of_code": 0,
                "completeness": 0,
                "flake8_issues": -1,
                "num_execution_errors": len(
                    challenge.get("test_cases", [])
                ),  # Marca todos os testes como erro
                "first_execution_error": api_error,
                "code": "",
                "full_response": "",
                "execution_results_raw": [],
            }
            # Adiciona detalhes de erro para cada caso de teste
            num_test_cases = len(challenge.get("test_cases", []))
            for i in range(num_test_cases if challenge["name"] != "LRU Cache" else 1):
                self.detailed_test_results.append(
                    {  # Adiciona diretamente à lista global
                        "run_index": run_idx,
                        "ai": ai_name,
                        "challenge": challenge["name"],
                        "difficulty": challenge["difficulty"],
                        "test_case_index": i if challenge["name"] != "LRU Cache" else 0,
                        "input_args_str": "N/A",
                        "expected_output_str": "N/A",
                        "actual_output_str": None,
                        "status": "API Error",
                        "execution_time_ms": -1,
                        "error_message": api_error,
                        "stdout": None,
                    }
                )
            return fail_result

        # Prossegue se houver conteúdo (mesmo que tenha havido erro na API não fatal)
        code = self.extract_code(content)
        lines = self.count_code_lines(code)
        completeness = self.measure_response_completeness(
            code, challenge["description"]
        )
        flake8 = self.analyze_code_quality_flake8(code)

        raw_results = []
        details_curr_run = []
        n_errors = 0
        first_err = ""
        can_exec = bool(code)
        func_name = None

        if can_exec and challenge["name"] != "LRU Cache":
            func_name = self._detect_function_name(code, challenge)
            if not func_name:
                can_exec = False

        if can_exec:
            if challenge["name"] == "LRU Cache":
                print(
                    f"  [Run {run_idx+1}] Executando LRU Cache de {ai_name} localmente..."
                )
                tc = challenge.get("test_cases", [])
                if tc and isinstance(tc[0], dict) and "sequence" in tc[0]:
                    lru_res = self._execute_lru_locally(code, tc[0]["sequence"])
                    raw_results = lru_res.get("results", [])
                    ok = lru_res.get("success", False)
                    n_errors = 0 if ok else 1
                    first_err = lru_res.get("error", "")
                    if not first_err and not ok:
                        for step in raw_results:
                            if step.get("err"):
                                first_err = (
                                    f"Step {step['op']}{step['args']}: {step['err']}"
                                )
                                break
                    if not first_err and not ok:
                        first_err = "LRU execution failed (unknown step error)"
                    t_ok = [
                        s["t_ms"]
                        for s in raw_results
                        if s.get("t_ms", -1) >= 0 and s.get("ok") is True
                    ]
                    avg_t = np.mean(t_ok) if t_ok else -1.0
                    details_curr_run.append(
                        {
                            "run_index": run_idx,  # Adiciona run_idx
                            "ai": ai_name,
                            "challenge": challenge["name"],
                            "difficulty": challenge["difficulty"],
                            "test_case_index": 0,
                            "input_args_str": json.dumps(tc[0]["sequence"]),
                            "expected_output_str": "Sequence Result",
                            "actual_output_str": f"Success: {ok}",
                            "status": (
                                "Pass" if ok else ("Fail" if raw_results else "Error")
                            ),
                            "execution_time_ms": avg_t,
                            "error_message": str(first_err)[:250],
                            "stdout": None,
                        }
                    )
                    print(
                        f"  [Run {run_idx+1}] Resultado LRU local: {'Sucesso' if ok else 'Falha/Erro'}"
                    )
                else:
                    print(f"  AVISO [Run {run_idx+1}]: Casos teste LRU inválidos.")
                    n_errors = 1
                    first_err = "Invalid LRU test case format."
                    details_curr_run.append(
                        {
                            "run_index": run_idx,
                            "ai": ai_name,
                            "challenge": challenge["name"],
                            "difficulty": challenge["difficulty"],
                            "test_case_index": 0,
                            "input_args_str": "N/A",
                            "expected_output_str": "N/A",
                            "actual_output_str": None,
                            "status": "Error",
                            "execution_time_ms": -1,
                            "error_message": first_err,
                            "stdout": None,
                        }
                    )
            else:  # Outros desafios
                print(
                    f"  [Run {run_idx+1}] Executando '{challenge['name']}' de {ai_name} via Executor Service..."
                )
                # Passa run_idx para _run_test_cases_fixed
                raw, details_from_tests, errors, first_e = self._run_test_cases_fixed(
                    ai_name, challenge, code, func_name, run_idx
                )
                raw_results = raw
                n_errors = errors
                first_err = first_e
                details_curr_run.extend(details_from_tests)
        else:  # Sem código ou nome função
            err_msg = (
                "Code not extracted" if not code else "Function/Class name not found"
            )
            print(
                f"  AVISO [Run {run_idx+1}]: {err_msg} para '{challenge['name']}' por {ai_name}. Pulando execução."
            )
            n_tests = len(challenge.get("test_cases", []))
            n_errors = n_tests
            first_err = err_msg
            for i in range(n_tests):
                details_curr_run.append(
                    {
                        "run_index": run_idx,  # Adiciona run_idx
                        "ai": ai_name,
                        "challenge": challenge["name"],
                        "difficulty": challenge["difficulty"],
                        "test_case_index": i,
                        "input_args_str": None,
                        "expected_output_str": None,
                        "actual_output_str": None,
                        "status": "No Code",
                        "execution_time_ms": -1,
                        "error_message": err_msg,
                        "stdout": None,
                    }
                )

        # Adiciona detalhes desta execução à lista GERAL da instância
        self.detailed_test_results.extend(details_curr_run)

        # Retorna dicionário agregado desta execução específica
        return {
            "run_index": run_idx,  # Adiciona run_idx
            "ai": ai_name,
            "challenge": challenge["name"],
            "difficulty": challenge["difficulty"],
            "api_execution_time": api_time,
            "lines_of_code": lines,
            "completeness": completeness,
            "flake8_issues": flake8,
            "num_execution_errors": n_errors,  # Erros/falhas NESTA execução
            "first_execution_error": first_err,
            "code": code,
            "full_response": content,
            "execution_results_raw": raw_results,
        }

    # --- FIM DA MODIFICAÇÃO evaluate_solution ---

    def _detect_function_name(self, code: str, challenge: dict) -> str | None:
        """Detecta nome da função/classe, priorizando assinatura e nome esperado."""
        # (Código _detect_function_name como estava antes)
        if not code:
            return None
        expected_name = None
        signature = challenge.get("function_signature", "")
        sig_match = re.match(r"^\s*(?:def|class)\s+(\w+)", signature)
        if sig_match:
            expected_name = sig_match.group(1)
            simple_def = f"def {expected_name}("
            simple_class = f"class {expected_name}:"
            simple_class_paren = f"class {expected_name}("
            if (
                re.search(rf"^\s*{re.escape(simple_def)}", code, re.MULTILINE)
                or re.search(rf"^\s*{re.escape(simple_class)}", code, re.MULTILINE)
                or re.search(
                    rf"^\s*{re.escape(simple_class_paren)}", code, re.MULTILINE
                )
            ):
                return expected_name
            pattern = rf"\s*(?:def|class)\s+{re.escape(expected_name)}\b"
            if re.search(pattern, code, re.MULTILINE):
                return expected_name
            if re.search(rf"\b{re.escape(expected_name)}\b", code):
                print(
                    f"  AVISO: Nome esperado '{expected_name}' encontrado, mas não como definição padrão. Usando '{expected_name}'."
                )
                return expected_name
        first_match = re.search(r"^\s*(?:def|class)\s+(\w+)", code, re.MULTILINE)
        if first_match:
            found_name = first_match.group(1)
            if expected_name and found_name != expected_name:
                print(
                    f"  AVISO: Usando 1ª definição encontrada '{found_name}' (Esperado: '{expected_name}')."
                )
            elif not expected_name:
                print(f"  AVISO: Usando 1ª definição encontrada '{found_name}'.")
            return found_name
        print(
            f"  ERRO: Nenhuma definição de função ou classe encontrada para {challenge['name']}."
        )
        return None

    # --- PASSO 3: Modificar _run_test_cases_fixed para aceitar e usar run_idx ---
    def _run_test_cases_fixed(
        self,
        ai_name: str,
        challenge: dict,
        code: str,
        actual_function_name: str,
        run_idx: int,
    ) -> tuple[list, list, int, str]:
        """Executa casos de teste via Executor Service, incluindo run_idx nos detalhes."""
        raw_results = []
        details = []
        n_fails_errors = 0
        first_err = ""
        test_cases = challenge.get("test_cases", [])
        if not test_cases:
            print(f"  Aviso: Sem casos de teste para '{challenge['name']}'.")
            return [], [], 0, ""

        for i, tc in enumerate(test_cases):
            args = tc.get("input_args")
            exp_out = tc.get("expected_output")
            if args is None:
                print(
                    f"  Aviso: Caso {i+1} '{challenge['name']}' sem 'input_args'. Pulando."
                )
                continue

            exec_res = self._call_executor_service(code, actual_function_name, args)
            raw_results.append(exec_res)

            t_det = {
                "run_index": run_idx,  # <<< ADICIONA run_index
                "ai": ai_name,
                "challenge": challenge["name"],
                "difficulty": challenge["difficulty"],
                "test_case_index": i,
                "input_args_str": json.dumps(args),
                "expected_output_str": json.dumps(exp_out),
                "actual_output_str": None,
                "status": "Unknown",  # Default
                "execution_time_ms": exec_res.get("execution_time_ms", -1),
                "error_message": None,
                "stdout": exec_res.get("stdout"),
            }
            case_err = None

            if exec_res.get("success") is True:
                act_out = exec_res.get("output")
                t_det["actual_output_str"] = json.dumps(act_out)
                try:
                    # Nota: Comparação np.array_equal pode falhar para Two Sum se a ordem dos índices for diferente.
                    # Uma solução seria ordenar ambos act_out e exp_out antes de comparar se forem listas de ints.
                    # if challenge["name"] == "Two Sum" and isinstance(act_out, list) and isinstance(exp_out, list):
                    #     are_equal = sorted(act_out) == sorted(exp_out)
                    # else:
                    #     are_equal = np.array_equal(np.array(act_out, dtype=object), np.array(exp_out, dtype=object))
                    are_equal = np.array_equal(
                        np.array(act_out, dtype=object), np.array(exp_out, dtype=object)
                    )

                    if are_equal:
                        t_det["status"] = "Pass"
                    else:
                        t_det["status"] = "Fail"
                        case_err = f"Output != Expected (Case {i})"
                        n_fails_errors += 1
                except Exception as comp_e:
                    print(
                        f"  AVISO: Erro ao comparar outputs {ai_name}/{challenge['name']}/Caso {i}: {comp_e}"
                    )
                    t_det["status"] = "Fail (Comparison Error)"
                    t_det["error_message"] = f"Error comparing outputs: {comp_e}"
                    case_err = t_det["error_message"]
                    n_fails_errors += 1

            elif exec_res.get("success") is False:
                t_det["status"] = "Error"
                t_det["error_message"] = exec_res.get(
                    "error", "Execution error reported by service"
                )
                case_err = str(t_det["error_message"])[:250]
                n_fails_errors += 1
            else:
                t_det["status"] = "Unknown Executor Status"
                t_det["error_message"] = exec_res.get(
                    "error", "Unexpected response from executor service"
                )
                case_err = str(t_det["error_message"])[:250]
                n_fails_errors += 1

            if case_err and not first_err:
                first_err = case_err
            details.append(
                t_det
            )  # Adiciona detalhes (com run_idx) à lista desta chamada

        if n_fails_errors > 0 and not first_err:
            first_err = "One or more test cases failed or errored."
        return raw_results, details, n_fails_errors, first_err

    # --- FIM DA MODIFICAÇÃO _run_test_cases_fixed ---

    def _call_executor_service(
        self, code_string: str, function_name: str, input_args: list
    ) -> dict:
        """Chama o serviço executor externo."""
        # (Código _call_executor_service como estava antes)
        imports = (
            "from typing import List, Dict, Tuple, Set, Optional, Any\n"
            "import re\n"
            "from collections import OrderedDict, deque\n\n"
        )
        code_to_exec = imports + code_string
        payload = {
            "code_string": code_to_exec,
            "function_name": function_name,
            "test_input_args": input_args,
        }
        headers = {"Content-Type": "application/json"}
        default_err = {
            "success": False,
            "output": None,
            "stdout": None,
            "error": "Failed to call executor service",
            "execution_time_ms": -1,
        }
        try:
            resp = requests.post(
                self.executor_url, headers=headers, json=payload, timeout=60
            )
            if resp.status_code == 200:
                try:
                    return resp.json()
                except json.JSONDecodeError as json_e:
                    print(f"  Erro JSON decode executor: {json_e}")
                    default_err["error"] = f"Executor JSON decode error: {json_e}"
                    return default_err
            else:
                err_txt = resp.text[:200]
                print(f"  Erro executor status {resp.status_code}: {err_txt}...")
                default_err["error"] = f"Executor status {resp.status_code}: {err_txt}"
                return default_err
        except requests.exceptions.Timeout:
            print(f"  Erro Timeout ao chamar executor ({self.executor_url}).")
            default_err["error"] = "Executor service request timeout"
            return default_err
        except requests.exceptions.RequestException as req_e:
            print(f"  Erro conexão/rede com executor: {req_e}")
            default_err["error"] = f"Executor connection error: {req_e}"
            return default_err
        except Exception as e:
            print(f"  Erro inesperado ao chamar executor: {type(e).__name__} - {e}")
            default_err["error"] = (
                f"Unexpected error calling executor: {type(e).__name__}"
            )
            return default_err

    # --- PASSO 3: Modificar run_benchmark para loop e passar run_idx ---
    def run_benchmark(self):
        """Executa o ciclo completo do benchmark por 'self.num_runs' vezes."""
        can_claude = bool(self.claude_api_key)
        can_openai = bool(self.openai_api_key)
        can_gemini = bool(self.gemini_api_key and self.gemini_model)
        enabled_ais = [
            n
            for n, can in [
                ("Claude", can_claude),
                ("ChatGPT", can_openai),
                ("Gemini", can_gemini),
            ]
            if can
        ]
        if not enabled_ais:
            print("ERRO: Nenhuma API configurada ou disponível. Saindo.")
            return

        print(
            f"Iniciando Benchmark para: {', '.join(enabled_ais)} ({self.num_runs} execução(ões) por desafio)"
        )
        print(f"  Claude Model: {self.claude_model if can_claude else 'N/A'}")
        print(f"  ChatGPT Model: {self.openai_model if can_openai else 'N/A'}")
        print(f"  Gemini Model: {self.gemini_model or 'N/A'}")
        print(f"  LRU Cache Execution: Local")
        print(f"  Executor Service URL: {self.executor_url}")
        print("=" * 42)

        # PASSO 2: Atualizar ch_details com novos desafios
        ch_details = {
            "Fibonacci Sequence": "Técnica: Geração de sequência numérica...\n  Finalidade: Avaliar algoritmos fundamentais...",
            "Binary Search": "Técnica: Algoritmo eficiente de busca...\n  Finalidade: Testar busca e comparação...",
            "Merge Sort": "Técnica: Algoritmo de ordenação eficiente...\n  Finalidade: Avaliar recursão e combinação...",
            "Count Islands in Grid": "Técnica: Travessia de grafo (DFS/BFS)...\n  Finalidade: Testar modelagem de grafos e travessia...",
            "Palindrome Check": "Técnica: Manipulação de strings...\n  Finalidade: Avaliar processamento de strings...",
            "Valid Parentheses": "Técnica: Uso de Pilha (Stack)...\n  Finalidade: Testar uso de pilhas...",
            "LRU Cache": "Técnica: Estrutura de dados customizada (OrderedDict)...\n  Finalidade: Avaliar criação de estruturas complexas...",
            "Two Sum": "Técnica: Uso de dicionário (hash map) para busca rápida de complemento.\n  Finalidade: Avaliar uso eficiente de dicionários.",
            "Longest Substring Without Repeating Characters": "Técnica: Janela deslizante (sliding window) com conjunto/dicionário.\n  Finalidade: Testar janelas deslizantes e verificação de unicidade.",
            "Minimum Edit Distance": "Técnica: Programação Dinâmica (PD) com matriz 2D (Levenshtein).\n  Finalidade: Avaliar resolução de problemas com PD.",
        }

        # Reinicia listas GLOBAIS antes de iniciar as execuções
        self.results = []
        self.detailed_test_results = []

        # Loop principal das execuções
        for run_idx in range(self.num_runs):
            print(
                f"\n{'='*20} INICIANDO EXECUÇÃO {run_idx + 1}/{self.num_runs} {'='*20}"
            )
            run_start_time = time.time()

            for chal in self.code_challenges:
                sig = chal.get(
                    "function_signature",
                    f"{chal['name'].lower().replace(' ','_')}(...)",
                )
                prompt = f"Implement the following Python {'class' if 'class' in sig else 'function'}.\nDescription: {chal['description']}\nSignature:\n{sig}\nProvide ONLY the Python code definition for the function/class and any necessary standard library imports (like `from collections import OrderedDict` if needed). Do not include any explanations, comments within the code, examples, or markdown formatting like ```python."

                print(
                    f"\n--- [Run {run_idx+1}] Desafio: {chal['name']} ({chal['difficulty']}) ---"
                )
                print(f"  {ch_details.get(chal['name'],'Explicação técnica N/A.')}")

                resps = {}
                if can_claude:
                    print(f"  [Run {run_idx+1}] Consultando Claude AI...")
                    resps["Claude"] = self.query_claude(prompt)
                    time.sleep(1)
                if can_openai:
                    print(f"  [Run {run_idx+1}] Consultando ChatGPT...")
                    resps["ChatGPT"] = self.query_chatgpt(prompt)
                    time.sleep(1)
                if can_gemini:
                    print(f"  [Run {run_idx+1}] Consultando Gemini AI...")
                    resps["Gemini"] = self.query_gemini(
                        prompt
                    )  # Tenacity lida com espera/retry

                summary = []
                for ai in enabled_ais:
                    resp_data = resps.get(
                        ai,
                        {
                            "content": "",
                            "execution_time": 0,
                            "last_error": "No Response Attempted",
                        },
                    )

                    # Avalia a solução passando run_idx
                    # A função evaluate_solution agora adiciona aos GLOBAIS self.results e self.detailed_test_results
                    result = self.evaluate_solution(ai, chal, resp_data, run_idx)

                    # Prepara linha de resumo para ESTA run/desafio/ia (usa o 'result' retornado)
                    n_err_fail = result.get("num_execution_errors", 0)
                    # Pega detalhes específicos desta AI/Challenge/RUN
                    det = [
                        d
                        for d in self.detailed_test_results
                        if d["ai"] == ai
                        and d["challenge"] == chal["name"]
                        and d["run_index"] == run_idx
                    ]
                    n_att = len(
                        [d for d in det if d["status"] not in ["No Code", "API Error"]]
                    )
                    n_pass = len([d for d in det if d["status"] == "Pass"])
                    f8 = (
                        str(result["flake8_issues"])
                        if result["flake8_issues"] != -1
                        else "N/A"
                    )

                    if (
                        result.get("first_execution_error") == "API Error or No Content"
                        or result.get("first_execution_error") == "Code not extracted"
                        or result.get("first_execution_error")
                        == "Function/Class name not found"
                    ):
                        status = result.get(
                            "first_execution_error"
                        )  # Mostra o erro da API/Extração
                    elif n_att == 0:
                        status = "Not Run (Unknown)"  # Caso não coberto acima
                    elif chal["name"] == "LRU Cache":
                        status = det[0]["status"] if det else "Error Evaluating"
                    else:
                        status = f"{n_pass}/{n_att} passed"

                    summary.append(
                        f"  {ai:<10}: Exec: {status}. Flake8: {f8}. API Time: {result['api_execution_time']:.2f}s."
                    )

                print(f"--- [Run {run_idx+1}] Resumo Desafio {chal['name']} ---")
                for s_line in summary:
                    print(s_line)
                print(f"--- [Run {run_idx+1}] Desafio {chal['name']} concluído ---")

            run_end_time = time.time()
            print(
                f"\n{'='*20} FIM EXECUÇÃO {run_idx + 1}/{self.num_runs} (Duração: {run_end_time - run_start_time:.2f}s) {'='*20}"
            )

        # Fim do loop de execuções

        if not self.results:
            print("\nNenhum resultado coletado durante o benchmark.")
            return
        # Gera relatório FINAL com dados acumulados de todas as execuções
        self.generate_report()

    # --- FIM DA MODIFICAÇÃO run_benchmark ---

    # --- Métodos de Geração de Relatório (AJUSTADOS para Múltiplas Execuções) ---

    # --- PASSO 5: Refatorar _prepare_dataframe ---
    def _prepare_dataframe(self) -> pd.DataFrame | None:
        """Prepara DataFrame AGREGADO (média sobre runs) a partir de self.results."""
        if not self.results:
            print("Não há resultados (self.results) para gerar DataFrame.")
            return None
        try:
            df_all_runs = pd.DataFrame(self.results)

            # Métricas para agregar pela MÉDIA sobre as execuções (runs)
            metrics_to_avg = ["api_execution_time", "lines_of_code", "completeness"]
            # Coluna Flake8 requer tratamento especial para -1
            metrics_to_avg_valid = ["flake8_issues"]

            # Agrupa por AI e Desafio
            grouped = df_all_runs.groupby(["ai", "challenge"])

            # Calcula médias simples para métricas diretas
            df_agg = grouped[metrics_to_avg].mean()

            # Calcula média de Flake8 ignorando -1
            def avg_valid_flake8(series):
                valid = series[series != -1]
                return valid.mean() if not valid.empty else -1

            df_agg["flake8_issues"] = grouped["flake8_issues"].apply(avg_valid_flake8)

            # Calcula taxa de sucesso GERAL e tempo médio de execução GERAL (sobre todos os casos/runs)
            # Usa a função _calculate_aggregated_test_metrics que já lê self.detailed_test_results (com run_idx)
            agg_test_metrics = self._calculate_aggregated_test_metrics()

            # Mapeia métricas de teste agregadas para o novo DataFrame
            df_agg["avg_exec_success_rate"] = df_agg.index.map(
                lambda idx: agg_test_metrics.get(idx, {}).get(
                    "avg_exec_success_rate", 0.0
                )
            )
            df_agg["avg_exec_time_ms"] = df_agg.index.map(
                lambda idx: agg_test_metrics.get(idx, {}).get(
                    "avg_exec_time_ms", np.nan
                )
            )
            df_agg["total_tests_passed"] = df_agg.index.map(
                lambda idx: agg_test_metrics.get(idx, {}).get("total_tests_passed", 0)
            )  # Total acumulado
            df_agg["total_tests_attempted"] = df_agg.index.map(
                lambda idx: agg_test_metrics.get(idx, {}).get(
                    "total_tests_attempted", 0
                )
            )  # Total acumulado

            # Calcula média de erros de execução por run
            df_agg["avg_num_execution_errors"] = grouped["num_execution_errors"].mean()
            # Pega o primeiro erro encontrado em qualquer run para exemplo
            df_agg["first_execution_error"] = grouped["first_execution_error"].apply(
                lambda x: x[x != ""].iloc[0] if not x[x != ""].empty else ""
            )

            # Adiciona difficulty de volta (já que group by removeu)
            # Pega difficulty do primeiro run para cada (ai, challenge)
            difficulty_map = df_all_runs.drop_duplicates(
                subset=["ai", "challenge"]
            ).set_index(["ai", "challenge"])["difficulty"]
            df_agg = df_agg.join(difficulty_map)  # Junta usando o índice

            df_agg = df_agg.reset_index()  # Reset index para normalização funcionar bem

            # Normaliza os tipos de dados no DataFrame agregado final
            # Renomeia avg_num_execution_errors -> num_execution_errors para compatibilidade com normalização/stats
            df_agg = df_agg.rename(
                columns={"avg_num_execution_errors": "num_execution_errors"}
            )
            self._normalize_dataframe_types(df_agg)

            return df_agg

        except Exception as e:
            print(f"Erro ao preparar DataFrame agregado: {e}")
            traceback.print_exc()
            return None

    # --- FIM DA REFATORAÇÃO _prepare_dataframe ---

    # _calculate_aggregated_test_metrics permanece o mesmo, pois já lê todos os detalhes acumulados
    def _calculate_aggregated_test_metrics(self) -> dict:
        """Calcula métricas agregadas a partir de self.detailed_test_results (TODAS as runs)."""
        agg = {}
        keys = set((d["ai"], d["challenge"]) for d in self.detailed_test_results)
        for k in keys:
            ai, chal = k
            dets = [
                d
                for d in self.detailed_test_results
                if d["ai"] == ai and d["challenge"] == chal
            ]
            if not dets:
                continue
            passed = [d for d in dets if d["status"] == "Pass"]
            attempted = [d for d in dets if d["status"] not in ["No Code", "API Error"]]
            times = [
                d["execution_time_ms"]
                for d in dets
                if d.get("execution_time_ms", -1) >= 0
            ]
            n_att = len(attempted)
            n_pass = len(passed)
            rate = n_pass / n_att if n_att > 0 else 0.0
            mean_t = np.mean(times) if times else np.nan
            agg[k] = {
                "avg_exec_success_rate": rate,
                "avg_exec_time_ms": mean_t,
                "total_tests_passed": n_pass,
                "total_tests_attempted": n_att,
            }
        return agg

    # _normalize_dataframe_types permanece o mesmo
    def _normalize_dataframe_types(self, df: pd.DataFrame):
        """Normaliza tipos de colunas numéricas e trata NaNs."""
        numeric_cols = [
            "api_execution_time",
            "lines_of_code",
            "completeness",
            "flake8_issues",
            "num_execution_errors",
            "avg_exec_success_rate",
            "avg_exec_time_ms",
            "total_tests_passed",
            "total_tests_attempted",
        ]
        for c in numeric_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
                fill_value = 0.0
                target_dtype = float
                if c == "flake8_issues":
                    fill_value = -1
                    target_dtype = int
                elif c in [
                    "lines_of_code",
                    "completeness",
                    "num_execution_errors",
                    "total_tests_passed",
                    "total_tests_attempted",
                ]:
                    fill_value = 0
                    target_dtype = int
                elif c == "avg_exec_time_ms":
                    fill_value = -1.0
                df.loc[:, c] = df[c].fillna(fill_value)
                if target_dtype == int:
                    try:
                        df.loc[:, c] = df[c].astype(int)
                    except ValueError as e:
                        print(
                            f"Aviso: Falha converter coluna '{c}' para int após fillna: {e}"
                        )
                        df.loc[:, c] = df[c].astype(float)
            # else: print(f"Aviso: Coluna numérica esperada '{c}' não encontrada no DataFrame para normalização.") # Opcional
        if "first_execution_error" in df.columns:
            df["first_execution_error"] = (
                df["first_execution_error"].fillna("").astype(str)
            )
        elif (
            "first_execution_error" not in df.columns
            and "num_execution_errors" in df.columns
        ):
            df["first_execution_error"] = ""

    def _save_environment_info(self, ts: str):
        """Salva info do ambiente em TXT."""
        p = os.path.join(self.output_dir, f"environment_info_{ts}.txt")
        try:
            with open(p, "w", encoding="utf-8") as f:
                # Adiciona o número de runs ao início da info
                f.write(f"Number of Runs per Challenge/AI: {self.num_runs}\n")
                f.write("-" * 30 + "\n")
                for k, v in self.environment_info.items():
                    # Evita duplicar a info do num_runs que já estava no dict (se houver)
                    if k == "Num Runs":
                        continue
                    if isinstance(v, dict):
                        f.write(f"{k}:\n")
                        for sk, sv in v.items():
                            f.write(f"  {sk}: {sv}\n")
                    else:
                        f.write(f"{k}: {v}\n")
            print(f"- Info Ambiente salvo: {p}")
        except Exception as e:
            print(f"Erro ao salvar info do ambiente: {e}")

    def _save_full_results_json(self, ts: str):
        """Salva resultados completos (run-a-run) em JSON."""
        p = os.path.join(
            self.output_dir, f"benchmark_results_all_runs_{ts}.json"
        )  # Nome modificado
        try:

            class NpEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, datetime):
                        return obj.isoformat()
                    elif isinstance(obj, np.bool_):
                        return bool(obj)
                    elif isinstance(obj, bytes):
                        return obj.decode("utf-8", "replace")
                    return super(NpEncoder, self).default(obj)

            with open(p, "w", encoding="utf-8") as f:
                json.dump(
                    self.results, f, indent=2, ensure_ascii=False, cls=NpEncoder
                )  # Salva self.results (todos os runs)
            print(f"- Resultados JSON (todos os runs) salvo: {p}")
        except Exception as e:
            print(f"Erro ao salvar JSON completo: {e}")

    def _save_detailed_tests_csv(self, ts: str):
        """Salva detalhes dos casos de teste individuais (todos os runs) em CSV."""
        p = os.path.join(
            self.output_dir, f"benchmark_test_details_all_runs_{ts}.csv"
        )  # Nome modificado
        if self.detailed_test_results:
            try:
                df = pd.DataFrame(self.detailed_test_results)
                # Adiciona run_index como primeira coluna
                cols = [
                    "run_index",
                    "ai",
                    "challenge",
                    "difficulty",
                    "test_case_index",
                    "status",
                    "execution_time_ms",
                    "error_message",
                    "stdout",
                    "input_args_str",
                    "expected_output_str",
                    "actual_output_str",
                ]
                for c in cols:
                    if c not in df.columns:
                        df[c] = pd.NA
                df[cols].to_csv(p, index=False, encoding="utf-8", float_format="%.3f")
                print(f"- Detalhes dos Testes CSV (todos os runs) salvo: {p}")
            except Exception as e:
                print(f"Erro ao salvar CSV de detalhes dos testes: {e}")
        else:
            print("- Nenhum resultado detalhado de teste para salvar em CSV.")

    def _create_csv_header(self) -> str:
        """Cria cabeçalho com info do ambiente (incluindo num_runs) para CSVs."""
        hdr = ["# --- Environment Info ---"]
        hdr.append(f"# Number of Runs per Challenge/AI: {self.num_runs}")
        for k, v in self.environment_info.items():
            if k == "Num Runs":
                continue  # Já adicionado
            if isinstance(v, dict):
                hdr.append(f"# {k}:")
                [hdr.append(f"#   {sk}: {sv}") for sk, sv in v.items()]
            else:
                hdr.append(f"# {k}: {v}")
        hdr.append("# --- Benchmark Data (Aggregated/Averaged over Runs) ---")
        return "\n".join(hdr) + "\n"  # Cabeçalho indica agregação

    def _save_summary_csv(self, df_agg: pd.DataFrame, ts: str):
        """Salva resumo agregado (média sobre runs) em CSV."""
        p = os.path.join(
            self.output_dir, f"benchmark_summary_agg_{ts}.csv"
        )  # Nome modificado
        if df_agg is None:
            print("Erro: DataFrame agregado é None. Pulando salvamento CSV.")
            return
        try:
            # Todas as colunas do df_agg são relevantes para o sumário agregado
            hdr = self._create_csv_header()
            with open(p, "w", encoding="utf-8", newline="") as f:
                f.write(hdr)
                df_agg.to_csv(
                    f,
                    index=False,
                    encoding="utf-8",
                    float_format="%.4f",
                    lineterminator="\n",
                )
            print(f"- Resumo Agregado CSV salvo: {p}")
        except Exception as e:
            print(f"Erro ao salvar resumo agregado CSV: {e}")

    # --- PASSO 5: Refatorar _calculate_statistics ---
    def _calculate_statistics(self, df_agg: pd.DataFrame) -> pd.DataFrame | None:
        """Calcula estatísticas gerais agregadas por IA a partir do DataFrame JÁ AGREGADO (média sobre runs) por desafio."""
        if df_agg is None:
            print("Erro: DataFrame agregado de entrada para estatísticas é None.")
            return None
        try:
            # Define métricas a calcular: (Nome Exibição, Coluna Original no df_agg, Tipo Agregação)
            # Adicionamos Std Dev para algumas métricas chave
            metrics_def = [
                ("Avg API Time (s)", "api_execution_time", "mean"),
                ("Std Dev API Time (s)", "api_execution_time", "std"),  # NOVO
                ("Avg LoC", "lines_of_code", "mean"),
                ("Std Dev LoC", "lines_of_code", "std"),  # NOVO
                ("Avg Completeness (%)", "completeness", "mean"),
                (
                    "Avg Flake8 Issues",
                    "flake8_issues",
                    "mean_valid",
                ),  # Média ignorando -1
                (
                    "Total Passed (All Runs)",
                    "total_tests_passed",
                    "sum",
                ),  # Soma total sobre desafios
                (
                    "Total Attempted (All Runs)",
                    "total_tests_attempted",
                    "sum",
                ),  # Soma total sobre desafios
                (
                    "Overall Success Rate (%)",
                    "avg_exec_success_rate",
                    "mean_percentage",
                ),  # Média das taxas por desafio
                (
                    "Std Dev Success Rate (%)",
                    "avg_exec_success_rate",
                    "std_percentage",
                ),  # NOVO
                (
                    "Avg Exec Time (ms)",
                    "avg_exec_time_ms",
                    "mean_valid",
                ),  # Média ignorando -1 / NaN
                (
                    "Avg Errors/Fails per Challenge",
                    "num_execution_errors",
                    "mean",
                ),  # Média dos erros por desafio (já é média das runs)
                (
                    "Example Error",
                    "first_execution_error",
                    "first_non_empty",
                ),  # Pega primeiro erro não vazio
            ]

            ais = sorted(df_agg["ai"].unique())
            stats_data = {
                "Metric": [m[0] for m in metrics_def]
            }  # Coluna com nomes das métricas

            for ai in ais:
                df_ai = df_agg[df_agg["ai"] == ai]
                col_data = []  # Filtra dados para a IA atual
                for _, col_key, agg_type in metrics_def:
                    value = np.nan  # Valor padrão
                    if col_key in df_ai.columns:
                        series = df_ai[col_key]
                        if (
                            series.empty
                        ):  # Lida com caso onde a IA não tem dados para uma coluna
                            value = np.nan
                        elif agg_type == "mean":
                            value = series.mean()
                        elif agg_type == "sum":
                            value = series.sum()
                        elif agg_type == "std":
                            value = series.std()  # Calcula Std Dev
                        elif agg_type == "mean_valid":
                            valid_data = series[series.notna() & (series != -1)]
                            value = (
                                valid_data.mean() if not valid_data.empty else np.nan
                            )
                        elif agg_type == "mean_percentage":
                            value = series.mean() * 100 if series.notna().any() else 0.0
                        elif agg_type == "std_percentage":  # Std Dev da taxa de sucesso
                            value = series.std() * 100 if series.notna().any() else 0.0
                        elif agg_type == "first_non_empty":
                            errors = series[series.fillna("").astype(str) != ""]
                            value = errors.iloc[0] if not errors.empty else ""
                    col_data.append(value)
                stats_data[ai] = col_data

            return pd.DataFrame(stats_data)
        except Exception as e:
            print(f"Erro ao calcular estatísticas: {e}")
            traceback.print_exc()
            return None

    # --- FIM DA REFATORAÇÃO _calculate_statistics ---

    def _save_statistics_csv(self, stats_df: pd.DataFrame, ts: str):
        """Salva estatísticas gerais agregadas em CSV."""
        if stats_df is None:
            print("Erro: DataFrame de estatísticas é None. Pulando salvamento CSV.")
            return
        p = os.path.join(
            self.output_dir, f"benchmark_stats_agg_{ts}.csv"
        )  # Nome modificado
        try:
            hdr = self._create_csv_header()
            # Cabeçalho indica agregação
            with open(p, "w", encoding="utf-8", newline="") as f:
                f.write(hdr)
                stats_df.to_csv(
                    f,
                    index=False,
                    encoding="utf-8",
                    float_format="%.2f",
                    lineterminator="\n",
                )
            print(f"- Estatísticas Gerais CSV salvo: {p}")
        except Exception as e:
            print(f"Erro ao salvar estatísticas CSV: {e}")

    def _display_summary_table(self, stats_df: pd.DataFrame):
        """Mostra tabela de resumo formatada no console."""
        print(
            f"\n===== SUMÁRIO DO BENCHMARK (ESTATÍSTICAS GERAIS - {self.num_runs} RUNS) ====="
        )  # Adiciona info de runs
        if stats_df is None or stats_df.empty:
            print("Não foi possível gerar sumário (DataFrame vazio ou None).")
            return

        disp = stats_df.copy()
        # Formata colunas numéricas para exibição
        for col_name in disp.columns:
            if col_name == "Metric":
                continue
            formatted_values = []
            for index, row in disp.iterrows():
                metric_name = row["Metric"]
                raw_value = row[col_name]
                fmt_value = "N/A"
                if not pd.isna(raw_value):
                    try:
                        # Adiciona formatação para Std Dev
                        if "Std Dev" in metric_name:
                            if "(%)" in metric_name:
                                fmt_value = f"±{raw_value:.1f}%"
                            elif "(s)" in metric_name:
                                fmt_value = f"±{raw_value:.2f}s"
                            else:
                                fmt_value = f"±{raw_value:.1f}"
                        elif "Flake8" in metric_name:
                            fmt_value = f"{raw_value:.1f}" if raw_value != -1 else "N/A"
                        elif "Time (ms)" in metric_name:
                            fmt_value = f"{raw_value:.1f}" if raw_value != -1 else "N/A"
                        elif "(%)" in metric_name:
                            fmt_value = f"{raw_value:.1f}%"
                        elif "Total " in metric_name:
                            fmt_value = f"{int(raw_value)}"
                        elif "Time (s)" in metric_name:
                            fmt_value = f"{raw_value:.2f}"
                        elif "Avg Errors/Fails" in metric_name:
                            fmt_value = f"{raw_value:.2f}"  # Média de erros
                        elif "Example Error" in metric_name:
                            s = str(raw_value)
                            fmt_value = s[:60] + ("..." if len(s) > 60 else "")
                        else:
                            fmt_value = f"{raw_value:.1f}"  # Default para LoC
                    except (ValueError, TypeError):
                        fmt_value = str(raw_value)
                formatted_values.append(fmt_value)
            disp[col_name] = formatted_values
        try:
            from tabulate import tabulate

            print(tabulate(disp, headers="keys", tablefmt="grid", showindex=False))
        except ImportError:
            print("(Instale 'tabulate' para tabela formatada)")
            print(disp.to_string(index=False))
        except Exception as e:
            print(f"Erro ao imprimir tabela com tabulate: {e}")
            print(disp.to_string(index=False))

    def generate_report(self):
        """Gera todos os relatórios e visualizações (AGREGADOS sobre runs)."""
        if not self.results:
            print("Não há resultados para gerar relatório.")
            return
        print("\n--- Gerando Relatórios Finais (Agregados) ---")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"Timestamp: {ts}")
        print(f"Diretório: '{self.output_dir}/'")
        self._save_environment_info(ts)

        # Prepara DF AGREGADO (média sobre runs)
        df_agg = self._prepare_dataframe()
        if df_agg is None:
            print(
                "Falha ao preparar DataFrame agregado. Abortando geração de relatórios."
            )
            return

        # Salva os diferentes arquivos de dados (acumulados e agregados)
        self._save_full_results_json(ts)
        # Todos os runs
        self._save_detailed_tests_csv(ts)
        # Todos os runs
        self._save_summary_csv(df_agg, ts)
        # Agregado (média sobre runs)

        # Calcula e salva estatísticas GERAIS (baseadas no df_agg)
        stats = self._calculate_statistics(df_agg)
        if stats is not None:
            self._save_statistics_csv(stats, ts)
            # Gera visualizações (baseadas no df_agg)
            try:
                print("\n--- Gerando Visualizações (baseadas em médias sobre runs) ---")
                self.create_visualizations(
                    df_agg, ts, self.output_dir
                )  # Passa o df AGREGADO
                print(f"- Visualizações salvas em '{self.output_dir}/'")
            except Exception as e:
                print(f"Erro ao gerar visualizações: {e}")
            # Exibe a tabela de sumário no console
            self._display_summary_table(stats)
        else:
            print("- Não foi possível calcular/salvar estatísticas.")
            print(
                "AVISO: Tabela de sumário e visualizações podem não ter sido geradas devido a erro no cálculo de estatísticas."
            )

    # --- Funções de Plotagem (operam sobre df_agg) ---

    def _plot_bar_charts(self, df_plot, ai_list, cmap, challenge_order, ts, out_dir):
        """Gera gráficos de barras comparativos por desafio (usando médias sobre runs)."""
        print("Gerando Gráficos de Barras...")
        # (Código _plot_bar_charts como estava antes, agora opera sobre df_agg)
        metrics_def = {
            "1_api_t": ("API Time", "s", "api_execution_time"),
            "2_loc": ("LoC", "Linhas", "lines_of_code"),
            "3_comp": ("Completude", "(%)", "completeness"),
            "4_f8": ("Flake8 Issues", "Issues", "flake8_issues"),
            "7_err": (
                "Avg Errors/Fails",
                "Avg Num Errors/Fails",
                "num_execution_errors",
            ),
        }
        if (
            "avg_exec_success_rate" in df_plot.columns
            and df_plot["avg_exec_success_rate"].notna().any()
        ):
            metrics_def["5_ok"] = (
                "Exec Success Rate",
                "Success Rate (%)",
                "avg_exec_success_rate",
            )
        if (
            "avg_exec_time_ms" in df_plot.columns
            and (df_plot["avg_exec_time_ms"] != -1).any()
        ):
            metrics_def["6_exec_t"] = (
                "Avg Exec Time (Steps)",
                "Time (ms)",
                "avg_exec_time_ms",
            )
        metrics_def = dict(sorted(metrics_def.items()))
        for key, (title, ylabel, col_name) in metrics_def.items():
            if col_name not in df_plot.columns:
                print(f"  Skip bar '{title}': Coluna '{col_name}' não encontrada.")
                continue
            data = df_plot.copy()
            if col_name == "flake8_issues":
                data = data[data[col_name] != -1].copy()
            elif col_name == "avg_exec_time_ms":
                data = data[data[col_name].notna() & (data[col_name] != -1)].copy()
            elif col_name == "num_execution_errors":
                data = data[
                    data[col_name].notna()
                ].copy()  # num_execution_errors agora é média
            elif col_name == "avg_exec_success_rate":
                data.loc[:, col_name] = data[col_name] * 100
            if data.empty or data[col_name].isnull().all():
                print(f"  Skip bar '{title}': Sem dados válidos após filtro.")
                continue
            try:
                plt.figure()
                pivot_table = data.pivot_table(
                    index="challenge", columns="ai", values=col_name, aggfunc="mean"
                )
                pivot_table = pivot_table.reindex(challenge_order, axis=0).reindex(
                    ai_list, axis=1
                )
                fill_val = (
                    0.0
                    if col_name
                    not in ["flake8_issues", "avg_exec_time_ms", "num_execution_errors"]
                    else np.nan
                )
                pivot_table.fillna(value=fill_val, inplace=True)
                if pivot_table.isnull().all().all():
                    print(f"  Skip bar '{title}': Tabela pivot apenas com NaN.")
                    plt.close()
                    continue
                num_ais = len(ai_list)
                ax = pivot_table.plot(
                    kind="bar",
                    figsize=(max(12, num_ais * 2), 7),
                    color=[cmap.get(ai, "grey") for ai in pivot_table.columns],
                    width=0.8,
                )
                ax.set_title(f"{title} por Desafio (Média sobre {self.num_runs} runs)")
                ax.set_ylabel(ylabel)
                ax.set_xlabel("Desafio")
                ax.tick_params(axis="x", rotation=45, labelsize=9)
                ax.tick_params(axis="y", labelsize=9)
                plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
                if col_name in ["completeness", "avg_exec_success_rate"]:
                    ax.set_ylim(0, 105)
                elif (
                    col_name in ["flake8_issues", "num_execution_errors"]
                    and not pivot_table.isnull().all().all()
                ):
                    max_val = pivot_table.max(skipna=True).max(skipna=True)
                    ax.set_ylim(
                        bottom=0,
                        top=max(1, max_val * 1.1) if not pd.isna(max_val) else 1,
                    )
                elif (
                    col_name == "avg_exec_time_ms"
                    and not pivot_table.isnull().all().all()
                ):
                    ax.set_ylim(bottom=0)
                ax.legend(title="IA", bbox_to_anchor=(1.02, 1), loc="upper left")
                plt.tight_layout(rect=[0, 0, 0.88, 1])
                path = os.path.join(out_dir, f"{key}_avg_{ts}.png")
                plt.savefig(path)
                plt.close()  # Nome indica média
            except Exception as e:
                print(f"  Erro ao gerar gráfico de barras para '{title}': {e}")
                plt.close()

    def _plot_lru_times_chart(self, detailed_df, ai_list, cmap, ts, out_dir):
        """Gera gráfico de barras para o tempo médio de operação do LRU Cache (média sobre runs)."""
        print("Gerando Gráfico de Tempo Médio - LRU Cache...")
        challenge_name = "LRU Cache"
        lru_data = detailed_df[
            (detailed_df["challenge"] == challenge_name)
            & (detailed_df["execution_time_ms"].notna())
            & (detailed_df["execution_time_ms"] >= 0)
        ].copy()
        if lru_data.empty:
            print(f"  Skip LRU Times chart: Sem dados válidos para '{challenge_name}'.")
            return

        # Calcula a MÉDIA do tempo médio por IA sobre todas as runs
        avg_times_per_ai = lru_data.groupby("ai")["execution_time_ms"].mean()
        avg_times_per_ai = avg_times_per_ai.reindex(ai_list).dropna()

        if avg_times_per_ai.empty:
            print(
                f"  Skip LRU Times chart: Sem dados válidos após agrupar/reindexar por IA."
            )
            return

        try:
            fig, ax = plt.subplots(figsize=(max(8, len(avg_times_per_ai) * 1.5), 6))
            colors_for_plot = [cmap.get(ai, "grey") for ai in avg_times_per_ai.index]
            avg_times_per_ai.plot(kind="bar", ax=ax, color=colors_for_plot, width=0.6)
            ax.set_title(
                f"Tempo Médio por Operação - {challenge_name} (Média sobre {self.num_runs} runs)"
            )  # Título atualizado
            ax.set_ylabel("Tempo Médio (ms)")
            ax.set_xlabel("Inteligência Artificial")
            ax.tick_params(axis="x", rotation=0, labelsize=10)
            ax.tick_params(axis="y", labelsize=9)
            ax.grid(axis="y", linestyle="--", alpha=0.7)
            for container in ax.containers:
                ax.bar_label(
                    container, fmt="%.3f", label_type="edge", padding=3, fontsize=8
                )
            ax.set_ylim(bottom=0, top=ax.get_ylim()[1] * 1.1)
            plt.tight_layout()
            plot_prefix = "8_lru_exec_time_avg"  # Nome indica média
            path = os.path.join(out_dir, f"{plot_prefix}_{ts}.png")
            plt.savefig(path)
            plt.close(fig)
            print(f"- Gráfico Tempos LRU (Média) salvo: {path}")
        except Exception as e:
            print(f"  Erro ao gerar gráfico de tempos LRU: {e}")
            plt.close()

    def _plot_box_plots(self, df_agg, ai_list, cmap, ts, out_dir):
        """Gera box plots para distribuição das métricas MÉDIAS por desafio."""
        print("Gerando Box Plots (distribuição das médias por desafio)...")
        # Usa df_agg (médias por desafio/IA sobre runs)
        df_plot = df_agg.copy()
        # (Resto do código _plot_box_plots como estava antes, operando sobre df_plot que agora contém médias)
        metrics_def = {
            "api_execution_time": {
                "t": "Distribuição API Time (Média)",
                "y": "Tempo(s)",
                "v": None,
            },
            "lines_of_code": {
                "t": "Distribuição LoC (Média)",
                "y": "Linhas",
                "v": (lambda d, c: d[c] > 0),
            },
            "completeness": {
                "t": "Distribuição Completude (Média)",
                "y": "Score (%)",
                "v": None,
            },
        }
        if (
            "flake8_issues" in df_plot.columns
            and (df_plot["flake8_issues"] != -1).any()
        ):
            metrics_def["flake8_issues"] = {
                "t": "Distribuição Flake8 Issues (Média)",
                "y": "Issues",
                "v": (lambda d, c: d[c] != -1),
            }
        if (
            "avg_exec_time_ms" in df_plot.columns
            and (
                df_plot["avg_exec_time_ms"].notna()
                & (df_plot["avg_exec_time_ms"] != -1)
            ).any()
        ):
            metrics_def["avg_exec_time_ms"] = {
                "t": "Distribuição Exec Time (Média)",
                "y": "Tempo (ms)",
                "v": (lambda d, c: d[c].notna() & (d[c] != -1)),
            }
        if (
            "num_execution_errors" in df_plot.columns
            and df_plot["num_execution_errors"].notna().any()
        ):
            if (df_plot["num_execution_errors"] > 0).any():
                metrics_def["num_execution_errors"] = {
                    "t": "Distribuição Erros/Falhas Exec (Média)",
                    "y": "Avg Num Errors/Fails",
                    "v": (lambda d, c: d[c].notna() & (d[c] > 0)),
                }
            else:
                print(
                    "  Skip box 'Distribuição Erros/Falhas Exec': Nenhum erro registrado."
                )
        plot_counter = 8
        if not metrics_def:
            print("Nenhuma métrica válida definida para box plots.")
            return
        for col_name, config in metrics_def.items():
            plot_counter += 1
            if col_name not in df_plot.columns:
                print(
                    f"  Skip box '{config['t']}': Coluna '{col_name}' N/A no df_plot."
                )
                continue
            plot_data = []
            plot_labels = []
            temp_df = df_plot.copy()
            filter_func = config.get("v")
            if filter_func:
                try:
                    filter_condition = filter_func(temp_df, col_name)
                    temp_df = temp_df[filter_condition]
                except Exception as e:
                    print(
                        f"  AVISO: Erro ao aplicar filtro para '{col_name}': {e}. Pulando filtro."
                    )
            if temp_df.empty or temp_df[col_name].isnull().all():
                print(
                    f"  Skip box '{config['t']}': Sem dados válidos após filtro para coluna '{col_name}'."
                )
                continue
            valid_ai_data_found = False
            for ai in ai_list:
                ai_data = temp_df[temp_df["ai"] == ai][col_name].dropna()
                if not ai_data.empty:
                    plot_data.append(ai_data.values)
                    plot_labels.append(ai)
                    valid_ai_data_found = True
            if not valid_ai_data_found:
                print(
                    f"  Skip box '{config['t']}': Sem dados válidos por IA após filtro para '{col_name}'."
                )
                continue
            try:
                fig, ax = plt.subplots(figsize=(max(8, len(plot_labels) * 1.2), 6))
                bp = ax.boxplot(
                    plot_data,
                    tick_labels=plot_labels,
                    patch_artist=True,
                    showfliers=True,
                )
                ax.set_title(config["t"], fontsize=14)
                ax.set_ylabel(config["y"], fontsize=10)
                ax.set_xlabel("IA", fontsize=10)
                ax.tick_params(axis="x", rotation=30, labelsize=9)
                ax.tick_params(axis="y", labelsize=9)
                plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
                ax.grid(axis="y", linestyle="--", alpha=0.7)
                colors = [cmap.get(lbl, "grey") for lbl in plot_labels]
                [
                    patch.set_facecolor(color) or patch.set_alpha(0.6)
                    for patch, color in zip(bp["boxes"], colors)
                ]
                plt.tight_layout()
                fname = f"{plot_counter}_boxplot_avg_{col_name}_{ts}.png"
                p = os.path.join(out_dir, fname)
                plt.savefig(p)
                plt.close(fig)
                print(f"  Gráfico box plot salvo: {p}")
            except Exception as e:
                print(f"  Erro ao gerar box plot para '{config['t']}': {e}")
                plt.close() if "fig" in locals() else None
        print("Geração Box Plots concluída.")

    def _plot_radar_chart(self, df_agg, ai_list, cmap, ts, out_dir):
        """Gera gráfico de radar comparativo (Normalizado, baseado em médias sobre runs)."""
        n_ais = len(ai_list)
        if n_ais < 2:
            print("Radar chart pulado (< 2 IAs para comparar).")
            return
        print(f"Gerando gráfico de radar para {n_ais} IAs...")
        df = df_agg  # Usa o DataFrame já agregado
        plot_counter = 14  # Define contador inicial para nome do arquivo

        radar_cols_def = {
            "api_execution_time": {
                "label": "Veloc API (Norm)",
                "lower_is_better": True,
            },
            "lines_of_code": {"label": "Conc LoC (Norm)", "lower_is_better": True},
            "completeness": {"label": "Completude (%)", "lower_is_better": False},
            "flake8_issues": {
                "label": "Qualid Flake8 (Norm)",
                "lower_is_better": True,
                "ignore_val": -1,
            },
            "avg_exec_success_rate": {
                "label": "Taxa Sucesso Exec (%)",
                "lower_is_better": False,
            },
        }
        valid_radar_cols = {k: v for k, v in radar_cols_def.items() if k in df.columns}
        if len(valid_radar_cols) < 3:
            print(
                f"Radar chart pulado (< 3 métricas válidas encontradas no DF: {list(valid_radar_cols.keys())})."
            )
            return

        categories = [v["label"] for v in valid_radar_cols.values()]
        n_cats = len(categories)
        # Calcula médias GERAIS por IA sobre as médias dos desafios
        means = df.groupby("ai")[list(valid_radar_cols.keys())].mean()
        norm_data = pd.DataFrame(index=means.index)
        for col, config in valid_radar_cols.items():
            series = means[col].copy()
            ignore_val = config.get("ignore_val")
            if ignore_val is not None:
                series = series[series != ignore_val]
            min_val = series.min()
            max_val = series.max()
            range_val = max_val - min_val if max_val > min_val else 1
            if config["lower_is_better"]:
                norm_series = 100 * (max_val - means[col]) / range_val
            else:
                norm_series = 100 * (means[col] - min_val) / range_val
            if max_val == min_val:
                norm_series = 100 if not config["lower_is_better"] else 0
            if ignore_val is not None:
                norm_series = norm_series.reindex(means.index).fillna(
                    0 if config["lower_is_better"] else 100
                )
            norm_data[config["label"]] = norm_series.clip(0, 100)

        angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
        angles += angles[:1]
        fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.set_ylim(0, 100)
        for idx, ai in enumerate(norm_data.index):
            if ai not in ai_list:
                continue
            values = norm_data.loc[ai].values.flatten().tolist()
            values += values[:1]
            color = cmap.get(ai, plt.cm.tab10(idx / n_ais))
            ax.plot(angles, values, "o-", linewidth=2, label=ai, color=color)
            ax.fill(angles, values, color, alpha=0.2)
        plt.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15))
        plt.title("Comparação Geral Normalizada (Maior = Melhor)", size=14, y=1.18)
        plt.tight_layout()
        path = os.path.join(out_dir, f"{plot_counter + 1}_radar_chart_avg_{ts}.png")
        plt.savefig(path)
        plt.close(fig)  # Nome indica média
        print(f"  Gráfico Radar salvo: {path}")

    def create_visualizations(self, df_agg, ts, out_dir):
        """Cria e salva visualizações (baseadas em médias sobre runs)."""
        if df_agg is None or df_agg.empty:
            print(
                "AVISO: DataFrame agregado vazio para gráficos. Pulando visualizações."
            )
            return

        plt.style.use("ggplot")
        ais = sorted(df_agg["ai"].unique())
        n_ais = len(ais)
        if n_ais == 0:
            print("AVISO: Nenhuma AI encontrada nos dados agregados para plotar.")
            return

        colors = (
            plt.cm.viridis(np.linspace(0, 1, n_ais))
            if n_ais > 10
            else plt.cm.tab10(np.linspace(0, 1, n_ais))
        )
        cmap = {ai: colors[i] for i, ai in enumerate(ais)}

        required_cols = ["challenge", "ai", "difficulty"]
        if not all(c in df_agg.columns for c in required_cols):
            missing = [c for c in required_cols if c not in df_agg.columns]
            print(
                f"AVISO: Faltando colunas essenciais no DF agregado: {missing}. Pulando visualizações."
            )
            return

        challenge_order = sorted(df_agg["challenge"].unique())
        if "difficulty" in df_agg.columns:
            try:
                difficulty_order = pd.CategoricalDtype(
                    ["Easy", "Medium", "Hard"], ordered=True
                )
                # Cria coluna temporária para ordenação se não existir
                if "diff_cat" not in df_agg.columns:
                    df_agg["diff_cat"] = df_agg["difficulty"].astype(difficulty_order)
                challenge_order = df_agg.sort_values("diff_cat")["challenge"].unique()
            except Exception as e_cat:
                print(
                    f"AVISO: Erro ao ordenar desafios por dificuldade ({e_cat}). Usando ordem alfabética."
                )

        # Prepara DataFrame Detalhado APENAS para Plot LRU
        detailed_df = None
        if self.detailed_test_results:
            try:
                detailed_df = pd.DataFrame(self.detailed_test_results)
                detailed_df["execution_time_ms"] = pd.to_numeric(
                    detailed_df["execution_time_ms"], errors="coerce"
                )
            except Exception as e:
                print(f"AVISO: Erro ao preparar DataFrame detalhado para plot LRU: {e}")
                detailed_df = None
        else:
            print(
                "AVISO: Sem dados detalhados (`detailed_test_results`) para plot LRU."
            )

        # Plots gerais usam df_agg (médias)
        self._plot_bar_charts(df_agg, ais, cmap, challenge_order, ts, out_dir)
        self._plot_box_plots(df_agg, ais, cmap, ts, out_dir)
        self._plot_radar_chart(df_agg, ais, cmap, ts, out_dir)

        # Plot LRU usa detailed_df (todos os runs) para calcular a média
        if detailed_df is not None:
            self._plot_lru_times_chart(detailed_df, ais, cmap, ts, out_dir)


# --- Ponto de Entrada ---
if __name__ == "__main__":
    print("=" * 59)
    print(" AI Programming Benchmark: Claude vs ChatGPT vs Gemini")
    print(
        f"       (Incluindo Análise Flake8 e Execução Externa/Local)"
    )  # Título ajustado
    print("=" * 59)

    multiprocessing.freeze_support()

    if not os.path.exists(".env"):
        print(
            "AVISO: Ficheiro .env não encontrado. Chaves API podem não ser carregadas."
        )

    try:
        # --- PASSO 3: Definir número de execuções ---
        n_runs = 3  # Exemplo: executar 3 vezes (pode vir do .env ou argparse)
        benchmark = AIBenchmark(num_runs=n_runs)

        if (
            benchmark.claude_api_key
            or benchmark.openai_api_key
            or benchmark.gemini_api_key
        ):
            benchmark.run_benchmark()
            print("\nBenchmark concluído.")
            print(f"Relatórios salvos em: '{benchmark.output_dir}'")
        else:
            print(
                "\nERRO FATAL: Nenhuma chave API encontrada ou configurada corretamente."
            )
            print(
                "Verifique o seu ficheiro .env e as variáveis CLAUDE_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY."
            )
    except Exception as e:
        print("\nERRO INESPERADO DURANTE A EXECUÇÃO DO BENCHMARK:")
        print(f"Tipo: {type(e).__name__}")
        print(f"Erro: {e}")
        print("Traceback:")
        traceback.print_exc()

    print("=" * 59)
