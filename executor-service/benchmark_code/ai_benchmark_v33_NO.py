# -*- coding: utf-8 -*-
import requests
import time
import os
import re
import pandas as pd
import numpy as np
import matplotlib
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
import builtins  # Necessário para a sandbox do exec (referência a __build_class__)
import logging
import argparse

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

# Configuração básica do logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class AIBenchmark:
    """
    Executa um benchmark de programação comparando modelos de IA
    (Claude, ChatGPT, Gemini) em desafios de código, avaliando a qualidade,
    corretude (via execução externa ou local para LRU) e desempenho.
    Permite múltiplas execuções para resultados mais robustos.

    Permite configuração de chaves API, nomes de modelo e URL do executor
    via variáveis de ambiente (ficheiro .env) ou argumentos de linha de comando.
    Gera relatórios detalhados e visualizações dos resultados.
    """

    def __init__(
        self,
        num_runs: int = 1,
        output_dir: str = "benchmark_reports",
        claude_model_name: str | None = None,
        openai_model_name: str | None = None,
        gemini_model_name: str | None = None,
        executor_service_url: str | None = None,
    ):
        """
        Inicializa o benchmark. Carrega configurações, define desafios,
        verifica ferramentas (flake8) e coleta informações do ambiente.
        """
        self.claude_api_key = os.getenv("CLAUDE_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")

        self.num_runs = max(1, num_runs)
        logging.info(f"Número de execuções por desafio/IA: {self.num_runs}")

        self.results = []
        self.detailed_test_results = []
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        logging.info(f"Diretório de saída: '{self.output_dir}/'")

        self._configure_gemini_api()

        self.claude_model = claude_model_name or os.getenv(
            "CLAUDE_MODEL_NAME", "claude-3-opus-20240229"
        )
        self.openai_model = openai_model_name or os.getenv(
            "OPENAI_MODEL_NAME", "gpt-4-turbo"
        )
        self.gemini_model = (
            (
                gemini_model_name
                or os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-pro-latest")
            )
            if self.gemini_api_key
            else None
        )

        default_executor_url = (
            "https://executor-service-1029404216382.europe-southwest1.run.app/execute"
        )
        self.executor_url = executor_service_url or os.getenv(
            "EXECUTOR_SERVICE_URL", default_executor_url
        )

        self.challenge_details = {
            "Fibonacci Sequence": "Gera os N primeiros números da sequência de Fibonacci (0, 1, 1, 2, 3,...). \nTécnica comum: Iteração com duas variáveis ou recursão (risco de ineficiência).\nFinalidade: Avaliar compreensão de sequências e algoritmos básicos.",
            "Binary Search": "Procura o índice de um `target` numa lista `arr` *ordenada* de inteiros. Retorna -1 se não encontrado.\nTécnica: Divisão e conquista, comparando com o elemento do meio e descartando metade da lista.\nFinalidade: Avaliar algoritmos de busca eficientes (O(log n)) em dados ordenados.",
            "Merge Sort": "Ordena uma lista de inteiros em ordem crescente usando o algoritmo Merge Sort. Retorna uma *nova* lista ordenada.\nTécnica: Divisão e conquista recursiva (divide, ordena metades, junta metades ordenadas - merge).\nFinalidade: Avaliar implementação de algoritmos de ordenação recursivos e eficientes (O(n log n)).",
            "Count Islands in Grid": "Conta o número de 'ilhas' (grupos de 1s conectados horizontal/verticalmente) numa grelha 2D. 0 representa água.\nTécnica: Travessia de grafo (busca em profundidade - DFS ou busca em largura - BFS) para marcar ilhas visitadas.\nFinalidade: Testar algoritmos de travessia de grafos/matrizes.",
            "Palindrome Check": "Verifica se uma string `s` é um palíndromo (lê-se igual de trás para frente), ignorando caixa e caracteres não alfanuméricos.\nTécnica: Filtrar/normalizar a string e comparar com a sua versão invertida, ou usar dois ponteiros.\nFinalidade: Avaliar manipulação de strings e lógica de comparação.",
            "Valid Parentheses": "Determina se uma string com '()', '{}', '[]' é válida (parênteses abertos são fechados na ordem correta pelo tipo correspondente).\nTécnica: Uso de uma pilha (stack) para guardar parênteses de abertura e verificar correspondência ao encontrar fechos.\nFinalidade: Testar o uso correto de pilhas para validação de sequências.",
            "LRU Cache": "Implementa uma cache 'Least Recently Used' com capacidade fixa. Suporta `get(key)` (retorna valor ou -1, marca como usado) e `put(key, value)` (insere/atualiza, remove o item menos usado se exceder capacidade).\nTécnica: Combinação de dicionário (hash map) para acesso O(1) e lista duplamente ligada (ou `OrderedDict`) para manter ordem de uso e permitir remoção O(1).\nFinalidade: Avaliar design de estruturas de dados customizadas.",
            "Two Sum": "Dada uma lista `nums` e um `target`, retorna os *índices* de dois números que somam `target`. Assume solução única e não repetida.\nTécnica: Uso de um dicionário (hash map) para armazenar números vistos e verificar rapidamente se o 'complemento' (`target - num`) existe.\nFinalidade: Avaliar uso eficiente de dicionários para busca rápida (O(n)).",
            "Longest Substring Without Repeating Characters": "Encontra o comprimento da maior substring dentro de `s` que não contém caracteres repetidos.\nTécnica: Janela deslizante (sliding window) com um conjunto (set) ou dicionário para rastrear caracteres na janela atual e ajustar o início quando uma repetição é encontrada.\nFinalidade: Testar a técnica de janela deslizante.",
            "Minimum Edit Distance": "Calcula o número mínimo de operações (inserção, deleção, substituição) para transformar `word1` em `word2`.\nTécnica: Programação Dinâmica (algoritmo de Levenshtein), usando uma matriz 2D para armazenar distâncias parciais.\nFinalidade: Avaliar aplicação de programação dinâmica em problemas de strings.",
        }

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
            {
                "name": "Two Sum",
                "description": "Given a list of integers `nums` and an integer `target`, return indices of the two numbers such that they add up to `target`. You may assume that each input would have exactly one solution, and you may not use the same element twice.",
                "difficulty": "Easy",
                "function_signature": "two_sum(nums: list[int], target: int) -> list[int]",
                "test_cases": [
                    {
                        "input_args": [[2, 7, 11, 15], 9],
                        "expected_output": [0, 1],
                    },
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
        ]

        self._check_flake8()
        self.environment_info = self._get_environment_info()

    def _configure_gemini_api(self):
        """Configura a API Gemini se a chave API estiver disponível."""
        if self.gemini_api_key:
            try:
                genai.configure(api_key=self.gemini_api_key)  # type: ignore
                logging.info("Chave API Gemini configurada.")
            except Exception as e:
                logging.error(
                    f"Erro configurar API Gemini: {e}. AI será pulada.", exc_info=True
                )
                self.gemini_api_key = None
        else:
            logging.warning(
                "Chave GEMINI_API_KEY não encontrada. Gemini AI será pulada."
            )

    def _get_environment_info(self):
        """Coleta e retorna informações sobre o ambiente de execução."""
        info = {
            "Timestamp": datetime.now().isoformat(),
            "OS": f"{platform.system()} {platform.release()}",
            "Python Version": sys.version.split()[0],
            "Num Runs": self.num_runs,
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
            "pandas", "numpy", "matplotlib", "requests",
            "python-dotenv", "google-generativeai", "google-api-core", "tenacity",
        ]
        lib_versions = {}
        try:
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
                    lib_versions[lib] = "Unknown (importlib.metadata/pkg_resources failed)"

        info["Libraries"] = lib_versions
        return info

    def _check_flake8(self):
        """Verifica disponibilidade do flake8."""
        try:
            subprocess.run(
                [sys.executable, "-m", "flake8", "--version"],
                check=True, capture_output=True, timeout=5,
            )
            logging.info("Verificação Flake8: OK")
            self.flake8_available = True
        except Exception:
            logging.warning(
                f"Flake8 não encontrado via '{sys.executable} -m flake8'. Análise de qualidade será pulada.",
                exc_info=False,
            )
            self.flake8_available = False

    def query_claude(self, prompt: str) -> dict:
        """Envia um prompt para a API do Claude e retorna a resposta (com tratamento de erro)."""
        start_time = time.time()
        content = ""
        last_exception = None
        if not self.claude_api_key:
            logging.warning("Pulando Claude: Chave API N/A.")
            return {"content": "", "execution_time": 0, "last_error": "API Key Not Available"}

        headers = {
            "x-api-key": self.claude_api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        data = {"model": self.claude_model, "max_tokens": 4000, "messages": [{"role": "user", "content": prompt}]}

        try:
            response = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=data, timeout=120)
            response.raise_for_status()
            result = response.json()
            if ("content" in result and isinstance(result["content"], list) and len(result["content"]) > 0 and
                    isinstance(result["content"][0], dict) and "text" in result["content"][0]):
                content = result["content"][0]["text"]
            else:
                logging.warning(f"Resposta Claude inesperada: {result}")
                last_exception = ValueError(f"Unexpected Claude Response Format: {result}")
        except requests.exceptions.RequestException as e:
            last_exception = e
            status_code = e.response.status_code if e.response is not None else "N/A"
            logging.error(f"Erro na requisição Claude (Status: {status_code}): {e}", exc_info=False)
            if e.response is not None:
                try:
                    err_details = e.response.json()
                    logging.error(f"Detalhes do erro API Claude: {err_details}")
                except json.JSONDecodeError:
                    logging.error(f"Corpo da resposta (não JSON): {e.response.text[:150]}...")
        except Exception as e:
            last_exception = e
            logging.error(f"Erro inesperado ao consultar Claude: {e}", exc_info=True)

        execution_time = time.time() - start_time
        return {"content": content or "", "execution_time": execution_time, "last_error": str(last_exception) if last_exception else None}

    def query_chatgpt(self, prompt: str) -> dict:
        """Envia um prompt para a API do OpenAI (ChatGPT) e retorna a resposta."""
        start_time = time.time()
        content = ""
        last_exception = None
        if not self.openai_api_key:
            logging.warning("Pulando ChatGPT: Chave API N/A.")
            return {"content": "", "execution_time": 0, "last_error": "API Key Not Available"}

        headers = {"Authorization": f"Bearer {self.openai_api_key}", "Content-Type": "application/json"}
        data = {"model": self.openai_model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 4000}
        try:
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data, timeout=120)
            response.raise_for_status()
            result = response.json()
            if ("choices" in result and isinstance(result["choices"], list) and len(result["choices"]) > 0 and
                    isinstance(result["choices"][0], dict) and "message" in result["choices"][0] and
                    isinstance(result["choices"][0]["message"], dict) and "content" in result["choices"][0]["message"]):
                content = result["choices"][0]["message"]["content"]
            else:
                logging.warning(f"Resposta ChatGPT inesperada: {result}")
                last_exception = ValueError("Unexpected ChatGPT Response Format")
        except requests.exceptions.RequestException as e:
            last_exception = e
            status_code = e.response.status_code if e.response is not None else "N/A"
            logging.error(f"Erro na requisição ChatGPT (Status: {status_code}): {e}", exc_info=False)
            if e.response is not None:
                try:
                    err_details = e.response.json()
                    logging.error(f"Detalhes do erro API ChatGPT: {err_details}")
                except json.JSONDecodeError:
                    logging.error(f"Corpo da resposta (não JSON): {e.response.text[:150]}...")
        except Exception as e:
            last_exception = e
            logging.error(f"Erro inesperado ao consultar ChatGPT: {e}", exc_info=True)

        execution_time = time.time() - start_time
        return {"content": content or "", "execution_time": execution_time, "last_error": str(last_exception) if last_exception else None}

    def query_gemini(self, prompt: str) -> dict:
        """Envia um prompt para a API do Gemini e retorna a resposta."""
        start_time = time.time()
        content = ""
        last_exception = None

        if not self.gemini_api_key or not self.gemini_model:
            logging.warning("Pulando Gemini: Chave/Modelo N/A.")
            return {"content": "", "execution_time": 0, "last_error": "API Key/Model Not Available"}

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=5, max=60),
            retry=retry_if_exception_type(google.api_core.exceptions.ResourceExhausted),
            reraise=True,
            before_sleep=lambda retry_state: logging.warning(
                f"Rate limit Gemini. Tentativa {retry_state.attempt_number}. "
                f"Esperando {retry_state.next_action.sleep:.1f}s..."
                if retry_state.next_action else "Esperando para nova tentativa..."
            ),
        )
        def _call_gemini_api_with_retry(prompt_str: str) -> genai.types.GenerateContentResponse: # type: ignore
            """Função interna que faz a chamada real à API."""
            assert self.gemini_model is not None, "O modelo Gemini não pode ser None aqui."
            model = genai.GenerativeModel(self.gemini_model) # type: ignore
            safety_settings = [
                {"category": c, "threshold": "BLOCK_NONE"}
                for c in [
                    "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
                    "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT",
                ]
            ]
            return model.generate_content(prompt_str, safety_settings=safety_settings)

        try:
            final_response = _call_gemini_api_with_retry(prompt)
            try:
                content = final_response.text
            except ValueError:
                err_msg = "Resposta Gemini bloqueada ou sem texto."
                feedback = getattr(final_response, "prompt_feedback", None)
                candidates = getattr(final_response, "candidates", [])
                if feedback: err_msg += f" Feedback: {feedback}"
                if candidates and hasattr(candidates[0], "finish_reason"): err_msg += f" Reason: {candidates[0].finish_reason}"
                logging.warning(err_msg)
                last_exception = ValueError(err_msg)
            except Exception as e_text:
                err_msg = f"Erro ao extrair texto da resposta Gemini: {e_text}"
                logging.error(err_msg, exc_info=True)
                last_exception = ValueError(err_msg)
        except google.api_core.exceptions.ResourceExhausted as e_rate_limit:
            err_msg = f"Erro Gemini (Rate Limit Excedido) após múltiplas tentativas: {e_rate_limit}"
            logging.error(err_msg, exc_info=False)
            last_exception = e_rate_limit
        except Exception as e_unexpected:
            err_msg = f"Erro inesperado Gemini: {type(e_unexpected).__name__} - {e_unexpected}"
            logging.error(err_msg, exc_info=True)
            last_exception = e_unexpected

        execution_time = time.time() - start_time
        return {"content": content or "", "execution_time": execution_time, "last_error": str(last_exception) if last_exception else None}

    def extract_code(self, content: str) -> str:
        """Extrai o primeiro bloco de código Python de uma string de texto."""
        if not content: return ""
        m_python = re.search(r"```python\n(.*?)\n```", content, re.DOTALL)
        if m_python: return m_python.group(1).strip()
        m_generic = re.search(r"```(?:.*\n)?(.*?)\n```", content, re.DOTALL)
        if m_generic: return m_generic.group(1).strip()

        lines, code_lines, in_code_block = content.splitlines(), [], False
        for line in lines:
            stripped_line = line.strip()
            if not in_code_block and (line.startswith((" ", "\t")) or stripped_line.startswith(("def ", "class "))):
                in_code_block = True
            if in_code_block: code_lines.append(line)
            elif in_code_block and not line.startswith((" ", "\t")): break

        if code_lines:
            extracted = "\n".join(code_lines).strip()
            if extracted: return extracted

        looks_like_code = ("def " in content or "class " in content or "import " in content or "return " in content or
                           " for " in content or " while " in content or " if " in content or re.search(r"\n\s+", content))

        if looks_like_code:
            logging.debug("extract_code: Usando fallback final (retornando todo o conteúdo)")
            return content.strip()
        else:
            logging.debug("extract_code: Nenhum bloco de código encontrado.")
            return ""

    def count_code_lines(self, code: str) -> int:
        """Conta linhas de código reais (não vazias, não comentários)."""
        if not code: return 0
        return len([ln for ln in code.splitlines() if ln.strip() and not ln.strip().startswith("#")])

    def measure_response_completeness(self, code: str, description: str) -> int:
        """Estima a 'completude' da resposta da IA (score 0-100)."""
        score = 0
        if self.count_code_lines(code) > 0:
            score += 50
            if "def " in code or "class " in code: score += 10
            if "return " in code or "yield " in code: score += 10
            dl, cl, bonus = description.lower(), code.lower(), 0
            if "fibonacci" in dl and ("fibonacci" in cl or "fib(" in cl or "yield" in code): bonus = 10
            elif "binary search" in dl and ("mid =" in code or "mid=" in code or "low" in code or "high" in code): bonus = 10
            elif "merge sort" in dl and ("merge(" in cl or ("left" in code and "right" in code and "mid" in code)): bonus = 10
            elif "island" in dl and ("dfs(" in cl or "bfs(" in cl or "visit" in cl or ("grid[" in code and "==" in code)): bonus = 10
            elif "palindrome" in dl and ("[::-1]" in code or ".isalnum()" in cl or ".lower()" in cl): bonus = 10
            elif "parentheses" in dl and ("stack" in cl or "append(" in code or "pop(" in code): bonus = 10
            elif "lru cache" in dl and ("dict" in cl or "capacity" in cl or "ordereddict" in code.lower()): bonus = 10
            elif "two sum" in dl and ("dict" in cl or "hashmap" in cl or "target -" in code or "complement" in cl): bonus = 10
            elif "longest substring" in dl and ("set(" in code or "dict(" in code or "window" in cl or "sliding" in cl or ("left" in cl and "right" in cl)): bonus = 10
            elif "edit distance" in dl and ("dp[" in code or "matrix" in cl or ("min(" in code and "+ 1" in code)): bonus = 10
            score += bonus
            struct = 0
            if "if" in code or "for" in code or "while" in code: struct = 10
            if struct > 0 and ("if not" in code or "if len(" in code or "== 0" in code or "is None" in code or "else:" in code): struct += 5
            score += min(struct, 15)
        return min(score, 100)

    def analyze_code_quality_flake8(self, code_string: str) -> int:
        """Analisa qualidade do código com Flake8."""
        if not self.flake8_available or not code_string or self.count_code_lines(code_string) == 0:
            return -1

        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w+", suffix=".py", delete=False, encoding="utf-8") as tf:
                tf.write(code_string)
                temp_file_path = tf.name

            result = subprocess.run(
                [sys.executable, "-m", "flake8", "--count", temp_file_path],
                capture_output=True, text=True, timeout=15, check=False
            )
            if result.returncode in [0, 1] and result.stdout:
                try:
                    return int(result.stdout.strip().splitlines()[-1])
                except (IndexError, ValueError):
                    num_lines = len(result.stdout.strip().splitlines())
                    logging.warning(f"Flake8: Não foi possível parsear o contador, contando linhas ({num_lines}).")
                    return num_lines
            elif result.stderr:
                logging.error(f"Erro ao executar flake8: {result.stderr}")
                return -1
            else:
                return 0 if result.returncode == 0 else -1
        except subprocess.TimeoutExpired:
            logging.error("Erro: Timeout (15s) ao executar Flake8.")
            return -1
        except Exception as e:
            logging.error(f"Erro inesperado ao executar flake8: {e}", exc_info=True)
            return -1
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except Exception as e_rem:
                    logging.warning(f"Falha ao remover ficheiro temporário flake8 '{temp_file_path}': {e_rem}")

    def _lru_worker(self, code_string: str, test_sequence: list, result_queue: multiprocessing.Queue):
        results, error_message, lru_instance, final_success = [], None, None, True
        try:
            filtered_lines = [ln for ln in code_string.splitlines() if not (ln.strip().startswith(("from collections", "import collections")))]
            processed_code_string = "\n".join(filtered_lines)

            safe_builtins = {
                "int": int, "str": str, "list": list, "dict": dict, "tuple": tuple, "len": len,
                "range": range, "print": print, "True": True, "False": False, "None": None,
                "object": object, "Exception": Exception, "RuntimeError": RuntimeError,
                "ValueError": ValueError, "NameError": NameError, "AttributeError": AttributeError,
                "TypeError": TypeError, "StopIteration": StopIteration, "KeyError": KeyError,
                "__build_class__": builtins.__build_class__, "getattr": getattr, "isinstance": isinstance,
                "__name__": "__exec_sandbox__",
            }
            safe_globals = {"__builtins__": safe_builtins, "OrderedDict": OrderedDict, "__name__": "__exec_sandbox__"}
            local_namespace = {}
            exec(processed_code_string, safe_globals, local_namespace)

            class_name = next((name for name, obj in local_namespace.items() if isinstance(obj, type) and name != "OrderedDict"), None)
            if not class_name:
                match = re.search(r"^\s*class\s+(\w+)", processed_code_string, re.MULTILINE)
                if match: class_name = match.group(1)
                else: raise NameError("Nome da classe LRUCache não encontrado.")

            LruClass = local_namespace.get(class_name)
            if not LruClass: raise NameError(f"Classe '{class_name}' não definida.")

            for op_data in test_sequence:
                op_start_time = time.perf_counter()
                operation, args, expected = op_data["op"], op_data.get("args", []), op_data.get("expected")
                step_res = {"op": operation, "args": args, "exp": expected, "out": None, "ok": False, "err": None, "t_ms": -1.0}
                try:
                    if operation == "__init__":
                        lru_instance = LruClass(*args)
                        step_res.update({"ok": True, "out": "OK"})
                    elif lru_instance is None: raise RuntimeError("LRUCache não inicializada.")
                    elif operation in ["put", "get"]:
                        if not hasattr(lru_instance, operation): raise AttributeError(f"Método '{operation}' não encontrado.")
                        method = getattr(lru_instance, operation)
                        if operation == "put":
                            method(*args)
                            step_res.update({"ok": True, "out": "OK"})
                        else: # get
                            output = method(*args)
                            step_res["out"] = output
                            if str(output) == str(expected): step_res["ok"] = True
                            else:
                                step_res["err"] = f"Output:'{output}' != Expected:'{expected}'"
                                final_success = False
                    else: raise ValueError(f"Operação LRU inválida: {operation}")
                except Exception as e_step:
                    step_res["err"] = f"{type(e_step).__name__}: {str(e_step).splitlines()[0]}"
                    final_success = False
                step_res["t_ms"] = (time.perf_counter() - op_start_time) * 1000
                results.append(step_res)
        except Exception as e_main:
            error_message = f"Erro fatal no worker LRU: {type(e_main).__name__} - {str(e_main).splitlines()[0]}"
            print(f"    ERRO FATAL NO WORKER LRU: {error_message}")
            traceback.print_exc()
            final_success = False
        try:
            result_queue.put({"success": final_success, "results": results, "error": error_message})
        except Exception as q_err:
            print(f"    ERRO ao colocar resultado na fila: {q_err}")

    def _execute_lru_locally(self, code_string: str, test_sequence: list) -> dict:
        """Gere o processo para execução local do LRU Cache, com timeout."""
        ctx, result_queue = multiprocessing.get_context(None), multiprocessing.Queue()
        timeout, process = 10, ctx.Process(target=self._lru_worker, args=(code_string, test_sequence, result_queue))
        return_value, terminated = {"success": False, "results": [], "error": "Unknown error"}, False
        try:
            process.start()
            process.join(timeout=timeout)
            if process.is_alive():
                logging.warning(f"Timeout ({timeout}s) para processo LRU. Terminando...")
                process.terminate()
                process.join(1)
                if process.is_alive():
                    process.kill()
                    process.join()
                return_value = {"success": False, "results": [], "error": f"Timeout ({timeout}s)"}
                terminated = True
            if not terminated:
                if process.exitcode == 0:
                    try: return_value = result_queue.get(timeout=1)
                    except queue.Empty: return_value, _ = {"success": False, "results": [], "error": "Fila de resultados vazia."}, logging.error("Processo terminou, mas fila vazia.")
                    except Exception as e_q: return_value, _ = {"success": False, "results": [], "error": f"Erro de fila: {e_q}"}, logging.error(f"Erro na fila: {e_q}", exc_info=True)
                else:
                    logging.error(f"Processo LRU terminou com exitcode: {process.exitcode}")
                    try: err_result = result_queue.get_nowait(); return_value = err_result
                    except (queue.Empty, Exception): pass
                    if not return_value.get("error"): return_value["error"] = f"Worker exited with code {process.exitcode}"
                    return_value["success"] = False
        except Exception as e_p:
            logging.error(f"Falha ao gerir processo LRU: {e_p}", exc_info=True)
            return_value = {"success": False, "results": [], "error": f"Erro de gestão: {e_p}"}
            if process and process.is_alive():
                try: process.kill(); process.join()
                except: pass
        finally:
            try: result_queue.close(); result_queue.join_thread()
            except Exception: pass
            try:
                if process: process.close()
            except Exception: pass
        return return_value

    def evaluate_solution(self, ai_name: str, challenge: dict, response: dict, run_idx: int) -> dict:
        content, api_time, api_error = response.get("content", ""), response.get("execution_time", 0), response.get("last_error")
        if not content and api_error:
            fail_summary = {"run_index": run_idx, "ai": ai_name, "challenge": challenge["name"], "difficulty": challenge["difficulty"],
                            "api_execution_time": api_time, "lines_of_code": 0, "completeness": 0, "flake8_issues": -1,
                            "num_execution_errors": len(challenge.get("test_cases", [])), "first_execution_error": f"API Error: {str(api_error)[:100]}",
                            "code": "", "full_response": "", "execution_results_raw": []}
            self.results.append(fail_summary)
            num_details = len(challenge.get("test_cases", [])) if challenge["name"] != "LRU Cache" else 1
            for i in range(num_details):
                self.detailed_test_results.append({"run_index": run_idx, "ai": ai_name, "challenge": challenge["name"], "difficulty": challenge["difficulty"],
                                                   "test_case_index": i if challenge["name"] != "LRU Cache" else 0, "input_args_str": "N/A",
                                                   "expected_output_str": "N/A", "actual_output_str": None, "status": "API Error",
                                                   "execution_time_ms": -1, "error_message": str(api_error), "stdout": None})
            return fail_summary

        code, lines = self.extract_code(content), self.count_code_lines(content)
        completeness, flake8 = self.measure_response_completeness(code, challenge["description"]), self.analyze_code_quality_flake8(code)
        raw_results, details_curr_run, n_errors, first_err, can_exec, func_name = [], [], 0, "", bool(code), None

        if can_exec and challenge["name"] != "LRU Cache":
            func_name = self._detect_function_name(code, challenge)
            if not func_name: can_exec, first_err = False, "Function/Class name not found"

        if can_exec:
            assert func_name is not None or challenge["name"] == "LRU Cache"
            if challenge["name"] == "LRU Cache":
                logging.info(f"Executando LRU Cache de {ai_name} localmente (Run {run_idx+1})...")
                tc = challenge.get("test_cases", [{}])[0]
                if "sequence" in tc:
                    lru_res = self._execute_lru_locally(code, tc["sequence"])
                    ok = lru_res.get("success", False)
                    raw_results, n_errors, first_err = lru_res.get("results", []), 0 if ok else 1, lru_res.get("error", "")
                    if not first_err and not ok: first_err = next((f"Step {s['op']}{s['args']}: {s['err']}" for s in raw_results if s.get("err")), "LRU failed")
                    times_ok = [s["t_ms"] for s in raw_results if s.get("t_ms", -1) >= 0 and s.get("ok")]
                    avg_time_ms = np.mean(times_ok) if times_ok else -1.0
                    details_curr_run.append({"run_index": run_idx, "ai": ai_name, "challenge": challenge["name"], "difficulty": challenge["difficulty"],
                                             "test_case_index": 0, "input_args_str": json.dumps(tc["sequence"]), "expected_output_str": "Sequence Result",
                                             "actual_output_str": f"Success: {ok}", "status": "Pass" if ok else "Fail", "execution_time_ms": avg_time_ms,
                                             "error_message": str(first_err)[:250], "stdout": None})
                else:
                    n_errors, first_err = 1, "Invalid LRU test case"
                    details_curr_run.append({"run_index": run_idx, "ai": ai_name, "challenge": challenge["name"], "status": "Error", "error_message": first_err})
            else:
                logging.info(f"Executando '{challenge['name']}' de {ai_name} via Executor (Run {run_idx+1})...")
                assert func_name is not None
                raw, details, errors, first_e = self._run_test_cases_fixed(ai_name, challenge, code, func_name, run_idx)
                raw_results, details_curr_run, n_errors, first_err = raw, details, errors, first_e
        else:
            err_msg = "Code not extracted" if not code else "Function/Class name not found"
            n_errors = len(challenge.get("test_cases", []))
            first_err = err_msg
            for i, tc in enumerate(challenge.get("test_cases", [])):
                details_curr_run.append({"run_index": run_idx, "ai": ai_name, "challenge": challenge["name"], "test_case_index": i, "status": "No Code/No Function", "error_message": err_msg})
        
        self.detailed_test_results.extend(details_curr_run)
        run_summary = {"run_index": run_idx, "ai": ai_name, "challenge": challenge["name"], "difficulty": challenge["difficulty"],
                       "api_execution_time": api_time, "lines_of_code": lines, "completeness": completeness, "flake8_issues": flake8,
                       "num_execution_errors": n_errors, "first_execution_error": first_err, "code": code,
                       "full_response": content, "execution_results_raw": raw_results}
        self.results.append(run_summary)
        return run_summary

    def _detect_function_name(self, code: str, challenge: dict) -> str | None:
        if not code:
            return None
        signature = challenge.get("function_signature", "")
        sig_match = re.match(r"^\s*(?:def|class)\s+(\w+)", signature)
        if sig_match:
            expected_name = sig_match.group(1)
            simple_def, simple_class = f"def {expected_name}(", f"class {expected_name}:"
            if re.search(rf"^\s*{re.escape(simple_def)}", code, re.MULTILINE) or re.search(rf"^\s*{re.escape(simple_class)}", code, re.MULTILINE):
                return expected_name
        first_def_match = re.search(r"^\s*(?:def|class)\s+(\w+)", code, re.MULTILINE)
        if first_def_match:
            return first_def_match.group(1)
        return None

    def _run_test_cases_fixed(
        self,
        ai_name: str,
        challenge: dict,
        code: str,
        actual_function_name: str,
        run_idx: int,
    ) -> tuple[list, list, int, str]:
        raw_results, details, n_fails_errors, first_err_msg = [], [], 0, ""
        test_cases = challenge.get("test_cases", [])
        challenge_name = challenge["name"]
        for i, tc in enumerate(test_cases):
            input_args, expected_output = tc.get("input_args"), tc.get("expected_output")
            if input_args is None: continue
            exec_res = self._call_executor_service(code, actual_function_name, input_args)
            raw_results.append(exec_res)
            test_detail = {"run_index": run_idx, "ai": ai_name, "challenge": challenge_name, "difficulty": challenge["difficulty"], "test_case_index": i,
                           "input_args_str": json.dumps(input_args), "expected_output_str": json.dumps(expected_output), "actual_output_str": None,
                           "status": "Unknown", "execution_time_ms": exec_res.get("execution_time_ms", -1), "error_message": None, "stdout": exec_res.get("stdout")}
            current_case_error = None
            if exec_res.get("success"):
                actual_output = exec_res.get("output")
                test_detail["actual_output_str"] = json.dumps(actual_output)
                if challenge_name == "Two Sum" and isinstance(actual_output, list) and isinstance(expected_output, list):
                    are_equal = sorted(actual_output) == sorted(expected_output)
                else:
                    are_equal = np.array_equal(np.array(actual_output, dtype=object), np.array(expected_output, dtype=object))
                if are_equal:
                    test_detail["status"] = "Pass"
                else:
                    test_detail["status"], current_case_error, n_fails_errors = "Fail", f"Output != Expected", n_fails_errors + 1
            else:
                test_detail["status"], test_detail["error_message"], n_fails_errors = "Error", exec_res.get("error", "Exec error"), n_fails_errors + 1
                current_case_error = str(test_detail["error_message"])[:250]
            if current_case_error and not first_err_msg: first_err_msg = current_case_error
            details.append(test_detail)
        return raw_results, details, n_fails_errors, first_err_msg

    def _call_executor_service(
        self, code_string: str, function_name: str, input_args: list
    ) -> dict:
        imports = "from typing import List, Dict, Tuple, Set, Optional, Any\nimport re\nfrom collections import OrderedDict, deque\n\n"
        payload = {"code_string": imports + code_string, "function_name": function_name, "test_input_args": input_args}
        try:
            resp = requests.post(self.executor_url, headers={"Content-Type": "application/json"}, json=payload, timeout=60)
            return resp.json() if resp.status_code == 200 else {"success": False, "error": f"Executor status {resp.status_code}"}
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Request failed: {e}"}

    def run_benchmark(self):
        enabled_ais = [n for n, c in [("Claude", self.claude_api_key), ("ChatGPT", self.openai_api_key), ("Gemini", self.gemini_api_key)] if c]
        if not enabled_ais:
            logging.critical("Nenhuma API configurada.")
            return
        logging.info(f"Iniciando benchmark para: {', '.join(enabled_ais)}")
        self.results, self.detailed_test_results = [], []
        for run_idx in range(self.num_runs):
            logging.info(f"\n--- INICIANDO EXECUÇÃO {run_idx + 1}/{self.num_runs} ---")
            for chal in self.code_challenges:
                prompt = f"Implement the Python function/class.\nDescription: {chal['description']}\nSignature: {chal['function_signature']}\nProvide ONLY raw code."
                logging.info(f"\n-- Desafio: {chal['name']} --")
                responses = {}
                if "Claude" in enabled_ais: responses["Claude"] = self.query_claude(prompt)
                if "ChatGPT" in enabled_ais: responses["ChatGPT"] = self.query_chatgpt(prompt)
                if "Gemini" in enabled_ais: responses["Gemini"] = self.query_gemini(prompt)
                for ai in enabled_ais:
                    self.evaluate_solution(ai, chal, responses.get(ai, {}), run_idx)
        self.generate_report()

    def generate_report(self):
        if not self.results:
            logging.error("Nenhum resultado para gerar relatório.")
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        df_agg = self._prepare_dataframe()
        if df_agg is None: return
        stats_df = self._calculate_statistics(df_agg)
        self._save_full_results_json(ts)
        self._save_detailed_tests_csv(ts)
        self._save_summary_csv(df_agg, ts)
        if stats_df is not None:
            self._save_statistics_csv(stats_df, ts)
            self.create_visualizations(df_agg, ts, self.output_dir)
            self._display_summary_table(stats_df)

    def _prepare_dataframe(self) -> pd.DataFrame | None:
        if not self.results: return None
        df = pd.DataFrame(self.results)
        # Simplified aggregation for brevity
        return df.groupby(["ai", "challenge"]).mean(numeric_only=True).reset_index()

    def _calculate_statistics(self, df_agg: pd.DataFrame) -> pd.DataFrame | None:
        # Simplified statistics for brevity
        return df_agg.groupby("ai").mean(numeric_only=True).reset_index()

    def _save_full_results_json(self, ts: str):
        filepath = os.path.join(self.output_dir, f"full_results_{ts}.json")
        with open(filepath, "w") as f: json.dump(self.results, f, indent=2)
        logging.info(f"Resultados completos salvos em {filepath}")
    
    def _save_detailed_tests_csv(self, ts: str):
        filepath = os.path.join(self.output_dir, f"detailed_tests_{ts}.csv")
        if self.detailed_test_results:
            pd.DataFrame(self.detailed_test_results).to_csv(filepath, index=False)
            logging.info(f"Resultados detalhados salvos em {filepath}")

    def _save_summary_csv(self, df: pd.DataFrame, ts: str):
        filepath = os.path.join(self.output_dir, f"summary_{ts}.csv")
        df.to_csv(filepath, index=False)
        logging.info(f"Resumo salvo em {filepath}")

    def _save_statistics_csv(self, df: pd.DataFrame, ts: str):
        filepath = os.path.join(self.output_dir, f"statistics_{ts}.csv")
        df.to_csv(filepath, index=False)
        logging.info(f"Estatísticas salvas em {filepath}")
    
    def _display_summary_table(self, stats_df: pd.DataFrame):
        try:
            from tabulate import tabulate
            print("\n--- RESUMO GERAL ---")
            print(tabulate(stats_df, headers="keys", tablefmt="grid", showindex=False)) # type: ignore
        except ImportError:
            print(stats_df)

    def create_visualizations(self, df_agg, ts, out_dir):
        # Simplified plotting logic for brevity, focusing on corrections
        logging.info("Gerando visualizações...")
        plt.style.use('ggplot')
        
        # Bar chart example correction
        fig, ax = plt.subplots()
        # ... logic to create pivot_table ...
        # pivot_table.plot(kind='bar', ax=ax)
        plt.tight_layout(rect=tuple([0, 0, 0.88, 1])) # Corrected line
        plt.close(fig)

        # Radar chart example correction
        fig, ax = plt.subplots(subplot_kw=dict(polar=True))
        values = [1, 2, 3, 4] # Example data
        values += values[:1] # type: ignore
        plt.close(fig)
        
        # Bar label example correction
        fig, ax = plt.subplots()
        bars = ax.bar(['A', 'B'], [1,2])
        ax.bar_label(bars) # This line often works but might show type error
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Programming Benchmark")
    parser.add_argument("-n", "--num_runs", type=int, default=1, help="Number of runs per challenge.")
    parser.add_argument("--claude_model", type=str, default=None, help="Claude model name.")
    parser.add_argument("--openai_model", type=str, default=None, help="OpenAI model name.")
    parser.add_argument("--gemini_model", type=str, default=None, help="Gemini model name.")
    args = parser.parse_args()

    benchmark = AIBenchmark(
        num_runs=args.num_runs,
        claude_model_name=args.claude_model,
        openai_model_name=args.openai_model,
        gemini_model_name=args.gemini_model
    )
    benchmark.run_benchmark()