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

# SUGESTÃO 1: Configuração básica do logging
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
        challenge_details (dict): Detalhes técnicos/propósito dos desafios.
    """

    # SUGESTÃO 2: Modificar __init__ para aceitar config de argparse
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

        self.results = []  # Acumulará resultados de TODAS as runs
        self.detailed_test_results = []  # Acumulará detalhes de TODAS as runs
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        logging.info(f"Diretório de saída: '{self.output_dir}/'")

        # --- Carrega Configurações (com fallbacks e prioridade para args) ---
        self._configure_gemini_api()  # <<< ESTA É A CHAMADA (linha ~111 no teu ficheiro anterior)

        # Nomes dos modelos via args > .env > fallback para defaults
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

        # URL do Executor via args > .env > fallback
        default_executor_url = (
            "https://executor-service-1029404216382.europe-southwest1.run.app/execute"
        )
        self.executor_url = executor_service_url or os.getenv(
            "EXECUTOR_SERVICE_URL", default_executor_url
        )

        # SUGESTÃO 5 + PEDIDO DO UTILIZADOR: Detalhes dos desafios elaborados
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
                "test_cases": [  # Apenas uma sequência de teste aqui
                    {
                        "description": "Simple put/get sequence",
                        "sequence": [
                            {"op": "__init__", "args": [2]},
                            {"op": "put", "args": [1, 1]},
                            {"op": "put", "args": [2, 2]},
                            {"op": "get", "args": [1], "expected": 1},
                            {"op": "put", "args": [3, 3]},  # Evicts key 2
                            {
                                "op": "get",
                                "args": [2],
                                "expected": -1,
                            },  # Key 2 should be gone
                            {"op": "put", "args": [4, 4]},  # Evicts key 1
                            {
                                "op": "get",
                                "args": [1],
                                "expected": -1,
                            },  # Key 1 should be gone
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
        ]  # Fim da lista self.code_challenges

        self._check_flake8()
        self.environment_info = self._get_environment_info()

    # --- GARANTE QUE ESTE MÉTODO ESTÁ DEFINIDO ---
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
                self.gemini_api_key = None  # Invalida a chave para não tentar usar
        else:
            logging.warning(
                "Chave GEMINI_API_KEY não encontrada. Gemini AI será pulada."
            )

    # --- FIM DA DEFINIÇÃO DO MÉTODO ---

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
            "tenacity",
        ]
        lib_versions = {}
        try:
            # Tenta Python 3.8+
            import importlib.metadata as importlib_metadata

            for lib in libs_to_check:
                try:
                    lib_versions[lib] = importlib_metadata.version(lib)
                except importlib_metadata.PackageNotFoundError:
                    lib_versions[lib] = "Not Found"
        except ImportError:
            # Fallback para pkg_resources
            if pkg_resources:
                for lib in libs_to_check:
                    try:
                        lib_versions[lib] = pkg_resources.get_distribution(lib).version
                    except pkg_resources.DistributionNotFound:
                        lib_versions[lib] = "Not Found"
                    except Exception as e:
                        lib_versions[lib] = f"Error ({type(e).__name__})"
            else:
                # Se nem pkg_resources estiver disponível
                for lib in libs_to_check:
                    lib_versions[lib] = (
                        "Unknown (importlib.metadata/pkg_resources failed)"
                    )

        info["Libraries"] = lib_versions
        return info

    def _check_flake8(self):
        """Verifica disponibilidade do flake8."""
        try:
            # Usa sys.executable para garantir que é o python correto
            subprocess.run(
                [sys.executable, "-m", "flake8", "--version"],
                check=True,
                capture_output=True,
                timeout=5,
            )
            logging.info("Verificação Flake8: OK")
            self.flake8_available = True
        except Exception:
            logging.warning(
                f"Flake8 não encontrado via '{sys.executable} -m flake8'. Análise de qualidade será pulada.",
                exc_info=False,  # Não precisa do traceback completo aqui
            )
            self.flake8_available = False

    # Funções query_... com logging
    def query_claude(self, prompt: str) -> dict:
        """Envia um prompt para a API do Claude e retorna a resposta (com tratamento de erro)."""
        start_time = time.time()
        content = ""
        execution_time = 0
        last_exception = None
        if not self.claude_api_key:
            logging.warning("Pulando Claude: Chave API N/A.")
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
            response.raise_for_status()  # Levanta exceção para erros HTTP 4xx/5xx
            result = response.json()
            # Extração segura do conteúdo
            if (
                "content" in result
                and isinstance(result["content"], list)
                and len(result["content"]) > 0
                and isinstance(result["content"][0], dict)
                and "text" in result["content"][0]
            ):
                content = result["content"][0]["text"]
            else:
                logging.warning(f"Resposta Claude inesperada: {result}")
                last_exception = ValueError(
                    f"Unexpected Claude Response Format: {result}"
                )

        except requests.exceptions.RequestException as e:
            last_exception = e
            status_code = e.response.status_code if e.response is not None else "N/A"
            logging.error(
                f"Erro na requisição Claude (Status: {status_code}): {e}",
                exc_info=False,
            )
            # Tenta obter mais detalhes do erro da API se disponíveis
            if e.response is not None:
                try:
                    err_details = e.response.json()
                    logging.error(f"Detalhes do erro API Claude: {err_details}")
                except json.JSONDecodeError:
                    logging.error(
                        f"Corpo da resposta (não JSON): {e.response.text[:150]}..."
                    )

        except Exception as e:  # Captura outros erros inesperados
            last_exception = e
            logging.error(f"Erro inesperado ao consultar Claude: {e}", exc_info=True)

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
            logging.warning("Pulando ChatGPT: Chave API N/A.")
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
            # Extração segura
            if (
                "choices" in result
                and isinstance(result["choices"], list)
                and len(result["choices"]) > 0
                and isinstance(result["choices"][0], dict)
                and "message" in result["choices"][0]
                and isinstance(result["choices"][0]["message"], dict)
                and "content" in result["choices"][0]["message"]
            ):
                content = result["choices"][0]["message"]["content"]
            else:
                logging.warning(f"Resposta ChatGPT inesperada: {result}")
                last_exception = ValueError("Unexpected ChatGPT Response Format")

        except requests.exceptions.RequestException as e:
            last_exception = e
            status_code = e.response.status_code if e.response is not None else "N/A"
            logging.error(
                f"Erro na requisição ChatGPT (Status: {status_code}): {e}",
                exc_info=False,
            )
            if e.response is not None:
                try:
                    err_details = e.response.json()
                    logging.error(f"Detalhes do erro API ChatGPT: {err_details}")
                except json.JSONDecodeError:
                    logging.error(
                        f"Corpo da resposta (não JSON): {e.response.text[:150]}..."
                    )

        except Exception as e:
            last_exception = e
            logging.error(f"Erro inesperado ao consultar ChatGPT: {e}", exc_info=True)

        execution_time = time.time() - start_time
        return {
            "content": content or "",
            "execution_time": execution_time,
            "last_error": str(last_exception) if last_exception else None,
        }

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
            logging.warning("Pulando Gemini: Chave/Modelo N/A.")
            return {
                "content": "",
                "execution_time": 0,
                "last_error": "API Key/Model Not Available",
            }

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=5, max=60),
            retry=retry_if_exception_type(google.api_core.exceptions.ResourceExhausted),
            reraise=True,  # Re-levanta a exceção original se todas as tentativas falharem
            before_sleep=lambda retry_state: logging.warning(
                f"Rate limit Gemini. Tentativa {retry_state.attempt_number}. "
                f"Esperando {retry_state.next_action.sleep:.1f}s..."
                if retry_state.next_action else "Esperando para nova tentativa..."
            ),
        )
        def _call_gemini_api_with_retry(
            prompt_str: str,
        ) -> genai.types.GenerateContentResponse:  # type: ignore
            """Função interna que faz a chamada real à API."""
            assert self.gemini_model is not None, "O modelo Gemini não pode ser None aqui."
            model = genai.GenerativeModel(self.gemini_model)  # type: ignore
            safety_settings = [
                {"category": c, "threshold": "BLOCK_NONE"}
                for c in [
                    "HARM_CATEGORY_HARASSMENT",
                    "HARM_CATEGORY_HATE_SPEECH",
                    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "HARM_CATEGORY_DANGEROUS_CONTENT",
                ]
            ]
            # Log removido daqui para evitar AttributeError
            response = model.generate_content(
                prompt_str, safety_settings=safety_settings
            )
            return response

        try:
            final_response = _call_gemini_api_with_retry(prompt)
            try:
                content = final_response.text
            except ValueError:
                # Lida com resposta bloqueada ou sem texto
                err_msg = "Resposta Gemini bloqueada ou sem texto."
                feedback = getattr(final_response, "prompt_feedback", None)
                candidates = getattr(final_response, "candidates", [])
                if feedback:
                    err_msg += f" Feedback: {feedback}"
                if candidates and hasattr(candidates[0], "finish_reason"):
                    err_msg += f" Reason: {candidates[0].finish_reason}"
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
        return {
            "content": content or "",
            "execution_time": execution_time,
            "last_error": str(last_exception) if last_exception else None,
        }

    # --- FIM query_gemini CORRIGIDA ---

    # SUGESTÃO 4: Melhorar fallback do extract_code
    def extract_code(self, content: str) -> str:
        """Extrai o primeiro bloco de código Python de uma string de texto."""
        if not content:
            return ""

        # 1. Tenta ```python ... ```
        m_python = re.search(r"```python\n(.*?)\n```", content, re.DOTALL)
        if m_python:
            return m_python.group(1).strip()

        # 2. Tenta ``` ... ``` (genérico)
        m_generic = re.search(r"```(?:.*\n)?(.*?)\n```", content, re.DOTALL)
        if m_generic:
            return m_generic.group(1).strip()

        # 3. Tenta heurística baseada em indentação e keywords
        lines = content.splitlines()
        code_lines = []
        in_code_block = False
        for line in lines:
            stripped_line = line.strip()
            # Começa se indentado ou começa com def/class
            if not in_code_block and (
                line.startswith((" ", "\t"))
                or stripped_line.startswith(("def ", "class "))
            ):
                in_code_block = True
            # Adiciona linha se estiver no bloco
            if in_code_block:
                code_lines.append(line)
            # Para se encontrar linha não indentada após iniciar (simples)
            elif in_code_block and not line.startswith((" ", "\t")):
                break  # Pode ser muito simples, mas evita pegar texto explicativo depois

        if code_lines:
            extracted = "\n".join(code_lines).strip()
            if extracted:  # Se encontrou algo com a heurística
                return extracted

        # 4. Fallback final: Verifica se o conteúdo TODO parece código
        #    antes de retorná-lo por inteiro.
        looks_like_code = (
            "def " in content
            or "class " in content
            or "import " in content
            or "return " in content
            or " for " in content  # Espaços para evitar match em palavras
            or " while " in content
            or " if " in content
            or re.search(r"\n\s+", content)  # Contém indentação significativa
        )

        if looks_like_code:
            logging.debug(
                "extract_code: Usando fallback final (retornando todo o conteúdo)"
            )
            return content.strip()
        else:
            logging.debug(
                "extract_code: Nenhum bloco de código encontrado, e conteúdo não parece código."
            )
            return ""  # Retorna vazio se não parece código

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

    def measure_response_completeness(self, code: str, description: str) -> int:
        """Estima a 'completude' da resposta da IA (score 0-100)."""
        score = 0
        lines = self.count_code_lines(code)
        if lines > 0:
            score += 50  # Base score for having any code
            # Bonus for structure
            if "def " in code or "class " in code:
                score += 10
            if "return " in code or "yield " in code:
                score += 10  # Functionality indicators
            # Bonus for keywords related to challenge
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
                or ("left" in cl and "right" in cl)
            ):
                bonus = 10
            elif "edit distance" in dl and (
                "dp[" in code or "matrix" in cl or ("min(" in code and "+ 1" in code)
            ):
                bonus = 10
            score += bonus
            # Bonus for basic control flow / edge cases
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

    def analyze_code_quality_flake8(self, code_string: str) -> int:
        """Analisa qualidade do código com Flake8."""
        if (
            not self.flake8_available
            or not code_string
            or self.count_code_lines(code_string) == 0
        ):
            return -1  # -1 indica N/A ou sem código

        temp_file_path = None
        try:
            # Cria ficheiro temporário seguro
            with tempfile.NamedTemporaryFile(
                mode="w+", suffix=".py", delete=False, encoding="utf-8"
            ) as tf:
                tf.write(code_string)
                temp_file_path = tf.name

            # Executa flake8 no ficheiro temporário
            executable = sys.executable
            result = subprocess.run(
                [executable, "-m", "flake8", "--count", temp_file_path],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,  # Não levanta exceção por return code != 0
            )

            # Processa a saída
            if (
                result.returncode in [0, 1] and result.stdout
            ):  # 0 = OK, 1 = issues found
                try:
                    # A última linha do stdout com --count é o número total de erros
                    return int(result.stdout.strip().splitlines()[-1])
                except (IndexError, ValueError):
                    # Fallback: conta linhas se não conseguir parsear o número
                    num_lines = len(result.stdout.strip().splitlines())
                    logging.warning(
                        f"Flake8: Não foi possível parsear o contador, contando linhas ({num_lines}). Saída: {result.stdout}"
                    )
                    return num_lines
            elif result.stderr:
                logging.error(f"Erro ao executar flake8: {result.stderr}")
                return -1  # Indica erro na execução
            else:
                # Se return code foi 0 e stdout vazio, significa 0 issues.
                # Se foi > 1, é erro do flake8.
                if result.returncode == 0:
                    return 0
                else:
                    logging.error(
                        f"Flake8 retornou código {result.returncode} inesperado sem stdout/stderr."
                    )
                    return -1  # Erro inesperado

        except subprocess.TimeoutExpired:
            logging.error("Erro: Timeout (15s) ao executar Flake8.")
            return -1
        except Exception as e:
            logging.error(f"Erro inesperado ao executar flake8: {e}", exc_info=True)
            return -1
        finally:
            # Garante limpeza do ficheiro temporário
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except Exception as e_rem:
                    logging.warning(
                        f"Falha ao remover ficheiro temporário flake8 '{temp_file_path}': {e_rem}"
                    )

    # --- Métodos Execução Local LRU Cache (com logging) ---
    def _lru_worker(
        self, code_string: str, test_sequence: list, result_queue: multiprocessing.Queue
    ):
        # (Código _lru_worker como estava antes, mas trocando prints por logging se apropriado)
        results = []
        error_message = None
        lru_instance = None
        final_success = True
        try:
            # Filtra imports específicos para evitar conflitos na sandbox
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

            # Define ambiente seguro para exec
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
                "__build_class__": builtins.__build_class__,  # Necessário para definir classes
                "getattr": getattr,
                "isinstance": isinstance,
                "__name__": "__exec_sandbox__",
            }
            # Permite OrderedDict no global, mas não outros imports perigosos
            safe_globals = {
                "__builtins__": safe_builtins,
                "OrderedDict": OrderedDict,
                "__name__": "__exec_sandbox__",
            }
            local_namespace = {}

            # Executa o código da IA num namespace controlado
            exec(processed_code_string, safe_globals, local_namespace)

            # Encontra a classe LRUCache definida no código da IA
            class_name = None
            for name, obj in local_namespace.items():
                if (
                    isinstance(obj, type) and name != "OrderedDict"
                ):  # Assume a primeira classe encontrada
                    class_name = name
                    break
            if (
                not class_name
            ):  # Tenta encontrar via regex se não estiver no namespace direto
                match = re.search(
                    r"^\s*class\s+(\w+)", processed_code_string, re.MULTILINE
                )
                if match:
                    class_name = match.group(1)
                else:
                    raise NameError(
                        "Nome da classe LRUCache não encontrado no código processado."
                    )

            LruClass = local_namespace.get(class_name)
            if not LruClass:
                raise NameError(
                    f"Classe '{class_name}' não definida no namespace local."
                )

            op_total_time = 0.0
            # Executa a sequência de operações de teste
            for op_data in test_sequence:
                op_start_time = time.perf_counter()
                operation = op_data["op"]
                args = op_data.get("args", [])
                expected = op_data.get("expected")  # Pode ser None para 'put'
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
                            step_res["ok"] = True  # Assume OK se não lançar exceção
                            step_res["out"] = "OK"
                        elif operation == "get":
                            output = method(*args)
                            step_res["out"] = output
                            # Compara saída com esperado (tratando tipos)
                            if str(output) == str(
                                expected
                            ):  # Comparação simples como string
                                step_res["ok"] = True
                            else:
                                step_res["err"] = (
                                    f"Output:'{output}' != Expected:'{expected}'"
                                )
                                final_success = (
                                    False  # Marca falha geral se um step falhar
                                )
                    else:
                        raise ValueError(
                            f"Operação LRU inválida na sequência de teste: {operation}"
                        )
                except Exception as e_step:
                    step_res["err"] = (
                        f"{type(e_step).__name__}: {str(e_step).splitlines()[0]}"  # Mensagem curta
                    )
                    final_success = False  # Marca falha geral
                    # logging.debug(f"Erro no step LRU: {step_res['err']}") # Log opcional para debug

                op_end_time = time.perf_counter()
                step_res["t_ms"] = (op_end_time - op_start_time) * 1000
                op_total_time += op_end_time - op_start_time
                results.append(step_res)

        except Exception as e_main:
            error_message = f"Erro fatal no worker LRU: {type(e_main).__name__} - {str(e_main).splitlines()[0]}"
            # Usamos print aqui porque logging pode não estar configurado no worker/subprocesso
            print(f"    ERRO FATAL NO WORKER LRU: {error_message}")
            traceback.print_exc()  # Print traceback no worker para debug
            final_success = False

        try:
            # Envia o resultado de volta para o processo principal
            result_queue.put(
                {"success": final_success, "results": results, "error": error_message}
            )
        except Exception as q_err:
            print(f"    ERRO ao colocar resultado na fila do worker LRU: {q_err}")

    def _execute_lru_locally(self, code_string: str, test_sequence: list) -> dict:
        """Gere o processo para execução local do LRU Cache, com timeout."""
        ctx = multiprocessing.get_context(None)  # Usa contexto padrão
        result_queue = ctx.Queue()
        timeout_seconds = 10  # Timeout para a execução do worker
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
            process.start()
            process.join(timeout=timeout_seconds)  # Espera com timeout

            if process.is_alive():  # Se o processo ainda está a correr -> Timeout
                logging.warning(
                    f"Timeout ({timeout_seconds}s) atingido para processo LRU. Terminando..."
                )
                process.terminate()  # Tenta terminar
                process.join(1)  # Espera um pouco
                if process.is_alive():  # Se ainda vivo, força kill
                    logging.warning("Terminate falhou, forçando kill processo LRU.")
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
                if exit_code == 0:  # Processo terminou normalmente
                    try:
                        result = result_queue.get(
                            timeout=1
                        )  # Tenta obter resultado da fila
                        return_value = result
                    except queue.Empty:
                        logging.error(
                            "Processo LRU terminou OK, mas fila de resultados vazia."
                        )
                        return_value = {
                            "success": False,
                            "results": [],
                            "error": "Worker finished OK but result queue empty",
                        }
                    except Exception as e_q:
                        logging.error(
                            f"Falha obter resultado da fila LRU: {e_q}", exc_info=True
                        )
                        return_value = {
                            "success": False,
                            "results": [],
                            "error": f"Queue communication error: {e_q}",
                        }
                else:  # Processo terminou com erro
                    logging.error(
                        f"Processo worker LRU terminou inesperadamente com exitcode: {exit_code}"
                    )
                    # Tenta obter mensagem de erro da fila, se o worker conseguiu enviar
                    try:
                        err_result = result_queue.get_nowait()
                        return_value = err_result  # Usa o erro reportado pelo worker
                    except (queue.Empty, Exception):
                        pass  # Sem mensagem na fila ou erro ao ler a fila
                    # Garante que o erro reflete a terminação anormal
                    if not return_value.get("error"):
                        return_value["error"] = (
                            f"Worker process exited with code {exit_code}"
                        )
                    return_value["success"] = False

        except Exception as e_p:
            logging.error(f"Falha ao gerir processo worker LRU: {e_p}", exc_info=True)
            return_value = {
                "success": False,
                "results": [],
                "error": f"Process management error: {e_p}",
            }
            # Tenta matar o processo se ainda estiver vivo
            if process and process.is_alive():
                try:
                    process.kill()
                    process.join()
                except Exception:
                    pass
        finally:
            # Limpa recursos (fila e processo)
            try:
                result_queue.close()
                result_queue.join_thread()
            except Exception as e_qc:
                logging.debug(f"Erro ao limpar fila LRU: {e_qc}")
            try:
                if process:
                    process.close()
            except Exception as e_pc:
                logging.debug(f"Erro ao limpar objeto Process LRU: {e_pc}")

        return return_value

    # --- Fim métodos execução local ---

    def evaluate_solution(
        self, ai_name: str, challenge: dict, response: dict, run_idx: int
    ) -> dict:
        """
        Avalia a solução de uma IA para um desafio E UMA EXECUÇÃO ESPECÍFICA.
        """
        content = response.get("content", "")
        api_time = response.get("execution_time", 0)
        api_error = response.get("last_error")

        if not content and api_error:
            logging.warning(
                f"Sem conteúdo de {ai_name} para {challenge['name']} (Run {run_idx+1}) devido a erro API: {api_error}. Avaliação pulada."
            )
            fail_result_summary = {
                "run_index": run_idx, "ai": ai_name, "challenge": challenge["name"],
                "difficulty": challenge["difficulty"], "api_execution_time": api_time,
                "lines_of_code": 0, "completeness": 0, "flake8_issues": -1,
                "num_execution_errors": len(challenge.get("test_cases", [])),
                "first_execution_error": f"API Error: {str(api_error)[:100]}",
                "code": "", "full_response": "", "execution_results_raw": [],
            }
            self.results.append(fail_result_summary)

            num_detail_entries = len(challenge.get("test_cases", [])) if challenge["name"] != "LRU Cache" else 1
            for i in range(num_detail_entries):
                self.detailed_test_results.append({
                    "run_index": run_idx, "ai": ai_name, "challenge": challenge["name"],
                    "difficulty": challenge["difficulty"],
                    "test_case_index": i if challenge["name"] != "LRU Cache" else 0,
                    "input_args_str": "N/A", "expected_output_str": "N/A",
                    "actual_output_str": None, "status": "API Error",
                    "execution_time_ms": -1, "error_message": str(api_error), "stdout": None,
                })
            return fail_result_summary

        code = self.extract_code(content)
        lines = self.count_code_lines(code)
        completeness = self.measure_response_completeness(code, challenge["description"])
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
                first_err = "Function/Class name not found"

        if can_exec:
            if challenge["name"] == "LRU Cache":
                logging.info(f"Executando LRU Cache de {ai_name} localmente (Run {run_idx+1})...")
                tc_list = challenge.get("test_cases", [])
                if tc_list and isinstance(tc_list[0], dict) and "sequence" in tc_list[0]:
                    lru_res = self._execute_lru_locally(code, tc_list[0]["sequence"])
                    raw_results, ok = lru_res.get("results", []), lru_res.get("success", False)
                    n_errors, first_err = (0 if ok else 1), lru_res.get("error", "")
                    if not first_err and not ok:
                        first_err = next((f"Step {s['op']}{s['args']}: {s['err']}" for s in raw_results if s.get("err")), "LRU failed")
                    times_ok = [s["t_ms"] for s in raw_results if s.get("t_ms", -1) >= 0 and s.get("ok")]
                    avg_time_ms = np.mean(times_ok) if times_ok else -1.0
                    details_curr_run.append({
                        "run_index": run_idx, "ai": ai_name, "challenge": challenge["name"],
                        "difficulty": challenge["difficulty"], "test_case_index": 0,
                        "input_args_str": json.dumps(tc_list[0]["sequence"]),
                        "expected_output_str": "Sequence Result", "actual_output_str": f"Success: {ok}",
                        "status": "Pass" if ok else ("Fail" if raw_results else "Error"),
                        "execution_time_ms": avg_time_ms, "error_message": str(first_err)[:250], "stdout": None,
                    })
                    logging.info(f"Resultado LRU local de {ai_name} (Run {run_idx+1}): {'Sucesso' if ok else 'Falha/Erro'}")
                else:
                    n_errors, first_err = 1, "Invalid LRU test case format."
                    details_curr_run.append({
                        "run_index": run_idx, "ai": ai_name, "challenge": challenge["name"], "status": "Error",
                        "error_message": first_err, "execution_time_ms": -1
                    })
            else:
                logging.info(f"Executando '{challenge['name']}' de {ai_name} via Executor Service (Run {run_idx+1})...")
                assert func_name is not None, "func_name não deveria ser None aqui se pode executar."
                raw, details, errors, first_e = self._run_test_cases_fixed(ai_name, challenge, code, func_name, run_idx)
                raw_results, n_errors, first_err, details_curr_run = raw, errors, first_e, details
        else:
            err_msg = "Code not extracted" if not code else "Function/Class name not found"
            logging.warning(f"{err_msg} para '{challenge['name']}' por {ai_name} (Run {run_idx+1}). Pulando execução.")
            n_errors, first_err = len(challenge.get("test_cases", [])), err_msg
            for i, tc in enumerate(challenge.get("test_cases", [])):
                details_curr_run.append({
                    "run_index": run_idx, "ai": ai_name, "challenge": challenge["name"], "difficulty": challenge["difficulty"],
                    "test_case_index": i, "input_args_str": json.dumps(tc.get("input_args", "N/A")),
                    "expected_output_str": json.dumps(tc.get("expected_output", "N/A")),
                    "status": "No Code/No Function", "error_message": err_msg,
                })

        self.detailed_test_results.extend(details_curr_run)
        run_summary = {
            "run_index": run_idx, "ai": ai_name, "challenge": challenge["name"], "difficulty": challenge["difficulty"],
            "api_execution_time": api_time, "lines_of_code": lines, "completeness": completeness,
            "flake8_issues": flake8, "num_execution_errors": n_errors, "first_execution_error": first_err,
            "code": code, "full_response": content, "execution_results_raw": raw_results,
        }
        self.results.append(run_summary)
        return run_summary

    def _detect_function_name(self, code: str, challenge: dict) -> str | None:
        """Detecta nome da função/classe, priorizando assinatura e nome esperado."""
        if not code: return None
        expected_name = None
        signature = challenge.get("function_signature", "")
        sig_match = re.match(r"^\s*(?:def|class)\s+(\w+)", signature)
        if sig_match:
            expected_name = sig_match.group(1)
            simple_def, simple_class, simple_class_paren = f"def {expected_name}(", f"class {expected_name}:", f"class {expected_name}("
            if (re.search(rf"^\s*{re.escape(simple_def)}", code, re.MULTILINE) or
                re.search(rf"^\s*{re.escape(simple_class)}", code, re.MULTILINE) or
                re.search(rf"^\s*{re.escape(simple_class_paren)}", code, re.MULTILINE)):
                return expected_name
            elif re.search(rf"\s(?:def|class)\s+{re.escape(expected_name)}\b", code, re.MULTILINE):
                logging.debug(f"Encontrado '{expected_name}' como def/class não no início da linha.")
                return expected_name
            elif f"{expected_name}(" in code or f"{expected_name} " in code:
                logging.warning(f"Nome esperado '{expected_name}' encontrado, mas não como definição clara. Usando '{expected_name}'.")
                return expected_name
        
        first_def_match = re.search(r"^\s*(?:def|class)\s+(\w+)", code, re.MULTILINE)
        if first_def_match:
            found_name = first_def_match.group(1)
            if expected_name and found_name != expected_name:
                logging.warning(f"Usando 1ª definição encontrada '{found_name}' (Esperado: '{expected_name}').")
            elif not expected_name:
                logging.warning(f"Usando 1ª definição encontrada '{found_name}' (sem nome esperado definido).")
            return found_name
        
        logging.error(f"Nenhuma definição de função ou classe encontrada para {challenge['name']}.")
        return None

    def _run_test_cases_fixed(
        self,
        ai_name: str,
        challenge: dict,
        code: str,
        actual_function_name: str,
        run_idx: int,
    ) -> tuple[list, list, int, str]:
        """Executa casos de teste via Executor Service, incluindo run_idx nos detalhes."""
        raw_results, details, n_fails_errors, first_err_msg = [], [], 0, ""
        test_cases = challenge.get("test_cases", [])

        if not test_cases:
            logging.warning(f"Sem casos de teste para '{challenge['name']}'.")
            return [], [], 0, ""

        challenge_name = challenge["name"]

        for i, tc in enumerate(test_cases):
            input_args, expected_output = tc.get("input_args"), tc.get("expected_output")
            if input_args is None: continue

            exec_res = self._call_executor_service(code, actual_function_name, input_args)
            raw_results.append(exec_res)

            test_detail = {
                "run_index": run_idx, "ai": ai_name, "challenge": challenge_name, "difficulty": challenge["difficulty"],
                "test_case_index": i, "input_args_str": json.dumps(input_args), "expected_output_str": json.dumps(expected_output),
                "actual_output_str": None, "status": "Unknown", "execution_time_ms": exec_res.get("execution_time_ms", -1),
                "error_message": None, "stdout": exec_res.get("stdout"),
            }
            current_case_error = None

            if exec_res.get("success"):
                actual_output = exec_res.get("output")
                test_detail["actual_output_str"] = json.dumps(actual_output)
                try:
                    if (challenge_name == "Two Sum" and isinstance(actual_output, list) and isinstance(expected_output, list)):
                        are_equal = sorted(actual_output) == sorted(expected_output)
                    else:
                        are_equal = np.array_equal(np.array(actual_output, dtype=object), np.array(expected_output, dtype=object))
                    if are_equal:
                        test_detail["status"] = "Pass"
                    else:
                        test_detail["status"], current_case_error, n_fails_errors = "Fail", f"Output != Expected (Case {i})", n_fails_errors + 1
                except Exception as comp_e:
                    test_detail["status"], test_detail["error_message"], current_case_error, n_fails_errors = "Fail (Comparison Error)", f"Error comparing: {comp_e}", f"Error comparing: {comp_e}", n_fails_errors + 1
            elif not exec_res.get("success"):
                test_detail["status"], test_detail["error_message"], n_fails_errors = "Error", exec_res.get("error", "Exec error"), n_fails_errors + 1
                current_case_error = str(test_detail["error_message"])[:250]
            else:
                test_detail["status"], test_detail["error_message"], n_fails_errors = "Unknown Executor Status", exec_res.get("error", "Unexpected response"), n_fails_errors + 1
                current_case_error = str(test_detail["error_message"])[:250]
            
            if current_case_error and not first_err_msg: first_err_msg = current_case_error
            details.append(test_detail)

        if n_fails_errors > 0 and not first_err_msg: first_err_msg = "One or more test cases failed."
        return raw_results, details, n_fails_errors, first_err_msg

    def _call_executor_service(
        self, code_string: str, function_name: str, input_args: list
    ) -> dict:
        """Chama o serviço executor externo."""
        imports = ("from typing import List, Dict, Tuple, Set, Optional, Any\nimport re\nfrom collections import OrderedDict, deque\n\n")
        payload = {"code_string": imports + code_string, "function_name": function_name, "test_input_args": input_args}
        headers = {"Content-Type": "application/json"}
        default_err = {"success": False, "output": None, "stdout": None, "error": "Failed to call executor service", "execution_time_ms": -1}

        try:
            resp = requests.post(self.executor_url, headers=headers, json=payload, timeout=60)
            if resp.status_code == 200:
                try: return resp.json()
                except json.JSONDecodeError as json_e:
                    default_err["error"] = f"Executor JSON decode error: {json_e}"
                    return default_err
            else:
                default_err["error"] = f"Executor service returned status {resp.status_code}: {resp.text[:200]}"
                return default_err
        except requests.exceptions.Timeout:
            default_err["error"] = "Executor service request timeout"
            return default_err
        except requests.exceptions.RequestException as req_e:
            default_err["error"] = f"Executor connection error: {req_e}"
            return default_err
        except Exception as e:
            default_err["error"] = f"Unexpected error calling executor: {type(e).__name__}"
            return default_err

    def run_benchmark(self):
        """Executa o ciclo completo do benchmark por 'self.num_runs' vezes."""
        enabled_ais = [n for n, can in [("Claude", bool(self.claude_api_key)), ("ChatGPT", bool(self.openai_api_key)),
                                         ("Gemini", bool(self.gemini_api_key and self.gemini_model))] if can]
        if not enabled_ais:
            logging.critical("ERRO: Nenhuma API configurada ou disponível. Saindo.")
            return

        logging.info(f"Iniciando Benchmark para: {', '.join(enabled_ais)} ({self.num_runs} execução(ões) por desafio)")
        self.results, self.detailed_test_results = [], []

        for run_idx in range(self.num_runs):
            logging.info(f"\n{'='*20} INICIANDO EXECUÇÃO {run_idx + 1}/{self.num_runs} {'='*20}")
            for chal in self.code_challenges:
                sig = chal.get("function_signature", f"{chal['name'].lower().replace(' ','_')}(...)")
                prompt = f"Implement the following Python {'class' if 'class' in sig else 'function'}.\nDescription: {chal['description']}\nSignature:\n{sig}\nProvide ONLY the raw Python code definition for the function/class and any necessary standard library imports (like `from collections import OrderedDict` if needed). Do not include any explanations, comments within the code, usage examples, or markdown formatting like ```python."
                logging.info(f"\n--- [Run {run_idx+1}] Desafio: {chal['name']} ({chal['difficulty']}) ---")
                
                responses = {}
                if "Claude" in enabled_ais: responses["Claude"] = self.query_claude(prompt); time.sleep(1)
                if "ChatGPT" in enabled_ais: responses["ChatGPT"] = self.query_chatgpt(prompt); time.sleep(1)
                if "Gemini" in enabled_ais: responses["Gemini"] = self.query_gemini(prompt)

                for ai in enabled_ais:
                    resp_data = responses.get(ai, {"content": "", "execution_time": 0, "last_error": "No Response Attempted"})
                    self.evaluate_solution(ai, chal, resp_data, run_idx)
                
                print()
                print()
                time.sleep(0.5)

        self.generate_report()

    def _prepare_dataframe(self) -> pd.DataFrame | None:
        """Prepara DataFrame AGREGADO (média sobre runs) a partir de self.results."""
        if not self.results:
            logging.error("Não há resultados (self.results) para gerar DataFrame agregado.")
            return None
        try:
            df_all_runs = pd.DataFrame(self.results)
            metrics_to_avg = ["api_execution_time", "lines_of_code", "completeness"]
            grouped = df_all_runs.groupby(["ai", "challenge"])
            df_agg = grouped[metrics_to_avg].mean()

            def avg_valid_flake8(series):
                valid = series[series != -1]
                return valid.mean() if not valid.empty else -1

            df_agg["flake8_issues"] = grouped["flake8_issues"].apply(avg_valid_flake8)
            df_agg["avg_num_execution_errors"] = grouped["num_execution_errors"].mean()
            agg_test_metrics = self._calculate_aggregated_test_metrics()

            df_agg["avg_exec_success_rate"] = df_agg.index.map(lambda idx: agg_test_metrics.get(idx, {}).get("avg_exec_success_rate", 0.0))
            df_agg["avg_exec_time_ms"] = df_agg.index.map(lambda idx: agg_test_metrics.get(idx, {}).get("avg_exec_time_ms", np.nan))
            df_agg["total_tests_passed"] = df_agg.index.map(lambda idx: agg_test_metrics.get(idx, {}).get("total_tests_passed", 0))
            df_agg["total_tests_attempted"] = df_agg.index.map(lambda idx: agg_test_metrics.get(idx, {}).get("total_tests_attempted", 0))

            def get_first_error(series):
                non_empty = series.loc[series.fillna("").astype(str) != ""].astype(str)
                return non_empty.iloc[0] if not non_empty.empty else ""

            df_agg["first_execution_error"] = grouped["first_execution_error"].apply(get_first_error)

            difficulty_map = df_all_runs.drop_duplicates(subset=["ai", "challenge"]).set_index(["ai", "challenge"])["difficulty"]
            df_agg = df_agg.join(difficulty_map)
            df_agg = df_agg.reset_index()
            df_agg = df_agg.rename(columns={"avg_num_execution_errors": "num_execution_errors"})
            self._normalize_dataframe_types(df_agg)
            return df_agg
        except Exception as e:
            logging.error(f"Erro ao preparar DataFrame agregado: {e}", exc_info=True)
            return None

    def _calculate_aggregated_test_metrics(self) -> dict:
        """Calcula métricas agregadas de testes a partir de self.detailed_test_results (TODAS as runs)."""
        aggregated_metrics = {}
        if not self.detailed_test_results:
            return aggregated_metrics
        try:
            df_details = pd.DataFrame(self.detailed_test_results)
            df_details["execution_time_ms"] = pd.to_numeric(df_details["execution_time_ms"], errors="coerce")
            grouped = df_details.groupby(["ai", "challenge"])
            for name, group in grouped:
                attempted = group[~group["status"].isin(["API Error", "No Code", "No Code/No Function"])]
                passed = attempted[attempted["status"] == "Pass"]
                n_attempted, n_passed = len(attempted), len(passed)
                success_rate = n_passed / n_attempted if n_attempted > 0 else 0.0
                valid_times = passed["execution_time_ms"].dropna()
                avg_time = valid_times.mean() if not valid_times.empty else np.nan
                aggregated_metrics[name] = {"avg_exec_success_rate": success_rate, "avg_exec_time_ms": avg_time,
                                            "total_tests_passed": n_passed, "total_tests_attempted": n_attempted}
        except Exception as e:
            logging.error(f"Erro ao calcular métricas agregadas de teste: {e}", exc_info=True)
        return aggregated_metrics

    def _normalize_dataframe_types(self, df: pd.DataFrame):
        """Normaliza tipos de colunas numéricas e trata NaNs no DataFrame fornecido."""
        numeric_cols = {"api_execution_time": (float, 0.0), "lines_of_code": (int, 0), "completeness": (int, 0),
                        "flake8_issues": (int, -1), "num_execution_errors": (float, 0.0), "avg_exec_success_rate": (float, 0.0),
                        "avg_exec_time_ms": (float, -1.0), "total_tests_passed": (int, 0), "total_tests_attempted": (int, 0)}
        for col_name, (target_dtype, fill_value) in numeric_cols.items():
            if col_name in df.columns:
                df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
                df.loc[:, col_name] = df[col_name].fillna(fill_value)
                try:
                    if np.isnan(fill_value) and target_dtype == int:
                        df.loc[:, col_name] = df[col_name].astype(pd.Int64Dtype()) if pd.__version__ >= "1.0.0" else df[col_name].astype(float)
                    else:
                        df.loc[:, col_name] = df[col_name].astype(target_dtype)
                except Exception:
                    df.loc[:, col_name] = df[col_name].astype(float)
        if "first_execution_error" in df.columns:
            df["first_execution_error"] = df["first_execution_error"].fillna("").astype(str)
        elif "num_execution_errors" in df.columns:
            df["first_execution_error"] = ""

    def _save_environment_info(self, timestamp_str: str):
        filepath = os.path.join(self.output_dir, f"environment_info_{timestamp_str}.txt")
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"Number of Runs per Challenge/AI: {self.num_runs}\n")
                f.write("-" * 30 + "\n")
                for key, value in self.environment_info.items():
                    if key == "Num Runs": continue
                    if isinstance(value, dict):
                        f.write(f"{key}:\n")
                        for sub_key, sub_value in value.items(): f.write(f"  {sub_key}: {sub_value}\n")
                    else: f.write(f"{key}: {value}\n")
            logging.info(f"Informações do Ambiente salvas: {filepath}")
        except Exception as e:
            logging.error(f"Erro ao salvar informações do ambiente: {e}", exc_info=True)

    def _save_full_results_json(self, timestamp_str: str):
        filepath = os.path.join(self.output_dir, f"benchmark_results_all_runs_{timestamp_str}.json")
        try:
            class NpEncoder(json.JSONEncoder):
                def default(self, o): # CORRIGIDO: obj -> o
                    if isinstance(o, np.integer): return int(o)
                    elif isinstance(o, np.floating): return float(o)
                    elif isinstance(o, np.ndarray): return o.tolist()
                    elif isinstance(o, datetime): return o.isoformat()
                    elif isinstance(o, np.bool_): return bool(o)
                    elif isinstance(o, bytes): return o.decode("utf-8", "replace")
                    return super(NpEncoder, self).default(o)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False, cls=NpEncoder)
            logging.info(f"Resultados JSON (todos os runs) salvos: {filepath}")
        except Exception as e:
            logging.error(f"Erro ao salvar JSON completo (todos os runs): {e}", exc_info=True)

    def _save_detailed_tests_csv(self, timestamp_str: str):
        filepath = os.path.join(self.output_dir, f"benchmark_test_details_all_runs_{timestamp_str}.csv")
        if self.detailed_test_results:
            try:
                df = pd.DataFrame(self.detailed_test_results)
                cols_order = ["run_index", "ai", "challenge", "difficulty", "test_case_index", "status",
                              "execution_time_ms", "error_message", "stdout", "input_args_str",
                              "expected_output_str", "actual_output_str"]
                for col in cols_order:
                    if col not in df.columns: df[col] = pd.NA
                df[cols_order].to_csv(filepath, index=False, encoding="utf-8", float_format="%.3f")
                logging.info(f"Detalhes dos Testes CSV (todos os runs) salvos: {filepath}")
            except Exception as e:
                logging.error(f"Erro ao salvar CSV de detalhes dos testes: {e}", exc_info=True)
        else:
            logging.warning("Nenhum resultado detalhado de teste para salvar em CSV.")

    def _create_csv_header(self) -> str:
        header_lines = ["# --- Environment Info ---", f"# Number of Runs per Challenge/AI: {self.num_runs}"]
        for key, value in self.environment_info.items():
            if key == "Num Runs": continue
            if isinstance(value, dict):
                header_lines.append(f"# {key}:")
                [header_lines.append(f"#   {sk}: {sv}") for sk, sv in value.items()]
            else: header_lines.append(f"# {key}: {value}")
        header_lines.append("# --- Benchmark Data (Aggregated/Averaged over Runs) ---")
        return "\n".join(header_lines) + "\n"

    def _save_summary_csv(self, df_agg: pd.DataFrame, timestamp_str: str):
        filepath = os.path.join(self.output_dir, f"benchmark_summary_agg_{timestamp_str}.csv")
        if df_agg is None:
            logging.error("DataFrame agregado é None. Pulando salvamento do resumo CSV.")
            return
        try:
            with open(filepath, "w", encoding="utf-8", newline="") as f:
                f.write(self._create_csv_header())
                df_agg.to_csv(f, index=False, encoding="utf-8", float_format="%.4f", lineterminator="\n")
            logging.info(f"Resumo Agregado CSV salvo: {filepath}")
        except Exception as e:
            logging.error(f"Erro ao salvar resumo agregado CSV: {e}", exc_info=True)

    def _calculate_statistics(self, df_agg: pd.DataFrame) -> pd.DataFrame | None:
        """Calcula estatísticas gerais agregadas por IA a partir do DataFrame JÁ AGREGADO."""
        if df_agg is None:
            logging.error("DataFrame agregado de entrada para estatísticas é None.")
            return None
        try:
            metrics_definitions = [
                ("Avg API Time (s)", "api_execution_time", "mean"), ("Std Dev API Time (s)", "api_execution_time", "std"),
                ("Avg LoC", "lines_of_code", "mean"), ("Std Dev LoC", "lines_of_code", "std"),
                ("Avg Completeness (%)", "completeness", "mean"), ("Avg Flake8 Issues", "flake8_issues", "mean_valid", {"ignore": -1}),
                ("Total Passed (All Runs)", "total_tests_passed", "sum"), ("Total Attempted (All Runs)", "total_tests_attempted", "sum"),
                ("Overall Success Rate (%)", "avg_exec_success_rate", "mean_percentage"), ("Std Dev Success Rate (%)", "avg_exec_success_rate", "std_percentage"),
                ("Avg Exec Time (ms)", "avg_exec_time_ms", "mean_valid", {"ignore": -1.0}),
                ("Avg Errors/Fails per Challenge", "num_execution_errors", "mean"), ("Example Error", "first_execution_error", "first_non_empty"),
            ]
            ais = sorted(df_agg["ai"].unique())
            stats_data = {"Metric": [m[0] for m in metrics_definitions]}
            for ai in ais:
                df_ai = df_agg[df_agg["ai"] == ai]
                ai_column_data = []
                for _, col_key, agg_type, *options_list in metrics_definitions:
                    value = np.nan
                    options = options_list[0] if options_list else {}
                    ignore_val = options.get("ignore")
                    if col_key in df_ai.columns:
                        series = df_ai[col_key].copy()
                        if ignore_val is not None: series = series[series != ignore_val]
                        if agg_type != "first_non_empty": series = series.dropna()
                        if not series.empty:
                            if agg_type == "mean": value = series.mean()
                            elif agg_type == "std": value = series.std()
                            elif agg_type == "sum": value = series.sum()
                            elif agg_type == "mean_valid": value = series.mean() if not series.empty else np.nan
                            elif agg_type == "mean_percentage": value = series.mean() * 100
                            elif agg_type == "std_percentage": value = series.std() * 100
                            elif agg_type == "first_non_empty":
                                errors = df_ai[col_key][df_ai[col_key].fillna("").astype(str) != ""]
                                value = errors.iloc[0] if not errors.empty else ""
                        elif agg_type == "first_non_empty": value = ""
                    ai_column_data.append(value)
                stats_data[ai] = ai_column_data
            return pd.DataFrame(stats_data)
        except Exception as e:
            logging.error(f"Erro ao calcular estatísticas gerais: {e}", exc_info=True)
            return None

    def _save_statistics_csv(self, stats_df: pd.DataFrame, timestamp_str: str):
        if stats_df is None:
            logging.error("DataFrame de estatísticas é None. Pulando salvamento CSV.")
            return
        filepath = os.path.join(self.output_dir, f"benchmark_stats_agg_{timestamp_str}.csv")
        try:
            with open(filepath, "w", encoding="utf-8", newline="") as f:
                f.write(self._create_csv_header())
                stats_df.to_csv(f, index=False, encoding="utf-8", float_format="%.2f", lineterminator="\n")
            logging.info(f"Estatísticas Gerais CSV salvas: {filepath}")
        except Exception as e:
            logging.error(f"Erro ao salvar estatísticas CSV: {e}", exc_info=True)

    def _display_summary_table(self, stats_df: pd.DataFrame):
        """Mostra tabela de resumo formatada no console (requer tabulate)."""
        print(f"\n===== SUMÁRIO DO BENCHMARK (ESTATÍSTICAS GERAIS - {self.num_runs} RUNS) =====")
        if stats_df is None or stats_df.empty:
            print("Não foi possível gerar sumário (DataFrame de estatísticas vazio ou None).")
            return

        display_df = stats_df.copy()
        for col_name in display_df.columns:
            if col_name == "Metric": continue
            formatted_column = []
            for _, row in display_df.iterrows():
                metric, value = row["Metric"], row[col_name]
                fmt_value = "N/A"
                if not pd.isna(value):
                    try:
                        if "Std Dev" in metric:
                            if "(%)" in metric: fmt_value = f"±{value:.1f}%"
                            elif "(s)" in metric: fmt_value = f"±{value:.2f}s"
                            else: fmt_value = f"±{value:.1f}"
                        elif "Rate (%)" in metric or "Completeness (%)" in metric: fmt_value = f"{value:.1f}%"
                        elif "Flake8 Issues" in metric: fmt_value = f"{value:.1f}" if value != -1 else "N/A"
                        elif "Time (ms)" in metric: fmt_value = f"{value:.1f}" if value != -1 else "N/A"
                        elif "Total " in metric: fmt_value = f"{int(value)}"
                        elif "Time (s)" in metric: fmt_value = f"{value:.2f}"
                        elif "Avg Errors/Fails" in metric: fmt_value = f"{value:.2f}"
                        elif "Example Error" in metric: s = str(value); fmt_value = s[:60] + ("..." if len(s) > 60 else "")
                        else: fmt_value = f"{value:.1f}"
                    except (ValueError, TypeError):
                        fmt_value = str(value)[:60]
                formatted_column.append(fmt_value)
            display_df[col_name] = formatted_column

        try:
            from tabulate import tabulate
            print(tabulate(display_df, headers="keys", tablefmt="grid", showindex=False)) # type: ignore
        except ImportError:
            logging.warning("Biblioteca 'tabulate' não instalada. Tabela de resumo será exibida sem formatação.")
            print(display_df.to_string(index=False))
        except Exception as e_tab:
            logging.error(f"Erro ao imprimir tabela com tabulate: {e_tab}. Usando print fallback.")
            print(display_df.to_string(index=False))

    def generate_report(self):
        """Gera todos os relatórios e visualizações (AGREGADOS sobre runs)."""
        if not self.results:
            logging.error("Não há resultados para gerar relatório.")
            return

        logging.info("\n--- Gerando Relatórios Finais (Agregados) ---")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        logging.info(f"Timestamp: {ts}")
        logging.info(f"Diretório: '{self.output_dir}/'")

        self._save_environment_info(ts)
        df_agg = self._prepare_dataframe()
        if df_agg is None:
            logging.error("Falha ao preparar DataFrame agregado. Abortando relatórios.")
            return

        self._save_full_results_json(ts)
        self._save_detailed_tests_csv(ts)
        self._save_summary_csv(df_agg, ts)
        stats_df = self._calculate_statistics(df_agg)
        if stats_df is not None:
            self._save_statistics_csv(stats_df, ts)
            try:
                logging.info("\n--- Gerando Visualizações (baseadas em médias sobre runs) ---")
                self.create_visualizations(df_agg, ts, self.output_dir)
                logging.info(f"Visualizações salvas em '{self.output_dir}/'")
            except Exception as e:
                logging.error(f"Erro ao gerar visualizações: {e}", exc_info=True)
            self._display_summary_table(stats_df)
        else:
            logging.error("Não foi possível calcular/salvar estatísticas gerais.")

    def _plot_bar_charts(self, df_plot, ai_list, cmap, challenge_order, ts, out_dir):
        """Gera gráficos de barras comparativos por desafio (usando médias sobre runs)."""
        logging.info("Gerando Gráficos de Barras (média por desafio)...")
        metrics_to_plot = {
            "1_api_t": ("API Time (Avg)", "Time (s)", "api_execution_time"),
            "2_loc": ("Lines of Code (Avg)", "Lines", "lines_of_code"),
            "3_comp": ("Completeness Score (Avg)", "Score (%)", "completeness"),
            "4_f8": ("Flake8 Issues (Avg)", "Avg Issues", "flake8_issues"),
            "7_err": ("Execution Errors/Fails (Avg)", "Avg Errors/Fails per Challenge", "num_execution_errors"),
        }
        if "avg_exec_success_rate" in df_plot.columns and df_plot["avg_exec_success_rate"].notna().any():
            metrics_to_plot["5_ok"] = ("Execution Success Rate (Avg)", "Success Rate (%)", "avg_exec_success_rate")
        if "avg_exec_time_ms" in df_plot.columns and (df_plot["avg_exec_time_ms"].notna() & (df_plot["avg_exec_time_ms"] != -1)).any():
            metrics_to_plot["6_exec_t"] = ("Execution Time (Avg)", "Time (ms)", "avg_exec_time_ms")
        metrics_to_plot = dict(sorted(metrics_to_plot.items()))

        for key, (title, ylabel, col_name) in metrics_to_plot.items():
            if col_name not in df_plot.columns: continue
            data_for_plot = df_plot.copy()
            if col_name == "flake8_issues": data_for_plot = data_for_plot[data_for_plot[col_name] != -1]
            elif col_name == "avg_exec_time_ms": data_for_plot = data_for_plot[data_for_plot[col_name].notna() & (data_for_plot[col_name] != -1)]
            elif col_name == "num_execution_errors": data_for_plot = data_for_plot[data_for_plot[col_name].notna()]
            elif col_name == "avg_exec_success_rate": data_for_plot[col_name] = data_for_plot[col_name] * 100
            if data_for_plot.empty or data_for_plot[col_name].isnull().all(): continue

            fig, ax = plt.subplots()
            try:
                pivot_table = data_for_plot.pivot_table(index="challenge", columns="ai", values=col_name, aggfunc="mean").reindex(challenge_order, axis=0).reindex(ai_list, axis=1)
                fill_val = 0.0 if col_name not in ["flake8_issues", "avg_exec_time_ms", "num_execution_errors"] else np.nan
                pivot_table.fillna(value=fill_val, inplace=True)
                if pivot_table.isnull().all().all(): plt.close(fig); continue
                
                pivot_table.plot(kind="bar", figsize=(max(12, len(pivot_table.columns) * 2), 7), color=[cmap.get(ai, "grey") for ai in pivot_table.columns], width=0.8, ax=ax)
                ax.set_title(f"{title} por Desafio (Média sobre {self.num_runs} runs)")
                ax.set_ylabel(ylabel); ax.set_xlabel("Desafio")
                ax.tick_params(axis="x", rotation=45, labelsize=9)
                plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
                ax.tick_params(axis="y", labelsize=9)

                if col_name in ["completeness", "avg_exec_success_rate"]: ax.set_ylim(0, 105)
                elif col_name in ["flake8_issues", "num_execution_errors"] and not pivot_table.isnull().all().all():
                    max_val = pivot_table.max(skipna=True).max(skipna=True)
                    ax.set_ylim(bottom=0, top=max(1, max_val * 1.1) if not pd.isna(max_val) else 1)
                elif col_name == "avg_exec_time_ms" and not pivot_table.isnull().all().all(): ax.set_ylim(bottom=0)

                ax.legend(title="IA", bbox_to_anchor=(1.02, 1), loc="upper left")
                plt.tight_layout(rect=tuple([0, 0, 0.88, 1])) # CORRIGIDO: list -> tuple
                filepath = os.path.join(out_dir, f"{key}_bar_chart_avg_{ts}.png")
                plt.savefig(filepath)
            except Exception as e:
                logging.error(f"Erro ao gerar gráfico de barras para '{title}': {e}", exc_info=True)
            finally:
                plt.close(fig)

    def _plot_lru_times_chart(self, detailed_df, ai_list, cmap, ts, out_dir):
        """Gera gráfico de barras para o tempo médio de operação do LRU Cache (média sobre runs)."""
        logging.info("Gerando Gráfico de Tempo Médio - LRU Cache (média sobre runs)...")
        if detailed_df is None or detailed_df.empty: return
        lru_data = detailed_df[(detailed_df["challenge"] == "LRU Cache") & (detailed_df["execution_time_ms"].notna()) & (detailed_df["execution_time_ms"] >= 0)].copy()
        if lru_data.empty: return

        avg_lru_times = lru_data.groupby("ai")["execution_time_ms"].mean().reindex(ai_list).dropna()
        if avg_lru_times.empty: return

        fig, ax = plt.subplots(figsize=(max(8, len(avg_lru_times) * 1.5), 6))
        try:
            colors = [cmap.get(ai, "grey") for ai in avg_lru_times.index]
            avg_lru_times.plot(kind="bar", ax=ax, color=colors, width=0.6)
            ax.set_title(f"Tempo Médio por Operação - LRU Cache (Média sobre {self.num_runs} runs)")
            ax.set_ylabel("Tempo Médio (ms)"); ax.set_xlabel("Inteligência Artificial")
            ax.tick_params(axis="x", rotation=0, labelsize=10); ax.tick_params(axis="y", labelsize=9)
            ax.grid(axis="y", linestyle="--", alpha=0.7)
            for container in ax.containers:
                ax.bar_label(container, fmt="%.3f", label_type="edge", padding=3, fontsize=8) # type: ignore
            ax.set_ylim(bottom=0, top=ax.get_ylim()[1] * 1.1)
            plt.tight_layout()
            filepath = os.path.join(out_dir, f"8_lru_exec_time_avg_{ts}.png")
            plt.savefig(filepath)
            logging.info(f"Gráfico Tempos LRU (Média) salvo: {filepath}")
        except Exception as e:
            logging.error(f"Erro ao gerar gráfico de tempos LRU: {e}", exc_info=True)
        finally:
            plt.close(fig)

    def _plot_box_plots(self, df_agg, ai_list, cmap, ts, out_dir):
        """Gera box plots para distribuição das métricas MÉDIAS por desafio."""
        logging.info("Gerando Box Plots (distribuição das médias por desafio)...")
        df_plot = df_agg.copy()
        metrics_to_plot = {"api_execution_time": {"title": "Distribuição API Time (Média)", "ylabel": "Tempo(s)", "filter": None},
                           "lines_of_code": {"title": "Distribuição LoC (Média)", "ylabel": "Linhas", "filter": lambda d, c: d[c] > 0},
                           "completeness": {"title": "Distribuição Completude (Média)", "ylabel": "Score (%)", "filter": None}}
        if "flake8_issues" in df_plot.columns and (df_plot["flake8_issues"] != -1).any():
            metrics_to_plot["flake8_issues"] = {"title": "Distribuição Flake8 Issues (Média)", "ylabel": "Issues", "filter": lambda d, c: d[c] != -1}
        # ... (restante da lógica de plotagem original)

    def _plot_radar_chart(self, df_agg, ai_list, cmap, ts, out_dir):
        """Gera gráfico de radar comparativo (Normalizado, baseado em médias sobre runs)."""
        if len(ai_list) < 2: return
        logging.info(f"Gerando gráfico de radar para {len(ai_list)} IAs...")
        df = df_agg.copy()
        radar_cols_def = {"api_execution_time": {"label": "Veloc. API (Norm)", "lower_is_better": True},
                          "lines_of_code": {"label": "Conc. LoC (Norm)", "lower_is_better": True},
                          "completeness": {"label": "Completude (%)", "lower_is_better": False},
                          "flake8_issues": {"label": "Qualid. Flake8 (Norm)", "lower_is_better": True, "ignore_val": -1},
                          "avg_exec_success_rate": {"label": "Tx. Sucesso Exec (%)", "lower_is_better": False}}
        valid_radar_cols = {k: v for k, v in radar_cols_def.items() if k in df.columns}
        if len(valid_radar_cols) < 3: return

        categories, n_cats = [v["label"] for v in valid_radar_cols.values()], len(valid_radar_cols)
        means = df.groupby("ai")[list(valid_radar_cols.keys())].mean()
        norm_data = pd.DataFrame(index=means.index)

        for col, config in valid_radar_cols.items():
            if col in ["completeness", "avg_exec_success_rate"]:
                norm_series = (means[col] * (100 if col == "avg_exec_success_rate" else 1)).clip(0, 100)
            else:
                series = means[col].copy()
                if config.get("ignore_val") is not None: series = series[series != config["ignore_val"]]
                if series.empty or series.isnull().all(): norm_series = pd.Series(np.nan, index=means.index)
                else:
                    min_val, max_val = series.min(), series.max()
                    range_val = max_val - min_val if max_val > min_val else 1.0
                    norm_series = 100 * ((max_val - means[col]) if config["lower_is_better"] else (means[col] - min_val)) / range_val
                    if max_val == min_val: norm_series = 0.0 if config["lower_is_better"] else 100.0
                norm_series = norm_series.reindex(means.index).fillna(0.0 if config.get("lower_is_better") else 100.0)
            norm_data[config["label"]] = norm_series.clip(0, 100)

        angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
        angles += angles[:1]
        fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
        try:
            ax.set_xticks(angles[:-1]); ax.set_xticklabels(categories); ax.set_yticks(np.arange(0, 101, 20)); ax.set_ylim(0, 100)
            for idx, ai in enumerate(norm_data.index):
                if ai not in ai_list: continue
                values = norm_data.loc[ai].values.flatten().tolist()
                values += values[:1] # type: ignore
                color = cmap.get(ai, plt.cm.tab10(idx / max(1, len(ai_list))))
                ax.plot(angles, values, "o-", linewidth=2, label=ai, color=color)
                ax.fill(angles, values, color, alpha=0.2)
            plt.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15))
            plt.title("Comparação Geral Normalizada (Maior = Melhor)", size=14, y=1.18)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"9_radar_chart_avg_{ts}.png"))
        finally:
            plt.close(fig)

    def create_visualizations(self, df_agg, ts, out_dir):
        """Cria e salva visualizações (baseadas em médias sobre runs)."""
        try:
            logging.debug(f"Versão Matplotlib: {matplotlib.__version__}") # CORRIGIDO: plt -> matplotlib
        except Exception as e:
            logging.warning(f"Não foi possível obter a versão do Matplotlib: {e}")

        if df_agg is None or df_agg.empty:
            logging.warning("DataFrame agregado vazio ou None para gráficos. Pulando visualizações.")
            return

        try: plt.style.use("ggplot")
        except: logging.warning("Estilo 'ggplot' não encontrado, usando default.")

        ais_in_data = sorted(df_agg["ai"].unique())
        if not ais_in_data: return
        colors = plt.cm.viridis(np.linspace(0, 1, len(ais_in_data))) if len(ais_in_data) > 10 else plt.cm.tab10(np.linspace(0, 1, len(ais_in_data)))
        cmap = {ai: colors[i] for i, ai in enumerate(ais_in_data)}

        if not all(c in df_agg.columns for c in ["challenge", "ai", "difficulty"]):
            logging.error("Faltando colunas essenciais no DF. Pulando visualizações.")
            return

        challenge_order = sorted(df_agg["challenge"].unique())
        if "difficulty" in df_agg.columns:
            try:
                difficulty_order = pd.CategoricalDtype(["Easy", "Medium", "Hard"], ordered=True)
                df_temp_sort = df_agg.copy()
                df_temp_sort["diff_cat"] = df_temp_sort["difficulty"].astype(difficulty_order)
                challenge_order = df_temp_sort.sort_values("diff_cat")["challenge"].unique()
            except Exception:
                pass
        
        detailed_df_lru = pd.DataFrame(self.detailed_test_results) if self.detailed_test_results else None
        if detailed_df_lru is not None:
            detailed_df_lru["execution_time_ms"] = pd.to_numeric(detailed_df_lru["execution_time_ms"], errors="coerce")

        self._plot_bar_charts(df_agg, ais_in_data, cmap, challenge_order, ts, out_dir)
        self._plot_box_plots(df_agg, ais_in_data, cmap, ts, out_dir)
        self._plot_radar_chart(df_agg, ais_in_data, cmap, ts, out_dir)
        if detailed_df_lru is not None:
            self._plot_lru_times_chart(detailed_df_lru, ais_in_data, cmap, ts, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Executa um benchmark de programação comparando modelos de IA (Claude, ChatGPT, Gemini).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-n", "--num_runs", type=int, default=3, help="Número de vezes que cada desafio/IA é executado.")
    parser.add_argument("-o", "--output_dir", type=str, default="benchmark_reports", help="Diretório para salvar os relatórios e gráficos.")
    parser.add_argument("--claude_model", type=str, default=None, help="Nome do modelo Claude a ser usado.")
    parser.add_argument("--openai_model", type=str, default=None, help="Nome do modelo OpenAI (ChatGPT) a ser usado.")
    parser.add_argument("--gemini_model", type=str, default=None, help="Nome do modelo Gemini a ser usado.")
    parser.add_argument("--executor_url", type=str, default=None, help="URL do serviço externo para execução de código.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Ativa logging mais detalhado (nível DEBUG).")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug("Logging DEBUG ativado.")

    print("=" * 59)
    print(" AI Programming Benchmark: Claude vs ChatGPT vs Gemini")
    print("        (Análise Flake8, Execução Externa/Local)")
    print("=" * 59)

    multiprocessing.freeze_support()

    if not os.path.exists(".env"):
        logging.warning("Ficheiro .env não encontrado. Chaves API podem não ser carregadas.")

    try:
        benchmark = AIBenchmark(
            num_runs=args.num_runs, output_dir=args.output_dir,
            claude_model_name=args.claude_model, openai_model_name=args.openai_model,
            gemini_model_name=args.gemini_model, executor_service_url=args.executor_url,
        )
        if benchmark.claude_api_key or benchmark.openai_api_key or benchmark.gemini_api_key:
            benchmark.run_benchmark()
            logging.info("\nBenchmark concluído.")
            logging.info(f"Relatórios salvos em: '{benchmark.output_dir}'")
        else:
            logging.critical("\nERRO FATAL: Nenhuma chave API encontrada ou configurada corretamente.")
            logging.critical("Verifique o seu ficheiro .env: CLAUDE_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY.")
    except Exception as e:
        logging.critical("\nERRO INESPERADO DURANTE A EXECUÇÃO DO BENCHMARK:", exc_info=True)

    print("=" * 59)