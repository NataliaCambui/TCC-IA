# -*- coding: utf-8 -*-
import requests
from string import ascii_lowercase, ascii_uppercase

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
import builtins  # Necessário para a sandbox do exec (referência a __build_class__)
import logging  # SUGESTÃO 1: Usar logging
import argparse  # SUGESTÃO 2: Usar argparse

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
                "description": "Count the number of islands in a 2D grid (list of lists) where 1 is land and 0 is water. An island is formed by connecting adjacent lands horizontally or vertically. “Trate também o caso em que a matriz grid é vazia, retornando 0.”",
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
        self.environment_info = self._get_environment_info()  # Coleta info do ambiente

    # --- GARANTE QUE ESTE MÉTODO ESTÁ DEFINIDO ---
    def _configure_gemini_api(self):
        """Configura a API Gemini se a chave API estiver disponível."""
        if self.gemini_api_key:
            try:
                genai.configure(api_key=self.gemini_api_key)
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
                f"Rate limit Gemini. Tentativa {retry_state.attempt_number}. Esperando {retry_state.next_action.sleep:.1f}s..."
            ),
        )
        def _call_gemini_api_with_retry(
            prompt_str: str,
        )-> object:
            """Função interna que faz a chamada real à API."""
            model = genai.GenerativeModel(self.gemini_model or 'gemini-1.5-flash')
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
            
                "ascii_lowercase": ascii_lowercase,
                "ascii_uppercase": ascii_uppercase,}
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
            start_time = time.time()
            process.start()
            process.join(timeout=timeout_seconds)  # Espera com timeout
            elapsed_time = time.time() - start_time

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
                    except queue.Empty:
                        pass  # Sem mensagem na fila
                    except Exception:
                        pass  # Erro ao ler a fila
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
                except:
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

        Extrai código, analisa com Flake8, mede completude, e executa testes.
        Adiciona resultados detalhados com 'run_idx' à lista GERAL `self.detailed_test_results`.

        Args:
            ai_name (str): Nome da IA.
            challenge (dict): Dicionário do desafio.
            response (dict): Resposta da API {'content': str, 'execution_time': float, 'last_error': str|None}.
            run_idx (int): Índice da execução atual (0 a num_runs-1).

        Returns:
            dict: Dicionário com resultados agregados para este AI/Desafio/Execução.
                  Estes resultados são também adicionados a `self.results`.
        """
        content = response.get("content", "")
        api_time = response.get("execution_time", 0)
        api_error = response.get("last_error")  # Captura erro da API, se houver

        # Se não houve conteúdo E houve erro na API, regista falha e retorna
        # Adiciona detalhes de erro para cada caso de teste na lista global
        if not content and api_error:
            logging.warning(
                f"Sem conteúdo de {ai_name} para {challenge['name']} (Run {run_idx+1}) devido a erro API: {api_error}. Avaliação pulada."
            )
            fail_result_summary = {
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
                ),  # Marca todos como erro
                "first_execution_error": f"API Error: {str(api_error)[:100]}",  # Erro API como causa
                "code": "",
                "full_response": "",
                "execution_results_raw": [],
            }
            self.results.append(fail_result_summary)  # Adiciona sumário da falha

            num_test_cases = len(challenge.get("test_cases", []))
            # Para LRU, há apenas 1 "caso de teste" (a sequência)
            num_detail_entries = (
                num_test_cases if challenge["name"] != "LRU Cache" else 1
            )
            for i in range(num_detail_entries):
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
                        "error_message": str(api_error),
                        "stdout": None,
                    }
                )
            return fail_result_summary  # Retorna o sumário da falha

        # Prossegue se houver conteúdo
        code = self.extract_code(content)
        lines = self.count_code_lines(code)
        completeness = self.measure_response_completeness(
            code, challenge["description"]
        )
        flake8 = self.analyze_code_quality_flake8(code)

        raw_results = []
        details_curr_run = []  # Detalhes apenas desta execução/ai/challenge
        n_errors = 0
        first_err = ""
        can_exec = bool(code)
        func_name = None

        # Tenta detetar nome da função/classe (exceto para LRU que é tratado diferente)
        if can_exec and challenge["name"] != "LRU Cache":
            func_name = self._detect_function_name(code, challenge)
            if not func_name:
                can_exec = False
                first_err = "Function/Class name not found"

        if can_exec:
            if challenge["name"] == "LRU Cache":
                logging.info(
                    f"Executando LRU Cache de {ai_name} localmente (Run {run_idx+1})..."
                )
                tc_list = challenge.get("test_cases", [])
                if (
                    tc_list
                    and isinstance(tc_list[0], dict)
                    and "sequence" in tc_list[0]
                ):
                    lru_res = self._execute_lru_locally(code, tc_list[0]["sequence"])
                    raw_results = lru_res.get("results", [])  # Detalhes dos steps
                    ok = lru_res.get("success", False)
                    n_errors = 0 if ok else 1  # 1 erro se falhar, 0 se sucesso
                    first_err = lru_res.get("error", "")
                    # Se falhou mas não tem erro geral, pega o erro do primeiro step falhado
                    if not first_err and not ok:
                        for step in raw_results:
                            if step.get("err"):
                                first_err = (
                                    f"Step {step['op']}{step['args']}: {step['err']}"
                                )
                                break
                    if not first_err and not ok:
                        first_err = "LRU execution failed (unknown step error)"

                    # Calcula tempo médio dos steps OK
                    times_ok = [
                        s["t_ms"]
                        for s in raw_results
                        if s.get("t_ms", -1) >= 0 and s.get("ok") is True
                    ]
                    avg_time_ms = np.mean(times_ok) if times_ok else -1.0

                    # Cria a entrada detalhada para este run do LRU
                    details_curr_run.append(
                        {
                            "run_index": run_idx,
                            "ai": ai_name,
                            "challenge": challenge["name"],
                            "difficulty": challenge["difficulty"],
                            "test_case_index": 0,  # Index único para a sequência
                            "input_args_str": json.dumps(
                                tc_list[0]["sequence"]
                            ),  # Salva a sequência como input
                            "expected_output_str": "Sequence Result",
                            "actual_output_str": f"Success: {ok}",
                            "status": (
                                "Pass" if ok else ("Fail" if raw_results else "Error")
                            ),  # Status geral da sequência
                            "execution_time_ms": avg_time_ms,  # Tempo médio dos steps OK
                            "error_message": str(first_err)[:250],
                            "stdout": None,
                        }
                    )
                    logging.info(
                        f"Resultado LRU local de {ai_name} (Run {run_idx+1}): {'Sucesso' if ok else 'Falha/Erro'}"
                    )
                else:
                    logging.warning(
                        f"Caso de teste LRU inválido para {ai_name} (Run {run_idx+1})."
                    )
                    n_errors = 1
                    first_err = "Invalid LRU test case format."
                    details_curr_run.append(
                        {  # Regista erro no detalhe
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
            else:  # Outros desafios (via Executor Service)
                logging.info(
                    f"Executando '{challenge['name']}' de {ai_name} via Executor Service (Run {run_idx+1})..."
                )
                # _run_test_cases_fixed retorna os detalhes JÁ com run_idx
                raw, details_from_tests, errors, first_e = self._run_test_cases_fixed(
                    ai_name, challenge, code, func_name, run_idx
                )
                raw_results = raw
                n_errors = errors
                first_err = first_e
                details_curr_run.extend(
                    details_from_tests
                )  # Adiciona detalhes desta execução
        else:  # Sem código ou nome de função não encontrado
            err_msg = (
                "Code not extracted" if not code else "Function/Class name not found"
            )
            logging.warning(
                f"{err_msg} para '{challenge['name']}' por {ai_name} (Run {run_idx+1}). Pulando execução."
            )
            n_tests = len(challenge.get("test_cases", []))
            n_errors = n_tests  # Considera todos os testes como erro
            first_err = err_msg
            # Adiciona entradas de erro para cada caso de teste nos detalhes
            for i in range(n_tests):
                test_case_info = (
                    challenge["test_cases"][i]
                    if i < len(challenge.get("test_cases", []))
                    else {}
                )
                details_curr_run.append(
                    {
                        "run_index": run_idx,
                        "ai": ai_name,
                        "challenge": challenge["name"],
                        "difficulty": challenge["difficulty"],
                        "test_case_index": i,
                        "input_args_str": json.dumps(
                            test_case_info.get("input_args", "N/A")
                        ),
                        "expected_output_str": json.dumps(
                            test_case_info.get("expected_output", "N/A")
                        ),
                        "actual_output_str": None,
                        "status": "No Code/No Function",  # Status mais específico
                        "execution_time_ms": -1,
                        "error_message": err_msg,
                        "stdout": None,
                    }
                )

        # Adiciona detalhes desta execução à lista GERAL da instância
        self.detailed_test_results.extend(details_curr_run)

        # Cria dicionário de sumário para ESTA execução específica
        run_summary = {
            "run_index": run_idx,  # Adiciona run_idx
            "ai": ai_name,
            "challenge": challenge["name"],
            "difficulty": challenge["difficulty"],
            "api_execution_time": api_time,
            "lines_of_code": lines,
            "completeness": completeness,
            "flake8_issues": flake8,
            "num_execution_errors": n_errors,  # Erros/falhas NESTA execução/desafio
            "first_execution_error": first_err,
            "code": code,  # Código extraído
            "full_response": content,  # Resposta completa da API
            "execution_results_raw": raw_results,  # Resposta crua do executor/LRU worker
        }
        # Adiciona o sumário desta execução à lista GERAL self.results
        self.results.append(run_summary)

        return run_summary  # Retorna o sumário desta execução

    def _detect_function_name(self, code: str, challenge: dict) -> str | None:
        """Detecta nome da função/classe, priorizando assinatura e nome esperado."""
        if not code:
            return None

        expected_name = None
        signature = challenge.get("function_signature", "")
        sig_match = re.match(r"^\s*(?:def|class)\s+(\w+)", signature)
        if sig_match:
            expected_name = sig_match.group(1)
            # Tenta encontrar exatamente a definição esperada
            simple_def = f"def {expected_name}("
            simple_class = f"class {expected_name}:"
            simple_class_paren = (
                f"class {expected_name}("  # Com parênteses para herança
            )
            # Procura no início das linhas (mais confiável)
            if (
                re.search(rf"^\s*{re.escape(simple_def)}", code, re.MULTILINE)
                or re.search(rf"^\s*{re.escape(simple_class)}", code, re.MULTILINE)
                or re.search(
                    rf"^\s*{re.escape(simple_class_paren)}", code, re.MULTILINE
                )
            ):
                return expected_name
            # Procura definição em qualquer lugar (menos confiável mas útil se houver código antes)
            # Usa \b para garantir que é a palavra completa
            elif re.search(
                rf"\s(?:def|class)\s+{re.escape(expected_name)}\b", code, re.MULTILINE
            ):
                logging.debug(
                    f"Encontrado '{expected_name}' como def/class não no início da linha."
                )
                return expected_name
            # Fallback: Se o nome esperado existir mas não como definição, avisa mas usa
            elif f"{expected_name}(" in code or f"{expected_name} " in code:
                logging.warning(
                    f"Nome esperado '{expected_name}' encontrado, mas não como definição clara. Usando '{expected_name}'."
                )
                return expected_name

        # Se não encontrou o nome esperado, pega a primeira definição encontrada
        first_def_match = re.search(r"^\s*(?:def|class)\s+(\w+)", code, re.MULTILINE)
        if first_def_match:
            found_name = first_def_match.group(1)
            if expected_name and found_name != expected_name:
                logging.warning(
                    f"Usando 1ª definição encontrada '{found_name}' (Esperado: '{expected_name}')."
                )
            elif not expected_name:
                logging.warning(
                    f"Usando 1ª definição encontrada '{found_name}' (sem nome esperado definido)."
                )
            return found_name

        logging.error(
            f"Nenhuma definição de função ou classe encontrada para {challenge['name']}."
        )
        return None

    # SUGESTÃO 3: Ajustar comparação Two Sum em _run_test_cases_fixed
    def _run_test_cases_fixed(
        self,
        ai_name: str,
        challenge: dict,
        code: str,
        actual_function_name: str,
        run_idx: int,
    ) -> tuple[list, list, int, str]:
        """Executa casos de teste via Executor Service, incluindo run_idx nos detalhes."""
        raw_results = []  # Respostas cruas do executor
        details = []  # Resultados detalhados formatados para esta execução
        n_fails_errors = 0
        first_err_msg = ""
        test_cases = challenge.get("test_cases", [])

        if not test_cases:
            logging.warning(f"Sem casos de teste para '{challenge['name']}'.")
            return [], [], 0, ""

        challenge_name = challenge["name"]  # Nome do desafio para verificações

        for i, tc in enumerate(test_cases):
            input_args = tc.get("input_args")
            expected_output = tc.get("expected_output")

            if input_args is None:
                logging.warning(
                    f"Caso {i+1} '{challenge_name}' sem 'input_args'. Pulando."
                )
                continue

            # Chama o serviço executor
            exec_res = self._call_executor_service(
                code, actual_function_name, input_args
            )
            raw_results.append(exec_res)  # Guarda resultado cru

            # Prepara dicionário de detalhes para este caso de teste
            test_detail = {
                "run_index": run_idx,  # << ADICIONA run_index
                "ai": ai_name,
                "challenge": challenge_name,
                "difficulty": challenge["difficulty"],
                "test_case_index": i,
                "input_args_str": json.dumps(input_args),
                "expected_output_str": json.dumps(expected_output),
                "actual_output_str": None,
                "status": "Unknown",  # Default
                "execution_time_ms": exec_res.get("execution_time_ms", -1),
                "error_message": None,
                "stdout": exec_res.get("stdout"),
            }
            current_case_error = None

            if exec_res.get("success") is True:
                actual_output = exec_res.get("output")
                test_detail["actual_output_str"] = json.dumps(actual_output)
                try:
                    are_equal = False
                    # --- SUGESTÃO 3: Lógica de comparação específica para Two Sum ---
                    if (
                        challenge_name == "Two Sum"
                        and isinstance(actual_output, list)
                        and isinstance(expected_output, list)
                    ):
                        # Compara ignorando a ordem para Two Sum
                        are_equal = sorted(actual_output) == sorted(expected_output)
                        if not are_equal:
                            logging.debug(
                                f"Two Sum Fail: Expected (sorted) {sorted(expected_output)}, Got (sorted) {sorted(actual_output)}"
                            )
                    else:
                        # Comparação padrão para outros desafios
                        # Usar np.array_equal é robusto para listas, números, etc.
                        # dtype=object ajuda a lidar com tipos mistos ou aninhados
                        are_equal = np.array_equal(
                            np.array(actual_output, dtype=object),
                            np.array(expected_output, dtype=object),
                        )
                    # --- Fim da lógica específica ---

                    if are_equal:
                        test_detail["status"] = "Pass"
                    else:
                        test_detail["status"] = "Fail"
                        current_case_error = f"Output != Expected (Case {i})"
                        n_fails_errors += 1

                except Exception as comp_e:
                    logging.warning(
                        f"Erro ao comparar outputs {ai_name}/{challenge_name}/Caso {i}: {comp_e}",
                        exc_info=False,
                    )
                    test_detail["status"] = "Fail (Comparison Error)"
                    test_detail["error_message"] = f"Error comparing outputs: {comp_e}"
                    current_case_error = test_detail["error_message"]
                    n_fails_errors += 1

            elif exec_res.get("success") is False:  # Erro reportado pelo executor
                test_detail["status"] = "Error"
                test_detail["error_message"] = exec_res.get(
                    "error", "Execution error reported by service"
                )
                current_case_error = str(test_detail["error_message"])[
                    :250
                ]  # Limita tamanho
                n_fails_errors += 1
            else:  # Resposta inesperada do executor
                test_detail["status"] = "Unknown Executor Status"
                test_detail["error_message"] = exec_res.get(
                    "error", "Unexpected response from executor service"
                )
                current_case_error = str(test_detail["error_message"])[:250]
                n_fails_errors += 1

            # Guarda a primeira mensagem de erro encontrada
            if current_case_error and not first_err_msg:
                first_err_msg = current_case_error

            details.append(test_detail)  # Adiciona detalhes deste caso à lista

        # Se houve falhas mas nenhuma mensagem específica foi capturada
        if n_fails_errors > 0 and not first_err_msg:
            first_err_msg = "One or more test cases failed or errored."

        return raw_results, details, n_fails_errors, first_err_msg

    def _call_executor_service(
        self, code_string: str, function_name: str, input_args: list
    ) -> dict:
        """Chama o serviço executor externo."""
        # Adiciona imports comuns que podem ser necessários
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
            )  # Timeout de 60s
            if resp.status_code == 200:
                try:
                    return resp.json()  # Retorna a resposta JSON do serviço
                except json.JSONDecodeError as json_e:
                    logging.error(
                        f"Erro JSON decode executor: {json_e}. Resposta: {resp.text[:200]}...",
                        exc_info=False,
                    )
                    default_err["error"] = f"Executor JSON decode error: {json_e}"
                    return default_err
            else:  # Erro HTTP do serviço executor
                err_txt = resp.text[:200]  # Pega início do texto do erro
                logging.error(
                    f"Erro do Executor Service status {resp.status_code}: {err_txt}..."
                )
                default_err["error"] = (
                    f"Executor service returned status {resp.status_code}: {err_txt}"
                )
                # Tenta extrair erro JSON se possível
                try:
                    default_err["error_details"] = resp.json()
                except:
                    pass
                return default_err
        except requests.exceptions.Timeout:
            logging.error(f"Timeout ao chamar executor service ({self.executor_url}).")
            default_err["error"] = "Executor service request timeout"
            return default_err
        except requests.exceptions.RequestException as req_e:
            logging.error(
                f"Erro de conexão/rede com executor service: {req_e}", exc_info=False
            )
            default_err["error"] = f"Executor connection error: {req_e}"
            return default_err
        except Exception as e:  # Outros erros inesperados
            logging.error(
                f"Erro inesperado ao chamar executor service: {type(e).__name__} - {e}",
                exc_info=True,
            )
            default_err["error"] = (
                f"Unexpected error calling executor: {type(e).__name__}"
            )
            return default_err

    def run_benchmark(self):
        """Executa o ciclo completo do benchmark por 'self.num_runs' vezes."""
        can_claude = False  # Claude desativado por solicitação
        can_openai = bool(self.openai_api_key)
        can_gemini = bool(self.gemini_api_key and self.gemini_model)
        enabled_ais = [
            n
            for n, can in [
                
                ("ChatGPT", can_openai),
                ("Gemini", can_gemini),
            ]
            if can
        ]

        if not enabled_ais:
            logging.critical("ERRO: Nenhuma API configurada ou disponível. Saindo.")
            return

        logging.info(
            f"Iniciando Benchmark para: {', '.join(enabled_ais)} ({self.num_runs} execução(ões) por desafio)"
        )
        logging.info(f"  Claude Model: {self.claude_model if can_claude else 'N/A'}")
        logging.info(f"  ChatGPT Model: {self.openai_model if can_openai else 'N/A'}")
        logging.info(f"  Gemini Model: {self.gemini_model or 'N/A'}")
        logging.info(f"  LRU Cache Execution: Local")
        logging.info(f"  Executor Service URL: {self.executor_url}")
        logging.info("=" * 42)

        # Reinicia listas GLOBAIS antes de iniciar as execuções
        self.results = []
        self.detailed_test_results = []

        # Loop principal das execuções
        for run_idx in range(self.num_runs):
            logging.info(
                f"\n{'='*20} INICIANDO EXECUÇÃO {run_idx + 1}/{self.num_runs} {'='*20}"
            )
            run_start_time = time.time()

            for chal in self.code_challenges:
                sig = chal.get(
                    "function_signature",
                    f"{chal['name'].lower().replace(' ','_')}(...)",
                )
                # Prompt aprimorado para pedir apenas o código
                prompt = f"Implement the following Python {'class' if 'class' in sig else 'function'}.\nDescription: {chal['description']}\nSignature:\n{sig}\nProvide ONLY the raw Python code definition for the function/class and any necessary standard library imports (like `from collections import OrderedDict` if needed). Do not include any explanations, comments within the code, usage examples, or markdown formatting like ```python."

                logging.info(
                    f"\n--- [Run {run_idx+1}] Desafio: {chal['name']} ({chal['difficulty']}) ---"
                )
                # SUGESTÃO 5: Usar self.challenge_details
                logging.info(
                    f"  {self.challenge_details.get(chal['name'],'Explicação técnica N/A.')}"
                )

                responses = {}  # Respostas das APIs para este desafio/run
                if can_claude:
                    logging.info(f"  Consultando Claude AI (Run {run_idx+1})...")
                    responses["Claude"] = self.query_claude(prompt)
                    time.sleep(1)  # Pequena pausa entre APIs
                if can_openai:
                    logging.info(f"  Consultando ChatGPT (Run {run_idx+1})...")
                    responses["ChatGPT"] = self.query_chatgpt(prompt)
                    time.sleep(1)
                if can_gemini:
                    logging.info(f"  Consultando Gemini AI (Run {run_idx+1})...")
                    responses["Gemini"] = self.query_gemini(prompt)
                    # Tenacity já lida com espera/retry no query_gemini

                # Avalia as respostas e acumula resultados
                summary_lines_run = []
                for ai in enabled_ais:
                    resp_data = responses.get(
                        ai,
                        {
                            "content": "",
                            "execution_time": 0,
                            "last_error": "No Response Attempted",
                        },
                    )

                    # evaluate_solution faz a avaliação e adiciona aos GLOBAIS
                    # self.results e self.detailed_test_results
                    # Retorna o sumário APENAS desta execução/ai/challenge
                    result_summary_this_run = self.evaluate_solution(
                        ai, chal, resp_data, run_idx
                    )

                    # Prepara linha de resumo para ESTA run/desafio/ia
                    n_err_fail = result_summary_this_run.get("num_execution_errors", 0)
                    # Obtém detalhes específicos desta AI/Challenge/RUN da lista global
                    details_this_eval = [
                        d
                        for d in self.detailed_test_results
                        if d["ai"] == ai
                        and d["challenge"] == chal["name"]
                        and d["run_index"] == run_idx
                    ]

                    n_attempted_tests = len(
                        [
                            d
                            for d in details_this_eval
                            if d["status"] not in ["API Error", "No Code/No Function"]
                        ]
                    )
                    n_passed_tests = len(
                        [d for d in details_this_eval if d["status"] == "Pass"]
                    )
                    f8_issues = (
                        str(result_summary_this_run["flake8_issues"])
                        if result_summary_this_run["flake8_issues"] != -1
                        else "N/A"
                    )

                    # Define o status para o log baseado nos resultados detalhados
                    if (
                        not details_this_eval
                    ):  # Caso não haja entrada detalhada (ex: erro API)
                        status_log = result_summary_this_run.get(
                            "first_execution_error", "Unknown Error"
                        )
                    elif chal["name"] == "LRU Cache":
                        status_log = (
                            details_this_eval[0]["status"]
                            if details_this_eval
                            else "Error Evaluating"
                        )
                    elif (
                        n_attempted_tests == 0
                    ):  # Se houve entradas mas nenhuma foi executada (No Code/Func)
                        status_log = (
                            details_this_eval[0]["status"]
                            if details_this_eval
                            else "Not Run"
                        )
                    else:
                        status_log = f"{n_passed_tests}/{n_attempted_tests} passed"

                    summary_lines_run.append(
                        f"    {ai:<10}: Exec: {status_log}. Flake8: {f8_issues}. API Time: {result_summary_this_run['api_execution_time']:.2f}s."
                    )

                # Imprime o resumo para este desafio DESTA execução
                logging.info(
                    f"--- [Run {run_idx+1}] Resumo Desafio '{chal['name']}' ---"
                )
                for s_line in summary_lines_run:
                    logging.info(s_line)
                logging.info(
                    f"--- [Run {run_idx+1}] Desafio '{chal['name']}' concluído ---"
                )
                print() # <---- ADICIONA ESTA LINHA PARA PULAR UMA LINHA VISUALMENTE
                print() # <---- ADICIONA ESTA LINHA PARA PULAR UMA LINHA VISUALMENTE
                time.sleep(0.5)  # Pausa entre desafios

            run_end_time = time.time()
            logging.info(
                f"\n{'='*20} FIM EXECUÇÃO {run_idx + 1}/{self.num_runs} (Duração: {run_end_time - run_start_time:.2f}s) {'='*20}"
            )

        # Fim do loop de execuções

        if not self.results:
            logging.error("Nenhum resultado foi coletado durante o benchmark.")
            return

        # Gera relatório FINAL com dados acumulados de todas as execuções
        self.generate_report()

    # --- Métodos de Geração de Relatório (AJUSTADOS para Múltiplas Execuções) ---

    def _prepare_dataframe(self) -> pd.DataFrame | None:
        """Prepara DataFrame AGREGADO (média sobre runs) a partir de self.results."""
        if not self.results:
            logging.error(
                "Não há resultados (self.results) para gerar DataFrame agregado."
            )
            return None
        try:
            df_all_runs = pd.DataFrame(self.results)

            # Métricas para agregar pela MÉDIA sobre as execuções (runs)
            metrics_to_avg = ["api_execution_time", "lines_of_code", "completeness"]
            # Coluna Flake8 requer tratamento especial para -1 (N/A)
            # num_execution_errors também será agregado pela média

            # Agrupa por AI e Desafio
            grouped = df_all_runs.groupby(["ai", "challenge"])

            # Calcula médias simples
            df_agg = grouped[metrics_to_avg].mean()

            # Calcula média de Flake8 ignorando -1
            def avg_valid_flake8(series):
                valid = series[series != -1]
                return (
                    valid.mean() if not valid.empty else -1
                )  # Retorna -1 se só houver N/A

            df_agg["flake8_issues"] = grouped["flake8_issues"].apply(avg_valid_flake8)

            # Calcula média de erros de execução por run
            df_agg["avg_num_execution_errors"] = grouped["num_execution_errors"].mean()

            # Calcula taxa de sucesso GERAL e tempo médio de execução GERAL (sobre todos os casos/runs)
            # Usa a função _calculate_aggregated_test_metrics que lê self.detailed_test_results (com run_idx)
            agg_test_metrics = self._calculate_aggregated_test_metrics()

            # Mapeia métricas de teste agregadas para o novo DataFrame
            # O índice de df_agg é (ai, challenge), que corresponde às chaves de agg_test_metrics
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
            )
            df_agg["total_tests_attempted"] = df_agg.index.map(
                lambda idx: agg_test_metrics.get(idx, {}).get(
                    "total_tests_attempted", 0
                )
            )

            # Pega o primeiro erro encontrado em qualquer run para exemplo
            # Pega o primeiro valor não vazio na série agrupada
            def get_first_error(series):
                non_empty = series.loc[series.fillna("").astype(str) != ""].astype(str)
                return non_empty.iloc[0] if not non_empty.empty else ""

            df_agg["first_execution_error"] = grouped["first_execution_error"].apply(
                get_first_error
            )

            # Adiciona difficulty de volta (group by remove colunas não agregadas)
            # Pega difficulty do primeiro run para cada (ai, challenge)
            difficulty_map = df_all_runs.drop_duplicates(
                subset=["ai", "challenge"]
            ).set_index(["ai", "challenge"])["difficulty"]
            df_agg = df_agg.join(difficulty_map)  # Junta usando o índice multi-nível

            df_agg = (
                df_agg.reset_index()
            )  # Reset index ('ai', 'challenge' viram colunas)

            # Renomeia avg_num_execution_errors -> num_execution_errors para consistência
            df_agg = df_agg.rename(
                columns={"avg_num_execution_errors": "num_execution_errors"}
            )

            # Normaliza os tipos de dados no DataFrame agregado final
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

        # Cria um DataFrame temporário para facilitar o cálculo
        try:
            df_details = pd.DataFrame(self.detailed_test_results)
            # Converte tempo para numérico, tratando erros
            df_details["execution_time_ms"] = pd.to_numeric(
                df_details["execution_time_ms"], errors="coerce"
            )

            # Agrupa por AI e Desafio
            grouped = df_details.groupby(["ai", "challenge"])

            for name, group in grouped:
                # Filtra casos que foram efetivamente tentados (exclui API Error, No Code)
                attempted = group[
                    ~group["status"].isin(
                        ["API Error", "No Code", "No Code/No Function"]
                    )
                ]
                passed = attempted[attempted["status"] == "Pass"]

                n_attempted = len(attempted)
                n_passed = len(passed)

                # Calcula taxa de sucesso sobre os tentados
                success_rate = n_passed / n_attempted if n_attempted > 0 else 0.0

                # Calcula tempo médio apenas dos casos que passaram
                valid_times = passed["execution_time_ms"].dropna()
                avg_time = valid_times.mean() if not valid_times.empty else np.nan

                aggregated_metrics[name] = {
                    "avg_exec_success_rate": success_rate,
                    "avg_exec_time_ms": avg_time,
                    "total_tests_passed": n_passed,  # Total passado em todas as runs
                    "total_tests_attempted": n_attempted,  # Total tentado em todas as runs
                }
        except Exception as e:
            logging.error(
                f"Erro ao calcular métricas agregadas de teste: {e}", exc_info=True
            )

        return aggregated_metrics

    def _normalize_dataframe_types(self, df: pd.DataFrame):
        """Normaliza tipos de colunas numéricas e trata NaNs no DataFrame fornecido."""
        numeric_cols = {
            "api_execution_time": (float, 0.0),
            "lines_of_code": (int, 0),
            "completeness": (int, 0),
            "flake8_issues": (int, -1),  # -1 para N/A
            "num_execution_errors": (float, 0.0),  # Mantem float pois é média
            "avg_exec_success_rate": (float, 0.0),
            "avg_exec_time_ms": (float, -1.0),  # -1 ou NaN para N/A
            "total_tests_passed": (int, 0),
            "total_tests_attempted": (int, 0),
        }
        for col_name, (target_dtype, fill_value) in numeric_cols.items():
            if col_name in df.columns:
                # Converte para numérico, forçando erros para NaN
                df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
                # Preenche NaNs com o valor definido
                df.loc[:, col_name] = df[col_name].fillna(fill_value)
                # Tenta converter para o tipo desejado (int ou float)
                try:
                    if np.isnan(fill_value) and target_dtype == int:
                        # Não pode converter NaN para int, mantém float ou usa Int64 (nullable)
                        if pd.__version__ >= "1.0.0":
                            df.loc[:, col_name] = df[col_name].astype(pd.Int64Dtype())
                        else:  # Versões antigas do pandas não suportam Int64, mantém float
                            df.loc[:, col_name] = df[col_name].astype(float)
                    else:
                        df.loc[:, col_name] = df[col_name].astype(target_dtype)
                except Exception as e:
                    logging.warning(
                        f"Falha ao converter coluna '{col_name}' para {target_dtype} após fillna: {e}. Mantendo como float."
                    )
                    df.loc[:, col_name] = df[col_name].astype(
                        float
                    )  # Fallback para float
            # else: logging.debug(f"Coluna numérica esperada '{col_name}' não encontrada para normalização.")

        # Normaliza coluna de erro (string)
        if "first_execution_error" in df.columns:
            df["first_execution_error"] = (
                df["first_execution_error"].fillna("").astype(str)
            )
        elif (
            "num_execution_errors" in df.columns
        ):  # Adiciona coluna vazia se não existir
            df["first_execution_error"] = ""

    def _save_environment_info(self, timestamp_str: str):
        """Salva info do ambiente em TXT."""
        filepath = os.path.join(
            self.output_dir, f"environment_info_{timestamp_str}.txt"
        )
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                # Adiciona o número de runs ao início da info
                f.write(f"Number of Runs per Challenge/AI: {self.num_runs}\n")
                f.write("-" * 30 + "\n")
                for key, value in self.environment_info.items():
                    # Evita duplicar a info do num_runs que já estava no dict
                    if key == "Num Runs":
                        continue
                    if isinstance(value, dict):
                        f.write(f"{key}:\n")
                        for sub_key, sub_value in value.items():
                            f.write(f"  {sub_key}: {sub_value}\n")
                    else:
                        f.write(f"{key}: {value}\n")
            logging.info(f"Informações do Ambiente salvas: {filepath}")
        except Exception as e:
            logging.error(f"Erro ao salvar informações do ambiente: {e}", exc_info=True)

    def _save_full_results_json(self, timestamp_str: str):
        """Salva resultados completos (run-a-run, de self.results) em JSON."""
        filepath = os.path.join(
            self.output_dir, f"benchmark_results_all_runs_{timestamp_str}.json"
        )
        try:
            # Encoder customizado para lidar com tipos numpy, datetime, etc.
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
                        return obj.decode("utf-8", "replace")  # Tenta decodificar bytes
                    return super(NpEncoder, self).default(obj)

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(
                    self.results, f, indent=2, ensure_ascii=False, cls=NpEncoder
                )  # Salva self.results
            logging.info(f"Resultados JSON (todos os runs) salvos: {filepath}")
        except Exception as e:
            logging.error(
                f"Erro ao salvar JSON completo (todos os runs): {e}", exc_info=True
            )

    def _save_detailed_tests_csv(self, timestamp_str: str):
        """Salva detalhes dos casos de teste individuais (todos os runs, de self.detailed_test_results) em CSV."""
        filepath = os.path.join(
            self.output_dir, f"benchmark_test_details_all_runs_{timestamp_str}.csv"
        )
        if self.detailed_test_results:
            try:
                df = pd.DataFrame(self.detailed_test_results)
                # Define e ordena colunas para o CSV
                cols_order = [
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
                # Garante que todas as colunas existem, preenchendo com NA se necessário
                for col in cols_order:
                    if col not in df.columns:
                        df[col] = pd.NA
                # Salva CSV
                df[cols_order].to_csv(
                    filepath, index=False, encoding="utf-8", float_format="%.3f"
                )
                logging.info(
                    f"Detalhes dos Testes CSV (todos os runs) salvos: {filepath}"
                )
            except Exception as e:
                logging.error(
                    f"Erro ao salvar CSV de detalhes dos testes: {e}", exc_info=True
                )
        else:
            logging.warning("Nenhum resultado detalhado de teste para salvar em CSV.")

    def _create_csv_header(self) -> str:
        """Cria cabeçalho com info do ambiente (incluindo num_runs) para CSVs agregados."""
        header_lines = ["# --- Environment Info ---"]
        header_lines.append(f"# Number of Runs per Challenge/AI: {self.num_runs}")
        for key, value in self.environment_info.items():
            if key == "Num Runs":
                continue  # Já adicionado
            if isinstance(value, dict):
                header_lines.append(f"# {key}:")
                [header_lines.append(f"#   {sk}: {sv}") for sk, sv in value.items()]
            else:
                header_lines.append(f"# {key}: {value}")
        header_lines.append("# --- Benchmark Data (Aggregated/Averaged over Runs) ---")
        return "\n".join(header_lines) + "\n"  # Adiciona linha extra no final

    def _save_summary_csv(self, df_agg: pd.DataFrame, timestamp_str: str):
        """Salva resumo agregado (média sobre runs) em CSV."""
        filepath = os.path.join(
            self.output_dir, f"benchmark_summary_agg_{timestamp_str}.csv"
        )
        if df_agg is None:
            logging.error(
                "DataFrame agregado é None. Pulando salvamento do resumo CSV."
            )
            return
        try:
            csv_header = self._create_csv_header()
            with open(filepath, "w", encoding="utf-8", newline="") as f:
                f.write(csv_header)  # Escreve cabeçalho comentado
                df_agg.to_csv(
                    f,
                    index=False,
                    encoding="utf-8",
                    float_format="%.4f",
                    lineterminator="\n",
                )
            logging.info(f"Resumo Agregado CSV salvo: {filepath}")
        except Exception as e:
            logging.error(f"Erro ao salvar resumo agregado CSV: {e}", exc_info=True)

    def _calculate_statistics(self, df_agg: pd.DataFrame) -> pd.DataFrame | None:
        """Calcula estatísticas gerais agregadas por IA a partir do DataFrame JÁ AGREGADO (média sobre runs) por desafio."""
        if df_agg is None:
            logging.error("DataFrame agregado de entrada para estatísticas é None.")
            return None
        try:
            # Define métricas a calcular: (Nome Exibição, Coluna Original df_agg, Tipo Agregação, [Opções])
            metrics_definitions = [
                ("Avg API Time (s)", "api_execution_time", "mean"),
                ("Std Dev API Time (s)", "api_execution_time", "std"),
                ("Avg LoC", "lines_of_code", "mean"),
                ("Std Dev LoC", "lines_of_code", "std"),
                ("Avg Completeness (%)", "completeness", "mean"),
                ("Avg Flake8 Issues", "flake8_issues", "mean_valid", {"ignore": -1}),
                ("Total Passed (All Runs)", "total_tests_passed", "sum"),
                ("Total Attempted (All Runs)", "total_tests_attempted", "sum"),
                (
                    "Overall Success Rate (%)",
                    "avg_exec_success_rate",
                    "mean_percentage",
                ),
                ("Std Dev Success Rate (%)", "avg_exec_success_rate", "std_percentage"),
                (
                    "Avg Exec Time (ms)",
                    "avg_exec_time_ms",
                    "mean_valid",
                    {"ignore": -1.0},
                ),  # Ignora -1.0 ou NaN
                ("Avg Errors/Fails per Challenge", "num_execution_errors", "mean"),
                ("Example Error", "first_execution_error", "first_non_empty"),
            ]

            ais = sorted(df_agg["ai"].unique())
            stats_data = {
                "Metric": [m[0] for m in metrics_definitions]
            }  # Coluna com nomes das métricas

            for ai in ais:
                df_ai = df_agg[df_agg["ai"] == ai]  # Filtra dados para a IA atual
                ai_column_data = []
                for (
                    metric_name,
                    col_key,
                    agg_type,
                    *options_list,
                ) in metrics_definitions:
                    value = np.nan  # Valor padrão
                    options = (
                        options_list[0] if options_list else {}
                    )  # Pega dict de opções se existir
                    ignore_val = options.get("ignore")

                    if col_key in df_ai.columns:
                        series = df_ai[col_key].copy()  # Trabalha com cópia da série
                        # Remove valores a ignorar ANTES de calcular
                        if ignore_val is not None:
                            series = series[series != ignore_val]
                        # Remove NaNs para cálculos (exceto para first_non_empty)
                        if agg_type != "first_non_empty":
                            series = series.dropna()

                        if not series.empty:
                            if agg_type == "mean":
                                value = series.mean()
                            elif agg_type == "std":
                                value = series.std()
                            elif agg_type == "sum":
                                value = series.sum()
                            # mean_valid já foi tratado pela remoção de ignore_val e dropna
                            elif agg_type == "mean_valid":
                                value = series.mean() if not series.empty else np.nan
                            elif agg_type == "mean_percentage":
                                value = series.mean() * 100
                            elif agg_type == "std_percentage":
                                value = series.std() * 100
                            elif agg_type == "first_non_empty":
                                # Usa a série original (com NaNs) para first_non_empty
                                errors = df_ai[col_key][
                                    df_ai[col_key].fillna("").astype(str) != ""
                                ]
                                value = errors.iloc[0] if not errors.empty else ""
                        elif (
                            agg_type == "first_non_empty"
                        ):  # Se série original era vazia ou só tinha vazios
                            value = ""
                        # else: value permanece np.nan se a série filtrada ficou vazia

                    ai_column_data.append(value)
                stats_data[ai] = ai_column_data  # Adiciona coluna da IA ao dicionário

            return pd.DataFrame(stats_data)

        except Exception as e:
            logging.error(f"Erro ao calcular estatísticas gerais: {e}", exc_info=True)
            return None

    def _save_statistics_csv(self, stats_df: pd.DataFrame, timestamp_str: str):
        """Salva estatísticas gerais agregadas em CSV."""
        if stats_df is None:
            logging.error("DataFrame de estatísticas é None. Pulando salvamento CSV.")
            return
        filepath = os.path.join(
            self.output_dir, f"benchmark_stats_agg_{timestamp_str}.csv"
        )
        try:
            csv_header = self._create_csv_header()  # Cabeçalho indica agregação
            with open(filepath, "w", encoding="utf-8", newline="") as f:
                f.write(csv_header)
                stats_df.to_csv(
                    f,
                    index=False,
                    encoding="utf-8",
                    float_format="%.2f",
                    lineterminator="\n",
                )
            logging.info(f"Estatísticas Gerais CSV salvas: {filepath}")
        except Exception as e:
            logging.error(f"Erro ao salvar estatísticas CSV: {e}", exc_info=True)

    def _display_summary_table(self, stats_df: pd.DataFrame):
        """Mostra tabela de resumo formatada no console (requer tabulate)."""
        # Usa print aqui para formatação direta no console
        print(
            f"\n===== SUMÁRIO DO BENCHMARK (ESTATÍSTICAS GERAIS - {self.num_runs} RUNS) ====="
        )
        if stats_df is None or stats_df.empty:
            print(
                "Não foi possível gerar sumário (DataFrame de estatísticas vazio ou None)."
            )
            return

        # Prepara DataFrame para exibição
        display_df = stats_df.copy()
        # Formata valores numéricos e trata NaNs/Erros para exibição
        for col_name in display_df.columns:
            if col_name == "Metric":
                continue  # Pula a coluna de métricas

            formatted_column = []
            for index, row in display_df.iterrows():
                metric = row["Metric"]
                value = row[col_name]
                fmt_value = "N/A"  # Default

                if not pd.isna(value):
                    try:
                        if "Std Dev" in metric:
                            if "(%)" in metric:
                                fmt_value = f"±{value:.1f}%"
                            elif "(s)" in metric:
                                fmt_value = f"±{value:.2f}s"
                            else:
                                fmt_value = f"±{value:.1f}"
                        elif "Rate (%)" in metric or "Completeness (%)" in metric:
                            fmt_value = f"{value:.1f}%"
                        elif "Flake8 Issues" in metric:
                            fmt_value = f"{value:.1f}" if value != -1 else "N/A"
                        elif "Time (ms)" in metric:
                            fmt_value = f"{value:.1f}" if value != -1 else "N/A"
                        elif "Total " in metric:
                            fmt_value = f"{int(value)}"
                        elif "Time (s)" in metric:
                            fmt_value = f"{value:.2f}"
                        elif "Avg Errors/Fails" in metric:
                            fmt_value = f"{value:.2f}"
                        elif "Example Error" in metric:
                            s = str(value)
                            fmt_value = s[:60] + ("..." if len(s) > 60 else "")
                        else:  # Default para LoC, etc.
                            fmt_value = f"{value:.1f}"
                    except (
                        ValueError,
                        TypeError,
                    ):  # Caso o valor não seja numérico como esperado
                        fmt_value = str(value)[:60]  # Mostra como string (truncado)

                formatted_column.append(fmt_value)
            display_df[col_name] = (
                formatted_column  # Substitui coluna pelos valores formatados
            )

        # Tenta usar tabulate para tabela bonita, fallback para print simples
        try:
            from tabulate import tabulate

            print(
                tabulate(display_df, headers="keys", tablefmt="grid", showindex=False)
            )
        except ImportError:
            logging.warning(
                "Biblioteca 'tabulate' não instalada. Tabela de resumo será exibida sem formatação."
            )
            print(display_df.to_string(index=False))
        except Exception as e_tab:
            logging.error(
                f"Erro ao imprimir tabela com tabulate: {e_tab}. Usando print fallback."
            )
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

        # 1. Salva Informações do Ambiente
        self._save_environment_info(ts)

        # 2. Prepara DataFrame AGREGADO (média sobre runs)
        df_agg = self._prepare_dataframe()
        if df_agg is None:
            logging.error(
                "Falha ao preparar DataFrame agregado. Abortando geração de relatórios restantes."
            )
            return

        # 3. Salva os diferentes arquivos de dados
        self._save_full_results_json(ts)  # Todos os runs (sumário por run)
        self._save_detailed_tests_csv(ts)  # Todos os runs (detalhes por teste)
        self._save_summary_csv(df_agg, ts)  # Agregado (média sobre runs)

        # 4. Calcula e salva estatísticas GERAIS (baseadas no df_agg)
        stats_df = self._calculate_statistics(df_agg)
        if stats_df is not None:
            self._save_statistics_csv(stats_df, ts)  # Salva estatísticas agregadas

            # 5. Gera visualizações (baseadas no df_agg e df_details para LRU)
            try:
                logging.info(
                    "\n--- Gerando Visualizações (baseadas em médias sobre runs) ---"
                )
                self.create_visualizations(
                    df_agg, ts, self.output_dir
                )  # Passa o df AGREGADO
                logging.info(f"Visualizações salvas em '{self.output_dir}/'")
            except Exception as e:
                logging.error(f"Erro ao gerar visualizações: {e}", exc_info=True)

            # 6. Exibe a tabela de sumário no console
            self._display_summary_table(stats_df)
        else:
            logging.error("Não foi possível calcular/salvar estatísticas gerais.")
            logging.warning(
                "Tabela de sumário e visualizações podem não ter sido geradas."
            )

    # --- Funções de Plotagem (COM CORREÇÃO E DEBUG LOGGING) ---

    # CORRIGIDO para erro 'ha' em tick_params
    def _plot_bar_charts(self, df_plot, ai_list, cmap, challenge_order, ts, out_dir):
        """Gera gráficos de barras comparativos por desafio (usando médias sobre runs)."""
        logging.info("Gerando Gráficos de Barras (média por desafio)...")
        metrics_to_plot = {
            "1_api_t": ("API Time (Avg)", "Time (s)", "api_execution_time"),
            "2_loc": ("Lines of Code (Avg)", "Lines", "lines_of_code"),
            "3_comp": ("Completeness Score (Avg)", "Score (%)", "completeness"),
            "4_f8": ("Flake8 Issues (Avg)", "Avg Issues", "flake8_issues"),
            "7_err": (
                "Execution Errors/Fails (Avg)",
                "Avg Errors/Fails per Challenge",
                "num_execution_errors",
            ),
        }
        if (
            "avg_exec_success_rate" in df_plot.columns
            and df_plot["avg_exec_success_rate"].notna().any()
        ):
            metrics_to_plot["5_ok"] = (
                "Execution Success Rate (Avg)",
                "Success Rate (%)",
                "avg_exec_success_rate",
            )
        if (
            "avg_exec_time_ms" in df_plot.columns
            and (
                df_plot["avg_exec_time_ms"].notna()
                & (df_plot["avg_exec_time_ms"] != -1)
            ).any()
        ):
            metrics_to_plot["6_exec_t"] = (
                "Execution Time (Avg)",
                "Time (ms)",
                "avg_exec_time_ms",
            )

        metrics_to_plot = dict(sorted(metrics_to_plot.items()))

        for key, (title, ylabel, col_name) in metrics_to_plot.items():
            if col_name not in df_plot.columns:
                logging.warning(
                    f"Pulando gráfico de barras '{title}': Coluna '{col_name}' não encontrada."
                )
                continue

            data_for_plot = df_plot.copy()
            if col_name == "flake8_issues":
                data_for_plot = data_for_plot[data_for_plot[col_name] != -1]
            elif col_name == "avg_exec_time_ms":
                data_for_plot = data_for_plot[
                    data_for_plot[col_name].notna() & (data_for_plot[col_name] != -1)
                ]
            elif col_name == "num_execution_errors":
                data_for_plot = data_for_plot[data_for_plot[col_name].notna()]
            elif col_name == "avg_exec_success_rate":
                data_for_plot[col_name] = data_for_plot[col_name] * 100

            if data_for_plot.empty or data_for_plot[col_name].isnull().all():
                logging.warning(
                    f"Pulando gráfico de barras '{title}': Sem dados válidos após filtro para '{col_name}'."
                )
                continue

            fig, ax = plt.subplots()  # Cria figura e eixo ANTES de plotar
            try:
                pivot_table = data_for_plot.pivot_table(
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
                    logging.warning(
                        f"Pulando gráfico de barras '{title}': Tabela pivot só contém NaNs."
                    )
                    plt.close(fig)  # Fecha a figura criada
                    continue

                num_ais = len(pivot_table.columns)
                pivot_table.plot(
                    kind="bar",
                    figsize=(max(12, num_ais * 2), 7),
                    color=[cmap.get(ai, "grey") for ai in pivot_table.columns],
                    width=0.8,
                    ax=ax,
                )

                ax.set_title(f"{title} por Desafio (Média sobre {self.num_runs} runs)")
                ax.set_ylabel(ylabel)
                ax.set_xlabel("Desafio")

                # --- DEBUG LOGGING ADICIONADO ---
                logging.debug(f"Bar chart '{title}': Configurando tick_params...")
                ax.tick_params(
                    axis="x", rotation=45, labelsize=9
                )  # Linha corrigida (sem ha, rotation_mode)
                logging.debug(
                    f"Bar chart '{title}': tick_params configurado. Configurando setp..."
                )
                plt.setp(
                    ax.get_xticklabels(), ha="right", rotation_mode="anchor"
                )  # Linha adicionada
                logging.debug(f"Bar chart '{title}': setp configurado.")
                # --- FIM DO DEBUG LOGGING ---

                ax.tick_params(axis="y", labelsize=9)

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
                plot_filename = f"{key}_bar_chart_avg_{ts}.png"
                filepath = os.path.join(out_dir, plot_filename)
                plt.savefig(filepath)

            except Exception as e:
                logging.error(
                    f"Erro ao gerar gráfico de barras para '{title}': {e}",
                    exc_info=True,
                )
            finally:
                plt.close(fig)  # Garante que fecha a figura

    def _plot_lru_times_chart(self, detailed_df, ai_list, cmap, ts, out_dir):
        """Gera gráfico de barras para o tempo médio de operação do LRU Cache (média sobre runs)."""
        logging.info("Gerando Gráfico de Tempo Médio - LRU Cache (média sobre runs)...")
        challenge_name = "LRU Cache"

        if detailed_df is None or detailed_df.empty:
            logging.warning(
                f"Pulando gráfico LRU Times: DataFrame detalhado vazio ou None."
            )
            return

        lru_data = detailed_df[
            (detailed_df["challenge"] == challenge_name)
            & (detailed_df["execution_time_ms"].notna())
            & (detailed_df["execution_time_ms"] >= 0)
        ].copy()

        if lru_data.empty:
            logging.warning(
                f"Pulando gráfico LRU Times: Sem dados válidos para '{challenge_name}'."
            )
            return

        avg_lru_times_per_ai = lru_data.groupby("ai")["execution_time_ms"].mean()
        avg_lru_times_per_ai = avg_lru_times_per_ai.reindex(ai_list).dropna()

        if avg_lru_times_per_ai.empty:
            logging.warning(
                f"Pulando gráfico LRU Times: Sem dados válidos após agrupar/reindexar por IA."
            )
            return

        fig, ax = plt.subplots(figsize=(max(8, len(avg_lru_times_per_ai) * 1.5), 6))
        try:
            colors_for_plot = [
                cmap.get(ai, "grey") for ai in avg_lru_times_per_ai.index
            ]
            avg_lru_times_per_ai.plot(
                kind="bar", ax=ax, color=colors_for_plot, width=0.6
            )

            ax.set_title(
                f"Tempo Médio por Operação - {challenge_name} (Média sobre {self.num_runs} runs)"
            )
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
            plot_filename = f"8_lru_exec_time_avg_{ts}.png"
            filepath = os.path.join(out_dir, plot_filename)
            plt.savefig(filepath)
            logging.info(f"Gráfico Tempos LRU (Média) salvo: {filepath}")

        except Exception as e:
            logging.error(f"Erro ao gerar gráfico de tempos LRU: {e}", exc_info=True)
        finally:
            plt.close(fig)  # Garante que fecha a figura

    # CORRIGIDO para erro 'ha' em tick_params
    def _plot_box_plots(self, df_agg, ai_list, cmap, ts, out_dir):
        """Gera box plots para distribuição das métricas MÉDIAS por desafio."""
        logging.info("Gerando Box Plots (distribuição das médias por desafio)...")
        df_plot = df_agg.copy()

        metrics_to_plot = {
            "api_execution_time": {
                "title": "Distribuição API Time (Média)",
                "ylabel": "Tempo(s)",
                "filter": None,
            },
            "lines_of_code": {
                "title": "Distribuição LoC (Média)",
                "ylabel": "Linhas",
                "filter": lambda d, c: d[c] > 0,
            },
            "completeness": {
                "title": "Distribuição Completude (Média)",
                "ylabel": "Score (%)",
                "filter": None,
            },
        }
        if (
            "flake8_issues" in df_plot.columns
            and (df_plot["flake8_issues"] != -1).any()
        ):
            metrics_to_plot["flake8_issues"] = {
                "title": "Distribuição Flake8 Issues (Média)",
                "ylabel": "Issues",
                "filter": lambda d, c: d[c] != -1,
            }
        if (
            "avg_exec_time_ms" in df_plot.columns
            and (
                df_plot["avg_exec_time_ms"].notna()
                & (df_plot["avg_exec_time_ms"] != -1)
            ).any()
        ):
            metrics_to_plot["avg_exec_time_ms"] = {
                "title": "Distribuição Exec Time (Média)",
                "ylabel": "Tempo (ms)",
                "filter": lambda d, c: d[c].notna() & (d[c] != -1),
            }
        if (
            "num_execution_errors" in df_plot.columns
            and df_plot["num_execution_errors"].notna().any()
        ):
            if (df_plot["num_execution_errors"] > 0).any():
                metrics_to_plot["num_execution_errors"] = {
                    "title": "Distribuição Erros/Falhas Exec (Média)",
                    "ylabel": "Avg Num Errors/Fails",
                    "filter": lambda d, c: d[c].notna() & (d[c] > 0),
                }
            else:
                logging.info(
                    "Pulando box plot 'Distribuição Erros/Falhas Exec': Nenhum erro médio > 0 registrado."
                )

        plot_counter = 8
        if not metrics_to_plot:
            logging.warning("Nenhuma métrica válida definida para box plots.")
            return

        for col_name, config in metrics_to_plot.items():
            plot_counter += 1
            if col_name not in df_plot.columns:
                logging.warning(
                    f"Pulando box plot '{config['title']}': Coluna '{col_name}' N/A no df_plot."
                )
                continue

            temp_df = df_plot.copy()
            filter_func = config.get("filter")
            if filter_func:
                try:
                    temp_df = temp_df[filter_func(temp_df, col_name)]
                except Exception as e_filter:
                    logging.warning(
                        f"Erro ao aplicar filtro para box plot '{col_name}': {e_filter}. Usando dados não filtrados."
                    )

            if temp_df.empty or temp_df[col_name].isnull().all():
                logging.warning(
                    f"Pulando box plot '{config['title']}': Sem dados válidos após filtro para '{col_name}'."
                )
                continue

            plot_data = []
            plot_labels = []
            valid_ai_data_found = False
            for ai in ai_list:
                ai_data = temp_df[temp_df["ai"] == ai][col_name].dropna()
                if not ai_data.empty:
                    plot_data.append(ai_data.values)
                    plot_labels.append(ai)
                    valid_ai_data_found = True

            if not valid_ai_data_found:
                logging.warning(
                    f"Pulando box plot '{config['title']}': Sem dados válidos por IA após filtro/dropna para '{col_name}'."
                )
                continue

            fig, ax = plt.subplots(figsize=(max(8, len(plot_labels) * 1.2), 6))
            try:
                bp = ax.boxplot(
                    plot_data,
                    tick_labels=plot_labels,
                    patch_artist=True,
                    showfliers=True,
                )

                ax.set_title(config["title"], fontsize=14)
                ax.set_ylabel(config["ylabel"], fontsize=10)
                ax.set_xlabel("IA", fontsize=10)

                # --- ADICIONADO LOGGING DEBUG ---
                logging.debug(
                    f"Box plot '{config['title']}': Configurando tick_params..."
                )
                ax.tick_params(
                    axis="x", rotation=30, labelsize=9
                )  # Linha corrigida (sem ha, rotation_mode)
                logging.debug(
                    f"Box plot '{config['title']}': tick_params configurado. Configurando setp..."
                )
                plt.setp(
                    ax.get_xticklabels(), ha="right", rotation_mode="anchor"
                )  # Linha adicionada
                logging.debug(f"Box plot '{config['title']}': setp configurado.")
                # --- FIM DO LOGGING DEBUG ---

                ax.tick_params(axis="y", labelsize=9)
                ax.grid(axis="y", linestyle="--", alpha=0.7)

                colors = [cmap.get(lbl, "grey") for lbl in plot_labels]
                for patch, color in zip(bp["boxes"], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.6)

                plt.tight_layout()
                plot_filename = f"{plot_counter}_boxplot_avg_{col_name}_{ts}.png"
                filepath = os.path.join(out_dir, plot_filename)
                plt.savefig(filepath)
                logging.debug(f"Gráfico box plot salvo: {filepath}")

            except Exception as e:
                logging.error(
                    f"Erro ao gerar box plot para '{config['title']}': {e}",
                    exc_info=True,
                )
            finally:
                plt.close(fig)  # Garante que fecha a figura

        logging.info("Geração Box Plots concluída.")

    def _plot_radar_chart(self, df_agg, ai_list, cmap, ts, out_dir):
        """Gera gráfico de radar comparativo (Normalizado, baseado em médias sobre runs)."""
        n_ais = len(ai_list)
        if n_ais < 2:
            logging.warning("Gráfico de radar pulado (< 2 IAs para comparar).")
            return

        logging.info(f"Gerando gráfico de radar para {n_ais} IAs...")
        df = df_agg.copy()

        radar_cols_def = {
            "api_execution_time": {
                "label": "Veloc. API (Norm)",
                "lower_is_better": True,
            },
            "lines_of_code": {"label": "Conc. LoC (Norm)", "lower_is_better": True},
            "completeness": {"label": "Completude (%)", "lower_is_better": False},
            "flake8_issues": {
                "label": "Qualid. Flake8 (Norm)",
                "lower_is_better": True,
                "ignore_val": -1,
            },
            "avg_exec_success_rate": {
                "label": "Tx. Sucesso Exec (%)",
                "lower_is_better": False,
            },
        }
        valid_radar_cols = {k: v for k, v in radar_cols_def.items() if k in df.columns}

        if len(valid_radar_cols) < 3:
            logging.warning(
                f"Gráfico de radar pulado (< 3 métricas válidas encontradas no DF: {list(valid_radar_cols.keys())})."
            )
            return

        categories = [v["label"] for v in valid_radar_cols.values()]
        n_cats = len(categories)
        means = df.groupby("ai")[list(valid_radar_cols.keys())].mean()
        norm_data = pd.DataFrame(index=means.index)

        for col, config in valid_radar_cols.items():
            if col == "completeness":
                norm_series = means[col].clip(0, 100)
            elif col == "avg_exec_success_rate":
                norm_series = (means[col] * 100).clip(0, 100)
            else:
                series = means[col].copy()
                ignore_val = config.get("ignore_val")
                if ignore_val is not None:
                    series = series[series != ignore_val]

                if series.empty or series.isnull().all():
                    norm_series = pd.Series(np.nan, index=means.index)
                else:
                    min_val, max_val = series.min(), series.max()
                    range_val = max_val - min_val if max_val > min_val else 1.0

                    if config["lower_is_better"]:
                        norm_series = 100 * (max_val - means[col]) / range_val
                    else:
                        norm_series = 100 * (means[col] - min_val) / range_val

                    if max_val == min_val:
                        norm_series = 0.0 if config["lower_is_better"] else 100.0

                fill_value = 0.0 if config.get("lower_is_better", False) else 100.0
                norm_series = norm_series.reindex(means.index).fillna(fill_value)

            norm_data[config["label"]] = norm_series.clip(0, 100)

        angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
        try:
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            ax.set_yticks(np.arange(0, 101, 20))
            ax.set_ylim(0, 100)

            for idx, ai in enumerate(norm_data.index):
                if ai not in ai_list:
                    continue
                values = norm_data.loc[ai].values.flatten().tolist()
                values += values[:1]
                color = cmap.get(ai, plt.plt.get_cmap('tab10')(idx / max(1, n_ais)))
                ax.plot(angles, values, "o-", linewidth=2, label=ai, color=color)
                ax.fill(angles, values, color, alpha=0.2)

            plt.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15))
            plt.title("Comparação Geral Normalizada (Maior = Melhor)", size=14, y=1.18)
            plt.tight_layout()
            plot_filename = f"9_radar_chart_avg_{ts}.png"
            filepath = os.path.join(out_dir, plot_filename)
            plt.savefig(filepath)
            logging.info(f"Gráfico Radar salvo: {filepath}")

        except Exception as e:
            logging.error(f"Erro ao gerar gráfico de radar: {e}", exc_info=True)
        finally:
            plt.close(fig)  # Garante que fecha a figura

    def create_visualizations(self, df_agg, ts, out_dir):
        """Cria e salva visualizações (baseadas em médias sobre runs)."""
        # --- DEBUG: Imprimir versão do Matplotlib ---
        try:
            # Usar logging.debug para só aparecer com -v
            logging.debug(f"Versão Matplotlib: {plt.__version__}")
        except Exception as e:
            logging.warning(f"Não foi possível obter a versão do Matplotlib: {e}")
        # --- FIM DEBUG ---

        if df_agg is None or df_agg.empty:
            logging.warning(
                "DataFrame agregado vazio ou None para gráficos. Pulando visualizações."
            )
            return

        try:
            plt.style.use("ggplot")
        except:
            logging.warning("Estilo 'ggplot' não encontrado, usando default.")

        ais_in_data = sorted(df_agg["ai"].unique())
        n_ais = len(ais_in_data)
        if n_ais == 0:
            logging.warning("Nenhuma AI encontrada nos dados agregados para plotar.")
            return

        colors = (
            plt.plt.get_cmap('viridis')(np.linspace(0, 1, n_ais))
            if n_ais > 10
            else plt.plt.get_cmap('tab10')(np.linspace(0, 1, n_ais))
        )
        cmap = {ai: colors[i] for i, ai in enumerate(ais_in_data)}

        required_cols = ["challenge", "ai", "difficulty"]
        if not all(c in df_agg.columns for c in required_cols):
            missing = [c for c in required_cols if c not in df_agg.columns]
            logging.error(
                f"Faltando colunas essenciais no DF agregado: {missing}. Pulando visualizações."
            )
            return

        challenge_order = sorted(df_agg["challenge"].unique())
        if "difficulty" in df_agg.columns:
            try:
                difficulty_order = pd.CategoricalDtype(
                    ["Easy", "Medium", "Hard"], ordered=True
                )
                # Usa cópia para não modificar df_agg original permanentemente
                df_temp_sort = df_agg.copy()
                df_temp_sort["diff_cat"] = df_temp_sort["difficulty"].astype(
                    difficulty_order
                )
                challenge_order = df_temp_sort.sort_values("diff_cat")[
                    "challenge"
                ].unique()
            except Exception as e_cat:
                logging.warning(
                    f"Erro ao ordenar desafios por dificuldade ({e_cat}). Usando ordem alfabética."
                )

        detailed_df_lru = None
        if self.detailed_test_results:
            try:
                detailed_df_lru = pd.DataFrame(self.detailed_test_results)
                detailed_df_lru["execution_time_ms"] = pd.to_numeric(
                    detailed_df_lru["execution_time_ms"], errors="coerce"
                )
            except Exception as e_prep:
                logging.warning(
                    f"Erro ao preparar DataFrame detalhado para plot LRU: {e_prep}. Gráfico LRU pode falhar."
                )
                detailed_df_lru = None
        else:
            logging.warning(
                "Sem dados detalhados (`detailed_test_results`) para plot LRU."
            )

        # Chama as funções de plotagem
        self._plot_bar_charts(df_agg, ais_in_data, cmap, challenge_order, ts, out_dir)
        self._plot_box_plots(df_agg, ais_in_data, cmap, ts, out_dir)
        self._plot_radar_chart(df_agg, ais_in_data, cmap, ts, out_dir)

        if detailed_df_lru is not None:
            self._plot_lru_times_chart(detailed_df_lru, ais_in_data, cmap, ts, out_dir)


# --- Ponto de Entrada ---
if __name__ == "__main__":
    # SUGESTÃO 2: Usar argparse para configuração
    parser = argparse.ArgumentParser(
        description="Executa um benchmark de programação comparando modelos de IA (Claude, ChatGPT, Gemini).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # Mostra defaults na ajuda
    )
    parser.add_argument(
        "-n",
        "--num_runs",
        type=int,
        default=3,
        help="Número de vezes que cada desafio/IA é executado.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="benchmark_reports",
        help="Diretório para salvar os relatórios e gráficos.",
    )
    parser.add_argument(
        "--claude_model",
        type=str,
        default=None,  # Default None para pegar do .env ou fallback
        help="Nome do modelo Claude a ser usado (sobrescreve .env e fallback).",
    )
    parser.add_argument(
        "--openai_model",
        type=str,
        default=None,
        help="Nome do modelo OpenAI (ChatGPT) a ser usado (sobrescreve .env e fallback).",
    )
    parser.add_argument(
        "--gemini_model",
        type=str,
        default=None,
        help="Nome do modelo Gemini a ser usado (sobrescreve .env e fallback).",
    )
    parser.add_argument(
        "--executor_url",
        type=str,
        default=None,
        help="URL do serviço externo para execução de código (sobrescreve .env e fallback).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Ativa logging mais detalhado (nível DEBUG).",
    )

    args = parser.parse_args()

    # SUGESTÃO 1: Ajusta nível de logging se -v for usado
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug("Logging DEBUG ativado.")

    # --- Fim do setup do argparse ---

    print("=" * 59)  # Use print para o título principal
    print(" AI Programming Benchmark: Claude vs ChatGPT vs Gemini")
    print("        (Análise Flake8, Execução Externa/Local)")
    print("=" * 59)

    # Necessário para multiprocessing no Windows se congelado
    multiprocessing.freeze_support()

    if not os.path.exists(".env"):
        logging.warning(
            "Ficheiro .env não encontrado. Chaves API podem não ser carregadas se não estiverem no ambiente."
        )

    try:
        # Cria instância do benchmark passando argumentos
        benchmark = AIBenchmark(
            num_runs=args.num_runs,
            output_dir=args.output_dir,
            claude_model_name=args.claude_model,
            openai_model_name=args.openai_model,
            gemini_model_name=args.gemini_model,
            executor_service_url=args.executor_url,
        )

        # Verifica se alguma API está disponível antes de rodar
        if (
            benchmark.claude_api_key
            or benchmark.openai_api_key
            or benchmark.gemini_api_key
        ):
            benchmark.run_benchmark()
            logging.info("\nBenchmark concluído.")
            logging.info(f"Relatórios salvos em: '{benchmark.output_dir}'")
        else:
            logging.critical(
                "\nERRO FATAL: Nenhuma chave API encontrada ou configurada corretamente."
            )
            logging.critical(
                "Verifique o seu ficheiro .env ou as variáveis de ambiente: CLAUDE_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY."
            )

    except Exception as e:
        logging.critical(
            "\nERRO INESPERADO DURANTE A EXECUÇÃO DO BENCHMARK:", exc_info=True
        )
        # Traceback já é logado pelo logging.critical com exc_info=True

    print("=" * 59)  # Use print para a linha final
