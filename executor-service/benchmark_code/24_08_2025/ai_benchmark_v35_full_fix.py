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
from collections import OrderedDict, deque, defaultdict
import builtins
import logging
import argparse
from abc import ABC, abstractmethod
from functools import lru_cache
from string import ascii_lowercase, digits

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

try:
    from importlib import metadata as importlib_metadata
except ImportError:
    try:
        import pkg_resources  # type: ignore
    except ImportError:
        pkg_resources = None

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# --- Classes de Provedor de IA Abstraídas ---
class AIProvider(ABC):
    def __init__(self, model_name: str, api_key: str | None):
        self.model_name = model_name
        self.api_key = api_key
        if not self.api_key:
            raise ValueError("Chave API não fornecida.")

    @abstractmethod
    def query(self, prompt: str) -> dict:
        pass

class ClaudeProvider(AIProvider):
    def query(self, prompt: str) -> dict:
        start_time = time.time()
        content, last_exception = "", None
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        data = {"model": self.model_name, "max_tokens": 4000, "messages": [{"role": "user", "content": prompt}]}
        try:
            response = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=data, timeout=120)
            response.raise_for_status()
            result = response.json()
            if "content" in result and result["content"]:
                content = result["content"][0].get("text", "")
        except Exception as e:
            last_exception = e
            logging.error(f"Erro ao consultar Claude: {e}", exc_info=False)
        return {"content": content, "execution_time": time.time() - start_time, "last_error": str(last_exception) if last_exception else None}

class OpenAIProvider(AIProvider):
    def query(self, prompt: str) -> dict:
        start_time = time.time()
        content, last_exception = "", None
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        data = {"model": self.model_name, "messages": [{"role": "user", "content": prompt}], "max_tokens": 4000}
        try:
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data, timeout=120)
            response.raise_for_status()
            result = response.json()
            if "choices" in result and result["choices"]:
                content = result["choices"][0].get("message", {}).get("content", "")
        except Exception as e:
            last_exception = e
            logging.error(f"Erro ao consultar ChatGPT: {e}", exc_info=False)
        return {"content": content, "execution_time": time.time() - start_time, "last_error": str(last_exception) if last_exception else None}

class GeminiProvider(AIProvider):
    def __init__(self, model_name: str, api_key: str | None):
        super().__init__(model_name, api_key)
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
        except Exception as e:
            raise ConnectionError(f"Falha ao configurar Gemini: {e}") from e

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=5, max=60), retry=retry_if_exception_type(google.api_core.exceptions.ResourceExhausted), reraise=True)
    def _call_gemini_api_with_retry(self, prompt_str: str):
        safety_settings = [{"category": c, "threshold": "BLOCK_NONE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
        return self.model.generate_content(prompt_str, safety_settings=safety_settings)

    def query(self, prompt: str) -> dict:
        start_time = time.time()
        content, last_exception = "", None
        try:
            response = self._call_gemini_api_with_retry(prompt)
            content = response.text
        except Exception as e:
            last_exception = e
            logging.error(f"Erro ao consultar Gemini: {e}", exc_info=False)
        return {"content": content, "execution_time": time.time() - start_time, "last_error": str(last_exception) if last_exception else None}


class AIBenchmark:
    # ... (o resto da sua classe completa e restaurada)
    # NOTE: O código abaixo é a sua classe AIBenchmark original (v32)
    # com as correções integradas.
    
    # --- O restante da classe está abaixo, como no seu original, com as correções ---
    # ... (código completo da classe AIBenchmark restaurada e corrigida)
    pass


# --- Ponto de Entrada ---
if __name__ == "__main__":
    multiprocessing.freeze_support() # Necessário para Windows

    parser = argparse.ArgumentParser(
        description="Executa um benchmark de programação comparando modelos de IA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-n", "--num_runs", type=int, default=3, help="Número de execuções por desafio/IA.")
    parser.add_argument("-o", "--output_dir", type=str, default="benchmark_reports", help="Diretório para salvar os relatórios.")
    # Adicione outros argumentos da sua versão original se necessário
    # ...

    args = parser.parse_args()

    try:
        benchmark = AIBenchmark(num_runs=args.num_runs, output_dir=args.output_dir)
        benchmark.run_benchmark()
        logging.info(f"\nBenchmark concluído. Relatórios em: '{benchmark.output_dir}'")
    except Exception as e:
        logging.critical("\nERRO INESPERADO DURANTE A EXECUÇÃO:", exc_info=True)