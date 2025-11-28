
# -*- coding: utf-8 -*-
import os
import json
import traceback
import io
import contextlib
import time
import subprocess # <<< Adicionado import >>>
import tempfile # <<< Adicionado import >>>
import sys # <<< Adicionado import (usado no wrapper) >>>
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/execute', methods=['POST'])
def execute_code():
    """
    Recebe código, nome da função e argumentos via JSON,
    executa a função em um subprocesso Python isolado
    e retorna o resultado.
    """
    start_process_time = time.perf_counter() # Mede o tempo total da rota
    result_data = {
        "success": False,
        "output": None,
        "stdout": "",
        "error": None,
        "execution_time_ms": -1.0
    }

    try:
        data = request.get_json()
        if not data:
            result_data["error"] = "Request body must be JSON"
            return jsonify(result_data), 400

        # code_string já deve vir com os imports adicionados pelo benchmark
        code_string = data.get('code_string')
        function_name = data.get('function_name')
        test_input_args = data.get('test_input_args', [])

        if not code_string or not function_name:
            result_data["error"] = "Missing 'code_string' or 'function_name' in request"
            return jsonify(result_data), 400

        # Garante que test_input_args seja uma lista (para consistência no wrapper)
        if not isinstance(test_input_args, (list, tuple)):
             test_input_args = [test_input_args]

        # --- Bloco de Execução via Subprocesso ---
        try:
            # Cria um arquivo temporário para o script wrapper
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.py', delete=True, encoding='utf-8') as temp_script:

                # Código Wrapper que será executado no subprocesso
                # Ele recebe os dados via stdin, executa o código da IA, captura resultados e imprime JSON no stdout
                wrapper_code = f"""
import json
import sys
import traceback
import io
import contextlib
import time

# Tenta importar módulos comuns (caso não estejam no code_string - redundância segura)
try:
    from typing import List, Dict, Tuple, Set, Optional, Any
    import re
    from collections import OrderedDict, deque
except ImportError:
    pass # Ignora se falhar aqui, confia nos imports do code_string

# --- Código da IA recebido ---
# (Já deve incluir imports adicionados pelo benchmark como List, re, deque)
{code_string}
# --- Fim do Código da IA ---

# Argumentos recebidos via JSON no stdin
input_data = json.load(sys.stdin)
func_name = input_data['function_name']
args = input_data['args']

output = None
stdout_capture = io.StringIO()
error = None
success = False
exec_start_time = time.perf_counter()

try:
    # Encontra a função no escopo local/global após o exec implícito do import
    # Usamos locals() pois 'exec' no nível do módulo normalmente define no escopo global (que se torna local aqui)
    target_func = locals().get(func_name)

    # Tenta globals() como fallback, embora menos provável de funcionar corretamente com exec
    if not target_func:
         target_func = globals().get(func_name)

    if not target_func:
        # Se ainda não achou, verifica se é uma classe (para futuras implementações)
        target_cls = locals().get(func_name) or globals().get(func_name)
        if not target_cls:
             raise NameError(f"Function or Class '{{func_name}}' not defined after code execution.")
        # Lógica para instanciar/chamar classe iria aqui se suportada
        # Por ora, lançamos erro se não for função
        if not callable(target_cls):
             raise NameError(f"Name '{{func_name}}' points to a non-callable object (likely a class, not supported).")
        # Se chegou aqui, era callable mas não achou antes? Improvável. Lançar erro.
        raise NameError(f"Callable '{{func_name}}' found but mechanism unclear.")


    # Captura stdout da função da IA
    with contextlib.redirect_stdout(stdout_capture):
         # Chama a função alvo com desempacotamento de argumentos
         output = target_func(*args)
    success = True # Sucesso na execução (não significa que o output está correto)

except Exception as e:
    # Captura erro DENTRO da execução do código da IA
    error = f"{{type(e).__name__}}: {{str(e)}}\\n{{traceback.format_exc()}}" # Inclui traceback detalhado
    success = False

# Imprime resultado como JSON no stdout do subprocesso
result = {{
    'success': success,
    'output': output,
    'stdout': stdout_capture.getvalue(), # Stdout capturado da função da IA
    'error': error
    # Nota: O tempo de execução é medido no processo pai
}}

# Tratamento especial para saída None (json.dumps falha com None diretamente às vezes)
# e para tipos não serializáveis (embora improvável para os desafios)
try:
    print(json.dumps(result))
except TypeError as te:
    # Tenta converter tipos não serializáveis para string
    def safe_encoder(obj):
        if isinstance(obj, (set, complex)): # Adicione outros tipos se necessário
            return str(obj)
        raise TypeError(f"Object of type {{obj.__class__.__name__}} is not JSON serializable")
    print(json.dumps(result, default=safe_encoder))

"""
                temp_script.write(wrapper_code)
                temp_script.flush() # Garante que o conteúdo foi escrito

                # Prepara os dados de entrada para o subprocesso (JSON via stdin)
                process_input_data = json.dumps({
                    "function_name": function_name,
                    "args": test_input_args
                })

                # Define o timeout para o subprocesso (ex: 30 segundos)
                timeout_seconds = 30

                # Executa o script wrapper como um subprocesso Python
                completed_process = subprocess.run(
                    [sys.executable, temp_script.name], # Usa o mesmo executável python
                    input=process_input_data,
                    capture_output=True, # Captura stdout e stderr
                    text=True,           # Decodifica output/error como texto
                    timeout=timeout_seconds
                )

            # Tempo total desde o início da rota até o fim do subprocesso
            total_time_ms = (time.perf_counter() - start_process_time) * 1000

            # Processa o resultado do subprocesso
            if completed_process.returncode == 0:
                # Subprocesso terminou com sucesso, tenta ler o JSON do stdout
                try:
                    exec_result = json.loads(completed_process.stdout)
                    # Atualiza o dicionário principal com os resultados da execução interna
                    result_data.update(exec_result)
                    # O tempo de execução retornado é o tempo total da rota aqui
                    result_data["execution_time_ms"] = total_time_ms
                except json.JSONDecodeError:
                    # Se o stdout do subprocesso não for JSON válido
                    result_data["error"] = f"Executor subprocess internal error: Failed to decode JSON output. stdout: {completed_process.stdout[:500]}"
                    result_data["stdout"] = completed_process.stdout # Guarda o stdout bruto
                    result_data["execution_time_ms"] = total_time_ms
            else:
                # Subprocesso terminou com erro (código de retorno != 0)
                result_data["error"] = f"Executor subprocess failed with code {completed_process.returncode}. stderr: {completed_process.stderr[:500]}"
                result_data["stdout"] = completed_process.stdout # Pode ter algo útil
                result_data["execution_time_ms"] = total_time_ms

        except subprocess.TimeoutExpired:
            total_time_ms = (time.perf_counter() - start_process_time) * 1000
            result_data["error"] = f"TimeoutExpired: Code execution exceeded the {timeout_seconds}s time limit."
            result_data["execution_time_ms"] = total_time_ms
        except Exception as e_outer:
            # Captura erros na lógica do Flask antes/depois do subprocesso
            total_time_ms = (time.perf_counter() - start_process_time) * 1000
            result_data["error"] = f"Flask handler error: {type(e_outer).__name__}: {e_outer}"
            result_data["execution_time_ms"] = total_time_ms
            print(f"Internal Server Error in Flask handler: {e_outer}")
            traceback.print_exc()
        # --- Fim do Bloco de Execução via Subprocesso ---

        # Retorna o dicionário final de resultados
        return jsonify(result_data)

    except Exception as e_request:
        # Erro ao processar a requisição inicial (ex: JSON inválido)
        print(f"Error processing request: {e_request}")
        traceback.print_exc()
        return jsonify({"error": f"Error processing request: {str(e_request)}"}), 500


if __name__ == "__main__":
    # O Cloud Run define a variável PORT. Localmente, usa 8080.
    port = int(os.environ.get("PORT", 8080))
    # Escuta em todas as interfaces de rede ('0.0.0.0')
    # debug=False é importante para produção e Cloud Run
    app.run(debug=False, host='0.0.0.0', port=port)