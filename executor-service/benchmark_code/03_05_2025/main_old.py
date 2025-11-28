import os
import json
import traceback
import io
import contextlib
import time
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/execute', methods=['POST'])
def execute_code():
    """
    Recebe código, nome da função e argumentos via JSON,
    executa a função e retorna o resultado.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request body must be JSON"}), 400

        code_string = data.get('code_string')
        function_name = data.get('function_name')
        # test_input pode ser um único valor ou uma lista/dicionário de argumentos
        test_input_args = data.get('test_input_args', []) # Espera uma lista de args

        if not code_string or not function_name:
            return jsonify({"error": "Missing 'code_string' or 'function_name' in request"}), 400

        # --- Bloco de Execução (ATENÇÃO: RISCOS COM exec()) ---
        local_namespace = {}
        execution_output = None
        execution_stdout = ""
        execution_error = None
        execution_success = False
        start_time = time.perf_counter()

        # Redireciona stdout para capturar prints
        stdout_capture = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout_capture):
                # Executa o código da IA para definir a função no namespace local
                exec(code_string, globals(), local_namespace)

                # Verifica se a função esperada foi definida
                if function_name not in local_namespace:
                    raise NameError(f"Function '{function_name}' not defined in the provided code.")

                # Obtém a função definida
                target_function = local_namespace[function_name]

                # Chama a função com os argumentos fornecidos
                # Garante que test_input_args seja uma lista ou tupla para desempacotamento
                if not isinstance(test_input_args, (list, tuple)):
                     # Se for um único argumento, coloca em uma lista
                     test_input_args = [test_input_args]

                execution_output = target_function(*test_input_args) # Desempacota args

            execution_success = True # Se chegou aqui sem exceção, considera sucesso
        except Exception as e:
            execution_error = f"{type(e).__name__}: {str(e)}"
            # Opcional: Capturar traceback completo para debug
            # execution_error += f"\n{traceback.format_exc()}"
        finally:
             execution_time_ms = (time.perf_counter() - start_time) * 1000
             execution_stdout = stdout_capture.getvalue()
        # --- Fim do Bloco de Execução ---

        # Retorna resultado detalhado
        return jsonify({
            "success": execution_success,
            "output": execution_output,
            "stdout": execution_stdout,
            "error": execution_error,
            "execution_time_ms": execution_time_ms
        })

    except Exception as e:
        # Erro na própria lógica do serviço executor
        print(f"Internal Server Error: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

if __name__ == "__main__":
    # O Cloud Run define a variável PORT. Localmente, usa 8080.
    port = int(os.environ.get("PORT", 8080))
    # Escuta em todas as interfaces de rede ('0.0.0.0')
    app.run(debug=False, host='0.0.0.0', port=port)