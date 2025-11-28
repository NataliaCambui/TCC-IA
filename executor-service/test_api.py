import google.generativeai as genai
import os

# --- IMPORTANTE ---
# Cole sua chave de API aqui, entre as aspas.
# Ou certifique-se de que a variável de ambiente GOOGLE_API_KEY está definida.
try:
    # Tenta pegar a chave da variável de ambiente primeiro
    api_key = os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        # Se não encontrar, use a chave abaixo (COLE A SUA AQUI)
        api_key = "AIzaSyB9rS9Uwpgj54AzK85a5ScXeoFVQ-DdkrY" # <--- COLOQUE SUA CHAVE AQUI
        if api_key == "SUA_CHAVE_API_AQUI":
             raise ValueError("Por favor, substitua 'SUA_CHAVE_API_AQUI' pela sua chave de API real.")

    genai.configure(api_key=api_key)

    print("--- Modelos disponíveis para sua chave de API ---")

    # Itera e lista todos os modelos que suportam o método 'generateContent'
    model_count = 0
    for model in genai.list_models():
        if 'generateContent' in model.supported_generation_methods:
            print(f"- {model.name}")
            model_count += 1

    if model_count == 0:
        print("\nNENHUM modelo compatível encontrado.")
        print("Isso geralmente indica um problema de configuração no seu projeto Google Cloud.")
        print("Verifique se a 'Generative Language API' ou 'Vertex AI API' está ativada corretamente.")

    print("-------------------------------------------------")

except Exception as e:
    print(f"\nOcorreu um erro ao tentar listar os modelos: {e}")