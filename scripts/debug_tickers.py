# scripts/debug_tickers.py
# Comando para rodar o script: python scripts/debug_tickers.py
import os

# --- CONFIGURAÇÃO ---
# Coloque o caminho para a MESMA pasta dos seus arquivos COTAHIST
COTAHIST_FOLDER_PATH = r"D:\UERJ\Programacao_e_Codigos\Artigo - Development ofstock correlation networks\Data_SpotPrice_hist"
# Especifique UM arquivo para analisar
FILE_TO_DEBUG = "COTAHIST_A2020.TXT"
# --------------------

def find_unique_tickers(file_path):
    if not os.path.exists(file_path):
        print(f"ERRO: Arquivo não encontrado em: {file_path}")
        return

    print(f"Analisando o arquivo: {file_path}")
    unique_tickers = set()

    with open(file_path, 'r', encoding='latin-1') as f:
        for line in f:
            # Apenas linhas de cotação (tipo '01')
            if line.startswith('01'):
                # FILTRO ADICIONAL: Pega o código BDI da sua posição fixa
                codbdi = line[10:12]
                
                # Processa apenas se for do lote-padrão (código '02')
                if codbdi == '02':
                    ticker = line[12:24].strip()
                    unique_tickers.add(ticker)

    print(f"\nForam encontrados {len(unique_tickers)} tickers únicos no mercado à vista (BDI '02').")
    
    print("Amostra de tickers encontrados:")
    # Imprime a lista completa ordenada
    print(sorted(list(unique_tickers)))

if __name__ == "__main__":
    full_path = os.path.join(COTAHIST_FOLDER_PATH, FILE_TO_DEBUG)
    find_unique_tickers(full_path)