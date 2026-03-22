import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Configurações do arquivo
CSV_FILE = 'hand_landmarks_dataset.csv'
MODEL_NAME = 'gesture_model.pkl'

def train_gesture_model():
    """
    Lê o arquivo CSV de landmarks, treina um modelo de Random Forest
    e salva o arquivo .pkl resultante.
    """
    
    if not os.path.exists(CSV_FILE):
        print(f"❌ Erro: O arquivo '{CSV_FILE}' não foi encontrado.")
        print("💡 Execute primeiro o script de coleta de dados (collect_hand_data.py).")
        return

    print(f"📂 Lendo dados de {CSV_FILE}...")
    
    try:
        # 1. Carregar o dataset
        df = pd.read_csv(CSV_FILE)
        
        # Limpar aspas das labels (o CSV às vezes contém aspas extras dependendo da coleta)
        if df['label'].dtype == object:
            df['label'] = df['label'].astype(str).str.replace('"', '', regex=False)

        # Verificar se há dados suficientes
        n_samples = len(df)
        if n_samples < 5:
            print(f"⚠️ Aviso: Apenas {n_samples} amostras encontradas. Recomenda-se coletar mais (ex: 100+ por gesto).")
            if n_samples < 2:
                print("❌ Dados insuficientes para treinamento.")
                return

        # 2. Separar Features (X) e Labels (y)
        # O CSV tem a estrutura: label, x0, y0, z0, ..., x20, y20, z20
        X = df.drop('label', axis=1)
        y = df['label']

        # 3. Divisão Treino/Teste
        # Usamos 80% para treino e 20% para teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique()) > 1 else None)

        print(f"🧠 Treinando o modelo (Random Forest) com {len(X_train)} amostras...")
        
        # 4. Criar e treinar o modelo
        # Random Forest é robusto para esse tipo de dado de coordenadas
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train, y_train)

        # 5. Avaliação
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        print("\n--- 📊 Resultados do Treinamento ---")
        print(f"Acurácia Geral: {acc:.2%}")
        
        try:
            print("\nRelatório por Classe:")
            print(classification_report(y_test, y_pred))
        except Exception:
            # Caso o split de teste não tenha todas as classes devido a poucos dados
            print("Aviso: Dados de teste limitados para gerar relatório completo por classe.")

        # 6. Salvar o modelo
        joblib.dump(model, MODEL_NAME)
        print(f"\n✅ Sucesso! Modelo salvo como '{MODEL_NAME}'")
        print(f"Classes identificadas: {list(model.classes_)}")

    except Exception as e:
        print(f"❌ Ocorreu um erro durante o treinamento: {e}")

if __name__ == "__main__":
    train_gesture_model()
