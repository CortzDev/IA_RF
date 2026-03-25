import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from sqlalchemy import create_engine
import time
from sklearn.ensemble import RandomForestRegressor
import os
import joblib
import threading
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Configuración de variables con la URL directa
DB_URL = "postgresql://postgres:VaLqxGBzdzZmBTddchzzryKgNeQmoPfI@switchback.proxy.rlwy.net:14573/railway?sslmode=require"

sensores = ['temp_current', 'humidity_value', 'co2_value', 'pm25_value', 'pm10']
RUTA_MODELO = '/app/data/cerebro_sensores.joblib'

# ==========================================
# 1. EL MOTOR DE LA IA (Se mantiene igual)
# ==========================================
def cargar_y_entrenar(engine):
    print("🧠 Entrenando el modelo con los datos más recientes...")
    query = "SELECT * FROM sensor_metrics ORDER BY recorded_at ASC"
    df = pd.read_sql(query, engine)
    
    for s in sensores:
        df[f'{s}_pasado'] = df[s].shift(1)
        df[f'{s}_futuro'] = df[s].shift(-1)
        
    df_train = df.dropna()
    columnas_X = sensores + [f'{s}_pasado' for s in sensores]
    columnas_Y = [f'{s}_futuro' for s in sensores]
    
    X = df_train[columnas_X]
    Y = df_train[columnas_Y]
    
    modelo = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    modelo.fit(X, Y)
    
    # Nos aseguramos de que el directorio exista antes de guardar
    os.makedirs(os.path.dirname(RUTA_MODELO), exist_ok=True)
    joblib.dump(modelo, RUTA_MODELO)
    print(f"💾 Memoria guardada exitosamente en: {RUTA_MODELO}")
    
    verdadero_presente = df.iloc[-1:][columnas_X].copy()
    return modelo, columnas_X, verdadero_presente

def fusion_ia_monitor():
    engine = create_engine(DB_URL)
    
    if os.path.exists(RUTA_MODELO):
        print(f"📦 Cargando IA desde archivo existente ({RUTA_MODELO})...")
        modelo = joblib.load(RUTA_MODELO)
        query = f"SELECT * FROM sensor_metrics ORDER BY recorded_at DESC LIMIT 2"
        df_reciente = pd.read_sql(query, engine).sort_values(by='recorded_at', ascending=True)
        for s in sensores:
            df_reciente[f'{s}_pasado'] = df_reciente[s].shift(1)
        columnas_X = sensores + [f'{s}_pasado' for s in sensores]
        datos_actuales = df_reciente.iloc[-1:][columnas_X].copy()
    else:
        print("🌱 No hay IA guardada. Iniciando entrenamiento...")
        modelo, columnas_X, datos_actuales = cargar_y_entrenar(engine)
    
    prediccion_actual = modelo.predict(datos_actuales)[0]
    nuevos_registros_count = 0
    LIMITE_PARA_REENTRENAR = 50 
    conn = None
    cursor = None
    
    try:
        conn = psycopg2.connect(DB_URL)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("SELECT id FROM sensor_metrics ORDER BY id DESC LIMIT 1;")
        row = cursor.fetchone()
        last_id = row['id'] if row else 0
        
        print("\n🚀 Iniciando ciclo de Predicción vs Realidad...")
        while True:
            cursor.execute("SELECT * FROM sensor_metrics WHERE id > %s ORDER BY id ASC;", (last_id,))
            nuevos_registros = cursor.fetchall()
            
            for registro in nuevos_registros:
                print("\n✨ ¡NUEVO DATO DETECTADO!")
                for i, sensor in enumerate(sensores):
                    valor_real = float(registro[sensor])
                    valor_predicho = float(prediccion_actual[i])
                    diferencia = abs(valor_real - valor_predicho)
                    estado_booleano = bool(diferencia <= (abs(valor_real) * 0.05))
                    
                    cursor.execute("""
                        INSERT INTO predicciones_log (sensor, valor_real, valor_predicho, margen_error, acertado)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (sensor, valor_real, valor_predicho, diferencia, estado_booleano))
                
                conn.commit() 
                
                nuevos_datos_X = pd.DataFrame([{
                    **{s: registro[s] for s in sensores}, 
                    **{f'{s}_pasado': datos_actuales[s].values[0] for s in sensores} 
                }], columns=columnas_X)
                
                prediccion_actual = modelo.predict(nuevos_datos_X)[0]
                datos_actuales = nuevos_datos_X
                last_id = registro['id']
                nuevos_registros_count += 1
                print(f"🔮 Predicción calculada. Esperando dato ({nuevos_registros_count}/{LIMITE_PARA_REENTRENAR})")
                
                if nuevos_registros_count >= LIMITE_PARA_REENTRENAR:
                    print("\n🔄 La IA está aprendiendo de los nuevos datos...")
                    modelo, columnas_X, datos_actuales = cargar_y_entrenar(engine)
                    prediccion_actual = modelo.predict(datos_actuales)[0]
                    nuevos_registros_count = 0 
            time.sleep(2)
    except Exception as e:
        print(f"\n❌ Ocurrió un error en el monitor: {e}")
    finally:
        if cursor: cursor.close()
        if conn: conn.close()


# ==========================================
# 2. EL SERVIDOR WEB (Para tu página HTML)
# ==========================================
app = FastAPI(title="API Sensores IA")

# Permite que tu HTML consulte esta API sin bloqueos de seguridad (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"mensaje": "Servidor de IA Activo. El modelo está corriendo en segundo plano."}

@app.get("/api/estadisticas")
def obtener_estadisticas():
    # Esta es la ruta a la que tu HTML hará la petición
    try:
        conn = psycopg2.connect(DB_URL)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            SELECT 
                SUM(CASE WHEN acertado = true THEN 1 ELSE 0 END) as aciertos,
                SUM(CASE WHEN acertado = false THEN 1 ELSE 0 END) as fallos
            FROM predicciones_log;
        """)
        stats = cursor.fetchone()
        cursor.close()
        conn.close()
        
        return {
            "aciertos": stats["aciertos"] or 0,
            "fallos": stats["fallos"] or 0,
            "total": (stats["aciertos"] or 0) + (stats["fallos"] or 0)
        }
    except Exception as e:
        return {"error": str(e)}

# ==========================================
# 3. ARRANQUE DEL SISTEMA UNIFICADO
# ==========================================
if __name__ == "__main__":
    # 1. Arrancamos el motor de la IA en un "hilo" de fondo para que no bloquee
    hilo_ia = threading.Thread(target=fusion_ia_monitor, daemon=True)
    hilo_ia.start()
    
    # 2. Arrancamos el servidor web en el puerto que asigne Railway
    puerto = int(os.environ.get("PORT", 8000))
    print(f"🌐 Iniciando servidor web en el puerto {puerto}...")
    uvicorn.run(app, host="0.0.0.0", port=puerto)