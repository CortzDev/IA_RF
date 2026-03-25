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
from fastapi.responses import FileResponse

# Configuración de variables
DB_URL = "postgresql://postgres:VaLqxGBzdzZmBTddchzzryKgNeQmoPfI@switchback.proxy.rlwy.net:14573/railway?sslmode=require"
sensores = ['temp_current', 'humidity_value', 'co2_value', 'pm25_value', 'pm10']
RUTA_MODELO = '/app/data/cerebro_sensores.joblib'

# ==========================================
# 1. MOTOR DE IA (CON PRINTS REFORZADOS)
# ==========================================
def cargar_y_entrenar(engine):
    print("🧠 [IA] Entrenando con datos recientes...")
    query = "SELECT * FROM sensor_metrics ORDER BY recorded_at ASC"
    df = pd.read_sql(query, engine)
    
    for s in sensores:
        df[f'{s}_pasado'] = df[s].shift(1)
        df[f'{s}_futuro'] = df[s].shift(-1)
        
    df_train = df.dropna()
    columnas_X = sensores + [f'{s}_pasado' for s in sensores]
    columnas_Y = [f'{s}_futuro' for s in sensores]
    
    modelo = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    modelo.fit(df_train[columnas_X], df_train[columnas_Y])
    
    os.makedirs(os.path.dirname(RUTA_MODELO), exist_ok=True)
    joblib.dump(modelo, RUTA_MODELO)
    print(f"💾 [IA] Memoria guardada en: {RUTA_MODELO}")
    
    return modelo, columnas_X, df.iloc[-1:][columnas_X].copy()

def fusion_ia_monitor():
    print("⚙️ [SISTEMA] Iniciando Monitor de Sensores...")
    engine = create_engine(DB_URL)
    
    # Bucle de conexión inicial
    while True:
        try:
            if os.path.exists(RUTA_MODELO):
                print(f"📦 [IA] Cargando modelo existente...")
                modelo = joblib.load(RUTA_MODELO)
                query = "SELECT * FROM sensor_metrics ORDER BY recorded_at DESC LIMIT 2"
                df_reciente = pd.read_sql(query, engine).sort_values(by='recorded_at')
                for s in sensores:
                    df_reciente[f'{s}_pasado'] = df_reciente[s].shift(1)
                columnas_X = sensores + [f'{s}_pasado' for s in sensores]
                datos_actuales = df_reciente.iloc[-1:][columnas_X].copy()
            else:
                modelo, columnas_X, datos_actuales = cargar_y_entrenar(engine)

            prediccion_actual = modelo.predict(datos_actuales)[0]
            
            # Conexión principal con Autocommit para ver datos en tiempo real
            conn = psycopg2.connect(DB_URL)
            conn.autocommit = True 
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("SELECT id FROM sensor_metrics ORDER BY id DESC LIMIT 1;")
            last_id = cursor.fetchone()['id']
            
            print(f"✅ [CONEXIÓN] Base de datos lista. ID actual: {last_id}")
            print("🚀 [IA] ¡Iniciando ciclo de Predicción vs Realidad!")
            
            while True:
                cursor.execute("SELECT * FROM sensor_metrics WHERE id > %s ORDER BY id ASC;", (last_id,))
                nuevos = cursor.fetchall()
                
                for registro in nuevos:
                    print(f"\n✨ [DETECCIÓN] ¡Nuevo dato recibido! (ID: {registro['id']})")
                    for i, s in enumerate(sensores):
                        v_real, v_pred = float(registro[s]), float(prediccion_actual[i])
                        error = abs(v_real - v_pred)
                        acierto = bool(error <= (abs(v_real) * 0.05))
                        
                        cursor.execute("""
                            INSERT INTO predicciones_log (sensor, valor_real, valor_predicho, margen_error, acertado)
                            VALUES (%s, %s, %s, %s, %s)
                        """, (s, v_real, v_pred, error, acierto))
                        print(f"   📊 {s.upper()}: Real {v_real} | Predicho {v_pred:.2f} | {'✅' if acierto else '⚠️'}")

                    # Preparar siguiente ciclo
                    nuevos_X = pd.DataFrame([{
                        **{s: registro[s] for s in sensores}, 
                        **{f'{s}_pasado': datos_actuales[s].values[0] for s in sensores} 
                    }], columns=columnas_X)
                    prediccion_actual = modelo.predict(nuevos_X)[0]
                    datos_actuales = nuevos_X
                    last_id = registro['id']
                
                time.sleep(5) # Espera 5 seg entre chequeos
                
        except Exception as e:
            print(f"❌ [ERROR] Fallo en el monitor: {e}. Reintentando en 10s...")
            time.sleep(10)

# ==========================================
# 2. SERVIDOR WEB (FASTAPI)
# ==========================================
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def home(): return {"estado": "online", "ia": "activa"}

@app.get("/api/estadisticas")
def stats():
    conn = psycopg2.connect(DB_URL); cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT SUM(CASE WHEN acertado THEN 1 ELSE 0 END) as aciertos, COUNT(*) as total FROM predicciones_log")
    res = cur.fetchone(); cur.close(); conn.close()
    return {"aciertos": res['aciertos'] or 0, "fallos": (res['total'] or 0) - (res['aciertos'] or 0), "total": res['total'] or 0}

@app.get("/api/historial")
def historial():
    conn = psycopg2.connect(DB_URL); cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT * FROM predicciones_log ORDER BY id DESC LIMIT 50"); logs = cur.fetchall()
    cur.close(); conn.close()
    hist = []
    for i in range(0, len(logs), 5):
        g = logs[i:i+5]
        if g: hist.append({"nombre": f"Lectura", "fecha": g[0]['fecha'], "datos": g})
    return hist

# ==========================================
# 3. ARRANQUE
# ==========================================
if __name__ == "__main__":
    # Hilo de la IA
    threading.Thread(target=fusion_ia_monitor, daemon=True).start()
    # Servidor Web
    puerto = int(os.environ.get("PORT", 8000))
    print(f"🌐 [WEB] Servidor en puerto {puerto}")
    uvicorn.run(app, host="0.0.0.0", port=puerto)