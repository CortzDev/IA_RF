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

# ==========================================
# CONFIGURACIÓN
# ==========================================
DB_URL = "postgresql://postgres:VaLqxGBzdzZmBTddchzzryKgNeQmoPfI@switchback.proxy.rlwy.net:14573/railway?sslmode=require"
sensores = ['temp_current', 'humidity_value', 'co2_value', 'pm25_value', 'pm10']
RUTA_MODELO = '/app/data/cerebro_sensores.joblib'

def cargar_y_entrenar(engine):
    print("🧠 Entrenando modelo con datos históricos...", flush=True)
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
    print("✅ Modelo entrenado y guardado en volumen.", flush=True)
    
    return modelo, columnas_X, df.iloc[-1:][columnas_X].copy()

def fusion_ia_monitor():
    print("🚀 Iniciando servicios de inteligencia artificial...", flush=True)
    engine = create_engine(DB_URL)
    
    while True:
        try:
            if os.path.exists(RUTA_MODELO):
                print("📦 Cargando cerebro de la IA desde el volumen...", flush=True)
                modelo = joblib.load(RUTA_MODELO)
                query = "SELECT * FROM sensor_metrics ORDER BY recorded_at DESC LIMIT 2"
                df_reciente = pd.read_sql(query, engine).sort_values(by='recorded_at')
                for s in sensores:
                    df_reciente[f'{s}_pasado'] = df_reciente[s].shift(1)
                columnas_X = sensores + [f'{s}_pasado' for s in sensores]
                datos_actuales = df_reciente.iloc[-1:][columnas_X].copy()
            else:
                print("🌱 Iniciando primer entrenamiento del sistema...", flush=True)
                modelo, columnas_X, datos_actuales = cargar_y_entrenar(engine)

            prediccion_actual = modelo.predict(datos_actuales)[0]
            
            conn = psycopg2.connect(DB_URL)
            conn.autocommit = True 
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("SELECT id FROM sensor_metrics ORDER BY id DESC LIMIT 1;")
            last_id = cursor.fetchone()['id']
            
            print(f"✅ Conexión establecida. Monitoreando desde ID: {last_id}", flush=True)
            
            nuevos_registros_count = 0
            
            while True:
                cursor.execute("SELECT * FROM sensor_metrics WHERE id > %s ORDER BY id ASC;", (last_id,))
                nuevos = cursor.fetchall()
                
                if not nuevos:
                    print("🔍 Escaneando base de datos: buscando nuevas lecturas...", flush=True)
                
                for registro in nuevos:
                    print(f"✨ Nueva lectura detectada: reading_id={registro['id']} at {pd.Timestamp.now()}", flush=True)
                    
                    for i, s in enumerate(sensores):
                        v_real = float(registro[s])
                        v_pred = float(prediccion_actual[i])
                        error = abs(v_real - v_pred)
                        acierto = bool(error <= (abs(v_real) * 0.05))
                        
                        cursor.execute("""
                            INSERT INTO predicciones_log (sensor, valor_real, valor_predicho, margen_error, acertado)
                            VALUES (%s, %s, %s, %s, %s)
                        """, (s, v_real, v_pred, error, acierto))
                        
                        icon = "✅" if acierto else "⚠️"
                        print(f"   {icon} Procesando {s}: Real={v_real} | IA={v_pred:.2f}", flush=True)

                    # Preparar siguiente
                    nuevos_X = pd.DataFrame([{
                        **{s: registro[s] for s in sensores}, 
                        **{f'{s}_pasado': datos_actuales[s].values[0] for s in sensores} 
                    }], columns=columnas_X)
                    
                    prediccion_actual = modelo.predict(nuevos_X)[0]
                    datos_actuales = nuevos_X
                    last_id = registro['id']
                    nuevos_registros_count += 1
                    
                    if nuevos_registros_count >= 50:
                        print("🔄 Reentrenamiento periódico: actualizando IA con nuevos datos...", flush=True)
                        modelo, columnas_X, datos_actuales = cargar_y_entrenar(engine)
                        nuevos_registros_count = 0

                time.sleep(10) 
                
        except Exception as e:
            print(f"❌ Error en el servicio: {e}. Reintentando...", flush=True)
            time.sleep(15)

# ==========================================
# SERVIDOR WEB
# ==========================================
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def home(): return {"status": "online"}

@app.get("/api/estadisticas")
def stats():
    conn = psycopg2.connect(DB_URL); cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT SUM(CASE WHEN acertado THEN 1 ELSE 0 END) as aciertos, COUNT(*) as total FROM predicciones_log")
    res = cur.fetchone(); cur.close(); conn.close()
    t = res['total'] or 0; a = res['aciertos'] or 0
    return {"aciertos": a, "fallos": t - a, "total": t}

@app.get("/api/historial")
def historial():
    conn = psycopg2.connect(DB_URL); cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT * FROM predicciones_log ORDER BY id DESC LIMIT 50"); logs = cur.fetchall()
    cur.close(); conn.close()
    hist = []
    for i in range(0, len(logs), 5):
        g = logs[i:i+5]
        if g: hist.append({"nombre": f"Lectura #{ (len(logs)//5) - (i//5) }", "fecha": g[0]['fecha'], "datos": g})
    return hist

if __name__ == "__main__":
    threading.Thread(target=fusion_ia_monitor, daemon=True).start()
    puerto = int(os.environ.get("PORT", 8080))
    print(f"🌐 Servidor web iniciado en puerto {puerto}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=puerto)