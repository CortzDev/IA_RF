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
# CONFIGURACIÓN GLOBAL
# ==========================================
DB_URL = "postgresql://postgres:VaLqxGBzdzZmBTddchzzryKgNeQmoPfI@switchback.proxy.rlwy.net:14573/railway?sslmode=require"
sensores = ['temp_current', 'humidity_value', 'co2_value', 'pm25_value', 'pm10']
RUTA_MODELO = '/app/data/cerebro_sensores.joblib'

# ==========================================
# 1. MOTOR DE IA (CON SALIDA DE LOGS FORZADA)
# ==========================================
def cargar_y_entrenar(engine):
    print("🧠 [IA] Entrenando modelo con datos históricos...", flush=True)
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
    print(f"💾 [IA] Memoria guardada exitosamente en: {RUTA_MODELO}", flush=True)
    
    return modelo, columnas_X, df.iloc[-1:][columnas_X].copy()

def fusion_ia_monitor():
    print("⚙️ [SISTEMA] Iniciando Monitor de Sensores...", flush=True)
    engine = create_engine(DB_URL)
    
    while True:
        try:
            # Cargar o Entrenar
            if os.path.exists(RUTA_MODELO):
                print(f"📦 [IA] Cargando modelo desde el volumen...", flush=True)
                modelo = joblib.load(RUTA_MODELO)
                query = "SELECT * FROM sensor_metrics ORDER BY recorded_at DESC LIMIT 2"
                df_reciente = pd.read_sql(query, engine).sort_values(by='recorded_at')
                for s in sensores:
                    df_reciente[f'{s}_pasado'] = df_reciente[s].shift(1)
                columnas_X = sensores + [f'{s}_pasado' for s in sensores]
                datos_actuales = df_reciente.iloc[-1:][columnas_X].copy()
            else:
                print("🌱 [IA] No se encontró modelo. Entrenando...", flush=True)
                modelo, columnas_X, datos_actuales = cargar_y_entrenar(engine)

            prediccion_actual = modelo.predict(datos_actuales)[0]
            
            # Conexión a la base de datos
            conn = psycopg2.connect(DB_URL)
            conn.autocommit = True # Clave para ver datos nuevos sin "congelarse"
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("SELECT id FROM sensor_metrics ORDER BY id DESC LIMIT 1;")
            last_id = cursor.fetchone()['id']
            
            print(f"✅ [CONEXIÓN] Base de datos vinculada. ID inicial: {last_id}", flush=True)
            print("🚀 [IA] ¡Iniciando ciclo activo de predicción!", flush=True)
            
            nuevos_registros_count = 0
            
            while True:
                cursor.execute("SELECT * FROM sensor_metrics WHERE id > %s ORDER BY id ASC;", (last_id,))
                nuevos = cursor.fetchall()
                
                if not nuevos:
                    # Mensaje de pulso para saber que el hilo no ha muerto
                    print("📡 [IA] Escaneando base de datos... (Sin cambios)", flush=True)
                
                for registro in nuevos:
                    print(f"\n✨ [DETECCIÓN] ¡Nuevo dato encontrado! ID: {registro['id']}", flush=True)
                    
                    for i, s in enumerate(sensores):
                        v_real = float(registro[s])
                        v_pred = float(prediccion_actual[i])
                        error = abs(v_real - v_pred)
                        # Margen del 5% para considerar acierto
                        acierto = bool(error <= (abs(v_real) * 0.05))
                        
                        cursor.execute("""
                            INSERT INTO predicciones_log (sensor, valor_real, valor_predicho, margen_error, acertado)
                            VALUES (%s, %s, %s, %s, %s)
                        """, (s, v_real, v_pred, error, acierto))
                        
                        print(f"   📊 {s.upper()}: Real {v_real} | Predicho {v_pred:.2f} | {'✅' if acierto else '⚠️'}", flush=True)

                    # Calcular predicción para el siguiente dato que vendrá
                    nuevos_X = pd.DataFrame([{
                        **{s: registro[s] for s in sensores}, 
                        **{f'{s}_pasado': datos_actuales[s].values[0] for s in sensores} 
                    }], columns=columnas_X)
                    
                    prediccion_actual = modelo.predict(nuevos_X)[0]
                    datos_actuales = nuevos_X
                    last_id = registro['id']
                    nuevos_registros_count += 1
                    
                    if nuevos_registros_count >= 50:
                        print("🔄 [IA] Umbral de 50 datos alcanzado. Reentrenando...", flush=True)
                        modelo, columnas_X, datos_actuales = cargar_y_entrenar(engine)
                        nuevos_registros_count = 0

                time.sleep(10) # Pausa de 10 segundos entre escaneos
                
        except Exception as e:
            print(f"❌ [ERROR CRÍTICO] {e}. Reiniciando monitor en 15s...", flush=True)
            time.sleep(15)

# ==========================================
# 2. SERVIDOR API (FASTAPI)
# ==========================================
app = FastAPI(title="Dashboard IA Sensores")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "online", "message": "API de IA para Sensores funcionando"}

@app.get("/api/estadisticas")
def stats():
    try:
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT 
                SUM(CASE WHEN acertado THEN 1 ELSE 0 END) as aciertos,
                COUNT(*) as total
            FROM predicciones_log
        """)
        res = cur.fetchone()
        cur.close(); conn.close()
        total = res['total'] or 0
        aciertos = res['aciertos'] or 0
        return {"aciertos": aciertos, "fallos": total - aciertos, "total": total}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/historial")
def historial():
    try:
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        # Traemos los últimos 50 registros de la tabla de log
        cur.execute("SELECT * FROM predicciones_log ORDER BY id DESC LIMIT 50")
        logs = cur.fetchall()
        cur.close(); conn.close()
        
        # Agrupamos los datos de 5 en 5 (cada toma de datos de los 5 sensores)
        agrupados = []
        for i in range(0, len(logs), 5):
            grupo = logs[i:i+5]
            if grupo:
                agrupados.append({
                    "nombre": f"Predicción #{ (len(logs)//5) - (i//5) }",
                    "fecha": grupo[0]['fecha'],
                    "datos": grupo
                })
        return agrupados
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/descargar-modelo")
def descargar():
    if os.path.exists(RUTA_MODELO):
        return FileResponse(RUTA_MODELO, media_type='application/octet-stream', filename='cerebro_ia.joblib')
    return {"error": "Archivo no encontrado"}

# ==========================================
# 3. ARRANQUE UNIFICADO
# ==========================================
if __name__ == "__main__":
    print("🚀 [SISTEMA] Iniciando despliegue unificado...", flush=True)
    
    # Lanzar el hilo de la IA
    hilo = threading.Thread(target=fusion_ia_monitor, daemon=True)
    hilo.start()
    print("🧵 [HILO] Motor de IA lanzado en segundo plano.", flush=True)

    # Lanzar el servidor Web
    puerto = int(os.environ.get("PORT", 8080))
    print(f"🌐 [WEB] Servidor escuchando en puerto {puerto}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=puerto)