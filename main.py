import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from sqlalchemy import create_engine
import time
from sklearn.ensemble import RandomForestRegressor
import os
import joblib
import threading
from flask import Flask, jsonify, send_file
from flask_cors import CORS
import logging
import sys # <-- Importamos sys

# ==========================================
# CONFIGURACIÓN DE LOGS (Corregido para Railway)
# ==========================================
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)] # <-- Forzamos el canal normal
)
logger = logging.getLogger(__name__)

# ==========================================
# CONFIGURACIÓN DE APP Y DB
# ==========================================
app = Flask(__name__)
CORS(app)

# Toma la URL de Railway automáticamente, o usa la tuya por defecto
DB_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:VaLqxGBzdzZmBTddchzzryKgNeQmoPfI@switchback.proxy.rlwy.net:14573/railway?sslmode=require")
sensores = ['temp_current', 'humidity_value', 'co2_value', 'pm25_value', 'pm10']

# --- AQUÍ ESTÁ EL CAMBIO PARA TU VOLUMEN ---
RUTA_MODELO = '/app/data/cerebro_sensores.joblib'

# Por seguridad, le decimos a Python que cree la carpeta si por alguna razón Railway no la ha inicializado
os.makedirs('/app/data', exist_ok=True)

# ==========================================
# LÓGICA DE INTELIGENCIA ARTIFICIAL (IA)
# ==========================================

def cargar_y_entrenar(engine):
    logger.info("🧠 IA: Entrenando modelo con datos históricos...")
    try:
        query = "SELECT * FROM sensor_metrics ORDER BY recorded_at ASC"
        df = pd.read_sql(query, engine)
        
        if df.empty or len(df) < 10:
            logger.warning("⚠️ IA: Datos insuficientes (< 10 registros).")
            return None, None, None

        for s in sensores:
            df[f'{s}_pasado'] = df[s].shift(1)
            df[f'{s}_futuro'] = df[s].shift(-1)
        
        df_train = df.dropna()
        columnas_X = sensores + [f'{s}_pasado' for s in sensores]
        columnas_Y = [f'{s}_futuro' for s in sensores]
        
        modelo = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        modelo.fit(df_train[columnas_X], df_train[columnas_Y])
        
        joblib.dump(modelo, RUTA_MODELO)
        datos_actuales = df.iloc[-1:][columnas_X].copy()
        logger.info("✅ IA: Modelo entrenado y guardado correctamente.")
        return modelo, columnas_X, datos_actuales
    except Exception as e:
        logger.error(f"❌ IA Error en entrenamiento: {e}")
        return None, None, None

def fusion_ia_monitor():
    """Hilo que vigila la DB y genera predicciones"""
    logger.info("🚀 IA: Iniciando monitor en segundo plano...")
    engine = create_engine(DB_URL)
    proximo_latido = time.time() + 300
    
    modelo = None
    columnas_X = None
    prediccion_actual = None
    last_id = None

    while True:
        try:
            # 1. CARGAR MODELO
            if modelo is None:
                if os.path.exists(RUTA_MODELO):
                    modelo = joblib.load(RUTA_MODELO)
                    columnas_X = sensores + [f'{s}_pasado' for s in sensores]
                    query = "SELECT * FROM sensor_metrics ORDER BY recorded_at DESC LIMIT 2"
                    df_reciente = pd.read_sql(query, engine).sort_values(by='recorded_at')
                    if len(df_reciente) >= 2:
                        for s in sensores:
                            df_reciente[f'{s}_pasado'] = df_reciente[s].shift(1)
                        datos_actuales = df_reciente.iloc[-1:][columnas_X].copy()
                        prediccion_actual = modelo.predict(datos_actuales)[0]
                    logger.info("✅ IA: Modelo cargado desde disco.")
                else:
                    modelo, columnas_X, datos_actuales = cargar_y_entrenar(engine)
                    if modelo:
                        prediccion_actual = modelo.predict(datos_actuales)[0]

            # 2. CONEXIÓN Y MONITOREO DIRECTO
            conn = psycopg2.connect(DB_URL)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            if last_id is None:
                cursor.execute("SELECT id FROM sensor_metrics ORDER BY id DESC LIMIT 1;")
                res = cursor.fetchone()
                last_id = res['id'] if res else 0
                logger.info(f"📡 IA: Monitoreando desde ID actual: {last_id}")

            while True:
                cursor.execute("SELECT * FROM sensor_metrics WHERE id > %s ORDER BY id ASC;", (last_id,))
                nuevos = cursor.fetchall()

                if nuevos and prediccion_actual is not None:
                    for registro in nuevos:
                        logger.info(f"✨ IA: Nueva lectura ID {registro['id']} - Comparando y guardando...")
                        
                        # Comparamos e insertamos log para cada sensor
                        for i, s in enumerate(sensores):
                            v_real = float(registro[s] or 0)
                            v_pred = float(prediccion_actual[i])
                            error = abs(v_real - v_pred)
                            acierto = bool(error <= (abs(v_real) * 0.05)) # Margen del 5%
                            
                            cursor.execute(
                                "INSERT INTO predicciones_log (sensor, valor_real, valor_predicho, margen_error, acertado) VALUES (%s, %s, %s, %s, %s)",
                                (s, v_real, v_pred, error, acierto)
                            )
                        
                        # --- ¡AQUÍ ESTÁ LA MAGIA QUE FALTABA! ---
                        # Forzamos a PostgreSQL a escribir los datos en el disco duro inmediatamente
                        conn.commit()
                        logger.info(f"💾 IA: Resultados guardados en la BD para ID {registro['id']}.")
                        
                        # Calcular predicción para el PRÓXIMO evento
                        cursor.execute("SELECT * FROM sensor_metrics WHERE id < %s ORDER BY id DESC LIMIT 1", (registro['id'],))
                        anterior = cursor.fetchone()
                        if anterior:
                            inputs = [float(registro[s] or 0) for s in sensores] + [float(anterior[s] or 0) for s in sensores]
                            df_X = pd.DataFrame([inputs], columns=columnas_X)
                            prediccion_actual = modelo.predict(df_X)[0]
                            logger.info(f"🔮 IA: Nueva predicción calculada para el futuro.")

                        last_id = registro['id']
                        proximo_latido = time.time() + 300

                elif time.time() >= proximo_latido:
                    logger.info("💓 IA: Sistema estable. Esperando nuevas lecturas...")
                    proximo_latido = time.time() + 300
                
                time.sleep(15) # Pausa de escaneo

        except Exception as e:
            logger.error(f"❌ IA: Error detectado: {e}")
            modelo = None # Forzar recarga completa en caso de error grave
            try:
                if 'conn' in locals():
                    conn.close()
            except:
                pass
            time.sleep(20) # Esperar antes de intentar reconectar

# ==========================================
# RUTAS DE LA API (FLASK)
# ==========================================

@app.route('/')
def home():
    return jsonify({"status": "online", "service": "Sensor IA Monitor"})

@app.route('/api/estadisticas')
def stats():
    try:
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT SUM(CASE WHEN acertado THEN 1 ELSE 0 END) as aciertos, COUNT(*) as total FROM predicciones_log")
        res = cur.fetchone()
        cur.close(); conn.close()
        t = res['total'] or 0
        a = res['aciertos'] or 0
        return jsonify({"aciertos": int(a), "fallos": int(t - a), "total": int(t)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/descargar-cerebro', methods=['GET'])
def descargar_cerebro():
    """Descarga el archivo del modelo de IA (.joblib) directamente a tu PC"""
    try:
        # Verificamos si el archivo realmente existe antes de enviarlo
        if os.path.exists(RUTA_MODELO):
            # send_file empaqueta el archivo y fuerza la descarga en el navegador
            return send_file(
                RUTA_MODELO, 
                as_attachment=True, 
                download_name='mi_cerebro_entrenado.joblib'
            )
        else:
            return jsonify({"success": False, "error": "El modelo aún no se ha generado en el servidor."}), 404
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
# ==========================================
# ARRANQUE
# ==========================================

# Iniciamos el hilo de la IA en segundo plano antes de arrancar Flask
t = threading.Thread(target=fusion_ia_monitor, daemon=True)
t.start()

if __name__ == "__main__":
    # Puerto dinámico para Railway, por defecto 5000 en local
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"🌐 Servidor Flask iniciando en puerto {port}")
    app.run(host="0.0.0.0", port=port, debug=False)