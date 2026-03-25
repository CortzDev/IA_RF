import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from sqlalchemy import create_engine
import time
from sklearn.ensemble import RandomForestRegressor
import os
import joblib # 👈 NUEVA LIBRERÍA PARA LA MEMORIA

# URL de tu base de datos (Recuerda usar variables de entorno en el futuro)
DB_URL = "postgresql://postgres:VaLqxGBzdzZmBTddchzzryKgNeQmoPfI@switchback.proxy.rlwy.net:14573/railway?sslmode=require"
sensores = ['temp_current', 'humidity_value', 'co2_value', 'pm25_value', 'pm10']

# 👈 RUTA DEL ARCHIVO DEL MODELO
# En tu PC local se guardará en la misma carpeta del script.
# Cuando lo subas a Railway, cambiaremos esto a: '/app/data/cerebro_sensores.joblib'
RUTA_MODELO = '/app/data/cerebro_sensores.joblib'

def cargar_y_entrenar(engine):
    print("🧠 Entrenando el modelo con los datos más recientes...")
    query = "SELECT * FROM sensor_metrics ORDER BY recorded_at ASC"
    df = pd.read_sql(query, engine)
    
    # Construir la "Máquina del tiempo"
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
    
    # 👈 GUARDAR EL CEREBRO FÍSICAMENTE
    joblib.dump(modelo, RUTA_MODELO)
    print(f"💾 Memoria guardada exitosamente en: {RUTA_MODELO}")
    
    # Devolvemos el modelo, las columnas que usa, y el ÚLTIMO registro válido
    verdadero_presente = df.iloc[-1:][columnas_X].copy()
    return modelo, columnas_X, verdadero_presente

def fusion_ia_monitor():
    engine = create_engine(DB_URL)
    
    # 👈 LÓGICA DE MEMORIA: ¿Existe un cerebro guardado?
    if os.path.exists(RUTA_MODELO):
        print(f"📦 Cargando IA desde archivo existente ({RUTA_MODELO})...")
        modelo = joblib.load(RUTA_MODELO)
        
        # Necesitamos reconstruir 'datos_actuales' y 'columnas_X' para saber dónde nos quedamos
        query = f"SELECT * FROM sensor_metrics ORDER BY recorded_at DESC LIMIT 2"
        df_reciente = pd.read_sql(query, engine).sort_values(by='recorded_at', ascending=True)
        for s in sensores:
            df_reciente[f'{s}_pasado'] = df_reciente[s].shift(1)
        
        columnas_X = sensores + [f'{s}_pasado' for s in sensores]
        datos_actuales = df_reciente.iloc[-1:][columnas_X].copy()
        
    else:
        print("🌱 No hay IA guardada. Iniciando entrenamiento desde cero...")
        modelo, columnas_X, datos_actuales = cargar_y_entrenar(engine)
    
    # Primera predicción 
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
                print("\n" + "="*50)
                print("✨ ¡NUEVO DATO DETECTADO!")
                
                print("\n📊 COMPARATIVA: PREDICCIÓN VS REALIDAD")
                for i, sensor in enumerate(sensores):
                    valor_real = float(registro[sensor])
                    valor_predicho = float(prediccion_actual[i])
                    diferencia = abs(valor_real - valor_predicho)
                    
                    # Lógica de acierto (5% de margen de error)
                    estado_booleano = bool(diferencia <= (abs(valor_real) * 0.05))
                    estado_txt = "✅ Preciso" if estado_booleano else "⚠️ Necesita ajuste"
                    
                    print(f"{sensor.upper()}:")
                    print(f"  -> Predijo: {valor_predicho:.2f} | Real: {valor_real:.2f} | Error: {diferencia:.2f} ({estado_txt})")
                    
                    # 👈 GUARDAR EL RESULTADO EN LA BASE DE DATOS
                    cursor.execute("""
                        INSERT INTO predicciones_log (sensor, valor_real, valor_predicho, margen_error, acertado)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (sensor, valor_real, valor_predicho, diferencia, estado_booleano))
                
                # Confirmar los guardados en la base de datos
                conn.commit() 
                
                # Preparar la predicción para el SIGUIENTE registro
                nuevos_datos_X = pd.DataFrame([{
                    **{s: registro[s] for s in sensores}, 
                    **{f'{s}_pasado': datos_actuales[s].values[0] for s in sensores} 
                }], columns=columnas_X)
                
                prediccion_actual = modelo.predict(nuevos_datos_X)[0]
                datos_actuales = nuevos_datos_X
                last_id = registro['id']
                nuevos_registros_count += 1
                
                print(f"\n🔮 Siguiente predicción calculada. Esperando nuevo dato... ({nuevos_registros_count}/{LIMITE_PARA_REENTRENAR} para reentrenar)")
                print("="*50)
                
                # Lógica de "Aprendizaje" y sobrescritura de memoria
                if nuevos_registros_count >= LIMITE_PARA_REENTRENAR:
                    print("\n🔄 Alcanzado el límite de datos nuevos. La IA está aprendiendo...")
                    modelo, columnas_X, datos_actuales = cargar_y_entrenar(engine) # Esto sobrescribe el archivo .joblib
                    prediccion_actual = modelo.predict(datos_actuales)[0]
                    nuevos_registros_count = 0 
                
            time.sleep(2)

    except KeyboardInterrupt:
        print("\n🛑 Monitoreo detenido por el usuario.")
    except Exception as e:
        print(f"\n❌ Ocurrió un error: {e}")
    finally:
        if cursor: cursor.close()
        if conn: conn.close()

if __name__ == "__main__":
    fusion_ia_monitor()