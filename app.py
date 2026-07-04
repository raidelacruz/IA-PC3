""" 
UTP · Java Script Avanzado · Semana 15 
API de modelos IA para evaluación React + Spring Boot + Ciencia de Datos 
  
Cómo ejecutar: 
1) python -m venv venv 
2) venv\Scripts\activate          (Windows) 
   source venv/bin/activate       (Mac/Linux) 
3) pip install -r requirements.txt 
4) uvicorn app:app --reload --port 8001 
  
Swagger: 
http://localhost:8001/docs 
  
Idea de arquitectura: 
React -> Spring Boot -> esta API Python -> modelo entrenado -> respuesta JSON 
""" 
  
from typing import Dict, List
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
  
  
app = FastAPI( 
    title="UTP IA Models API - Java Script Avanzado Semana 15", 
    version="1.0.0", 
    description="Modelos sintéticos entrenados en memoria para proyectos React + Spring Boot." 
) 
  
RANDOM_STATE = 42 
  
  
class ModelPack: 
    def __init__(self, features: List[str], labels: List[str], model: RandomForestClassifier, encoder: LabelEncoder): 
        self.features = features 
        self.labels = labels 
        self.model = model 
        self.encoder = encoder 
  
    def predict(self, values: Dict[str, float]) -> Dict: 
        x = np.array([[float(values[f]) for f in self.features]]) 
        pred = self.model.predict(x)[0] 
        probs = self.model.predict_proba(x)[0] 
        label = self.encoder.inverse_transform([pred])[0] 
        confidence = round(float(np.max(probs)), 4) 
        ranking = sorted( 
            [ 
                {"clase": self.encoder.inverse_transform([i])[0], "probabilidad": round(float(p), 4)} 
                for i, p in enumerate(probs) 
            ], 
            key=lambda item: item["probabilidad"], 
            reverse=True 
        ) 
        return {"label": label, "confidence": confidence, "ranking": ranking} 
  
  
MODELS: Dict[str, ModelPack] = {} 
  
  
# ========================================================== 
# 1) UTP RISK AI - Riesgo académico 
# ========================================================== 
class UtpRiskRequest(BaseModel): 
    promedio_actual: float = Field(..., ge=0, le=20) 
    asistencia_pct: float = Field(..., ge=0, le=100) 
    tareas_entregadas_pct: float = Field(..., ge=0, le=100) 
    participacion_pct: float = Field(..., ge=0, le=100) 
    horas_estudio_semana: float = Field(..., ge=0, le=40) 
    nota_pc_anterior: float = Field(..., ge=0, le=20) 
  
  
def label_utp_risk(row): 
    promedio, asistencia, tareas, participacion, horas, pc = row 
    score = ( 
        promedio * 3.2 
        + pc * 2.5 
        + asistencia * 0.18 
        + tareas * 0.16 
        + participacion * 0.10 
 + horas * 1.20 
    ) 
    if score < 105: 
        return "RIESGO_ALTO" 
    if score < 150: 
        return "RIESGO_MEDIO" 
    return "RIESGO_BAJO" 
  
  
def generate_utp_risk_data(): 
    rng = np.random.default_rng(RANDOM_STATE) 
    X = [] 
    y = [] 
    for _ in range(260): 
        row = [ 
            rng.uniform(5, 20),      # promedio_actual 
            rng.uniform(40, 100),    # asistencia_pct 
            rng.uniform(20, 100),    # tareas_entregadas_pct 
            rng.uniform(10, 100),    # participacion_pct 
            rng.uniform(0, 25),      # horas_estudio_semana 
            rng.uniform(4, 20),      # nota_pc_anterior 
        ] 
        X.append(row) 
        y.append(label_utp_risk(row)) 
    return np.array(X), np.array(y) 
  
  
def recommendation_utp(payload: UtpRiskRequest, label: str) -> List[str]: 
    tips = [] 
    if payload.asistencia_pct < 75: 
        tips.append("Subir asistencia: el sistema detecta pérdida de continuidad académica.") 
    if payload.tareas_entregadas_pct < 70: 
        tips.append("Regularizar entregas: las tareas evidencian práctica sostenida.") 
    if payload.horas_estudio_semana < 6: 
        tips.append("Aumentar horas de práctica semanal con ejercicios guiados.") 
    if payload.nota_pc_anterior < 12: 
        tips.append("Reforzar puntos débiles de la práctica anterior antes de la evaluación final.") 
    if not tips: 
        tips.append("Mantener el ritmo y resolver un reto adicional de integración React + servicios.") 
    return tips 
  
  
# ========================================================== 
# 2) FRAUD SHIELD - Transacciones sospechosas 
# ========================================================== 
class FraudRequest(BaseModel): 
    monto: float = Field(..., ge=1, le=20000) 
    hora_24: int = Field(..., ge=0, le=23) 
    intentos_previos: int = Field(..., ge=0, le=10) 
    antiguedad_cliente_meses: int = Field(..., ge=0, le=120) 
    dispositivo_nuevo: int = Field(..., ge=0, le=1, description="0=no, 1=sí") 
    pais_riesgo: int = Field(..., ge=0, le=1, description="0=bajo, 1=alto") 
    compras_ultima_hora: int = Field(..., ge=0, le=20) 
  
  
def label_fraud(row): 
    monto, hora, intentos, antig, disp, pais, compras = row 
    risk = 0 
    risk += 35 if monto > 5000 else 12 if monto > 1500 else 0 
    risk += 18 if hora <= 5 else 0 
    risk += intentos * 8 
    risk += 18 if antig < 3 else 8 if antig < 12 else 0 
    risk += 20 if disp == 1 else 0 
    risk += 18 if pais == 1 else 0 
    risk += 12 if compras > 5 else 0 
    if risk >= 70: 
        return "FRAUDE_PROBABLE" 
    if risk >= 40: 
        return "REVISION_MANUAL" 
    return "TRANSACCION_SEGURA" 
  
  
def generate_fraud_data(): 
    rng = np.random.default_rng(RANDOM_STATE + 1) 
    X = [] 
    y = [] 
    for _ in range(320): 
        row = [ 
            rng.uniform(10, 15000), 
            rng.integers(0, 24), 
            rng.integers(0, 8), 
            rng.integers(0, 96), 
            rng.integers(0, 2), 
            rng.integers(0, 2), 
            rng.integers(0, 12), 
        ] 
        X.append(row) 
        y.append(label_fraud(row)) 
    return np.array(X), np.array(y) 
  
  
def recommendation_fraud(payload: FraudRequest, label: str) -> List[str]: 
    if label == "FRAUDE_PROBABLE": 
        return ["Bloquear temporalmente la operación.", "Solicitar autenticación reforzada.", "Registrar alerta para revisión del analista."] 
    if label == "REVISION_MANUAL": 
        return ["Solicitar confirmación adicional del cliente.", "Comparar con historial de compras.", "Permitir solo si la verificación es exitosa."] 
    return ["Autorizar operación y continuar monitoreo pasivo."] 
  
  
# ========================================================== 
# 3) CYBER SENTINEL - Severidad de incidente de ciberseguridad 
# ========================================================== 
class CyberRequest(BaseModel): 
    intentos_login_fallidos: int = Field(..., ge=0, le=200) 
    puertos_abiertos: int = Field(..., ge=0, le=100) 
    vulnerabilidades_criticas: int = Field(..., ge=0, le=20) 
    trafico_anomalo_pct: float = Field(..., ge=0, le=100) 
    equipos_afectados: int = Field(..., ge=0, le=500) 
    parcheado_pct: float = Field(..., ge=0, le=100) 
  
  
def label_cyber(row): 
    logins, ports, vulns, traf, equipos, patch = row 
    risk = logins * 0.25 + ports * 0.45 + vulns * 7 + traf * 0.8 + equipos * 0.18 + (100 - patch) * 0.65 
    if risk >= 115: 
        return "CRITICO" 
    if risk >= 75: 
        return "ALTO" 
    if risk >= 40: 
        return "MEDIO" 
    return "BAJO" 
  
  
def generate_cyber_data(): 
    rng = np.random.default_rng(RANDOM_STATE + 2) 
    X = [] 
    y = [] 
    for _ in range(300): 
        row = [ 
            rng.integers(0, 180), 
            rng.integers(1, 80), 
            rng.integers(0, 18), 
            rng.uniform(0, 100), 
            rng.integers(1, 250), 
            rng.uniform(30, 100), 
        ] 
        X.append(row) 
        y.append(label_cyber(row)) 
    return np.array(X), np.array(y) 
  
  
def recommendation_cyber(payload: CyberRequest, label: str) -> List[str]: 
    tips = [] 
    if payload.vulnerabilidades_criticas >= 5: 
        tips.append("Priorizar parcheo de vulnerabilidades críticas.") 
    if payload.trafico_anomalo_pct >= 50: 
        tips.append("Analizar tráfico y aislar segmentos con comportamiento anómalo.") 
    if payload.intentos_login_fallidos >= 50: 
        tips.append("Activar bloqueo temporal y revisión de credenciales.") 
    if payload.parcheado_pct < 70: 
        tips.append("Elevar porcentaje de equipos parchados antes de cerrar el incidente.") 
    if not tips: 
        tips.append("Continuar monitoreo y documentar evidencias del incidente.") 
    return tips 
  
  
# ========================================================== 
# 4) SMART STOCK 360 - Demanda e inventario 
# ========================================================== 
class StockRequest(BaseModel): 
    precio: float = Field(..., ge=1, le=5000) 
    stock_actual: int = Field(..., ge=0, le=10000) 
    ventas_7d: int = Field(..., ge=0, le=5000) 
    descuento_pct: float = Field(..., ge=0, le=90) 
    temporada: int = Field(..., ge=0, le=2, description="0=normal, 1=campaña, 2=feriado/alta demanda") 
    dias_sin_reabastecer: int = Field(..., ge=0, le=120) 
    rating_producto: float = Field(..., ge=1, le=5) 
  
  
def label_stock(row): 
    precio, stock, ventas, desc, temp, dias, rating = row 
    demand = ventas * 0.09 + desc * 0.7 + temp * 18 + rating * 8 - precio * 0.006 + dias * 0.22 
    cobertura = stock / max(ventas, 1) 
    if demand >= 70 and cobertura < 3: 
        return "DEMANDA_ALTA_REABASTECER" 
    if demand >= 45: 
        return "DEMANDA_MEDIA_MONITOREAR" 
    return "DEMANDA_BAJA_OPTIMIZAR" 
  
  
def generate_stock_data(): 
    rng = np.random.default_rng(RANDOM_STATE + 3) 
    X = [] 
    y = [] 
    for _ in range(300): 
        row = [ 
            rng.uniform(10, 3500), 
            rng.integers(0, 5000), 
            rng.integers(0, 1200), 
            rng.uniform(0, 70),
             rng.integers(0, 3), 
            rng.integers(0, 90), 
            rng.uniform(1, 5), 
        ] 
        X.append(row) 
        y.append(label_stock(row)) 
    return np.array(X), np.array(y) 
  
  
def recommendation_stock(payload: StockRequest, label: str) -> List[str]: 
    if label == "DEMANDA_ALTA_REABASTECER": 
        return ["Reabastecer en las próximas 48 horas.", "Mantener promoción si el margen lo permite.", "Mostrar alerta roja en el dashboard."] 
    if label == "DEMANDA_MEDIA_MONITOREAR": 
        return ["Monitorear ventas diarias.", "Preparar pedido moderado si la temporada continúa.", "Comparar contra productos sustitutos."] 
    return ["Reducir compra inmediata.", "Evaluar promoción o bundle.", "Evitar sobrestock y revisar precio."] 
  
  
# ========================================================== 
# 5) TALENT MATCH AI - Perfil laboral tecnológico 
# ========================================================== 
class TalentRequest(BaseModel): 
    javascript: int = Field(..., ge=0, le=100) 
    react: int = Field(..., ge=0, le=100) 
    spring_boot: int = Field(..., ge=0, le=100) 
    python_datos: int = Field(..., ge=0, le=100) 
    sql: int = Field(..., ge=0, le=100) 
    experiencia_proyectos: int = Field(..., ge=0, le=10) 
    preferencia: int = Field(..., ge=0, le=3, description="0=front, 1=back, 2=datos, 3=fullstack") 
  
  
def label_talent(row): 
    js, react, spring, py, sql, exp, pref = row 
    front = js * 0.35 + react * 0.45 + exp * 3 + (12 if pref == 0 else 0) 
    back = spring * 0.50 + sql * 0.25 + js * 0.10 + exp * 3 + (12 if pref == 1 else 0) 
    data = py * 0.55 + sql * 0.25 + exp * 2 + (12 if pref == 2 else 0) 
    full = (js + react + spring + sql) * 0.20 + exp * 4 + (12 if pref == 3 else 0) 
    scores = {"FRONTEND_REACT": front, "BACKEND_SPRING": back, "DATA_ANALYST_JUNIOR": data, "FULLSTACK_JUNIOR": full} 
    return max(scores, key=scores.get) 
  
  
def generate_talent_data(): 
    rng = np.random.default_rng(RANDOM_STATE + 4) 
    X = [] 
    y = [] 
    for _ in range(320): 
        row = [ 
            rng.integers(10, 101), 
            rng.integers(10, 101), 
            rng.integers(0, 101), 
            rng.integers(0, 101), 
            rng.integers(10, 101), 
            rng.integers(0, 11), 
            rng.integers(0, 4), 
        ] 
        X.append(row) 
        y.append(label_talent(row)) 
    return np.array(X), np.array(y) 
  
  
def recommendation_talent(payload: TalentRequest, label: str) -> List[str]: 
    tips = { 
        "FRONTEND_REACT": ["Construir un dashboard React con rutas, hooks y consumo REST.", "Reforzar diseño de componentes y manejo de estado."], 
        "BACKEND_SPRING": ["Crear endpoints REST sólidos y documentación Swagger.", "Reforzar validaciones, DTOs y conexión con servicios externos."], 
        "DATA_ANALYST_JUNIOR": ["Explicar variables, modelo y métricas de predicción.", "Reforzar Python, limpieza de datos y visualización."], 
        "FULLSTACK_JUNIOR": ["Integrar React + Spring Boot + Python de extremo a extremo.", "Reforzar despliegue y manejo de errores entre capas."] 
    } 
    return tips[label] 
  
  
# ========================================================== 
# Entrenamiento común 
# ========================================================== 
def train_pack(features: List[str], X: np.ndarray, y: np.ndarray) -> ModelPack: 
    encoder = LabelEncoder() 
    y_enc = encoder.fit_transform(y) 
    model = RandomForestClassifier( 
        n_estimators=180, 
        max_depth=7, 
        random_state=RANDOM_STATE, 
        class_weight="balanced_subsample" 
    ) 
    model.fit(X, y_enc) 
    return ModelPack(features, list(encoder.classes_), model, encoder) 
  
  
@app.on_event("startup") 
def startup_train_models(): 
    X, y = generate_utp_risk_data() 
    MODELS["utp-risk"] = train_pack( 
        ["promedio_actual", "asistencia_pct", "tareas_entregadas_pct", "participacion_pct", "horas_estudio_semana", "nota_pc_anterior"], X, y 
    ) 

    X, y = generate_fraud_data()
    MODELS["fraud-shield"] = train_pack( 
        ["monto", "hora_24", "intentos_previos", "antiguedad_cliente_meses", "dispositivo_nuevo", "pais_riesgo", "compras_ultima_hora"], X, y 
    ) 
  
    X, y = generate_cyber_data() 
    MODELS["cyber-sentinel"] = train_pack( 
        ["intentos_login_fallidos", "puertos_abiertos", "vulnerabilidades_criticas", "trafico_anomalo_pct", "equipos_afectados", "parcheado_pct"], X, y 
    ) 
  
    X, y = generate_stock_data() 
    MODELS["smart-stock"] = train_pack( 
        ["precio", "stock_actual", "ventas_7d", "descuento_pct", "temporada", "dias_sin_reabastecer", "rating_producto"], X, y 
    ) 
  
    X, y = generate_talent_data() 
    MODELS["talent-match"] = train_pack( 
        ["javascript", "react", "spring_boot", "python_datos", "sql", "experiencia_proyectos", "preferencia"], X, y 
    ) 
  
  
@app.get("/health") 
def health(): 
    return {"status": "ok", "modelos_cargados": list(MODELS.keys())} 
  
  
@app.get("/metadata") 
def metadata(): 
    return { 
        key: {"features": pack.features, "labels": pack.labels} 
        for key, pack in MODELS.items() 
    } 
  
  
@app.post("/predict/utp-risk") 
def predict_utp_risk(payload: UtpRiskRequest): 
    values = payload.dict() 
    pred = MODELS["utp-risk"].predict(values) 
    return { 
        "caso": "UTP RiskAI", 
        "prediccion": pred["label"], 
        "confianza": pred["confidence"], 
        "ranking": pred["ranking"], 
        "recomendaciones": recommendation_utp(payload, pred["label"]), 
        "entrada": values 
    } 
  
  
@app.post("/predict/fraud-shield") 
def predict_fraud(payload: FraudRequest): 
    values = payload.dict() 
    pred = MODELS["fraud-shield"].predict(values) 
    return { 
        "caso": "FraudShield", 
        "prediccion": pred["label"], 
        "confianza": pred["confidence"], 
        "ranking": pred["ranking"], 
        "recomendaciones": recommendation_fraud(payload, pred["label"]), 
        "entrada": values 
    } 
  
  
@app.post("/predict/cyber-sentinel") 
def predict_cyber(payload: CyberRequest): 
    values = payload.dict() 
    pred = MODELS["cyber-sentinel"].predict(values) 
    return { 
        "caso": "CyberSentinel", 
        "prediccion": pred["label"], 
        "confianza": pred["confidence"], 
        "ranking": pred["ranking"], 
        "recomendaciones": recommendation_cyber(payload, pred["label"]), 
        "entrada": values 
    } 
  
  
@app.post("/predict/smart-stock") 
def predict_stock(payload: StockRequest): 
    values = payload.dict() 
    pred = MODELS["smart-stock"].predict(values) 
    return { 
        "caso": "SmartStock360", 
        "prediccion": pred["label"], 
        "confianza": pred["confidence"], 
        "ranking": pred["ranking"], 
        "recomendaciones": recommendation_stock(payload, pred["label"]), 
        "entrada": values 
    } 
  
  
@app.post("/predict/talent-match") 
def predict_talent(payload: TalentRequest): 
    values = payload.dict() 
    pred = MODELS["talent-match"].predict(values) 
    return { 
"caso": "TalentMatchAI", 
"prediccion": pred["label"], 
"confianza": pred["confidence"], 
"ranking": pred["ranking"], 
"recomendaciones": recommendation_talent(payload, pred["label"]), 
"entrada": values 
}