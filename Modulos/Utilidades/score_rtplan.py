"""
score_rtplan.py
─────────────────────────────────────────────────────────────────────────────
Score a single RTPLAN .dcm file using a trained ensemble pipeline.

Public API
──────────
    extract_features(archivo, id3="Manual")
        → df : pd.DataFrame  (one row per beam)

    score_rtplan(archivo, pipeline, id3="Manual")
        → df_result : pd.DataFrame  (features + scores, one row per beam)

    score_rtplan_batch(archivos, pipeline, id3="Manual")
        → df_result : pd.DataFrame  (all beams from all files)

Usage
─────
    from score_rtplan import score_rtplan

    result = score_rtplan("path/to/plan.dcm", pipeline)
    print(result[["ID", "ID2", "Composite_score", "Risk_percentile", "Composite_label"]])

Version
─────
    Numero: 2
    Fecha:  10/07/2026
    Cambios: Añade complexity index

"""

from __future__ import annotations

import os
import sys
import warnings
from typing import Literal
import numpy as np

import pandas as pd
import shap

from IPython.display import HTML, display
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Ruta base del proyecto — sube dos niveles desde Utilidades/ hasta Modulos/
_MODULOS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _MODULOS_PATH not in sys.path:
    sys.path.insert(0, _MODULOS_PATH)

from Clases.clases import RTPLAN, BEAM
from Utilidades.CONSTANTES import (
    COMPLEXITY_DIRECTION,
    FEATURES,
    IDS,
    SCORES
)

def _is_notebook() -> bool:
    """True solo si hay un kernel de IPython/Jupyter corriendo (Colab, notebook).
    En un servidor (ej. Streamlit en Render) devuelve False, evitando llamadas
    a display() que ahí no muestran nada y solo ensucian los logs."""
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# PLAN GROUP CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

_PLAN_REGLAS: dict[str, str] = {
    "MAMA":      r"MD|MI|MAMA",
    "PROSTATA":  r"PROSTATA|PTTA|PELVISPROSTATA|PELVISPTTA",
    "PELVIS":    r"PELVIS|RECTO",
    "METSOSEA":  r"METS|MTTS|MTS",
    "MEDIASTINO":r"MEDIASTINO|ESOFAGO",
    "SNC":       r"SNC|CRANEORAQUIS",
    "CYC":       r"CYC|HN|LARINGE",
}


def _assign_plan_grupo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derives PLAN (from filename) and PLAN_GRUPO (anatomical group)
    from the ID4 column (filename).
    """
    df = df.copy()

    df["PLAN"] = (
        df["ID4"]
        .str.upper()
        .str.extract(r"_(.*?)\.DCM$", expand=False)
        .str.replace(r"^ET\d+", "", regex=True)
    )

    df["PLAN_GRUPO"] = "OTROS"
    for grupo, patron in _PLAN_REGLAS.items():
        mask = df["PLAN"].str.contains(patron, na=False, regex=True)
        df.loc[mask, "PLAN_GRUPO"] = grupo

    return df


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(
    archivo: str,
    id3: str = "Manual",
) -> pd.DataFrame:
    """
    Parse a single RTPLAN .dcm file and return a DataFrame with one row
    per beam, including metadata columns ID, ID2, ID3, ID4, PLAN, PLAN_GRUPO.

    Parameters
    ----------
    archivo : str
        Path to the .dcm file.
    id3 : str
        Label for the ID3 metadata column (e.g. "Manual", "Validation").

    Returns
    -------
    df : pd.DataFrame or None if parsing fails.
    """
    nombre_archivo = os.path.basename(archivo)

    try:
        rtplan    = RTPLAN(archivo)
        pat_id    = rtplan.header.get("PatientID", "unknown")
        plan_name = rtplan.header.get("RTPlanName", "unknown")
        num_beams = len(rtplan.beams)

        lista_dfs = []
        for beam_idx in range(1, num_beams + 1):
            beam      = rtplan.beams[beam_idx]
            beam_name = beam.info.get("BeamName", f"Beam_{beam_idx}")
            atributos = beam.atributos_beam.copy()

            atributos["ID"]  = pat_id
            atributos["ID2"] = beam_name
            atributos["ID3"] = id3
            atributos["ID4"] = nombre_archivo

            lista_dfs.append(pd.DataFrame([atributos]))

        df = pd.concat(lista_dfs, ignore_index=True)
        df = _assign_plan_grupo(df)

        return df

    except Exception as e:
        warnings.warn(f"Error procesando {nombre_archivo}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# BUILD SHAP
# ─────────────────────────────────────────────────────────────────────────────

def build_shap(
    df_caso: pd.DataFrame,
    resultados: pd.DataFrame,
    pipeline: dict
)-> pd.DataFrame:

  import shap
  import warnings

  df_train       = pipeline['df_train']
  X_train_scaled = pipeline['X_train_scaled']
  modelo_if      = pipeline['modelos']['IF']
  scaler         = pipeline['scaler']

  features = FEATURES

  X_caso_scaled = scaler.transform(df_caso[features])

  with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      explainer   = shap.TreeExplainer(modelo_if, data=X_train_scaled)
      shap_values = explainer.shap_values(X_caso_scaled)

  train_mean  = df_train[features].mean()
  train_std   = df_train[features].std()
  caso_scores = modelo_if.decision_function(X_caso_scaled)

  rows = []

  for i, (idx, row) in enumerate(df_caso.iterrows()):

    sv = shap_values[i]
    av = row[features]
    df_beam_shap = pd.DataFrame({
        "Study_idx":    idx,
        "Feature":      features,
        "SHAP":         sv,
        "ABS_SHAP":     np.abs(sv),
        "Actual_value": av.values,
        "Train_mean":   train_mean.values,
        "Train_std":    train_std.values,
        "Composite_score": resultados.loc[idx, "Composite_score"],
        "Risk_percentile": resultados.loc[idx, "Risk_percentile"],
        "Composite_label": resultados.loc[idx, "Composite_label"]
    })

    df_beam_shap['Zscore'] = (df_beam_shap['Actual_value'] - df_beam_shap['Train_mean']) / df_beam_shap['Train_std']
    df_beam_shap['Anomalous_score'] =  caso_scores[i]
    rows.append(df_beam_shap.sort_values("ABS_SHAP", ascending = False))

  df_shap_caso = pd.concat(rows, ignore_index=True)
  df_shap_caso['Feature_direction'] = df_shap_caso['Feature'].map(COMPLEXITY_DIRECTION)
  df_shap_caso['Complexity'] = np.sign(df_shap_caso['Zscore']) * df_shap_caso['Feature_direction']

  return df_shap_caso


# ─────────────────────────────────────────────────────────────────────────────
# SCORING
# ─────────────────────────────────────────────────────────────────────────────

def _build_output_row(df_features: pd.DataFrame, resultados_val: pd.DataFrame) -> pd.DataFrame:
    """Merge features with scores side by side."""
    available_meta = [c for c in IDS if c in df_features.columns]

    return pd.concat(
        [df_features[available_meta].reset_index(drop=True),
         resultados_val.reset_index(drop=True)],
        axis=1,
    )



def score_rtplan(
    archivo: str,
    pipeline: dict,
    id3: str = "Manual",
    return_features = False,
    plot_features = False,
    theme: str = 'dark' # o light
) -> pd.DataFrame | None:
    """
    Extract features from a single .dcm file and score it with the
    trained ensemble pipeline.

    Parameters
    ----------
    archivo : str
        Path to the RTPLAN .dcm file.
    pipeline : dict
        Trained pipeline from fit_ensemble().
    id3 : str
        Label for the ID3 metadata column.
    return_features: Boolean
        devuelve features

    Returns
    -------
    pipeline_caso : dict
        pipeline actualizado con resultado, df_caso y df_shap
        Returns None if feature extraction fails.
    """
    # ── 1. Extract features ───────────────────────────────────────
    df_features = extract_features(archivo, id3=id3)
    if df_features is None:
        return None

    features = FEATURES

    missing = [f for f in features if f not in df_features.columns]
    if missing:
        warnings.warn(
            f"Missing features in extracted data: {missing}\n"
            "Scores will not be computed."
        )
        return df_features

    df_caso = df_features.copy()

    scaler         = pipeline['scaler']
    X_caso_scaled = scaler.transform(df_caso[features])
    

    # ── 2. Score with ensemble ────────────────────────────────────
    from Utilidades.ensemble import predict_ensemble

    resultados_val = predict_ensemble(pipeline, df_features)
    df_resultados = _build_output_row(df_features, resultados_val)
    
    pipeline['caso']       = df_caso

    df_shap_caso = build_shap(df_caso, df_resultados, pipeline)

    pipeline['df_shap']    = df_shap_caso


    # - 5. Complexity Index

    summary = (
        df_shap_caso
        .groupby("Study_idx")
        .apply(
            lambda g: pd.Series({
                "complexity_index":
                    (g["Complexity"] * g["ABS_SHAP"]).sum()
                    / g["ABS_SHAP"].sum(),

                "complexity_pct":
                    g.loc[g["Complexity"] > 0, "ABS_SHAP"].sum()
                    / g["ABS_SHAP"].sum()
            })
        )
        .reset_index()
    )

    summary2 = summary.set_index("Study_idx")

    # Columnas de etiqueta binaria por modelo (Score_IF_label, Score_KNN_label,
    # etc.), generadas por predict_ensemble() comparando cada score contra su
    # propio p80 de entrenamiento. Se agregan explícitamente porque no siempre
    # están incluidas en la constante SCORES, y son necesarias para colorear
    # las barras de score de forma coherente con la etiqueta de anomalía real
    # de cada modelo (en vez de usar umbrales arbitrarios comunes).
    label_cols = [c for c in df_resultados.columns if c.endswith("_label")]
    score_cols = ['Score_IF', 'Score_KNN', 'Score_PCA', 'Score_LOF']

    df_resultados_final = (
        df_caso[IDS]
        .join(df_resultados[['Composite_score', 'Risk_percentile']], how = "left")
        .join(df_resultados[score_cols + label_cols], how = "left")
        .join(summary2[["complexity_index", "complexity_pct"]], how = "left")
    )
                      
    df_resultados_final["tipo_anomalia"] = pd.cut(
        df_resultados_final["complexity_index"],
        bins=[-np.inf, -0.2, 0.2, np.inf],
        labels=["Simple", "Neutra", "Compleja"]
    )
    
    pipeline['resultados'] = df_resultados_final

    # Genera el HTML del resumen y lo guarda en el pipeline (además de
    # mostrarlo con display() si estamos en un notebook). Esto es lo que
    # permite que un consumidor externo -como una app de Streamlit- pueda
    # renderizarlo, ya que fuera de un notebook display() no muestra nada.
    summary_html = _print_summary_html(df_resultados_final, os.path.basename(archivo), theme = theme)
    pipeline['summary_html'] = summary_html

    # Proyección 2D (PCA) del espacio de features (67 variables escaladas):
    # dónde cae el caso evaluado respecto a la nube de ~300 casos de train.
    embedding_html = plot_2d_embedding_html(
        pipeline, df_resultados_final, X_caso_scaled, theme=theme
    )
    pipeline['embedding_html'] = embedding_html

    if plot_features:
        plot_html = plot_case_explanation_html(
            df_shap_all=df_shap_caso,
            top_n=10,
            theme=theme,
        )
        pipeline['plot_html'] = plot_html

    return pipeline


# ─────────────────────────────────────────────────────────────────────────────
# 2D EMBEDDING: TRAIN vs CASO EVALUADO (espacio de scores, 4 modelos)
# ─────────────────────────────────────────────────────────────────────────────

def plot_2d_embedding_html(
    pipeline: dict,
    df_resultados_final: pd.DataFrame,
    X_caso_scaled: np.ndarray,
    theme: str = "dark",
    save_path: str | None = None,
) -> str:
    """
    Proyección 2D (PCA) del espacio de FEATURES (las ~67 variables clínicas,
    escaladas con el mismo StandardScaler del entrenamiento), mostrando los
    ~300 casos de entrenamiento y el caso evaluado (uno o más haces) sobre
    los mismos ejes.

    A diferencia de una versión anterior que proyectaba el espacio de scores
    (4 dimensiones, una por modelo), acá el PCA se calcula sobre el espacio
    de features original. Esto refleja similitud/disimilitud "clínica" entre
    planes (geometría real de las 67 variables), no solo cómo lo puntuó cada
    modelo. El color de cada punto sigue indicando si el ensemble lo marcó
    como anómalo o no.

    Color:
        Verde -> ningún modelo marcó anomalía (label 0 en los 4 modelos)
        Rojo  -> al menos un modelo marcó anomalía (mismo criterio que las
                 barras de score del resumen, p80 de entrenamiento por modelo)
    El caso evaluado se dibuja con marcador tipo estrella, más grande, y
    etiquetado con el nombre del haz.

    Parameters
    ----------
    pipeline : dict
        Salida de fit_ensemble(). Debe contener 'score_cols', 'X_train_scaled',
        'df_labels_train'.
    df_resultados_final : pd.DataFrame
        DataFrame armado en score_rtplan() con, al menos, las columnas de
        label (Score_IF_label, ..., Score_LOF_label) del caso evaluado.
    X_caso_scaled : np.ndarray
        Features del caso evaluado (una fila por haz) ya escaladas con
        pipeline['scaler'], mismas columnas/orden que pipeline['features'].
    theme : str
        "dark" o "light".
    save_path : str | None
        Si se especifica, guarda el HTML en disco.
    """
    t = _get_theme_html(theme)

    GREEN, RED = "#4caf50", "#ef5350"

    score_cols       = pipeline["score_cols"]
    X_train_scaled   = pipeline["X_train_scaled"]
    df_labels_train  = pipeline["df_labels_train"]
    label_cols       = [f"{c}_label" for c in score_cols]

    faltantes = [c for c in label_cols if c not in df_resultados_final.columns]
    if faltantes:
        raise ValueError(f"df_resultados_final no tiene las columnas necesarias: {faltantes}")

    # ── PCA 2D sobre el espacio de features (train) ──────────────────
    pca       = PCA(n_components=2, random_state=42)
    emb_train = pca.fit_transform(X_train_scaled)
    emb_case  = pca.transform(X_caso_scaled)
    var_exp   = pca.explained_variance_ratio_ * 100

    # ── Colores según label (0/1) por caso ───────────────────────────
    n_flag_train  = df_labels_train["Anomalia"]
    color_train   = np.where(n_flag_train > 0, RED, GREEN)

    n_flag_case = df_resultados_final[label_cols].sum(axis=1)
    color_case  = np.where(n_flag_case > 0, RED, GREEN)

    beam_labels = df_resultados_final["ID2"].astype(str) if "ID2" in df_resultados_final.columns \
        else [f"Beam_{i}" for i in range(len(df_resultados_final))]

    fig = go.Figure()

    # Nube de entrenamiento
    fig.add_trace(go.Scatter(
        x=emb_train[:, 0], y=emb_train[:, 1],
        mode="markers",
        marker=dict(size=6, color=color_train, opacity=0.55, line=dict(width=0)),
        hovertext=[f"Train · modelos que marcan anomalía: {int(n)}/4" for n in n_flag_train],
        hovertemplate="%{hovertext}<extra></extra>",
        showlegend=False,
    ))

    # Entradas "fantasma" solo para la leyenda (no queremos duplicar hover)
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                              marker=dict(size=9, color=GREEN),
                              name="Train · sin anomalía (0/4 modelos)"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                              marker=dict(size=9, color=RED),
                              name="Train · marcado por ≥1 modelo"))

    # Caso evaluado
    fig.add_trace(go.Scatter(
        x=emb_case[:, 0], y=emb_case[:, 1],
        mode="markers+text",
        marker=dict(size=18, symbol="star", color=color_case,
                    line=dict(width=2, color=t["text"])),
        text=beam_labels,
        textposition="top center",
        textfont=dict(color=t["text"], size=10),
        hovertext=[f"Beam: {b} · modelos que marcan anomalía: {int(n)}/4"
                   for b, n in zip(beam_labels, n_flag_case)],
        hovertemplate="%{hovertext}<extra></extra>",
        name="Caso evaluado",
    ))

    fig.update_layout(
        template=t["template"],
        paper_bgcolor=t["bg"], plot_bgcolor=t["panel_bg"],
        font=dict(color=t["text"], family="Segoe UI, Roboto, Arial, sans-serif"),
        title=dict(text="Ubicación del caso vs. entrenamiento (espacio de features, PCA)",
                   font=dict(size=14)),
        xaxis_title=f"PC1 ({var_exp[0]:.1f}% var.)",
        yaxis_title=f"PC2 ({var_exp[1]:.1f}% var.)",
        height=480,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    embedding_html = fig.to_html(full_html=False, include_plotlyjs="cdn")

    if _is_notebook():
        display(HTML(embedding_html))

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(embedding_html)

    return embedding_html

# ─────────────────────────────────────────────────────────────────────────────
# PRETTY PRINT
# ─────────────────────────────────────────────────────────────────────────────


def _get_theme_html(theme: str = "dark") -> dict:
    if theme == "dark":
        return {
            "bg": "#0e1117", "panel_bg": "#161a23", "text": "#eaeaea", "muted": "#8a8f98",
            "shap_neg": "#ef5350",   # anomalía (SHAP < 0)
            "shap_pos": "#42a5f5",   # normaliza (SHAP > 0)
            "cx_complex": "#ff9800", # aporta a COMPLEJA
            "cx_simple":  "#26c6da", # aporta a SIMPLE
            "cx_neutral": "#5c6370", # sin dirección definida
            "template": "plotly_dark",
        }
    else:
        return {
            "bg": "#ffffff", "panel_bg": "#f7f7f9", "text": "#1a1a1a", "muted": "#666666",
            "shap_neg": "#d32f2f",
            "shap_pos": "#1565c0",
            "cx_complex": "#ef6c00",
            "cx_simple":  "#00838f",
            "cx_neutral": "#9e9e9e",
            "template": "plotly_white",
        }
        

def _print_summary_html(
    df: pd.DataFrame,
    filename: str,
    theme: str = "dark",
    save_path: str | None = None,
) -> str:
    """
    Genera un reporte HTML elegante a partir del DataFrame de resultados
    y lo muestra directamente en la celda de Colab.

    Usa la misma paleta que plot_case_explanation_html (_get_theme_html),
    de modo que el color de "tipo_anomalia" (Simple/Neutra/Compleja) sea
    idéntico al del panel de "Índice de complejidad" en los gráficos.

    Parámetros
    ----------
    df : DataFrame con las columnas esperadas (ID2, Score_IF, Score_KNN, ...)
    filename : nombre del archivo/lote que se está resumiendo (se muestra en el header)
    theme : "dark" o "light" — debe coincidir con el theme usado en los gráficos
    save_path : si se pasa una ruta (ej. "reporte.html"), además guarda el HTML a disco

    Retorna
    -------
    El string HTML generado (por si querés reutilizarlo).
    """
    t = _get_theme_html(theme)

    # Colores de riesgo (semáforo BAJO/INTERMEDIO/ALTO) — variantes por theme
    if theme == "dark":
        RIESGO_COLORS = {
            "BAJO":       {"bg": "#1b2e1f", "border": "#4caf50", "text": "#a5d6a7", "badge": "#2e7d32"},
            "INTERMEDIO": {"bg": "#332b12", "border": "#f9a825", "text": "#ffe082", "badge": "#f9a825"},
            "ALTO":       {"bg": "#3a1414", "border": "#ef5350", "text": "#ef9a9a", "badge": "#c62828"},
        }
    else:
        RIESGO_COLORS = {
            "BAJO":       {"bg": "#e8f5e9", "border": "#2e7d32", "text": "#1b5e20", "badge": "#2e7d32"},
            "INTERMEDIO": {"bg": "#fff8e1", "border": "#f9a825", "text": "#7a5c00", "badge": "#f9a825"},
            "ALTO":       {"bg": "#ffebee", "border": "#c62828", "text": "#8e0000", "badge": "#c62828"},
        }

    # Colores de tipo_anomalia — MISMA paleta que el panel de complejidad del gráfico
    TIPO_COLORS = {
        "Simple":   t["cx_simple"],
        "Neutra":   t["cx_neutral"],
        "Compleja": t["cx_complex"],
    }

    def interpretar(label: int, tipo: str):
        if label == 0:
            interpretacion = "BAJO"
        elif label in (1, 2):
            interpretacion = "INTERMEDIO"
        else:
            interpretacion = "ALTO"

        if label == 0:
            anomalia, riesgo = "NO", "BAJO"
        elif tipo == "Simple":
            anomalia, riesgo = "SI - PLAN SIMPLE", "BAJO"
        elif tipo == "Neutra":
            anomalia, riesgo = "SI - PLAN NEUTRO", "INTERMEDIO"
        elif tipo == "Compleja":
            anomalia, riesgo = "SI - PLAN COMPLEJO", "ALTO"
        else:
            anomalia, riesgo = "SI", interpretacion
        return interpretacion, anomalia, riesgo

    def score_bar(valor: float, label: int) -> str:
        pct = max(0, min(100, valor * 100))
        # Verde si el modelo NO marcó anomalía (label 0), rojo si SÍ (label 1),
        # en vez de umbrales arbitrarios sobre el valor escalado. Así el color
        # de la barra coincide siempre con la etiqueta real de ese modelo
        # (score >= p80 de su distribución de entrenamiento).
        color = "#ef5350" if label else "#4caf50"
        track_bg = "#2a2e38" if theme == "dark" else "#eee"
        return f"""
        <div class="score-row">
          <div class="score-track" style="background:{track_bg};"><div class="score-fill" style="width:{pct:.1f}%;background:{color};"></div></div>
        </div>"""

    cards_html = []
    for _, row in df.iterrows():
        beam = row.get("ID2", "—")
        score_if = row["Score_IF"]
        score_knn = row["Score_KNN"]
        score_pca = row["Score_PCA"]
        score_lof = row["Score_LOF"]
        score = row["Composite_score"]
        percentile = row["Risk_percentile"]
        label = int(row["Composite_label"])
        complexity = row["complexity_index"]
        tipo = row["tipo_anomalia"]

        # Etiquetas binarias por modelo (0/1), calculadas por predict_ensemble
        # comparando cada score contra el p80 de su propia distribución de
        # entrenamiento. Si por algún motivo no llegaron en el DataFrame, se
        # asume 0 (no marca anomalía) para no romper el reporte.
        label_if  = int(row["Score_IF_label"])  if "Score_IF_label"  in row and pd.notna(row["Score_IF_label"])  else 0
        label_knn = int(row["Score_KNN_label"]) if "Score_KNN_label" in row and pd.notna(row["Score_KNN_label"]) else 0
        label_pca = int(row["Score_PCA_label"]) if "Score_PCA_label" in row and pd.notna(row["Score_PCA_label"]) else 0
        label_lof = int(row["Score_LOF_label"]) if "Score_LOF_label" in row and pd.notna(row["Score_LOF_label"]) else 0
        # Composite_label es la suma de labels individuales (0..N modelos);
        # para la barra compuesta se pinta rojo si al menos un modelo marcó.
        label_composite = int(label > 0)

        interpretacion, anomalia, riesgo = interpretar(label, tipo)
        c = RIESGO_COLORS.get(riesgo, RIESGO_COLORS["INTERMEDIO"])
        tipo_color = TIPO_COLORS.get(tipo, t["cx_neutral"])

        cards_html.append(f"""
        <div class="card" style="border-left:6px solid {c['border']};background:{c['bg']};">
          <div class="card-header">
            <span class="beam-title">Beam: {beam}</span>
            <span class="badge" style="background:{c['badge']};">{riesgo}</span>
          </div>

          <div class="section">
            <div class="section-title">Scores <span class="hint">[0–1] 0: normal · 1: anómalo</span></div>
            <div class="score-grid">
              <div class="score-label">Score_IF</div><div class="score-val">{score_if:.4f}</div>{score_bar(score_if, label_if)}
              <div class="score-label">Score_KNN</div><div class="score-val">{score_knn:.4f}</div>{score_bar(score_knn, label_knn)}
              <div class="score-label">Score_PCA</div><div class="score-val">{score_pca:.4f}</div>{score_bar(score_pca, label_pca)}
              <div class="score-label">Score_LOF</div><div class="score-val">{score_lof:.4f}</div>{score_bar(score_lof, label_lof)}
              <div class="score-label"><b>Composite_score</b></div><div class="score-val"><b>{score:.4f}</b></div>{score_bar(score, label_composite)}
            </div>
          </div>

          <div class="section">
            <div class="section-title">Riesgo</div>
            <table class="kv">
              <tr><td>Percentil</td><td>{percentile:.1f} %</td></tr>
              <tr><td>Etiqueta</td><td>{label}</td></tr>
              <tr><td>Interpretación</td><td>{interpretacion}</td></tr>
            </table>
          </div>

          <div class="section">
            <div class="section-title">Índice de complejidad</div>
            <table class="kv">
              <tr><td>complexity_index</td><td>{complexity:.3f}</td></tr>
              <tr><td>tipo_anomalia</td>
                  <td><span class="badge" style="background:{tipo_color};">{tipo}</span></td></tr>
            </table>
          </div>

          <div class="final" style="border-top:2px dashed {c['border']};color:{c['text']};">
            <div class="final-title">★ INTERPRETACIÓN FINAL ★</div>
            <table class="kv">
              <tr><td>ANOMALÍA</td><td><b>{anomalia}</b></td></tr>
              <tr><td>RIESGO</td><td><b>{riesgo}</b></td></tr>
            </table>
          </div>
        </div>
        """)

    header_bg = "#0a0d14" if theme == "dark" else "#1a1a2e"
    body_border = f"{t['muted']}33"

    html = f"""
    <div style="font-family:'Segoe UI',Roboto,Arial,sans-serif;max-width:760px;margin:0 auto;
                background:{t['bg']};color:{t['text']};padding:4px;border-radius:12px;">
      <style>
        .report-header {{
          background:{header_bg};color:#fff;padding:16px 20px;border-radius:10px 10px 0 0;
          font-size:16px;font-weight:600;letter-spacing:.5px;
        }}
        .report-body {{
          background:{t['bg']};padding:18px;border-radius:0 0 10px 10px;
          border:1px solid {body_border};border-top:none;
        }}
        .card {{
          background:{t['panel_bg']};border-radius:8px;padding:16px 20px;margin-bottom:16px;
          box-shadow:0 1px 3px rgba(0,0,0,.25);
        }}
        .card-header {{
          display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;
        }}
        .beam-title {{ font-size:15px;font-weight:700;color:{t['text']}; }}
        .badge {{
          color:#fff;padding:3px 12px;border-radius:12px;font-size:11px;font-weight:700;letter-spacing:.5px;
        }}
        .section {{ margin-top:10px; }}
        .section-title {{
          font-size:12px;font-weight:700;text-transform:uppercase;color:{t['muted']};
          letter-spacing:.5px;margin-bottom:6px;border-bottom:1px solid {body_border};padding-bottom:3px;
        }}
        .hint {{ font-weight:400;text-transform:none;color:{t['muted']};font-size:11px; }}
        .score-grid {{
          display:grid;grid-template-columns:110px 60px 1fr;row-gap:6px;column-gap:8px;align-items:center;
        }}
        .score-label {{ font-size:12px;color:{t['text']}; }}
        .score-val {{ font-size:12px;font-family:monospace;color:{t['text']}; }}
        .score-row {{ display:flex;align-items:center; }}
        .score-track {{ border-radius:4px;height:8px;width:100%;overflow:hidden; }}
        .score-fill {{ height:100%;border-radius:4px; }}
        table.kv {{ width:100%;border-collapse:collapse;font-size:12px; }}
        table.kv td {{ padding:3px 0;color:{t['text']}; }}
        table.kv td:first-child {{ color:{t['muted']}; }}
        table.kv td:last-child {{ text-align:right;font-family:monospace; }}
        .final {{ margin-top:12px;padding-top:10px; }}
        .final-title {{ font-weight:700;font-size:12px;text-align:center;margin-bottom:6px;letter-spacing:.5px; }}
      </style>
      <div class="report-header">📊 {filename}</div>
      <div class="report-body">
        {''.join(cards_html)}
      </div>
    </div>
    """

    if _is_notebook():
        display(HTML(html))

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"HTML guardado en: {save_path}")

    return html






#-- PLOT

def plot_case_explanation_html(
    df_shap_all: pd.DataFrame,
    top_n: int = 10,
    theme: str = "dark",
    save_path: str | None = None,
) -> str:
    """
    Versión HTML/Plotly de plot_case_explanation, con el panel derecho
    reemplazado por el "Índice de complejidad" en vez del Z-score.

    Recorre TODOS los Study_idx presentes en df_shap_all y genera
    un bloque (SHAP waterfall + índice de complejidad) por cada uno.

    Panel izquierdo  — SHAP waterfall (rojo = anomalía, azul = normaliza)
    Panel derecho    — Complexity_value = sign(Zscore * Feature_direction) * |SHAP|
                        naranja = aporta a COMPLEJA, cian = aporta a SIMPLE, gris = sin dirección definida

    Parameters
    ----------
    df_shap_all : pd.DataFrame
        Debe contener una o más Study_idx; se grafican todas.
    top_n : int
        Cantidad de features top a mostrar por caso.
    theme : str
        "dark" o "light".
    save_path : str | None
        Si se especifica, guarda el HTML combinado (todos los casos) en disco.
    """
    t = _get_theme_html(theme)

    study_idxs = df_shap_all["Study_idx"].unique()
    if len(study_idxs) == 0:
        if _is_notebook():
            display(HTML("<p style='color:#c62828'>[!] df_shap_all está vacío.</p>"))
        return ""

    legend_html = f"""
    <div style="display:flex;gap:18px;justify-content:center;margin-top:6px;
                font-family:'Segoe UI',Roboto,Arial,sans-serif;font-size:12px;color:{t['muted']};">
      <span><span style="display:inline-block;width:10px;height:10px;background:{t['shap_neg']};
             border-radius:2px;margin-right:5px;"></span>Anomaly driver (SHAP&lt;0)</span>
      <span><span style="display:inline-block;width:10px;height:10px;background:{t['shap_pos']};
             border-radius:2px;margin-right:5px;"></span>Normalizing (SHAP&gt;0)</span>
      <span style="border-left:1px solid {t['muted']};padding-left:18px;">
        <span style="display:inline-block;width:10px;height:10px;background:{t['cx_complex']};
             border-radius:2px;margin-right:5px;"></span>Aporta a COMPLEJA</span>
      <span><span style="display:inline-block;width:10px;height:10px;background:{t['cx_simple']};
             border-radius:2px;margin-right:5px;"></span>Aporta a SIMPLE</span>
      <span><span style="display:inline-block;width:10px;height:10px;background:{t['cx_neutral']};
             border-radius:2px;margin-right:5px;"></span>Sin dirección (N/D)</span>
    </div>
    """

    sections = []

    for i, study_idx in enumerate(study_idxs):
        #case = df_shap_all[df_shap_all["Study_idx"] == study_idx].head(top_n).copy()
        case = (
              df_shap_all[df_shap_all["Study_idx"] == study_idx]
              .sort_values("ABS_SHAP", ascending = False)
              .head(top_n)
              .copy()
              )
        if case.empty:
            continue

        score      = case["Composite_score"].iloc[0]
        percentile = case["Risk_percentile"].iloc[0]
        label      = case["Composite_label"].iloc[0]

        # ── Cálculo del índice de complejidad ─────────────────────
        case["Feature_direction"] = case["Feature"].map(COMPLEXITY_DIRECTION)
        has_direction = case["Feature_direction"].notna()
        case["Complexity_sign"] = 0.0
        case.loc[has_direction, "Complexity_sign"] = (
            np.sign(case.loc[has_direction, "Zscore"]) * case.loc[has_direction, "Feature_direction"]
        )
        case["Complexity_value"] = case["Complexity"] * case["ABS_SHAP"]

        case_sorted = case.sort_values("ABS_SHAP")
        features_order = case_sorted["Feature"].tolist()
        case_reindexed = case.set_index("Feature").reindex(features_order)

        shap_colors = [t["shap_neg"] if v < 0 else t["shap_pos"] for v in case_sorted["SHAP"]]

        def cx_color(sign):
            if sign > 0:  return t["cx_complex"]
            if sign < 0:  return t["cx_simple"]
            return t["cx_neutral"]

        def cx_label(sign):
            if sign > 0:  return "COMPLEJA"
            if sign < 0:  return "SIMPLE"
            return "N/D"

        cx_colors = [cx_color(s) for s in case_reindexed["Complexity_sign"]]

        shap_hover = [
            f"<b>{feat}</b><br>SHAP: {val:+.3f}<br>"
            f"{'Anomaly driver' if val < 0 else 'Normalizing'}"
            for feat, val in zip(case_sorted["Feature"], case_sorted["SHAP"])
        ]

        cx_hover = [
            f"<b>{feat}</b><br>Complexity idx: {row['Complexity_value']:+.3f}<br>"
            f"Aporta a: {cx_label(row['Complexity_sign'])}"
            for feat, row in case_reindexed.iterrows()
        ]

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("SHAP  (negativo → driver de anomalía)", "Índice de complejidad  (−SIMPLE / +COMPLEJA)"),
            horizontal_spacing=0.04,
        )

        fig.add_trace(
            go.Bar(
                y=case_sorted["Feature"], x=case_sorted["SHAP"], orientation="h",
                marker_color=shap_colors, text=[f"{v:+.3f}" for v in case_sorted["SHAP"]],
                textposition="outside", hovertext=shap_hover, hoverinfo="text",
                name="SHAP",
            ),
            row=1, col=1,
        )

        fig.add_trace(
            go.Bar(
                y=features_order, x=case_reindexed["Complexity_value"], orientation="h",
                marker_color=cx_colors, text=[f"{v:+.3f}" for v in case_reindexed["Complexity_value"]],
                textposition="outside", hovertext=cx_hover, hoverinfo="text",
                name="Complexity index",
            ),
            row=1, col=2,
        )

        fig.add_vline(x=0, line_width=1, line_color=t["muted"], row=1, col=1)
        fig.add_vline(x=0, line_width=1, line_color=t["muted"], row=1, col=2)

        fig.update_layout(
            template=t["template"],
            title=dict(
                text=f"Study_idx={study_idx}   score={score:.3f}   "
                     f"percentile={percentile:.0f}   label={label}",
                x=0.5, xanchor="center",
            ),
            height=max(420, 46 * top_n),
            width=1050,
            showlegend=False,
            margin=dict(l=40, r=40, t=100, b=40),
            paper_bgcolor=t["bg"],
            plot_bgcolor=t["panel_bg"],
        )
        fig.update_yaxes(categoryorder="array", categoryarray=features_order, row=1, col=1)
        fig.update_yaxes(categoryorder="array", categoryarray=features_order, row=1, col=2)

        max_abs = max(
            abs(case_sorted["SHAP"]).max(),
            abs(case_reindexed["Complexity_value"]).max()
        ) + 0.2

        fig.update_xaxes(range=[-max_abs, max_abs], row=1, col=1)
        fig.update_xaxes(range=[-max_abs, max_abs], row=1, col=2)
        fig.update_yaxes(showticklabels=False, row=1, col=2)

        # Solo el primer caso carga plotly.js; los siguientes lo reutilizan
        chart_html = fig.to_html(full_html=False, include_plotlyjs=("cdn" if i == 0 else False))

        sections.append(f"""
        <div style="background:{t['bg']};padding:12px;border-radius:10px;
                    margin-bottom:18px;border:1px solid {t['muted']}33;">
            {chart_html}
            {legend_html}
        </div>
        """)

    full_html = "".join(sections)

    if _is_notebook():
        display(HTML(full_html))

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(f"<html><body style='margin:0;background:{t['bg']};'>{full_html}</body></html>")
        print(f"HTML guardado en: {save_path}")

    return full_html




# ─────────────────────────────────────────────────────────────────────────────
# USAGE EXAMPLE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(__doc__)
