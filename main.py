# main.py (Streamlit Application)
# Adaptado desde "Codigo_Propuesto.txt" (Google Colab) para despliegue en
# GitHub + Streamlit (onrender.com).
#
# Reemplaza:
#   - google.colab.drive / files.upload()  ->  st.file_uploader
#   - rutas de Drive ('/Modulos/', etc.)   ->  rutas relativas al repo
#   - !pip install                         ->  requirements.txt

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import tempfile

st.set_page_config(layout="wide", page_title="Detección de Anomalías RTPLAN")

# ============================================
# RUTAS DEL REPOSITORIO (equivalentes a las de Drive)
# ============================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ruta_modulos  = os.path.join(BASE_DIR, "Modulos")
ruta_datasets = os.path.join(BASE_DIR, "Datasets")
ruta_output   = os.path.join(BASE_DIR, "Outputs")

# Para poder importar "from Clases.clases import ..." y
# "from Utilidades.ensemble import ..." tal como en Colab
sys.path.append(ruta_modulos)

from Utilidades.ensemble import fit_ensemble
from Utilidades.score_rtplan import score_rtplan
from Utilidades.CONSTANTES import (
    COMPLEXITY_DIRECTION,
    FEATURES,
    IDS,
    SCORES,
    AGRUPAMIENTO_FEATURES,
)

from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.pca import PCA
from pyod.models.lof import LOF


# ============================================
# ENTRENAMIENTO DEL ENSEMBLE (se cachea: se entrena
# una sola vez por sesión del servidor, no en cada request)
# ============================================
@st.cache_resource(show_spinner="Entrenando modelos de ensemble...")
def cargar_pipeline():
    dataset_normales = os.path.join(ruta_datasets, "dataset_train_safim.csv")
    df_train = pd.read_csv(dataset_normales)

    modelos = {
        "IF": IForest(contamination=0.01, random_state=42),
        "KNN": KNN(contamination=0.01),
        "PCA": PCA(contamination=0.01, random_state=42),
        "LOF": LOF(contamination=0.01),
    }

    pipeline = fit_ensemble(df_train, FEATURES, modelos)
    return pipeline


# ============================================
# LÓGICA PRINCIPAL STREAMLIT
# ============================================
def main():
    st.title("Sistema de Detección de Anomalías en RTPLAN (Ensemble)")
    st.markdown(
        "Cargue un archivo DICOM RTPLAN (.dcm) para evaluar la complejidad "
        "y el riesgo de anomalía de cada haz."
    )

    pipeline = cargar_pipeline()

    uploaded_file = st.file_uploader(
        "Subir Archivo DICOM RTPLAN",
        type="dcm",
        help="Solo se aceptan archivos RTPLAN en formato DICOM.",
    )

    if uploaded_file is not None:
        st.subheader(f"Procesando archivo: {uploaded_file.name}")

        # score_rtplan espera un nombre/ruta de archivo en disco (igual que en
        # Colab, donde 'nombre' viene de files.upload()). Por eso el archivo
        # subido se guarda temporalmente antes de pasarlo a la función.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_path = tmp_file.name

        try:
            resultado = score_rtplan(
                tmp_path,
                pipeline,
                plot_features=False,
                theme="dark",
            )

            # score_rtplan devuelve el pipeline actualizado (dict) con, entre
            # otras cosas, 'summary_html' (el reporte HTML por haz) y
            # 'resultados' (DataFrame con los scores). Devuelve None si no
            # pudo extraer features del DICOM.
            if resultado is None:
                st.error(
                    "No se pudieron extraer las características del archivo "
                    "DICOM. Verifique que sea un RTPLAN válido."
                )
            else:
                summary_html = resultado.get("summary_html")
                embedding_html = resultado.get("embedding_html")
                df_resumen = resultado.get("resultados")

                if summary_html:
                    st.components.v1.html(summary_html, height=900, scrolling=True)
                else:
                    st.warning("No se generó el resumen HTML del caso.")
 
                if embedding_html:
                    st.subheader("Ubicación del caso vs. entrenamiento (2D)")
                    st.components.v1.html(embedding_html, height=520, scrolling=False)
                    
                if df_resumen is not None:
                    with st.expander("Ver tabla de resultados"):
                        st.dataframe(df_resumen, use_container_width=True)

        except Exception as e:
            st.error(f"Ocurrió un error durante el procesamiento del DICOM: {e}")
            st.exception(e)
        finally:
            os.remove(tmp_path)


if __name__ == "__main__":
    main()
