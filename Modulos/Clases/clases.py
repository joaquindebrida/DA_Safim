"""
Definicion de las clases RTPLAN y Beam
"""

import pydicom
import pandas as pd
from io import BytesIO
from PIL import Image
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString
from shapely.affinity import translate
from scipy.stats import skew, kurtosis
from shapely.ops import polygonize, unary_union
from matplotlib.artist import get
from scipy.stats.distributions import f

# HELPERS
# CLASE RTPLAN
# CLASE BEAM


# ---------------------------------------
# Helpers
# ---------------------------------------

def to_int_safe(value):
    """Convierte a int sin romper si es None o no convertible."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return None

def calc_poligono(fila, size=(128,128)):
    # ====== 1. Construir polígono de apertura ======
    spacing = 5
    shift = fila['J1']

    inicio = 41 + round(fila['J1'] / spacing)
    fin    = 40 + round(fila['J2'] / spacing)

    B1 = [fila[f'B1_L{i}'] for i in range(inicio, fin+1)]
    B2 = [fila[f'B2_L{i}'] for i in range(inicio, fin+1)]

    coords = []

    # Banco izquierdo
    for i,v in enumerate(B1):
        y0 = i*spacing + shift
        y1 = (i+1)*spacing + shift
        coords.append((v, y0))
        coords.append((v, y1))

    # Banco derecho
    for i,v in reversed(list(enumerate(B2))):
        y0 = i*spacing + shift
        y1 = (i+1)*spacing + shift
        coords.append((v, y1))
        coords.append((v, y0))

    coords.append(coords[0])
    poly = Polygon(coords)

    return poly


def contar_self_intersections(poly):

    exterior = LineString(poly.exterior.coords)

    merged = unary_union(exterior)

    polys = list(polygonize(merged))

    # polígono normal -> 1
    # self-intersectado -> >1
    return max(len(polys) - 1, 0)

def _ampliar_fila(fila):
            # Índices basados en Jaws (5mm ancho)
            idx_ini = 41 + int(round(fila['J1'] / 5))
            idx_fin = 40 + int(round(fila['J2'] / 5))

            # Clipping de seguridad (Láminas 1-80)
            idx_ini = max(1, min(80, idx_ini))
            idx_fin = max(1, min(80, idx_fin))

            if idx_ini > idx_fin:
                return idx_ini, idx_fin, 0.0

            # Calcular apertura usando columnas dinámicas
            # Nota: Usamos indices numéricos si fuera posible, pero strings es seguro
            cols_b1 = [f'B1_L{i}' for i in range(idx_ini, idx_fin + 1)]
            cols_b2 = [f'B2_L{i}' for i in range(idx_ini, idx_fin + 1)]

            # Suma de posiciones para apertura
            try:
                ancho = fila[cols_b2].sum() - fila[cols_b1].sum()
                apertura = ancho * 5 # mm
            except KeyError:
                apertura = 0.0

            return idx_ini, idx_fin, apertura

def _crear_poligono_row(row):
    """
    Crea un objeto shapely.Polygon basado en las láminas y Jaws de la fila.
    Recicla lp_ini y lp_fin calculados previamente.
    """
    # 1. Recuperar índices (asegurar int)
    try:
        idx_ini = int(row['lp_ini'])
        idx_fin = int(row['lp_fin'])
    except (ValueError, TypeError):
        return None # O Polygon() vacio

    # Si los índices están cruzados, no hay apertura
    if idx_ini > idx_fin:
        return Polygon()

    # 2. Definir parámetros geométricos
    spacing = 5.0 # mm
    # Usamos J1 como el 'shift' en Y (inicio del bloque de láminas visibles)
    shift = row['J1']

    # 3. Obtener nombres de columnas para el rango activo
    # Nota: range va hasta idx_fin + 1 para incluir la última lámina
    cols_b1 = [f'B1_L{i}' for i in range(idx_ini, idx_fin + 1)]
    cols_b2 = [f'B2_L{i}' for i in range(idx_ini, idx_fin + 1)]

    # Extraer valores numéricos de las columnas (manejo de errores si faltan cols)
    try:
        vals_b1 = row[cols_b1].values.astype(float)
        vals_b2 = row[cols_b2].values.astype(float)
    except KeyError:
        return Polygon() # Faltan columnas

    coords = []
    n_leaves = len(vals_b1)

    # --- Construir coordenadas Banco 1 (Izquierdo) ---
    # Recorrido de abajo hacia arriba (i=0 es la lámina inferior)
    for i in range(n_leaves):
        x = vals_b1[i]
        y_bottom = i * spacing + shift
        y_top    = (i + 1) * spacing + shift

        coords.append((x, y_bottom))
        coords.append((x, y_top))

    # --- Construir coordenadas Banco 2 (Derecho) ---
    # Recorrido inverso (de arriba hacia abajo) para cerrar el loop correctamente
    for i in range(n_leaves - 1, -1, -1):
        x = vals_b2[i]
        y_bottom = i * spacing + shift
        y_top    = (i + 1) * spacing + shift

        coords.append((x, y_top))
        coords.append((x, y_bottom))

    # --- Cerrar y crear Polígono ---
    if len(coords) > 2:
        coords.append(coords[0]) # Cerrar loop
        return Polygon(coords)

    return Polygon()

def calcular_recorrido_gantry(angles_list):
    if len(angles_list) < 2:
        return 0.0


    rads = np.deg2rad(angles_list)

    rads_unwrapped = np.unwrap(rads)

    degrees_continuous = np.rad2deg(rads_unwrapped)

    deltas = np.diff(degrees_continuous)

    recorrido_total = np.sum(np.abs(deltas))

    return recorrido_total


def extract_features(rtplan):
    # 1. Crear una lista vacía fuera del bucle para acumular los datos
    lista_filas = []

    # 2. Iterar sobre las KEYS (índices), no values, para poder llamar a atributos_globales(idx)
    for beam_idx in rtplan.beams.keys():

        beam = rtplan.beams[beam_idx]
        beam_name = getattr(beam.raw, 'BeamName', f"Beam_{beam_idx}")

        # 3. Obtener diccionarios de métricas
        ag = rtplan.atributos_globales(beam_idx)
        al = beam.atributos_beam

        # 4. Crear el diccionario combinado ordenadamente
        fila_dict = {}

        # A) Primero identificadores
        fila_dict['ID'] = rtplan.header['PatientID']
        fila_dict['nuevoID'] = f'{rtplan.header["PatientID"]}_{beam_name}'

        # B) Luego mezclamos los diccionarios (update agrega sin borrar lo anterior)
        fila_dict.update(ag)
        fila_dict.update(al)

        # 5. Guardar el diccionario en la lista acumuladora
        lista_filas.append(fila_dict)

    # 6. Crear el DataFrame UNA SOLA VEZ al final con toda la lista
    df_final = pd.DataFrame(lista_filas)

    return
def cross_bank_edge_coincidence(df, threshold_mm=1.0):
        """
        Calcula el score de coincidencia cross-bank a partir de un DataFrame
        donde cada fila es un control point del campo.

        Parámetros
        ----------
        df : DataFrame con columnas B1_L{i}, B2_L{i}, UM, lp_ini, lp_fin
        threshold_mm : distancia en mm para considerar coincidencia de bordes

        Retorna
        -------
        coincidence_score : float ponderado por MU
        """

        # --- Extraer rango de láminas activas del plan ---
        l_min = int(df['lp_ini'].min())
        l_max = int(df['lp_fin'].max())
        leaf_cols_A = [f'B1_L{i}' for i in range(l_min, l_max + 1)]
        leaf_cols_B = [f'B2_L{i}' for i in range(l_min, l_max + 1)]

        # --- Arrays numpy: shape (n_cp, n_leaves) ---
        edges_A = df[leaf_cols_A].values   # banco A
        edges_B = df[leaf_cols_B].values   # banco B

        # --- Pesos MU normalizados: shape (n_cp,) ---
        mu = df['UM'].values
        mu_weights = mu / mu.sum()

        # --- Máscara de láminas activas (dentro del campo): shape (n_cp, n_leaves) ---
        active = (edges_B - edges_A) > 0

        # --- Score con broadcasting vectorizado ---
        n_leaves = edges_A.shape[1]
        coincidence_score = 0.0

        # Matriz MU_i * MU_j: shape (n_cp, n_cp)
        mu_matrix = mu_weights[:, None] * mu_weights[None, :]

        for k in range(n_leaves):
            a = edges_A[:, k]   # shape (n_cp,)
            b = edges_B[:, k]   # shape (n_cp,)

            # Distancias banco_A[i] vs banco_B[j], i != j
            dist_matrix = np.abs(a[:, None] - b[None, :])   # (n_cp, n_cp)
            np.fill_diagonal(dist_matrix, np.inf)            # excluir i == j

            # Máscara de coincidencia espacial
            coincide = dist_matrix < threshold_mm            # (n_cp, n_cp)

            # Máscara de actividad: lámina k activa en CP_i Y en CP_j
            active_k = active[:, k]
            active_matrix = active_k[:, None] & active_k[None, :]  # (n_cp, n_cp)

            coincidence_score += ((coincide & active_matrix) * mu_matrix).sum()

        return coincidence_score

# ---------------------------------------
# CLASE RTPLAN
# ---------------------------------------
class RTPLAN:
    def __init__(self, dicom_file):
        self.raw_data = pydicom.dcmread(dicom_file, force=True)
        self.doses = self.extract_dose_dic()
        self.header = self.extract_header_dic()
        self.beams = self.extract_beams_dic()

    # --------------------------
    def extract_dose_dic(self):
        dosis = []
        Dose_sequence = getattr(self.raw_data, 'DoseReferenceSequence', [])

        for Dose_element in Dose_sequence:
            dosis.append({
                "Numero": getattr(Dose_element, "DoseReferenceNumber", None),
                "Tipo_1": getattr(Dose_element, "DoseReferenceStructureType", None),
                "Descripcion": getattr(Dose_element, "DoseReferenceDescription", None),
                "Tipo_2": getattr(Dose_element, "DoseReferenceType", None),
                "Valor": getattr(Dose_element, "TargetPrescriptionDose", None)
            })
        return dosis


    # --------------------------
    def extract_header_dic(self):
        header = {}

        # Datos básicos del plan
        header['PatientID'] = getattr(self.raw_data, 'PatientID', None)
        header['PatientName'] = getattr(self.raw_data, 'PatientName', None)
        header['RTPlanName'] = getattr(self.raw_data, 'RTPlanName', None)
        header['RTPlanDescription'] = getattr(self.raw_data, 'RTPlanDescription', None)

        #   OBTENER LA DOSIS TARGET
        target_dose = next((d for d in self.doses if d.get("Tipo_2") == "TARGET"), None)

        if target_dose:
            header['DoseDescription'] = target_dose.get('Descripcion')
            header['DoseValue'] = to_int_safe(target_dose.get('Valor'))
        else:
            header['DoseDescription'] = None
            header['DoseValue'] = None


        #   FRACCIONES Y NÚMERO DE HACES
        fx_group = getattr(self.raw_data, 'FractionGroupSequence', None)

        if fx_group and len(fx_group) > 0:
            fx_item = fx_group[0]
            fx_planned = to_int_safe(getattr(fx_item, 'NumberOfFractionsPlanned', None))
            beams = to_int_safe(getattr(fx_item, 'NumberOfBeams', None))
        else:
            fx_planned = None
            beams = None

        header['FxPlanned'] = fx_planned
        header['Beams'] = beams

        #     CÁLCULO DE TARGET DOSE POR FX

        if header['DoseValue'] and fx_planned:
            header['DoseFx'] = header['DoseValue'] / fx_planned
        else:
            header['DoseFx'] = None

        return header

    # --------------------------
    def extract_beams_dic(self):
        beams_dic = {}

        fx = getattr(self.raw_data, 'FractionGroupSequence', [])[0]
        Beam_sequence = getattr(fx, 'ReferencedBeamSequence', [])

        Beam_list = [{
            "Numero": getattr(B, "ReferencedBeamNumber", None),
            "BeamDose": getattr(B, "BeamDose", None),
            "BeamMeterset": getattr(B, "BeamMeterset", None)
        } for B in Beam_sequence]

        BS = getattr(self.raw_data, 'BeamSequence', [])

        for i, beam_ds in enumerate(BS):
            beams_dic[i + 1] = BEAM(beam_ds, Beam_list[i])

        return beams_dic
     #----------------------------
    def atributos_globales(self, idx):

        beam = self.beams[idx]

        atributos_dict = {}
        atributos_dict['ID'] = self.header['PatientID']
        atributos_dict['DS'] = float(beam.info['BeamDose'])  # OJO: dicom usa 0-index
        atributos_dict['UM'] = float(beam.um)
        atributos_dict['UM_DS'] = float(beam.um) / atributos_dict['DS']
        atributos_dict['CP'] = beam.info['NumberOfCP']
        #atributos_dict['AM'] = beam.apertura_maxima
        atributos_dict['MCS'] = float(beam.mcs)
        atributos_dict['LP1'] = beam.tabla['lp_ini'].min()
        atributos_dict['LP2'] = beam.tabla['lp_fin'].max()
        atributos_dict['NLP'] = 1 + atributos_dict['LP2'] - atributos_dict['LP1']

        return atributos_dict

# ---------------------------------------
# CLASE BEAM
# ---------------------------------------
class BEAM:
    def __init__(self, beam_ds, beam_info):
        """
        beam_ds   = dataset original del BeamSequence[i]
        beam_info = diccionario con {"Numero", "BeamDose", "BeamMeterset"}
        """
        self.raw = beam_ds
        self.numero = beam_info["Numero"]
        self.dosis = float(beam_info["BeamDose"] )
        self.um = float(beam_info["BeamMeterset"])
        #
        self.is_wedged = self._check_wedged()

        #
        self.tabla = self.extract_table()
        self.mcs = self.extract_mcs()
        self.lt_mcs = self.extract_lt_mcs()
        self.info = self.extract_beam_info()

    def _check_wedged(self):
        """
        Función bandera: Retorna True si el campo tiene una cuña (wedge).
        Se basa en la existencia del atributo 'WedgePositionSequence' en self.raw.
        """
        # Se busca WedgePositionSequence en el haz (beam)
        if hasattr(self.raw.ControlPointSequence[0], 'WedgePositionSequence'):
            return True

        return False



    def extract_table(self):

        CP_sequence = self.raw.ControlPointSequence
        num_cp = len(CP_sequence)

        if self.is_wedged:
            cp_indices = [0]

        else:
            # Si NO hay cuña (VMAT o STATIC)
            # Mantener la lógica anterior: Estático de 2 CPs usa solo el CP[0]. VMAT usa todos.
            cp_indices = [0] if num_cp == 2 else range(num_cp)


        # Columnas
        jaws_col_names = ['J1', 'J2']
        mlc_col_names = [
          f'B1_L{i}' if i <= 80 else f'B2_L{i-80}'
          for i in range(1, 161)
        ]

        registros = []

        for idx in cp_indices:
          fila = {"ControlPoint": idx}
          fila['GantryAngle'] = float(getattr(CP_sequence[idx], 'GantryAngle', None))
          cum_weight = float(getattr(CP_sequence[idx], 'CumulativeMetersetWeight', 0.0))
          fila['cumMetersetW'] = cum_weight

          # Iterar sobre los dispositivos directamente
          for dev in CP_sequence[idx].BeamLimitingDevicePositionSequence:
              vals = dev.LeafJawPositions
              dtype = dev.RTBeamLimitingDeviceType

              if dtype.startswith("ASY"):  # Mordazas ASYMX / ASYMY
                  fila.update({name: v for name, v in zip(jaws_col_names, vals)})

              elif dtype.startswith("MLC"):  # MLCX / MLCY
                  fila.update({name: v for name, v in zip(mlc_col_names, vals)})

          registros.append(fila)

        df = pd.DataFrame(registros)

        # 2. "cumMeterset" a partir de self.um * cumMetersetW
        df['cumMeterset'] = df['cumMetersetW'] * self.um

        # 3. "UM" (MU por Control Point) usando diff()
        df['UM'] = df['cumMeterset'].diff().fillna(0)

        # Borarr cumMeterset y cumMetersetW:
        df = df.drop(columns=['cumMetersetW', 'cumMeterset'])

        # 3 nuevas columnas
        res_exp = df.apply(_ampliar_fila, axis=1, result_type='expand')
        df[['lp_ini', 'lp_fin', 'apertura']] = res_exp

        # 4 crear poligono
        df['poligono'] = df.apply(_crear_poligono_row, axis=1)


        return df

    def extract_mcs(self):

        if self.is_wedged:
            return np.nan

        df = self.tabla.copy()

        # --- 1. Creación de DF Efectivo ---
        df['lp_ini'] = df['lp_ini'].astype(int)
        df['lp_fin'] = df['lp_fin'].astype(int)

        l_min = df['lp_ini'].min()
        l_max = df['lp_fin'].max()

        # Selección de columnas críticas
        cols_meta = ['ControlPoint', 'GantryAngle', 'UM', 'J1', 'J2', 'lp_ini', 'lp_fin', 'apertura']
        cols_laminas = ([f'B1_L{i}' for i in range(l_min, l_max + 1)] +
                        [f'B2_L{i}' for i in range(l_min, l_max + 1)])
        # Filtramos columnas existentes para evitar errores si l_min/max exceden lo disponible
        cols_laminas = [c for c in cols_laminas if c in df.columns]

        df_ef = df[cols_meta + cols_laminas].copy()

        # --- 2. Procesamiento Geométrico (Arcos) ---
        # Unwrap Gantry
        gantry_rad = np.deg2rad(df_ef['GantryAngle'])
        df_ef['GantryContinuous'] = np.rad2deg(np.unwrap(gantry_rad))

        # Detección de Arcos: UM=0 indica el FIN de un arco.
        # El inicio del nuevo arco es la fila siguiente.
        #fin_de_arco = (df_ef['UM'] == 0)
        inicio_de_arco = (df_ef['UM'] == 0)
        df_ef['ArcID'] = inicio_de_arco.cumsum()

        # --- 3. Cálculo de Métricas (LSV / AAV) Optimizado ---

        def _get_metrics_row(row):
            i_start, i_end = int(row['lp_ini']), int(row['lp_fin'])

            if i_start > i_end: return 1.0, 0.0 # LSV=1, AAV=0

            # Generamos nombres locales
            c_b1 = [f'B1_L{k}' for k in range(i_start, i_end + 1)]
            c_b2 = [f'B2_L{k}' for k in range(i_start, i_end + 1)]

            try:
                # Extracción rápida
                p1 = row[c_b1].values.astype(float)
                p2 = row[c_b2].values.astype(float)
            except KeyError:
                return 1.0, 0.0

            N = len(p1)
            if N < 1: return 1.0, 0.0

            # --- AAV (Leaf-Based Bounding Box) ---
            # Numerador: Suma de anchos (abs para robustez)
            width_sum = np.sum(np.abs(p2 - p1))

            # Denominador: Bounding Box (Max Global - Min Global)
            # Evitamos concatenar arrays grandes
            min_glob = min(np.min(p1), np.min(p2))
            max_glob = max(np.max(p1), np.max(p2))
            bb_width = max_glob - min_glob

            denom_aav = N * bb_width
            aav = min(width_sum / denom_aav, 1.0) if denom_aav > 1e-3 else 0.0

            # --- LSV (Dynamic p_max) ---
            if N < 2:
                lsv = 1.0
            else:
                def _lsv_b(arr):
                    p_mx = np.max(arr) - np.min(arr)
                    if p_mx <= 1e-3: return 1.0
                    # McNiven Adaptado: Sum(p_max - |diff|) / ((N-1)*p_max)
                    return np.sum(p_mx - np.abs(np.diff(arr))) / ((N - 1) * p_mx)

                lsv = _lsv_b(p1) * _lsv_b(p2)

            return lsv, aav

        # Aplicamos cálculo vectorial row-wise
        metrics = df_ef.apply(_get_metrics_row, axis=1, result_type='expand')

        # --- 4. Promedio Dinámico y Ponderación ---
        lsv_snap, aav_snap = metrics[0], metrics[1]

        # Shift por grupo de Arco (Compara i con i+1 dentro del mismo arco)
        lsv_next = lsv_snap.groupby(df_ef['ArcID']).shift(-1)
        aav_next = aav_snap.groupby(df_ef['ArcID']).shift(-1)

        # Promedio Dinámico (Viola et al.)
        lsv_dyn = ((lsv_snap + lsv_next) / 2.0).fillna(0)
        aav_dyn = ((aav_snap + aav_next) / 2.0).fillna(0)

        UM_dyn = df_ef['UM'].groupby(df_ef['ArcID']).shift(-1)

        # MCS Ponderado = Sum(LSV_dyn * AAV_dyn * UM_intervalo) / UM_Total
        #weighted_sum = (lsv_dyn * aav_dyn * df_ef['UM']).sum()
        weighted_sum = (lsv_dyn * aav_dyn * UM_dyn).sum()

        # Retorno final
        if self.um > 0:
            return weighted_sum / self.um
        else:
            return 0.0


    #
    def extract_lt_mcs(self):

      if self.is_wedged:
            return np.nan

      # Copia de la tabla original
      df = self.tabla.copy()

      # ---------------------------------------------------------
      # 1. Creación de dataframe efectivo
      # ---------------------------------------------------------
      df['lp_ini'] = df['lp_ini'].astype(int)
      df['lp_fin'] = df['lp_fin'].astype(int)

      l_min = df['lp_ini'].min()
      l_max = df['lp_fin'].max()

      # Columnas de láminas dinámicas B1 y B2
      cols_laminas = (
          [f'B1_L{i}' for i in range(l_min, l_max + 1)] +
          [f'B2_L{i}' for i in range(l_min, l_max + 1)]
      )

      # Filtrar solo columnas existentes
      cols_laminas = [c for c in cols_laminas if c in df.columns]

      # DataFrame solo con posiciones MLC
      df_mlc = df[cols_laminas].copy()

      # Distancia entre CP_k y CP_(k+1)
      diff_mlc_futuro = df_mlc.diff(periods=-1).abs().fillna(0)

      # ---------------------------------------------------------
      # 2. Procesamiento geométrico del gantry (arcos)
      # ---------------------------------------------------------
      gantry_rad = np.deg2rad(df['GantryAngle'])
      df['GantryContinuous'] = np.rad2deg(np.unwrap(gantry_rad))

      # UM = 0 marca un fin de arco → el siguiente CP inicia uno nuevo
      inicio_de_arco = (df['UM'] == 0)
      df['ArcID'] = inicio_de_arco.cumsum()

      # ---------------------------------------------------------
      # 3. LEAF TRAVEL NORMALIZADO
      # ---------------------------------------------------------
      lt_k = diff_mlc_futuro.sum(axis=1)
      max_lt = lt_k.max()

      if max_lt < 1e-6:
          lt_k_norm = lt_k * 0.0
      else:
          lt_k_norm = lt_k / max_lt

      # ---------------------------------------------------------
      # 4. Cálculo de geometría de polígonos
      # ---------------------------------------------------------

      poligonos = df['poligono'].values

      areas_raw = np.array([p.area for p in poligonos])
      bb_areas_raw = np.array([p.envelope.area for p in poligonos])

      perim_raw = np.array([p.length for p in poligonos])
      bb_perim_raw = np.array([p.envelope.length for p in poligonos])

      # máscara válida
      mask = (
          (areas_raw > 1e-3) &
          (bb_areas_raw > 1e-3) &
          (~np.isnan(areas_raw))
      )

      areas = areas_raw[mask]
      bb_areas = bb_areas_raw[mask]
      perims = perim_raw[mask]
      bb_perims = bb_perim_raw[mask]

      area_ratio = areas / bb_areas
      perim_ratio = perims / bb_perims


      # ---------------------------------------------------------
      # 5. Promedios forward-looking
      # ---------------------------------------------------------
      s_area_ratio = pd.Series(area_ratio)
      avg_area = (s_area_ratio + s_area_ratio.shift(-1)) / 2.0
      avg_area = avg_area.fillna(0)

      s_perim_ratio = pd.Series(perim_ratio)
      avg_perim = (s_perim_ratio + s_perim_ratio.shift(-1)) / 2.0
      avg_perim = avg_perim.fillna(0)


      UM_dyn = df['UM'].groupby(df['ArcID']).shift(-1)

      # ---------------------------------------------------------
      # 6. Producto de factores (serie)
      # ---------------------------------------------------------
      LT_mcs_s  = (1-lt_k_norm )* avg_area * avg_perim * UM_dyn

       # Normalizado por UM total del plan
      LT_mcs = LT_mcs_s.sum() / self.um

      return LT_mcs

    #
    def extract_beam_info(self):
        beam_info = {}

        beam_info['BeamNumber'] = self.numero
        beam_info['BeamName'] = getattr(self.raw, 'BeamName', None)
        beam_info['BeamDescription'] = getattr(self.raw, 'BeamDescription', None)

        Tipo = getattr(self.raw, 'BeamType', None)
        beam_info['BeamType'] = Tipo

        beam_info['Wedge'] = self.is_wedged

        beam_info['MCS'] = self.mcs
        beam_info['LT_MCS'] = self.lt_mcs

        beam_info['BeamMeterset'] = self.um
        beam_info['BeamDose'] = self.dosis
        beam_info['NumberOfCP'] = to_int_safe(getattr(self.raw, 'NumberOfControlPoints', None))
        beam_info['RadiationType'] = getattr(self.raw, 'RadiationType', None)

        # Solo para haces estáticos (3D)
        if Tipo == 'STATIC':
            CP_0 = self.raw.ControlPointSequence[0]
            beam_info['GantryAngle'] = getattr(CP_0, 'GantryAngle', None)
            beam_info['Energia'] = to_int_safe(getattr(CP_0, 'NominalBeamEnergy', None))
            beam_info['ColimatorAngle'] = getattr(CP_0, 'BeamLimitingDeviceAngle', None)

        return beam_info

    import numpy as np



    @property
    def atributos_beam(self):
        """
        Calcula un diccionario con métricas avanzadas de complejidad (QA/Anomalías).
        """
        # 1. Recuperar datos básicos
        df = self.tabla
        beam_dose = float(self.info.get('BeamDose', 0))
        um_total = float(self.um)
        cp_total = len(df)

        # Evitar divisiones por cero globales
        if um_total == 0: um_total = 1e-6
        if cp_total == 0: cp_total = 1


        #-----------------
        # -- UMS:
        #-----------------
        ums = df['UM'].values
        w_ums = ums / um_total
        w1_ums = ums / np.average(ums)
        w2_ums = ums / np.max(ums)

        # ---------------------------------------------------------
        # A. GEOMETRÍA Y FORMA (Polígonos)
        # ---------------------------------------------------------
        poligonos = df['poligono'].values

        # Cálculos vectorizados previos (más eficiente que hacerlo uno por uno después)
        # Usamos listas por comprensión que son rápidas para iterar objetos Shapely
        areas_raw = np.array([p.area for p in poligonos])
        bb_areas_raw = np.array([p.envelope.area for p in poligonos])
        ch_areas_raw = np.array([p.convex_hull.area for p in poligonos])
        perim_raw = np.array([p.length for p in poligonos])
        bb_perim_raw = np.array([p.envelope.length for p in poligonos])

        # Máscara de validez (evitar áreas 0 o nulas)
        mask = (areas_raw > 1e-3) & (bb_areas_raw > 1e-3) & (~np.isnan(areas_raw))

        # Filtrado de arrays
        areas = areas_raw[mask]
        bb_areas = bb_areas_raw[mask]
        ch_areas = ch_areas_raw[mask]
        perims = perim_raw[mask]
        bb_perims = bb_perim_raw[mask]

        # Métricas derivadas (solo con datos válidos)
        if len(areas) > 0:
            area_ratio = areas / bb_areas         # Extent (Rectangularidad)
            A3 = ch_areas / areas                 # Convexidad (Solidity inversa)
            area_norm = areas / (100*100)         # Tamaño relativo al 10x10
            perim_ratio = perims / bb_perims      # Perímetro relativo
            area_perim_ratio = areas / perims     # Area / Perímetro
        else:
            # Fallback seguro si no hay aperturas válidas
            area_ratio = np.zeros(1)
            A3 = np.zeros(1)
            area_norm = np.zeros(1)
            perim_ratio = np.zeros(1)

        Area_prop_very_small = np.mean(areas / areas.mean() < 0.25)

        # ---------------------------------------------------------
        # A2.  INTERSECCIONES EN UN POLIGONO (INTERDIGITACION DE MLC)
        # ---------------------------------------------------------
        n_self_intersections = np.array([contar_self_intersections(p)
            for p in poligonos])

        # ---------------------------------------------------------
        # A3.  CENTROIDES EN UN POLIGONO (OFF_AXIS DE MLC)
        # ---------------------------------------------------------
        centroides_x = np.array([p.centroid.x for p in poligonos])

        centroides_y = np.array([p.centroid.y for p in poligonos])
        cx_norm = centroides_x / 200 # relativo al tamaño maximo
        cy_norm = centroides_y / 200 # relativo al tamaño maximo

        # ---------------------------------------------------------
        # B.  AREAS Ponderadas por UM
        # ---------------------------------------------------------

        # area_ratio esta entr 0 y 1. siendo 1 equivalente a su envolvente = bajo riesgo

        A_wavg = w1_ums*(1-area_ratio)   #ums / np.average(ums)
        A_wmax = w2_ums*(1-area_ratio)   #ums / np.max(ums)

        # conteos
        A_wmax_50 = (A_wmax >= 0.5).sum()/len(A_wmax)   #R2 superiores a 0.5
        A_wavg_100 = (A_wavg >= 1).sum()/len(A_wavg)    #R1 superiores a 1

        A_wavg_tot = np.sum(A_wavg)
        A_wmax_tot = np.sum(A_wmax)

        # ---------------------------------------------------------
        # C.  Otras FORMAS
        # ---------------------------------------------------------
        #..compacidad
        compacidad = (4 * np.pi * areas) / (perims**2 + 1e-8)

        #..elongacion
        #dx = bounds[:, 2] - bounds[:, 0]
        #dy = bounds[:, 3] - bounds[:, 1]

        #elongacion = np.maximum(dx, dy) / (np.minimum(dx, dy) + 1e-8)

        #..rugosidad
        #rugosidad = perims / (perims_convex + 1e-8)
        rugosidad_area = perims / (np.sqrt(areas) + 1e-8)

        # ---------------------------------------------------------
        # D. CINEMÁTICA (Velocidades MLC y Jaws)
        # ---------------------------------------------------------


        # ---  Creación de DF Efectivo ---
        df['lp_ini'] = df['lp_ini'].astype(int)
        df['lp_fin'] = df['lp_fin'].astype(int)

        l_min = df['lp_ini'].min()
        l_max = df['lp_fin'].max()

        # Selección de columnas efectivas
        cols_laminas = ([f'B1_L{i}' for i in range(l_min, l_max + 1)] +
                        [f'B2_L{i}' for i in range(l_min, l_max + 1)])

        df_mlc = df[cols_laminas].copy()

        # Columnas Jaws
        jaw_cols = ['J1', 'J2']
        df_jaws = df[jaw_cols]

        # Velocidad MLC
        diff_mlc = df_mlc.diff().abs().fillna(0)
        serie_um = df['UM'].replace(0, np.nan) # Evitar div por cero

        vel_mlc = diff_mlc.div(serie_um, axis=0)
        vals_vel_mlc = vel_mlc.values.flatten()
        vals_vel_mlc = vals_vel_mlc[~np.isnan(vals_vel_mlc)] # Limpiar NaNs

        if len(vals_vel_mlc) > 0:
            media_vel = np.mean(vals_vel_mlc)
            vals_vel_mlc_norm = vals_vel_mlc / media_vel if media_vel > 0 else vals_vel_mlc
        else:
            vals_vel_mlc = np.array([0])
            vals_vel_mlc_norm = np.array([0])

        # B.2 alto y ancho promedio para normalizar la distancia de laminas:
        cant_pares_activos = (df['lp_fin'] - df['lp_ini'] + 1)
        largo_y_mm = cant_pares_activos * 5.0

        # Ancho X promedio del CP = Area / Largo Y
        anchos_x_cp = df['apertura'] / largo_y_mm

        ancho_promedio_x = np.nanmean(anchos_x_cp)
        largo_promedio_y = np.nanmean(largo_y_mm)

        # ---------------------------------------------------------
        # SAS FEATURES (SMALL APERTURE SCORE)
        # ---------------------------------------------------------
        # Separar bancos
        cols_b1 = [f'B1_L{i}' for i in range(l_min, l_max + 1)]
        cols_b2 = [f'B2_L{i}' for i in range(l_min, l_max + 1)]

        B1 = df_mlc[cols_b1].values
        B2 = df_mlc[cols_b2].values

        # APERTURE valor del par de lamina
        ap = B2 - B1

        # SAS Clasicas A nivel completo del haz
        SAS4 = np.mean(ap < 4)
        SAS10 = np.mean(ap < 10)
        SAS20 = np.mean(ap < 20)

        # SAS4 por apertura/poligoino/punto de control
        SAS4_CP = np.mean(ap < 4 , axis = 1)
        # -SAS4_CP ponderado por UM/UM_total
        SAS4_wUM = np.average(SAS4_CP, weights = w_ums)

        #SAS VECTOR hasta 20 mm
        SAS_VECTOR = np.array([np.mean(ap < t) for t in np.arange(1,21)])
        # Pendiente del sas_vector tomado desde 4 a 10
        SAS_SLOPE = (SAS_VECTOR[9] - SAS_VECTOR[3])/(10 - 4)
        # Area bajo la curva (SAS_VECTOR)
        SAS_AUC = np.trapezoid(SAS_VECTOR, np.arange(1,21))



        # ---------------------------------------------------------
        # E. GANTRY
        # ---------------------------------------------------------

        gantry = df['GantryAngle'].values
        gantry_cont = np.rad2deg(np.unwrap(np.deg2rad(gantry)))
        inicio_arco = (ums == 0)
        arc_id = np.cumsum(inicio_arco)
        dgantry = np.diff(gantry_cont)
        dgantry = np.insert(dgantry, 0, np.nan)

        #anular incrementos cuando cambia el arco:
        cambio_arco = np.insert(arc_id[1:] != arc_id[:-1], 0, True)
        dgantry[cambio_arco] = np.nan

        #valor abs
        ga_incremento = np.abs(dgantry)

        # cociente R = deltaUM / DeltaGantry
        R = np.where(
            (~np.isnan(ga_incremento)) & (ums != 0),
            ums / ga_incremento,
            np.nan
        )



        # Asumiendo que tienes la función helper disponible o es un método estático
        gantry_recorrido = np.nansum(ga_incremento)

        # ---------------------------------------------------------
        # F. OTRAS
        # ---------------------------------------------------------

        # Fraccion de puntos de control que poseen menos de umbral UM
        ums = df['UM'][df['UM'] > 0]

        conteo_3 = (ums <= 3).sum() / len(df)
        conteo_5 = (ums <= 5).sum() / len(df)
        conteo_10 = (ums <= 10).sum() / len(df)

        # ---------------------------------------------------------
        # G. VELOCIDADES LAMINA (NUEVO)
        # ---------------------------------------------------------

        #definicion de deltas um y delta gantry ('seguros')
        delta_um = df['UM'].values
        delta_gantry = ga_incremento


        #definicion de delta mlc
        delta_mlc = diff_mlc.values

        delta_um_col = delta_um.reshape(-1, 1)
        delta_gantry_col = delta_gantry.reshape(-1, 1)

        L_ums = np.divide(delta_mlc, delta_um_col,
                          out = np.zeros_like(delta_mlc),
                          where = delta_um_col != 0).flatten()

        L_gan = np.divide(delta_mlc, delta_gantry_col,
                          out = np.zeros_like(delta_mlc),
                          where = ~np.isnan(delta_gantry_col) & (delta_gantry_col != 0)).flatten()


        # ---------------------------------------------------------
        # H. RECORRIDOS JAWs (NUEVO)
        # ---------------------------------------------------------

        J1 = df_jaws['J1'].diff().abs().fillna(0).values
        J2 = df_jaws['J2'].diff().abs().fillna(0).values


        l_min = df['lp_ini'].min()
        l_max = df['lp_fin'].max()
        largo_y_total_mm = (l_max - l_min +1) * 5.0

        # Recorrido total de una mordaza normalizado con respecto a N laminas
        recorrido_J1 = np.sum(J1)
        recorrido_J2 = np.sum(J2)

        # Recorrido norm por largo_y_total_mm
        eps = 1e-8
        J1_norm = recorrido_J1 / (largo_y_total_mm + eps)
        J2_norm = recorrido_J2 / (largo_y_total_mm + eps)

        #---Velocidades (UM)
        #J1_vel = J1 / delta_gantry_safe
        #J2_vel = J2 / delta_gantry_safe
        J1_vel = np.divide(J1, delta_gantry,
                          out = np.zeros_like(J1),
                          where = ~np.isnan(delta_gantry) & (delta_gantry != 0))
        J2_vel = np.divide(J2, delta_gantry,
                          out = np.zeros_like(J2),
                          where = ~np.isnan(delta_gantry) & (delta_gantry != 0))

        # ---------------------------------------------------------
        # I. RECORRIDOS MLC (NUEVO)
        # ---------------------------------------------------------
        L_recorrido_tot = diff_mlc.values.sum()
        L_recorrido_prom = diff_mlc.values.mean()
        L_recorrido_normx = L_recorrido_prom / (ancho_promedio_x + 1e-8)
        L_recorrido_normxy = L_recorrido_tot / (ancho_promedio_x * largo_promedio_y + 1e-8)


        # ---------------------------------------------------------
        # K. PUNTAJE DE COINCIDENCIA ESPACIAL DE LAMINAS OPUESTAS
        # ---------------------------------------------------------

        CBEC = cross_bank_edge_coincidence(df)

        # ---------------------------------------------------------
        #  CONSTRUCCIÓN DEL DICCIONARIO
        # ---------------------------------------------------------
        metrics = {
            # --- Factor UM  ---
            "UM_5":           conteo_5,
            "UM_10":          conteo_10,
            "UM_asim":        stats.skew(ums, nan_policy='omit'),
            "UM_kurt":        stats.kurtosis(ums, nan_policy='omit'),

            # --- Demanda de Gantry ---
            "R_media":        np.nanmean(R),
            "R_cv":           np.nanstd(R) / np.nanmean(R),
            "R_asim":         stats.skew(R, nan_policy='omit'),
            "R_kurt":         stats.kurtosis(R, nan_policy='omit'),

            # --- Modulation Complexity Scores  ---
            "MCS": self.mcs,
            #"LT_MCS": self.lt_mcs,

            # --- Velocidad de Láminas y mordazas nuevo ---
            "L_ums_mean":    np.mean(L_ums),
            "L_ums_cv":      np.std(L_ums) / np.mean(L_ums),
            "L_ums_asim":    stats.skew(L_ums, nan_policy='omit'),
            "L_ums_kurt":    stats.kurtosis(L_ums, nan_policy='omit'),

            "L_gan_mean":    np.mean(L_gan),
            "L_gan_cv":      np.std(L_gan) / np.mean(L_gan),
            "L_gan_asim":    stats.skew(L_gan, nan_policy='omit'),
            "L_gan_kurt":    stats.kurtosis(L_gan, nan_policy='omit'),

            'J1_ums_mean':   np.mean(J1_vel),
            'J1_ums_cv':     np.std(J1_vel) / np.mean(J1_vel),

            'J2_ums_mean':   np.mean(J2_vel),
            'J2_ums_cv':     np.std(J2_vel) / np.mean(J2_vel),

            # --- Recorridos ---
            'J1_recorrido':  J1_norm,
            'J2_recorrido':  J2_norm,
            'L_recorrido_normx': L_recorrido_normx,
            'L_recorrido_normxy': L_recorrido_normxy,

             # --- Forma poligono (Area Ratio / Rectangularidad) ---
            "AREA_ratio_mean":     np.nanmean(area_ratio),
            "AREA_ratio_cv":       np.nanstd(area_ratio) / np.nanmean(area_ratio),
            "AREA_ratio_asim":     stats.skew(area_ratio, nan_policy='omit'),
            "AREA_ratio_kurt":     stats.kurtosis(area_ratio, nan_policy='omit'),

            # --- Forma poligono (Perim Ratio) ---
            "PERIM_ratio_mean":   np.nanmean(perim_ratio),
            "PERIM_ratio_cv":     np.nanstd(perim_ratio) / np.nanmean(perim_ratio),
            "PERIM_rati_asim":    stats.skew(perim_ratio, nan_policy='omit'),
            "PERIM_rati_kurt":    stats.kurtosis(perim_ratio, nan_policy='omit'),

            # --- Forma area/perim
            "AREA_PERIM_mean":    np.nanmean(area_perim_ratio),
            "AREA_PERIM_cv":      np.nanstd(area_perim_ratio) / np.nanmean(area_perim_ratio),
            "AREA_PERIM_asim":    stats.skew(area_perim_ratio, nan_policy='omit'),
            "AREA_PERIM_kurt":    stats.kurtosis(area_perim_ratio, nan_policy='omit'),

            # --- Forma poligono (Convexidad / A3) ---
            "A3_med":          np.nanmean(A3),
            "A3_cv":           np.nanstd(A3)/np.nanmean(A3),
            "A3_asim":         stats.skew(A3, nan_policy='omit'),
            "A3_kurt":         stats.kurtosis(A3, nan_policy='omit'),

            # --- Compacidad ---
            "COM_mean":       np.nanmean(compacidad),
            "COM_cv":         np.nanstd(compacidad) / np.nanmean(compacidad),
            "COM_asim":       stats.skew(compacidad, nan_policy='omit'),
            "COM_kurt":       stats.kurtosis(compacidad, nan_policy='omit'),

            # -- Elongacion ---
            #"ELO_mean":       np.nanmean(elongacion),
            #"ELO_cv":         np.nanstd(elongacion) / np.nanmean(elongacion),
            #"ELO_asim":       stats.skew(elongacion, nan_policy='omit'),
            #"ELO_kurt":       stats.kurtosis(elongacion, nan_policy='omit'),

            # -- Rugosidad segun area:
            "RU_area_mean":   np.nanmean(rugosidad_area),
            "RU_area_cv":     np.nanstd(rugosidad_area) / np.nanmean(rugosidad_area),
            "RU_area_asim":   stats.skew(rugosidad_area, nan_policy='omit'),
            "RU_area_kurt":   stats.kurtosis(rugosidad_area, nan_policy='omit'),

            # --- Aperturas (Tamaño) ---
            "Area_cv":          np.nanstd(areas)/np.nanmean(areas),
            "Area_skew":        stats.skew(areas, nan_policy='omit'),
            "Area_kurt":        stats.kurtosis(areas, nan_policy='omit'),
            "Area_prop_vs":     Area_prop_very_small,


            # --- Areas ponderadas  wavg y wmax con pesos diferentes ---

            "A_wavg_mean":     np.nanmean(A_wavg),
            "A_wmax_mean":      np.nanmean(A_wmax),
            "A_wavg_cv":       np.nanstd(A_wavg) / np.nanmean(A_wavg),

            "A_wavg_asim":     stats.skew(A_wavg, nan_policy='omit'),

            "A_wavg_kurt":     stats.kurtosis(A_wavg, nan_policy='omit'),


            "A_avg_tot":      A_wavg_tot,
            "A_wmax_tot":     A_wmax_tot,


            "A_wmax_conteo":  A_wmax_50,
            "A_wavg_conteo":  A_wavg_100,

            # --- Dosis y Modulación ---
            "DOSE_cp":(um_total / beam_dose) / cp_total,
            #"DOSE_gan":  (um_total / beam_dose) / gantry_recorrido

            # --- Small Aperture Scores ---
            'SAS4':         SAS4,
            'SAS10':        SAS10,
            'SAS20':        SAS20,
            'SAS4_SLOPE':   SAS_SLOPE,
            'SAS4_wUM':     SAS4_wUM,
            'SAS_AUC':      SAS_AUC,

            # --- LINT interdigitaciones de  laminas ---
            'LINT_mean':     np.mean(n_self_intersections),  #interdigfit promedio en una apertura
            'LINT_max':      np.max(n_self_intersections),  #maximo num de interdigit en un moismo apertura
            'LINT_frac':     np.mean(n_self_intersections > 0), #fraccion de aperturas con interdigit

            # --- Centroides ---
            "CX_mean":       np.nanmean(np.mean(cx_norm)),
            "CX_cv":         np.nanstd(np.abs(cx_norm)) / np.nanmean(np.abs(cx_norm)),
            #"CX_asim":       stats.skew(centroides_x, nan_policy='omit'),
            #"CX_kurt":       stats.kurtosis(centroides_x, nan_policy='omit'),
            "CY_mean":       np.nanmean(np.mean(cy_norm)),
            "CY_cv":         np.nanstd(np.abs(cy_norm)) / np.nanmean(np.abs(cy_norm)),
            #"CY_asim":       stats.skew(centroides_y, nan_policy='omit'),
            #"CY_kurt":       stats.kurtosis(centroides_y, nan_policy='omit'),

            # --- Cross Bank Edge Coincidence ---
            "CBEC": CBEC


            # --- Desplazamiento Total Promedio ---
            # Promedio del movimiento total acumulado por hoja/mordaza
            #"MLC_DIST_mean":   diff_mlc.sum().mean()/ancho_promedio_x,
            #"JAW_DIST_mean":  df_jaws.diff().abs().sum().mean()/largo_promedio_y
        }

        return metrics

