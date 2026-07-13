"""
CONTIENE LA DEFINICION DE
1) FEATURES
2) SCORES
3) IDS
4) COMPLEXITY DIRECTION

"""

# ------------------------------------------------------------------------------
# COMPLEXITY DIRECTION 
# ------------------------------------------------------------------------------
COMPLEXITY_DIRECTION = {
 'UM_5':          +1,
 'UM_asim':       +1,
 'UM_kurt':       +1,
 'R_media':       +1,
 'R_cv':          +1,
 'R_asim':        +1,
 'R_kurt':        +1,
 'MCS':           -1,
 'L_ums_mean':    +1,
 'L_ums_cv':      +1,
 'L_ums_asim':    +1,
 'L_ums_kurt':    +1,
 'L_gan_mean':    +1,
 'L_gan_cv':      +1,
 'L_gan_asim':    +1,
 'L_gan_kurt':    +1,
 'J1_ums_mean':   +1,
 'J1_ums_cv':     +1,
 'J2_ums_mean':   +1,
 'J2_ums_cv':     +1,
 'J1_recorrido':  +1,
 'J2_recorrido':  +1,
 'L_recorrido_normx':+1,
 'L_recorrido_normxy':+1,
 'AREA_ratio_mean': -1,
 'AREA_ratio_cv':  +1,
 'AREA_ratio_asim': -1,
 'AREA_ratio_kurt': +1,
 'PERIM_ratio_mean':+1,
 'PERIM_ratio_cv':  +1,
 'PERIM_rati_asim': +1,
 'PERIM_rati_kurt': +1,
 'AREA_PERIM_mean': -1,
 'AREA_PERIM_cv':   +1,
 'AREA_PERIM_asim': -1,
 'AREA_PERIM_kurt': +1,
 'A3_med':          +1,
 'A3_cv':           +1,
 'A3_asim':         +1,
 'A3_kurt':         +1,
 'COM_mean':        -1,
 'COM_cv':          +1,
 'COM_asim':        -1,
 'COM_kurt':        +1,
 'RU_area_mean':    +1,
 'RU_area_cv':      +1,
 'RU_area_asim':    +1, 
 'RU_area_kurt':    +1,
 'Area_cv':         +1,
 'Area_skew':       +1,
 'Area_kurt':       +1,
 'Area_prop_vs':    +1,
 'A_wavg_mean':     +1,
 'A_wmax_mean':     +1,
 'A_wavg_cv':       +1,
 'A_wavg_asim':     +1,
 'A_wavg_kurt':     +1,
 'A_avg_tot':       +1,
 'A_wmax_tot':      +1,
 'A_wmax_conteo':   +1,
 'A_wavg_conteo':   +1,
 'DOSE_cp':         +1,
 'SAS4':            +1,
 'LINT_mean':       +1,
 'LINT_max':        +1,
 'LINT_frac':       +1,
 'CBEC':            +1
}

# ------------------------------------------------------------------------------
# FEATURES
# ------------------------------------------------------------------------------
FEATURES = [
       'UM_5','UM_asim', 'UM_kurt',
       'R_media', 'R_cv', 'R_asim','R_kurt',
       'MCS',
       'L_ums_mean', 'L_ums_cv', 'L_ums_asim', 'L_ums_kurt',
       'L_gan_mean', 'L_gan_cv', 'L_gan_asim', 'L_gan_kurt',
       'J1_ums_mean','J1_ums_cv', 'J2_ums_mean', 'J2_ums_cv',
       'J1_recorrido', 'J2_recorrido','L_recorrido_normx', 'L_recorrido_normxy',
       'AREA_ratio_mean', 'AREA_ratio_cv', 'AREA_ratio_asim', 'AREA_ratio_kurt',
       'PERIM_ratio_mean', 'PERIM_ratio_cv', 'PERIM_rati_asim', 'PERIM_rati_kurt',
       'AREA_PERIM_mean', 'AREA_PERIM_cv', 'AREA_PERIM_asim', 'AREA_PERIM_kurt',
       'A3_med', 'A3_cv', 'A3_asim','A3_kurt',
       'COM_mean', 'COM_cv', 'COM_asim', 'COM_kurt',
       'RU_area_mean', 'RU_area_cv','RU_area_asim', 'RU_area_kurt',
       'Area_cv', 'Area_skew', 'Area_kurt','Area_prop_vs',
       'A_wavg_mean', 'A_wmax_mean', 'A_wavg_cv','A_wavg_asim', 'A_wavg_kurt', 'A_avg_tot', 'A_wmax_tot',
       'A_wmax_conteo', 'A_wavg_conteo', 'DOSE_cp',
       'SAS4',
       'LINT_mean', 'LINT_max', 'LINT_frac', 'CBEC'
]
# ------------------------------------------------------------------------------
# IDS
# ------------------------------------------------------------------------------

IDS = ['ID', 'ID2', 'ID3', 'ID4', 'PLAN', 'PLAN_GRUPO']


# ------------------------------------------------------------------------------
# SCORES
# ------------------------------------------------------------------------------

SCORES = ['Score_IF', 'Score_KNN', 'Score_PCA', 'Score_LOF', 'Composite_score', 'Composite_label','Risk_percentile']



# ------------------------------------------------------------------------------
# MAPEO PARA AGRUPAR FEATURES
# ------------------------------------------------------------------------------

AGRUPAMIENTO_FEATURES = {

    # ==========================
    # Dosis
    # ==========================
    'UM_5': 'Dosis',
    'UM_asim': 'Dosis',
    'UM_kurt': 'Dosis',
    'DOSE_cp': 'Dosis',


    # ==========================
    # Velocidad
    # ==========================
    'R_media': 'Velocidad',
    'R_cv': 'Velocidad',
    'R_asim': 'Velocidad',
    'R_kurt': 'Velocidad',

    'L_ums_mean': 'Velocidad',
    'L_ums_cv': 'Velocidad',
    'L_ums_asim': 'Velocidad',
    'L_ums_kurt': 'Velocidad',

    'L_gan_mean': 'Velocidad',
    'L_gan_cv': 'Velocidad',
    'L_gan_asim': 'Velocidad',
    'L_gan_kurt': 'Velocidad',

    'J1_ums_mean': 'Velocidad',
    'J1_ums_cv': 'Velocidad',

    'J2_ums_mean': 'Velocidad',
    'J2_ums_cv': 'Velocidad',


    # ==========================
    # Recorrido
    # ==========================
    'J1_recorrido': 'Recorrido',
    'J2_recorrido': 'Recorrido',
    'L_recorrido_normx': 'Recorrido',
    'L_recorrido_normxy': 'Recorrido',


    # ==========================
    # Forma
    # ==========================
    'AREA_ratio_mean': 'Forma',
    'AREA_ratio_cv': 'Forma',
    'AREA_ratio_asim': 'Forma',
    'AREA_ratio_kurt': 'Forma',

    'PERIM_ratio_mean': 'Forma',
    'PERIM_ratio_cv': 'Forma',
    'PERIM_rati_asim': 'Forma',
    'PERIM_rati_kurt': 'Forma',

    'AREA_PERIM_mean': 'Forma',
    'AREA_PERIM_cv': 'Forma',
    'AREA_PERIM_asim': 'Forma',
    'AREA_PERIM_kurt': 'Forma',

    'A3_med': 'Forma',
    'A3_cv': 'Forma',
    'A3_asim': 'Forma',
    'A3_kurt': 'Forma',

    'COM_mean': 'Forma',
    'COM_cv': 'Forma',
    'COM_asim': 'Forma',
    'COM_kurt': 'Forma',

    'RU_area_mean': 'Forma',
    'RU_area_cv': 'Forma',
    'RU_area_asim': 'Forma',
    'RU_area_kurt': 'Forma',

    'Area_cv': 'Forma',
    'Area_skew': 'Forma',
    'Area_kurt': 'Forma',
    'Area_prop_vs': 'Forma',

    'SAS4': 'Forma',


    # ==========================
    # Forma + Dosis
    # ==========================
    'A_wavg_mean': 'Forma&Dosis',
    'A_wmax_mean': 'Forma&Dosis',
    'A_wavg_cv': 'Forma&Dosis',
    'A_wavg_asim': 'Forma&Dosis',
    'A_wavg_kurt': 'Forma&Dosis',

    'A_avg_tot': 'Forma&Dosis',
    'A_wmax_tot': 'Forma&Dosis',
    'A_wmax_conteo': 'Forma&Dosis',
    'A_wavg_conteo': 'Forma&Dosis',


    # ==========================
    # MCS
    # ==========================
    'MCS': 'MCS',


    # ==========================
    # Interdigitación
    # ==========================
    'LINT_mean': 'Interdigitacion',
    'LINT_max': 'Interdigitacion',
    'LINT_frac': 'Interdigitacion',
    'CBEC': 'Interdigitacion'
}