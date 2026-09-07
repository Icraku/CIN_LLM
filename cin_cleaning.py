"""
cleaning_cin.py
"""

import json
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(".env")
data_dir = Path(os.getenv("DATA_DIR"))

INPUT_CSV = data_dir / "cin_peads.csv"
OUTPUT_DIR = data_dir

# =================================================================

cin = pd.read_csv(INPUT_CSV, keep_default_na=False, dtype=str)

cin = cin.replace(['', 'Empty'], np.nan)

NUMERIC_COLUMNS = [
    'age_mths', 'age_years', 'age_days', 'temp', 'resp_rate', 'pulse_rate',
    'oxygen_sat', 'weight', 'height', 'muac', 'cough_dur', 'diarrhoea_dur',
    'convulsions_no',
]
for col in NUMERIC_COLUMNS:
    cin[col] = pd.to_numeric(cin[col], errors='coerce')

cin['date_adm'] = pd.to_datetime(cin['date_adm'], errors='coerce')

cin['is_minimum'] = cin['is_minimum'].astype(str)
cin = cin[(cin.is_minimum == '0') &
          (cin.date_adm > pd.Timestamp("2014-03-01")) &
          (cin.surgical_burns != "Yes")].copy()

# Age in months
cin['age_mths'] = cin['age_mths'].map(lambda x: np.nan if (x <= 0) | (x > 18 * 12) else x, na_action='ignore')
cin['age_years'] = cin['age_years'].map(lambda x: np.nan if (x <= 0) | (x > 18) else x, na_action='ignore')
cin['age'] = cin[['age_mths', 'age_years', 'age_days']].apply(
    lambda row: row['age_mths'] if row['age_mths'] > 11 else np.nansum([row['age_mths'], 12 * row['age_years']]),
    axis=1)
cin['age_cat'] = pd.cut(cin['age'], [0, 2, 12, 24, 60, 120, 217], right=False,
                         labels=["0-2mths", "2-11mths", "12-23mths", "24-59mths", "59-120 mths", "120 + mths"],
                         include_lowest=True)

# Gender
cin['child_sex'] = cin['child_sex'].replace({"Male": "M", "Female": "F"})

cin['vaccine_period'] = cin['date_adm'].map(lambda x: "pre" if x.year < 2010 else "post", na_action="ignore")

# Deep/acidotic breathing
"""
acidotic breathing: merge two variables (acidotic_breathing and acidotic__breathing)
"""
cin['acidotic__breathing'] = cin['acidotic__breathing'].fillna(cin['acidotic_breathing'])

# Pulse
cin['pulse'] = cin['pulse'].replace({"Normal": "No", "weak": "Yes"})

# cap refill category **needs guidance
cin['cap_refill_cat'] = cin['cap_refill_cat'].replace(
    {"1 Sec": "<=3 Sec",
     "2 Sec": "<=3 Sec",
     "3 Sec": "<=3 Sec",
     "4 Sec": ">3 Sec",
     "5 Sec": ">3 Sec",
     "6 Sec": ">3 Sec",
     "6+ Sec": ">3 Sec",
     "Indeterminate": np.nan
     }).replace({"<=3 Sec": "No", ">3 Sec": "Yes"})

# temp_gradient
cin['skin_temp'] = cin.skin_temp.replace({'hand': 'No', 'elbow': 'Yes', 'Shoulder': 'Yes'})

# pallor
cin['pallor'] = cin.pallor.replace({'none': 'No', '+ (mild/moderate)': 'Yes', '+ + + (severe)': 'Yes'})

# Decreased skin turgor
cin['skin_pinch'] = cin.skin_pinch.replace({'Immediate': 'No',
                                             '1 -2 secs': 'Yes',
                                             'more than or equal to 2secs': 'Yes'})

# AVPU/bsc score
cin['avpu'] = cin.avpu.replace({"Alert": "Yes", "Pain response": 'No',
                                 'Verbal response': 'No', 'Unresponsive': 'No',
                                 'Other scale': np.nan})

# can drink
cin['can_drink'] = cin.can_drink.replace({"No": "Yes", "Yes": "No"})
# stiff neck
cin['stiff_neck'] = cin.stiff_neck.replace({'No/soft': 'No', "Yes": 'Yes'})
# bulging_font
cin['bulging_font'] = cin.bulging_font.replace({'No/flat': 'No', "Yes": "Yes"})

# Temperature
cin['temp'] = cin.temp.map(lambda x: x if (x > 30) & (x < 50) else np.nan). \
    map(lambda x: "Yes" if (x < 36.5) | (x > 37.5) else "No", na_action='ignore')

# Respiratory rate
resp_rate_min = 20
resp_rate_max = 60


def resp_rate_recode(resp_rate, age_months):
    if (age_months == np.nan) | (resp_rate == np.nan):
        return np.nan
    elif (age_months >= 1) & (age_months < 12):
        return "Yes" if resp_rate >= 50 else "No"
    elif (age_months > 12) & (age_months <= 59):
        return "Yes" if resp_rate >= 40 else "No"
    elif (age_months > 59) & (age_months <= 12 * 12):
        return "Yes" if resp_rate >= 30 else "No"
    else:
        return np.nan


cin['resp_rate'] = cin.resp_rate.map(lambda x: x if (x > resp_rate_min) & (x < resp_rate_max) else np.nan)
cin['resp_rate'] = cin[['resp_rate', 'age']].apply(lambda row: resp_rate_recode(row['resp_rate'], row['age']), axis=1)

# pulse_rate -- stays numeric, not converted to Yes/No
cin['pulse_rate'] = cin.pulse_rate.map(lambda x: x if (x > 50) & (x < 250) else np.nan)

# oxygen sat
ox_sat_min = 70
cin['oxygen_sat'] = cin.oxygen_sat.map(lambda x: x if (x >= ox_sat_min) & (x <= 100) else np.nan). \
    map(lambda x: "Yes" if x < 90 else "No", na_action='ignore')

# jaundice
cin['jaundice'] = cin.jaundice.replace({'none': "No", '+ (mild/moderate)': "Yes", '+ + + (severe)': "Yes"})

# Weight and height
cin['weight'] = cin.weight.map(lambda x: np.nan if (x < 1.5) | (x > 60) else x)
cin['height'] = cin.height.map(lambda x: np.nan if (x < 30) | (x > 180) else x)

# oedema - Kwashiorkor
cin['oedema'] = cin.oedema.replace({'None': "No", 'Foot': "Yes", 'Face': "Yes", 'Knee': "Yes"})

# MUAC
cin['muac'] = cin.muac.map(lambda x: x if (x > 5) & (x < 30) else np.nan)

# Fever, cough, diarrhoea, vomiting, convulsions durations
cin.loc[cin.cough == "No", 'cough_dur'] = 0
cin.loc[(cin.cough_dur < 0) | (cin.cough_dur > 90), 'cough_dur'] = np.nan
cin['cough_2wks'] = cin['cough_2wks'].fillna(cin.cough_dur.map(lambda x: "Yes" if x >= 14 else "No", na_action='ignore'))

cin.loc[cin.diarrhoea == "No", 'diarrhoea_dur'] = 0
cin['diarrhoea_dur'] = cin.diarrhoea_dur.map(lambda x: x if (x >= 0) & (x <= 90) else np.nan)
cin.loc[cin.diarrhoea == "No", 'diarrhoea_bloody'] = "No"
cin['diarrhoea_14d'] = cin['diarrhoea_14d'].fillna(cin.diarrhoea_dur.map(lambda x: "Yes" if x >= 14 else "No", na_action='ignore'))

cin.loc[cin.convulsions == "No", 'convulsions_no'] = 0
cin['convulsions_no'] = cin.convulsions_no.map(lambda x: x if (x >= 0) & (x <= 25) else np.nan)
cin['partial_fits'] = cin.fits
cin.loc[cin.convulsions == "No", 'partial_fits'] = "No"

# outcome at discharge
cin['died'] = cin['outcome'].map({"Alive": 0, "Died": 1})

# save the fully cleaned CIN data as both CSV and Parquet
cin.to_csv(os.path.join(OUTPUT_DIR, "cin_cleaned_full.csv"), index=False)