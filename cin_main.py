"""
cin_main.py,
"""

import os
import pandas as pd
import dotenv
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(".env")
data_dir = Path(os.getenv("DATA_DIR"))

CLEANED_CSV = data_dir / "cin_cleaned_full.csv"
OUTPUT_TEXT = data_dir / "extracted_records.jsonl"

FIELDS_TO_EXTRACT = [
    "age_years",
    "age_mths",
    "child_sex",
    "dob",
    "weight",
    "height",
    "muac",
    "par",
    "fever",
    "cough",
    "tb_contact",  # not sure of this
    "diff_breath",
    "diarrhoea",
    "vomits",
    "diff_feed",
    "convulsions",
    "fits",
    "urine_color",
    "pmtct",
    "temp",
    "resp_rate",
    "pulse_rate",
    "oxygen_sat",
    "thrush",
    "lymph_nd",
    "wrist_sign",
    "f_clubbing",
    "jaundice",
    "sev_wasting",
    "oedema",
    "umbil",
    "stridor",
    "c_cyanosis",
    "indrawing",
    "grunting",
    "acidotic_breathing",
    "wheeze",
    "crackles",
    "pulse",
    "cap_refill_cat",
    "skin_temp",
    "pallor",
    "skin_pinch",
    "can_sit",
    "posture",
    "blantyre_score",
    "bcs_motor",
    "bcs_verbal",
    "bcs_eye",
    "avpu",
    "can_drink",
    "stiff_neck",
    "bulging_font",
    "irrit",
    "red_mov",
]

# Values that count as "missing".
EMPTY_VALUES = {"", "-1", "null"}

# Fields that get a unit appended to their VALUE, e.g. "3.2" -> "3.2 kg".
FIELD_UNITS = {
    "weight": "kg",
    "height": "cm",
    "muac": "cm",
    "temp": "c",
    "resp_rate": "bpm",
    "pulse_rate": "bpm",
    "oxygen_sat": "pct",
}

# =============================================================
def is_present(value):
    """True if a cell has a real value, False if missing."""
    return value.strip().lower() not in EMPTY_VALUES


def extract_row(row, fields_wanted):
    """
    A dict for keeping only fields that: are in `fields_wanted` and those that have a value in this row.
    """
    record = {}
    columns_to_check = fields_wanted if fields_wanted else row.index

    for field in columns_to_check:
        if field not in row.index:
            # If column isn't in the CSV at all, skip.
            continue
        value = row[field]
        if is_present(value):
            record[field] = value.strip()

    return record

def add_units(record):
    """
    Add units to fields in the record, e.g. "weight" -> "3.2 kg".
    """
    years = record.pop("age_years", None)
    months = record.pop("age_mths", None)
    age_parts = []
    if years is not None:
        age_parts.append(f"{years}Yr")
    if months is not None:
        age_parts.append(f"{months}M")
    if age_parts:
        record["age"] = " ".join(age_parts)

    for field, unit in FIELD_UNITS.items():
        if field in record:
            record[field] = f"{record[field]} {unit}"

    return record

def record_to_text(record):
    """Format record"""
    return "\n".join(f"{key}: {value}" for key, value in record.items())

def main():
    """
    Read the CSV, extract the desired fields and write to JSONL.
    :return: JSONL file with one JSON object per row, containing only the fields that were present and non-empty in the CSV.
    """
    df = pd.read_csv(CLEANED_CSV, dtype=str, keep_default_na=False)

    rows_written = 0
    with open(OUTPUT_TEXT, "w", encoding="utf-8") as jsonl_out:
        for _, row in df.iterrows():
            record = extract_row(row, FIELDS_TO_EXTRACT)

            if not record:
                continue

            record = add_units(record)

            jsonl_out.write(record_to_text(record) + "\n\n")
            rows_written += 1

            if rows_written % 5000 == 0:
                print(f"...processed {rows_written} rows so far")

    print(f"Wrote {rows_written} records to {OUTPUT_TEXT}")


if __name__ == "__main__":
    main()