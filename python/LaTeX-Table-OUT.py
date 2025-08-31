#!/usr/bin/env python
# coding: utf-8

# In[77]:


import pandas as pd
import os
from astropy.coordinates import SkyCoord
import astropy.units as u

# ------------------------------------------------------------------
# 1.  Load main photometry file
# ------------------------------------------------------------------
phot_df = pd.read_csv(
    "../utils/photometry_matches.txt",
    sep="\t"
)

# ------------------------------------------------------------------
# 2.  Load reference data  (now includes to_AB_offset)
# ------------------------------------------------------------------
references_path = "../utils/references.txt"
if os.path.exists(references_path):
    try:
        ref_df = pd.read_csv(
            references_path,
            sep="\t",
            dtype={"instrument": str, "reference": str},
            na_values=["--------", "NaN", "nan", ""],
            keep_default_na=True
        )
        ref_coords = SkyCoord(
            ra=ref_df["ra_deg"].values * u.deg,
            dec=ref_df["dec_deg"].values * u.deg
        )
    except Exception as e:
        print(f"Could not load references.txt: {e}")
        ref_df = None
        ref_coords = None
else:
    ref_df = None
    ref_coords = None

# ------------------------------------------------------------------
# 3.  Exposure‑time dictionary
# ------------------------------------------------------------------
exposure_times = {row["Filename"][:10]: 1440 for _, row in phot_df.iterrows()}

# ------------------------------------------------------------------
# 4.  Helpers for consistent output
# ------------------------------------------------------------------
def dash(val):
    if pd.isnull(val) or str(val).strip() in {"", "—", "--------", "nan", "NaN"}:
        return "-"
    return str(val)

def safe_fmt(val, fmt=":.2f"):
    if pd.isnull(val) or str(val).strip() in {"", "—", "--------", "nan", "NaN"}:
        return "-"
    try:
        return format(float(val), fmt)
    except (ValueError, TypeError):
        return dash(val)

# ------------------------------------------------------------------
# 5.  Write LaTeX rows
# ------------------------------------------------------------------
out_path = "utils/latex_table_rows.txt"
with open(out_path, "w") as f:
    for _, row in phot_df.iterrows():
        object_name = row["Filename"][:10]
        ra_deg, dec_deg = row["PreciseRA"], row["PreciseDEC"]
        ra_sex, dec_sex = row["PreciseRA_sex"], row["PreciseDEC_sex"]
        redshift = row["Redshift"]
        mag, err_up, err_low = row["MAG_APER"], row["MAGERRTOT_upper"], row["MAGERRTOT_lower"]
        obs_date = dash(row.get("ObservationDate", "-"))

        if any(pd.isnull(x) for x in [mag, err_up, err_low, redshift]):
            continue

        # RA/Dec LaTeX strings
        ra_h, ra_m, ra_s = ra_sex.split(":")
        ra_ltx = f"{int(ra_h)}\\,\\mathrm{{h}}\\,{int(ra_m)}\\,\\mathrm{{m}}\\,{float(ra_s):.2f}\\,\\mathrm{{s}}"
        dec_sign, dec_body = dec_sex[0], dec_sex[1:]
        dec_d, dec_m, dec_s = dec_body.split(":")
        dec_ltx = f"{dec_sign}{int(dec_d)}^\\circ\\,{int(dec_m)}\\,\\mathrm{{'}}\\,{float(dec_s):.2f}\\,\\mathrm{{''}}"

        # Reference placeholders
        filt = ref_source = date = ref_mag_ltx = delta_J = "-"

        # ----------------------------------------------------------
        #  Reference lookup and AB conversion
        # ----------------------------------------------------------
        if ref_df is not None:
            target = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
            sep  = target.separation(ref_coords)
            idx  = sep.argmin()
            if sep[idx] < 1.5 * u.arcsec:
                ref_row = ref_df.iloc[idx]

                ref_mag  = pd.to_numeric(ref_row.get("mag_ref"),    errors="coerce")
                mag_err  = pd.to_numeric(ref_row.get("magerr_ref"), errors="coerce")
                offset   = pd.to_numeric(ref_row.get("to_AB_offset"), errors="coerce")
                if pd.isnull(offset):         # treat blank as zero
                    offset = 0.0

                filt       = dash(ref_row.get("instrument"))
                ref_source = dash(ref_row.get("reference"))
                date       = dash(ref_row.get("date"))

                if pd.notnull(ref_mag):
                    ref_mag_AB = ref_mag + offset          # <-- convert to AB
                    if pd.notnull(mag_err):
                        ref_mag_ltx = f"${ref_mag_AB:.2f} \\pm {mag_err:.2f}$"
                    else:
                        ref_mag_ltx = f"${ref_mag_AB:.2f}$"
                    delta_J = f"${ref_mag_AB - mag:.2f}$"  

        exposure_time = exposure_times.get(object_name, 1440)

        # Output LaTeX row
        latex_row = (
            f"{object_name} & "
            f"{safe_fmt(ra_deg)} & {safe_fmt(dec_deg)} & "
            f"${ra_ltx}$ & ${dec_ltx}$ & "
            f"{safe_fmt(redshift, ':.3f')} & "
            f"{exposure_time} & "
            f"{obs_date} & "
            f"${mag:.2f}^{{+{err_up:.2f}}}_{{-{err_low:.2f}}}$ & "
            f"{dash(ref_mag_ltx)} & {dash(delta_J)} & "
            f"{filt} & {ref_source} & {date} \\\\\n"
        )
        f.write(latex_row)


# In[1]:


#!/usr/bin/env python3
# -*- coding: utf‑8 -*-

import os, re, numpy as np, pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

# Load instrument offsets
OFFSETS_PATH = "../utils/Instrument_to_GROND.txt" 
if not os.path.exists(OFFSETS_PATH):
    raise FileNotFoundError(f"{OFFSETS_PATH} not found")

offset_cols = ["Redshift",
               "MKO->GROND", "MMT->GROND", "VISTA->GROND",
               "SUBARU->GROND", "UKIRT->GROND", "Fourstar->GROND",
               "SOFI->GROND"]

offset_df = pd.read_csv(OFFSETS_PATH,
                        delim_whitespace=True, comment="#",
                        header=None, names=offset_cols)

offset_df["Redshift"] = pd.to_numeric(offset_df["Redshift"], errors="coerce")
offset_df["z_key"]    = offset_df["Redshift"].apply(lambda v: f"{v:.2f}")
offset_lookup         = offset_df.set_index("z_key") 


# truncate redshift to 2 decimal places, so direct match can be forced
def nearest_offset(z, column):
    try:
        z = float(z)
    except (ValueError, TypeError):
        return np.nan
    if column not in offset_df.columns:
        return np.nan
    idx = np.abs(offset_df["Redshift"] - z).argmin()
    return offset_df.iloc[idx][column]

# ---------------------------------------------------------
# 1.  Load photometry_matches.txt 
# ---------------------------------------------------------
phot_df = pd.read_csv("../utils/photometry_matches.txt",
                      sep="\t")

# -------------------------------------------------------------------------
# 2.  Load reference catalogue
# -------------------------------------------------------------------------
REF_PATH = "../utils/references.txt"
ref_df, ref_coords = None, None
if os.path.exists(REF_PATH):
    ref_df = pd.read_csv(REF_PATH, sep="\t",
                         dtype={"instrument": str, "reference": str},
                         keep_default_na=True,
                         na_values=["--------", "nan", "NaN", ""])
    ref_coords = SkyCoord(ref_df["ra_deg"].values * u.deg,
                          ref_df["dec_deg"].values * u.deg)


EXPTIME_PATH = "../utils/exptimes.txt"   # format: name\texptime

if os.path.exists(EXPTIME_PATH):
    try:
        exp_df = pd.read_csv(
            EXPTIME_PATH,
            sep="\t",
            dtype={"name": str, "exptime": float},
            na_values=["", "nan", "NaN"]
        )
        exptime = dict(zip(exp_df["name"], exp_df["exptime"]))
    except Exception as e:
        print(f"Could not load exptimes.txt: {e} — using default 1440 s.")
        exptime = {}
else:
    print("exptimes.txt not found — using default 1440 s.")
    exptime = {}

# -----------------------------------------------------------------------------
#  Helper functions
# ----------------------------------------------------------------------------
def dash(v):
    return "-" if pd.isnull(v) or str(v).strip() in {"", "—", "--------", "nan", "NaN"} else str(v)

def safe_fmt(v, fmt=":.3f"):
    if pd.isnull(v):
        return "-"
    try:
        return format(float(v), fmt)
    except Exception:
        return dash(v)

# --------------------------------------------
# 5. Mapping of Instruments
# ---------------------------------------------
_INSTRUMENT_PATTERNS = {
    "MKO->GROND":      [r"\bNIRI\b"],
    "MMT->GROND":      [r"\bSWIRC\b"],
    "VISTA->GROND":    [r"\bVISTA\b", r"\bVHS\b", r"\bVIKING\b",
                        r"\bVIKINGDR\d+\b", r"\bVHSDR\d+\b"],
    "SUBARU->GROND":   [r"\bMOIRCS\b"],
    "UKIRT->GROND":    [r"\bUKIRT\b", r"\bWFCAM\b", r"\bUHS\b",
                        r"\bUKIDSS\b", r"\bUHSDR\d+\b"],
    "Fourstar->GROND": [r"\bFOURSTAR\b"],
    "SOFI->GROND":     [r"\bSOFI\b"],
}

_COMPILED_PATTS = {
    col: [re.compile(p, re.I) for p in patterns]
    for col, patterns in _INSTRUMENT_PATTERNS.items()
}

# orders entries in references.txt into their instrument 'bins'
def get_offset_column(inst_raw: str | None) -> str | None:
    if not inst_raw:
        return None
    s = inst_raw.upper()
    for col, regex_list in _COMPILED_PATTS.items():
        if any(rgx.search(s) for rgx in regex_list):
            return col
    return None


# -------------------------------------------
# 6.  Outputs
# -------------------------------------------
LATEX_OUT = "../utils/latex_table_rows.txt"
TSV_OUT   = "../utils/table_rows.tsv"

with open(LATEX_OUT, "w") as latex_f, open(TSV_OUT, "w") as tsv_f:

    tsv_header = ["Object", "RA_deg", "DEC_deg", "RA_sex", "DEC_sex",
                  "Redshift", "Exposure_s", "ObsDate",
                  "Mag", "MagErr_plus", "MagErr_minus",
                  "SNR",
                  "RefMag_AB_raw", "RefMagErr", "ColourTerm", "RefMag_AB_corr",
                  "DeltaJ_corr", "DeltaJErr_plus", "DeltaJErr_minus",
                  "Instrument", "RefSource", "RefDate"]
    tsv_f.write("\t".join(tsv_header) + "\n")

    # ------------------------------------------------------------------
    # 7.  Iterate over rows in photometry_matches.txt
    # ------------------------------------------------------------------
    for _, row in phot_df.iterrows():

        m, du, dl, z = row["MAG_APER"], row["MAGERRTOT_upper"], \
                       row["MAGERRTOT_lower"], row["Redshift"]
        if any(pd.isnull(v) for v in [m, du, dl, z]):
            continue
        
        
        snr = row.get("SNR") or row.get("snr")

        obj   = row["Filename"][:10]
        ra_d, dec_d = row["PreciseRA"], row["PreciseDEC"]
        ra_s, dec_s = row["PreciseRA_sex"], row["PreciseDEC_sex"]
        obsdate     = dash(row.get("ObservationDate", "-"))

        # Define latex strings for RA and DEC
        ra_h, ra_m, ra_sec  = ra_s.split(":")
        ra_ltx  = f"{int(ra_h)}\\,\\mathrm{{h}}\\,{int(ra_m)}\\,\\mathrm{{m}}\\,{float(ra_sec):.2f}\\,\\mathrm{{s}}"
        sign, rest = dec_s[0], dec_s[1:]
        dd, dm, ds = rest.split(":")
        dec_ltx = f"{sign}{int(dd)}^\\circ\\,{int(dm)}\\,\\mathrm{{'}}\\,{float(ds):.2f}\\,\\mathrm{{''}}"

        ref_mag_AB_raw = ref_mag_AB_corr = colour_term = np.nan
        delta_J_corr = delta_J_err_plus = delta_J_err_minus = np.nan
        filt_disp = ref_src = ref_date = "-"
        ref_mag_tex = dash(np.nan)

        # Reference match
        if ref_df is not None:
            target = SkyCoord(ra=ra_d*u.deg, dec=dec_d*u.deg)
            dists  = target.separation(ref_coords)
            idx    = dists.argmin()
            if dists[idx] < 1.5*u.arcsec:
                ref = ref_df.iloc[idx]

                m_ref = pd.to_numeric(ref.get("mag_ref"),    errors="coerce")
                m_err = pd.to_numeric(ref.get("magerr_ref"), errors="coerce")
                toAB  = pd.to_numeric(ref.get("to_AB_offset"), errors="coerce")
                if pd.isnull(toAB): toAB = 0.0

                inst_raw = ref.get("instrument", "")
                filt_disp = dash(inst_raw)
                ref_src   = dash(ref.get("reference"))
                ref_date  = dash(ref.get("date"))
                
                # AB conversion
                if pd.notnull(m_ref):
                    ref_mag_AB_raw = float(m_ref) + toAB

                    col_col = get_offset_column(inst_raw)
                    if col_col:
                        magoffset = nearest_offset(z, col_col)
                        if pd.notnull(magoffset):
                            colour_term = float(magoffset)
                            ref_mag_AB_corr = ref_mag_AB_raw + colour_term
                        else:
                            ref_mag_AB_corr = ref_mag_AB_raw
                    else:
                        ref_mag_AB_corr = ref_mag_AB_raw

                    delta_J_corr = m - ref_mag_AB_corr

                    if pd.notnull(m_err):
                        delta_J_err_plus  = np.sqrt(m_err**2 + dl**2)
                        delta_J_err_minus = np.sqrt(m_err**2 + du**2)
                    else:
                        delta_J_err_plus, delta_J_err_minus = du, dl

                    ref_mag_tex = (f"${ref_mag_AB_corr:.2f} \\pm {m_err:.2f}$"
                                   if pd.notnull(m_err)
                                   else f"${ref_mag_AB_corr:.2f}$")
                    
                    
#------------------------------------------------------------------------------
# Writing output lines                    
#------------------------------------------------------------------------------                    
                    
        # Write LaTeX lines
        if pd.isnull(delta_J_corr):
            delta_ltx = "-"
        else:
            delta_ltx = f"${delta_J_corr:.3f}^{{+{delta_J_err_plus:.3f}}}_{{-{delta_J_err_minus:.3f}}}$"

        snr_ltx = dash(snr)
        if snr_ltx != "-":
            try:
                snr_ltx = f"${float(snr):.3f}$"
            except Exception:
                snr_ltx = dash(snr)

        latex_f.write(
            f"{obj} & "
            f"{safe_fmt(ra_d, ':.6f')} & {safe_fmt(dec_d, ':.6f')} & "
            f"${ra_ltx}$ & ${dec_ltx}$ & "
            f"{safe_fmt(z, ':.3f')} & "
            f"{int(exptime.get(obj, 1440))} & "
            f"{dash(obsdate)} & "
            f"${m:.2f}^{{+{du:.2f}}}_{{-{dl:.2f}}}$ & "
            f"{snr_ltx} & "            
            f"{ref_mag_tex} & {delta_ltx} & "
            f"{filt_disp} & {ref_src} & {ref_date} \\\\\n"
        )

        
        # Write TSV lines
        tsv_f.write("\t".join([
            obj, f"{ra_d:.6f}", f"{dec_d:.6f}", ra_s, dec_s,
            f"{z:.2f}", str(int(exptime.get(obj, 1440))), obsdate,
            f"{m:.3f}", f"{du:.3f}", f"{dl:.3f}",
            safe_fmt(snr, ".3f"),
            safe_fmt(ref_mag_AB_raw, ".3f"),
            safe_fmt(m_err, ".3f"),
            safe_fmt(colour_term, ".3f"),
            safe_fmt(ref_mag_AB_corr, ".3f"),
            safe_fmt(delta_J_corr, ".3f"),
            safe_fmt(delta_J_err_plus, ".3f"),
            safe_fmt(delta_J_err_minus, ".3f"),
            filt_disp, ref_src, ref_date
        ]) + "\n")


print("✓  Finished  – LaTeX rows →", LATEX_OUT,
      "   and TSV table →", TSV_OUT)


# In[76]:





# In[55]:





# In[ ]:




