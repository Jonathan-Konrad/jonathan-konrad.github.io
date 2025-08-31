#!/usr/bin/env python
# coding: utf-8

# In[16]:


import os
import re
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table
import numpy as np
from astropy.io import fits
import time
from photutils.aperture import CircularAperture, aperture_photometry

start = time.time()

# === SETTINGS ===
catalog_folder = "../utils/fits_folder/catalogs"
background_rms_folder = "../utils/fits_folder/backgrounds_rms"
ods_file_path = "../utils/aa61_fan_qso_database.ods"
output_file = "../utils/photometry_matches.txt"

def find_closest_in_ods(ods_file, target_ra, target_dec):
    df = pd.read_excel(ods_file, engine="odf", header=[0, 1])
    ra_cols = [col for col in df.columns if col[0].lower() == "ra"]
    dec_cols = [col for col in df.columns if col[0].lower() == "dec"]

    if not ra_cols or not dec_cols:
        raise ValueError("ODS file must contain 'ra' and 'dec' columns")

    ra_col, dec_col = ra_cols[0], dec_cols[0]
    df[ra_col] = pd.to_numeric(df[ra_col], errors="coerce")
    df[dec_col] = pd.to_numeric(df[dec_col], errors="coerce")

    target = SkyCoord(ra=target_ra * u.deg, dec=target_dec * u.deg)
    coords = SkyCoord(ra=df[ra_col].values * u.deg, dec=df[dec_col].values * u.deg)

    idx = target.separation(coords).argmin()
    return df.iloc[idx][ra_col], df.iloc[idx][dec_col]

def find_closest_in_catalog(cat_path, target_ra, target_dec, max_sep_arcsec=2.0):
    cat = Table.read(cat_path, format="ascii")

    if not {"ALPHA_J2000", "DELTA_J2000"}.issubset(cat.colnames):
        raise ValueError(f"{cat_path} missing ALPHA_J2000 or DELTA_J2000")

    target = SkyCoord(ra=target_ra * u.deg, dec=target_dec * u.deg)
    cat_coords = SkyCoord(ra=cat["ALPHA_J2000"], dec=cat["DELTA_J2000"], unit=(u.deg, u.deg))

    sep = target.separation(cat_coords)
    min_sep = sep.min()

    if min_sep > max_sep_arcsec * u.arcsec:
        raise ValueError(f"No match within {max_sep_arcsec} arcsec (closest = {min_sep.arcsec:.2f} arcsec)")

    idx = sep.argmin()
    row = cat[idx]

    return {
        "ALPHA_J2000": float(row["ALPHA_J2000"]),
        "DELTA_J2000": float(row["DELTA_J2000"]),
        "MAG_APER": row.get("MAG_APER", None),
        "MAGERR_APER": row.get("MAGERR_APER", None),
        "FLUX_APER": row.get("FLUX_APER", None),
        "FLUXERR_APER": row.get("FLUXERR_APER", None),
        "X_IMAGE": row.get("X_IMAGE", None),
        "Y_IMAGE": row.get("Y_IMAGE", None),
        "FILENAME": os.path.basename(cat_path)
    }

def parse_filename_coords(filename):
    match = re.search(r"J(\d{2})(\d{2})([+-])(\d{2})(\d{2})", filename)
    if not match:
        raise ValueError(f"Filename {filename} does not match pattern JHHMM+DDMM")

    ra_h = int(match.group(1))
    ra_m = int(match.group(2))
    sign = 1 if match.group(3) == "+" else -1
    dec_d = int(match.group(4))
    dec_m = int(match.group(5))

    ra_deg = (ra_h + ra_m / 60) * 15
    dec_deg = sign * (dec_d + dec_m / 60)

    return ra_deg, dec_deg

def extract_zpd_from_filename(filename):
    zpd_match = re.search(r"_ZPD([0-9p\-]+)", filename)
    zpd_err_match = re.search(r"_ERR([0-9p\-]+)", filename)

    def convert_str_to_float(s):
        return float(s.replace('p', '.')) if s else None

    zpd = convert_str_to_float(zpd_match.group(1) if zpd_match else None)
    zpd_err = convert_str_to_float(zpd_err_match.group(1) if zpd_err_match else None)
    return zpd, zpd_err


#DEFAULT method can be center, subpixel or exact (chose exact to best emulate SExtractor)
#The in-use method is set further below
def compute_background_error_photutils(rms_path, x_center, y_center, diameter, method="center"):    
    try:
        with fits.open(rms_path) as hdul:
            data = hdul[0].data

        radius = diameter / 2.0
        positions = [(x_center, y_center)]
        aperture = CircularAperture(positions, r=radius)

        squared_data = data**2
        phot_table = aperture_photometry(squared_data, aperture, method=method)
        sum_of_squares = phot_table['aperture_sum'][0]
        background_error = np.sqrt(sum_of_squares)
        
        return background_error
    except Exception as e:
        print(f"Failed to compute background error from {rms_path}: {e}")
        return None

def main():
    all_data = []

    for cat_filename in sorted(os.listdir(catalog_folder)):
        if not cat_filename.endswith(".cat"):
            continue

        base_match = re.match(r"(J\d{4}[+-]\d{4}(\(dup\))?)", cat_filename)
        if base_match:
            base = base_match.group(1)
        else:
            print(f"Skipping {cat_filename}: could not extract base name")
            continue

        try:
            rough_ra, rough_dec = parse_filename_coords(base)
            precise_ra, precise_dec = find_closest_in_ods(ods_file_path, rough_ra, rough_dec)
            cat_path = os.path.join(catalog_folder, cat_filename)
            phot = find_closest_in_catalog(cat_path, precise_ra, precise_dec)
        except Exception as e:
            print(f"Skipping {cat_filename}: {e}")
            continue

        ap_match = re.search(r"_ap([0-9p]+)", cat_filename)
        aperture_diam = float(ap_match.group(1).replace('p', '.')) if ap_match else 1.0

        zpd, zpd_err = extract_zpd_from_filename(cat_filename)

        rms_filename = f"{base}_ap{ap_match.group(1)}_background_rms.fits"
        rms_path = os.path.join(background_rms_folder, rms_filename)
        background_error = compute_background_error_photutils(
            rms_path, phot["X_IMAGE"], phot["Y_IMAGE"], aperture_diam, method="center"    
        )

        all_data.append({
            "Base": base,
            "Filename": cat_filename,
            "PreciseRA": precise_ra,
            "PreciseDEC": precise_dec,
            "ALPHA_J2000": phot["ALPHA_J2000"],
            "DELTA_J2000": phot["DELTA_J2000"],
            "MAG_APER": phot["MAG_APER"],
            "MAGERR_APER": phot["MAGERR_APER"],
            "FLUX_APER": phot["FLUX_APER"],
            "FLUXERR_APER": phot["FLUXERR_APER"],
            "X_IMAGE": phot["X_IMAGE"],
            "Y_IMAGE": phot["Y_IMAGE"],
            "Aperture": aperture_diam,
            "ZPD": zpd,
            "ZPD_ERR": zpd_err,
            "Background_ERR": background_error
        })

    df = pd.DataFrame(all_data)

    with open(output_file, "w") as f:
        header = (
            "Filename\tPreciseRA\tPreciseDEC\tALPHA_J2000\tDELTA_J2000\t"
            "MAG_APER\tMAGERR_APER\tFLUX_APER\tFLUXERR_APER\tX_IMAGE\tY_IMAGE\t"
            "Aperture\tZPD\tZPD_ERR\tBackground_ERR\n"
        )
        f.write(header)
        for _, r in df.iterrows():
            line = (
                f"{r['Filename']}\t{r['PreciseRA']:.6f}\t{r['PreciseDEC']:.6f}\t"
                f"{r['ALPHA_J2000']:.6f}\t{r['DELTA_J2000']:.6f}\t"
                f"{r['MAG_APER']}\t{r['MAGERR_APER']}\t{r['FLUX_APER']}\t{r['FLUXERR_APER']}\t"
                f"{r['X_IMAGE']}\t{r['Y_IMAGE']}\t{r['Aperture']}\t"
                f"{r['ZPD']}\t{r['ZPD_ERR']}\t{r['Background_ERR']}\n"
            )
            f.write(line)

    print(f"Finished! Results saved to {output_file}")

if __name__ == "__main__":
    main()

end = time.time()
print(f"Total computation time: {end - start:.2f} seconds")


# In[26]:


import os
import re
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table
import numpy as np
from astropy.io import fits
import math
import time
from photutils.aperture import CircularAperture, aperture_photometry

start = time.time()

catalog_folder = "../utils/fits_folder/catalogs"
background_rms_folder = "../utils/fits_folder/backgrounds_rms"
ods_file_path = "../utils/aa61_fan_qso_database.ods"
output_file = "../utils/photometry_matches.txt"
FIDUCIAL_ZPD_ERR = 0.1

def find_closest_in_ods(ods_file, target_ra, target_dec):
    df = pd.read_excel(ods_file, engine="odf", header=[0, 1])
    ra_cols = [col for col in df.columns if col[0].lower() == "ra"]
    dec_cols = [col for col in df.columns if col[0].lower() == "dec"]
    redshift_cols = [col for col in df.columns if col[0].lower() == "redshift"]

    if not ra_cols or not dec_cols:
        raise ValueError("ODS file must contain 'ra' and 'dec' columns")

    ra_col, dec_col = ra_cols[0], dec_cols[0]
    redshift_col = redshift_cols[0] if redshift_cols else None

    df[ra_col] = pd.to_numeric(df[ra_col], errors="coerce")
    df[dec_col] = pd.to_numeric(df[dec_col], errors="coerce")

    target = SkyCoord(ra=target_ra * u.deg, dec=target_dec * u.deg)
    coords = SkyCoord(ra=df[ra_col].values * u.deg, dec=df[dec_col].values * u.deg)

    idx = target.separation(coords).argmin()
    row = df.iloc[idx]
    redshift = row[redshift_col] if redshift_col else None
    return row[ra_col], row[dec_col], redshift



def main():
    all_data = []

    for cat_filename in sorted(os.listdir(catalog_folder)):
        if not cat_filename.endswith(".cat"):
            continue

        base_match = re.match(r"(J\d{4}[+-]\d{4}(\(dup\))?)", cat_filename)
        if base_match:
            base = base_match.group(1)
        else:
            print(f"Skipping {cat_filename}: could not extract base name")
            continue

        try:
            rough_ra, rough_dec = parse_filename_coords(base)
            precise_ra, precise_dec, redshift = find_closest_in_ods(ods_file_path, rough_ra, rough_dec)

            skycoord = SkyCoord(ra=precise_ra * u.deg, dec=precise_dec * u.deg)
            precise_ra_sex = skycoord.ra.to_string(unit=u.hourangle, sep=':', precision=2, pad=True)
            precise_dec_sex = skycoord.dec.to_string(unit=u.deg, sep=':', precision=2, alwayssign=True, pad=True)

            cat_path = os.path.join(catalog_folder, cat_filename)
            phot = find_closest_in_catalog(cat_path, precise_ra, precise_dec, max_sep=MAX_SEP_ARCSEC)
        except Exception as e:
            print(f"Skipping {cat_filename}: {e}")
            continue

        ap_match = re.search(r"_ap([0-9p]+)", cat_filename)
        aperture_diam = float(ap_match.group(1).replace('p', '.')) if ap_match else 1.0

        zpd, zpd_err = extract_zpd_from_filename(cat_filename)

        rms_filename = f"{base}_ap{ap_match.group(1)}_background_rms.fits"
        rms_path = os.path.join(background_rms_folder, rms_filename)
        background_error = compute_background_error_photutils(
            rms_path, phot["X_IMAGE"], phot["Y_IMAGE"], aperture_diam, method="center"
        )

        # NEW: Extract DATEOBS1 from background FITS header
        try:
            with fits.open(rms_path) as hdul:
                dateobs_full = hdul[0].header.get("DATEOBS1", "")
                observation_date = dateobs_full.split("T")[0] if "T" in dateobs_full else dateobs_full
        except Exception as e:
            observation_date = ""

        magerr_plus, magerr_minus = flux_to_magerr_asymmetric(
            phot["FLUX_APER"], phot["FLUXERR_APER"]
        )

        bg_magerr_upper, bg_magerr_lower = background_flux_to_magerr_asymmetric(
            phot["FLUX_APER"], background_error
        )

        magerrtot_upper = safe_quad_sum(zpd_err, bg_magerr_upper, magerr_plus)
        magerrtot_lower = safe_quad_sum(zpd_err, bg_magerr_lower, magerr_minus)

        all_data.append({
            "Base": base,
            "Filename": cat_filename,
            "PreciseRA": precise_ra,
            "PreciseDEC": precise_dec,
            "PreciseRA_sex": precise_ra_sex,
            "PreciseDEC_sex": precise_dec_sex,
            "Redshift": redshift,
            "ALPHA_J2000": phot["ALPHA_J2000"],
            "DELTA_J2000": phot["DELTA_J2000"],
            "MAG_APER": phot["MAG_APER"],
            "MAGERR_APER": phot["MAGERR_APER"],
            "FLUX_APER": phot["FLUX_APER"],
            "FLUXERR_APER": phot["FLUXERR_APER"],
            "X_IMAGE": phot["X_IMAGE"],
            "Y_IMAGE": phot["Y_IMAGE"],
            "Aperture": aperture_diam,
            "ZPD": zpd,
            "ZPD_ERR": zpd_err,
            "Background_ERR": background_error,
            "MAGERR_FROMFLUXAPER_PLUS": magerr_plus,
            "MAGERR_FROMFLUXAPER_MINUS": magerr_minus,
            "Background_ERR_upper": bg_magerr_upper,
            "Background_ERR_lower": bg_magerr_lower,
            "MAGERRTOT_upper": magerrtot_upper,
            "MAGERRTOT_lower": magerrtot_lower,
            "ObservationDate": observation_date
        })

    df = pd.DataFrame(all_data)

    with open(output_file, "w") as f:
        header = (
            "Filename\tPreciseRA\tPreciseDEC\tPreciseRA_sex\tPreciseDEC_sex\tRedshift\t"
            "ALPHA_J2000\tDELTA_J2000\tMAG_APER\tMAGERR_APER\tFLUX_APER\tFLUXERR_APER\t"
            "X_IMAGE\tY_IMAGE\tAperture\tZPD\tZPD_ERR\tBackground_ERR\t"
            "MAGERR_FROMFLUXAPER_PLUS\tMAGERR_FROMFLUXAPER_MINUS\t"
            "Background_ERR_upper\tBackground_ERR_lower\tMAGERRTOT_upper\tMAGERRTOT_lower\t"
            "ObservationDate\n"
        )
        f.write(header)
        for _, r in df.iterrows():
            line = (
                f"{r['Filename']}\t{r['PreciseRA']:.6f}\t{r['PreciseDEC']:.6f}\t"
                f"{r['PreciseRA_sex']}\t{r['PreciseDEC_sex']}\t{r['Redshift']}\t"
                f"{r['ALPHA_J2000']:.6f}\t{r['DELTA_J2000']:.6f}\t"
                f"{r['MAG_APER']}\t{r['MAGERR_APER']}\t{r['FLUX_APER']}\t{r['FLUXERR_APER']}\t"
                f"{r['X_IMAGE']}\t{r['Y_IMAGE']}\t{r['Aperture']}\t"
                f"{r['ZPD']}\t{r['ZPD_ERR']}\t{r['Background_ERR']}\t"
                f"{r['MAGERR_FROMFLUXAPER_PLUS']}\t{r['MAGERR_FROMFLUXAPER_MINUS']}\t"
                f"{r['Background_ERR_upper']}\t{r['Background_ERR_lower']}\t"
                f"{r['MAGERRTOT_upper']}\t{r['MAGERRTOT_lower']}\t"
                f"{r['ObservationDate']}\n"
            )
            f.write(line)

    print(f"Finished! Results saved to {output_file}")

if __name__ == "__main__":
    main()

end = time.time()
print(f"Total computation time: {end - start:.2f} seconds")


# In[39]:


#!/usr/bin/env python3
# -*- coding: utf‑8 -*-

"""
Collect photometry from THELI *.cat files, match to fiducial quasar
coordinates from an .ods spreadsheet, apply background‑noise estimates
and zero‑point data, and write a tab‑separated output table.

• matching radius is configurable with MAX_SEP_ARCSEC
• observation date (DATEOBS1) is read from the *_background_rms.fits header
• asymmetric flux and background errors are propagated in quadrature
"""

import os, re, time, math
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table
from astropy.io import fits
from photutils.aperture import CircularAperture, aperture_photometry

# ----------------------------------------------------------------------
# 0.  SETTINGS (edit folder paths here)
# ----------------------------------------------------------------------
catalog_folder        = "../utils/fits_folder/catalogs"
background_rms_folder = "../utils/fits/folder/backgrounds_rms"
ods_file_path         = "../utils/aa61_fan_qso_database.ods"
output_file           = "../utils/photometry_matches.txt"

MAX_SEP_ARCSEC = 2.0          # <-- configurable match radius
FIDUCIAL_ZPD_ERR = 0.10       # fallback ZP‑error if none in filename

start = time.time()

# ----------------------------------------------------------------------
# 1.  HELPER FUNCTIONS
# ----------------------------------------------------------------------
def parse_filename_coords(name: str):
    """Parse 'JHHMM±DDMM...' string ⇒ (RA_deg, Dec_deg) rough coords."""
    m = re.search(r"J(\d{2})(\d{2})([+-])(\d{2})(\d{2})", name)
    if not m:
        raise ValueError(f"Cannot parse J‑name coordinates from '{name}'")
    ra_h, ra_m   = int(m.group(1)), int(m.group(2))
    sign         = 1 if m.group(3) == "+" else -1
    dec_d, dec_m = int(m.group(4)), int(m.group(5))
    ra_deg  = (ra_h + ra_m/60) * 15.0
    dec_deg = sign * (dec_d + dec_m/60)
    return ra_deg, dec_deg


def find_closest_in_ods(ods_file, target_ra, target_dec):
    """Return (precise RA, Dec, redshift) of closest entry in the ODS file."""
    df = pd.read_excel(ods_file, engine="odf", header=[0, 1])
    ra_col  = [c for c in df.columns if c[0].lower() == "ra"][0]
    dec_col = [c for c in df.columns if c[0].lower() == "dec"][0]
    z_col   = [c for c in df.columns if c[0].lower() == "redshift"]
    z_col   = z_col[0] if z_col else None

    df[ra_col]  = pd.to_numeric(df[ra_col],  errors="coerce")
    df[dec_col] = pd.to_numeric(df[dec_col], errors="coerce")

    target  = SkyCoord(target_ra*u.deg, target_dec*u.deg)
    coords  = SkyCoord(df[ra_col].values*u.deg, df[dec_col].values*u.deg)
    idx     = target.separation(coords).argmin()
    row     = df.iloc[idx]
    z_val   = row[z_col] if z_col else np.nan
    return float(row[ra_col]), float(row[dec_col]), z_val


def find_closest_in_catalog(cat_path, target_ra, target_dec, max_sep=2.0):
    """Return dict with photometry of closest source (≤ max_sep arcsec)."""
    cat = Table.read(cat_path, format="ascii")
    if not {"ALPHA_J2000", "DELTA_J2000"}.issubset(cat.colnames):
        raise ValueError(f"{cat_path}: missing RA/Dec columns")

    target    = SkyCoord(target_ra*u.deg, target_dec*u.deg)
    cat_coord = SkyCoord(cat["ALPHA_J2000"], cat["DELTA_J2000"], unit="deg")
    sep       = target.separation(cat_coord)
    if sep.min() > max_sep*u.arcsec:
        raise ValueError(f"No match within {max_sep} arcsec (closest={sep.min().arcsec:.2f})")
    row = cat[sep.argmin()]
    # return only numeric types or None (float(row[col]) if present else None)
    def get(col):
        return float(row[col]) if col in row.colnames and row[col] is not None else np.nan
    return dict(
        ALPHA_J2000 = get("ALPHA_J2000"),
        DELTA_J2000 = get("DELTA_J2000"),
        MAG_APER    = get("MAG_APER"),
        MAGERR_APER = get("MAGERR_APER"),
        FLUX_APER   = get("FLUX_APER"),
        FLUXERR_APER= get("FLUXERR_APER"),
        X_IMAGE     = get("X_IMAGE"),
        Y_IMAGE     = get("Y_IMAGE"),
    )


def extract_zpd_from_filename(fname):
    """Return (ZP, ZP_err).  If missing, err falls back to FIDUCIAL_ZPD_ERR."""
    z  = re.search(r"_ZPD([0-9p\-]+)", fname)
    ze = re.search(r"_ERR([0-9p\-]+)", fname)
    zpd     = float(z.group(1).replace('p','.'))  if z else np.nan
    zpd_err = float(ze.group(1).replace('p','.')) if ze else FIDUCIAL_ZPD_ERR
    return zpd, zpd_err


def flux_to_magerr_asymmetric(flux, flux_err):
    """Return (+err, -err) in magnitudes from flux ± flux_err."""
    if flux <= 0 or flux_err <= 0 or flux_err >= flux:
        return np.nan, np.nan
    mag     = -2.5 * np.log10(flux)
    mag_plus  = -2.5 * np.log10(flux - flux_err)
    mag_minus = -2.5 * np.log10(flux + flux_err)
    err_plus  = mag_plus - mag
    err_minus = mag - mag_minus
    return err_plus, err_minus


def background_flux_to_magerr_asymmetric(flux, bg_err):
    """Return (+err, -err) in magnitudes due to background noise."""
    if flux <= 0 or bg_err is None or bg_err <= 0 or bg_err >= flux:
        return np.nan, np.nan
    mag     = -2.5 * np.log10(flux)
    mag_plus  = -2.5 * np.log10(flux - bg_err)
    mag_minus = -2.5 * np.log10(flux + bg_err)
    err_plus  = mag_plus - mag
    err_minus = mag - mag_minus
    return err_plus, err_minus


def safe_quad_sum(*errs):
    """Quadrature sum, ignoring NaNs."""
    errs_clean = [e for e in errs if np.isfinite(e)]
    return np.sqrt(np.sum(np.square(errs_clean))) if errs_clean else np.nan


def compute_background_error_photutils(rms_path, x, y, diameter, method="center"):
    """Integrate RMS map inside circular aperture → background noise."""
    try:
        with fits.open(rms_path) as hdul:
            data = hdul[0].data
        r   = diameter/2.0
        ap  = CircularAperture([(x, y)], r=r)
        s2  = aperture_photometry(data**2, ap, method=method)['aperture_sum'][0]
        return np.sqrt(s2)
    except Exception as e:
        print(f"Background error failed ({os.path.basename(rms_path)}): {e}")
        return np.nan

# ----------------------------------------------------------------------
# 2.  MAIN LOOP
# ----------------------------------------------------------------------
records = []

for cat_file in sorted(os.listdir(catalog_folder)):
    if not cat_file.endswith(".cat"):
        continue

    base_match = re.match(r"(J\d{4}[+-]\d{4}(?:\(dup\))?)", cat_file)
    if not base_match:
        print(f"Skipping {cat_file}: cannot extract J‑name")
        continue
    base = base_match.group(1)

    try:
        rough_ra, rough_dec = parse_filename_coords(base)
        ra_prec, dec_prec, z = find_closest_in_ods(ods_file_path, rough_ra, rough_dec)

        cat_path = os.path.join(catalog_folder, cat_file)
        phot = find_closest_in_catalog(cat_path, ra_prec, dec_prec, max_sep=MAX_SEP_ARCSEC)
    except Exception as e:
        print(f"Skipping {cat_file}: {e}")
        continue

    # aperture diameter from filename
    ap_match = re.search(r"_ap([0-9p]+)", cat_file)
    aper_diam = float(ap_match.group(1).replace('p','.')) if ap_match else 1.0

    zpd, zpd_err = extract_zpd_from_filename(cat_file)

    # background RMS
    rms_file = f"{base}_ap{ap_match.group(1)}_background_rms.fits"
    rms_path = os.path.join(background_rms_folder, rms_file)
    bg_err   = compute_background_error_photutils(
                    rms_path, phot["X_IMAGE"], phot["Y_IMAGE"], aper_diam, method="exact") #<-- set to exact ot center 

    # DATEOBS1 -> observation date
    try:
        with fits.open(rms_path) as hd:
            dateobs1 = hd[0].header.get("DATEOBS1", "")
        obs_date = dateobs1.split('T')[0] if "T" in dateobs1 else dateobs1
    except Exception:
        obs_date = ""

    # magnitude‑error budget
    m_err_plus, m_err_minus = flux_to_magerr_asymmetric(
                                phot["FLUX_APER"], phot["FLUXERR_APER"])
    bg_err_plus, bg_err_minus = background_flux_to_magerr_asymmetric(
                                phot["FLUX_APER"], bg_err)

    magerrtot_plus  = safe_quad_sum(zpd_err, bg_err_plus,  m_err_plus)
    magerrtot_minus = safe_quad_sum(zpd_err, bg_err_minus, m_err_minus)

    # SkyCoord pretty strings
    c_prec = SkyCoord(ra_prec*u.deg, dec_prec*u.deg)
    ra_sex = c_prec.ra.to_string(unit=u.hourangle, sep=':', precision=2, pad=True)
    dec_sex= c_prec.dec.to_string(unit=u.deg,       sep=':', precision=2, pad=True, alwayssign=True)

    records.append(dict(
        Filename          = cat_file,
        PreciseRA         = ra_prec,
        PreciseDEC        = dec_prec,
        PreciseRA_sex     = ra_sex,
        PreciseDEC_sex    = dec_sex,
        Redshift          = z,
        ALPHA_J2000       = phot["ALPHA_J2000"],
        DELTA_J2000       = phot["DELTA_J2000"],
        MAG_APER          = phot["MAG_APER"],
        MAGERR_APER       = phot["MAGERR_APER"],
        FLUX_APER         = phot["FLUX_APER"],
        FLUXERR_APER      = phot["FLUXERR_APER"],
        X_IMAGE           = phot["X_IMAGE"],
        Y_IMAGE           = phot["Y_IMAGE"],
        Aperture          = aper_diam,
        ZPD               = zpd,
        ZPD_ERR           = zpd_err,
        Background_ERR    = bg_err,
        MAGERR_FROMFLUXAPER_PLUS   = m_err_plus,
        MAGERR_FROMFLUXAPER_MINUS  = m_err_minus,
        Background_ERR_upper       = bg_err_plus,
        Background_ERR_lower       = bg_err_minus,
        MAGERRTOT_upper            = magerrtot_plus,
        MAGERRTOT_lower            = magerrtot_minus,
        ObservationDate            = obs_date
    ))

# ----------------------------------------------------------------------
# 3.  WRITE TSV
# ----------------------------------------------------------------------
df = pd.DataFrame(records)
df.to_csv(
    output_file, sep="\t", index=False,
    columns=[
        "Filename","PreciseRA","PreciseDEC","PreciseRA_sex","PreciseDEC_sex","Redshift",
        "ALPHA_J2000","DELTA_J2000","MAG_APER","MAGERR_APER",
        "FLUX_APER","FLUXERR_APER","X_IMAGE","Y_IMAGE","Aperture",
        "ZPD","ZPD_ERR","Background_ERR",
        "MAGERR_FROMFLUXAPER_PLUS","MAGERR_FROMFLUXAPER_MINUS",
        "Background_ERR_upper","Background_ERR_lower",
        "MAGERRTOT_upper","MAGERRTOT_lower","ObservationDate"
    ]
)

print(f"✓ Finished – output saved to {output_file}")
print(f"Total runtime: {time.time()-start:.2f} s")


# In[2]:


#!/usr/bin/env python3
# -*- coding: utf‑8 -*-

import os, re, time, math
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table
from astropy.io import fits
from photutils.aperture import CircularAperture, aperture_photometry

# ----------------------------------------------------------------------
# 0.  SETTINGS
# ----------------------------------------------------------------------
catalog_folder        = "../utils/fits_folder/catalogs"
background_rms_folder = "../utils/fits_folder/backgrounds_rms"
ods_file_path         = "../utils/aa61_fan_qso_database.ods"
output_file           = "../utils/photometry_matches.tsv"

MAX_SEP_ARCSEC = 2.0          # match radius of precise RA/DEC from Fan23 with catalog detections

start = time.time()

# ----------------------------------------------------------------------
# 1.  HELPER FUNCTIONS
# ----------------------------------------------------------------------

# reading rough coordinates from .cat filenames
def parse_filename_coords(name: str):
    m = re.search(r"J(\d{2})(\d{2})([+-])(\d{2})(\d{2})", name)
    if not m:
        raise ValueError(f"Cannot parse J‑name coordinates from '{name}'")
    ra_h, ra_m   = int(m.group(1)), int(m.group(2))
    sign         = 1 if m.group(3) == "+" else -1
    dec_d, dec_m = int(m.group(4)), int(m.group(5))
    ra_deg  = (ra_h + ra_m/60) * 15.0
    dec_deg = sign * (dec_d + dec_m/60)
    return ra_deg, dec_deg


# first cone-search on the Fan 2023 database to extract precise coordinates
def find_closest_in_ods(ods_file, target_ra, target_dec):
    df = pd.read_excel(ods_file, engine="odf", header=[0, 1])
    ra_col  = [c for c in df.columns if c[0].lower() == "ra"][0]
    dec_col = [c for c in df.columns if c[0].lower() == "dec"][0]
    z_col   = [c for c in df.columns if c[0].lower() == "redshift"]
    z_col   = z_col[0] if z_col else None

    df[ra_col]  = pd.to_numeric(df[ra_col],  errors="coerce")
    df[dec_col] = pd.to_numeric(df[dec_col], errors="coerce")

    target  = SkyCoord(target_ra*u.deg, target_dec*u.deg)
    coords  = SkyCoord(df[ra_col].values*u.deg, df[dec_col].values*u.deg)
    idx     = target.separation(coords).argmin()
    row     = df.iloc[idx]
    z_val   = row[z_col] if z_col else np.nan
    return float(row[ra_col]), float(row[dec_col]), z_val


# cone-search with precise ra/dec values from Fan 2023 database on the catalog entries
# SET MAXIMUM SEPARATION AT SETTINGS ABOVE, max_sep here is only default value!!!
def find_closest_in_catalog(cat_path, target_ra, target_dec, max_sep=2.0):
    cat = Table.read(cat_path, format="ascii")
    if not {"ALPHA_J2000", "DELTA_J2000"}.issubset(cat.colnames):
        raise ValueError(f"{cat_path}: missing RA/Dec columns")

    target    = SkyCoord(target_ra*u.deg, target_dec*u.deg)
    cat_coord = SkyCoord(cat["ALPHA_J2000"], cat["DELTA_J2000"], unit="deg")
    sep       = target.separation(cat_coord)
    if sep.min() > max_sep*u.arcsec:
        raise ValueError(f"No match within {max_sep} arcsec (closest={sep.min().arcsec:.2f})")
    row = cat[sep.argmin()]
    # return only numeric types or None (float(row[col]) if present else None)
    def get(col):
        return float(row[col]) if col in row.colnames and row[col] is not None else np.nan
    return dict(
        ALPHA_J2000 = get("ALPHA_J2000"),
        DELTA_J2000 = get("DELTA_J2000"),
        MAG_APER    = get("MAG_APER"),
        MAGERR_APER = get("MAGERR_APER"),
        FLUX_APER   = get("FLUX_APER"),
        FLUXERR_APER= get("FLUXERR_APER"),
        X_IMAGE     = get("X_IMAGE"),
        Y_IMAGE     = get("Y_IMAGE"),
    )


# reverse engineered from bash-script output
def extract_zpd_from_filename(fname):
    z  = re.search(r"_ZPD([0-9p\-]+)", fname)
    ze = re.search(r"_ERR([0-9p\-]+)", fname)
    zpd     = float(z.group(1).replace('p','.'))  if z else np.nan
    zpd_err = float(ze.group(1).replace('p','.')) if ze else np.nan
    return zpd, zpd_err


# poisson error (flux) to asymmetric magerr
def flux_to_magerr_asymmetric(flux, flux_err):
    if flux <= 0 or flux_err <= 0 or flux_err >= flux:
        return np.nan, np.nan
    mag     = -2.5 * np.log10(flux)
    mag_plus  = -2.5 * np.log10(flux - flux_err)
    mag_minus = -2.5 * np.log10(flux + flux_err)
    err_plus  = mag_plus - mag
    err_minus = mag - mag_minus
    return err_plus, err_minus


# asymmetric error calculation
def background_flux_to_magerr_asymmetric(flux, bg_err):
    if flux <= 0 or bg_err is None or bg_err <= 0 or bg_err >= flux:
        return np.nan, np.nan
    mag     = -2.5 * np.log10(flux)
    mag_plus  = -2.5 * np.log10(flux - bg_err)
    mag_minus = -2.5 * np.log10(flux + bg_err)
    err_plus  = mag_plus - mag
    err_minus = mag - mag_minus
    return err_plus, err_minus

# safe quadratic summation, where it prints nan for failed calculation instead of stopping (very neat)
def safe_quad_sum(*errs):
    errs_clean = [e for e in errs if np.isfinite(e)]
    return np.sqrt(np.sum(np.square(errs_clean))) if errs_clean else np.nan


# background error estimation by quadrature summing of background_rms map. SET ONLY DEFAULT METHOD HERE!!!
def compute_background_error_photutils(rms_path, x, y, diameter, method="center"):
    try:
        with fits.open(rms_path) as hdul:
            data = hdul[0].data
        r   = diameter/2.0
        ap  = CircularAperture([(x, y)], r=r)
        s2  = aperture_photometry(data**2, ap, method=method)['aperture_sum'][0]
        return np.sqrt(s2)
    except Exception as e:
        print(f"Background error failed ({os.path.basename(rms_path)}): {e}")
        return np.nan

# ----------------------------------------------------------------------
# 2.  MAIN LOOP
# ----------------------------------------------------------------------
records = []

for cat_file in sorted(os.listdir(catalog_folder)):
    if not cat_file.endswith(".cat"):
        continue

    base_match = re.match(r"(J\d{4}[+-]\d{4}(?:\(dup\))?)", cat_file)
    if not base_match:
        print(f"Skipping {cat_file}: cannot extract J‑name")
        continue
    base = base_match.group(1)

    try:
        rough_ra, rough_dec = parse_filename_coords(base)
        ra_prec, dec_prec, z = find_closest_in_ods(ods_file_path, rough_ra, rough_dec)

        cat_path = os.path.join(catalog_folder, cat_file)
        phot = find_closest_in_catalog(cat_path, ra_prec, dec_prec, max_sep=MAX_SEP_ARCSEC)
    except Exception as e:
        print(f"Skipping {cat_file}: {e}")
        continue

    # grab aperture diameter from filename
    ap_match = re.search(r"_ap([0-9p]+)", cat_file)
    aper_diam = float(ap_match.group(1).replace('p','.')) if ap_match else 1.0

    zpd, zpd_err = extract_zpd_from_filename(cat_file)

    # background RMS calculation (SET IN-USE method HERE, ABOVE IS DEFAULT METHOD!!!)
    rms_file = f"{base}_ap{ap_match.group(1)}_background_rms.fits"
    rms_path = os.path.join(background_rms_folder, rms_file)
    bg_err   = compute_background_error_photutils(
                    rms_path, phot["X_IMAGE"], phot["Y_IMAGE"], aper_diam, method="exact") 

    # read and truncate DATEOBS1 to give exposure date
    try:
        with fits.open(rms_path) as hd:
            dateobs1 = hd[0].header.get("DATEOBS1", "")
        obs_date = dateobs1.split('T')[0] if "T" in dateobs1 else dateobs1
    except Exception:
        obs_date = ""

    # magnitude‑error budget
    m_err_plus, m_err_minus = flux_to_magerr_asymmetric(
                                phot["FLUX_APER"], phot["FLUXERR_APER"])
    bg_err_plus, bg_err_minus = background_flux_to_magerr_asymmetric(
                                phot["FLUX_APER"], bg_err)

    magerrtot_plus  = safe_quad_sum(zpd_err, bg_err_plus,  m_err_plus)
    magerrtot_minus = safe_quad_sum(zpd_err, bg_err_minus, m_err_minus)

    # Calculate SNR
    snr = phot["FLUX_APER"] / bg_err if bg_err > 0 else np.nan

    # SkyCoord formatted clean strings
    c_prec = SkyCoord(ra_prec*u.deg, dec_prec*u.deg)
    ra_sex = c_prec.ra.to_string(unit=u.hourangle, sep=':', precision=2, pad=True)
    dec_sex= c_prec.dec.to_string(unit=u.deg,       sep=':', precision=2, pad=True, alwayssign=True)

    records.append(dict(
        Filename          = cat_file,
        PreciseRA         = ra_prec,
        PreciseDEC        = dec_prec,
        PreciseRA_sex     = ra_sex,
        PreciseDEC_sex    = dec_sex,
        Redshift          = z,
        ALPHA_J2000       = phot["ALPHA_J2000"],
        DELTA_J2000       = phot["DELTA_J2000"],
        MAG_APER          = phot["MAG_APER"],
        MAGERR_APER       = phot["MAGERR_APER"],
        FLUX_APER         = phot["FLUX_APER"],
        FLUXERR_APER      = phot["FLUXERR_APER"],
        X_IMAGE           = phot["X_IMAGE"],
        Y_IMAGE           = phot["Y_IMAGE"],
        Aperture          = aper_diam,
        ZPD               = zpd,
        ZPD_ERR           = zpd_err,
        Background_ERR    = bg_err,
        MAGERR_FROMFLUXAPER_PLUS   = m_err_plus,
        MAGERR_FROMFLUXAPER_MINUS  = m_err_minus,
        Background_ERR_upper       = bg_err_plus,
        Background_ERR_lower       = bg_err_minus,
        MAGERRTOT_upper            = magerrtot_plus,
        MAGERRTOT_lower            = magerrtot_minus,
        ObservationDate            = obs_date,
        SNR                       = snr 
    ))

# ----------------------------------------------------------------------
# 3.  WRITE TSV
# ----------------------------------------------------------------------
df = pd.DataFrame(records)
df.to_csv(
    output_file, sep="\t", index=False,
    columns=[
        "Filename","PreciseRA","PreciseDEC","PreciseRA_sex","PreciseDEC_sex","Redshift",
        "ALPHA_J2000","DELTA_J2000","MAG_APER","MAGERR_APER",
        "FLUX_APER","FLUXERR_APER","X_IMAGE","Y_IMAGE","Aperture",
        "ZPD","ZPD_ERR","Background_ERR",
        "MAGERR_FROMFLUXAPER_PLUS","MAGERR_FROMFLUXAPER_MINUS",
        "Background_ERR_upper","Background_ERR_lower",
        "MAGERRTOT_upper","MAGERRTOT_lower","ObservationDate",
        "SNR"               
    ]
)

print(f"✓ Finished – output saved to {output_file}")
print(f"Total runtime: {time.time()-start:.2f} s")


# In[9]:


# crackcode, last minute

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, time, math
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table
from astropy.io import fits
from photutils.aperture import CircularAperture, aperture_photometry

# ----------------------------------------------------------------------
# 0.  SETTINGS
# ----------------------------------------------------------------------
catalog_folder        = "../utils/fits_folder/catalogs"
background_rms_folder = "../utils/fits_folder/backgrounds_rms"
ods_file_path         = "../utils/fits_folder/aa61_fan_qso_database.ods"
output_file           = "../utils/fits_folder/photometry_matches.tsv"

MAX_SEP_ARCSEC = 2.0          # match radius of precise RA/DEC from Fan23 with catalog detections

start = time.time()

# ----------------------------------------------------------------------
# 1.  HELPER FUNCTIONS
# ----------------------------------------------------------------------

def parse_filename_coords(name: str):
    """Parse truncated J-name and return rough RA/Dec in degrees."""
    m = re.search(r"J(\d{2})(\d{2})([+-])(\d{2})(\d{2})", name)
    if not m:
        raise ValueError(f"Cannot parse J-name coordinates from '{name}'")
    ra_h, ra_m   = int(m.group(1)), int(m.group(2))
    sign         = 1 if m.group(3) == "+" else -1
    dec_d, dec_m = int(m.group(4)), int(m.group(5))
    ra_deg  = (ra_h + ra_m/60) * 15.0
    dec_deg = sign * (dec_d + dec_m/60)
    return ra_deg, dec_deg


def find_closest_in_ods(ods_file, target_ra, target_dec):
    """
    Cone-search the Fan+2023 ODS catalogue. 
    Returns precise RA, Dec, redshift, **and BH mass (10^8 M_sun)**.
    """
    df = pd.read_excel(ods_file, engine="odf", header=[0, 1])

    ra_col  = [c for c in df.columns if c[0].lower() == "ra"][0]
    dec_col = [c for c in df.columns if c[0].lower() == "dec"][0]
    z_col   = [c for c in df.columns if c[0].lower() == "redshift"]
    z_col   = z_col[0] if z_col else None

    # Debug BHmass lookup with safer match
    bh_cols = [c for c in df.columns if c[0].strip().lower() == "bhmass"]
    bh_col = bh_cols[0] if bh_cols else None


    target  = SkyCoord(target_ra*u.deg, target_dec*u.deg)
    coords  = SkyCoord(df[ra_col].values*u.deg, df[dec_col].values*u.deg)
    idx     = target.separation(coords).argmin()
    row     = df.iloc[idx]
    z_val   = row[z_col] if z_col else np.nan
    bh_mass = row[bh_col] if bh_col else np.nan

    return float(row[ra_col]), float(row[dec_col]), z_val, bh_mass


def find_closest_in_catalog(cat_path, target_ra, target_dec, max_sep=2.0):
    """Cone-search inside one SExtractor catalogue."""
    cat = Table.read(cat_path, format="ascii")
    if not {"ALPHA_J2000", "DELTA_J2000"}.issubset(cat.colnames):
        raise ValueError(f"{cat_path}: missing RA/Dec columns")

    target    = SkyCoord(target_ra*u.deg, target_dec*u.deg)
    cat_coord = SkyCoord(cat["ALPHA_J2000"], cat["DELTA_J2000"], unit="deg")
    sep       = target.separation(cat_coord)
    if sep.min() > max_sep*u.arcsec:
        raise ValueError(f"No match within {max_sep} arcsec (closest={sep.min().arcsec:.2f})")
    row = cat[sep.argmin()]
    def get(col):
        return float(row[col]) if col in row.colnames and row[col] is not None else np.nan
    return dict(
        ALPHA_J2000 = get("ALPHA_J2000"),
        DELTA_J2000 = get("DELTA_J2000"),
        MAG_APER    = get("MAG_APER"),
        MAGERR_APER = get("MAGERR_APER"),
        FLUX_APER   = get("FLUX_APER"),
        FLUXERR_APER= get("FLUXERR_APER"),
        X_IMAGE     = get("X_IMAGE"),
        Y_IMAGE     = get("Y_IMAGE"),
    )

def extract_zpd_from_filename(fname):
    z  = re.search(r"_ZPD([0-9p\-]+)", fname)
    ze = re.search(r"_ERR([0-9p\-]+)", fname)
    zpd     = float(z.group(1).replace('p','.'))  if z else np.nan
    zpd_err = float(ze.group(1).replace('p','.')) if ze else np.nan
    return zpd, zpd_err

# --- (the rest of helper functions unchanged) ---
def flux_to_magerr_asymmetric(flux, flux_err):
    if flux <= 0 or flux_err <= 0 or flux_err >= flux:
        return np.nan, np.nan
    mag     = -2.5 * np.log10(flux)
    mag_plus  = -2.5 * np.log10(flux - flux_err)
    mag_minus = -2.5 * np.log10(flux + flux_err)
    err_plus  = mag_plus - mag
    err_minus = mag - mag_minus
    return err_plus, err_minus

def background_flux_to_magerr_asymmetric(flux, bg_err):
    if flux <= 0 or bg_err is None or bg_err <= 0 or bg_err >= flux:
        return np.nan, np.nan
    mag     = -2.5 * np.log10(flux)
    mag_plus  = -2.5 * np.log10(flux - bg_err)
    mag_minus = -2.5 * np.log10(flux + bg_err)
    err_plus  = mag_plus - mag
    err_minus = mag - mag_minus
    return err_plus, err_minus

def safe_quad_sum(*errs):
    errs_clean = [e for e in errs if np.isfinite(e)]
    return np.sqrt(np.sum(np.square(errs_clean))) if errs_clean else np.nan

def compute_background_error_photutils(rms_path, x, y, diameter, method="center"):
    try:
        with fits.open(rms_path) as hdul:
            data = hdul[0].data
        r   = diameter/2.0
        ap  = CircularAperture([(x, y)], r=r)
        s2  = aperture_photometry(data**2, ap, method=method)['aperture_sum'][0]
        return np.sqrt(s2)
    except Exception as e:
        print(f"Background error failed ({os.path.basename(rms_path)}): {e}")
        return np.nan

# ----------------------------------------------------------------------
# 2.  MAIN LOOP
# ----------------------------------------------------------------------
records = []

for cat_file in sorted(os.listdir(catalog_folder)):
    if not cat_file.endswith(".cat"):
        continue

    base_match = re.match(r"(J\d{4}[+-]\d{4}(?:\(dup\))?)", cat_file)
    if not base_match:
        print(f"Skipping {cat_file}: cannot extract J-name")
        continue
    base = base_match.group(1)

    try:
        rough_ra, rough_dec = parse_filename_coords(base)
        ra_prec, dec_prec, z, bh_mass = find_closest_in_ods(
            ods_file_path, rough_ra, rough_dec)

        cat_path = os.path.join(catalog_folder, cat_file)
        phot = find_closest_in_catalog(cat_path, ra_prec, dec_prec,
                                       max_sep=MAX_SEP_ARCSEC)
    except Exception as e:
        print(f"Skipping {cat_file}: {e}")
        continue

    # grab aperture diameter from filename
    ap_match = re.search(r"_ap([0-9p]+)", cat_file)
    aper_diam = float(ap_match.group(1).replace('p','.')) if ap_match else 1.0

    zpd, zpd_err = extract_zpd_from_filename(cat_file)

    # background RMS calculation
    rms_file = f"{base}_ap{ap_match.group(1)}_background_rms.fits"
    rms_path = os.path.join(background_rms_folder, rms_file)
    bg_err   = compute_background_error_photutils(
                    rms_path, phot["X_IMAGE"], phot["Y_IMAGE"],
                    aper_diam, method="exact") 

    # read and truncate DATEOBS1 to give exposure date
    try:
        with fits.open(rms_path) as hd:
            dateobs1 = hd[0].header.get("DATEOBS1", "")
        obs_date = dateobs1.split('T')[0] if "T" in dateobs1 else dateobs1
    except Exception:
        obs_date = ""

    # magnitude-error budget
    m_err_plus, m_err_minus = flux_to_magerr_asymmetric(
                                phot["FLUX_APER"], phot["FLUXERR_APER"])
    bg_err_plus, bg_err_minus = background_flux_to_magerr_asymmetric(
                                phot["FLUX_APER"], bg_err)

    magerrtot_plus  = safe_quad_sum(zpd_err, bg_err_plus,  m_err_plus)
    magerrtot_minus = safe_quad_sum(zpd_err, bg_err_minus, m_err_minus)

    # Calculate SNR
    snr = phot["FLUX_APER"] / bg_err if bg_err > 0 else np.nan

    # SkyCoord formatted clean strings
    c_prec = SkyCoord(ra_prec*u.deg, dec_prec*u.deg)
    ra_sex = c_prec.ra.to_string(unit=u.hourangle, sep=':', precision=2, pad=True)
    dec_sex= c_prec.dec.to_string(unit=u.deg,       sep=':', precision=2, pad=True,
                                  alwayssign=True)

    records.append(dict(
        Filename          = cat_file,
        PreciseRA         = ra_prec,
        PreciseDEC        = dec_prec,
        PreciseRA_sex     = ra_sex,
        PreciseDEC_sex    = dec_sex,
        Redshift          = z,
        BHmass_1e8Msun    = bh_mass,             
        ALPHA_J2000       = phot["ALPHA_J2000"],
        DELTA_J2000       = phot["DELTA_J2000"],
        MAG_APER          = phot["MAG_APER"],
        MAGERR_APER       = phot["MAGERR_APER"],
        FLUX_APER         = phot["FLUX_APER"],
        FLUXERR_APER      = phot["FLUXERR_APER"],
        X_IMAGE           = phot["X_IMAGE"],
        Y_IMAGE           = phot["Y_IMAGE"],
        Aperture          = aper_diam,
        ZPD               = zpd,
        ZPD_ERR           = zpd_err,
        Background_ERR    = bg_err,
        MAGERR_FROMFLUXAPER_PLUS   = m_err_plus,
        MAGERR_FROMFLUXAPER_MINUS  = m_err_minus,
        Background_ERR_upper       = bg_err_plus,
        Background_ERR_lower       = bg_err_minus,
        MAGERRTOT_upper            = magerrtot_plus,
        MAGERRTOT_lower            = magerrtot_minus,
        ObservationDate            = obs_date,
        SNR                        = snr 
    ))

# ----------------------------------------------------------------------
# 3.  WRITE TSV
# ----------------------------------------------------------------------
df = pd.DataFrame(records)
df.to_csv(
    output_file, sep="\t", index=False,
    columns=[
        "Filename","PreciseRA","PreciseDEC","PreciseRA_sex","PreciseDEC_sex","Redshift",
        "BHmass_1e8Msun",                     
        "ALPHA_J2000","DELTA_J2000","MAG_APER","MAGERR_APER",
        "FLUX_APER","FLUXERR_APER","X_IMAGE","Y_IMAGE","Aperture",
        "ZPD","ZPD_ERR","Background_ERR",
        "MAGERR_FROMFLUXAPER_PLUS","MAGERR_FROMFLUXAPER_MINUS",
        "Background_ERR_upper","Background_ERR_lower",
        "MAGERRTOT_upper","MAGERRTOT_lower","ObservationDate",
        "SNR"               
    ]
)

print(f"✓ Finished – output saved to {output_file}")
print(f"Total runtime: {time.time() - start:.2f} s")


# In[ ]:





# In[ ]:




