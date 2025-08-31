#!/bin/bash

# Master folder with only FITS files
master_dir="../utils/sextractor"

# SExtractor configuration files
sex_config="test.sex"
param_file="test.param"
conv_file="test_halfnonorm.conv"

# nominal factor of 2 to capture most of the light
aperture_factor=2.0


fwhm_list=(3.5)  # <-- either scalar (list with one entry) or full fledged list


# Output directories
catalog_dir="${master_dir}/catalogs"
checkimg_dir="${master_dir}/check_images"
background_rms_dir="${master_dir}/backgrounds_rms"
mkdir -p "$catalog_dir" "$checkimg_dir" "$background_rms_dir"

# Loop over all FWHM values/ useless by now, I am only using one aperture informed by theli
for fwhm in "${fwhm_list[@]}"; do

  # Compute aperture diameter (FWHM × aperture_factor)
  aper=$(python3 -c "print(round($fwhm * $aperture_factor, 2))")
  aper_tag=$(echo "$aper" | sed 's/\./p/')

  # Loop through all .fits files in the master directory
  for fitsfile in "$master_dir"/*.fits; do
    base=$(basename "$fitsfile" .fits)

    # Extract ZPD and ZPD_ERR from FITS header using Python (all on one line, no indents)
    read zpd zpd_err < <(python3 -c "from astropy.io import fits; h = fits.getheader('$fitsfile'); print(f'{h.get(\"ZPD\", 0.0)} {h.get(\"ZPD_ERR\", 0.0)}')")

    # Warn if ZPD was missing or zero
    if [[ "$zpd" == "0.0" ]]; then
      echo "⚠️ Warning: ZPD missing or zero in $fitsfile"
    fi



    # Format ZPD and ZPD_ERR for filenames (4 decimal places, replace '.' with 'p')
    zpd_fmt=$(printf "%.4f" "$zpd" | sed 's/\./p/')
    zpd_err_fmt=$(printf "%.4f" "$zpd_err" | sed 's/\./p/')

    # Output filenames
    catalog="${catalog_dir}/${base}_ap${aper_tag}_ZPD${zpd_fmt}_ERR${zpd_err_fmt}.cat"
    checkimg="${checkimg_dir}/${base}_ap${aper_tag}_aperture.fits"
    background_rms_img="${background_rms_dir}/${base}_ap${aper_tag}_background_rms.fits"

    # Running sextractor
    source-extractor "$fitsfile" -c "$sex_config" \
      -PARAMETERS_NAME "$param_file" \
      -CATALOG_NAME "$catalog" \
      -CATALOG_TYPE ASCII_HEAD \
      -CHECKIMAGE_TYPE APERTURES,BACKGROUND_RMS \
      -CHECKIMAGE_NAME "$checkimg","$background_rms_img" \
      -FILTER_NAME "$conv_file" \
      -PHOT_APERTURES "$aper" \
      -MAG_ZEROPOINT "$zpd"

    echo "Processed $fitsfile with aperture $aper pixels (FWHM × $aperture_factor)"
    echo "  → Catalog:          $catalog"
    echo "  → CheckImg:         $checkimg"
    echo "  → Background RMS:   $background_rms_img"

  done
done
