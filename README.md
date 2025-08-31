# 📦 Variability of VHzQs

> Photometry and quantitative analysis of J-Band magnitude variability over z-scaled time intervals. Later upload version will include further analysis of the observed (non-)variability.
>
>The code and subsequent master's thesis were compiled at the Frontier research group at ITP Heidelberg.

![License](https://img.shields.io/github/license/Jonathan-Konrad/jonathan-konrad.github.io?style=flat-square)
![Build Status](https://img.shields.io/github/actions/workflow/status/Jonathan-Konrad/jonathan-konrad.github.io/main.yml)
![Version](https://img.shields.io/github/v/release/Jonathan-Konrad/jonathan-konrad.github.io?include_prereleases)

---

## 🚀 Table of Contents

- [About](#about)
- [Features](#features)
- [Contact](#contact)

---

## 🧠 About

This is a collection of python code used to reduce and analyze scientific raw data from the GROND observatory to measure explicitly J-Band magnitudes in high z (z>5.3) quasars. Other utilities are also uploaded here, of wich I do not claim ownership. 
These include modified configuration and parameter files (SExtractor), a 3x3 gaussian convolution map and raw public data from the GROND observatory, accessed through the ESO archive.
J-Band exposures were reduced with theli v3, a reduction pipeline for several observatories by Mischa Schirmer. 

---

## ✨ Features

Iterative SExtractor bash-shell (variables: aperture, 3x3 convolution matrix), also grabs zero-points from .fits header

Collection of three major codes, interlocking to order, extract and calculate errors on J-Band magnitudes, J-Band magnitudes themselves and variability from a compiled list of archival J-Band magnitudes. Also z-scales time intervals to the quasar restframe.

---
##   Contact

jojokonrad@googlemail.com
