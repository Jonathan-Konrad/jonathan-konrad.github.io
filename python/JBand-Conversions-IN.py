#!/usr/bin/env python
# coding: utf-8

# In[2]:


import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from numpy import nan
from astropy.io import fits
from scipy.integrate import simps

angstrom = 'Å'
alpha = 'α'

plt.xlabel('wavelength ({})'.format(angstrom))
plt.ylabel('flux density (erg/s¹/cm²/{}¹)'.format(angstrom))

data = fits.open("/utils/optical_nir_qso_sed_001.fits")[1].data
 
data2=[]                  # testdata to check normalizations (flat profile from 5000A-15000A)
for j in range(10001):
    data2.append(tuple((1000+j, 1*1e-12)))
    
z=5.84
lyman_alpha=1215.67
lyman_alpha_z=lyman_alpha*(1+z)

x=[]
flux=[]
for a in data:
    x.append(a[0]*(1+z))
    flux.append(a[1])

plt.plot(x, np.array(flux), 'k-', linewidth=.6, label='composite QSO spectrum Glikman(2006)')



GRONDcurve = np.loadtxt('/utils/filtercurves/GROND_filtercurves_trunc.txt', skiprows=1)
headers = GRONDcurve[0,:]
xG = GRONDcurve[:,0]
GFilter = GRONDcurve[:,5]
# plt.plot(10*xG, GFilter*1e-12, 'c-', linewidth=.6, label='GROND filtercurve')

VISTAcurve = np.loadtxt('/utils/filtercurves/VISTA_filtercurves/Paranal_VISTA.J_trunc.dat')
xV = VISTAcurve[:,0]
VFilter = VISTAcurve[:,1]
# plt.plot(10*xV, VFilter*1e-14, 'b-', linewidth=.6, label='VISTA filtercurve')        # -14 comes from -12 * -2 because VFilter was normalized to 100, not 1

UKIDSScurve = np.loadtxt('/utils/filtercurves/UKIDDS_filtercurve_J_trunc.dat')
xU = UKIDSScurve[:,0]
UFilter = UKIDSScurve[:,1]
# plt.plot(xU, UFilter*1e-12, 'm-', linewidth=.6, label='UKIDSS filtercurve')

FourStarscurve = np.loadtxt('/utils/filtercurves/LCO_FourStar.J.dat')
xF = FourStarscurve[:,0]
FFilter = FourStarscurve[:,1]
# plt.plot(xF, FFilter*1e-12, 'k-', linewidth=.6, label='FourStars filtercurve')

GNIRScurve = np.loadtxt('/utils/filtercurves/Gemini_GNIRS.J_trunc.dat')
xGN = GNIRScurve[:,0]
GNFilter = GNIRScurve[:,1]

TwoMASScurve = np.loadtxt('/utils/filtercurves/2MASS_2MASS.J_trunc.dat')
xT = TwoMASScurve[:,0]
TFilter = TwoMASScurve[:,1]



GFilter_int  = np.interp(x, 10*xG, GFilter)           #Interpolate to common wavelength axis
GFilterSpec  = GFilter_int * flux                     #Calculate throughput
GFlux      = simps(GFilterSpec, x)                    #Integrate over wavelength
GFFlux = simps(GFilter_int, x)
Gmag = GFlux/GFFlux

#print('Total GFlux is {0:8.3e} erg/s¹/cm² and the normalization: {2}; the corresponding Gmag is {1:8.3e}'.format(GFlux, Gmag, GFFlux))



VFilter_int = np.interp(x, xV, VFilter)        #Interpolate to common wavelength axis
VFilterSpec = VFilter_int * flux                      #Calculate throughput
VFlux = simps(VFilterSpec, x)                         #Integrate over wavelength
VFFlux = simps(VFilter_int, x)                    #Integrate over the filter (for normalization)
Vmag = VFlux/VFFlux
conv_fac_GV = Gmag/Vmag



UFilter_int = np.interp(x, xU, UFilter)
UFilterSpec = UFilter_int * flux
UFlux = simps(UFilterSpec, x)
UFFlux = simps(UFilter_int, x)
Umag = UFlux/UFFlux
conv_fac_GU = Gmag/Umag



FFilter_int = np.interp(x, xF, FFilter)
FFilterSpec = FFilter_int* flux
FFlux = simps(FFilterSpec, x)
FFFlux = simps(FFilter_int, x)
Fmag = FFlux/FFFlux
conv_fac_GF = Gmag/Fmag



GNFilter_int = np.interp(x, xGN, GNFilter)
GNFilterSpec = GNFilter_int* flux
GNFlux = simps(GNFilterSpec, x)
GNFFlux = simps(GNFilter_int, x)
GNmag = GNFlux/GNFFlux
conv_fac_GGN = Gmag/GNmag



TFilter_int = np.interp(x, xT, TFilter)
TFilterSpec = TFilter_int* flux
TFlux = simps(TFilterSpec, x)
TFFlux = simps(TFilter_int, x)
Tmag = TFlux/TFFlux
conv_fac_GT = Gmag/Tmag

plt.axvline(x=lyman_alpha_z, ymin=0, ymax=1, c='k', linewidth=.3, label='Lyman-{}'.format(alpha))

plt.plot(x, GFilter_int/GFFlux*1e-9, 'c-', linewidth=.5, label='GROND filtercurve')
#plt.plot(x, VFilter_int/VFFlux*1e-9, 'b-', linewidth=.5, label='VISTA filtercurve')
#plt.plot(x, UFilter_int/UFFlux*1e-9, 'm-', linewidth=.5, label='UKIDSS filtercurve')
plt.plot(x, FFilter_int/FFFlux*1e-9, 'g-', linewidth=.5, label='FourStars filtercurve')
#plt.plot(x, GNFilter_int/GNFFlux*1e-9, 'r-', linewidth=.5, label='GNIRS filtercurve')
#plt.plot(x, TFilter_int/TFFlux*1e-9, 'k-', linewidth=.5, label='2MASS filtercurve')


#plt.plot(10*xG, GFilter*1e-12, 'c-', linewidth=.6, label='GROND filtercurve')
#plt.plot(10*xV, VFilter*1e-14, 'b-', linewidth=.6, label='VISTA filtercurve')
#plt.plot(xU, UFilter*1e-12, 'm-', linewidth=.6, label='UKIDSS filtercurve')
#plt.plot(xF, FFilter*1e-12, 'k-', linewidth=.6, label='FourStars filtercurve')



#for a, b, c in zip(x, GFilterSpec, UFilterSpec):
#    print(a, b, c)
    
    
plt.plot(x, GFilterSpec, 'c-', linewidth=.3, label='throughput in GROND')
#plt.plot(x, VFilterSpec, 'b-', linewidth=.3, label='throughput in VISTA')
#plt.plot(x, UFilterSpec, 'm-', linewidth=.3, label='throughput in UKIDSS')
plt.plot(x, FFilterSpec, 'g-', linewidth=.3, label='throughput in FourStars')
#plt.plot(x, GNFilterSpec, 'r-', linewidth=.3, label='throughput in GNIRS')
#plt.plot(x, TFilterSpec, 'k-', linewidth=.3, label='throughput in 2MASS')

#print('Conversion factor between GROND/ VISTA is {}'.format(conv_fac_GV))
#print('Conversion factor between GROND/ UKIDSS is {}'.format(conv_fac_GU))




'''
data2 = np.loadtxt('/home/jk/Desktop/Colina_Quasar_z=0_ETC-Glikman+Hernan-Caballero.txt')

x2=[]
flux2=[]

for b in data2:
    x2.append(b[0]*1e4)                                                        # redshifting knocks the QSO far into the infrared
    c = b[1]*3*1e-5/(b[0]*1e4)**2
    d = b[1]
    flux2.append(c)                                                            # chose carefully and mind the units
    
plt.plot(x2, flux2, 'k-', linewidth=.6, label='Colina Quasar')
'''




plt.xlim(8000, 16000)
plt.ylim(0, 8*1e-13)

plt.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1,1), ncol=1, fancybox=True, shadow=True)
plt.show()


# In[2]:


import math
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.integrate import simps
import time

start = time.time()

# Helper function: load 2-column text filter files
def load_2col_filter(path):
    arr = np.loadtxt(path)
    return arr[:, 0], arr[:, 1]

# Load QSO spectrum from FITS file
data = fits.open("/utils/optical_nir_qso_sed_001.fits")[1].data

# Load filter curves from old exact paths
xMKO, MKOFilter = load_2col_filter('/utils/filtercurves/MKO_NSFCam.J.dat')
xMMT, MMTFilter = load_2col_filter('/utils/filtercurves/MMT_SWIRC.J.dat')
xV, VFilter = load_2col_filter('/utils/filtercurves/Paranal_VISTA.J.dat')
xSubaru, SubaruFilter = load_2col_filter('/utils/filtercurves/Subaru_MOIRCS.J.dat')
xUKIRT, UKIRTFilter = load_2col_filter('/utils/filtercurves/UKIRT_WFCAM.J.dat')
xFour, FourFilter = load_2col_filter('/utils/filtercurves/LCO_FourStar.J_trunc.dat')
xSOFI, SOFIFilter = load_2col_filter('/utils/filtercurves/LaSilla_SOFI.J.dat')

# Load GROND filter from multi-column file, column 0 = wavelength, column 5 = J-band filter
GRONDcurve = np.loadtxt('/utils/filtercurves/GROND_filtercurves_trunc.txt', skiprows=1)
xG = GRONDcurve[:, 0]
GFilter = GRONDcurve[:, 5]

# Initialize arrays to hold results
zlist = []
magnitude_diffGMKO = []
magnitude_diffGMMT = []
magnitude_diffGV = []
magnitude_diffGSubaru = []
magnitude_diffGUKIRT = []
magnitude_diffGFour = []
magnitude_diffGSOFI = []

# Compute magnitude differences (X → GROND)
for i in range(831):
    z = 5.7 + 0.01 * i
    zlist.append(round(z, 2))

    # Redshifted wavelengths: data[0] is wavelength, data[1] is flux
    x = [a[0] * (1 + z) for a in data]
    flux = [a[1] for a in data]

    # Interpolate GROND filter to redshifted wavelengths (note factor 10 applied on old xG)
    GFilter_int = np.interp(x, 10 * xG, GFilter)
    GFlux = simps(np.array(GFilter_int) * np.array(flux), x)
    GFFlux = simps(GFilter_int, x)
    Gmag = GFlux / GFFlux

    # MKO magnitude
    MKOFilter_int = np.interp(x, xMKO, MKOFilter)
    MKOmag = simps(MKOFilter_int * flux, x) / simps(MKOFilter_int, x)
    magnitude_diffGMKO.append(-2.5 * math.log10(Gmag/MKOmag))

    # MMT magnitude
    MMTFilter_int = np.interp(x, xMMT, MMTFilter)
    MMTmag = simps(MMTFilter_int * flux, x) / simps(MMTFilter_int, x)
    magnitude_diffGMMT.append(-2.5 * math.log10(Gmag / MMTmag))

    # VISTA magnitude
    VFilter_int = np.interp(x, xV, VFilter)
    Vmag = simps(VFilter_int * flux, x) / simps(VFilter_int, x)
    magnitude_diffGV.append(-2.5 * math.log10(Gmag / Vmag))

    # Subaru magnitude
    SubaruFilter_int = np.interp(x, xSubaru, SubaruFilter)
    Subarumag = simps(SubaruFilter_int * flux, x) / simps(SubaruFilter_int, x)
    magnitude_diffGSubaru.append(-2.5 * math.log10(Gmag / Subarumag))

    # UKIRT magnitude
    UKIRTFilter_int = np.interp(x, xUKIRT, UKIRTFilter)
    UKIRTmag = simps(UKIRTFilter_int * flux, x) / simps(UKIRTFilter_int, x)
    magnitude_diffGUKIRT.append(-2.5 * math.log10(Gmag / UKIRTmag))

    # FourStar magnitude
    FourFilter_int = np.interp(x, xFour, FourFilter)
    Fourmag = simps(FourFilter_int * flux, x) / simps(FourFilter_int, x)
    magnitude_diffGFour.append(-2.5 * math.log10(Gmag / Fourmag))

    # SOFI magnitude
    SOFIFilter_int = np.interp(x, xSOFI, SOFIFilter)
    SOFImag = simps(SOFIFilter_int * flux, x) / simps(SOFIFilter_int, x)
    magnitude_diffGSOFI.append(-2.5 * math.log10(Gmag / SOFImag))

# Plot results
plt.figure(figsize=(8, 5))
plt.xlabel('Redshift')
plt.ylabel(r'$\Delta m$ (Instrument → GROND)', fontsize=12)

plt.plot(zlist, magnitude_diffGMKO, color='orange', linestyle='-', linewidth=0.6, label='NIRI → GROND')
plt.plot(zlist, magnitude_diffGMMT, color='blue', linestyle='-', linewidth=0.6, label='SWIRC → GROND')
plt.plot(zlist, magnitude_diffGV, color='cyan', linestyle='-', linewidth=0.6, label='VIRCAM → GROND')
plt.plot(zlist, magnitude_diffGSubaru, color='red', linestyle='-', linewidth=0.6, label='MOIRCS → GROND')
plt.plot(zlist, magnitude_diffGUKIRT, color='purple', linestyle='-', linewidth=0.6, label='WFCAM → GROND')
plt.plot(zlist, magnitude_diffGFour, color='olive', linestyle='-', linewidth=0.6, label='FourStar → GROND')
plt.plot(zlist, magnitude_diffGSOFI, color='green', linestyle='-', linewidth=0.6, label='SOFI → GROND')

plt.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1, 1),
           fancybox=True, shadow=True)
plt.tick_params(axis='both', direction='in', which='both', top=True, right=True)
plt.tight_layout()
plt.show()

# Save results to file
out_file = '/utils/Instrument_to_GROND.txt'
with open(out_file, 'w') as f_out:
    f_out.write("# Redshift  MKO->GROND  MMT->GROND  VISTA->GROND  "
                "SUBARU->GROND  UKIRT->GROND  FourStar->GROND  SOFI->GROND\n")
    for i in range(len(zlist)):
        f_out.write(f"{zlist[i]:.2f}  {magnitude_diffGMKO[i]:.6f}  {magnitude_diffGMMT[i]:.6f}  "
                    f"{magnitude_diffGV[i]:.6f}  {magnitude_diffGSubaru[i]:.6f}  "
                    f"{magnitude_diffGUKIRT[i]:.6f}  {magnitude_diffGFour[i]:.6f}  "
                    f"{magnitude_diffGSOFI[i]:.6f}\n")

end = time.time()
print(f"Total computation time: {end - start:.2f} seconds")


# In[4]:


import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# Set redshift value
z = 5.3

def load_2col_filter(path):
    arr = np.loadtxt(path)
    return arr[:,0], arr[:,1]

# Load filters
xMKO, MKOFilter = load_2col_filter('/utils/filtercurves/MKO_NSFCam.J.dat')
xMMT, MMTFilter = load_2col_filter('/utils/filtercurves/MMT_SWIRC.J.dat')
xV, VFilter   = load_2col_filter('/utils/filtercurves/Paranal_VISTA.J.dat')
xSubaru, SubaruFilter = load_2col_filter('/utils/filtercurves/Subaru_MOIRCS.J.dat')
xUKIRT, UKIRTFilter   = load_2col_filter('/utils/filtercurves/UKIRT_WFCAM.J.dat')
xFour, FourFilter     = load_2col_filter('/utils/filtercurves/LCO_FourStar.J_trunc.dat')
xSOFI, SOFIFilter     = load_2col_filter('/utils/filtercurves/LaSilla_SOFI.J.dat')

GRONDcurve = np.loadtxt('/utils/filtercurves/GROND_filtercurves_trunc.txt', skiprows=1)
xG = GRONDcurve[:,0]
GFilter = GRONDcurve[:,5]

# Load QSO SED
with fits.open('/utils/optical_nir_qso_sed_001.fits') as hdul:
    data = hdul[1].data
    sed_wave_rest = data['wavelength']        # Å, rest‑frame
    sed_flux      = 4e12 * data['flux']       # scaled flux
    sed_wave_obs  = sed_wave_rest * (1 + z)   # redshift to observer frame

# Interpolator
def sed_flux_interp(wl):
    return np.interp(wl, sed_wave_obs, sed_flux, left=0, right=0)

plt.figure(figsize=(12,4))

# Plot redshifted SED
plt.plot(sed_wave_obs, sed_flux, 'k-', label=f'QSO SED (z={z})', alpha=0.7)

# Filter list
filters = [
    ('MKO_NSFCam', xMKO, MKOFilter, 'orange'),
    ('MMT_SWIRC',  xMMT, MMTFilter, 'blue'),
    ('Paranal_VISTA', xV, VFilter, 'cyan'),
    ('Subaru_MOIRCS', xSubaru, SubaruFilter, 'red'),
    ('UKIRT_WFCAM',  xUKIRT, UKIRTFilter, 'purple'),
    ('LCO_FourStar', xFour, FourFilter, 'olive'),
    ('LaSilla_SOFI', xSOFI, SOFIFilter, 'green'),
    ('GROND_JBand', 10*xG, GFilter, 'brown')
]


for name, fwave, fthroughput, col in filters:
    norm_throughput = fthroughput / fthroughput.max()
    sed_at_filter   = sed_flux_interp(fwave)
    weighted_spec   = sed_at_filter * norm_throughput
    plt.plot(fwave, weighted_spec, color=col, linewidth=1.2,
             label=f'{name} × SED')

# --- vertical emission‑line markers ---
lya_obs  = 1216 * (1 + z)
civ_obs  = 1549 * (1 + z)
mgii_obs = 2800 * (1 + z)
ciiialiii_obs = 1906 * (1 + z)

plt.axvline(lya_obs, color='grey', linestyle='--')
plt.axvline(civ_obs, color='grey', linestyle='--', label='CIV doublet')
plt.axvline(mgii_obs, color='grey', linestyle='-.', label='MgII doublet')
plt.axvline(ciiialiii_obs, color='grey', linestyle=':', label='CIII+AlIII blend')



plt.xlabel('Wavelength (Å)')
plt.ylabel('Flux (relative units)')
plt.title(f'Redshifted QSO SED (z={z}) and Filter‑Weighted Spectra')
plt.legend(fontsize=8, loc='upper right', ncol=2)
plt.xlim(8000, 17000)
plt.ylim(0, 1)          


# Ticks settings
plt.tick_params(axis='both',        # 'x', 'y', or 'both'
                direction='in',     # 'in', 'out', or 'inout'
                length=6,           # tick length in points
                width=1,            # tick width
                which='both')       # 'major', 'minor', or 'both'


plt.tight_layout()
plt.show()


# In[16]:


import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.integrate import simps

# ---------------------------------------------------------------
# helper to load 2‑column filter curves
# ---------------------------------------------------------------
def load_2col_filter(path):
    arr = np.loadtxt(path)
    return arr[:, 0], arr[:, 1]

# ---------------------------------------------------------------
# load all filter curves (unchanged)
# ---------------------------------------------------------------
xMKO,   MKOFilter    = load_2col_filter('/utils/filtercurves/MKO_NSFCam.J.dat')
xMMT,   MMTFilter    = load_2col_filter('/utils/filtercurves/MMT_SWIRC.J.dat')
xV,     VFilter      = load_2col_filter('/utils/filtercurves/Paranal_VISTA.J.dat')
xSubaru,SubaruFilter = load_2col_filter('/utils/filtercurves/Subaru_MOIRCS.J.dat')
xUKIRT, UKIRTFilter  = load_2col_filter('/utils/filtercurves/UKIRT_WFCAM.J.dat')
xFour,  FourFilter   = load_2col_filter('/utils/filtercurves/LCO_FourStar.J_trunc.dat')
xSOFI,  SOFIFilter   = load_2col_filter('/utils/filtercurves/LaSilla_SOFI.J.dat')

GRONDcurve = np.loadtxt('/utils/filtercurves/GROND_filtercurves_trunc.txt',
                        skiprows=1)
xG = GRONDcurve[:, 0]
GFilter = GRONDcurve[:, 5]

# ---------------------------------------------------------------
# area‑normalize all filter curves
# ---------------------------------------------------------------
def area_normalize(w, t):
    return t / simps(t, w)

MKOFilter_norm    = area_normalize(xMKO,   MKOFilter)
MMTFilter_norm    = area_normalize(xMMT,   MMTFilter)
VFilter_norm      = area_normalize(xV,     VFilter)
SubaruFilter_norm = area_normalize(xSubaru,SubaruFilter)
UKIRTFilter_norm  = area_normalize(xUKIRT, UKIRTFilter)
FourFilter_norm   = area_normalize(xFour,  FourFilter)
SOFIFilter_norm   = area_normalize(xSOFI,  SOFIFilter)
GFilter_norm      = area_normalize(10*xG,  GFilter)  # ×10 wavelength scaling

# ---------------------------------------------------------------
# load the rest‑frame quasar SED
# ---------------------------------------------------------------
spec = fits.open('/utils/optical_nir_qso_sed_001.fits')[1].data
lam_rest = spec['wavelength']
flux_rest = spec['flux']

# ---------------------------------------------------------------
# Redshift and normalize the spectrum (area normalization)
# ---------------------------------------------------------------
def redshift_spectrum(lam_rest, flux_rest, z):
    lam_obs = lam_rest * (1 + z)
    area = simps(flux_rest, lam_rest)
    flux_norm = flux_rest / area
    return lam_obs, flux_norm

lam_z53, flux_z53 = redshift_spectrum(lam_rest, flux_rest, 5.52)
lam_z70, flux_z70 = redshift_spectrum(lam_rest, flux_rest, 7.64)

# ---------------------------------------------------------------
# PLOT
# ---------------------------------------------------------------
plt.figure(figsize=(10, 6))

# Calculate observed wavelength for M1450 at each redshift
#m1450_z53 = 1450 * (1 + 5.3)
#m1450_z70 = 1450 * (1 + 7.0)

# Add vertical lines for M1450 rest-frame UV continuum at observed wavelengths
#plt.axvline(m1450_z53, color='tab:gray', linestyle='--', lw=1, label=r'$M_{1450}$ (z=5.3)')
#plt.axvline(m1450_z70, color='tab:orange', linestyle='--', lw=1, label=r'$M_{1450}$ (z=7.0)')


# filter curves
plt.plot(xMKO,   MKOFilter_norm,    'tab:orange',  lw=0.8, label='Gemini NIRI J')
plt.plot(xMMT,   MMTFilter_norm,    'tab:blue',    lw=0.8, label='MMT SWIRC J')
plt.plot(xV,     VFilter_norm,      'tab:cyan',    lw=0.8, label='VISTA VIRCAM J')
plt.plot(xSubaru,SubaruFilter_norm, 'tab:red',    lw=0.8, label='Subaru MOIRCS J')
plt.plot(xUKIRT, UKIRTFilter_norm,  'tab:purple',  lw=0.8, label='UKIRT WFCAM J')
plt.plot(xFour,  FourFilter_norm,   'tab:olive',   lw=0.8, label='Magellan FourStar J')
plt.plot(xSOFI,  SOFIFilter_norm,   'tab:green',   lw=0.8, label='NTT SOFI J')
plt.plot(10*xG,  GFilter_norm,      'tab:brown',   lw=0.8, label='MPGESO GROND J')

# red‑shifted spectra
plt.plot(lam_z53, 0.5*flux_z53, '0.3',  lw=1.2, label='Quasar SED (z=5.3)')
plt.plot(lam_z70, 0.5*flux_z70, '0.3',  lw=1.2, ls='--', label='Quasar SED (z=7.0)')


plt.xlabel('Observed Wavelength (Å)', fontsize=14)
plt.ylabel('Normalized Throughput', fontsize=14)
plt.legend(fontsize=12, loc='upper right', frameon=False)
plt.tight_layout()
plt.tick_params(axis='both',        # 'x', 'y', or 'both'
                direction='in',     # 'in', 'out', or 'inout'
                length=6,           # tick length in points
                width=1,            # tick width
                which='both')       # 'major', 'minor', or 'both'

plt.xlim(8000, 17000)
plt.show()


# In[23]:


import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

def load_2col_filter(path):
    arr = np.loadtxt(path)
    return arr[:,0], arr[:,1]

# Load filters
xMKO, MKOFilter = load_2col_filter('/utils/filtercurves/MKO_NSFCam.J.dat')
xMMT, MMTFilter = load_2col_filter('/utils/filtercurves/MMT_SWIRC.J.dat')
xV, VFilter   = load_2col_filter('/utils/filtercurves/Paranal_VISTA.J.dat')
xSubaru, SubaruFilter = load_2col_filter('/utils/filtercurves/Subaru_MOIRCS.J.dat')
xUKIRT, UKIRTFilter   = load_2col_filter('/utils/filtercurves/UKIRT_WFCAM.J.dat')
xFour, FourFilter     = load_2col_filter('/utils/filtercurves/LCO_FourStar.J_trunc.dat')
xSOFI, SOFIFilter     = load_2col_filter('/utils/filtercurves/LaSilla_SOFI.J.dat')

GRONDcurve = np.loadtxt('/utils/filtercurves/GROND_filtercurves_trunc.txt', skiprows=1)
xG = GRONDcurve[:,0]
GFilter = GRONDcurve[:,5]

filters = [
    ('Gemini NIRI', xMKO, MKOFilter, 'orange'),
    ('MMT SWIRC',  xMMT, MMTFilter, 'blue'),
    ('Paranal VISTA', xV, VFilter, 'cyan'),
    ('Subaru MOIRCS', xSubaru, SubaruFilter, 'red'),
    ('UKIRT WFCAM',  xUKIRT, UKIRTFilter, 'purple'),
    ('LCO FourStar', xFour, FourFilter, 'olive'),
    ('LaSilla SOFI', xSOFI, SOFIFilter, 'green'),
    ('MPGESO GROND', 10*xG, GFilter, 'brown')
]

with fits.open('/utils/optical_nir_qso_sed_001.fits') as hdul:
    data = hdul[1].data
    sed_wave_rest = data['wavelength']
    sed_flux      = 4e12 * data['flux']

def sed_flux_interp(wl, sed_wave_obs, sed_flux):
    return np.interp(wl, sed_wave_obs, sed_flux, left=0, right=0)

redshifts = [6.4, 6.9, 7.6]

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 10), sharex=True)

for i, (ax, z) in enumerate(zip(axes, redshifts)):
    sed_wave_obs = sed_wave_rest * (1 + z)
    ax.plot(sed_wave_obs, sed_flux, 'k-', alpha=0.7)  # no label here

    for name, fwave, fthroughput, col in filters:
        norm_throughput = fthroughput / fthroughput.max()
        sed_at_filter = sed_flux_interp(fwave, sed_wave_obs, sed_flux)
        weighted_spec = sed_at_filter * norm_throughput

        # Add label only in first subplot for combined legend
        if i == 0:
            ax.plot(fwave, weighted_spec, color=col, linewidth=1.2, label=f'{name} × SED')
        else:
            ax.plot(fwave, weighted_spec, color=col, linewidth=1.2)

    # Emission line vertical lines
    lya_obs  = 1216 * (1 + z)
    civ_obs  = 1549 * (1 + z)
    mgii_obs = 2800 * (1 + z)
    ciiialiii_obs = 1906 * (1 + z)

    ax.axvline(lya_obs, color='grey', linestyle='-.', label='Lyman-$\\alpha$')
    ax.axvline(civ_obs, color='grey', linestyle='--', label='CIV doublet')
    #ax.axvline(mgii_obs, color='grey', linestyle='-.', label='MgII doublet')
    ax.axvline(ciiialiii_obs, color='grey', linestyle=':', label='CIII+AlIII blend')

    ax.set_ylabel('Flux (rel. units)', fontsize=14)
    ax.set_xlim(8000, 17000)
    ax.set_ylim(0, 0.99)
    ax.tick_params(axis='both', direction='in', length=6, width=1, which='both')

    # Small legend on each subplot showing just the redshift
    ax.legend([f'$z={z}$'], loc='upper right', fontsize=12, framealpha=0.3, borderpad=0.3)

axes[-1].set_xlabel('Wavelength (Å)', fontsize=14)

# Remove vertical spacing between subplots
fig.subplots_adjust(hspace=0)

# Combined legend below plots (filters only, no redshift labels here)
handles, labels = [], []
for ax in axes:
    h, l = ax.get_legend_handles_labels()
    handles += h
    labels += l

seen = set()
unique = []
for h, l in zip(handles, labels):
    if l not in seen and 'z=' not in l:  # exclude the redshift legends from combined legend
        unique.append((h, l))
        seen.add(l)

if unique:  # just to be safe
    handles, labels = zip(*unique)
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=12, bbox_to_anchor=(0.5, 0.035))
    
fig.subplots_adjust(bottom=0.18, top=0.95)
plt.show()


# In[18]:


import numpy as np
import matplotlib.pyplot as plt

# Load data (skiprows=1 since you said headers are on first line)
GRONDcurve = np.loadtxt('/utils/filtercurves/GROND_filtercurves.txt', skiprows=1)

xG = 10*GRONDcurve[:,0]  # wavelengths in Å, presumably starting at 3480, step 10
filters = {
    'g':  GRONDcurve[:,1],
    'r':  GRONDcurve[:,2],
    'i':  GRONDcurve[:,3],
    'z':  GRONDcurve[:,4],
    'J':  GRONDcurve[:,5],
    'H':  GRONDcurve[:,6],
    'K':  GRONDcurve[:,7],
}

# Define cutoff wavelength ranges for each filter (in Å)
cutoffs = {
    'g': (3500, 5700),
    'r': (4800, 8000),
    'i': (6800, 8400),
    'z': (7900, 11000),
    'J': (10000, 15000),
    'H': (14000, 19500),
    'K': (19000, 26000),
}

# Function to convert wavelength to index
def wl_to_idx(wl, wl_start=3480, step=10):
    return int((wl - wl_start) / step)

plt.figure(figsize=(10,6))

for band, filt_curve in filters.items():
    low, high = cutoffs[band]
    start_idx = wl_to_idx(low)
    end_idx = wl_to_idx(high) + 1
    x_trunc = xG[start_idx:end_idx]
    filt_trunc = filt_curve[start_idx:end_idx]

    # Normalize filter curve
    filt_norm = filt_trunc / np.max(filt_trunc)

    plt.plot(x_trunc, filt_norm, label=band)

plt.xlabel('Wavelength (Å)')
plt.ylabel('Normalized Transmission (relative units)')
plt.title('GROND Filter Curves (Truncated and Normalized)')
plt.legend()

plt.grid(True)
plt.tick_params(axis='both', direction='in', length=6, width=1, which='both')
plt.tight_layout()
plt.show()


# In[4]:


#cracked out plot: last minute use with care:

import numpy as np
import matplotlib.pyplot as plt

# -------- SETTINGS --------
RESULT_FILE = "/utils/Instrument_to_GROND.txt"
FIGSIZE     = (10, 6)
CUTOUT_FILE = "/utils/Instrument_to_GROND_plot.png"  # optional save
# --------------------------

# ---------- 1. LOAD THE DATA ----------
data = np.genfromtxt(
    RESULT_FILE,
    comments="#",
    dtype=float,
    unpack=True
)

(
    zlist,
    diff_MKO,
    diff_MMT,
    diff_VISTA,
    diff_SUBARU,
    diff_UKIRT,
    diff_FOUR,
    diff_SOFI,
) = data

# ---------- 2. PLOT ----------
plt.figure(figsize=FIGSIZE)
plt.xlabel("Redshift", fontsize=14)
plt.ylabel(r"$\Delta m$ (Instrument $\rightarrow$ GROND)", fontsize=14)

plt.plot(zlist, diff_MKO,   color="orange", linestyle="-", linewidth=0.6, label="NIRI → GROND")
plt.plot(zlist, diff_MMT,   color="blue",   linestyle="-", linewidth=0.6, label="SWIRC → GROND")
plt.plot(zlist, diff_VISTA, color="cyan",   linestyle="-", linewidth=0.6, label="VIRCAM → GROND")
plt.plot(zlist, diff_SUBARU,color="red",    linestyle="-", linewidth=0.6, label="MOIRCS → GROND")
plt.plot(zlist, diff_UKIRT, color="purple", linestyle="-", linewidth=0.6, label="WFCAM → GROND")
plt.plot(zlist, diff_FOUR,  color="olive",  linestyle="-", linewidth=0.6, label="FourStar → GROND")
plt.plot(zlist, diff_SOFI,  color="green",  linestyle="-", linewidth=0.6, label="SOFI → GROND")

# Legend centred below the plot (two rows, four columns)
plt.legend(
    fontsize=12,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.1),
    ncol=4,
    fancybox=True,
    shadow=True,
)

plt.tick_params(axis="both", direction="in", which="both", top=True, right=True)
plt.tight_layout(rect=[0, 0.05, 1, 1])   # leave space for the legend


plt.show()


# In[ ]:





# In[ ]:




