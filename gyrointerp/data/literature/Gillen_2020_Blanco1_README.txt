MNRAS                   NGTS Clusters Survey. I.                  (Gillen, 2019)
================================================================================
NGTS clusters survey. I. Rotation in the young benchmark open cluster Blanco 1
================================================================================

Description:
    Photometric and period information for stars in Blanco 1.

File Summary:
--------------------------------------------------------------------------------
  FileName   Lrecl  Records   Explanations
--------------------------------------------------------------------------------
ReadMe          80       99   This file
table1.dat     192      128   Information for stars detected to be periodic
table2.dat     110       44   Information for stars not detected to be periodic
LCs_*.tar.gz   ---      ---   Light curves and period results for periodic stars
--------------------------------------------------------------------------------

Byte-by-byte Description of file: table1.dat
--------------------------------------------------------------------------------
   Bytes  Format Units         Label     Explanations
--------------------------------------------------------------------------------
   3-  7  I5     ---           NGTS_ID   Internal six digit NGTS ID
  10- 28  I19    ---           Gaia_ID   Gaia DR2 Source Identifier
  31- 42  S12    hh:mm:ss.sss  RA        Right ascension (J2000)
  45- 57  S13    ddd:mm:ss.sss Dec       Declination (J2000)
  60- 64  F5.2   mag           NGTS_mag  NGTS band magnitude
  67- 71  F5.2   mag           G_mag     Gaia G band magnitude
  74- 78  F5.2   mag           BP_mag    Gaia G_BP band magnitude
  81- 85  F5.2   mag           RP_mag    Gaia G_RP band magnitude
  88- 92  F5.2   mag           Ks_mag    2MASS Ks band magnitude
  96- 99  F4.2   mag           GmK       G-Ks colour
 102-105  S4     ---           SpT       Estimated spectral type (see Note)
 108-109  S2     ---           mult      Identified as likely multiple system? *
 112-117  F6.2   ppt           amp_data  Amplitude of the data **
 120-125  F6.2   ppt           amp_GP    Amplitude of the GP model **
 128-135  F8.5   days          P_GP      Period estimate from GP regression
 139-145  F7.5   days          P_GPue    Upper 1-sigma GP uncertainty
 149-155  F7.5   days          P_GPle    Lower 1-sigma GP uncertainty
 158-165  F8.5   days          P_LS      Period estimate from Lomb-Scargle
 168-175  F8.5   days          P_GACF    Period estimate from G-ACF
 178-181  S4     ---           Method    Method selected for final period ***
 184-191  F8.5   days          P_adopt   Period adopted
--------------------------------------------------------------------------------
*   c = CMD and r = RV. For example, "cr" would indicate a system that 
    was highlighted as a likely multiple system by both methods.
**  amplitude is defined as the 10-90th percentile spread and ppt is parts per 
    thousand
*** i.e. the GP, G-ACF or LS method
Note: the magnitudes and colours quoted are as observed (i.e. not dereddened).
Note: spectral types are based purely on dereddened G-Ks colours and are  
      therefore estimates only.
--------------------------------------------------------------------------------

Byte-by-byte Description of file: table2.dat
--------------------------------------------------------------------------------
   Bytes  Format Units         Label     Explanations
--------------------------------------------------------------------------------
   3-  7  I5     ---           NGTS_ID   Internal six digit NGTS ID
  10- 28  I19    ---           Gaia_ID   Gaia DR2 Source Identifier
  31- 42  S12    hh:mm:ss.sss  RA        Right ascension (J2000)
  45- 57  S13    ddd:mm:ss.sss Dec       Declination (J2000)
  60- 64  F5.2   mag           NGTS_mag  NGTS band magnitude
  67- 71  F5.2   mag           G_mag     Gaia G band magnitude
  74- 78  F5.2   mag           BP_mag    Gaia G_BP band magnitude
  81- 85  F5.2   mag           RP_mag    Gaia G_RP band magnitude
  88- 92  F5.2   mag           Ks_mag    2MASS Ks band magnitude
  95- 99  F5.2   mag           GmK       G-Ks colour
 102-105  S4     ---           SpT       Estimated spectral type (see Note)
 108-109  S2     ---           mult      Identified as likely multiple system? *
--------------------------------------------------------------------------------
*   c = CMD and r = RV. For example, "cr" would indicate a system that 
    was highlighted as a likely multiple system by both methods.
Note: the magnitudes and colours quoted are as observed (i.e. not dereddened).
Note: spectral types are based purely on dereddened G-Ks colours and are  
therefore estimates only. The spectral types of stars listed as B9* are suspect 
because this is the earliest spectral type for which G-Ks colours (and 
corresponding spectral types) are available in:
http://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt 
(as of June 2019). We therefore advise caution when interpreting these.
--------------------------------------------------------------------------------

Figure caption: 
--------------------------------------------------------------------------------
NGTS light curves and period comparison for GPs, G-ACF and LS. Left: relative 
flux NGTS light curve (black) with GP model (the orange line and shaded region 
represent the maximum a posteriori GP mean and 1-sigma uncertainty). Masked 
points are shown in blue. Middle: NGTS light curve phase-folded on the adopted 
period, with the rainbow colour scheme indicating data from the beginning 
(indigo) to end (red) of the observations. Right: 1D GP posterior MCMC period 
distribution (orange) with the median period and 1-sigma uncertainties (solid 
and dashed orange lines) shown. The period predictions of G-ACF and LS are 
shown by the vertical blue and green solid lines, respectively.
--------------------------------------------------------------------------------


================================================================================
(End)          Edward Gillen [University of Cambridge]               23-Aug-2019
