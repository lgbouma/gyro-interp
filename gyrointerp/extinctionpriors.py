"""
This module defines a dictionary of mean extinction A_V values adopted for the
reference open clusters when calibrating the gyrochronal model.
"""
#
# Pleiades, Praesepe, and NGC-6811 are as quoted in Appendix A.3 of
# Curtis+2020, ApJ 904 140.
# NGC-3532 is from Fritzewski+2021.
# Group-X is from Messina+2022, E(B-V)=0.005 and R_V = A_V / E(B - V) = 3.1
# Blanco-1 is from Gillen+2020 (who got it from Gaia Collab+2018), E(B-V)=0.010
#
# For Pleiades, the age (and presumably A_V) references are Stauffer+1998 and
# Dahm 2015.
# For Praesepe, it is Appendix A of Douglas+2019.
# For NGC-6811, it's Curtis+2019a. (The NGC-6811 paper, not the Psc-Eri one).
#
# For alpha-Per, E(B-V) = 0.090 from Gaia Collab+2018 -> A_V = 0.279
#
# For M34, E(B-V) = 0.07 from Meibom+2011 -> A_V = 0.217
#
# For M67, E(B-V) = 0.04 from Gruner+2023 / Taylor2007 -> A_V = 0.124.

extinction_A_V_dict = {
    "Pleiades": 0.12,
    "Praesepe": 0.035, # https://ui.adsabs.harvard.edu/abs/2019ApJ...879..100D/abstract
    "NGC-6811": 0.15,
    "NGC-3532": 0.034,
    "Group-X": 0.0155,
    'Blanco-1': 0.031,
    'Psc-Eri': 0, # Fig 5 of Curtis+2019, zero reddening assumed
    'alpha-Per': 0.279, # Gaia Collaboration+2018
    'Ruprecht-147': 0.30, # Curtis+2020, table 5
    'NGC-6819': 0.44, # Curtis+2020, table 5 (Meibom+2015 periods)
    'M34': 0.217, # Meibom+2011
    'M67': 0.124
}
