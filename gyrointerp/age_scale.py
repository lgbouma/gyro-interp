"""
This module contains a dictionary defining the default cluster age scale, as
well as the assumed +1sigma and -1sigma uncertainties on those ages.  These
data are the same as Table 1 in BPH23.
"""

agedict = {
    'default': {
        'α Per': 80,
        'Pleiades': 120,
        'Psc-Eri': 120,
        'Blanco-1': 120,
        'NGC-3532': 300,
        'Group-X': 300,
        'Praesepe': 670,
        'NGC-6811': 1000,
        'NGC-6819': 2600,
        'Rup-147': 2600,
        'M67': 4000,
        'reference_ages': [80, 120, 300, 670, 1000, 2600, 4000]
    },
    '1sigmaolder': {
        'α Per': 80+5,
        'Pleiades': 120+12,
        'Psc-Eri': 120+12,
        'Blanco-1': 120+12,
        'NGC-3532': 300+60,
        'Group-X': 300+60,
        'Praesepe': 670+67,
        'NGC-6811': 1000+100,
        'NGC-6819': 2600+200,
        'Rup-147': 2600+200,
        'M67': 5000, # Dungee+2022
        'reference_ages': [80+5, 120+12, 300+60, 670+67, 1000+100, 2600+200, 4000+1000]
    },
    '1sigmayounger': {
        'α Per': 80-5,
        'Pleiades': 120-12,
        'Psc-Eri': 120-12,
        'Blanco-1': 120-12,
        'NGC-3532': 300-50,
        'Group-X': 300-70,
        'Praesepe': 670-67,
        'NGC-6811': 1000-100,
        'NGC-6819': 2600-200,
        'Rup-147': 2600-200,
        'M67': 3500, # Dungee+2022
        'reference_ages': [80-5, 120-12, 300-60, 670-67, 1000-100, 2600-200, 4000-500]
    }
}
