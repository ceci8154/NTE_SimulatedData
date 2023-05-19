'''
This file is for getting the HgAr lines for simulating the lamp.
'''

from astroquery.nist import Nist 
import astropy.units as u
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

"""
This code is copied from pyechelle
"""

def get_lines(element, min_wl, max_wl, output_order='wavelength', wavelength_type='vacuum'):
    table_lines = Nist.query(min_wl * u.nm, max_wl * u.nm, linename=element, output_order=output_order, wavelength_type=wavelength_type)['Ritz', 'Rel.']
    df = table_lines.filled(0).to_pandas()
    df['Rel.'] = pd.to_numeric(df['Rel.'], downcast='float', errors='coerce')
    df['Ritz'] = pd.to_numeric(df['Ritz'], downcast='float', errors='coerce')
    df.dropna(inplace=True)
    idx = np.logical_and(df['Rel.'] > 0, df['Ritz'] > 0)
    final1, final2 = df['Ritz'].values[idx], df['Rel.'].values[idx]
    return final1, final2

def get_spectral_density(minwl, maxwl, ar_to_hg_factor):
    hgwl, hgint = get_lines(element='Hg I', min_wl=minwl, max_wl=maxwl)
    arwl, arint = get_lines(element='Ar I', min_wl=minwl, max_wl=maxwl)
    arint *= ar_to_hg_factor
    return np.hstack((hgwl, arwl)), np.hstack((hgint * 226627.84748700162, arint * 226627.84748700162))

def plot_spectral_density(minwl, maxwl, ar_to_hg_factor):
    wl, inten = get_spectral_density(minwl, maxwl, ar_to_hg_factor)
    plt.plot(wl, inten)
    plt.xlabel('Wavelength (microns)')
    plt.ylabel('Intensity')
    plt.show()

if __name__ == '__main__':
    x, y = get_spectral_density(300, 2600, 0.18818806285725248) # wavelength input in nm, output in microns
    
    output_df = pd.DataFrame({'wavelength': x, 'intensity': y})
    output_df.to_csv('HgArFinal.csv', index=False, header=False, sep=' ')