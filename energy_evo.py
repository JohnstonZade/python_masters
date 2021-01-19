import diagnostics as diag
default_prob = diag.DEFAULT_PROB


def get_energy_data(output_dir,
                    prob=default_prob,
                    B0=1.0,
                    Lx=1.0):
    hstData = diag.load_hst(output_dir, prob)

    # Volume to calculate energy density
    if prob == 'turb':
        vol = 1  # Athena already takes this into account for the turbulent sim
    else:
        vol = diag.get_vol(output_dir, prob)

    t_A = Lx  # true when v_A = 1 and B0 along the x axis
    t = hstData['time'] / t_A  # scale by Alfven period
    KE = (hstData['1-KE'] + hstData['2-KE'] + hstData['3-KE']) / vol  # kinetic
    ME = (hstData['2-ME'] + hstData['3-ME']) / vol  # perp magnetic energy

    return t, KE, ME
