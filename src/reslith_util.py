import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from tqdm.notebook import tqdm


def build_reslithtransform(all_relat, coarse_frac, nquantiles=1001, bin_size=0.001,
                           progress=True):
    """Build the resistivity-lithology transform
    
    Parameters
    ----------
        all_relat : array of float
            bootstrapped resistivity values of shape (nb, nc) for coarse fraction values
            on the range [0, 1]. nb is the number of bootstrapped samples. For each sample,
            resistivity is calculated at nc coarse fraction values.
        coarse_frac : array of float
            coarse fractions of shape (nc,) corresponding to the columns of all_relat
        nquantiles : int
            number of quantiles to use
        bin_size : float
            bin size used to bin log resistivity 
        progress : bool
            whether to print a progress bar

    Returns
    -------
        trans : array of float
            the resistivity-lithology transform of shape (nquantiles, nbins) containing the
            cdf of coarse fraction for each value of resistivity
        res_vals : array of float
            resistivity values corresponding to the columns of rpt
    """
     
    # quantiles at which to evaluate coarse fraction CDF
    quantiles = np.linspace(0, 1.0, nquantiles)
    
    # Convert rock physics model to log
    log_rpm = np.log10(all_relat)
    rounding = int(-np.log10(bin_size))
    
    # Create bins for digitizing log10(resistivity)
    bin_min = np.floor(log_rpm.min()*10**rounding)/10**rounding
    bin_max = np.ceil(log_rpm.max()*10**rounding)/10**rounding
    bin_edges = np.arange(bin_min, bin_max+1e-9, bin_size)
    bin_centers = (bin_edges[1:] + bin_edges[:-1])/2

    # Construct coarsemat from coarse_frac
    coarsemat = np.tile(coarse_frac.reshape(1, -1), (all_relat.shape[0], 1))
    
    # Build rock physics transform, which contains cdf of each coarse fraction
    # Columns: Resistivity values (bin_centers)
    # Rows: For each resistivity value, coarse fraction value at each quantile
    trans = np.ones((quantiles.shape[0], bin_centers.shape[0]), dtype=float)*np.nan

    if progress:
        loop = tqdm(bin_edges[:-1], desc='Building resistivity-lithology transform')
    else:
        loop = bin_edges[:-1]

    nancols = []
    
    for ibin, bin_low in enumerate(loop):
        bin_high = bin_edges[ibin + 1]
        mask = (log_rpm >= bin_low)*(log_rpm < bin_high)
        if mask.sum() > 0:
            cdf = np.quantile(coarsemat[mask], quantiles)
            trans[:, ibin] = cdf.copy()
        else:
            nancols.append(ibin)
        
    # Replace any columns with nan with mean of quantile values on either side
    for icol in nancols:
        trans[:, icol] = np.mean(trans[:, [icol-1, icol+1]], axis=1)

    return trans, bin_centers


def plot_reslith(end_relat, all_relat, coarse_frac, bins=None, density=False):
    """Plot the resistivity-lithology relationship and transforms
    as separate subplots.
    
    Parameters
    ----------
        end_relat : array of float
            bootstrapped samples of resistivity for fine- and coarse-grained end members
            where the first column is fine and second column is coarse
        all_relat : array of float
            bootstrapped resistivity values of shape (nb, nc) for coarse fraction values
            on the range [0, 1]. nb is the number of bootstrapped samples. For each sample,
            resistivity is calculated at nc coarse fraction values.
        coarse_frac : array of float
            coarse fraction values of shape (nc,) corresponding to the columns of all_relat
        bins : int or array of float
            bins to use to plot for both sediment types, or the number of bins to use
        density : bool
            whether to color the points by their probability along each row

    Returns
    -------
        fig, ax : matplotlib objects
            matplotlib figure and axes handles
    """
    
    if density:
        pdfs = np.zeros(all_relat.shape, dtype=float)
        for i in range(all_relat.shape[1]):
            kde_model = gaussian_kde(all_relat[:, i])
            pdfs[:, i] = kde_model(all_relat[:, i])
        
    if bins is None:
        bins = 10

    fig, ax = plt.subplots(2, figsize=(9, 6), sharex='all')

    ax[0].hist(end_relat[:, 0], label='Fine', bins=bins, density=True,
               color='blue', edgecolor='k')
    ax[0].hist(end_relat[:, 1], label='Coarse', bins=bins, density=True,
               color='yellow', edgecolor='k')

    ax[0].set(ylabel='Percentage')
    ax[0].legend()
    
    coarsemat = np.tile(coarse_frac.reshape(1, -1), (all_relat.shape[0], 1))
    
    if density:
        ax[1].scatter(all_relat, coarsemat, marker='.', s=3, c=pdfs, cmap='cividis')
    else:
        ax[1].scatter(all_relat, coarsemat, marker='.', s=3, color='k', alpha=0.01)
    
    ax[1].set(xlabel='Resistivity(Ohm m)', ylabel='Coarse fraction')
    
    return fig, ax


def combine_cptttem(cpt, ttem, surface_elevation=90):
    """Combine CPT and tTEM data by averaging coarse fraction across each 
    tTEM cell for each collocated CPT and tTEM measurement.
    
    Parameters
    ----------
        cpt : array of float
            cpt data of shape (n, 4) organized in x, y, z, v where
            v is coarse fraction
        ttem : array of float
            tTEM data of shape (m, 4) organized in x, y, z, v where
            z is the elevation of the bottom of each tTEM cell and
            v is the inverted resistivity
        surface_elevation : float
            elevation of the surface. Used to determine the thickness 
            of the uppermost tTEM cell
            
    Returns
    -------
        cptttem : array of float
            combined CPT and tTEM data containing x, y, z, v1, v2 
            where v1 is the resitivity and v2 is the average coarse
            fraction across that tTEM cell
    """

    # Get unique cpt xy locations
    cptxy = np.unique(cpt[:, :2], axis=0)
    ncpt = cptxy.shape[0]
    
    # Build single array containing all collocated tTEM and CPT data
    # For loop is plenty fast here since ncpt is always small
    # (more readable than other methods)
    for i in range(ncpt):
        xy = cptxy[i, :]

        # Get ttem coordinates closest to each cpt coordinate
        dist = np.sqrt((xy[0]-ttem[:, 0])**2+(xy[1]-ttem[:, 1])**2)
        ttemxy = ttem[dist.argmin(), :2]

        # Create array of tTEM data at this xy location
        ttemmask = (ttem[:, 0] == ttemxy[0])*(ttem[:, 1] == ttemxy[1])
        ttem_at_this_loc = ttem[ttemmask, :].copy()
        
        # Create array of CPT data at this xy location
        cptmask = (cpt[:, 0] == cptxy[i, 0])*(cpt[:, 1] == cptxy[i, 1])
        cpt_at_this_loc = cpt[cptmask, :].copy()

        # Add in column containing the top of the ttem cell
        # ttem_at_this_loc columns are now:
        # [x    y    cell_top   cell_bot   rho]
        ttem_at_this_loc = np.insert(ttem_at_this_loc, 2, surface_elevation, axis=1)
        ttem_at_this_loc[1:, 2] = ttem_at_this_loc[:-1, 3]

        # Add additonal column of coarsefrac
        ttem_at_this_loc = np.insert(ttem_at_this_loc, 5, np.nan, axis=1)

        for j in range(ttem_at_this_loc.shape[0]):
            mask = (ttem_at_this_loc[j, 2] > cpt_at_this_loc[:, 2])*\
                   (ttem_at_this_loc[j, 3] <= cpt_at_this_loc[:, 2])
            if mask.sum() > 0:
                ttem_at_this_loc[j, -1] = cpt_at_this_loc[mask, -1].mean()

        if i == 0:
            cptttem = ttem_at_this_loc.copy()
        else:
            cptttem = np.vstack((cptttem, ttem_at_this_loc))
      
    # Delete column containing z coordinate of the top of the ttem cell
    cptttem = np.delete(cptttem, 2, axis=1)

    return cptttem


def apply_reslithtransform(res_real, trans, resvals, min_quantile=0.50, max_quantile=0.50):
    """Apply a resistivity-lithology transform to a realization of resistivity
    
    Parameters
    ----------
        res_real : array of float
            array of size (nx, ny, nz) containing the resistivity values
            to convert to coarse fraction
        trans : array of float
            array of size (nquantiles, nres) containing the coarse fraction
            at each quantile for a specific resistivity bin
        resvals : array of float
            center of the bins of resistivity corresponding to the columns
            of trans
        min_quantile, max_quantile : float
            the quantile values used to transform resistivity to coarse 
            fraction corresponding to the lowest (min_quantile) and 
            highest (max_quantile) resistivity values in trans. For example,
            if both min_ and max_quantile are set to 0.50, the median value
            will be used across all resisitvity bins
    
    Returns
    -------
        cf_real : array of float
            array of size (nx, ny, nz) containing the transformed coarse
            fraction values
    """

    # Determine quantiles based on trans shape
    # These correspond to the rows of trans
    tquantiles = np.linspace(0, 1, trans.shape[0])

    # Create coarsefrac_per_resval
    # This is the list of coarse fractions corresponding to each 
    # resistivity in resvals

    # create quantile used for each resistivity bin
    qvalues = np.linspace(min_quantile, max_quantile, trans.shape[1])
    qvalues = np.around(qvalues, 3) 
    a = np.arange(tquantiles.shape[0])

    # for each resistivity value, get row corresponding to chosen quantile 
    rowidx = a[np.digitize(qvalues, tquantiles, right=True)]

    # Convert this row to absolute index once the array is flattened
    idx = rowidx + np.arange(trans.shape[1]) * trans.shape[0]

    # Flatten array (in column-major order) and extract coarse fraction values
    coarsefrac_per_resval = trans.flatten(order='F')[idx]
    coarsefrac_per_resval = np.append(coarsefrac_per_resval, 1.0) # append high value
    coarsefrac_per_resval = np.insert(coarsefrac_per_resval, 0, 0.0) # prepend low value

    # Create bin edges and bin centers for digitizing realization
    # Round resval difference because of limits of float64 literals
    bin_size = np.around(resvals[1] - resvals[0], 10)
    rounding = int(-np.log10(bin_size))
    bin_min = np.around(resvals.min(), rounding)
    bin_max = np.around(resvals.max(), rounding) + bin_size
    bin_edges = np.arange(bin_min, bin_max+1e-9, bin_size)

    # Note that with np.digitize:
    #     if x < lowest bin, 0 is returned
    #     if x is in lowest bin, 1 is returned
    #     if x is in highest bin, len(bins) - 1 is returned
    #     if x is > highest bin, len(bins) is returned
    cf_real = coarsefrac_per_resval[np.digitize(res_real, bin_edges)]

    return cf_real
