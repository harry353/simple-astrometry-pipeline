import numpy as np
import os
import sys
import multiprocessing
import time
import galsim
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants_datagen import HAYSTACK_SIZE, GALAXY_DENSITY, PIXEL_SCALE, P_FAINT_GALAXY, P_STAR
import galsim.roman as roman  # roman module is used only for its collecting_area and exptime
                               # constants when loading COSMOS catalogs, not for Roman-specific
                               # optics or bandpasses

# This script simulates a synthetic VLT/MUSE WFM white-light image.
#
# References:
#   Reiss et al., 2012SPIE.8446E..2PR -- "The MUSE instrument detector system"
#   http://www.eso.org/sci/facilities/develop/detectors/optdet/docs/papers/RRE_SPIE2012-06-14.pdf
#
#   ESO MUSE instrument overview:
#   https://www.eso.org/sci/facilities/paranal/instruments/muse/overview.html


seed = 12345                            # Master random seed for reproducibility.
seed1 = galsim.BaseDeviate(seed).raw()  # Convert to raw uint64 for use with GalSim RNGs

n_pix       = HAYSTACK_SIZE
pixel_scale = PIXEL_SCALE     # arcsec/pixel
nobj        = int(round(GALAXY_DENSITY * (n_pix * pixel_scale / 60) ** 2))

# Cerro Paranal 50th percentile over 2016-2023~0.72", with the
# standard ESO "nominal" condition defined as 0.8". We use 0.8"
# as a representative "typical" observing condition.
#
# ESO Paranal astroclimatic statistics:
# https://www.eso.org/sci/facilities/paranal/astroclimate/site.html
seeing_fwhm = 0.8     # arcsec FWHM of the atmospheric PSF


# The Paranal dark-sky optical brightness in V-band is V ≈ 21.6 mag/arcsec²
# and in R ≈ 20.9 mag/arcsec² (Patat et al. 2003, A&A 400, 981).
# https://www.aanda.org/articles/aa/full/2003/12/aa3223/aa3223.right.html
#
# For MUSE with gain=1.5 e-/ADU and a typical 600 s exposure the dark sky
# contributes ~150-250 ADU/pixel in optical bands. We use 200 ADU/pixel as a
# representative figure consistent with these conditions.
sky_level = 200.0     # ADU/pixel, flat sky background to add before noise

# Read noise and gain given in:
#   Reiss et al., 2012SPIE.8446E..2PR -- "The MUSE instrument detector system"
#   http://www.eso.org/sci/facilities/develop/detectors/optdet/docs/papers/RRE_SPIE2012-06-14.pdf
read_noise = 3.0      # electrons RMS per pixel per read
gain = 1.0            # e-/ADU (electrons per analogue-to-digital unit)

_DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
os.makedirs(_DATASET_DIR, exist_ok=True)


# Tangent-plane (gnomonic) WCS centred on the COSMOS field.
# CRPIX is placed at the exact image centre (n_pix/2 + 0.5 gives the
# half-pixel shift to land between pixels, per the FITS standard).
# The CD matrix encodes the pixel scale with North up / East left.
_crpix = galsim.PositionD(n_pix / 2 + 0.5, n_pix / 2 + 0.5)
_affine = galsim.AffineTransform(
    -pixel_scale, 0.0,       # RA increases eastward = left
     0.0, pixel_scale,       # Dec increases northward = up
    origin=_crpix,
)
wcs = galsim.TanWCS(
    affine=_affine,
    world_origin=galsim.CelestialCoord(
        ra=0.0 * galsim.degrees,
        dec=0.0 * galsim.degrees,
    ),
    units=galsim.arcsec,
)


# Each worker initialises its own copies of the catalogs, SED, bandpass, and
# PSF so that none of these large GalSim objects need to be pickled and sent
# across process boundaries on every task.

_worker_cat1 = None
_worker_cat2 = None
_worker_vega_sed = None
_worker_bandpass = None
_worker_psf = None


def _init_worker():
    global _worker_cat1, _worker_cat2, _worker_vega_sed, _worker_bandpass, _worker_psf
    _worker_cat1 = galsim.COSMOSCatalog(sample='25.2', area=roman.collecting_area, exptime=roman.exptime)
    _worker_cat2 = galsim.COSMOSCatalog(sample='23.5', area=roman.collecting_area, exptime=roman.exptime)
    _worker_vega_sed = galsim.SED('vega.txt', 'nm', 'flambda')
    _worker_bandpass = galsim.Bandpass('1', wave_type='nm', blue_limit=480, red_limit=930).withZeropoint('AB')
    _worker_psf = galsim.Kolmogorov(fwhm=seeing_fwhm)


def _draw_one_object(i_obj):
    """Draw a single object stamp and return its pixel data and position."""
    obj_rng  = galsim.UniformDeviate(seed + 1 + 10**6 + i_obj)
    phot_rng = galsim.UniformDeviate(seed1 + 1 + i_obj)

    p = obj_rng()
    x, y = obj_rng() * n_pix, obj_rng() * n_pix
    image_pos = galsim.PositionD(x, y)

    if p < P_FAINT_GALAXY:
        # ---- Faint background galaxy (80% of objects) ----
        # Drawn from the deep I<25.2 COSMOS catalog, which is dominated
        # by small, faint high-redshift galaxies.
        obj = _worker_cat1.makeGalaxy(chromatic=True, gal_type='parametric', rng=obj_rng)
        obj = obj.rotate(obj_rng() * 2 * np.pi * galsim.radians)

    elif p < P_STAR:
        # ---- Star population (10% of objects) ----
        # Draw stellar flux from a log-normal distribution
        # See: https://en.wikipedia.org/wiki/Log-normal_distribution
        mu_x, sigma_x = 1.e5, 2.e5   # mean and std of flux in e-
        # Convert from (mean, std) of the linear variable to (mu, sigma)
        # of the underlying normal distribution using the log-normal
        # moment relations:
        #   mu    = log(mu_x² / sqrt(mu_x² + sigma_x²))
        #   sigma = sqrt(log(1 + sigma_x² / mu_x²))
        mu = np.log(mu_x**2 / (mu_x**2 + sigma_x**2)**0.5)
        sigma = (np.log(1 + sigma_x**2 / mu_x**2))**0.5
        flux = np.exp(galsim.GaussianDeviate(obj_rng, mean=mu, sigma=sigma)())
        # DeltaFunction() is a point source (star); multiplying by the SED
        # makes it chromatic so it can be rendered through a Bandpass.
        obj = galsim.DeltaFunction() * _worker_vega_sed.withFlux(flux, _worker_bandpass)

    else:
        # ---- Bright/extended galaxy (10% of objects) ----
        # Drawn from the shallower I<23.5 COSMOS catalog, which contains
        # more luminous, physically larger galaxies.
        obj = _worker_cat2.makeGalaxy(chromatic=True, gal_type='parametric', rng=obj_rng)
        obj = (obj.dilate(2) * 4).rotate(obj_rng() * 2 * np.pi * galsim.radians)

    final = galsim.Convolve(obj, _worker_psf)

    # Draw the object onto a small cutout image just large enough to contain it,
    # then add that cutout into the full image. This is cheaper than drawing each
    # object directly onto the full canvas. Objects near the edge may hang outside
    # the canvas, so we clip to the overlap.
    stamp = final.drawImage(
        _worker_bandpass, center=image_pos, wcs=wcs.local(image_pos), method='phot', rng=phot_rng
    )

    full_bounds = galsim.BoundsI(1, n_pix, 1, n_pix)
    bounds = stamp.bounds & full_bounds
    if not bounds.isDefined():
        return None

    # Return the pixel data as a numpy array together with the bounds so the
    # main process can paste it into the full image.
    return stamp[bounds].array.copy(), bounds.xmin, bounds.ymin, bounds.xmax, bounds.ymax


def generate_haystack(seed, output_dir, filename_prefix, save_clean=False):
    """Generate a GalSim haystack image and return (image_array, fits_header).

    Runs single-threaded to avoid nested multiprocessing when called from
    within a ProcessPoolExecutor worker.  The returned image is normalised to
    [0, 1] so it is compatible with the rest of the pipeline (03_create_needle,
    etc.).  The FITS header uses CDELT format for compatibility with the
    WCS helpers in 03_create_needle.py.

    Parameters
    ----------
    seed : int
        Master random seed for this image (use a different value per pair).
    output_dir : str
        Directory to write the output FITS file into.
    filename_prefix : str
        Base name for the output files (no extension).
    save_clean : bool
        If True, also save a copy without noise (mirroring 01b_create_haystack_gaussian).

    Returns
    -------
    image_array : np.ndarray, shape (n_pix, n_pix), dtype float32
        Normalised pixel data in [0, 1].
    fits_header : astropy.io.fits.Header
        FITS header with CDELT-based WCS (RA---TAN / DEC--TAN).
    """
    import galsim.roman as roman
    from astropy.io import fits as afits
    from astropy.wcs import WCS as aWCS

    _seed1 = galsim.BaseDeviate(seed).raw()

    cat1      = galsim.COSMOSCatalog(sample='25.2', area=roman.collecting_area, exptime=roman.exptime)
    cat2      = galsim.COSMOSCatalog(sample='23.5', area=roman.collecting_area, exptime=roman.exptime)
    vega_sed  = galsim.SED('vega.txt', 'nm', 'flambda')
    bandpass  = galsim.Bandpass('1', wave_type='nm', blue_limit=480, red_limit=930).withZeropoint('AB')
    psf       = galsim.Kolmogorov(fwhm=seeing_fwhm)

    full_image = galsim.ImageF(n_pix, n_pix, wcs=wcs)
    image_rng  = galsim.UniformDeviate(_seed1)

    for i_obj in range(nobj):
        obj_rng  = galsim.UniformDeviate(seed + 1 + 10**6 + i_obj)
        phot_rng = galsim.UniformDeviate(_seed1 + 1 + i_obj)

        p = obj_rng()
        x, y = obj_rng() * n_pix, obj_rng() * n_pix
        image_pos = galsim.PositionD(x, y)

        if p < P_FAINT_GALAXY:
            obj = cat1.makeGalaxy(chromatic=True, gal_type='parametric', rng=obj_rng)
            obj = obj.rotate(obj_rng() * 2 * np.pi * galsim.radians)
        elif p < P_STAR:
            mu_x, sigma_x = 1.e5, 2.e5
            mu    = np.log(mu_x**2 / (mu_x**2 + sigma_x**2)**0.5)
            sigma = (np.log(1 + sigma_x**2 / mu_x**2))**0.5
            flux  = np.exp(galsim.GaussianDeviate(obj_rng, mean=mu, sigma=sigma)())
            obj   = galsim.DeltaFunction() * vega_sed.withFlux(flux, bandpass)
        else:
            obj = cat2.makeGalaxy(chromatic=True, gal_type='parametric', rng=obj_rng)
            obj = (obj.dilate(2) * 4).rotate(obj_rng() * 2 * np.pi * galsim.radians)

        final = galsim.Convolve(obj, psf)
        stamp = final.drawImage(bandpass, center=image_pos, wcs=wcs.local(image_pos),
                                method='phot', rng=phot_rng)

        full_bounds = galsim.BoundsI(1, n_pix, 1, n_pix)
        bounds = stamp.bounds & full_bounds
        if not bounds.isDefined():
            continue
        full_image[bounds] += stamp[bounds].array.copy()

    # Add sky + Poisson noise + read noise, then sky-subtract (matches __main__)
    clean_array = full_image.array.copy()

    full_image += sky_level
    full_image.addNoise(galsim.PoissonNoise(image_rng))
    full_image -= sky_level
    full_image.addNoise(galsim.GaussianNoise(image_rng, sigma=read_noise))
    full_image /= gain
    full_image.quantize()

    noisy_array = full_image.array.copy()

    # Normalise both arrays to [0, 1] using a percentile-based stretch so that
    # faint galaxy profiles land well above 0 rather than being crushed by
    # bright stellar peaks.  The same reference (derived from the noisy image)
    # is applied to both arrays so they share a consistent scale.
    vmin = np.percentile(noisy_array, 1)
    vmax = np.percentile(noisy_array, 99)
    if vmax > vmin:
        noisy_norm = np.clip((noisy_array - vmin) / (vmax - vmin), 0, 1).astype(np.float32)
        clean_norm = np.clip((clean_array - vmin) / (vmax - vmin), 0, 1).astype(np.float32)
    else:
        noisy_norm = np.zeros_like(noisy_array, dtype=np.float32)
        clean_norm = np.zeros_like(clean_array, dtype=np.float32)

    # Build a CDELT-based WCS header compatible with 03_create_needle.build_needle_header
    cdelt_deg = pixel_scale / 3600.0   # arcsec/pix -> deg/pix
    w = aWCS(naxis=2)
    w.wcs.crpix = [n_pix / 2 + 0.5, n_pix / 2 + 0.5]
    w.wcs.crval = [0.0, 0.0]
    w.wcs.cdelt = [-cdelt_deg, cdelt_deg]   # RA decreases eastward
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    fits_header = w.to_header()

    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    noisy_path = os.path.join(output_dir, f"{filename_prefix}.fits")
    afits.writeto(noisy_path, noisy_norm, header=fits_header, overwrite=True)

    plt.imsave(os.path.join(output_dir, f"{filename_prefix}.png"), noisy_norm, cmap='viridis')

    if save_clean:
        clean_path = os.path.join(output_dir, f"{filename_prefix}_clean.fits")
        afits.writeto(clean_path, clean_norm, header=fits_header, overwrite=True)
        plt.imsave(os.path.join(output_dir, f"{filename_prefix}_clean.png"), clean_norm, cmap='viridis')

    # Return the noise-free image (matching 01b_create_haystack_gaussian.main()) so that
    # 03_create_needle can find galaxy-rich regions and extract a clean ground truth.
    return clean_norm, fits_header


# Main
if __name__ == '__main__':
    # Load catalogs once in the main process just to report object counts.
    cat1 = galsim.COSMOSCatalog(sample='25.2', area=roman.collecting_area, exptime=roman.exptime)
    cat2 = galsim.COSMOSCatalog(sample='23.5', area=roman.collecting_area, exptime=roman.exptime)
    print('Read in %d galaxies from I<25.2 catalog' % cat1.nobjects)
    print('Read in %d galaxies from I<23.5 catalog' % cat2.nobjects)

    print('Beginning white-light MUSE WFM simulation.')
    t_start = time.time()

    full_image = galsim.ImageF(n_pix, n_pix, wcs=wcs)
    image_rng  = galsim.UniformDeviate(seed1)

    n_workers = os.cpu_count()
    print('Drawing %d objects across %d workers.' % (nobj, n_workers))

    with multiprocessing.Pool(processes=n_workers, initializer=_init_worker) as pool:
        results = pool.map(_draw_one_object, range(nobj))

    # Accumulate stamps into the full image sequentially (no race conditions).
    for i_obj, result in enumerate(results):
        if result is None:
            continue
        arr, xmin, ymin, xmax, ymax = result
        bounds = galsim.BoundsI(xmin, xmax, ymin, ymax)
        full_image[bounds] += arr

    print('All objects drawn. Adding noise.')

    # Step 1: Add the sky background as a flat pedestal.
    # Poisson noise is then applied to the total (source + sky) signal,
    # which makes brighter pixels noisier.
    full_image += sky_level     # ADU/pixel

    # Step 2: Poisson noise. Each pixel's standard deviation is sqrt(N_photons),
    # where N_photons ≈ pixel_value (in electrons).
    full_image.addNoise(galsim.PoissonNoise(image_rng))

    # Step 3: Sky-subtract: remove the artificial sky pedestal added in
    # Step 1 so the output image has a mean background near zero, as real
    # science-ready images are delivered after background subtraction.
    full_image -= sky_level

    # Step 4: CCD read noise. Gaussian white noise added by the detector
    # electronics during analogue-to-digital conversion. GalSim's GaussianNoise
    # treats the image as already in electrons at this stage.
    full_image.addNoise(galsim.GaussianNoise(image_rng, sigma=read_noise))

    # Step 5: Apply detector gain to convert electrons → ADU.
    # After this step the image is in units of ADU (counts).
    full_image /= gain   # e- → ADU

    # Step 6: Quantise to integer ADU values. This rounds each pixel
    # to the nearest integer.
    full_image.quantize()

    outfile = os.path.join(_DATASET_DIR, 'muse_wfm_white.fits')
    full_image.write(outfile)
    print('Wrote %s' % outfile)
    print('Total time: %.1f s' % (time.time() - t_start))

    print('\nDone. View in DS9 with:')
    print('ds9 -zoom 2 -scale limits -10 100 %s' % outfile)
    print('\nOr in CARTA:')
    print('carta %s' % outfile)
