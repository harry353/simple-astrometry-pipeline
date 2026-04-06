import numpy as np
import sys, os
import logging
import galsim
import galsim.roman as roman  # still needed for COSMOS catalogs

# MUSE WFM Parameters
outpath = 'output'
nobj = 100
seed = 12345
seed1 = galsim.BaseDeviate(seed).raw()
n_pix = 300           # 1x1 arcmin FoV at 0.2 arcsec/px
pixel_scale = 0.2     # arcsec/px
seeing_fwhm = 0.8     # arcsec, typical Paranal seeing
sky_level = 200.0     # ADU/px, dark Paranal optical sky
read_noise = 3.0      # electrons
gain = 1.5            # e-/ADU

logging.basicConfig(format="%(message)s", stream=sys.stdout)
logger = logging.getLogger("muse_wfm_sim")
logger.setLevel(logging.INFO)

os.makedirs(outpath, exist_ok=True)

# MUSE WFM covers 480-930nm — use V, R, I as representative bands
filter_names = ['g', 'r', 'i']
bandpasses = {
    'g': galsim.Bandpass('LSST_g.dat', wave_type='nm').withZeropoint('AB'),
    'r': galsim.Bandpass('LSST_r.dat', wave_type='nm').withZeropoint('AB'),
    'i': galsim.Bandpass('LSST_i.dat', wave_type='nm').withZeropoint('AB'),
}

# Load COSMOS catalogs for realistic galaxy morphologies
cat1 = galsim.COSMOSCatalog(sample='25.2', area=roman.collecting_area, exptime=roman.exptime)
cat2 = galsim.COSMOSCatalog(sample='23.5', area=roman.collecting_area, exptime=roman.exptime)
logger.info('Read in %d galaxies from I<25.2 catalog' % cat1.nobjects)
logger.info('Read in %d galaxies from I<23.5 catalog' % cat2.nobjects)

vega_sed = galsim.SED('vega.txt', 'nm', 'flambda')
y_bandpass = bandpasses['r']  # use r as reference band for star fluxes

# Simple pixel WCS — no celestial coordinates needed
wcs = galsim.PixelScale(pixel_scale)

# Atmospheric PSF — Kolmogorov, appropriate for ground-based seeing
psf = galsim.Kolmogorov(fwhm=seeing_fwhm)

for ifilter, filter_name in enumerate(filter_names):
    logger.info('Beginning work for MUSE WFM {0}-band.'.format(filter_name))
    bandpass = bandpasses[filter_name]

    full_image = galsim.ImageF(n_pix, n_pix, wcs=wcs)
    image_rng = galsim.UniformDeviate(seed1 + ifilter * nobj)

    for i_obj in range(nobj):
        logger.info('Drawing object %d in band %s' % (i_obj, filter_name))
        obj_rng = galsim.UniformDeviate(seed + 1 + 10**6 + i_obj)
        phot_rng = galsim.UniformDeviate(seed1 + 1 + i_obj + ifilter * nobj)

        p = obj_rng()
        x, y = obj_rng() * n_pix, obj_rng() * n_pix
        image_pos = galsim.PositionD(x, y)

        if p < 0.8:
            # Faint galaxy from deep catalog
            obj = cat1.makeGalaxy(chromatic=True, gal_type='parametric', rng=obj_rng)
            obj = obj.rotate(obj_rng() * 2 * np.pi * galsim.radians)
        elif p < 0.9:
            # Star
            mu_x, sigma_x = 1.e5, 2.e5
            mu = np.log(mu_x**2 / (mu_x**2 + sigma_x**2)**0.5)
            sigma = (np.log(1 + sigma_x**2 / mu_x**2))**0.5
            flux = np.exp(galsim.GaussianDeviate(obj_rng, mean=mu, sigma=sigma)())
            obj = galsim.DeltaFunction() * vega_sed.withFlux(flux, y_bandpass)
        else:
            # Bright/large galaxy from shallower catalog
            obj = cat2.makeGalaxy(chromatic=True, gal_type='parametric', rng=obj_rng)
            obj = (obj.dilate(2) * 4).rotate(obj_rng() * 2 * np.pi * galsim.radians)

        final = galsim.Convolve(obj, psf)
        stamp = final.drawImage(
            bandpass, center=image_pos, wcs=wcs.local(image_pos), method='phot', rng=phot_rng
        )
        bounds = stamp.bounds & full_image.bounds
        full_image[bounds] += stamp[bounds]

    logger.info('All objects drawn for %s-band. Adding noise.' % filter_name)

    # Sky background + Poisson noise
    full_image += sky_level
    full_image.addNoise(galsim.PoissonNoise(image_rng))
    full_image -= sky_level  # sky-subtract

    # CCD read noise
    full_image.addNoise(galsim.GaussianNoise(image_rng, sigma=read_noise))

    # Apply gain
    full_image /= gain
    full_image.quantize()

    outfile = os.path.join(outpath, 'muse_wfm_{0}.fits'.format(filter_name))
    full_image.write(outfile)
    logger.info('Wrote %s' % outfile)

logger.info('Done. View with:')
logger.info('ds9 -zoom 2 -scale limits -10 100 -rgb '
            '-red output/muse_wfm_i.fits '
            '-green output/muse_wfm_r.fits '
            '-blue output/muse_wfm_g.fits')
logger.info('Or in CARTA:')
logger.info('carta output/muse_wfm_i.fits output/muse_wfm_r.fits output/muse_wfm_g.fits')