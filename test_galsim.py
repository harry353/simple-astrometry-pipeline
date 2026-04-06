import numpy as np
import os
import galsim
from constants_datagen import OUTPATH, NOBJ, N_PIX, PIXEL_SCALE, P_FAINT_GALAXY, P_STAR
import galsim.roman as roman  # roman module is used only for its collecting_area and exptime
                               # constants when loading COSMOS catalogs, not for Roman-specific
                               # optics or bandpasses

# This script simulates a synthetic VLT/MUSE WFM observation.
#
# References:
#   Reiss et al., 2012SPIE.8446E..2PR -- "The MUSE instrument detector system"
#   http://www.eso.org/sci/facilities/develop/detectors/optdet/docs/papers/RRE_SPIE2012-06-14.pdf
#
#   ESO MUSE instrument overview:
#   https://www.eso.org/sci/facilities/paranal/instruments/muse/overview.html


nobj = NOBJ                              # Number of objects to place in the simulated field
seed = 12345                            # Master random seed for reproducibility.
seed1 = galsim.BaseDeviate(seed).raw()  # Convert to raw uint64 for use with GalSim RNGs


# MUSE WFM field of view: 59.9" × 60.0" ≈ 1'×1' image side length 
# in pixels: 1500 px × 0.2"/px = 300" = 5'. We use 1500 px to have a 
# large enough canvas to cut out the 300×300 px needle.
n_pix = N_PIX
pixel_scale = PIXEL_SCALE     # arcsec/pixel

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

os.makedirs(OUTPATH, exist_ok=True)


# MUSE WFM covers 480-930 nm.
# https://www.eso.org/sci/facilities/paranal/instruments/muse/overview.html
#
# Because GalSim's chromatic rendering requires to give bandpass objects, we
# approximate the MUSE wavelength range using three broad-band LSST filters
# (g, r, i) that together span roughly 400-820 nm. Good enough for the purpose
# of producing a simulation of a colour image.
#
# LSST filter curves ship with GalSim as 'LSST_g.dat', 'LSST_r.dat', etc.
# GalSim Bandpass docs: https://galsim-developers.github.io/GalSim/_build/html/bandpass.html
filter_names = ['g', 'r', 'i']
bandpasses = {
    # g-band: ~320-560 nm (peaks ~480 nm) -- represents MUSE blue end
    'g': galsim.Bandpass('LSST_g.dat', wave_type='nm').withZeropoint('AB'),
    # r-band: ~550-700 nm (peaks ~620 nm) -- represents MUSE mid range
    'r': galsim.Bandpass('LSST_r.dat', wave_type='nm').withZeropoint('AB'),
    # i-band: ~680-830 nm (peaks ~750 nm) -- represents MUSE red end
    'i': galsim.Bandpass('LSST_i.dat', wave_type='nm').withZeropoint('AB'),
}


# We use GalSim's COSMOSCatalog, which wraps the HST ACS F814W (I-band) COSMOS
# morphology catalog. Two depth samples are available:
#
#   '25.2': F814W < 25.2  ~87,000 galaxies (faint, compact — typical field pop.)
#   '23.5': F814W < 23.5  ~26,000 galaxies (brighter, larger — more extended)
#
# GalSim COSMOSCatalog docs:
# https://galsim-developers.github.io/GalSim/_build/html/real_gal.html
#
# We pass Roman Space Telescope collecting area and exposure time as the
# reference flux normalisation. GalSim's COSMOSCatalog internally stores fluxes
# calibrated to a Roman reference observation (area + exptime), so we need to pass
# those same constants to makeGalaxy() returns correctly scaled chromatic SEDs.
#   roman.collecting_area: 4.54 m² (Roman primary mirror)
#   roman.exptime: 154.77 s (Roman exposure time for COSMOS sample)
cat1 = galsim.COSMOSCatalog(sample='25.2', area=roman.collecting_area, exptime=roman.exptime)
cat2 = galsim.COSMOSCatalog(sample='23.5', area=roman.collecting_area, exptime=roman.exptime)
print('Read in %d galaxies from I<25.2 catalog' % cat1.nobjects)
print('Read in %d galaxies from I<23.5 catalog' % cat2.nobjects)


# Vega SED used for stellar flux normalisation ('vega.txt' ships with GalSim data).
vega_sed = galsim.SED('vega.txt', 'nm', 'flambda')
# Use r-band as the reference passband when computing star fluxes from the
# Vega SED, since it sits near the middle of the MUSE WFM wavelength range.
y_bandpass = bandpasses['r']


# We use a simple linear (no distortion) pixel scale rather than a full WCS
# with RA/Dec, because the pipeline only needs relative pixel positions.
# galsim.PixelScale creates an affine WCS with equal plate scales in x and y.
wcs = galsim.PixelScale(pixel_scale)   # arcsec/pixel


# Atmospheric Point Spread Function
psf = galsim.Kolmogorov(fwhm=seeing_fwhm)   # fwhm in arcsec

# Main rendering loop -- one image per filter band
for ifilter, filter_name in enumerate(filter_names):
    print('Beginning work for MUSE WFM {0}-band.'.format(filter_name))
    bandpass = bandpasses[filter_name]

    # Use a blank 32-bit float image of the full field size.
    full_image = galsim.ImageF(n_pix, n_pix, wcs=wcs)

    # Per-band RNG for noise: offset by ifilter*nobj so each band and object has
    # different noise while also remaining deterministic.
    image_rng = galsim.UniformDeviate(seed1 + ifilter * nobj)

    
    # Draw each simulated object onto the full image
    for i_obj in range(nobj):
        print('Drawing object %d in band %s' % (i_obj, filter_name))

        # Independent RNG for object parameters (position, morphology).
        # The large offset (10**6) separates the obj_rng stream from the
        # phot_rng stream so they never share state.
        obj_rng  = galsim.UniformDeviate(seed + 1 + 10**6 + i_obj)

        # Independent RNG for photon shooting (method='phot'). Separating
        # this from obj_rng means morphology draws don't affect shot noise.
        phot_rng = galsim.UniformDeviate(seed1 + 1 + i_obj + ifilter * nobj)

        # Draw a uniform random number on [0, 1) to decide object type.
        p = obj_rng()

        # Random pixel position within the full image canvas.
        x, y = obj_rng() * n_pix, obj_rng() * n_pix
        image_pos = galsim.PositionD(x, y)

        if p < P_FAINT_GALAXY:
            # ---- Faint background galaxy (80% of objects) ----
            # Drawn from the deep I<25.2 COSMOS catalog, which is dominated
            # by small, faint high-redshift galaxies.
            obj = cat1.makeGalaxy(chromatic=True, gal_type='parametric', rng=obj_rng)
            # Apply a uniformly random position angle on [0, 2π).
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
            obj = galsim.DeltaFunction() * vega_sed.withFlux(flux, y_bandpass)

        else:
            # ---- Bright/extended galaxy (10% of objects) ----
            # Drawn from the shallower I<23.5 COSMOS catalog, which contains
            # more luminous, physically larger galaxies.
            obj = cat2.makeGalaxy(chromatic=True, gal_type='parametric', rng=obj_rng)
            obj = (obj.dilate(2) * 4).rotate(obj_rng() * 2 * np.pi * galsim.radians)

        # Convolve galaxy/star with the atmospheric PSF.
        final = galsim.Convolve(obj, psf)

        # Draw the object onto a small cutout image just large enough to contain it, 
        # then add that cutout into the full image. This is cheaper than drawing each 
        # object directly onto the full canvas. Objects near the edge may hang outside 
        # the canvas, so we clip to the overlap.
        stamp = final.drawImage(
            bandpass, center=image_pos, wcs=wcs.local(image_pos), method='phot', rng=phot_rng
        )
        bounds = stamp.bounds & full_image.bounds
        full_image[bounds] += stamp[bounds]

    print('All objects drawn for %s-band. Adding noise.' % filter_name)

    
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

    
    # Write output
    outfile = os.path.join(OUTPATH, 'muse_wfm_{0}.fits'.format(filter_name))
    full_image.write(outfile)
    print('Wrote %s' % outfile)


# Visualisation hints
print('Done. View with:')
# ds9 can display an RGB composite by assigning i→red, r→green, g→blue,
# which approximates a natural colour rendering of the MUSE wavelength range.
print('ds9 -zoom 2 -scale limits -10 100 -rgb '
      '-red output/muse_wfm_i.fits '
      '-green output/muse_wfm_r.fits '
      '-blue output/muse_wfm_g.fits')
print('Or in CARTA:')
print('carta output/muse_wfm_i.fits output/muse_wfm_r.fits output/muse_wfm_g.fits')
