"""
Tests for utils.py

Covers:
  - pad_to_size
  - prepare_images
  - match_template_subpixel
  - load_fits (file-not-found path)
"""

import tempfile

import numpy as np
import pytest
from astropy.io import fits


import utils


# Fixtures

def _gaussian_blob(shape, cy, cx, sigma=3.0, amplitude=1.0):
    """Return a 2-D array with a single Gaussian blob at (cy, cx)."""
    H, W = shape
    ys, xs = np.ogrid[:H, :W]
    return amplitude * np.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * sigma ** 2))


def _random_haystack(H=100, W=100, seed=42):
    """Random uniform noise; gives each sub-region a unique fingerprint for NCC."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 1, (H, W)).astype(np.float64)


def _extract_needle(haystack, cy, cx, nh, nw):
    """Extract a nh×nw needle centred at (cy, cx) from haystack."""
    y0, x0 = cy - nh // 2, cx - nw // 2
    return haystack[y0:y0 + nh, x0:x0 + nw].copy()


@pytest.fixture
def small_haystack():
    rng = np.random.default_rng(0)
    img = _gaussian_blob((64, 64), cy=64 // 3, cx=64 // 2, sigma=5)
    img += _gaussian_blob((64, 64), cy=2 * 64 // 3, cx=64 // 4, sigma=4)
    img += rng.normal(0, 0.05, (64, 64))
    return img.astype(np.float64)


@pytest.fixture
def small_needle(small_haystack):
    """A 20×20 needle cut from the centre of the 64×64 haystack."""
    return small_haystack[22:42, 22:42].copy()


# pad_to_size

class TestPadToSize:
    """
    Tests for pad_to_size.

    Verifies that a smaller image is correctly embedded in a larger canvas:
    the output has the requested shape, the original pixels land in the centre,
    and the border is filled with the image mean.
    """

    def test_output_shape(self):
        """Output array must have exactly the requested (H, W) shape."""
        img = np.ones((10, 10))
        result = utils.pad_to_size(img, 30, 40)
        assert result.shape == (30, 40)

    def test_image_content_centered(self):
        """Original pixels must appear centred inside the padded canvas."""
        img = np.full((4, 4), 2.0)
        result = utils.pad_to_size(img, 12, 12)
        y0 = (12 - 4) // 2
        x0 = (12 - 4) // 2
        np.testing.assert_array_almost_equal(result[y0:y0+4, x0:x0+4], img)

    def test_border_fill_equals_image_mean(self):
        """Border pixels outside the embedded image must equal the image mean."""
        rng = np.random.default_rng(42)
        img = rng.uniform(0.3, 0.7, (8, 8))
        H, W = 30, 30
        result = utils.pad_to_size(img, H, W)
        # Corners are outside the embedded image; should equal img.mean()
        mean_val = img.mean()
        assert result[0, 0] == pytest.approx(mean_val)
        assert result[0, W - 1] == pytest.approx(mean_val)
        assert result[H - 1, 0] == pytest.approx(mean_val)

    def test_same_size_is_copy(self):
        """Padding an image to its own size must return an identical array."""
        img = np.eye(8)
        result = utils.pad_to_size(img, 8, 8)
        np.testing.assert_array_equal(result, img)

    def test_uniform_image_entire_canvas_equals_fill(self):
        """When the image is uniform, the whole canvas should be that value."""
        img = np.full((6, 6), 0.5)
        result = utils.pad_to_size(img, 20, 20)
        np.testing.assert_array_almost_equal(result, 0.5)


# prepare_images

class TestPrepareImages:
    """
    Tests for prepare_images.

    Verifies that after preparation both images share the haystack shape,
    contain no NaN/Inf values, and are float arrays -- preconditions for NCC.
    """

    def test_haystack_shape_preserved(self, small_haystack, small_needle):
        """Haystack output shape must match the input haystack shape."""
        H, W = small_haystack.shape
        h_out, n_out = utils.prepare_images(small_haystack, small_needle)
        assert h_out.shape == (H, W)

    def test_needle_padded_to_haystack_shape(self, small_haystack, small_needle):
        """Needle must be padded to the same shape as the haystack."""
        H, W = small_haystack.shape
        h_out, n_out = utils.prepare_images(small_haystack, small_needle)
        assert n_out.shape == (H, W)

    def test_no_nans_or_infs(self, small_haystack, small_needle):
        """Both outputs must be finite; no NaN or Inf values."""
        h_out, n_out = utils.prepare_images(small_haystack, small_needle)
        assert np.all(np.isfinite(h_out)), "haystack output contains NaN or Inf"
        assert np.all(np.isfinite(n_out)), "needle output contains NaN or Inf"

    def test_outputs_are_float(self, small_haystack, small_needle):
        """Both outputs must have a floating-point dtype."""
        h_out, n_out = utils.prepare_images(small_haystack, small_needle)
        assert np.issubdtype(h_out.dtype, np.floating)
        assert np.issubdtype(n_out.dtype, np.floating)

    def test_uniform_haystack_does_not_crash(self):
        """Wiener filter on a flat image should not raise."""
        haystack = np.ones((32, 32))
        needle = np.ones((10, 10))
        h_out, n_out = utils.prepare_images(haystack, needle)
        assert h_out.shape == (32, 32)


# Match_template_subpixel

class TestMatchTemplateSubpixel:
    """
    Tests for match_template_subpixel.

    Strategy: build a random-noise haystack, extract a needle at a known 
    integer position, and verify the detected shift equals (cx - W/2, cy - H/2).

    A single smooth Gaussian blob is unsuitable here because the NCC surface
    is nearly flat and the argmax drifts to the image boundary.
    """

    # Needle size used across all shift tests (even, so bias correction is exercised)
    NH, NW = 30, 30

    def _run(self, H, W, true_shift_x, true_shift_y, nh=None, nw=None):
        """Build a haystack, plant a needle at the given shift, and run matching."""
        nh = nh or self.NH
        nw = nw or self.NW
        haystack = _random_haystack(H, W)
        cx = W // 2 + true_shift_x
        cy = H // 2 + true_shift_y
        needle = _extract_needle(haystack, cy, cx, nh, nw)
        shift_x, shift_y, corr = utils.match_template_subpixel(haystack, needle)
        return shift_x, shift_y, corr

    def test_zero_shift(self):
        """Needle centred in the haystack should produce a near-zero shift."""
        sx, sy, _ = self._run(100, 100, 0, 0)
        assert abs(sx) < 1.0
        assert abs(sy) < 1.0

    def test_positive_x_shift(self):
        """Needle displaced +10 px in x should be detected within 1 px."""
        sx, sy, _ = self._run(100, 100, 10, 0)
        assert abs(sx - 10) < 1.0

    def test_negative_x_shift(self):
        """Needle displaced −8 px in x should be detected within 1 px."""
        sx, sy, _ = self._run(100, 100, -8, 0)
        assert abs(sx - (-8)) < 1.0

    def test_positive_y_shift(self):
        """Needle displaced +8 px in y should be detected within 1 px."""
        sx, sy, _ = self._run(100, 100, 0, 8)
        assert abs(sy - 8) < 1.0

    def test_negative_y_shift(self):
        """Needle displaced −8 px in y should be detected within 1 px."""
        sx, sy, _ = self._run(100, 100, 0, -8)
        assert abs(sy - (-8)) < 1.0

    def test_diagonal_shift(self):
        """Needle displaced diagonally (+12, −10) should be detected within 1 px."""
        sx, sy, _ = self._run(120, 120, 12, -10)
        assert abs(sx - 12) < 1.0
        assert abs(sy - (-10)) < 1.0

    def test_returns_correlation_map_correct_shape(self):
        """Returned correlation map must have the same shape as the haystack."""
        H, W = 100, 100
        haystack = _random_haystack(H, W)
        needle = _extract_needle(haystack, 50, 50, self.NH, self.NW)
        _, _, corr = utils.match_template_subpixel(haystack, needle)
        assert corr.shape == (H, W)

    def test_correlation_map_peak_is_finite(self):
        """Every value in the correlation map must be finite."""
        H, W = 100, 100
        haystack = _random_haystack(H, W)
        needle = _extract_needle(haystack, 50, 50, self.NH, self.NW)
        _, _, corr = utils.match_template_subpixel(haystack, needle)
        assert np.isfinite(corr).all()

    def test_odd_template_size_bias_zero(self):
        """Odd-sized needle: x_bias and y_bias should both be 0.0."""
        sx, sy, _ = self._run(101, 101, 5, 3, nh=21, nw=21)
        assert abs(sx - 5) < 1.0
        assert abs(sy - 3) < 1.0

    def test_even_template_size_bias_half(self):
        """Even-sized needle: 0.5 px bias correction is applied internally."""
        sx, sy, _ = self._run(100, 100, 8, -8, nh=20, nw=20)
        assert abs(sx - 8) < 1.0
        assert abs(sy - (-8)) < 1.0

    def test_shift_sign_consistency(self):
        """Shift sign must be consistent: positive shift_x → needle right of centre."""
        sx, _, _ = self._run(100, 100, 15, 0)
        assert sx > 0

    def test_correlation_peak_near_known_position(self):
        """The argmax of the raw correlation map should be close to the true location."""
        H, W = 100, 100
        true_sx, true_sy = 12, -7
        cx = W // 2 + true_sx
        cy = H // 2 + true_sy
        haystack = _random_haystack(H, W)
        needle = _extract_needle(haystack, cy, cx, self.NH, self.NW)
        _, _, corr = utils.match_template_subpixel(haystack, needle)
        peak_yi, peak_xi = np.unravel_index(np.argmax(corr), corr.shape)
        # Raw argmax should be within 2 px of the true centre
        assert abs(peak_xi - cx) < 2
        assert abs(peak_yi - cy) < 2


# load_fits

class TestLoadFits:
    """
    Tests for load_fits.

    Covers the two failure modes (missing haystack, missing needle) and the
    happy path where both files exist and contain valid image data.
    """

    def test_missing_haystack_returns_none_tuple(self, tmp_path):
        """If the haystack file does not exist, all four return values are None."""
        result = utils.load_fits(
            str(tmp_path / "nonexistent_haystack.fits"),
            str(tmp_path / "nonexistent_needle.fits"),
        )
        assert result == (None, None, None, None)

    def test_loads_real_fits_files(self, tmp_path):
        """Valid FITS files must load correctly with matching pixel values."""
        data = np.ones((10, 10), dtype=np.float32)
        hdr = fits.Header()
        hdr["SIMPLE"] = True

        h_path = str(tmp_path / "haystack.fits")
        n_path = str(tmp_path / "needle.fits")
        fits.writeto(h_path, data * 2, hdr)
        fits.writeto(n_path, data * 3, hdr)

        h, hh, n, nh = utils.load_fits(h_path, n_path)
        assert h is not None
        assert n is not None
        assert h.shape == (10, 10)
        assert n.shape == (10, 10)
        np.testing.assert_array_almost_equal(h, 2.0)
        np.testing.assert_array_almost_equal(n, 3.0)

    def test_missing_needle_returns_none_tuple(self, tmp_path):
        """If the needle file is missing, all four return values are None."""
        data = np.ones((10, 10), dtype=np.float32)
        h_path = str(tmp_path / "haystack.fits")
        fits.writeto(h_path, data, fits.Header())
        result = utils.load_fits(h_path, str(tmp_path / "no_needle.fits"))
        assert result == (None, None, None, None)
