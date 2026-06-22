"""
realtime/channelizer.py
-----------------------
Polyphase DFT analysis filterbank (maximally-decimated, and 2x-oversampled).

Splits a wideband complex IQ stream into num_subbands equal sub-bands with a
single FFT per output block.

Math summary
~~~~~~~~~~~~
Prototype lowpass h of length N*M is reshaped into N polyphase branches:
    poly[k, m] = h[m*N + k],  k=0..N-1, m=0..M-1

For each block of N input samples the k-th branch accumulates the dot product
of its M taps against the M most-recent block-rows for path k.  An N-point FFT
across the N branch outputs then yields the N sub-band samples for that block.

fftshift is applied so output row 0 corresponds to the most-negative sub-band
centre (-fs/2), ascending to +fs/2*(1-1/N) for row N-1.

Convention settled on (see task-1-brief.md Note):
    * scipy.fft.fft  (not ifft) — matches the analysis direction for this
      commutator/path ordering.
    * Paths loaded in natural order (X[:, k] = x[r*N + k]).
    * subband_centers uses np.fft.fftshift on k*(fs/N) wrapped to [-fs/2, fs/2).
"""

import numpy as np
from scipy.signal import firwin
from scipy.fft import fft


class PolyphaseChannelizer:
    """Maximally-decimated (critically-sampled, oversample=1) polyphase DFT
    analysis filterbank.  Splits a wideband stream into num_subbands equal,
    overlapping baseband sub-bands in one FFT per output block.

    Parameters
    ----------
    sample_rate : float
        Input sample rate in Hz.
    num_subbands : int
        Number of sub-bands N (must be even).  Output row i is the i-th
        sub-band in ascending frequency order.
    taps_per_phase : int
        Number of taps M per polyphase branch; prototype filter length = N*M.
    oversample : int
        1 → critically sampled (implemented here).
        2 → 2x oversampled (Task 2; raises NotImplementedError for now).
    """

    def __init__(self, sample_rate: float, num_subbands: int = 32,
                 taps_per_phase: int = 12, oversample: int = 2):
        self.fs = float(sample_rate)
        self.N = int(num_subbands)
        self.M = int(taps_per_phase)
        self.oversample = int(oversample)

        # Prototype lowpass: normalised cutoff = 1/N in units of Nyquist.
        # This makes the per-sub-band passband ≈ the owning half-width fs/(2N)
        # for BOTH the critically-sampled and the 2x-oversampled paths.  The
        # oversampling factor changes the COMMUTATOR HOP (N vs N/2) and the
        # output rate, NOT the prototype bandwidth — using 2/N here was the
        # cause of the decimation-aliasing phantoms (no stopband before the
        # fold frequency).  Clamped to 0.99 to keep firwin well-conditioned.
        cutoff = min(0.99, 1.0 / self.N)
        proto = firwin(self.N * self.M, cutoff).astype(np.float64)

        # Flat prototype (length N*M) for the WOLA oversampled path.
        self.proto = proto
        # poly[k, m] = proto[m*N + k]  →  shape (N, M); used by critical path.
        self.poly = proto.reshape(self.M, self.N).T.copy()

        # Sub-band output rate
        self.subband_rate: float = self.fs / self.N * self.oversample

        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all streaming state (overlap buffer and polyphase memory)."""
        # Leftover samples that did not fill a complete block of N
        self._tail = np.zeros(0, dtype=np.complex128)
        # Previous (M-1) block-rows, one row per block of N samples processed
        self._state = np.zeros((self.M - 1, self.N), dtype=np.complex128)
        # WOLA oversampled path: leftover samples that did not fill a frame,
        # and a global frame counter for the circular-shift origin continuity.
        self._buf_os = np.zeros(0, dtype=np.complex128)
        self._frame_count_os = 0

    def subband_centers(self) -> np.ndarray:
        """Return sub-band centre frequencies in Hz, ascending, length N.

        Channel k (0-based, natural FFT order) has centre k*(fs/N).
        fftshift maps this to ascending order [-fs/2, -fs/2+fs/N, ..., fs/2-fs/N].
        """
        k = np.arange(self.N)
        centers = k * (self.fs / self.N)
        # Wrap k >= N/2 to negative frequencies
        centers = np.where(centers >= self.fs / 2, centers - self.fs, centers)
        return np.fft.fftshift(centers)

    def process(self, chunk: np.ndarray) -> np.ndarray:
        """Filter and channelise a chunk of wideband IQ samples.

        Parameters
        ----------
        chunk : np.ndarray, complex
            1-D array of input samples (any length).

        Returns
        -------
        np.ndarray, complex64, shape (num_subbands, n_out)
            Row i is the baseband IQ of the i-th sub-band (ascending freq).
            n_out = len(chunk) // N  (critically sampled, oversample=1),
            or approx 2*len(chunk) // N  (oversample=2).
        """
        x = np.asarray(chunk, dtype=np.complex128)
        if self.oversample == 1:
            return self._process_critical(x)
        if self.oversample == 2:
            return self._process_oversampled(x)
        raise ValueError(f"unsupported oversample={self.oversample}")

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------

    def _process_critical(self, x: np.ndarray) -> np.ndarray:
        """Critically-sampled (oversample=1) polyphase DFT filterbank."""
        N, M = self.N, self.M

        # Prepend leftover from previous call
        x = np.concatenate([self._tail, x])
        nblocks = len(x) // N
        self._tail = x[nblocks * N:].copy()

        if nblocks == 0:
            return np.zeros((N, 0), dtype=np.complex64)

        # X[r, k] = x[r*N + k]  →  shape (nblocks, N)
        X = x[:nblocks * N].reshape(nblocks, N)

        # Stack previous state rows above current blocks for the FIR dot product
        Xs = np.vstack([self._state, X])   # shape (M-1+nblocks, N)

        # Polyphase FIR: for output block r, accumulate M taps
        #   F[r, k] = sum_{m=0}^{M-1}  poly[k, m] * Xs[r + (M-1-m), k]
        F = np.zeros((nblocks, N), dtype=np.complex128)
        for m in range(M):
            F += self.poly[:, m][None, :] * Xs[(M - 1 - m):(M - 1 - m) + nblocks, :]

        # Save the last (M-1) rows for next call
        if M > 1:
            self._state = Xs[-(M - 1):, :].copy()

        # N-point FFT across polyphase branches → N sub-band samples per block
        # fft (analysis) convention matches the commutator/path order above.
        Y = fft(F, axis=1)                  # (nblocks, N)

        # fftshift: map to ascending frequency order
        Y = np.fft.fftshift(Y, axes=1)     # (nblocks, N)

        return Y.T.astype(np.complex64)    # (N, nblocks)

    def _process_oversampled(self, x: np.ndarray) -> np.ndarray:
        """2x oversampled (WOLA) polyphase DFT analysis filterbank.

        Weighted OverLap-Add structure (the standard correct construction):
        for each frame (hop H = N/2) take L = N*M samples, weight by the flat
        prototype, fold the M length-N segments by summation, apply a circular
        shift to track the sliding window origin, then an N-point FFT yields the
        N sub-band samples for that frame.

            frame_p = x[p*H : p*H + L] * proto      # weight
            u_p     = sum_{m=0}^{M-1} frame_p[m*N : (m+1)*N]   # fold to length N
            u_p     = roll(u_p, -((p*H) mod N))     # sliding-origin alignment
            X_p     = fft(u_p)                       # N channels

        Output rate = fs/N * 2 (hop is N/2).  Channel order is fftshifted to
        ascending frequency, matching the critical path and subband_centers().

        The earlier implementation fed N/2-stepped overlapping block-rows into
        an N-spaced polyphase dot product (a mismatched stride), which produced
        decimation-alias images and >0 dB out-of-band gain.  This WOLA form is
        the correct oversampled analysis bank and rejects those aliases.
        """
        N, M = self.N, self.M
        L = N * M
        H = N // 2  # hop = N/2 for 2x oversampling

        # Prepend leftover from previous call
        x = np.concatenate([self._buf_os, x])
        nframes = (len(x) - L) // H + 1 if len(x) >= L else 0
        if nframes <= 0:
            self._buf_os = x.copy()
            return np.zeros((N, 0), dtype=np.complex64)

        # Build the folded length-N frames WITHOUT materialising the full
        # (nframes, L) matrix — that would be M times larger and OOMs on a
        # whole-capture channelize.  Accumulate over the M prototype segments:
        #   u[p, :] = sum_{m} proto[m*N:(m+1)*N] * x[p*H + m*N : p*H + m*N + N]
        # Each segment contributes an (nframes, N) view, so peak memory is
        # O(nframes*N) instead of O(nframes*L).
        base = H * np.arange(nframes)[:, None]            # (nframes, 1)
        cols = np.arange(N)[None, :]                       # (1, N)
        u = np.zeros((nframes, N), dtype=np.complex128)
        for m in range(M):
            seg = x[base + cols + m * N]                   # (nframes, N)
            u += self.proto[m * N:(m + 1) * N][None, :] * seg

        # Sliding-origin circular shift: frame p starts at sample p*H, so the
        # DFT origin must be rotated by -((p*H) mod N) to keep phase continuous.
        f = self._frame_count_os + np.arange(nframes)
        shifts = (f * H) % N
        # Apply per-row circular shift via advanced indexing
        col = (np.arange(N)[None, :] + shifts[:, None]) % N
        u = np.take_along_axis(u, col, axis=1)

        # N-point FFT across the folded branches → N sub-band samples per frame
        Y = fft(u, axis=1)                               # (nframes, N)

        # Advance streaming state
        self._frame_count_os += nframes
        self._buf_os = x[nframes * H:].copy()

        # fftshift: map to ascending frequency order
        Y = np.fft.fftshift(Y, axes=1)                   # (nframes, N)

        return Y.T.astype(np.complex64)                  # (N, nframes)
