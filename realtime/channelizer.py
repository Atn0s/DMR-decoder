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

        # Prototype lowpass: normalised cutoff = 1/N (critically sampled)
        # or 2/N (2x oversampled) in units of Nyquist.
        # Clamped to 0.99 to keep firwin well-conditioned.
        cutoff = min(0.99, self.oversample / self.N)
        proto = firwin(self.N * self.M, cutoff).astype(np.float64)

        # poly[k, m] = proto[m*N + k]  →  shape (N, M)
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
        # State for oversampled path (same shape, separate buffer)
        self._state_os = np.zeros((self.M - 1, self.N), dtype=np.complex128)

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
        """2x oversampled (WOLA) polyphase DFT filterbank.

        Commutator steps by H = N/2 (50% overlap) so each output block
        sees a 50%-overlapping window of input.  A per-block WOLA phase
        correction rotates channel k of block r by exp(-1j*pi*k*r), which
        (under Task 1's fft convention) places a tone on the sub-band
        boundary into BOTH adjacent channels with equal energy.

        Convention note: Task 1 uses fft (analysis), no *N scaling.  The
        brief's snippet uses ifft*N; since we stay with fft we negate the
        exponent sign relative to the brief's exp(+1j*pi*k*r).
        """
        N, M = self.N, self.M
        H = N // 2  # commutator hop = N/2 for 2x oversampling

        # Prepend leftover from previous call
        x = np.concatenate([self._tail, x])
        nblocks = (len(x) - N) // H + 1 if len(x) >= N else 0
        if nblocks <= 0:
            self._tail = x.copy()
            return np.zeros((N, 0), dtype=np.complex64)

        consumed = nblocks * H
        self._tail = x[consumed:].copy()

        # Build overlapping blocks of length N stepped by H: shape (nblocks, N)
        idx = np.arange(N)[None, :] + H * np.arange(nblocks)[:, None]
        B = x[idx]   # (nblocks, N)

        # Stack previous (M-1) state rows above current blocks for FIR
        Xs = np.vstack([self._state_os, B])   # (M-1+nblocks, N)

        # Polyphase FIR: same dot-product as _process_critical
        #   F[r, k] = sum_{m=0}^{M-1}  poly[k, m] * Xs[r + (M-1-m), k]
        F = np.zeros((nblocks, N), dtype=np.complex128)
        for m in range(M):
            F += self.poly[:, m][None, :] * Xs[(M - 1 - m):(M - 1 - m) + nblocks, :]

        # Save last (M-1) rows for next call
        if M > 1:
            self._state_os = Xs[-(M - 1):, :].copy()

        # N-point FFT — same fft (analysis) convention as _process_critical
        Y = fft(F, axis=1)   # (nblocks, N)

        # WOLA phase correction: block r stepped by H=N/2 acquires a
        # linear phase ramp across channels.  Negated sign vs. brief's ifft
        # snippet because we use fft (analysis direction).
        r = np.arange(nblocks)[:, None]
        k = np.arange(N)[None, :]
        Y = Y * np.exp(-1j * np.pi * k * r)   # (nblocks, N)

        # fftshift: map to ascending frequency order
        Y = np.fft.fftshift(Y, axes=1)   # (nblocks, N)

        return Y.T.astype(np.complex64)   # (N, nblocks)
