from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.io import wavfile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.dsp import read_rawiq
from dpmr.decoder import decode, filter_stable_pdus
from dpmr.dsp import frontend_dpmr


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _write_disc_wavs(rawiq_path: str, tmpdir: Path) -> tuple[Path, Path]:
    y = frontend_dpmr(read_rawiq(rawiq_path))
    y = y - np.median(y)
    scale = np.percentile(np.abs(y), 99.5)
    if scale <= 0:
        scale = 1.0
    samples = np.clip(y / scale * 26000, -32767, 32767).astype(np.int16)
    norm_path = tmpdir / "dpmr_disc_norm.wav"
    inv_path = tmpdir / "dpmr_disc_inv.wav"
    wavfile.write(norm_path, 48000, samples)
    wavfile.write(inv_path, 48000, (-samples).astype(np.int16))
    return norm_path, inv_path


def _run_dsd_fme(wav_path: Path, inverted: bool) -> dict:
    cmd = ["dsd-fme", "-fm", "-i", str(wav_path), "-o", "null", "-Z"]
    if inverted:
        cmd.append("-xd")
    proc = subprocess.run(
        ["timeout", "30s", *cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        check=False,
    )
    out = ANSI_RE.sub("", proc.stdout)
    tg = re.findall(r"TG=([^\s]+)", out)
    src = re.findall(r"Src=([^\s]+)", out)
    audio_errors = re.search(r"Total audio errors: (\d+)", out)
    return {
        "sync_plus": out.count("Sync: +dPMR  dPMR Frame Sync 2"),
        "sync_minus": out.count("Sync: -dPMR  dPMR Frame Sync 2"),
        "cc02": out.count("Channel Code=02"),
        "audio_errors": int(audio_errors.group(1)) if audio_errors else None,
        "tg": Counter(tg).most_common(5),
        "src": Counter(src).most_common(5),
    }


def _run_local(rawiq_path: str) -> dict:
    pdus = filter_stable_pdus(decode(frontend_dpmr(read_rawiq(rawiq_path))))
    return {
        "pdus": len(pdus),
        "color": Counter(pdu["extra"].get("color_code") for pdu in pdus).most_common(),
        "polarity": Counter(
            "INV" if pdu["extra"].get("polarity_inverted") else "NORM"
            for pdu in pdus
        ).most_common(),
        "crc": Counter(
            pdu["extra"].get("quality", {}).get("crc_ok_count", 0)
            for pdu in pdus
        ).most_common(),
        "src": Counter(pdu.get("src") or "" for pdu in pdus).most_common(5),
        "dst": Counter(pdu.get("dst") or "" for pdu in pdus).most_common(5),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare local dPMR decode with DSD-FME.")
    parser.add_argument("rawiq", help="int16 interleaved IQ file at 48 kHz")
    args = parser.parse_args()

    if shutil.which("dsd-fme") is None:
        raise SystemExit("dsd-fme not found in PATH")

    with tempfile.TemporaryDirectory(prefix="dpmr_dsd_fme_") as tmp:
        norm_wav, inv_wav = _write_disc_wavs(args.rawiq, Path(tmp))
        rows = [
            ("norm", "no-xd", _run_dsd_fme(norm_wav, False)),
            ("norm", "-xd", _run_dsd_fme(norm_wav, True)),
            ("inv", "no-xd", _run_dsd_fme(inv_wav, False)),
            ("inv", "-xd", _run_dsd_fme(inv_wav, True)),
        ]

    print("DSD-FME")
    for audio, mode, row in rows:
        print(
            f"{audio:>4} {mode:>5} "
            f"sync+={row['sync_plus']:3d} sync-={row['sync_minus']:3d} "
            f"cc02={row['cc02']:3d} audio_err={row['audio_errors']}"
        )
        print(f"           TG {row['tg']}")
        print(f"          SRC {row['src']}")

    local = _run_local(args.rawiq)
    print("Local")
    print(f"  pdus={local['pdus']} color={local['color']} polarity={local['polarity']}")
    print(f"  crc={local['crc']}")
    print(f"  src={local['src']}")
    print(f"  dst={local['dst']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
