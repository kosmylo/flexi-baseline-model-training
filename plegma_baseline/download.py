"""Download and extract the Plegma clean archive (CC BY 4.0, University of Strathclyde)."""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

import requests

from .prepare import find_house_dirs

log = logging.getLogger(__name__)
SEVEN_ZIP_MAGIC = b"7z\xbc\xaf\x27\x1c"
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) plegma-baseline/0.1"}
REPORT_EVERY = 25 * 2**20


def download_archive(url: str, dest: Path, chunk: int = 2**20) -> Path:
    """Stream the archive to disk and resume a partial download when the server allows it."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    done = part.stat().st_size if part.exists() else 0
    headers = dict(HEADERS)
    if done:
        headers["Range"] = f"bytes={done}-"
    with requests.get(url, headers=headers, stream=True, timeout=(30, 300), allow_redirects=True) as r:
        if r.status_code != 416:  # 416: the partial file is already complete
            r.raise_for_status()
            if "text/html" in r.headers.get("Content-Type", ""):
                raise RuntimeError(f"The server returned a web page instead of the archive ({r.url}).")
            append = done > 0 and r.status_code == 206
            written = done if append else 0
            total = int(r.headers.get("Content-Length", 0)) + written
            next_report = written + REPORT_EVERY
            with open(part, "ab" if append else "wb") as fh:
                for block in r.iter_content(chunk_size=chunk):
                    fh.write(block)
                    written += len(block)
                    if written >= next_report:
                        log.info("downloaded %.0f of %.0f MB", written / 2**20, total / 2**20)
                        next_report += REPORT_EVERY
    with open(part, "rb") as fh:
        if fh.read(6) != SEVEN_ZIP_MAGIC:
            raise RuntimeError(f"{part} is not a 7z archive. Download it from the landing page and pass --archive.")
    os.replace(part, dest)
    log.info("saved %s (%.0f MB)", dest, dest.stat().st_size / 2**20)
    return dest


def extract_archive(archive: Path, out_dir: Path) -> None:
    """Extract with py7zr, or with a 7-Zip binary when py7zr cannot handle the archive."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import py7zr

        with py7zr.SevenZipFile(archive, mode="r") as z:
            z.extractall(path=out_dir)
        return
    except Exception as exc:
        error = exc
    exe = shutil.which("7z") or shutil.which("7za") or shutil.which("7zz")
    if not exe:
        raise RuntimeError(f"py7zr could not extract the archive ({error}). Install 7-Zip (p7zip-full) and retry.")
    log.info("py7zr failed (%s), extracting with %s", error, exe)
    subprocess.run([exe, "x", "-y", f"-o{out_dir}", str(archive)], check=True)


def find_clean_root(base: Path) -> Path:
    houses = find_house_dirs(base)
    if not houses:
        raise FileNotFoundError(f"No House_<i>/Electric_data folders under {base}")
    return Path(os.path.commonpath([str(p.parent) for p in houses.values()]))


def run_download(cfg: dict, archive: str | None = None) -> Path:
    raw_dir = Path(cfg["paths"]["raw_dir"])
    name = cfg["download"]["archive_name"]
    if archive and not Path(archive).exists():
        raise SystemExit(f"Archive not found: {archive}")
    candidates = [Path(archive)] if archive else [raw_dir / name, Path.cwd() / name, Path.home() / "Downloads" / name]
    arc = next((p for p in candidates if p.exists()), candidates[0])
    if not arc.exists():
        log.info("downloading %s", cfg["download"]["url"])
        try:
            download_archive(cfg["download"]["url"], arc)
        except requests.RequestException as exc:
            resp = getattr(exc, "response", None)
            reason = f"HTTP {resp.status_code}" if resp is not None else type(exc).__name__
            raise SystemExit(
                f"The scripted download failed ({reason}).\n"
                f"Download {name} in a browser from {cfg['download']['landing_page']}, then run\n"
                f"  python -m plegma_baseline download --archive /path/to/{name}\n"
                f"or copy the file to {raw_dir / name} and run the download command again.") from None
    log.info("using archive %s", arc)
    out = raw_dir / "extracted"
    if not (out.exists() and find_house_dirs(out)):
        log.info("extracting %s", arc)
        extract_archive(arc, out)
    root = find_clean_root(out)
    (raw_dir / "clean_root.txt").write_text(str(root), encoding="utf-8")
    log.info("clean dataset root: %s (%d houses)", root, len(find_house_dirs(root)))
    return root
