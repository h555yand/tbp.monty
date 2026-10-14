# Copyright 2026 [Aleksey Isaev]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Auto-download pre-trained hybrid_rl models from GitHub Releases."""

import logging
import os
import tarfile
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

MODELS_URL = (
    "https://github.com/h555yand/hybrid_rl/releases/"
    "download/v0.1.0/models_v0.1.0.zip"
)

DEFAULT_MODELS_DIR = Path(__file__).parent / "models"


def ensure_models(
    models_dir: Path = DEFAULT_MODELS_DIR,
    url: str = MODELS_URL,
) -> tuple:
    """Download and extract pre-trained models if not present.

    Supports both .tar.gz and .zip archives (auto-detected).

    Args:
        models_dir: Directory to store models.
        url: URL to download models archive.

    Returns:
        Tuple of (q_store_path, sac_path).
    """
    models_dir = Path(models_dir)
    q_store_dir = models_dir / "q_store"
    sac_dir = models_dir / "sac"

    if (q_store_dir / "config.json").exists():
        logger.info("Models found at %s", models_dir)
        return str(q_store_dir), str(sac_dir)

    logger.info("Models not found at %s", models_dir)
    logger.info("Downloading from %s ...", url)

    models_dir.mkdir(parents=True, exist_ok=True)
    archive_path = models_dir / "models_archive"

    try:
        urllib.request.urlretrieve(url, archive_path)
        logger.info(
            "Download complete: %.1f MB",
            archive_path.stat().st_size / 1e6,
        )
    except Exception as e:
        logger.error("Download failed: %s", e)
        if archive_path.exists():
            archive_path.unlink()
        raise RuntimeError(
            f"Failed to download models from {url}. "
            f"Check your internet connection or download manually:\n"
            f"  wget {url} -O /tmp/models_archive\n"
            f"  Extract to {models_dir}"
        ) from e

    try:
        import zipfile

        if zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(models_dir)
            logger.info("Extracted zip to %s", models_dir)
        else:
            with tarfile.open(archive_path, "r:*") as tar:
                tar.extractall(models_dir)
            logger.info("Extracted tar to %s", models_dir)
    finally:
        if archive_path.exists():
            archive_path.unlink()

    if not (q_store_dir / "config.json").exists():
        contents = list(models_dir.iterdir())
        raise RuntimeError(
            f"Models extracted but q_store/config.json not found. "
            f"Contents of {models_dir}: {[p.name for p in contents]}"
        )

    logger.info("Models ready: q_store=%s, sac=%s", q_store_dir, sac_dir)
    return str(q_store_dir), str(sac_dir)
