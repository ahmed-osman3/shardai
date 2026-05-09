"""Mock object storage (S3/R2 interface).

Copies files to a local directory and returns file:// URLs.
Lets pipeline code call upload/download without real cloud credentials.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


class MockStorage:
    """Local-filesystem stand-in for S3/R2 object storage.

    Files are stored under base_dir/{key}. All upload/download/signed_url
    calls operate on this local tree. Replace with a real boto3/cloudflare
    client when credentials are available.

    Args:
        base_dir: Root directory for mock storage (e.g. data/outputs/mock_storage/).
    """

    def __init__(self, base_dir: Path) -> None:
        self._base = Path(base_dir)
        self._base.mkdir(parents=True, exist_ok=True)

    def upload(self, local_path: Path, key: str) -> str:
        """Copy a local file into mock storage.

        Args:
            local_path: Source file on disk.
            key: Storage key (used as relative path under base_dir).

        Returns:
            file:// URL pointing to the stored file.
        """
        dest = self._base / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, dest)
        logger.debug("MockStorage: uploaded %s → %s", local_path, dest)
        return f"file://{dest.resolve()}"

    def download(self, key: str, local_path: Path) -> None:
        """Copy a file from mock storage to a local path.

        Args:
            key: Storage key of the file to retrieve.
            local_path: Destination path on disk.
        """
        src = self._base / key
        shutil.copy2(src, local_path)
        logger.debug("MockStorage: downloaded %s → %s", src, local_path)

    def signed_url(self, key: str, ttl_seconds: int = 3600) -> str:
        """Return a URL for direct access to a stored file.

        Args:
            key: Storage key.
            ttl_seconds: Expiry window (ignored in mock — always returns permanent URL).

        Returns:
            file:// URL pointing to the stored file.
        """
        return f"file://{(self._base / key).resolve()}"
