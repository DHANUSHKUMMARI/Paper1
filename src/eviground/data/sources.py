"""Network sources (Kaggle, Hugging Face) behind one small interface.

:class:`NetworkSources` is the only code that talks to the network. Tests and dry runs use
:class:`eviground.data.fake_sources.FakeSources`, which serves small synthetic archives with
the same methods. Credentials are read inside each call from Colab Secrets (or the
environment) and are never logged, returned or written to disk.
"""

from __future__ import annotations

import base64
import logging
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from eviground.env import MissingSecretError, get_secret

log = logging.getLogger(__name__)

KAGGLE_DOWNLOAD = "https://www.kaggle.com/api/v1/datasets/download/{slug}"
CHUNK = 8 << 20

# before_write(bytes_still_to_download, total_bytes) is called once the size is known and
# before anything is written; it raises to stop (e.g. InsufficientSpaceError).
SpaceCheck = Callable[[int, int], None]


class Sources(Protocol):
    def kaggle_download(self, slug: str, dest: Path, before_write: SpaceCheck) -> int: ...

    def hf_file_info(self, repo: str, filenames: list[str]) -> dict[str, dict[str, Any]]: ...

    def hf_list_files(self, repo: str) -> dict[str, int]: ...

    def hf_download(self, repo: str, filename: str, dest_dir: Path) -> Path: ...

    def hf_parquet_column(self, repo: str, filename: str, column: str) -> list[Any]: ...

    def url_download(self, url: str, dest: Path, before_write: SpaceCheck) -> int: ...


def kaggle_auth_header() -> str:
    """Authorization header value from KAGGLE_API_TOKEN, or KAGGLE_USERNAME + KAGGLE_KEY."""
    try:
        return f"Bearer {get_secret('KAGGLE_API_TOKEN')}"
    except MissingSecretError:
        pass
    try:
        user, key = get_secret("KAGGLE_USERNAME"), get_secret("KAGGLE_KEY")
    except MissingSecretError:
        raise MissingSecretError(
            "Kaggle credentials are not set. Add KAGGLE_API_TOKEN (new-style token), or "
            "KAGGLE_USERNAME and KAGGLE_KEY (legacy API key), in the Colab Secrets panel with "
            "notebook access enabled."
        ) from None
    return "Basic " + base64.b64encode(f"{user}:{key}".encode()).decode()


def optional_hf_token() -> str | None:
    """HF_TOKEN if set; public datasets download without one."""
    try:
        return get_secret("HF_TOKEN")
    except MissingSecretError:
        return None


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None  # surface 3xx as HTTPError so the signed URL is fetched without our token


def _open(url: str, headers: dict[str, str], follow: bool) -> Any:
    handlers = [] if follow else [_NoRedirect()]
    return urllib.request.build_opener(*handlers).open(
        urllib.request.Request(url, headers=headers), timeout=60
    )


class NetworkSources:
    """Real Kaggle and Hugging Face access (Colab only)."""

    def __init__(self, retries: int = 5) -> None:
        self.retries = retries

    # -- Kaggle ------------------------------------------------------------------------

    def _kaggle_response(self, slug: str, start: int) -> Any:
        """Open the dataset download at byte ``start``; the token goes to kaggle.com only."""
        rng = {"Range": f"bytes={start}-"} if start else {}
        try:
            return _open(
                KAGGLE_DOWNLOAD.format(slug=slug),
                {"Authorization": kaggle_auth_header(), **rng},
                follow=False,
            )
        except urllib.error.HTTPError as exc:
            if exc.code in (301, 302, 303, 307, 308) and exc.headers.get("Location"):
                return _open(exc.headers["Location"], rng, follow=True)
            if exc.code in (401, 403):
                raise MissingSecretError(
                    f"Kaggle refused the credentials (HTTP {exc.code}). Check KAGGLE_API_TOKEN "
                    "or KAGGLE_USERNAME/KAGGLE_KEY, and accept the dataset's terms on kaggle.com."
                ) from None
            raise

    def kaggle_download(self, slug: str, dest: Path, before_write: SpaceCheck) -> int:
        """Download the dataset archive to ``dest`` without extracting it; resumable."""
        return self._resumable(
            lambda have: self._kaggle_response(slug, have), dest, before_write, f"kaggle {slug}"
        )

    def url_download(self, url: str, dest: Path, before_write: SpaceCheck) -> int:
        """Download a public file (no credentials) to ``dest``; resumable."""

        def open_at(have: int) -> Any:
            return _open(url, {"Range": f"bytes={have}-"} if have else {}, follow=True)

        return self._resumable(open_at, dest, before_write, url)

    def _resumable(
        self, open_at: Callable[[int], Any], dest: Path, before_write: SpaceCheck, label: str
    ) -> int:
        """Bytes go to ``dest.part``; an interrupted download continues from its size with an
        HTTP Range request. ``dest`` appears only when the byte count equals the total."""
        dest = Path(dest)
        part = dest.with_name(dest.name + ".part")
        dest.parent.mkdir(parents=True, exist_ok=True)
        checked = False
        for attempt in range(self.retries + 1):
            have = part.stat().st_size if part.exists() else 0
            try:
                resp = open_at(have)
                with resp:
                    if resp.status == 206:
                        total = int(resp.headers["Content-Range"].rsplit("/", 1)[1])
                        mode = "ab"
                    else:  # the server ignored the Range header: start again
                        total, have, mode = int(resp.headers["Content-Length"]), 0, "wb"
                    if not checked:
                        before_write(total - have, total)
                        checked = True
                    log.info("%s: %.2f of %.2f GB already here", label, have / 1e9, total / 1e9)
                    with open(part, mode) as out:
                        while block := resp.read(CHUNK):
                            out.write(block)
                if part.stat().st_size == total:
                    os.replace(part, dest)
                    return total
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
                if isinstance(exc, urllib.error.HTTPError) and 400 <= exc.code < 500:
                    raise  # not found / bad request: retrying will not help
                log.warning(
                    "download interrupted (%s); retry %d of %d",
                    type(exc).__name__,
                    attempt + 1,
                    self.retries,
                )
                time.sleep(min(60, 2**attempt))
        raise RuntimeError(
            f"Download of {label} did not complete after {self.retries} retries; "
            "run the step again to resume."
        )

    # -- Hugging Face ------------------------------------------------------------------

    def _api(self) -> Any:
        from huggingface_hub import HfApi

        return HfApi(token=optional_hf_token())

    @staticmethod
    def _wrap(exc: Exception, repo: str) -> Exception:
        name = type(exc).__name__
        if name in {"GatedRepoError", "RepositoryNotFoundError"} or " 401" in str(exc):
            return MissingSecretError(
                f"Hugging Face refused {repo} ({name}). Accept the dataset's terms on its page "
                "and add HF_TOKEN in the Colab Secrets panel with notebook access enabled."
            )
        return exc

    def hf_file_info(self, repo: str, filenames: list[str]) -> dict[str, dict[str, Any]]:
        try:
            infos = self._api().get_paths_info(repo, filenames, repo_type="dataset", expand=True)
        except Exception as exc:
            raise self._wrap(exc, repo) from exc
        return {
            f.path: {"size": f.size, "sha256": f.lfs.sha256 if f.lfs else None}
            for f in infos
            if hasattr(f, "size")
        }

    def hf_list_files(self, repo: str) -> dict[str, int]:
        try:
            tree = self._api().list_repo_tree(repo, recursive=True, repo_type="dataset")
            return {f.path: f.size for f in tree if hasattr(f, "size")}
        except Exception as exc:
            raise self._wrap(exc, repo) from exc

    def hf_download(self, repo: str, filename: str, dest_dir: Path) -> Path:
        """Download one file into ``dest_dir`` (resumes a partial download)."""
        from huggingface_hub import hf_hub_download

        try:
            return Path(
                hf_hub_download(
                    repo,
                    filename,
                    repo_type="dataset",
                    local_dir=dest_dir,
                    token=optional_hf_token(),
                )
            )
        except Exception as exc:
            raise self._wrap(exc, repo) from exc

    def hf_parquet_column(self, repo: str, filename: str, column: str) -> list[Any]:
        """One column of a remote parquet file (reads only that column's pages)."""
        import pyarrow.parquet as pq
        from huggingface_hub import HfFileSystem

        try:
            fs = HfFileSystem(token=optional_hf_token())
            with fs.open(f"datasets/{repo}/{filename}", "rb") as handle:
                return pq.read_table(handle, columns=[column]).column(column).to_pylist()
        except Exception as exc:
            raise self._wrap(exc, repo) from exc
