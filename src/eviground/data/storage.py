"""Disk space, archives and verification for the data layer.

Storage rules (Research Specification, section 13, 12 Nov 2026): source archives stay on
Google Drive and are unpacked on Colab's local disk each session; no loose-file copies and
no duplicate shards on Drive. Every download checks free space on both Drive and the local
disk first, is resumable, and is verified (file counts and sizes) before ``_DONE`` is written.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tarfile
import zipfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

GB = 1e9
DEFAULT_RESERVE_GB = 5.0


class InsufficientSpaceError(RuntimeError):
    """A step would not fit on Drive or on the local disk. Nothing was written for it."""


class VerificationError(RuntimeError):
    """Downloaded or written data does not match what was expected; ``_DONE`` is not written."""


# --------------------------------------------------------------------------------------
# Free space
# --------------------------------------------------------------------------------------


def free_bytes(path: Path | str) -> int:
    """Free bytes on the file system holding ``path`` (or its nearest existing parent)."""
    probe = Path(path)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free


def check_space(
    path: Path | str,
    need_bytes: float,
    *,
    what: str,
    where: str,
    reserve_gb: float = DEFAULT_RESERVE_GB,
) -> dict[str, float]:
    """Raise :class:`InsufficientSpaceError` unless ``need_bytes`` fit with a reserve left.

    Returns the numbers used, for the run record.
    """
    free = free_bytes(path)
    info = {
        "need_gb": round(need_bytes / GB, 2),
        "free_gb": round(free / GB, 2),
        "reserve_gb": reserve_gb,
    }
    if free - need_bytes < reserve_gb * GB:
        raise InsufficientSpaceError(
            f"{what} needs {need_bytes / GB:.1f} GB on {where} ({path}) and a "
            f"{reserve_gb:.0f} GB reserve must stay free, but only {free / GB:.1f} GB is free. "
            f"Nothing was written for this step. Free at least "
            f"{(need_bytes + reserve_gb * GB - free) / GB:.1f} GB on {where} and run it again; "
            "finished parts are kept and skipped."
        )
    return info


# --------------------------------------------------------------------------------------
# Small file helpers
# --------------------------------------------------------------------------------------


def sha256_file(path: Path | str, chunk: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def write_text_atomic(path: Path | str, text: str) -> None:
    """Write via a temporary file in the same folder and rename (safe on Drive)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "w") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def write_json_atomic(path: Path | str, obj: Any) -> None:
    write_text_atomic(path, json.dumps(obj, indent=1, sort_keys=True))


def read_json(path: Path | str) -> Any | None:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


# --------------------------------------------------------------------------------------
# Archives: read zip or tar without extracting
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Member:
    name: str
    size: int


def archive_kind(path: Path | str) -> str:
    """'zip', 'tar' or 'unknown', from the file's bytes."""
    path = Path(path)
    if zipfile.is_zipfile(path):
        return "zip"
    try:
        if tarfile.is_tarfile(path):
            return "tar"
    except OSError:
        pass
    return "unknown"


def safe_member_name(name: str) -> str:
    """Reject absolute paths and '..' so extraction can never leave its folder."""
    pure = PurePosixPath(name)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise VerificationError(f"unsafe archive member name: {name!r}")
    return pure.as_posix()


class ArchiveReader:
    """Read members of a zip or tar archive without extracting it.

    zip members are read in any order (random access); for tar, :meth:`iter_read` makes
    one sequential pass, which is what a 90 GB archive needs.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.kind = archive_kind(self.path)
        if self.kind == "zip":
            self._zip: zipfile.ZipFile | None = zipfile.ZipFile(self.path)
            self._tar: tarfile.TarFile | None = None
        elif self.kind == "tar":
            self._zip = None
            self._tar = tarfile.open(self.path, "r:*")
        else:
            raise VerificationError(f"not a zip or tar archive: {self.path}")

    def __enter__(self) -> ArchiveReader:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        for handle in (self._zip, self._tar):
            if handle is not None:
                handle.close()

    def members(self) -> list[Member]:
        """Regular files only, in archive order."""
        if self._zip is not None:
            return [Member(i.filename, i.file_size) for i in self._zip.infolist() if not i.is_dir()]
        assert self._tar is not None
        return [Member(m.name, m.size) for m in self._tar.getmembers() if m.isfile()]

    def read(self, name: str) -> bytes:
        """One member's bytes (zip checks the CRC while reading)."""
        if self._zip is not None:
            return self._zip.read(name)
        assert self._tar is not None
        handle = self._tar.extractfile(name)
        if handle is None:
            raise KeyError(name)
        return handle.read()

    def iter_read(self, names: Iterable[str]) -> Iterator[tuple[str, bytes]]:
        """Yield ``(name, bytes)`` for the wanted members, in archive order."""
        wanted = set(names)
        if self._zip is not None:
            for info in self._zip.infolist():
                if info.filename in wanted:
                    yield info.filename, self._zip.read(info)
            return
        assert self._tar is not None
        for member in self._tar:
            if member.isfile() and member.name in wanted:
                handle = self._tar.extractfile(member)
                assert handle is not None
                yield member.name, handle.read()


# --------------------------------------------------------------------------------------
# Tar shards: write, verify, then publish
# --------------------------------------------------------------------------------------


def tar_index(path: Path | str) -> dict[str, int]:
    """Member name -> size for the regular files of a tar (headers only)."""
    with tarfile.open(path, "r:*") as tf:
        return {m.name: m.size for m in tf.getmembers() if m.isfile()}


def verify_tar(path: Path | str, expected: dict[str, int]) -> list[str]:
    """Problems (missing, unexpected or wrong-size members); empty when the tar matches."""
    try:
        found = tar_index(path)
    except (OSError, tarfile.TarError) as exc:
        return [f"unreadable: {type(exc).__name__}: {exc}"]
    problems = [f"missing {n}" for n in sorted(set(expected) - set(found))]
    problems += [f"unexpected {n}" for n in sorted(set(found) - set(expected))]
    problems += [
        f"size {n}: {found[n]} != {expected[n]}"
        for n in sorted(set(found) & set(expected))
        if found[n] != expected[n]
    ]
    return problems


def write_tar_shard(
    path: Path | str, items: Iterable[tuple[str, bytes]], expected: dict[str, int]
) -> list[dict[str, Any]]:
    """Write ``items`` into an uncompressed tar, verify it, then move it into place.

    The tar is written as ``<name>.part`` and renamed only after its member names and
    sizes equal ``expected``; a failure leaves no final file. Returns one record per
    member: name, bytes and sha256.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    records: list[dict[str, Any]] = []
    with open(tmp, "wb") as raw:
        with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as tf:
            for name, data in items:
                info = tarfile.TarInfo(safe_member_name(name))
                info.size = len(data)
                info.mtime = 0  # reproducible bytes
                tf.addfile(info, io.BytesIO(data))
                records.append(
                    {
                        "name": info.name,
                        "bytes": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(),
                    }
                )
        raw.flush()
        os.fsync(raw.fileno())
    problems = verify_tar(tmp, expected)
    if problems:
        tmp.unlink(missing_ok=True)
        raise VerificationError(f"{path.name}: {len(problems)} problems, e.g. {problems[:3]}")
    os.replace(tmp, path)
    return records


# --------------------------------------------------------------------------------------
# Unpacking on the local disk (each session)
# --------------------------------------------------------------------------------------


def extract_archive(
    archive: Path | str,
    dest: Path | str,
    *,
    reserve_gb: float = DEFAULT_RESERVE_GB,
    names: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Unpack ``archive`` (or only ``names``) into ``dest``; resumable and verified.

    Files already present with the right size are skipped. Free space on the destination
    is checked for what is still missing before anything is written. Each file is written
    as ``.part`` and renamed. Returns counts; raises :class:`VerificationError` if the
    files on disk do not match the archive afterwards.
    """
    dest = Path(dest)
    with ArchiveReader(archive) as reader:
        members = reader.members()
        if names is not None:
            wanted = set(names)
            members = [m for m in members if m.name in wanted]
        todo = []
        for m in members:
            target = dest / safe_member_name(m.name)
            if not (target.is_file() and target.stat().st_size == m.size):
                todo.append(m)
        check_space(
            dest,
            sum(m.size for m in todo),
            what=f"unpacking {Path(archive).name}",
            where="the local disk",
            reserve_gb=reserve_gb,
        )
        sizes = {m.name: m.size for m in todo}
        for name, data in reader.iter_read(sizes):
            target = dest / safe_member_name(name)
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".part")
            tmp.write_bytes(data)
            os.replace(tmp, target)
    bad = [
        m.name
        for m in members
        if not ((dest / m.name).is_file() and (dest / m.name).stat().st_size == m.size)
    ]
    if bad:
        raise VerificationError(
            f"{len(bad)} files missing or wrong size after unpacking, e.g. {bad[:3]}"
        )
    return {"files": len(members), "extracted": len(todo), "bytes": sum(m.size for m in members)}
