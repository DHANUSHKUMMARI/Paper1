"""Space checks, archive reading, verified tar shards and resumable unpacking."""

from __future__ import annotations

import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from eviground.data import storage
from eviground.data.storage import (
    GB,
    ArchiveReader,
    InsufficientSpaceError,
    VerificationError,
    check_space,
    extract_archive,
    tar_index,
    verify_tar,
    write_tar_shard,
)

FILES = {"a/x.png": b"x" * 10, "a/y.jpg": b"yy" * 7, "b/z.txt": b"zzz"}


def _zip(path: Path, files: dict[str, bytes] = FILES) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("a/", "")  # a directory entry must not count as a member
        for name, data in files.items():
            zf.writestr(name, data)
    return path


def _tar(path: Path, files: dict[str, bytes] = FILES) -> Path:
    with tarfile.open(path, "w") as tf:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return path


def test_check_space_message(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(storage, "free_bytes", lambda p: int(10 * GB))
    assert check_space(tmp_path, 4 * GB, what="x", where="Drive")["free_gb"] == 10.0
    with pytest.raises(InsufficientSpaceError) as err:
        check_space(tmp_path, 6 * GB, what="DiffSeg30k train.zip", where="Drive")
    msg = str(err.value)
    assert "DiffSeg30k train.zip needs 6.0 GB on Drive" in msg
    assert "only 10.0 GB is free" in msg and "Free at least 1.0 GB" in msg


def test_free_bytes_of_missing_path(tmp_path: Path) -> None:
    assert storage.free_bytes(tmp_path / "not" / "there") > 0


@pytest.mark.parametrize("make", [_zip, _tar])
def test_archive_reader(tmp_path: Path, make) -> None:
    path = make(tmp_path / "arc")
    with ArchiveReader(path) as reader:
        assert {m.name: m.size for m in reader.members()} == {k: len(v) for k, v in FILES.items()}
        assert reader.read("a/y.jpg") == FILES["a/y.jpg"]
        got = dict(reader.iter_read(["b/z.txt", "a/x.png"]))
        assert got == {"b/z.txt": b"zzz", "a/x.png": FILES["a/x.png"]}


def test_archive_reader_rejects_other_files(tmp_path: Path) -> None:
    other = tmp_path / "x.bin"
    other.write_bytes(b"not an archive")
    with pytest.raises(VerificationError):
        ArchiveReader(other)


def test_write_tar_shard_verified(tmp_path: Path) -> None:
    expected = {k: len(v) for k, v in FILES.items()}
    out = tmp_path / "drive" / "s-00000.tar"
    records = write_tar_shard(out, FILES.items(), expected)
    assert tar_index(out) == expected and verify_tar(out, expected) == []
    assert [r["name"] for r in records] == list(FILES)
    assert not out.with_name(out.name + ".part").exists()
    # same input gives byte-identical shards
    again = tmp_path / "again.tar"
    write_tar_shard(again, FILES.items(), expected)
    assert again.read_bytes() == out.read_bytes()


def test_write_tar_shard_mismatch_leaves_nothing(tmp_path: Path) -> None:
    out = tmp_path / "s.tar"
    expected = {k: len(v) for k, v in FILES.items()} | {"missing.png": 5}
    with pytest.raises(VerificationError):
        write_tar_shard(out, FILES.items(), expected)
    assert not out.exists() and not out.with_name("s.tar.part").exists()


def test_verify_tar_reports_problems(tmp_path: Path) -> None:
    path = _tar(tmp_path / "t.tar")
    problems = verify_tar(path, {"a/x.png": 11, "new": 1, "a/y.jpg": 14})
    assert "missing new" in problems and "unexpected b/z.txt" in problems
    assert any(p.startswith("size a/x.png") for p in problems)
    assert verify_tar(tmp_path / "nope.tar", {})[0].startswith("unreadable")


@pytest.mark.parametrize("make", [_zip, _tar])
def test_extract_is_resumable_and_verified(tmp_path: Path, make) -> None:
    arc = make(tmp_path / "arc")
    dest = tmp_path / "local"
    first = extract_archive(arc, dest)
    assert first == {"files": 3, "extracted": 3, "bytes": 27}
    assert (dest / "a/y.jpg").read_bytes() == FILES["a/y.jpg"]
    (dest / "b/z.txt").unlink()  # an interrupted session
    (dest / "a/x.png").write_bytes(b"short")  # a partial file
    second = extract_archive(arc, dest)
    assert second["extracted"] == 2
    assert (dest / "a/x.png").read_bytes() == FILES["a/x.png"]
    assert extract_archive(arc, dest)["extracted"] == 0
    assert extract_archive(arc, tmp_path / "some", names=["a/x.png"])["files"] == 1


def test_extract_checks_space_before_writing(tmp_path: Path, monkeypatch) -> None:
    arc = _zip(tmp_path / "arc.zip")
    monkeypatch.setattr(storage, "free_bytes", lambda p: int(1 * GB))
    with pytest.raises(InsufficientSpaceError):
        extract_archive(arc, tmp_path / "local")
    assert not (tmp_path / "local").exists()


def test_unsafe_member_names_are_rejected(tmp_path: Path) -> None:
    arc = _zip(tmp_path / "evil.zip", {"../escape.txt": b"x"})
    with pytest.raises(VerificationError):
        extract_archive(arc, tmp_path / "local")
    assert not (tmp_path / "escape.txt").exists()


def test_copy_files_resumable_and_checked(tmp_path: Path, monkeypatch) -> None:
    src = tmp_path / "drive"
    for name, data in FILES.items():
        (src / name).parent.mkdir(parents=True, exist_ok=True)
        (src / name).write_bytes(data)
    files = {k: len(v) for k, v in FILES.items()}
    dest = tmp_path / "local"
    assert storage.copy_files(src, files, dest)["copied"] == 3
    (dest / "a/x.png").write_bytes(b"cut")  # an interrupted copy
    assert storage.copy_files(src, files, dest)["copied"] == 1
    assert (dest / "a/x.png").read_bytes() == FILES["a/x.png"]
    with pytest.raises(VerificationError):
        storage.copy_files(src, {**files, "a/y.jpg": 999}, tmp_path / "other")
    monkeypatch.setattr(storage, "free_bytes", lambda p: 1)
    with pytest.raises(InsufficientSpaceError):
        storage.copy_files(src, files, tmp_path / "third")
    assert not (tmp_path / "third").exists()


def _loose(src: Path, n: int) -> dict[str, int]:
    files = {}
    for i in range(n):
        rel = f"split/cls/images/{i:05d}.jpg"
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        data = bytes([i % 256]) * (100 + i)
        (src / rel).write_bytes(data)
        files[rel] = len(data)
    return files


def test_parallel_copy_same_result_resume_and_progress(tmp_path: Path) -> None:
    src = tmp_path / "drive"
    files = _loose(src, 50)
    lines: list[str] = []
    res = storage.copy_files(src, files, tmp_path / "a", workers=8, progress=lines.append)
    assert res["copied"] == 50
    for rel, size in files.items():
        assert (tmp_path / "a" / rel).read_bytes() == (src / rel).read_bytes()
        assert (tmp_path / "a" / rel).stat().st_size == size
    assert not list((tmp_path / "a").rglob("*.part"))
    assert lines and lines[-1].startswith("copying drive: 50/50 files")
    # interrupted session: two files missing, one cut short -> only those three are copied
    (tmp_path / "a" / "split/cls/images/00003.jpg").unlink()
    (tmp_path / "a" / "split/cls/images/00007.jpg").unlink()
    (tmp_path / "a" / "split/cls/images/00011.jpg").write_bytes(b"cut")
    assert storage.copy_files(src, files, tmp_path / "a", workers=8)["copied"] == 3
    assert (tmp_path / "a" / "split/cls/images/00011.jpg").read_bytes() == (
        src / "split/cls/images/00011.jpg"
    ).read_bytes()


def test_parallel_copy_error_reaches_caller(tmp_path: Path) -> None:
    src = tmp_path / "drive"
    files = _loose(src, 20)
    (src / "split/cls/images/00005.jpg").unlink()  # vanished on Drive
    with pytest.raises(FileNotFoundError):
        storage.copy_files(src, files, tmp_path / "a", workers=4)


def test_parallel_copy_hides_per_file_latency(tmp_path: Path, monkeypatch) -> None:
    """The Drive mount costs a round trip per file; threads overlap them."""
    import shutil as _shutil
    import time as _time

    src = tmp_path / "drive"
    files = _loose(src, 40)
    real = _shutil.copyfile

    def slow_copy(a, b, *args, **kw):
        _time.sleep(0.03)
        return real(a, b, *args, **kw)

    monkeypatch.setattr(storage.shutil, "copyfile", slow_copy)
    t = _time.monotonic()
    storage.copy_files(src, files, tmp_path / "serial", workers=1)
    serial = _time.monotonic() - t
    t = _time.monotonic()
    storage.copy_files(src, files, tmp_path / "parallel", workers=16)
    parallel = _time.monotonic() - t
    assert serial > 1.0 and parallel < serial / 3


def test_progress_reports_at_intervals_and_end() -> None:
    lines: list[str] = []
    p = storage.Progress("x", total_files=10, total_bytes=1000, report=lines.append, every_files=4)
    for _ in range(10):
        p.add(100)
    assert [ln.split(":")[1].split(",")[0].strip() for ln in lines] == [
        "4/10 files",
        "8/10 files",
        "10/10 files",
    ]


def test_extract_archive_reports_progress(tmp_path: Path) -> None:
    arc = _zip(tmp_path / "d.zip")
    lines: list[str] = []
    extract_archive(arc, tmp_path / "local", progress=lines.append)
    assert lines[-1].startswith("unpacking d.zip: 3/3 files")
