"""ساخت و جایگزینی امن نسخهٔ نصب‌شده.

مراحل:
  1. اجرای تست‌ها (مگر با ``--skip-tests``)
  2. ساخت با PyInstaller (مگر با ``--skip-build``)
  3. کپی داده‌های جلسه (settings.json، history.csv، logs و…) از نسخهٔ نصب‌شده
     به بیلد تازه
  4. جابه‌جایی اتمیک: نسخهٔ قدیمی به ``*.old-<timestamp>`` تغییر نام می‌دهد و
     بیلد تازه جای آن می‌نشیند

داده‌ها هرگز حذف نمی‌شوند و پشتیبانِ نسخهٔ قبلی هم می‌ماند.

  python tools/release.py --dry-run
  python tools/release.py
  python tools/release.py --target "F:/CRASH WATCHER/FINAL VERSION/repo 1/CrashWatcher"
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = ROOT / "FINAL VERSION" / "repo 1" / "CrashWatcher"
PRESERVE = ("settings.json", "settings.json.bak", "history.csv", "self-test.json",
            "dev.log", "logs")
BUILD_COMMAND = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--windowed",
                 "--onedir", "--name", "CrashWatcher", "--icon", "assets/crash_watcher.ico",
                 "--collect-all", "rapidocr", "--collect-all", "onnxruntime",
                 "--paths", ".", "desktop_app.py"]


def digest(path: Path) -> str:
    if not path.exists():
        return "missing"
    if path.is_dir():
        hasher = hashlib.sha256()
        for item in sorted(path.rglob("*")):
            if item.is_file():
                hasher.update(str(item.relative_to(path)).encode("utf-8"))
                hasher.update(item.read_bytes())
        return hasher.hexdigest()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command: list[str], dry_run: bool) -> None:
    print("$", " ".join(str(part) for part in command))
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def copy_data(source: Path, destination: Path, dry_run: bool) -> None:
    """داده‌های جلسهٔ نسخهٔ نصب‌شده را به بیلد تازه منتقل کن."""
    for name in PRESERVE:
        item = source / name
        if not item.exists():
            continue
        target = destination / name
        print(f"  حفظ داده: {name}")
        if dry_run:
            continue
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def swap(built: Path, target: Path, dry_run: bool) -> Path:
    """بیلد تازه را جای نسخهٔ نصب‌شده بگذار و پشتیبان را برگردان."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    staging = target.with_name(target.name + ".new")
    backup = target.with_name(target.name + f".old-{stamp}")
    print(f"$ staging: {staging}")
    if not dry_run:
        if staging.exists():
            shutil.rmtree(staging)
        shutil.copytree(built, staging)
        copy_data(target, staging, dry_run=False)
        if target.exists():
            target.rename(backup)
        staging.rename(target)
    return backup


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="ساخت و جایگزینی امن CrashWatcher")
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET,
                        help=f"پوشهٔ نصب (پیش‌فرض: {DEFAULT_TARGET})")
    parser.add_argument("--skip-tests", action="store_true", help="تست‌ها را اجرا نکن")
    parser.add_argument("--skip-build", action="store_true",
                        help="فقط همین بیلد موجود در dist/ را نصب کن")
    parser.add_argument("--dry-run", action="store_true", help="فقط دستورها را نشان بده")
    args = parser.parse_args(argv)

    if not args.skip_tests:
        run([sys.executable, "-m", "pytest", "-q"], args.dry_run)
    if not args.skip_build:
        run(BUILD_COMMAND, args.dry_run)

    built = ROOT / "dist" / "CrashWatcher"
    if not built.is_dir() and not args.dry_run:
        print(f"بیلد پیدا نشد: {built}", file=sys.stderr)
        return 1
    before = {name: digest(args.target / name) for name in PRESERVE}
    backup = swap(built, args.target, args.dry_run)
    if not args.dry_run:
        after = {name: digest(args.target / name) for name in PRESERVE}
        changed = [name for name in PRESERVE if before[name] != after[name]
                   and before[name] != "missing"]
        if changed:
            print("خطا: این داده‌ها تغییر کردند: " + "، ".join(changed), file=sys.stderr)
            return 2
        print(f"نصب شد: {args.target}")
        print(f"پشتیبان نسخهٔ قبلی: {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
