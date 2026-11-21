"""ذخیرهٔ امن کلید API روی دیسک.

settings.json در پوشهٔ پروژه می‌ماند و ممکن است کپی یا به اشتراک گذاشته شود؛
کلید نباید متن‌ساده داخلش باشد. روی ویندوز از DPAPI (همان رمزنگاری کاربرِ
جاری) استفاده می‌کنیم: کلید فقط برای همان کاربر و همان رایانه باز می‌شود.
اگر DPAPI در دسترس نباشد (مثلاً لینوکس)، مقدار بدون تغییر برمی‌گردد تا برنامه
از کار نیفتد و متن‌سادهٔ قدیمی هم خوانده شود.
"""
from __future__ import annotations

import base64
import ctypes
import os

PREFIX = "dpapi:"


class _DataBlob(ctypes.Structure):
    _fields_ = [("cbData", ctypes.c_uint32),
                ("pbData", ctypes.POINTER(ctypes.c_ubyte))]


_CRYPT32 = None
_KERNEL32 = None
if os.name == "nt":
    try:  # pragma: no cover - exercised on the Windows build
        _CRYPT32 = ctypes.windll.crypt32
        _KERNEL32 = ctypes.windll.kernel32
        _CRYPT32.CryptProtectData.argtypes = [
            ctypes.POINTER(_DataBlob), ctypes.c_wchar_p, ctypes.POINTER(_DataBlob),
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32,
            ctypes.POINTER(_DataBlob)]
        _CRYPT32.CryptProtectData.restype = ctypes.c_int
        _CRYPT32.CryptUnprotectData.argtypes = [
            ctypes.POINTER(_DataBlob), ctypes.c_void_p, ctypes.POINTER(_DataBlob),
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32,
            ctypes.POINTER(_DataBlob)]
        _CRYPT32.CryptUnprotectData.restype = ctypes.c_int
        _KERNEL32.LocalFree.argtypes = [ctypes.c_void_p]
        _KERNEL32.LocalFree.restype = ctypes.c_void_p
    except Exception:  # pragma: no cover
        _CRYPT32 = _KERNEL32 = None


def _blob(data: bytes) -> _DataBlob:
    buffer = ctypes.create_string_buffer(data, len(data))
    return _DataBlob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)))


def is_protected(value: str) -> bool:
    return isinstance(value, str) and value.startswith(PREFIX)


def protect(text: str) -> str:
    """متن را برای ذخیره در settings.json رمز کن؛ در خطا همان متن را برگردان."""
    if not text or is_protected(text) or _CRYPT32 is None:
        return text
    try:
        incoming = _blob(text.encode("utf-8"))
        outgoing = _DataBlob()
        if not _CRYPT32.CryptProtectData(ctypes.byref(incoming), "CrashWatcher",
                                         None, None, None, 0, ctypes.byref(outgoing)):
            return text
        try:
            sealed = ctypes.string_at(outgoing.pbData, outgoing.cbData)
        finally:
            _KERNEL32.LocalFree(outgoing.pbData)
        return PREFIX + base64.b64encode(sealed).decode("ascii")
    except Exception:
        return text


def unprotect(value: str) -> str:
    """مقدار ذخیره‌شده را باز کن؛ متن‌سادهٔ قدیمی بدون تغییر می‌ماند."""
    if not is_protected(value) or _CRYPT32 is None:
        return value
    try:
        sealed = base64.b64decode(value[len(PREFIX):])
    except Exception:
        return value
    try:
        incoming = _blob(sealed)
        outgoing = _DataBlob()
        if not _CRYPT32.CryptUnprotectData(ctypes.byref(incoming), None, None,
                                           None, None, 0, ctypes.byref(outgoing)):
            return ""
        try:
            plain = ctypes.string_at(outgoing.pbData, outgoing.cbData)
        finally:
            _KERNEL32.LocalFree(outgoing.pbData)
        return plain.decode("utf-8", "replace")
    except Exception:
        return ""
