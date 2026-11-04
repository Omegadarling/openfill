# OpenFill for PhotoCraft
# SPDX-License-Identifier: Apache-2.0
"""Windows file drops scoped to a single native Tk control."""

from __future__ import annotations

import sys
import tkinter as tk
from collections.abc import Callable

_WM_DROPFILES = 0x0233
_WM_NCDESTROY = 0x0082
_callbacks: dict[int, object] = {}  # Keep native callback pointers alive until the window is destroyed.


def accept_file_drops(
    widget: tk.Misc,
    queue_drop: Callable[[tuple[str, ...]], None],
    queue_error: Callable[[str], None],
    max_files: int,
) -> bool:
    """Register one control's HWND. Callbacks may enqueue work but must never call Tcl."""
    if sys.platform != "win32":
        return False
    import ctypes as ct
    from ctypes import wintypes as wt

    proc_type = ct.WINFUNCTYPE(ct.c_ssize_t, wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM, ct.c_size_t, ct.c_size_t)
    try:
        shell = ct.WinDLL("shell32", use_last_error=True)
        controls = ct.WinDLL("comctl32", use_last_error=True)
        set_subclass = controls.SetWindowSubclass
        remove_subclass = controls.RemoveWindowSubclass
        forward = controls.DefSubclassProc
        accept, query, finish = shell.DragAcceptFiles, shell.DragQueryFileW, shell.DragFinish
        hwnd = widget.winfo_id()
    except (AttributeError, OSError, tk.TclError):
        return False
    if hwnd in _callbacks:
        return False

    set_subclass.argtypes, set_subclass.restype = (wt.HWND, proc_type, ct.c_size_t, ct.c_size_t), wt.BOOL
    remove_subclass.argtypes, remove_subclass.restype = (wt.HWND, proc_type, ct.c_size_t), wt.BOOL
    forward.argtypes, forward.restype = (wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM), ct.c_ssize_t
    accept.argtypes, accept.restype = (wt.HWND, wt.BOOL), None
    query.argtypes, query.restype = (wt.HANDLE, wt.UINT, wt.LPWSTR, wt.UINT), wt.UINT
    finish.argtypes, finish.restype = (wt.HANDLE,), None

    def read_paths(drop: int) -> tuple[str, ...]:
        count = query(drop, 0xFFFFFFFF, None, 0)
        if count > max_files:
            raise ValueError(f"Drop at most {max_files} reference images at once.")
        paths = []
        for index in range(count):
            length = query(drop, index, None, 0)
            if not 0 < length <= 32767:
                raise ValueError("Could not read the dropped file paths.")
            buffer = ct.create_unicode_buffer(length + 1)
            if query(drop, index, buffer, len(buffer)) != length:
                raise ValueError("Could not read the dropped file paths.")
            paths.append(buffer.value)
        return tuple(paths)

    @proc_type
    def handle(window, message, wparam, lparam, subclass_id, _data):
        if message == _WM_DROPFILES:
            try:
                paths = read_paths(wparam)
                if paths:
                    queue_drop(paths)
            except ValueError as err:
                queue_error(str(err))
            finally:
                finish(wparam)
            return 0
        if message == _WM_NCDESTROY:
            accept(window, False)
            remove_subclass(window, handle, subclass_id)
            try:
                return forward(window, message, wparam, lparam)
            finally:
                _callbacks.pop(window, None)
        return forward(window, message, wparam, lparam)

    if not set_subclass(hwnd, handle, 0, 0):
        return False
    _callbacks[hwnd] = handle
    accept(hwnd, True)
    return True
