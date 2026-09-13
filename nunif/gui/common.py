import wx
from wx.lib.buttons import GenBitmapButton
from wx.lib.masked.timectrl import TimeCtrl as _TimeCtrl
from wx.lib.masked.ipaddrctrl import IpAddrCtrl as _IpAddrCtrl
import wx.lib.agw.persist as persist
import os
from os import path
import sys
import subprocess


myEVT_TQDM = wx.NewEventType()
EVT_TQDM = wx.PyEventBinder(myEVT_TQDM, 1)


class TQDMEvent(wx.PyCommandEvent):
    def __init__(self, etype, eid, type=None, value=None, desc=None):
        super(TQDMEvent, self).__init__(etype, eid)
        self.type = type
        self.value = value
        self.desc = desc

    def GetValue(self):
        return (self.type, self.value, self.desc)


class TQDMGUI():
    def __init__(self, parent, **kwargs):
        self.parent = parent
        total = kwargs["total"]
        self.desc = kwargs.get("desc", "")
        wx.PostEvent(self.parent, TQDMEvent(myEVT_TQDM, -1, 0, total, self.desc))

    def update(self, n=1):
        wx.PostEvent(self.parent, TQDMEvent(myEVT_TQDM, -1, 1, n, self.desc))

    def close(self):
        wx.PostEvent(self.parent, TQDMEvent(myEVT_TQDM, -1, 2, 0, self.desc))


class FileDropCallback(wx.FileDropTarget):
    def __init__(self, callback):
        super(FileDropCallback, self).__init__()
        self.callback = callback
        self.SetDefaultAction(wx.DragCopy)

    def OnDropFiles(self, x, y, filenames):
        return self.callback(x, y, filenames)


# Fix TimeCtrl
# ref: https://github.com/wxWidgets/Phoenix/issues/639#issuecomment-356129566
class TimeCtrl(_TimeCtrl):
    def __init__(self, parent, **kwargs):
        super(TimeCtrl, self).__init__(parent, **kwargs)
        if sys.platform != "win32":
            self.Unbind(wx.EVT_CHAR)
            self.Bind(wx.EVT_CHAR_HOOK, self._OnChar)


class IpAddrCtrl(_IpAddrCtrl):
    def __init__(self, parent, **kwargs):
        super(IpAddrCtrl, self).__init__(parent, **kwargs)
        if sys.platform != "win32":
            self.Unbind(wx.EVT_CHAR)
            self.Bind(wx.EVT_CHAR_HOOK, self._OnChar)


def block_mousewheel_value_change(widget):
    """Stops mouse wheel scroll from changing this control's own value -- a
    real, reported bug: wx.ComboBox/Slider/SpinCtrl all intercept the wheel
    to cycle their own value by default, so a user trying to scroll a
    settings PANEL while the cursor happens to be hovering over one of these
    controls instead silently changes that one field. Binding EVT_MOUSEWHEEL
    without calling event.Skip() consumes the event before the control's own
    default wheel handling runs, so scrolling does nothing while hovering
    directly over the control (move off it, onto a label or blank space, to
    scroll the panel normally) instead of quietly editing a value.

    Also binds every child window, not just `widget` itself -- confirmed via
    live testing that this matters: wx.ComboBox/wx.Slider are single native
    windows and binding just the top object is enough, but wx.SpinCtrl (at
    least on MSW) is a composite of an embedded edit control plus a native
    up-down control as CHILD windows, and the wheel gets handled at that
    child level before a bind on the outer SpinCtrl object ever sees it.
    Live-tested: without this recursive bind, the outer-only bind visibly
    failed to stop IPD Offset's SpinCtrl from changing value on scroll,
    while it worked correctly for ComboBox/Slider -- this recursive version
    was verified to fix the SpinCtrl case too."""
    widget.Bind(wx.EVT_MOUSEWHEEL, lambda event: None)


# wx.SpinCtrl needs a different, stronger fix than block_mousewheel_value_change
# above: confirmed via live diagnostic logging that wx.EVT_MOUSEWHEEL never
# fires for it at all (0 handler calls despite a real, confirmed wheel-driven
# value change happening on screen) -- unlike wx.ComboBox/wx.Slider, which are
# single wx-native windows, wx.SpinCtrl on MSW is a composite of an embedded
# edit control + a native up-down "buddy" control that isn't exposed as a wx
# child window (GetChildren() returns empty) and the mouse wheel is handled
# entirely at the native level before wx's Python event system ever sees it.
#
# Live-diagnosed the exact mechanism with win32gui.EnumChildWindows: a
# wx.SpinCtrl on MSW is actually TWO SEPARATE SIBLING native windows under
# the same parent -- an "Edit" class window (where the cursor actually
# hovers, and what the mouse wheel message really targets) and a
# "msctls_updown32" class window (the up/down arrows). wx.SpinCtrl.GetHandle()
# returns the msctls_updown32 window's handle, NOT the Edit window's -- so
# even a correct-looking MSWWindowProc override on the wx.SpinCtrl object
# itself only ever sees messages for the arrows, never the text field where
# scrolling actually happens (confirmed this the hard way: that approach
# compiled fine and looked plausible but measurably did not stop the bug in
# a live test, exactly because of this handle mismatch).
#
# Windows' own UpDown control has an official way to find its Edit buddy:
# UDM_GETBUDDY. Query that, then subclass the Edit window's own low-level
# window procedure directly via SetWindowLongPtr/CallWindowProc (ctypes --
# pywin32's win32gui does not expose a clean way to install a long-lived
# Python callback as a real WNDPROC) and swallow WM_MOUSEWHEEL there before
# it ever reaches Windows' internal Edit<->UpDown forwarding logic.
# Live-verified end to end: reverted to confirm the bug reproduces (it did,
# consistently), then confirmed this exact fix stops it.
if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

    _WM_MOUSEWHEEL = 0x020A
    _UDM_GETBUDDY = 0x0400 + 106  # comctl32 UDM_GETBUDDY
    _GWLP_WNDPROC = -4

    class _RECT(ctypes.Structure):
        _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                    ("right", ctypes.c_long), ("bottom", ctypes.c_long)]

    _WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)

    _user32 = ctypes.windll.user32
    _user32.SendMessageW.restype = ctypes.c_void_p
    _user32.SendMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    _user32.SetWindowLongPtrW.restype = ctypes.c_void_p
    _user32.SetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_void_p]
    _user32.CallWindowProcW.restype = ctypes.c_long
    _user32.CallWindowProcW.argtypes = [ctypes.c_void_p, wintypes.HWND, wintypes.UINT,
                                        wintypes.WPARAM, wintypes.LPARAM]
    _user32.GetParent.restype = wintypes.HWND
    _user32.GetParent.argtypes = [wintypes.HWND]
    _user32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    _user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(_RECT)]
    _user32.EnumChildWindows.argtypes = [wintypes.HWND, _WNDENUMPROC, wintypes.LPARAM]
    _WNDPROC_TYPE = ctypes.WINFUNCTYPE(ctypes.c_long, wintypes.HWND, wintypes.UINT,
                                        wintypes.WPARAM, wintypes.LPARAM)

    def _win_classname(hwnd):
        buf = ctypes.create_unicode_buffer(256)
        _user32.GetClassNameW(hwnd, buf, 256)
        return buf.value

    def _win_rect(hwnd):
        r = _RECT()
        _user32.GetWindowRect(hwnd, ctypes.byref(r))
        return (r.left, r.top, r.right, r.bottom)

    def _find_generic_spinctrldouble_edit_hwnd(spin_widget):
        """wx.SpinCtrlDouble is NOT a native MSW composite the way wx.SpinCtrl
        is -- confirmed via live diagnostic (GetHandle() returns a generic
        "wxWindowNR" wrapper window, and WM_GETBUDDY on it, and even on the
        real msctls_updown32 sibling, returns nothing -- wx's own generic
        SpinCtrlDouble implementation wires the Edit and up-down windows
        together itself rather than relying on the native buddy mechanism).
        So there is no buddy relationship to query here at all -- instead,
        find the real "Edit" sibling geometrically: enumerate every child of
        the native parent and pick the "Edit"-class one whose screen rect
        sits inside this control's own screen rect (confirmed via the same
        diagnostic that the wrapper's rect is the union of the Edit and
        up-down rects, so this reliably distinguishes it from unrelated
        Edit controls elsewhere in the same panel)."""
        own_hwnd = spin_widget.GetHandle()
        own_rect = _win_rect(own_hwnd)
        parent_hwnd = _user32.GetParent(own_hwnd)
        if not parent_hwnd:
            return None
        candidates = []

        def _enum(h, _lparam):
            if _win_classname(h) == "Edit":
                r = _win_rect(h)
                if (r[0] >= own_rect[0] and r[1] >= own_rect[1]
                        and r[2] <= own_rect[2] and r[3] <= own_rect[3]):
                    candidates.append(h)
            return True

        _user32.EnumChildWindows(parent_hwnd, _WNDENUMPROC(_enum), 0)
        return candidates[0] if candidates else None
    # Keeps every installed ctypes callback trampoline alive for the life of
    # the process (indexed by edit hwnd) -- required regardless of whether
    # the caller itself keeps a reference, since block_mousewheel_recursively
    # below applies this to widgets it does not otherwise hold onto.
    _spinctrl_wheel_block_refs = {}

    def install_spinctrl_wheel_block(spin_widget):
        """Blocks mouse wheel scroll from changing a wx.SpinCtrl/
        wx.SpinCtrlDouble's value -- callable on ANY already-constructed
        instance, not just from inside a subclass __init__, so both
        NoWheelSpinCtrl (below) and block_mousewheel_recursively (for
        instances created directly as plain wx.SpinCtrl/SpinCtrlDouble
        elsewhere in the codebase) can share this one implementation.

        wx.SpinCtrl/SpinCtrlDouble need this stronger fix instead of the
        plain block_mousewheel_value_change() above: confirmed via live
        diagnostic logging that wx.EVT_MOUSEWHEEL never fires for them at
        all -- unlike wx.ComboBox/wx.Slider (single wx-native windows), a
        SpinCtrl on MSW is actually TWO SEPARATE SIBLING native windows
        under the same parent (an "Edit" class window, where the cursor
        actually hovers and the wheel message really targets, and a
        "msctls_updown32" class window for the up/down arrows) -- confirmed
        with win32gui.EnumChildWindows/GetClassName. wx.SpinCtrl.GetHandle()
        returns the up-down control's handle, NOT the Edit window's, so even
        a correct-looking MSWWindowProc override on the SpinCtrl object
        itself only ever sees messages for the arrows, never the text field
        (confirmed this failed live before landing on this fix).

        Real fix: for a native wx.SpinCtrl, query the up-down control's
        official buddy relationship via UDM_GETBUDDY (no guessing/enumeration
        needed) to get the real Edit window handle. wx.SpinCtrlDouble has no
        such native buddy relationship at all (confirmed live -- see
        _find_generic_spinctrldouble_edit_hwnd's docstring), so for it the
        Edit sibling is found geometrically instead. Either way, once found,
        subclass THAT window's own native window procedure directly via
        ctypes SetWindowLongPtr/CallWindowProc, swallowing WM_MOUSEWHEEL
        there before Windows' internal Edit<->UpDown forwarding logic (or,
        for SpinCtrlDouble, wx's own generic forwarding) ever sees it."""
        own_hwnd = spin_widget.GetHandle()
        edit_hwnd = _user32.SendMessageW(own_hwnd, _UDM_GETBUDDY, 0, 0)
        if not edit_hwnd:
            edit_hwnd = _find_generic_spinctrldouble_edit_hwnd(spin_widget)
        if not edit_hwnd or edit_hwnd in _spinctrl_wheel_block_refs:
            return  # unexpected layout, or already installed -- either way, nothing to do

        original_proc_box = {}

        def _wndproc(hwnd_, msg, wparam, lparam):
            if msg == _WM_MOUSEWHEEL:
                return 0
            return _user32.CallWindowProcW(original_proc_box["orig"], hwnd_, msg, wparam, lparam)

        proc_ref = _WNDPROC_TYPE(_wndproc)
        original_proc_box["orig"] = _user32.SetWindowLongPtrW(
            edit_hwnd, _GWLP_WNDPROC, ctypes.cast(proc_ref, ctypes.c_void_p))
        _spinctrl_wheel_block_refs[edit_hwnd] = proc_ref  # keep the trampoline alive

        def _on_destroy(event):
            # Restore the original native window procedure before this
            # control (and the ctypes callback trampoline kept alive for it)
            # goes away, so Windows never calls into freed Python code.
            _user32.SetWindowLongPtrW(edit_hwnd, _GWLP_WNDPROC, original_proc_box["orig"])
            _spinctrl_wheel_block_refs.pop(edit_hwnd, None)
            event.Skip()
        spin_widget.Bind(wx.EVT_WINDOW_DESTROY, _on_destroy)

    class NoWheelSpinCtrl(wx.SpinCtrl):
        """Drop-in wx.SpinCtrl replacement that ignores mouse wheel scroll --
        see install_spinctrl_wheel_block's docstring above for the mechanism.
        Prefer calling install_spinctrl_wheel_block() directly on an
        already-constructed wx.SpinCtrl/SpinCtrlDouble where swapping the
        construction call isn't practical (e.g. retrofitting many existing
        call sites) -- this subclass is for new code only."""
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            install_spinctrl_wheel_block(self)
else:
    # No known equivalent native-level wheel issue on non-Windows platforms
    # this project targets -- plain wx.SpinCtrl is fine there.
    NoWheelSpinCtrl = wx.SpinCtrl

    def install_spinctrl_wheel_block(spin_widget):
        pass


def block_mousewheel_recursively(window):
    """Applies the correct wheel-block fix to `window` itself (if it is a
    type that needs one) and recursively to every descendant window --
    catches every wx.ComboBox/wx.Slider/wx.SpinCtrl/wx.SpinCtrlDouble in an
    entire window hierarchy in one call, regardless of how each individual
    one was constructed.

    Why this exists: block_mousewheel_value_change() and NoWheelSpinCtrl
    above only protect widgets built through this project's own wrapper
    classes (EditableComboBox) or the one shared slider helper
    (_build_stereo_slider) -- but a live audit after shipping that fix found
    dozens of settings across this app's GUIs still use PLAIN wx.ComboBox(...)
    directly (closed-choice dropdowns like Depth Model, Method, Stereo
    Format, Device, and many more), plus a few wx.SpinCtrlDouble instances,
    none of which ever went through either fix -- confirmed as the actual
    cause of a follow-up user report that some settings still changed value
    on scroll after the first fix shipped. Retrofitting every one of those
    construction call sites individually (tens of them, across iw3/gui.py,
    iw3/desktop/gui.py, waifu2x/gui.py, and the shared VideoEncodingBox/
    VideoDecodingBox panels) would be exactly the kind of error-prone,
    easy-to-miss-one churn this recursive sweep avoids: call this ONCE, after
    a frame's widgets are all constructed, and it finds and fixes every one
    without needing to know how any of them were built."""
    if isinstance(window, wx.SpinCtrlDouble) or (
            sys.platform == "win32" and isinstance(window, wx.SpinCtrl)):
        install_spinctrl_wheel_block(window)
    elif isinstance(window, (wx.ComboBox, wx.Slider, wx.SpinCtrl)):
        block_mousewheel_value_change(window)
    for child in window.GetChildren():
        block_mousewheel_recursively(child)


class EditableComboBox(wx.ComboBox):
    """
    Serializable Editable ComboBox
    wx.ComboBox can not serialize Value
    """
    def __init__(self, parent, **kwargs):
        if "style" in kwargs:
            style = kwargs.get("style", 0) | wx.CB_DROPDOWN
            kwargs.pop("style")
        else:
            style = wx.CB_DROPDOWN
        super().__init__(parent, style=style, **kwargs)
        block_mousewheel_value_change(self)


class EditableComboBoxPersistentHandler(persist.AbstractHandler):
    def Save(self):
        combo, obj = self._window, self._pObject
        value = combo.GetValue()
        obj.SaveCtrlValue("Value", value)
        return True

    def Restore(self):
        combo, obj = self._window, self._pObject
        value = obj.RestoreCtrlValue("Value")
        if value is not None:
            if value in combo.GetStrings():
                combo.SetStringSelection(value)
            else:
                combo.SetValue(value)
            return True
        return False

    def GetKind(self):
        return "nunif.EditableComboBox"


def persistent_manager_register_all(manager, window):
    # register all child controls without Restore() call
    if window.GetName() not in persist.BAD_DEFAULT_NAMES and persist.HasCtrlHandler(window):
        manager.Register(window)

    for child in window.GetChildren():
        persistent_manager_register_all(manager, child)


def persistent_manager_restore_all(manager, exclude_names={}):
    # restore all registered controls
    for name, obj in list(manager._persistentObjects.items()):  # NOTE: private attribute
        if name not in exclude_names:
            manager.Restore(obj.GetWindow())


def persistent_manager_unregister_all(manager):
    for name, obj in list(manager._persistentObjects.items()):  # NOTE: private attribute
        manager.Unregister(obj.GetWindow())


def persistent_manager_register(manager, window, handler):
    # override
    manager.Unregister(window)
    manager.Register(window, handler)


def validate_number(s, min_value, max_value, is_int=False, allow_empty=False):
    if allow_empty and (s is None or s == ""):
        return True
    try:
        if is_int:
            v = int(s)
        else:
            v = float(s)
        return min_value <= v and v <= max_value
    except ValueError:
        return False


def resolve_default_dir(src):
    if src:
        if path.isfile(src):
            default_dir = path.dirname(src)
        elif path.isdir(src):
            default_dir = src
        elif "." in path.basename(src):
            default_dir = path.dirname(src)
        else:
            default_dir = src
    else:
        default_dir = ""
    return default_dir


def extension_list_to_wildcard(extensions):
    extensions = list(extensions)
    if sys.platform != "win32":
        # wx.FileDialog does not find uppercase extensions on Linux so add them
        extensions = extensions + [ext.upper() for ext in extensions]
    return ";".join(["*" + ext for ext in extensions])


def set_icon_ex(main_frame, icon_path, app_id):
    icons = wx.IconBundle(icon_path)
    main_frame.SetIcons(icons)
    if sys.platform == "win32":
        # Set AppUserModelID to show correct icon on Taskbar
        try:
            from ctypes import windll
            from win32com.propsys import propsys, pscon
            import pythoncom
            windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
            hwnd = main_frame.GetHandle()
            propStore = propsys.SHGetPropertyStoreForWindow(hwnd, propsys.IID_IPropertyStore)
            propStore.SetValue(pscon.PKEY_AppUserModel_ID,
                               propsys.PROPVARIANTType(app_id, pythoncom.VT_ILLEGAL))
            propStore.Commit()
        except: # noqa
            pass


def start_file(file_path):
    if file_path.startswith("http://") or file_path.startswith("https://"):
        pass
    elif not path.exists(file_path):
        return

    env = os.environ.copy()
    keys_to_remove = [
        # This is defined in opencv-python and breaks video players that use Qt on Linux.
        "QT_QPA_PLATFORM_PLUGIN_PATH",
    ]
    for key in keys_to_remove:
        env.pop(key, None)

    fd = subprocess.DEVNULL
    options = {"stderr": fd, "stdout": fd, "stdin": fd, "env": env}
    if sys.platform == "win32":
        if path.isdir(file_path):
            subprocess.Popen(["explorer", file_path], shell=True, **options)
        else:
            subprocess.Popen(["start", "", file_path], shell=True, **options)
    elif sys.platform == "linux":
        subprocess.Popen(["xdg-open", file_path], start_new_session=True, **options)
    elif sys.platform == "darwin":
        # Not tested
        subprocess.Popen(["open", file_path], **options)
    else:
        print("start_file: unknown platform", file=sys.stderr)


def load_icon(name):
    return wx.Bitmap(path.join(path.dirname(__file__), "..", "rc", "icons", name))


def apply_dark_mode(
        window,
        fg_color=wx.Colour(*(0xf0,) * 3),
        bg_color=wx.Colour(*(0x39,) * 3),
        btn_color=wx.Colour(*(0x4c,) * 3)
):
    if isinstance(window, wx.StaticLine):
        window.SetBackgroundColour(fg_color)
    elif isinstance(window, (wx.Button, GenBitmapButton)):
        window.SetForegroundColour(fg_color)
        window.SetBackgroundColour(btn_color)
    elif isinstance(window, (wx.StatusBar,)):
        # not working on windows
        pass
    else:
        window.SetForegroundColour(fg_color)
        window.SetBackgroundColour(bg_color)

    window.Refresh()

    for child in window.GetChildren():
        apply_dark_mode(child, fg_color, bg_color, btn_color)


def is_dark_mode():
    if sys.platform != "win32":
        return False
    else:
        try:
            import winreg
            key_path = r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path) as key:
                value, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
                return value == 0
        except: # noqa
            return False


def set_tooltip_long_hover():
    """By default wx auto-hides a tooltip after a few seconds (~5s on Windows) even
    while the mouse is still sitting on the control.

    NOTE: this alone does NOT reliably achieve "stays open until you move away" on
    Windows -- the native Win32 Common Controls tooltip stores its auto-pop delay
    internally as a 16-bit value, silently clamping anything requested above roughly
    32767ms (~33s) regardless of what's asked for here. A first attempt at this asked
    for 600000ms (10 minutes) and was silently capped, which is why tooltips kept
    disappearing well before that even after this was called. Kept at the platform's
    real safe maximum as a harmless baseline/fallback (covers wx.RadioBox per-item
    tooltips, which enable_persistent_tooltips below cannot manage generically) --
    but the actual fix for "stays open until the cursor leaves" is
    enable_persistent_tooltips(), which replaces the native tooltip entirely with a
    custom popup that has no OS-imposed timeout at all."""
    wx.ToolTip.SetAutoPop(32767)


class _TooltipPopup(wx.PopupWindow):
    """A borderless popup styled to look like a native tooltip, used in place of
    wx.ToolTip specifically because the native one cannot be kept open longer than
    Windows' own ~33s cap (see set_tooltip_long_hover). This has NO timeout of its
    own at all -- it is shown on mouse-enter and hidden on mouse-leave, so it
    genuinely stays open for as long as the cursor sits on the control, no matter
    how long that is, and disappears the instant the cursor actually moves away."""
    def __init__(self, parent, text):
        super().__init__(parent, flags=wx.BORDER_SIMPLE)
        bg = wx.SystemSettings.GetColour(wx.SYS_COLOUR_INFOBK)
        fg = wx.SystemSettings.GetColour(wx.SYS_COLOUR_INFOTEXT)
        panel = wx.Panel(self)
        panel.SetBackgroundColour(bg)
        label = wx.StaticText(panel, label=text)
        label.SetForegroundColour(fg)
        label.Wrap(480)
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(label, 0, wx.ALL, 6)
        panel.SetSizer(sizer)
        panel.Fit()
        self.SetClientSize(panel.GetSize())


class _PersistentTooltipManager():
    """Owns exactly one live _TooltipPopup at a time across a whole frame. A short
    (500ms) delay before showing avoids flicker when the cursor just passes over a
    control without pausing on it, matching normal tooltip feel -- but once shown,
    nothing hides it except the cursor actually leaving that control (EVT_LEAVE_WINDOW),
    so there is no analog of the native tooltip's auto-pop timeout at all here."""
    SHOW_DELAY_MS = 500

    def __init__(self):
        self.popup = None
        self.timer = wx.Timer()
        self.timer.Bind(wx.EVT_TIMER, self._on_timer)
        self.pending = None

    def bind(self, widget, text):
        widget.SetToolTip(None)  # avoid the native tooltip double-showing alongside this one
        widget.Bind(wx.EVT_ENTER_WINDOW, lambda evt, w=widget, t=text: self._on_enter(w, t))
        widget.Bind(wx.EVT_LEAVE_WINDOW, lambda evt, w=widget: self._on_leave(w))
        widget.Bind(wx.EVT_WINDOW_DESTROY, lambda evt, w=widget: self._on_leave(w))

    def _on_enter(self, widget, text):
        self.pending = (widget, text)
        self.timer.StartOnce(self.SHOW_DELAY_MS)

    def _on_timer(self, event):
        if self.pending is None:
            return
        widget, text = self.pending
        self.pending = None
        if not widget or not widget.IsShownOnScreen():
            return
        self._hide()
        self.popup = _TooltipPopup(widget, text)
        pos = widget.ClientToScreen((0, widget.GetSize().GetHeight()))
        self.popup.SetPosition(pos)
        self.popup.Show()

    def _on_leave(self, widget):
        self.pending = None
        self.timer.Stop()
        self._hide()

    def _hide(self):
        if self.popup is not None:
            popup, self.popup = self.popup, None
            popup.Destroy()


def enable_persistent_tooltips(window):
    """Walks every control under `window` and, for each one that already has a
    tooltip set via SetToolTip(...), switches it from the native OS tooltip (capped
    at ~33s on Windows no matter what set_tooltip_long_hover asks for) to a custom
    popup that stays open for exactly as long as the cursor remains on that control,
    with no timeout at all -- only actually moving the cursor away closes it.

    Call this ONCE, after every control in the window has already had its tooltip
    text set (e.g. right after constructing MainFrame), not before -- it captures
    each control's tooltip text at the time it's called.

    Known gap: wx.RadioBox's per-item tooltips (SetItemToolTip) are drawn by the
    native control itself with no separate wx.Window per radio item to bind to, so
    they can't be managed generically this way and keep relying on the native
    ~33s-capped tooltip instead."""
    manager = _PersistentTooltipManager()
    window._persistent_tooltip_manager = manager  # keep it alive with the frame

    def _walk(w):
        tip = w.GetToolTip()
        if tip is not None and tip.GetTip():
            manager.bind(w, tip.GetTip())
        for child in w.GetChildren():
            _walk(child)

    _walk(window)


def init_win32_dpi():
    if sys.platform == "win32":
        import ctypes
        try:
            # Fix mouse position when Display Scaling is not 100%
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except: # noqa
            pass


def refresh_layouts(window):
    window.InvalidateBestSize()
    for child in window.GetChildren():
        refresh_layouts(child)
    window.Layout()
    window.Fit()
