from packaging import version as packaging_version
import torch
import warnings
from datetime import datetime, timezone
from collections import OrderedDict
import torch.nn as nn
from . register import create_model
from . model import Model
from .. logger import logger
from .. device import create_device, autocast


PYTORCH2 = packaging_version.parse(torch.__version__).major >= 2


def save_model(model, model_path, updated_at=None, train_kwargs=None, **kwargs):
    if isinstance(model, nn.DataParallel):
        model = model.module
    assert (isinstance(model, Model))
    updated_at = str(updated_at or datetime.now(timezone.utc))
    if train_kwargs is not None:
        if not isinstance(train_kwargs, dict):
            # Namespace to dict
            train_kwargs = vars(train_kwargs)
        # Remove data that probably cannot be loaded in other environments
        # The current intended target is a handler
        remove_keys = [k for k in train_kwargs.keys()
                       if callable(train_kwargs[k])]
        for k in remove_keys:
            train_kwargs.pop(k)

    data = {
        "nunif_model": 1,
        "name": model.name,
        "updated_at": updated_at,
        "kwargs": model.get_kwargs(),
        "train_kwargs": train_kwargs,
        "state_dict": model.state_dict()}
    data.update(kwargs)
    torch.save(data, model_path)


def load_model(model_path, model=None, device_ids=None,
               strict=True, map_location="cpu", weights_only=False):
    if not PYTORCH2:
        # Disabled due to https://github.com/pytorch/pytorch/issues/94670
        weights_only = False
    if "mps" in str(map_location):
        map_location = "cpu"
    if model_path.startswith("http://") or model_path.startswith("https://"):
        # force weights_only=True to avoid security risk
        data = torch.hub.load_state_dict_from_url(model_path, weights_only=True, map_location=map_location)
    else:
        data = torch.load(model_path, map_location=map_location, weights_only=weights_only)

    assert ("nunif_model" in data)
    if model is None:
        model = create_model(data["name"], device_ids=device_ids, **data["kwargs"])
        model_predefine = False
    else:
        model_predefine = True
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(data["state_dict"], strict=strict)
    else:
        model.load_state_dict(data["state_dict"], strict=strict)
    logger.debug(f"load: {model.name} from {model_path}")
    if "updated_at" in data:
        model.updated_at = data["updated_at"]
    data.pop("state_dict")

    if not model_predefine and device_ids is not None:
        device = create_device(device_ids)
        model = model.to(device)

    return model, data


def get_model_kwargs(model, key=None):
    if isinstance(model, nn.DataParallel):
        model = model.module
    kwargs = model.get_kwargs()
    if key is None:
        return kwargs
    else:
        return kwargs[key]


def get_model_device(model):
    if isinstance(model, nn.DataParallel):
        model = model.module
    if hasattr(model, "get_device"):
        return model.get_device()
    else:
        return next(model.parameters()).device


_COMPILER_SUPPORTED_DEVICES = {}


def _test_func(x):
    return x + 1.0


def check_compile_support(device):
    device_name = device if isinstance(device, str) else device.type
    if device_name not in _COMPILER_SUPPORTED_DEVICES:
        try:
            model = torch.nn.Linear(32, 32, bias=False)
            model.weight.data.zero_()
            model = torch.compile(model.eval().to(device))
            func = torch.compile(_test_func)
            with torch.inference_mode(), autocast(device):
                model(torch.zeros((1, 32), device=device))
                func(torch.zeros((32,), dtype=torch.float32, device=device))
            _COMPILER_SUPPORTED_DEVICES[device_name] = True
        except Exception as e:
            # Any failure during this real compiler-toolchain probe (RuntimeError,
            # AssertionError, or an OSError from a Windows-level CreateProcess/cl.exe
            # invocation failure when the MSVC/Triton toolchain is missing or broken)
            # means torch.compile is not usable on this device right now.
            logger.debug(f"check_compile_support: {device_name}: {e.__class__.__name__}: {e}")
            _COMPILER_SUPPORTED_DEVICES[device_name] = False

    return _COMPILER_SUPPORTED_DEVICES[device_name]


_COMPILE_FALLBACK_WARNED = False


def _enable_compile_fallback():
    # check_compile_support()'s probe (ADR-068) only proves a tiny throwaway model can
    # be built-and-run on this device -- torch.compile(model, ...) below only *wraps*
    # the real model, it does not run the real compiler toolchain yet. That real
    # invocation (Triton/cl.exe) is deferred by torch to the model's first real
    # forward() call, deep inside whatever real pipeline (iw3 video/image processing,
    # waifu2x, etc.) goes on to use this model -- so a probe pass here does not
    # guarantee the real compile of a real, larger model graph will also succeed
    # (ADR-068 amendment). suppress_errors makes torch fall back to eager execution
    # for the failing call instead of raising BackendCompilerFailed all the way up
    # into caller code that has no reason to expect torch.compile could ever crash it.
    global _COMPILE_FALLBACK_WARNED
    torch._dynamo.config.suppress_errors = True
    if not _COMPILE_FALLBACK_WARNED:
        _COMPILE_FALLBACK_WARNED = True
        warnings.warn(
            "torch.compile is enabled. If the real compiler toolchain fails to build "
            "a model during this run, processing will automatically continue without "
            "the torch.compile speedup instead of stopping -- see docs/torch_compile.md.")


def compile_model(model, device=None, **kwargs):
    device = device or get_model_device(model)

    if not is_compiled_model(model) and check_compile_support(device):
        logger.debug(f"compile {model.__class__.__name__}, kwargs={kwargs}")
        try:
            _enable_compile_fallback()
            model = torch.compile(model, **kwargs)
        except Exception as e:
            # Any failure while *wrapping* the model (as opposed to the real compile
            # failures suppress_errors above handles once this model actually runs)
            # -- treat it the same way: keep the original, uncompiled model instead of
            # crashing whatever real pipeline called this.
            logger.debug(f"compile_model: {model.__class__.__name__}: {e.__class__.__name__}: {e}")
    return model


def compile_function(func, device, **kwargs):
    if not is_compiled_function(func) and check_compile_support(device):
        logger.debug(f"compile {func.__name__}, kwargs={kwargs}")
        try:
            _enable_compile_fallback()
            func = torch.compile(func, **kwargs)
        except Exception as e:
            logger.debug(f"compile_function: {func.__name__}: {e.__class__.__name__}: {e}")
    return func


def is_compiled_function(func):
    return hasattr(func, "_torchdynamo_orig_callable")


def is_compiled_model(model):
    # TODO: class name of compiled model is unclear
    return hasattr(model, "_orig_mod")


def merge_state_dict(a, b, alpha=0.5):
    """
    NOTE: This only works when `a` and `b` are finetuned models of the same original model.
          Also constraints may be broken. Should always be verified to work.
    """
    assert a.keys() == b.keys()
    c = OrderedDict()
    for k in a.keys():
        c[k] = a[k] * alpha + b[k] * (1. - alpha)
    return c


def mean_state_dict(dicts):
    assert len(dicts) > 0
    a = dicts[0]
    assert all(a.keys() == d.keys() for d in dicts)
    mean = OrderedDict()
    scale = 1. / len(dicts)
    for k in a.keys():
        for d in dicts:
            if k not in mean:
                mean[k] = d[k] * scale
            else:
                mean[k] += d[k] * scale
    return mean


def _test_check_compile():
    print(check_compile_support("cpu"))
    print(check_compile_support(torch.device("cpu")))
    print(check_compile_support(torch.device("cuda")))
    print(check_compile_support(torch.device("cuda:1")))
    print(check_compile_support("cuda:0"))
    print(check_compile_support("cuda:10"))
    print(check_compile_support("mps"))
    print(check_compile_support("xpu"))
    print(_COMPILER_SUPPORTED_DEVICES)


def _test_is_compiled_model(device):
    model = torch.nn.Linear(32, 32, bias=False).to(device)
    assert not is_compiled_model(model)
    compiled_model = compile_model(model)
    assert is_compiled_model(compiled_model)
    assert not is_compiled_model(model)
    model(torch.zeros(1, 32).to(device))
    assert is_compiled_model(compiled_model)
    assert not is_compiled_model(model)


if __name__ == "__main__":
    _test_check_compile()
    _test_is_compiled_model("cpu")
    _test_is_compiled_model("cuda")
