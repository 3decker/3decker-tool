"""Serializes schema.py -> JSON for the JS renderer, and self-checks the
schema against iw3.utils.create_parser()'s real registered arguments.

Coverage direction for Phase 1 (see schema.py's own docstring): this asserts
every schema Field with a real cli_arg actually exists on the parser, with
matching choices where the parser defines any. It does NOT yet assert the
reverse (every parser arg has schema coverage) -- that full-coverage check
is a Phase 2+ concern once the schema actually covers all 132 flags; doing
it now against a deliberately 7-field slice would just fail loudly for
every field Phase 1 hasn't ported yet, telling us nothing new.
"""
import json
from os import path

from iw3.utils import create_parser
from iw3.gui_web.schema import FIELDS, TABS


def _hw_device_choices():
    # Real choice list is only known at runtime (whatever ffmpeg on this
    # machine actually reports) -- see schema.py's Field.dynamic_choices.
    from nunif.utils.video import HW_DEVICES
    return list(HW_DEVICES)


def _scene_batch_ema_model_choices():
    # Same reasoning as hwaccel -- read from the one real source of truth
    # (iw3.scene_batch) instead of a copy that could go stale, exactly like
    # create_parser() itself does for --scene-batch-auto-ema-model.
    from iw3.scene_batch import EMA_BY_DURATION_TABLES
    return list(EMA_BY_DURATION_TABLES.keys())


_DYNAMIC_CHOICE_PROVIDERS = {
    "hw_devices": _hw_device_choices,
    "scene_batch_ema_models": _scene_batch_ema_model_choices,
}


def build_schema_dict():
    fields = []
    for f in FIELDS:
        d = f.to_dict()
        if f.dynamic_choices:
            d["choices"] = _DYNAMIC_CHOICE_PROVIDERS[f.dynamic_choices]()
        fields.append(d)
    return {
        "tabs": [{"key": key, "label": label} for key, label in TABS],
        "fields": fields,
    }


def write_schema_json(out_path=None):
    if out_path is None:
        out_path = path.join(path.dirname(__file__), "public", "schema.json")
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(build_schema_dict(), fp, indent=2)
    return out_path


def _self_test_schema_fields_exist_on_real_parser():
    """Every Field with a real cli_arg must correspond to an actual
    create_parser() argument, with matching choices when the parser
    restricts them -- this is what keeps schema.py from silently drifting
    out of sync with iw3's real CLI surface."""
    parser = create_parser(required_true=False)
    actions_by_flag = {}
    for action in parser._actions:
        for opt in action.option_strings:
            actions_by_flag[opt] = action

    for f in FIELDS:
        if f.cli_arg is None:
            # deliberately-special-cased fields (stereo_format today) --
            # worker.py owns the mapping, not the parser.
            continue
        assert f.cli_arg in actions_by_flag, \
            f"schema field {f.name!r} references {f.cli_arg!r}, which does not " \
            f"exist on create_parser() -- has the real CLI flag been renamed/removed?"
        action = actions_by_flag[f.cli_arg]
        if action.choices and f.choices and not f.dynamic_choices:
            # Compared as strings, not raw values: HTML <select> options are
            # always strings even when the real parser choice is an int
            # (e.g. --max-workers' [0, 1, 2, ...]) -- a type mismatch here
            # would be a false positive, not a real drift. "" is schema-only
            # UI sugar for "leave this blank/unset" and isn't expected to be
            # a real parser choice, so it's excluded from the comparison.
            schema_choices = {str(c) for c in f.choices if c != ""}
            parser_choices = {str(c) for c in action.choices}
            assert schema_choices <= parser_choices, (
                f"schema field {f.name!r} has choices not present in the real "
                f"parser action's choices: {schema_choices - parser_choices}"
            )

    print("_self_test_schema_fields_exist_on_real_parser: PASS")


def _self_test_special_cased_fields_have_no_cli_arg():
    """Documents/guards the fields worker.py deliberately special-cases
    instead of mapping 1:1 to a parser arg (stereo_format, anaglyph_method,
    metadata -- see schema.py's own docstring for why) -- if any of these
    ever trips, either that field grew a real 1:1 CLI arg (update worker.py's
    special-casing away) or someone accidentally gave it one by mistake."""
    special_cased = {"stereo_format", "anaglyph_method", "metadata"}
    for f in FIELDS:
        if f.name in special_cased:
            assert f.cli_arg is None, (
                f"schema field {f.name!r} is expected to be special-cased in "
                f"worker.py (no 1:1 cli_arg) -- if it now has a real cli_arg, "
                f"that special-casing needs to be reconciled, not left stale."
            )
    print("_self_test_special_cased_fields_have_no_cli_arg: PASS")


def _self_test_dynamic_choice_providers_resolve():
    """Every dynamic_choices name a Field references must have a real
    provider, and that provider must actually return something -- catches a
    typo'd dynamic_choices string or a provider that silently returns
    nothing useful."""
    for f in FIELDS:
        if not f.dynamic_choices:
            continue
        assert f.dynamic_choices in _DYNAMIC_CHOICE_PROVIDERS, (
            f"schema field {f.name!r} references dynamic_choices="
            f"{f.dynamic_choices!r}, which has no registered provider"
        )
        result = _DYNAMIC_CHOICE_PROVIDERS[f.dynamic_choices]()
        assert result, f"dynamic_choices provider {f.dynamic_choices!r} (for field {f.name!r}) returned nothing"
    print("_self_test_dynamic_choice_providers_resolve: PASS")


def _run_self_tests():
    _self_test_schema_fields_exist_on_real_parser()
    _self_test_special_cased_fields_have_no_cli_arg()
    _self_test_dynamic_choice_providers_resolve()
    print("All iw3.gui_web.schema_export self-tests PASSED")


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv[1:]:
        _run_self_tests()
    else:
        out = write_schema_json()
        print(f"wrote {out}")
