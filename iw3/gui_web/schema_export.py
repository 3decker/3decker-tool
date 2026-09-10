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


_DYNAMIC_CHOICE_PROVIDERS = {
    "hw_devices": _hw_device_choices,
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
            assert set(f.choices) <= set(action.choices), (
                f"schema field {f.name!r} has choices not present in the real "
                f"parser action's choices: {set(f.choices) - set(action.choices)}"
            )

    print("_self_test_schema_fields_exist_on_real_parser: PASS")


def _self_test_stereo_format_field_has_no_cli_arg():
    """Documents/guards the one deliberate special case -- if this ever
    trips, either stereo_format grew a real 1:1 CLI arg (update worker.py's
    special-casing away) or someone accidentally gave it one by mistake."""
    stereo_format = next(f for f in FIELDS if f.name == "stereo_format")
    assert stereo_format.cli_arg is None, \
        "stereo_format is expected to be the one schema field worker.py " \
        "special-cases via _apply_stereo_format() -- if it now has a real " \
        "cli_arg, that special-casing needs to be reconciled, not left stale."
    print("_self_test_stereo_format_field_has_no_cli_arg: PASS")


def _run_self_tests():
    _self_test_schema_fields_exist_on_real_parser()
    _self_test_stereo_format_field_has_no_cli_arg()
    print("All iw3.gui_web.schema_export self-tests PASSED")


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv[1:]:
        _run_self_tests()
    else:
        out = write_schema_json()
        print(f"wrote {out}")
