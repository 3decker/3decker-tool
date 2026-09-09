import os
import sys
from os import path
import hashlib
import json
from nunif.utils.home_dir import ensure_home_dir, is_nunif_home_set


def md5(s):
    return hashlib.md5((s + "iw3").encode()).hexdigest()


def get_cache_dir():
    if is_nunif_home_set():
        cache_dir = path.join(ensure_home_dir("iw3"), "scene_cache")
    else:
        cache_dir = path.join(path.dirname(__file__), "..", "tmp", "iw3_scene_cache")

    if not path.exists(cache_dir):
        os.makedirs(cache_dir)
    return cache_dir


def get_cache_path(input_video_path, max_fps, cache_dir=None):
    cache_dir = cache_dir or get_cache_dir()

    max_fps_key = str(max_fps)
    path_key = path.abspath(input_video_path)
    size_key = str(path.getsize(input_video_path))
    mtime_key = str(path.getmtime(input_video_path))
    param = f"{max_fps_key} {path_key} {size_key} {mtime_key}"
    cache_filename = md5(param) + ".json"

    cache_path = path.join(cache_dir, cache_filename)
    return cache_path


def save_cache_with_filename(cache_path, input_video_path, pts, max_fps, start_time, end_time):
    parent_dir = path.dirname(cache_path)
    if not path.exists(parent_dir):
        os.makedirs(parent_dir)
    data = {
        "pts": sorted(list(pts)),
        "max_fps": max_fps,
        "start_time": start_time,
        "end_time": end_time,
    }
    with open(cache_path, mode="w", encoding="utf-8") as f:
        json.dump(data, f)


def save_cache(input_video_path, pts, max_fps, start_time, end_time, cache_dir=None):
    cache_path = get_cache_path(input_video_path, max_fps, cache_dir=cache_dir)
    save_cache_with_filename(
        cache_path, input_video_path, pts,
        max_fps=max_fps,
        start_time=start_time,
        end_time=end_time
    )


def time_to_sec(val, default):
    if val is None:
        return default
    try:
        parts = str(val).split(":")
        if len(parts) > 3:
            raise ValueError
        units = [1, 60, 3600]
        total_sec = sum(float(c) * u for c, u in zip(reversed(parts), units))
        return max(total_sec, 0)
    except (ValueError, TypeError):
        raise ValueError("time must be hh:mm:ss, mm:ss or ss format")


def is_within_range(data, start_time, end_time):
    data_start = time_to_sec(data.get("start_time"), 0)
    data_end = time_to_sec(data.get("end_time"), float("inf"))
    query_start = time_to_sec(start_time, 0)
    query_end = time_to_sec(end_time, float("inf"))
    return data_start <= query_start and query_end <= data_end


def try_load_cache_with_filename(cache_path, input_video_path, max_fps, start_time, end_time):
    if path.exists(cache_path):
        try:
            with open(cache_path, mode="r", encoding="utf-8") as f:
                data = json.load(f)
            if not is_within_range(data, start_time, end_time):
                return None

            return set(data["pts"])
        except Exception as e:  # noqa
            print(f"{cache_path}: {e}", file=sys.stderr)
            return None
    else:
        return None


def get_cached_range_with_filename(cache_path):
    """
    Returns the (start_time, end_time) already covered by the cache file, regardless
    of whether it covers the currently requested range. Used to widen a rescan so the
    cache only ever grows instead of being overwritten with a narrower range.
    """
    if path.exists(cache_path):
        try:
            with open(cache_path, mode="r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("start_time"), data.get("end_time")
        except Exception:
            return None
    else:
        return None


def get_cached_range(input_video_path, max_fps, cache_dir=None):
    cache_path = get_cache_path(input_video_path, max_fps=max_fps, cache_dir=cache_dir)
    return get_cached_range_with_filename(cache_path)


def try_load_cache(input_video_path, max_fps, start_time, end_time, cache_dir=None):
    cache_path = get_cache_path(input_video_path, max_fps=max_fps, cache_dir=cache_dir)
    return try_load_cache_with_filename(
        cache_path, input_video_path,
        max_fps=max_fps,
        start_time=start_time,
        end_time=end_time
    )


def purge_cache(input_video_path, max_fps):
    cache_path = get_cache_path(input_video_path, max_fps)
    if path.exists(cache_path):
        os.unlink(cache_path)


def list_cache_files(cache_dir=None):
    cache_dir = cache_dir or get_cache_dir()
    return (path.join(cache_dir, fn)
            for fn in os.listdir(cache_dir)
            if fn.endswith(".json"))


def purge_cache_all(cache_dir=None):
    # NOTE: with cache_dir=None (the default) this deletes EVERY cached scene-boundary
    # scan for EVERY video the user has ever run --scene-detect on -- real, expensive,
    # regenerable-but-slow-to-rebuild data. Callers that need an isolated sandbox
    # (tests especially) MUST pass an explicit cache_dir, never rely on the default.
    for cache_path in list_cache_files(cache_dir=cache_dir):
        os.unlink(cache_path)


def _test():
    import unittest
    import tempfile
    import shutil

    class TestSceneCache(unittest.TestCase):
        def setUp(self):
            # Isolated temp cache_dir, passed explicitly to every call below -- this
            # test must NEVER touch the real production cache dir (get_cache_dir()'s
            # default). It used to call the no-argument purge_cache()/save_cache()/
            # try_load_cache() forms, which silently operated on that real directory
            # and wiped every real cached scene-detection scan (for every video the
            # user had ever run --scene-detect on) each time this test ran -- caught
            # for real during the ADR-059 investigation. See also list_cache_files()/
            # purge_cache_all()'s cache_dir parameter, added for the same reason.
            self.cache_dir = tempfile.mkdtemp(prefix="iw3_scene_cache_test_")
            self.tmp_dir = tempfile.mkdtemp(prefix="iw3_scene_cache_test_video_")
            self.video_path = path.join(self.tmp_dir, "scene_cache_test.mp4")
            self.max_fps = 30
            self.pts = [1, 2, 3, 4, 5]
            with open(self.video_path, "w") as f:
                f.write("dummy data")

        def tearDown(self):
            shutil.rmtree(self.tmp_dir, ignore_errors=True)
            shutil.rmtree(self.cache_dir, ignore_errors=True)

        def test_save_and_load_success(self):
            save_cache(self.video_path, self.pts, self.max_fps, "00:00:00", "00:00:10",
                       cache_dir=self.cache_dir)
            loaded_pts = try_load_cache(self.video_path, self.max_fps, "00:00:01", "00:00:05",
                                        cache_dir=self.cache_dir)
            # try_load_cache_with_filename() returns set(data["pts"]) -- compare against
            # a set, not the original list (pre-existing test bug, unrelated to this
            # investigation's fix; was failing before any change made here).
            self.assertEqual(loaded_pts, set(self.pts))

        def test_out_of_range(self):
            save_cache(self.video_path, self.pts, self.max_fps, "00:00:05", "00:00:10",
                       cache_dir=self.cache_dir)

            self.assertIsNone(try_load_cache(self.video_path, self.max_fps, "00:00:00", "00:00:10",
                                             cache_dir=self.cache_dir))
            self.assertIsNone(try_load_cache(self.video_path, self.max_fps, "00:00:05", "00:00:15",
                                             cache_dir=self.cache_dir))

        def test_corrupted_json(self):
            cache_path = get_cache_path(self.video_path, self.max_fps, cache_dir=self.cache_dir)

            with open(cache_path, "w") as f:
                f.write("{ invalid json ...")
            result = try_load_cache(self.video_path, self.max_fps, "00:00:00", "00:00:10",
                                    cache_dir=self.cache_dir)
            self.assertIsNone(result)

        def test_file_modified(self):
            save_cache(self.video_path, self.pts, self.max_fps, "00:00:00", "00:00:10",
                       cache_dir=self.cache_dir)

            mtime = os.path.getmtime(self.video_path)
            os.utime(self.video_path, (mtime + 1, mtime + 1))

            result = try_load_cache(self.video_path, self.max_fps, "00:00:00", "00:00:10",
                                    cache_dir=self.cache_dir)
            self.assertIsNone(result)

    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    suite.addTests(loader.loadTestsFromTestCase(TestSceneCache))
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)


if __name__ == "__main__":
    _test()
