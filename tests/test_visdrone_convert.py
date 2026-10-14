"""Dataset converter tests against a synthetic VisDrone-shaped directory.

No download: a tiny valid PNG is generated with zlib/struct so the converter's
image-size path works without Pillow or opencv.
"""

import os
import struct
import sys
import tempfile
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import visdrone_to_yolo as V

CAR = 4          # documented VisDrone index; the converter VERIFIES it (A1)


def write_png(path, w, h):
    """Minimal single-colour RGB PNG."""
    raw = b"".join(b"\x00" + b"\x7f" * (w * 3) for _ in range(h))

    def chunk(tag, data):
        body = tag + data
        return (struct.pack(">I", len(data)) + body
                + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    with open(path, "wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
                 + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def make_split(root, lines_per_image, w=200, h=100):
    img_dir = os.path.join(root, "images")
    ann_dir = os.path.join(root, "annotations")
    os.makedirs(img_dir)
    os.makedirs(ann_dir)
    for i, lines in enumerate(lines_per_image):
        stem = "img{:03d}".format(i)
        write_png(os.path.join(img_dir, stem + ".png"), w, h)
        with open(os.path.join(ann_dir, stem + ".txt"), "w") as fh:
            fh.write("\n".join(lines) + "\n")
    return root


def test_png_size_reader():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "a.png")
        write_png(p, 321, 123)
        assert V.image_size(p) == (321, 123)


def test_parse_annotation_fields():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "a.txt")
        open(p, "w").write("10,20,30,40,1,4,0,0\n50,60,5,5,0,4,1,2\n")
        rows = V.parse_annotation(p)
        assert len(rows) == 2
        assert rows[0]["left"] == 10 and rows[0]["category"] == 4
        assert rows[1]["score"] == 0


def test_xywh_conversion_is_normalised_and_centred():
    with tempfile.TemporaryDirectory() as tmp:
        src = make_split(os.path.join(tmp, "src"),
                         [["0,0,100,50,1,{},0,0".format(CAR)]], w=200, h=100)
        dst = os.path.join(tmp, "dst")
        V.convert_split(src, dst, "train", CAR)
        lbl = os.path.join(dst, "labels", "train", "img000.txt")
        cls, cx, cy, bw, bh = open(lbl).read().split()
        assert cls == "0"
        assert abs(float(cx) - 0.25) < 1e-6      # (0 + 100/2) / 200
        assert abs(float(cy) - 0.25) < 1e-6      # (0 +  50/2) / 100
        assert abs(float(bw) - 0.5) < 1e-6
        assert abs(float(bh) - 0.5) < 1e-6


def test_only_car_class_is_kept_no_merging():
    """DECISION D6 -- van(5)/truck(6)/bus(9) must NOT be merged into car."""
    with tempfile.TemporaryDirectory() as tmp:
        src = make_split(os.path.join(tmp, "src"), [[
            "0,0,20,20,1,4,0,0",    # car    -> kept
            "0,0,20,20,1,5,0,0",    # van    -> dropped
            "0,0,20,20,1,6,0,0",    # truck  -> dropped
            "0,0,20,20,1,9,0,0",    # bus    -> dropped
            "0,0,20,20,1,1,0,0",    # pedestrian -> dropped
        ]])
        dst = os.path.join(tmp, "dst")
        st = V.convert_split(src, dst, "train", CAR)
        assert st["boxes_kept"] == 1
        assert st["boxes_dropped_other_class"] == 4


def test_score_zero_boxes_are_dropped_by_default():
    """OUR ASSUMPTION A6 -- VisDrone ignored regions are not objects."""
    with tempfile.TemporaryDirectory() as tmp:
        src = make_split(os.path.join(tmp, "src"), [[
            "0,0,20,20,1,4,0,0",     # kept
            "30,30,20,20,0,4,0,0",   # score 0 -> dropped
        ]])
        dst = os.path.join(tmp, "dst")
        st = V.convert_split(src, dst, "train", CAR)
        assert st["boxes_kept"] == 1
        assert st["boxes_dropped_score_zero"] == 1


def test_keep_score_zero_override():
    with tempfile.TemporaryDirectory() as tmp:
        src = make_split(os.path.join(tmp, "src"),
                         [["0,0,20,20,0,4,0,0"]])
        dst = os.path.join(tmp, "dst")
        st = V.convert_split(src, dst, "train", CAR, drop_score_zero=False)
        assert st["boxes_kept"] == 1
        assert st["boxes_dropped_score_zero"] == 0


def test_truncation_and_occlusion_levels_are_all_kept():
    """OUR ASSUMPTION A7."""
    with tempfile.TemporaryDirectory() as tmp:
        src = make_split(os.path.join(tmp, "src"), [[
            "0,0,20,20,1,4,0,0", "30,0,20,20,1,4,1,1", "60,0,20,20,1,4,2,2",
        ]])
        dst = os.path.join(tmp, "dst")
        st = V.convert_split(src, dst, "train", CAR)
        assert st["boxes_kept"] == 3


def test_degenerate_boxes_are_dropped():
    with tempfile.TemporaryDirectory() as tmp:
        src = make_split(os.path.join(tmp, "src"), [[
            "0,0,0,20,1,4,0,0", "0,0,20,0,1,4,0,0", "0,0,20,20,1,4,0,0",
        ]])
        dst = os.path.join(tmp, "dst")
        st = V.convert_split(src, dst, "train", CAR)
        assert st["boxes_kept"] == 1
        assert st["boxes_dropped_degenerate"] == 2


def test_empty_label_file_written_for_images_without_cars():
    with tempfile.TemporaryDirectory() as tmp:
        src = make_split(os.path.join(tmp, "src"), [["0,0,20,20,1,1,0,0"]])
        dst = os.path.join(tmp, "dst")
        st = V.convert_split(src, dst, "train", CAR)
        lbl = os.path.join(dst, "labels", "train", "img000.txt")
        assert os.path.isfile(lbl) and open(lbl).read() == ""
        assert st["empty_label_files"] == 1


def test_category_survey_supports_a1_verification():
    """A1: the car index is VERIFIED against the data, never assumed."""
    with tempfile.TemporaryDirectory() as tmp:
        src = make_split(os.path.join(tmp, "src"), [[
            "0,0,20,20,1,4,0,0", "0,0,20,20,1,4,0,0", "0,0,20,20,1,1,0,0",
        ]])
        counts, nfiles = V.survey_categories(os.path.join(src, "annotations"))
        assert nfiles == 1
        assert counts[4] == 2 and counts[1] == 1
        assert 4 in counts          # what --verify-index asserts before converting


def test_visdrone_category_table_matches_the_paper_enumeration():
    """PAPER FACT 3.1.2 lists the 10 object categories in this order."""
    names = [V.VISDRONE_CATEGORIES[i] for i in range(1, 11)]
    assert names == ["pedestrian", "people", "bicycle", "car", "van", "truck",
                     "tricycle", "awning-tricycle", "bus", "motor"]


def test_summarise_reports_distribution():
    st = {"images": 2, "boxes_per_image": [1, 3], "box_area_frac": [0.1, 0.2, 0.3]}
    out = V.summarise(st)
    assert out["boxes_per_image_mean"] == 2.0
    assert out["boxes_per_image_max"] == 3
    assert "box_area_frac_median" in out
