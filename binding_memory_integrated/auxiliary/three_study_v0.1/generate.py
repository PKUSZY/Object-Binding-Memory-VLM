"""Deterministic, programmatically rendered object-binding benchmark. Python 3.9+."""
import argparse
import hashlib
import itertools
import json
import math
import platform
import random
import shutil
from collections import Counter
from pathlib import Path

import PIL
from PIL import Image, ImageChops, ImageDraw
from difficulty import SCHEME, annotations, write_index

VERSION = "0.1.0-preview"
DIMS = ("color", "shape", "texture")
COLORS = {
    "red": (213, 66, 68), "blue": (48, 114, 202),
    "green": (42, 153, 99), "yellow": (232, 190, 43),
    "purple": (142, 84, 186), "orange": (231, 123, 38),
}
SHAPES = ("circle", "square", "triangle", "star", "cross", "semicircle")
TEXTURES = ("solid", "horizontal_stripes", "vertical_stripes", "diagonal_stripes", "dots", "checkerboard")
VALUES = (tuple(COLORS), SHAPES, TEXTURES)
ZH = {
    "red": "红色", "blue": "蓝色", "green": "绿色", "yellow": "黄色", "purple": "紫色", "orange": "橙色",
    "circle": "圆形", "square": "正方形", "triangle": "三角形", "star": "五角星", "cross": "十字形", "semicircle": "半圆形",
    "solid": "纯色", "horizontal_stripes": "横条纹", "vertical_stripes": "竖条纹", "diagonal_stripes": "斜条纹", "dots": "圆点", "checkerboard": "棋盘格",
}
TYPES = ("target", "critical_2plus1", "critical_1plus1plus1", "near_2seen", "unrelated")
TYPE_INFO = {
    "target": ("Target", "真正出现过", "YES"),
    "critical_2plus1": ("Critical · 2+1", "两项同源，一项重组", "NO"),
    "critical_1plus1plus1": ("Critical · 1+1+1", "三项来自不同物体", "NO"),
    "near_2seen": ("Near lure", "两项已见，一项全新", "NO"),
    "unrelated": ("Unrelated lure", "三项属性均未见", "NO"),
}
BACKGROUND = (245, 247, 249)
SIZE = 448
INSTRUCTION = (
    "请判断测试物体的颜色、形状和纹理组合，是否在学习阶段的某一张图片中完整出现过。"
    "同一物体按颜色、形状、纹理三项均相同定义；位置、大小和纹理细节位置的轻微变化不影响身份。"
    "只有三个属性同时匹配同一个学习物体才回答 YES，否则回答 NO。仅输出 YES 或 NO。"
)


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(dumps(value).encode()).hexdigest()


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(dumps(row) + "\n" for row in rows), encoding="utf-8")


def attributes(values):
    return dict(zip(DIMS, values))


def triple(obj):
    return tuple(obj[dim] for dim in DIMS)


def vertices(shape):
    if shape == "square":
        return [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    if shape == "triangle":
        return [(0, -1), (math.sqrt(3) / 2, .5), (-math.sqrt(3) / 2, .5)]
    if shape == "star":
        return [((1 if k % 2 == 0 else .48) * math.cos(-math.pi/2 + k*math.pi/5),
                 (1 if k % 2 == 0 else .48) * math.sin(-math.pi/2 + k*math.pi/5)) for k in range(10)]
    if shape == "cross":
        a = 1 / 3
        return [(-a,-1),(a,-1),(a,-a),(1,-a),(1,a),(a,a),(a,1),(-a,1),(-a,a),(-1,a),(-1,-a),(-a,-a)]
    return None


def render_parameters(seed):
    rng = random.Random(seed)
    return {"seed": seed, "size": SIZE, "supersampling": 2,
            "center_x": SIZE/2 + rng.randint(-7, 7), "center_y": SIZE/2 + rng.randint(-7, 7),
            "area_pixels": round(47500 * rng.uniform(.95, 1.05)),
            "texture_phase_x": rng.randrange(24), "texture_phase_y": rng.randrange(24)}


def render_image(attrs, params):
    """Equal nominal object area; texture ink preserves the base color's hue."""
    ss = params["supersampling"]
    n = SIZE * ss
    cx, cy = params["center_x"] * ss, params["center_y"] * ss
    area = params["area_pixels"] * ss**2
    mask = Image.new("L", (n, n))
    draw = ImageDraw.Draw(mask)
    shape = attrs["shape"]
    if shape == "circle":
        r = math.sqrt(area / math.pi)
        draw.ellipse((cx-r, cy-r, cx+r, cy+r), fill=255)
    elif shape == "semicircle":
        r = math.sqrt(2 * area / math.pi)
        baseline = cy + 4*r/(3*math.pi)
        draw.pieslice((cx-r, baseline-r, cx+r, baseline+r), 180, 360, fill=255)
    else:
        points = vertices(shape)
        unit_area = abs(sum(points[i][0]*points[(i+1)%len(points)][1] - points[(i+1)%len(points)][0]*points[i][1]
                            for i in range(len(points)))) / 2
        r = math.sqrt(area / unit_area)
        draw.polygon([(cx+r*x, cy+r*y) for x,y in points], fill=255)
    bbox = mask.getbbox()
    if not bbox or min(bbox[:2]) < 16*ss or max(bbox[2:]) > n-16*ss:
        raise ValueError(f"Object clipped or too close to edge: {shape} {bbox}")
    base = COLORS[attrs["color"]]
    canvas = Image.new("RGB", (n, n), BACKGROUND)
    canvas.paste(Image.new("RGB", (n,n), base), mask=mask)
    texture = attrs["texture"]
    if texture != "solid":
        tex = Image.new("L", (n, n))
        pen = ImageDraw.Draw(tex)
        px, py = params["texture_phase_x"]*ss, params["texture_phase_y"]*ss
        pitch, width = 24*ss, 6*ss
        if texture == "horizontal_stripes":
            for y in range(py-pitch, n+pitch, pitch):
                pen.rectangle((0,y,n,y+width-1), fill=255)
        elif texture == "vertical_stripes":
            for x in range(px-pitch, n+pitch, pitch):
                pen.rectangle((x,0,x+width-1,n), fill=255)
        elif texture == "diagonal_stripes":
            # Lines of slope +1, with consistent perpendicular spacing.
            for offset in range(-n+px-2*pitch, n+2*pitch, round(pitch*math.sqrt(2))):
                pen.line((offset,0,offset+n,n), fill=255, width=width)
        elif texture == "dots":
            r = 6*ss
            for y in range(py-pitch, n+pitch, pitch):
                for x in range(px-pitch, n+pitch, pitch):
                    pen.ellipse((x-r,y-r,x+r,y+r), fill=255)
        elif texture == "checkerboard":
            cell = 20*ss
            for row,y in enumerate(range(py-cell, n+cell, cell)):
                for col,x in enumerate(range(px-cell, n+cell, cell)):
                    if (row+col)%2 == 0:
                        pen.rectangle((x,y,x+cell-1,y+cell-1), fill=255)
        else:
            raise ValueError(texture)
        # Checkerboard covers more area, so it uses weaker contrast.
        contrast = .34 if texture == "checkerboard" else .52
        ink = tuple(round(v*(1-contrast)) for v in base)
        canvas.paste(Image.new("RGB", (n,n), ink), mask=ImageChops.multiply(mask, tex))
    return canvas.resize((SIZE, SIZE), Image.Resampling.LANCZOS)


def image_record(out, attrs, seed, pixel_hashes):
    # Different floating-point area settings can rasterize to identical pixels.
    # Resample deterministically, so a rerendered target never reuses exact pixels.
    for attempt in range(100):
        render_seed = seed if attempt == 0 else int(digest({"seed":seed,"attempt":attempt})[:16],16)
        params = render_parameters(render_seed)
        image = render_image(attrs, params)
        pixel_hash = hashlib.sha256(image.tobytes()).hexdigest()
        if pixel_hash not in pixel_hashes:
            pixel_hashes.add(pixel_hash)
            break
    else:
        raise RuntimeError("Unable to render unique pixels")
    uid = digest({"version": VERSION, "attributes": attrs, "render": params})[:28]
    rel = f"images/{uid}.png"
    path = out / rel
    image.save(path, format="PNG", compress_level=6)
    return {"image_id": uid, "image": rel, "attributes": attrs, "render": params,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "pixel_sha256":pixel_hash,
            "render_resample_count":attempt}


def study_columns(n, rng):
    columns = []
    for values in VALUES:
        column = []
        while len(column) < n*3:
            deck = list(values)
            rng.shuffle(deck)
            column.extend(deck)
        columns.append(column[:n*3])
    return columns


def choose_new(values, rng, counts, kind, dim):
    choices = list(values)
    rng.shuffle(choices)
    return min(choices, key=lambda value: counts[(kind, dim, value)])


def query_specs(study, index, rng, counts):
    specs = []
    a = index % 3
    specs.append(("target", study[a], [a,a,a], None, a))
    changed = index % 3
    a = (index // 3) % 3
    b = (a + 1 + (index//9)%2) % 3
    q = list(study[a]); q[changed] = study[b][changed]
    sources = [a,a,a]; sources[changed] = b
    specs.append(("critical_2plus1", tuple(q), sources, DIMS[changed], a))
    donors = list(itertools.permutations(range(3)))[index % 6]
    specs.append(("critical_1plus1plus1", tuple(study[donors[d]][d] for d in range(3)), list(donors), None, None))
    changed = (index+1) % 3
    a = ((index//3)+1) % 3
    q = list(study[a])
    unseen = [set(VALUES[d]) - {s[d] for s in study} for d in range(3)]
    q[changed] = choose_new(sorted(unseen[changed]), rng, counts, "near_2seen", DIMS[changed])
    sources = [a,a,a]; sources[changed] = None
    specs.append(("near_2seen", tuple(q), sources, DIMS[changed], a))
    q = tuple(choose_new(sorted(unseen[d]), rng, counts, "unrelated", DIMS[d]) for d in range(3))
    specs.append(("unrelated", q, [None,None,None], None, None))
    for kind,q,*_ in specs:
        counts.update((kind, DIMS[d], q[d]) for d in range(3))
    return specs


def make_split(out, split, n, seed, used_sets, pixel_hashes):
    rng = random.Random(seed)
    columns = study_columns(n, rng)
    counts = Counter()
    episodes, trials, inputs, labels = [], [], [], []
    image_count = 0
    for i in range(n):
        cols = [c[i*3:(i+1)*3] for c in columns]
        for attempt in range(200):
            study = list(zip(*cols))
            signature = digest(sorted(study))
            if signature not in used_sets:
                used_sets.add(signature)
                break
            for col in cols:
                rng.shuffle(col)
        else:
            raise RuntimeError("Cannot create a unique study set without changing balancing constraints")
        episode_id = f"{split}-{i+1:04d}"
        records = [image_record(out, attributes(s), rng.getrandbits(63), pixel_hashes) for s in study]
        queries = []
        for kind,q,sources,changed,anchor in query_specs(study, i, rng, counts):
            record = image_record(out, attributes(q), rng.getrandbits(63), pixel_hashes)
            trial_id = digest({"episode": episode_id,"query":record["image_id"],"version":VERSION})[:24]
            familiarity = sum(q[d] in {s[d] for s in study} for d in range(3))
            hamming = min(sum(x!=y for x,y in zip(q,s)) for s in study)
            details = {"trial_id":trial_id, "test_type":kind, "answer":TYPE_INFO[kind][2], "query":record,
                       "feature_sources":dict(zip(DIMS,sources)), "changed_dimension":changed,
                       "anchor_study_index":anchor, "seen_feature_count":familiarity, "nearest_hamming_distance":hamming}
            details.update(annotations(familiarity, hamming))
            queries.append(details)
            trials.append({"dataset_version":VERSION,"split":split,"episode_id":episode_id,"study":records,**details})
            inputs.append({"trial_id":trial_id,"study_images":[s["image"] for s in records],
                           "test_image":record["image"],"instruction":INSTRUCTION})
            labels.append({"trial_id":trial_id,"episode_id":episode_id,"test_type":kind,"answer":details["answer"]})
        episodes.append({"episode_id":episode_id,"split":split,"study_signature":signature,"study":records,"tests":queries})
        image_count += 8
        if (i+1)%50 == 0:
            print(f"{split}: {i+1}/{n} episodes", flush=True)
    write_jsonl(out / "episodes" / f"{split}.jsonl", episodes)
    write_jsonl(out / "manifests" / f"{split}.jsonl", trials)
    write_jsonl(out / "inputs" / f"{split}.jsonl", inputs)
    write_jsonl(out / "labels" / f"{split}.jsonl", labels)
    write_index(out, split, trials)
    distribution = {kind:{dim:{value:counts[(kind,dim,value)] for value in VALUES[d]} for d,dim in enumerate(DIMS)} for kind in TYPES}
    return {"episodes":n,"trials":len(trials),"images":image_count,
            "structural_difficulty_counts":dict(Counter(t["structural_difficulty"] for t in trials)),
            "type_counts":dict(Counter(t["test_type"] for t in trials)),"query_attribute_counts":distribution}


def build(out, seed=20261007, dev=50, test=300, calibration_variants=2, overwrite=False):
    out = Path(out).resolve()
    if out.exists():
        marker = out / "dataset.json"
        if not overwrite or not marker.is_file() or json.loads(marker.read_text()).get("generator") != "binding_dataset/generate.py":
            raise SystemExit(f"Output already exists; use a fresh path, or --overwrite for this generator's dataset: {out}")
        shutil.rmtree(out)
    (out/"images").mkdir(parents=True)
    metadata = {"name":"Object Binding / 单物体属性绑定", "version":VERSION,"status":"preview_unvalidated_by_vlm",
                "generator":"binding_dataset/generate.py","seed":seed,"size":[SIZE,SIZE],"study_size":3,
                "python":platform.python_version(),"pillow":PIL.__version__,"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "rendering":{"background_rgb":BACKGROUND,"nominal_area_pixels":47500,"area_jitter_fraction":.05,"position_jitter_pixels":7,"supersampling":2},
                "attributes":dict(zip(DIMS, VALUES)),"colors_rgb":COLORS,"labels_zh":ZH,
                "test_types":{kind:{"label":info[0],"description":info[1],"answer":info[2]} for kind,info in TYPE_INFO.items()},
                "difficulty_scheme":SCHEME,
                "instruction":INSTRUCTION,"notes":["学习图在上下文中，不测原图撤除后的内部记忆。","原始图片没有标签；预览页仅供人工检查。","test 是候选测试划分，需在开发集完成感知校准后冻结版本。"]}
    (out/"dataset.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+"\n")
    signatures = set()
    pixel_hashes = set()
    summary = {"version":VERSION,"seed":seed,"difficulty_scheme_version":SCHEME["version"],"splits":{}}
    for offset,(split,n) in enumerate((("dev",dev),("test",test))):
        summary["splits"][split] = make_split(out, split, n, seed+offset*1000003, signatures, pixel_hashes)
    rng = random.Random(seed+2000006)
    calibration = []
    for values in itertools.product(*VALUES):
        for variant in range(calibration_variants):
            calibration.append({"split":"calibration","variant":variant,**image_record(out,attributes(values),rng.getrandbits(63),pixel_hashes)})
    write_jsonl(out/"manifests"/"calibration.jsonl",calibration)
    summary["calibration_images"] = len(calibration)
    summary["total_images"] = sum(x["images"] for x in summary["splits"].values()) + len(calibration)
    summary["total_trials"] = sum(x["trials"] for x in summary["splits"].values())
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"output":str(out),"images":summary["total_images"],"trials":summary["total_trials"]},ensure_ascii=False),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path(__file__).parent/"data"/"v0.1")
    parser.add_argument("--seed",type=int,default=20261007)
    parser.add_argument("--dev",type=int,default=50)
    parser.add_argument("--test",type=int,default=300)
    parser.add_argument("--calibration-variants",type=int,default=2)
    parser.add_argument("--overwrite",action="store_true")
    args=parser.parse_args()
    if min(args.dev,args.test,args.calibration_variants)<0:
        parser.error("Counts must be non-negative")
    build(args.output,args.seed,args.dev,args.test,args.calibration_variants,args.overwrite)
