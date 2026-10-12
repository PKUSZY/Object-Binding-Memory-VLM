"""Validate membership labels, feature provenance, splits, PNGs and safe inputs."""
import argparse
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path

from PIL import Image
from generate import DIMS, SIZE, TYPES, VALUES, VERSION, digest, triple
from difficulty import SCHEME, group_index


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(root, write_report=True):
    root=Path(root).resolve()
    meta=json.loads((root/"dataset.json").read_text())
    require(meta["version"]==VERSION,"Unsupported version")
    require(meta.get("difficulty_scheme")==SCHEME,"Difficulty scheme mismatch")
    seen_sets=set(); seen_trials=set(); seen_paths={}; all_pixel_hashes=set()
    report={"passed":False,"checks":[],"splits":{}}

    def image_check(record):
        path=record["image"]
        if path in seen_paths:
            require(seen_paths[path]==record,"Same image has inconsistent metadata")
            return
        require(Path(path).parts[0]=="images" and ".." not in Path(path).parts,"Unsafe image path")
        file=root/path
        require(file.is_file(),f"Missing {path}")
        require(hashlib.sha256(file.read_bytes()).hexdigest()==record["sha256"],f"Hash mismatch {path}")
        attrs=record["attributes"]
        require(set(attrs)==set(DIMS),"Incorrect attribute keys")
        require(all(attrs[d] in VALUES[i] for i,d in enumerate(DIMS)),"Unknown attribute value")
        require(record["image_id"]==digest({"version":VERSION,"attributes":attrs,"render":record["render"]})[:28],"Image id mismatch")
        with Image.open(file) as im:
            require(im.format=="PNG" and im.mode=="RGB" and im.size==(SIZE,SIZE),f"Incorrect format {path}")
            pixel_hash=hashlib.sha256(im.tobytes()).hexdigest()
            require(pixel_hash==record["pixel_sha256"],f"Pixel hash mismatch {path}")
            require(pixel_hash not in all_pixel_hashes,f"Duplicate pixels {path}")
            all_pixel_hashes.add(pixel_hash)
        seen_paths[path]=record

    for split in ("dev","test"):
        episodes=read_jsonl(root/"episodes"/f"{split}.jsonl")
        manifests=read_jsonl(root/"manifests"/f"{split}.jsonl")
        inputs=read_jsonl(root/"inputs"/f"{split}.jsonl")
        labels=read_jsonl(root/"labels"/f"{split}.jsonl")
        require(len(manifests)==len(episodes)*5==len(inputs)==len(labels),"Counts mismatch")
        by_id={t["trial_id"]:t for t in manifests}
        require(len(by_id)==len(manifests),"Duplicate manifest trial id")
        safe={t["trial_id"]:t for t in inputs}; answers={t["trial_id"]:t for t in labels}
        require(set(by_id)==set(safe)==set(answers),"Input/label ID mismatch")
        type_counts=Counter(); study_counts={dim:Counter() for dim in DIMS}
        target_positions=Counter(); changed_dimensions={kind:Counter() for kind in ("critical_2plus1","near_2seen")}
        for ep in episodes:
            require(ep["split"]==split,"Episode split mismatch")
            require(len(ep["study"])==3 and len(ep["tests"])==5,"Episode shape mismatch")
            study=[triple(s["attributes"]) for s in ep["study"]]
            signature=digest(sorted(study))
            require(signature==ep["study_signature"] and signature not in seen_sets,"Study set duplicate across episodes/splits")
            seen_sets.add(signature)
            require(all(len({s[d] for s in study})==3 for d in range(3)),"Study dimension repeats")
            require(Counter(t["test_type"] for t in ep["tests"])==Counter(TYPES),"Missing test type")
            for record in ep["study"]:
                image_check(record)
                for dim in DIMS: study_counts[dim][record["attributes"][dim]]+=1
            for test in ep["tests"]:
                uid=test["trial_id"]; kind=test["test_type"]
                require(uid not in seen_trials,"Trial id reused")
                seen_trials.add(uid); type_counts[kind]+=1
                q=triple(test["query"]["attributes"])
                require(test["answer"]==("YES" if q in study else "NO"),"Membership label wrong")
                sources=[]
                for d,dim in enumerate(DIMS):
                    matches=[i for i,s in enumerate(study) if s[d]==q[d]]
                    source=matches[0] if matches else None
                    sources.append(source)
                    require(test["feature_sources"][dim]==source,"Feature provenance wrong")
                familiarity=sum(s is not None for s in sources)
                distance=min(sum(x!=y for x,y in zip(q,s)) for s in study)
                expected={"target":(3,0),"critical_2plus1":(3,1),"critical_1plus1plus1":(3,2),"near_2seen":(2,1),"unrelated":(0,3)}[kind]
                require((familiarity,distance)==expected,"Wrong lure construction")
                require(test["seen_feature_count"]==familiarity and test["nearest_hamming_distance"]==distance,"Derived metadata wrong")
                expected_level={"target":"control","unrelated":"easy","near_2seen":"medium",
                                "critical_2plus1":"hard","critical_1plus1plus1":"hard"}[kind]
                require(test.get("structural_difficulty")==expected_level and test.get("novel_feature_count")==3-familiarity,"Structural difficulty wrong")
                if kind=="critical_2plus1": require(sorted(Counter(sources).values())==[1,2],"Not a 2+1 lure")
                if kind=="critical_1plus1plus1": require(len(set(sources))==3,"Not a 1+1+1 lure")
                if kind=="target": target_positions[sources[0]]+=1
                if kind in changed_dimensions: changed_dimensions[kind][test["changed_dimension"]]+=1
                if kind=="near_2seen": require(len({s for s in sources if s is not None})==1,"Near old features must share source")
                image_check(test["query"])
                require(by_id[uid]=={"dataset_version":VERSION,"split":split,"episode_id":ep["episode_id"],"study":ep["study"],**test},"Manifest disagrees with episode")
                require(set(safe[uid])=={"trial_id","study_images","test_image","instruction"},"Safe input contains extra fields/labels")
                require(safe[uid]["study_images"]==[s["image"] for s in ep["study"]] and safe[uid]["test_image"]==test["query"]["image"],"Input images mismatch")
                require(safe[uid]["instruction"]==meta["instruction"],"Input instruction mismatch")
                require(answers[uid]=={"trial_id":uid,"episode_id":ep["episode_id"],"test_type":kind,"answer":test["answer"]},"Answer key mismatch")
        for counts in [target_positions,*changed_dimensions.values()]:
            if counts: require(max(counts.values())-min(counts.values())<=1,"Position/dimension balance wrong")
        for counts in study_counts.values():
            if counts: require(max(counts.values())-min(counts.values())<=1,"Study feature balance wrong")
        report["splits"][split]={"episodes":len(episodes),"trials":len(manifests),"type_counts":dict(type_counts),
                                 "structural_difficulty_counts":dict(Counter(t["structural_difficulty"] for t in manifests)),
                                 "study_attribute_counts":study_counts,"target_source_positions":target_positions,
                                 "changed_dimensions":changed_dimensions}
        tier_index=json.loads((root/"difficulty"/f"{split}.json").read_text())
        require(tier_index==group_index(split,manifests),"Difficulty group index wrong")
    calibration=read_jsonl(root/"manifests"/"calibration.jsonl")
    combos=Counter()
    for record in calibration:
        # Calibration's split and variant are metadata outside the image record.
        image_check({k:v for k,v in record.items() if k not in ("split","variant")})
        combos[triple(record["attributes"])]+=1
    if calibration:
        require(set(combos)==set(itertools.product(*VALUES)) and len(set(combos.values()))==1,"Calibration coverage wrong")
    require({str(p.relative_to(root)) for p in (root/"images").glob("*.png")}==set(seen_paths),"Extra or missing PNGs")
    summary=json.loads((root/"summary.json").read_text())
    require(summary["total_images"]==len(seen_paths) and summary["total_trials"]==len(seen_trials),"Summary totals wrong")
    require(summary.get("difficulty_scheme_version")==SCHEME["version"],"Summary difficulty version wrong")
    for split in ("dev","test"):
        require(summary["splits"][split].get("structural_difficulty_counts")==report["splits"][split]["structural_difficulty_counts"],"Summary difficulty counts wrong")
    report.update(passed=True,images_checked=len(seen_paths),trials_checked=len(seen_trials),calibration_images=len(calibration),
                  checks=["object membership labels","lure definitions and feature provenance","no duplicate study set across splits",
                          "balanced types and source positions","balanced study attributes","PNG dimensions, mode and sha256",
                          "no identical image pixels","label-free model input allowlist","calibration combination coverage",
                          "structural difficulty annotations and shared-target group indexes"])
    if write_report:
        (root/"validation_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"passed":True,"images":len(seen_paths),"trials":len(seen_trials)},ensure_ascii=False))
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=Path(__file__).parent/"data"/"v0.1")
    validate(parser.parse_args().data)
