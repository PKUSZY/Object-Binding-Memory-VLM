"""Build an offline HTML explorer and labeled contact sheets; never feed these to VLMs."""
import argparse
import json
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from generate import COLORS, DIMS, SHAPES, TEXTURES, TYPE_INFO, ZH
from difficulty import SCHEME
from validate import read_jsonl


def font(size):
    candidates=["/System/Library/Fonts/STHeiti Medium.ttc", "/System/Library/Fonts/PingFang.ttc",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    for name in candidates:
        if Path(name).exists(): return ImageFont.truetype(name,size)
    return ImageFont.load_default(size=size)


def rounded(draw,box,fill="#ffffff"):
    draw.rounded_rectangle(box,radius=18,fill=fill,outline="#dce3eb",width=2)


def paste_thumb(canvas,data,record,x,y,size):
    with Image.open(data/record["image"]) as im:
        canvas.paste(im.resize((size,size),Image.Resampling.LANCZOS),(x,y))


def make_contact(data,out,ep):
    im=Image.new("RGB",(1600,1090),"#f3f6fa")
    d=ImageDraw.Draw(im)
    dark="#1b2b43"; mute="#64748b"
    d.text((40,26),"物体绑定数据集  /  一组学习图片，五类测试",font=font(32),fill=dark)
    d.text((40,77),f"{ep['episode_id']}    ·    原图无标签；此图仅供人工检查    ·    每个测试项应单独调用模型",font=font(20),fill=mute)
    d.text((40,119),"学习阶段",font=font(23),fill=dark)
    for i,record in enumerate(ep["study"]):
        x=40+i*514; y=160; w=492
        rounded(d,(x,y,x+w,y+314))
        d.text((x+20,y+16),f"学习 {i+1}",font=font(21),fill=dark)
        paste_thumb(im,data,record,x+(w-218)//2,y+50,218)
        text=" · ".join(ZH[record["attributes"][dim]] for dim in DIMS)
        d.text((x+20,y+276),text,font=font(20),fill=dark)
    d.text((40,509),"测试阶段",font=font(23),fill=dark)
    for i,test in enumerate(ep["tests"]):
        x=40+i*308; y=550; w=288
        rounded(d,(x,y,x+w,y+422))
        d.text((x+14,y+17),TYPE_INFO[test["test_type"]][0],font=font(19),fill=dark)
        d.text((x+14,y+49),TYPE_INFO[test["test_type"]][1],font=font(17),fill=mute)
        paste_thumb(im,data,test["query"],x+42,y+80,204)
        for j,dim in enumerate(DIMS):
            source=test["feature_sources"][dim]
            origin=f"学习 {source+1}" if source is not None else "未出现"
            value=ZH[test["query"]["attributes"][dim]]
            d.text((x+16,y+297+j*28),f"{value}  ←  {origin}",font=font(18),fill=dark if source is not None else "#a15b13")
        answer=test["answer"]
        level=SCHEME["levels"][test["structural_difficulty"]]["label"]
        d.text((x+16,y+389),f"{answer}  ·  {level}",font=font(19),fill="#167457" if answer=="YES" else "#a15b13")
    d.text((40,1010),"属性相同即视为同一物体；轻微的位置、大小和纹理相位变化不改变身份。",font=font(21),fill=mute)
    d.text((40,1046),"v0.1 · 结构难度尚未经模型实测校准；两种 Critical 同属 L3，Target 为共同对照。",font=font(18),fill=mute)
    im.save(out/"sample_episode.png")


def make_atlas(data,out,calibration):
    records={(r["attributes"]["color"],r["attributes"]["shape"],r["attributes"]["texture"]):r
             for r in calibration if r["variant"]==0}
    if not records: return
    im=Image.new("RGB",(1460,1515),"#f3f6fa")
    d=ImageDraw.Draw(im)
    d.text((34,25),"属性图谱  /  6 种形状 × 6 种纹理",font=font(31),fill="#1b2b43")
    d.text((34,73),"统一展示蓝色；完整校准集覆盖 6 种颜色、216 个组合，每个组合 2 张图。",font=font(19),fill="#64748b")
    for j,texture in enumerate(TEXTURES):
        d.text((180+j*207,124),ZH[texture],font=font(20),fill="#1b2b43")
    for i,shape in enumerate(SHAPES):
        y=171+i*210
        d.text((30,y+78),ZH[shape],font=font(21),fill="#1b2b43")
        for j,texture in enumerate(TEXTURES):
            x=160+j*207
            rounded(d,(x,y,x+193,y+193))
            paste_thumb(im,data,records[("blue",shape,texture)],x+7,y+7,180)
    im.save(out/"attribute_atlas.png")


def build(data,out):
    data=Path(data).resolve(); out=Path(out).resolve()
    out.mkdir(parents=True,exist_ok=True)
    meta=json.loads((data/"dataset.json").read_text())
    summary=json.loads((data/"summary.json").read_text())
    eps={split:read_jsonl(data/"episodes"/f"{split}.jsonl") for split in ("dev","test")}
    calibration=read_jsonl(data/"manifests"/"calibration.jsonl")
    payload={"meta":meta,"summary":summary,"episodes":eps,"calibration":[r for r in calibration if r["variant"]==0],
             "dataPrefix":Path(os.path.relpath(data,out)).as_posix()+"/"}
    template=(Path(__file__).parent/"preview_template.html").read_text()
    html=template.replace("__DATA_JSON__",json.dumps(payload,ensure_ascii=False).replace("</","<\\/"))
    (out/"index.html").write_text(html)
    first=next((e for split in eps.values() for e in split),None)
    if first: make_contact(data,out,first)
    make_atlas(data,out,calibration)
    print(f"Preview: {out/'index.html'}")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=Path(__file__).parent/"data"/"v0.1")
    parser.add_argument("--output",type=Path,default=Path(__file__).parent/"preview")
    args=parser.parse_args(); build(args.data,args.output)
