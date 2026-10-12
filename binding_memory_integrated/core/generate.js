const fs = require('fs');
const path = require('path');
const sharp = require('sharp');

const root = __dirname;
const imageDir = path.join(root, 'images');
const distractorDir = path.join(root, 'distractors');
const colors = [
  ['red', '#e84c56'], ['blue', '#3986e8'], ['green', '#37ab79'],
  ['yellow', '#f3be3e'], ['purple', '#a76acb'], ['orange', '#f09342'],
];
const shapes = ['circle', 'square', 'triangle', 'hexagon', 'star', 'diamond'];
const textures = ['dots', 'diagonal_stripes', 'grid', 'checkerboard', 'waves', 'chevrons'];
const zh = {
  color: { red: '红', blue: '蓝', green: '绿', yellow: '黄', purple: '紫', orange: '橙' },
  shape: { circle: '圆', square: '方', triangle: '三角', hexagon: '六边形', star: '星形', diamond: '菱形' },
  texture: { dots: '圆点', diagonal_stripes: '斜条纹', grid: '网格', checkerboard: '棋盘格', waves: '波纹', chevrons: '折线纹' },
};

function rng(seed) {
  let x = seed >>> 0;
  return () => {
    x += 0x6d2b79f5;
    let t = x;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function shuffle(a, random) {
  a = [...a];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}
function pick(a, random) { return a[Math.floor(random() * a.length)]; }
function key(o) { return [o.color, o.shape, o.texture].join('__'); }
function objectPath(o) { return `images/${key(o)}.png`; }
function object(color, shape, texture) { return { color, shape, texture }; }
function pairsAbsent(o, study) {
  return study.every(s => !(s.color === o.color && s.shape === o.shape)
    && !(s.color === o.color && s.texture === o.texture)
    && !(s.shape === o.shape && s.texture === o.texture));
}
function criticalCandidates(study) {
  const cs = [...new Set(study.map(x => x.color))];
  const ss = [...new Set(study.map(x => x.shape))];
  const ts = [...new Set(study.map(x => x.texture))];
  return cs.flatMap(color => ss.flatMap(shape => ts.map(texture => object(color, shape, texture))))
    .filter(o => pairsAbsent(o, study));
}
function makeStudy(seed) {
  const random = rng(seed * 0x9e3779b1);
  const cOrder = shuffle(colors.map(x => x[0]), random);
  const sOrder = shuffle(shapes, random);
  const tOrder = shuffle(textures, random);
  const reserved = object(cOrder[5], sOrder[5], tOrder[5]);
  const study = [];
  const used = new Set();
  for (let i = 0; i < 5; i++) {
    const o = object(cOrder[i], sOrder[(i + seed) % 5], tOrder[(i * 2 + seed) % 5]);
    study.push(o); used.add(key(o));
  }
  const active = cOrder.slice(0, 5).flatMap(color => sOrder.slice(0, 5)
    .flatMap(shape => tOrder.slice(0, 5).map(texture => object(color, shape, texture))));
  for (const o of shuffle(active, random)) {
    if (study.length === 12) break;
    if (!used.has(key(o))) { study.push(o); used.add(key(o)); }
  }
  for (const n of [4, 8, 12]) {
    if (criticalCandidates(study.slice(0, n)).length === 0) throw new Error(`No critical lure: ${seed}/${n}`);
  }
  return { study, reserved };
}
function makeProbes(seed, n, study, reserved) {
  const random = rng(seed * 1009 + n * 9176);
  const seen = new Set(study.map(key));
  const target = shuffle(study, random).slice(0, 3).map(o => ({ type: 'target', label: 'YES', object: o }));
  const critical = pick(criticalCandidates(study), random);
  const seenShapes = [...new Set(study.map(o => o.shape))];
  const seenTextures = [...new Set(study.map(o => o.texture))];
  // Match the document's example: shape and texture each occurred, but their pair did not.
  const nearOptions = seenShapes.flatMap(shape => seenTextures.map(texture => object(reserved.color, shape, texture)))
    .filter(o => study.every(s => !(s.shape === o.shape && s.texture === o.texture)));
  if (!nearOptions.length) throw new Error(`No near lure: ${seed}/${n}`);
  const near = pick(nearOptions, random);
  const unrelated = reserved;
  const probes = shuffle([
    ...target,
    { type: 'critical_recombination', label: 'NO', object: critical },
    { type: 'near_lure', label: 'NO', object: near },
    { type: 'unrelated_lure', label: 'NO', object: unrelated },
  ], random);
  if (probes.some(p => p.label === 'NO' && seen.has(key(p.object)))) throw new Error('Probe collision');
  return probes;
}
function shapeMarkup(shape) {
  switch (shape) {
    case 'circle': return '<circle cx="192" cy="192" r="116"/>';
    case 'square': return '<rect x="76" y="76" width="232" height="232"/>';
    case 'triangle': return '<path d="M192 57 L331 311 L53 311 Z"/>';
    case 'hexagon': return '<path d="M192 55 L312 124 L312 260 L192 329 L72 260 L72 124 Z"/>';
    case 'star': return '<path d="M192 54 L226 143 L322 150 L248 211 L272 306 L192 252 L112 306 L136 211 L62 150 L158 143 Z"/>';
    case 'diamond': return '<path d="M192 53 L331 192 L192 331 L53 192 Z"/>';
    default: throw new Error(shape);
  }
}
function textureMarkup(texture) {
  const ink = '#242a35';
  if (texture === 'dots') {
    let s = '';
    for (let y = 14; y <= 370; y += 20) for (let x = 14; x <= 370; x += 20)
      s += `<circle cx="${x}" cy="${y}" r="3.5" fill="${ink}"/>`;
    return s;
  }
  if (texture === 'diagonal_stripes') {
    let s = '';
    for (let x = -400; x < 600; x += 25)
      s += `<path d="M${x} -10 L${x + 404} 394" stroke="${ink}" stroke-width="4"/>`;
    return s;
  }
  if (texture === 'grid') {
    let s = '';
    for (let v = 0; v <= 384; v += 22)
      s += `<path d="M${v} 0 V384 M0 ${v} H384" stroke="${ink}" stroke-width="3"/>`;
    return s;
  }
  if (texture === 'checkerboard') {
    let s = '';
    for (let y = 0; y < 16; y++) for (let x = 0; x < 16; x++)
      if ((x + y) % 2 === 0) s += `<rect x="${x * 24}" y="${y * 24}" width="24" height="24" fill="${ink}" fill-opacity=".58"/>`;
    return s;
  }
  if (texture === 'waves') {
    let s = '';
    for (let y = 10; y <= 374; y += 24) {
      let d = `M-20 ${y}`;
      for (let x = -20; x < 400; x += 40) d += ` Q${x + 10} ${y - 10} ${x + 20} ${y} Q${x + 30} ${y + 10} ${x + 40} ${y}`;
      s += `<path d="${d}" fill="none" stroke="${ink}" stroke-width="3.5"/>`;
    }
    return s;
  }
  if (texture === 'chevrons') {
    let s = '';
    for (let y = 12; y <= 372; y += 25) {
      let d = `M-16 ${y}`;
      for (let x = -16; x < 400; x += 32) d += ` l16 -10 l16 10`;
      s += `<path d="${d}" fill="none" stroke="${ink}" stroke-width="3.5" stroke-linejoin="round"/>`;
    }
    return s;
  }
  throw new Error(texture);
}
function svg(o, variant = { dx: 0, dy: 0, scale: 1 }) {
  const fill = colors.find(x => x[0] === o.color)[1];
  const shape = shapeMarkup(o.shape);
  const transform = `translate(${variant.dx || 0} ${variant.dy || 0}) translate(192 192) scale(${variant.scale || 1}) translate(-192 -192)`;
  return `<svg xmlns="http://www.w3.org/2000/svg" width="384" height="384" viewBox="0 0 384 384">
<rect width="384" height="384" fill="#f6f7f9"/>
<defs><clipPath id="shape">${shape}</clipPath></defs>
<g transform="${transform}">
<g fill="${fill}">${shape}</g>
<g clip-path="url(#shape)">${textureMarkup(o.texture)}</g>
<g fill="none" stroke="#242a35" stroke-width="5" stroke-linejoin="round">${shape}</g>
</g>
</svg>`;
}
function distractorSvg(index) {
  const shapes = [
    '<path d="M192 75 C135 15 52 92 82 175 C101 230 166 274 192 310 C218 274 283 230 302 175 C332 92 249 15 192 75 Z"/>',
    '<path d="M160 58 H224 V155 H326 V219 H224 V326 H160 V219 H58 V155 H160 Z"/>',
    '<path d="M192 52 L308 138 L263 318 H121 L76 138 Z"/>',
    '<path d="M208 52 L102 208 H174 L144 333 L288 162 H211 L252 52 Z"/>',
  ];
  const sh = shapes[index];
  const lines = Array.from({ length: 16 }, (_, i) => `<path d="M0 ${i * 26} L384 ${i * 26}" stroke="#858d98" stroke-width="4"/>`).join('');
  return `<svg xmlns="http://www.w3.org/2000/svg" width="384" height="384"><rect width="384" height="384" fill="#f6f7f9"/><defs><clipPath id="s">${sh}</clipPath></defs><g fill="#b8bdc6">${sh}</g><g clip-path="url(#s)">${lines}</g><g fill="none" stroke="#4b5563" stroke-width="5" stroke-linejoin="round">${sh}</g></svg>`;
}
async function main() {
  fs.mkdirSync(imageDir, { recursive: true });
  fs.mkdirSync(distractorDir, { recursive: true });
  const all = colors.flatMap(([color]) => shapes.flatMap(shape => textures.map(texture => object(color, shape, texture))));
  for (const o of all) await sharp(Buffer.from(svg(o))).png().toFile(path.join(root, objectPath(o)));
  for (let i = 0; i < 4; i++) await sharp(Buffer.from(distractorSvg(i))).png().toFile(path.join(distractorDir, `distractor_${i + 1}.png`));

  const trials = [];
  const groups = [];
  for (let seed = 1; seed <= 24; seed++) {
    const design = makeStudy(seed);
    for (const n of [4, 8, 12]) {
      const study = design.study.slice(0, n);
      const probes = makeProbes(seed, n, study, design.reserved);
      for (const d of [0, 4]) {
        const groupId = `s${String(seed).padStart(2, '0')}_n${String(n).padStart(2, '0')}_d${d}`;
        const distractors = Array.from({ length: d }, (_, i) => `distractors/distractor_${i + 1}.png`);
        groups.push({ group_id: groupId, seed, study_count: n, distractor_count: d,
          study_objects: study, study_images: study.map(objectPath), distractor_images: distractors,
          reserved_unseen_attributes: design.reserved });
        for (let i = 0; i < probes.length; i++) {
          const p = probes[i];
          trials.push({ trial_id: `${groupId}_t${i + 1}`, group_id: groupId, seed,
            study_count: n, distractor_count: d, probe_order: i + 1,
            probe_type: p.type, answer: p.label,
            study_images: study.map(objectPath), distractor_images: distractors,
            probe_image: objectPath(p.object), probe_object: p.object,
            probe_object_zh: `${zh.color[p.object.color]}色${zh.shape[p.object.shape]}形＋${zh.texture[p.object.texture]}`,
            session_rule: 'fresh_context_for_every_trial' });
        }
      }
    }
  }
  fs.writeFileSync(path.join(root, 'trials.jsonl'), trials.map(x => JSON.stringify(x)).join('\n') + '\n', 'utf8');
  fs.writeFileSync(path.join(root, 'groups.jsonl'), groups.map(x => JSON.stringify(x)).join('\n') + '\n', 'utf8');
  fs.writeFileSync(path.join(root, 'metadata.json'), JSON.stringify({
    title: '物体绑定记忆数据集', version: '1.0', image_size: [384, 384], image_format: 'PNG',
    colors: Object.fromEntries(colors.map(([k]) => [k, zh.color[k]])),
    shapes: Object.fromEntries(shapes.map(k => [k, zh.shape[k]])),
    textures: Object.fromEntries(textures.map(k => [k, zh.texture[k]])),
    seeds: 24, study_counts: [4, 8, 12], distractor_counts: [0, 4], trials_per_group: 6,
    total_groups: groups.length, total_trials: trials.length,
    probe_types_per_group: { target: 3, critical_recombination: 1, near_lure: 1, unrelated_lure: 1 },
    answer_balance: { YES: 432, NO: 432 },
  }, null, 2) + '\n', 'utf8');
  console.log(`Generated ${all.length} object PNGs, 4 distractor PNGs, ${groups.length} groups, ${trials.length} trials.`);
}
if (require.main === module) main().catch(e => { console.error(e); process.exitCode = 1; });
module.exports = { svg };
