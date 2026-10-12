const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const sharp = require('sharp');
const { svg } = require('./core/generate');

const root = __dirname;
const core = path.join(root, 'core');
const readJsonl = p => fs.readFileSync(path.join(core, p), 'utf8').trim().split(/\r?\n/).map(JSON.parse);
const writeJsonl = (p, rows) => fs.writeFileSync(path.join(core, p), rows.map(x => JSON.stringify(x)).join('\n') + '\n');
const opaque = value => crypto.createHash('sha256').update(`binding-core-v1|${value}`).digest('hex').slice(0, 24);
const objKey = o => `${o.color}|${o.shape}|${o.texture}`;
const objectPath = o => `images/${opaque(objKey(o))}.png`;
const hash32 = s => { let h = 2166136261; for (const c of s) h = Math.imul(h ^ c.charCodeAt(0), 16777619); return h >>> 0; };
function variant(id) {
  const a = hash32(id), b = hash32(`${id}|y`), c = hash32(`${id}|scale`);
  const result = { dx: (a % 11) - 5, dy: (b % 11) - 5, scale: 0.974 + (c % 53) / 1000 };
  if (result.dx === 0 && result.dy === 0 && Math.abs(result.scale - 1) < 1e-8) result.scale = 1.017;
  return result;
}
function audit(o, study) {
  const attrs = ['color', 'shape', 'texture'];
  const sources = Object.fromEntries(attrs.map(a => [a, study.findIndex(s => s[a] === o[a])]));
  for (const a of attrs) if (sources[a] === -1) sources[a] = null;
  const seen = attrs.filter(a => sources[a] !== null).length;
  const nearest = Math.min(...study.map(s => attrs.filter(a => s[a] !== o[a]).length));
  const pairs = [['color', 'shape'], ['color', 'texture'], ['shape', 'texture']];
  const pairwiseSeen = pairs.filter(([a, b]) => study.some(s => s[a] === o[a] && s[b] === o[b])).length;
  return { feature_sources: sources, seen_feature_count: seen, novel_feature_count: 3 - seen,
    nearest_hamming_distance: nearest, pairwise_seen_count: pairwiseSeen };
}

async function main() {
  const groups = readJsonl('groups.jsonl');
  const trials = readJsonl('trials.jsonl');
  const groupMap = new Map(groups.map(g => [g.group_id, g]));
  const sourceObjectImages = new Map();
  for (const g of groups) for (const o of g.study_objects) sourceObjectImages.set(objKey(o), o);
  for (const t of trials) sourceObjectImages.set(objKey(t.probe_object), t.probe_object);
  // Copy the fixed catalog under nonsemantic filenames. The generator can recreate its source files.
  for (const o of sourceObjectImages.values()) {
    const from = path.join(core, `images/${o.color}__${o.shape}__${o.texture}.png`);
    const to = path.join(core, objectPath(o));
    fs.copyFileSync(from, to);
  }
  // Include every possible triple for a complete visual calibration catalog.
  const metadata = JSON.parse(fs.readFileSync(path.join(core, 'metadata.json'), 'utf8'));
  const allObjects = Object.keys(metadata.colors).flatMap(color => Object.keys(metadata.shapes)
    .flatMap(shape => Object.keys(metadata.textures).map(texture => ({ color, shape, texture }))));
  for (const o of allObjects) {
    const from = path.join(core, `images/${o.color}__${o.shape}__${o.texture}.png`);
    const to = path.join(core, objectPath(o));
    if (!fs.existsSync(to)) fs.copyFileSync(from, to);
  }
  for (const name of fs.readdirSync(path.join(core, 'images'))) {
    if (!name.includes('__') || !name.endsWith('.png')) continue;
    const sourcePath = path.join(core, 'images', name);
    for (let attempt = 0; ; attempt++) {
      try { await fs.promises.unlink(sourcePath); break; }
      catch (e) {
        if (!['EBUSY', 'EPERM'].includes(e.code) || attempt === 20) throw e;
        await new Promise(resolve => setTimeout(resolve, 250));
      }
    }
  }

  for (const g of groups) g.study_images = g.study_objects.map(objectPath);
  const probeDir = path.join(core, 'probes');
  fs.mkdirSync(probeDir, { recursive: true });
  const rendered = new Set();
  for (const t of trials) {
    const g = groupMap.get(t.group_id);
    t.study_images = g.study_images;
    t.split = t.seed <= 6 ? 'dev' : 'test';
    t.structural_difficulty = { target: 'control', critical_recombination: 'hard', near_lure: 'medium', unrelated_lure: 'easy' }[t.probe_type];
    Object.assign(t, audit(t.probe_object, g.study_objects));
    const pairId = `s${String(t.seed).padStart(2, '0')}_n${String(t.study_count).padStart(2, '0')}_t${t.probe_order}`;
    t.probe_image = `probes/${opaque(pairId)}.png`;
    if (!rendered.has(t.probe_image)) {
      await sharp(Buffer.from(svg(t.probe_object, variant(pairId)))).png().toFile(path.join(core, t.probe_image));
      rendered.add(t.probe_image);
    }
  }
  const instruction = '请依次观察学习图片。干扰图片不计入学习集合。测试图片的颜色、形状和纹理组合是否在某一张学习图片中完整出现过？位置、大小和纹理细节位置的轻微变化不改变身份。仅输出 YES 或 NO。';
  for (const split of ['dev', 'test']) {
    const rows = trials.filter(t => t.split === split);
    fs.mkdirSync(path.join(core, 'inputs'), { recursive: true });
    fs.mkdirSync(path.join(core, 'labels'), { recursive: true });
    writeJsonl(`inputs/${split}.jsonl`, rows.map(t => ({ trial_id: t.trial_id, instruction,
      study_images: t.study_images, distractor_images: t.distractor_images, test_image: t.probe_image })));
    writeJsonl(`labels/${split}.jsonl`, rows.map(t => ({ trial_id: t.trial_id, group_id: t.group_id,
      answer: t.answer, probe_type: t.probe_type, structural_difficulty: t.structural_difficulty,
      seed: t.seed, study_count: t.study_count, distractor_count: t.distractor_count })));
  }
  writeJsonl('groups.jsonl', groups);
  writeJsonl('trials.jsonl', trials);

  const calibrationDir = path.join(core, 'calibration', 'images');
  fs.mkdirSync(calibrationDir, { recursive: true });
  const calInputs = [], calLabels = [];
  for (const o of allObjects) for (let v = 0; v < 2; v++) {
    const id = opaque(`calibration|${objKey(o)}|${v}`);
    const image = `calibration/images/${id}.png`;
    await sharp(Buffer.from(svg(o, variant(`${id}|cal`)))).png().toFile(path.join(core, image));
    calInputs.push({ calibration_id: id, image, instruction: '请识别这张图片的颜色、形状和纹理。' });
    calLabels.push({ calibration_id: id, image, attributes: o, variant: v });
  }
  writeJsonl('calibration/inputs.jsonl', calInputs);
  writeJsonl('calibration/labels.jsonl', calLabels);
  metadata.version = '1.1-integrated';
  metadata.source_design = '物体绑定记忆数据集设计简述.docx';
  metadata.dev_seeds = [1, 2, 3, 4, 5, 6];
  metadata.test_seeds = Array.from({ length: 18 }, (_, i) => i + 7);
  metadata.dev_trials = 216;
  metadata.test_trials = 648;
  metadata.probe_images = rendered.size;
  metadata.calibration_images = calInputs.length;
  metadata.render_variation = 'probe and calibration images use deterministic ±5 px translation and 0.974–1.026 scale';
  metadata.model_input = 'inputs/{dev,test}.jsonl only; filenames opaque; labels kept separate';
  fs.writeFileSync(path.join(core, 'metadata.json'), JSON.stringify(metadata, null, 2) + '\n');
  console.log(`Core built: ${trials.length} trials, ${rendered.size} independent probe PNGs, ${calInputs.length} calibration PNGs.`);
}
main().catch(e => { console.error(e); process.exitCode = 1; });
