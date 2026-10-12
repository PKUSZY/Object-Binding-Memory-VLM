const fs = require('fs');
const path = require('path');
const sharp = require('sharp');

const root = __dirname;
const readJsonl = name => fs.readFileSync(path.join(root, name), 'utf8').trim().split(/\r?\n/).map(JSON.parse);
const check = (condition, message) => { if (!condition) throw new Error(message); };
const key = o => `${o.color}/${o.shape}/${o.texture}`;
const count = (items, value) => items.filter(x => x === value).length;

async function main() {
  const trials = readJsonl('trials.jsonl');
  const groups = readJsonl('groups.jsonl');
  const metadata = JSON.parse(fs.readFileSync(path.join(root, 'metadata.json'), 'utf8'));
  check(trials.length === 864, `Trial count: ${trials.length}`);
  check(groups.length === 144, `Group count: ${groups.length}`);
  check(new Set(trials.map(t => t.trial_id)).size === 864, 'Duplicate trial ID');
  check(new Set(groups.map(g => g.group_id)).size === 144, 'Duplicate group ID');
  check(count(trials.map(t => t.answer), 'YES') === 432, 'YES count');
  check(count(trials.map(t => t.answer), 'NO') === 432, 'NO count');
  check(metadata.total_trials === 864 && metadata.total_groups === 144, 'Metadata counts');

  const files = new Set();
  const allowedTypes = ['target', 'critical_recombination', 'near_lure', 'unrelated_lure'];
  const groupById = new Map(groups.map(g => [g.group_id, g]));
  for (const g of groups) {
    check(g.seed >= 1 && g.seed <= 24, `${g.group_id}: seed`);
    check([4, 8, 12].includes(g.study_count), `${g.group_id}: study count`);
    check([0, 4].includes(g.distractor_count), `${g.group_id}: distractor count`);
    check(g.study_objects.length === g.study_count && g.study_images.length === g.study_count, `${g.group_id}: study length`);
    check(new Set(g.study_objects.map(key)).size === g.study_count, `${g.group_id}: duplicate study object`);
    check(g.distractor_images.length === g.distractor_count, `${g.group_id}: distractor length`);
    for (const p of [...g.study_images, ...g.distractor_images]) files.add(p);
    const ts = trials.filter(t => t.group_id === g.group_id);
    check(ts.length === 6, `${g.group_id}: trial count`);
    check(count(ts.map(t => t.answer), 'YES') === 3 && count(ts.map(t => t.answer), 'NO') === 3, `${g.group_id}: answer balance`);
    for (const type of allowedTypes) check(count(ts.map(t => t.probe_type), type) === (type === 'target' ? 3 : 1), `${g.group_id}: ${type} count`);
    check(new Set(ts.map(t => t.probe_order)).size === 6, `${g.group_id}: duplicate order`);
    const seen = {
      color: new Set(g.study_objects.map(o => o.color)),
      shape: new Set(g.study_objects.map(o => o.shape)),
      texture: new Set(g.study_objects.map(o => o.texture)),
    };
    const studyKeys = new Set(g.study_objects.map(key));
    for (const t of ts) {
      check(t.seed === g.seed && t.study_count === g.study_count && t.distractor_count === g.distractor_count, `${t.trial_id}: condition`);
      check(JSON.stringify(t.study_images) === JSON.stringify(g.study_images), `${t.trial_id}: study sequence`);
      check(JSON.stringify(t.distractor_images) === JSON.stringify(g.distractor_images), `${t.trial_id}: distractor sequence`);
      check(t.session_rule === 'fresh_context_for_every_trial', `${t.trial_id}: session rule`);
      check(/^probes\/[a-f0-9]{24}\.png$/.test(t.probe_image), `${t.trial_id}: probe path`);
      files.add(t.probe_image);
      const o = t.probe_object;
      if (t.probe_type === 'target') {
        check(t.answer === 'YES' && studyKeys.has(key(o)), `${t.trial_id}: target invalid`);
      } else if (t.probe_type === 'critical_recombination') {
        check(t.answer === 'NO' && !studyKeys.has(key(o)), `${t.trial_id}: critical label`);
        check(seen.color.has(o.color) && seen.shape.has(o.shape) && seen.texture.has(o.texture), `${t.trial_id}: critical components`);
        for (const s of g.study_objects) {
          check(!(s.color === o.color && s.shape === o.shape), `${t.trial_id}: color-shape pair seen`);
          check(!(s.color === o.color && s.texture === o.texture), `${t.trial_id}: color-texture pair seen`);
          check(!(s.shape === o.shape && s.texture === o.texture), `${t.trial_id}: shape-texture pair seen`);
        }
      } else if (t.probe_type === 'near_lure') {
        check(t.answer === 'NO' && !seen.color.has(o.color) && seen.shape.has(o.shape) && seen.texture.has(o.texture), `${t.trial_id}: near components`);
        check(g.study_objects.every(s => !(s.shape === o.shape && s.texture === o.texture)), `${t.trial_id}: near pair seen`);
      } else if (t.probe_type === 'unrelated_lure') {
        check(t.answer === 'NO' && !seen.color.has(o.color) && !seen.shape.has(o.shape) && !seen.texture.has(o.texture), `${t.trial_id}: unrelated components`);
      } else throw new Error(`${t.trial_id}: unknown type`);
    }
  }
  for (let seed = 1; seed <= 24; seed++) for (const n of [4, 8, 12]) {
    const a = groupById.get(`s${String(seed).padStart(2, '0')}_n${String(n).padStart(2, '0')}_d0`);
    const b = groupById.get(`s${String(seed).padStart(2, '0')}_n${String(n).padStart(2, '0')}_d4`);
    check(a && b, `${seed}/${n}: missing condition`);
    check(JSON.stringify(a.study_objects) === JSON.stringify(b.study_objects), `${seed}/${n}: unmatched study`);
    const ta = trials.filter(t => t.group_id === a.group_id).map(t => `${t.probe_order}:${key(t.probe_object)}:${t.answer}`);
    const tb = trials.filter(t => t.group_id === b.group_id).map(t => `${t.probe_order}:${key(t.probe_object)}:${t.answer}`);
    check(JSON.stringify(ta) === JSON.stringify(tb), `${seed}/${n}: unmatched probes`);
  }
  for (const p of files) {
    const absolute = path.join(root, p);
    check(fs.existsSync(absolute), `Missing image: ${p}`);
    const meta = await sharp(absolute).metadata();
    check(meta.format === 'png' && meta.width === 384 && meta.height === 384, `Invalid PNG: ${p}`);
  }
  check(fs.readdirSync(path.join(root, 'images')).filter(x => x.endsWith('.png')).length === 216, 'Object image count');
  check(fs.readdirSync(path.join(root, 'distractors')).filter(x => x.endsWith('.png')).length === 4, 'Distractor image count');
  console.log(`PASS: ${groups.length} groups, ${trials.length} trials, ${files.size} referenced valid PNGs; labels and lure constraints verified.`);
}
main().catch(e => { console.error(e); process.exitCode = 1; });
