const fs = require('fs');
const path = require('path');
const cp = require('child_process');
const sharp = require('sharp');

const root = __dirname;
const core = path.join(root, 'core');
const aux = path.join(root, 'auxiliary', 'three_study_v0.1', 'data', 'v0.1');
const check = (v, m) => { if (!v) throw new Error(m); };
const jsonl = p => fs.readFileSync(p, 'utf8').trim().split(/\r?\n/).map(JSON.parse);
const n = (a, x) => a.filter(v => v === x).length;
const img = (base, p) => path.join(base, p);

async function main() {
  cp.execFileSync(process.execPath, [path.join(core, 'validate.js')], { stdio: 'inherit' });
  const trials = jsonl(path.join(core, 'trials.jsonl'));
  const groups = jsonl(path.join(core, 'groups.jsonl'));
  const byId = new Map(trials.map(t => [t.trial_id, t]));
  const coreReport = { groups: groups.length, trials: trials.length, dev_trials: 0, test_trials: 0,
    answers: { YES: n(trials.map(t => t.answer), 'YES'), NO: n(trials.map(t => t.answer), 'NO') },
    study_pngs: fs.readdirSync(path.join(core, 'images')).filter(x => x.endsWith('.png')).length,
    probe_pngs: fs.readdirSync(path.join(core, 'probes')).filter(x => x.endsWith('.png')).length,
    distractor_pngs: fs.readdirSync(path.join(core, 'distractors')).filter(x => x.endsWith('.png')).length };
  check(coreReport.study_pngs === 216 && coreReport.probe_pngs === 432 && coreReport.distractor_pngs === 4, 'Core image catalog');
  check(new Set(groups.map(g => g.study_objects.map(o => `${o.color}/${o.shape}/${o.texture}`).join('|'))).size === 72,
    'Expected 72 distinct study lists shared across distractor conditions');
  for (const split of ['dev', 'test']) {
    const inputs = jsonl(path.join(core, 'inputs', `${split}.jsonl`));
    const labels = jsonl(path.join(core, 'labels', `${split}.jsonl`));
    const want = split === 'dev' ? 216 : 648;
    check(inputs.length === want && labels.length === want, `${split}: split size`);
    check(new Set(inputs.map(x => x.trial_id)).size === want, `${split}: duplicate input ID`);
    check(new Set(labels.map(x => x.trial_id)).size === want, `${split}: duplicate label ID`);
    const labelMap = new Map(labels.map(x => [x.trial_id, x]));
    for (const x of inputs) {
      check(JSON.stringify(Object.keys(x).sort()) === JSON.stringify(['distractor_images','instruction','study_images','test_image','trial_id']), `${split}: input allowlist`);
      check(/^[\da-f]{24}\.png$/.test(path.basename(x.test_image)), `${split}: opaque query name`);
      check(x.study_images.every(p => /^[\da-f]{24}\.png$/.test(path.basename(p))), `${split}: opaque study names`);
      const t = byId.get(x.trial_id), l = labelMap.get(x.trial_id);
      check(t && l && t.split === split && l.answer === t.answer && l.probe_type === t.probe_type, `${split}: join mismatch`);
      check(JSON.stringify(x.study_images) === JSON.stringify(t.study_images) && x.test_image === t.probe_image, `${split}: input path mismatch`);
      if (t.answer === 'YES') for (const p of x.study_images) {
        check(!fs.readFileSync(img(core, p)).equals(fs.readFileSync(img(core, x.test_image))), `${split}: target pixel copy ${x.trial_id} ${p}`);
      }
    }
    coreReport[`${split}_trials`] = want;
  }
  const calInputs = jsonl(path.join(core, 'calibration', 'inputs.jsonl'));
  const calLabels = jsonl(path.join(core, 'calibration', 'labels.jsonl'));
  check(calInputs.length === 432 && calLabels.length === 432, 'Calibration count');
  const calMap = new Map(calLabels.map(x => [x.calibration_id, x]));
  const combos = new Map();
  for (const x of calInputs) {
    check(JSON.stringify(Object.keys(x).sort()) === JSON.stringify(['calibration_id','image','instruction']), 'Calibration input allowlist');
    const l = calMap.get(x.calibration_id);
    check(l && l.image === x.image, 'Calibration join');
    const k = `${l.attributes.color}/${l.attributes.shape}/${l.attributes.texture}`;
    combos.set(k, (combos.get(k) || 0) + 1);
    const m = await sharp(img(core, x.image)).metadata();
    check(m.format === 'png' && m.width === 384 && m.height === 384, 'Calibration PNG');
  }
  check(combos.size === 216 && [...combos.values()].every(v => v === 2), 'Calibration combination coverage');
  coreReport.calibration_pngs = 432;

  const imported = { dev_trials: 0, test_trials: 0, episodes: 0, images: 0,
    answers: { YES: 0, NO: 0 }, critical_2plus1_with_seen_pair: 0,
    critical_1plus1plus1_pairwise_unseen: 0 };
  const types = ['target','critical_2plus1','critical_1plus1plus1','near_2seen','unrelated'];
  const referenced = new Set();
  for (const split of ['dev', 'test']) {
    const episodes = jsonl(path.join(aux, 'episodes', `${split}.jsonl`));
    const manifest = jsonl(path.join(aux, 'manifests', `${split}.jsonl`));
    const inputs = jsonl(path.join(aux, 'inputs', `${split}.jsonl`));
    const labels = jsonl(path.join(aux, 'labels', `${split}.jsonl`));
    const expectedEpisodes = split === 'dev' ? 50 : 300;
    check(episodes.length === expectedEpisodes && manifest.length === expectedEpisodes * 5,
      `Aux ${split}: size`);
    check(inputs.length === manifest.length && labels.length === manifest.length, `Aux ${split}: input/label size`);
    check(episodes.every(e => e.study.length === 3 && e.tests.length === 5), `Aux ${split}: episode layout`);
    const byTrial = new Map(manifest.map(m => [m.trial_id, m]));
    check(byTrial.size === manifest.length, `Aux ${split}: duplicate trial ID`);
    for (const e of episodes) for (const type of types) check(n(e.tests.map(t => t.test_type), type) === 1, `Aux ${split}: ${type}`);
    for (const i of inputs) {
      check(JSON.stringify(Object.keys(i).sort()) === JSON.stringify(['instruction','study_images','test_image','trial_id']), `Aux ${split}: input allowlist`);
      const m = byTrial.get(i.trial_id);
      check(m && m.query.image === i.test_image && m.study.length === i.study_images.length, `Aux ${split}: input join`);
      check(i.study_images.every((p,j) => p === m.study[j].image), `Aux ${split}: study paths`);
    }
    for (const l of labels) {
      const m = byTrial.get(l.trial_id);
      check(m && m.answer === l.answer && m.test_type === l.test_type, `Aux ${split}: label join`);
    }
    for (const m of manifest) {
      check(types.includes(m.test_type) && m.study.length === 3, `Aux ${split}: type/study`);
      const o = m.query.attributes, study = m.study.map(s => s.attributes);
      const seen = ['color','shape','texture'].filter(a => study.some(s => s[a] === o[a])).length;
      const pairCount = [['color','shape'],['color','texture'],['shape','texture']]
        .filter(([a,b]) => study.some(s => s[a] === o[a] && s[b] === o[b])).length;
      if (m.test_type === 'target') check(m.answer === 'YES' && study.some(s => ['color','shape','texture'].every(a => s[a] === o[a])), 'Aux target');
      if (m.test_type === 'critical_2plus1') {
        check(m.answer === 'NO' && seen === 3 && pairCount >= 1, 'Aux 2+1');
        imported.critical_2plus1_with_seen_pair++;
      }
      if (m.test_type === 'critical_1plus1plus1') {
        check(m.answer === 'NO' && seen === 3 && pairCount === 0, 'Aux 1+1+1');
        imported.critical_1plus1plus1_pairwise_unseen++;
      }
      if (m.test_type === 'near_2seen') check(m.answer === 'NO' && seen === 2, 'Aux near');
      if (m.test_type === 'unrelated') check(m.answer === 'NO' && seen === 0, 'Aux unrelated');
      for (const s of m.study) referenced.add(s.image);
      referenced.add(m.query.image);
      imported.answers[m.answer]++;
    }
    imported[`${split}_trials`] = manifest.length;
    imported.episodes += episodes.length;
  }
  const cal = jsonl(path.join(aux, 'manifests', 'calibration.jsonl'));
  check(cal.length === 432, 'Aux calibration count');
  for (const x of cal) referenced.add(x.image);
  const imageFiles = fs.readdirSync(path.join(aux, 'images')).filter(x => x.endsWith('.png'));
  check(imageFiles.length === 3232 && referenced.size === 3232, 'Aux image count/references');
  for (const p of referenced) check(fs.existsSync(img(aux, p)), `Aux missing PNG ${p}`);
  imported.images = imageFiles.length;
  imported.calibration_images = cal.length;
  const report = { passed: true, core: coreReport, supplementary_three_study: imported,
    interpretation: 'Only core is conformant with the 864-trial design. Supplementary trials require separate reporting.' };
  fs.writeFileSync(path.join(root, 'validation_report.json'), JSON.stringify(report, null, 2) + '\n');
  console.log(`PASS integrated: core ${coreReport.trials} trials; auxiliary ${imported.dev_trials + imported.test_trials} trials and ${imported.images} images.`);
}
main().catch(e => { console.error(e); process.exitCode = 1; });
