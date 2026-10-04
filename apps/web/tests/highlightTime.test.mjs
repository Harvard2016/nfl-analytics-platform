import assert from 'node:assert/strict';
import { test } from 'node:test';
import fs from 'node:fs';
import { youtubeId, toSourceTime, toTimelineTime, sourceVideoLink } from '../lib/highlightTime.ts';

test('source mapping is per-game, preserves fractions, and round-trips', () => {
  for (const offset of [0, 27.123448, 61.75]) {
    const source = { full_video: 'https://www.youtube.com/watch?v=0FF_DbJ3G68', trim_start_s: offset };
    for (const time of [0, 408, 3566, 8000.25]) {
      assert.ok(Math.abs(toTimelineTime(source, toSourceTime(source,time), 9000)-time)<1e-9);
    }
  }
});
test('mapping clamps before/after the dataset, and rejects invalid inputs', () => {
  const source = { full_video: 'https://www.youtube.com/watch?v=0FF_DbJ3G68', trim_start_s: 27.123448 };
  assert.equal(toTimelineTime(source,10,100),0);
  assert.equal(toTimelineTime(source,500,100),100);
  assert.throws(()=>toSourceTime({...source,trim_start_s:NaN},10));
  assert.throws(()=>toSourceTime({...source,trim_start_s:-1},10));
  assert.throws(()=>toTimelineTime(source,Infinity,100));
  const link = new URL(sourceVideoLink(source,408));
  assert.equal(link.searchParams.get('v'),'0FF_DbJ3G68');
  assert.equal(link.searchParams.get('t'),'435s');
});
test('only recognized YouTube hosts and video IDs are accepted', () => {
  assert.equal(youtubeId('https://youtu.be/0FF_DbJ3G68?t=10'),'0FF_DbJ3G68');
  assert.equal(youtubeId('https://www.youtube.com/watch?v=0FF_DbJ3G68'),'0FF_DbJ3G68');
  assert.equal(youtubeId('https://youtube.com.evil.example/watch?v=0FF_DbJ3G68'),null);
  assert.equal(youtubeId('javascript:alert(1)'),null);
  assert.equal(youtubeId('https://youtube.com/watch?v=short'),null);
});
test('all exported candidate intervals map to their saved source times', () => {
  const index = JSON.parse(fs.readFileSync(new URL('../public/demo/highlights/index.json',import.meta.url),'utf8'));
  let intervals = 0;
  for (const item of index.games) {
    const game = JSON.parse(fs.readFileSync(new URL(`../public/demo/highlights/games/${item.id}.json`,import.meta.url),'utf8'));
    assert.ok(youtubeId(game.source.full_video), `game ${item.id}`);
    for (const c of game.candidates) {
      assert.ok(c.end_s>c.start_s && c.start_s>=0 && c.end_s<=game.clips*2);
      assert.ok(Math.abs(toSourceTime(game.source,c.start_s)-c.source_time_s)<=.051);
      assert.equal(c.start_clip*2,c.start_s);
      assert.equal(c.end_clip*2,c.end_s);
      intervals++;
    }
  }
  assert.ok(intervals>100);
});
