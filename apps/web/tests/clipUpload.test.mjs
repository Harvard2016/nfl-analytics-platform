import assert from 'node:assert/strict';
import test from 'node:test';
import { audioExtent, binBounds, timelineSeek } from '../lib/clipTimeline.ts';
import { initialClipEdits, reviewedCandidates } from '../lib/clipReview.ts';

test('a shorter audio track cannot stretch video timestamps', () => {
  const timeline = { loudness_db: Array(11).fill(-20), audio_start_s: 0, audio_end_s: 20.6 };
  assert.deepEqual(audioExtent(timeline, 2), [0, 20.6]);
  assert.deepEqual(binBounds(timeline, 10, 2), [20, 20.6]);
  assert.equal(timelineSeek(10 / 30, 30), 10);       // click the 10 s marker in a 30 s video
  assert.equal(timelineSeek(25 / 30, 30), 25);       // seeking beyond the audio still uses video time
});

test('explicit bin times preserve delayed audio and a final partial bin', () => {
  const timeline = { loudness_db: [-20, -10], bin_start_s: [5, 7], bin_end_s: [7, 7.6] };
  assert.deepEqual(audioExtent(timeline, 2), [5, 7.6]);
  assert.deepEqual(binBounds(timeline, 1, 2), [7, 7.6]);
  assert.equal(timelineSeek(-1, 30), 0);
  assert.equal(timelineSeek(2, 30), 30);
});

test('exporting alone never claims that a person reviewed a candidate', () => {
  const candidates = [
    { asset: 'a.mp4', candidate_moment_s: 10, clip_start_s: 8, clip_end_s: 12 },
    { asset: 'b.mp4', candidate_moment_s: 18, clip_start_s: 16, clip_end_s: 20 },
  ];
  const edits = initialClipEdits(candidates);
  assert.deepEqual(reviewedCandidates(candidates, edits).map(c => c.reviewed_by_a_person), [false, false]);
  edits['a.mp4'].reviewed = true;
  assert.deepEqual(reviewedCandidates(candidates, edits).map(c => c.reviewed_by_a_person), [true, false]);
});
