"""Dedupe check-set v2 QC-kept chips that are the same acquisition reprocessed by ESA.

Chip id format: BD_<date>_<S2 platform>_<product level>_<sensing time>_<orbit>_<tile>_<processing time>_<chip#>,
e.g. BD_20231016_S2B_MSIL2A_20231016T043749_R033_T45RYJ_20231016T094425_0. Splitting on "_" gives field 4 =
Sentinel-2 sensing time, field 6 = tile, field 7 = processing timestamp, field 8 = chip number within the scene
pair. Two rows with the same (sensing time, tile, chip#) but a different processing timestamp are the SAME
acquisition, reprocessed to a newer ESA baseline; they are not independent chips.

Among the 323 QC-kept chips (`qc_decisions.csv`, decision == keep) this happens 28 times. Rule: for each
duplicate group keep the row with the LATEST processing timestamp (current processing baseline) and drop the
rest. Dedup is applied only within QC-kept chips (a QC-rejected chip never blocks or is blocked by its twin).

This script does not modify qc_decisions.csv (it still records the pre-dedupe QC decision for every candidate
chip); it writes `dedupe_dropped.csv` (id, kept_twin_id) listing the 28 dropped ids and the id of the twin that
was kept instead. `bd_eval.py` and `bd_finetune.py` additionally exclude any id listed in dedupe_dropped.csv,
when the file is present, so the final check set used for scoring is 323 - 28 = 295 chips.
"""
import pandas as pd

q = pd.read_csv('qc_decisions.csv')
kept = q[q.decision == 'keep'].copy()
parts = kept.id.str.split('_')
kept['sensing_time'] = parts.str[4]
kept['tile'] = parts.str[6]
kept['proc_time'] = parts.str[7]
kept['chip_num'] = parts.str[8]
kept['key'] = list(zip(kept.sensing_time, kept.tile, kept.chip_num))

rows = []
for key, g in kept.groupby('key'):
    if len(g) == 1:
        continue
    g = g.sort_values('proc_time')
    keep_row = g.iloc[-1]  # latest processing timestamp = current ESA baseline
    for _, r in g.iloc[:-1].iterrows():
        rows.append(dict(id=r.id, kept_twin_id=keep_row.id))

dropped = pd.DataFrame(rows, columns=['id', 'kept_twin_id']).sort_values('id').reset_index(drop=True)
dropped.to_csv('dedupe_dropped.csv', index=False)

sizes = kept.groupby('key').size()
print(f'QC-kept chips: {len(kept)} | unique (sensing time, tile, chip#) keys: {sizes.shape[0]} | '
      f'duplicate groups: {int((sizes > 1).sum())} | dropped (excess) chips: {len(dropped)} | '
      f'final check-set size: {len(kept) - len(dropped)}')
