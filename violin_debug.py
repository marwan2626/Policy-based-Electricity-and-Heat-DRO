import os, pandas as pd, numpy as np
parq_dir = os.path.join('v4_oos','v4_loading')
files = [f for f in os.listdir(parq_dir) if f.endswith('.parquet')]
print('[debug] parquet files:', files)
for f in files:
    p = os.path.join(parq_dir,f)
    try:
        df = pd.read_parquet(p)
    except Exception as e:
        print(f'  {f}: read failed {e}')
        continue
    if not {'sample_id','t','trafo_index','loading_pct'} <= set(df.columns):
        print(f'  {f}: missing required cols, has {df.columns.tolist()}')
        continue
    lp = pd.to_numeric(df['loading_pct'], errors='coerce').to_numpy()
    lp = lp[np.isfinite(lp)]
    print(f'  {f}: count={lp.size}, min={lp.min() if lp.size else None:.2f}, max={lp.max() if lp.size else None:.2f}, mean={lp.mean() if lp.size else None:.2f}')
