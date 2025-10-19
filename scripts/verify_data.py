```python
import os, sys, glob


req_dirs = [
'data/raw/nflverse__nflfastR-play-by-play-data',
'data/external'
]
for d in req_dirs:
if not os.path.isdir(d):
print(f"ERROR: Missing directory {d}")
sys.exit(2)


pbp = glob.glob('data/raw/nflverse__nflfastR-play-by-play-data/play_by_play_*.csv.gz')
if len(pbp) < 5:
print("ERROR: Too few PBP files; fetch step likely failed")
sys.exit(2)


sched = glob.glob('data/external/schedules_*.csv')
if len(sched) < 5:
print("ERROR: schedules_*.csv not found; split step likely failed")
sys.exit(2)


print("OK: raw inputs present")
