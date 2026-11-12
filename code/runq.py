"""Minimal resumable job queue for the revision campaigns.

Usage: python3 runq.py WORKER_ID [QUEUE_FILE]    (default queue file: queue.txt)

The queue file holds one job per line: TAG<TAB>COMMAND. Workers take the first job
whose claim directory does not exist yet (os.mkdir is atomic), run it with
ICL_THREADS=1, log its output to logs/TAG.log, and write done/TAG on success. The
queue file is re-read before every claim, so jobs can be appended or reordered while
workers run. A worker exits when no unclaimed job is left; deleting claims/TAG makes
a failed job available again."""
import os, subprocess, sys, time

wid = sys.argv[1]
queue = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else 'queue.txt'
os.chdir(os.path.dirname(os.path.abspath(__file__)))
for d in ('claims', 'done', 'logs'):
    os.makedirs(d, exist_ok=True)
while True:
    if os.path.exists('STOP'):
        break
    job = None
    for line in open(queue):
        line = line.rstrip('\n')
        if not line.strip() or line.startswith('#'):
            continue
        tag, cmd = line.split('\t', 1)
        if os.path.exists(f'done/{tag}'):
            continue
        try:
            os.mkdir(f'claims/{tag}')
        except FileExistsError:
            continue
        job = (tag, cmd)
        break
    if job is None:
        break
    tag, cmd = job
    t0 = time.time()
    with open(f'logs/{tag}.log', 'a') as log:
        log.write(f'# worker {wid} start {time.ctime()}\n{cmd}\n'); log.flush()
        rc = subprocess.call(['bash', '-c', cmd], stdout=log, stderr=subprocess.STDOUT,
                             env={**os.environ, 'ICL_THREADS': '1'})
    with open('runq.log', 'a') as f:
        f.write(f'{time.strftime("%H:%M:%S")} worker {wid} {tag} rc={rc} {time.time()-t0:.0f}s\n')
    if rc == 0:
        open(f'done/{tag}', 'w').close()
