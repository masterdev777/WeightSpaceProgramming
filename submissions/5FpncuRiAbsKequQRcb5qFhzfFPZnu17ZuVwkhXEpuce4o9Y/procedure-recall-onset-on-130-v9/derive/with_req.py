"""with_req.py SCRIPT ARGS...: run a derive script with recombined requests added to the toolkit's item index
(REQ_ITEMS = the request file of #94's mkreq.py, credited; the same patch as #94's branch_req.py / train_req.py)."""
import json, os, runpy, sys
from wsp import data
extra = {it["id"]: it for it in json.load(open(os.environ["REQ_ITEMS"]))} if os.environ.get("REQ_ITEMS") else {}
_by_id = data.by_id
data.by_id = lambda *a, **k: {**_by_id(*a, **k), **extra}
script = sys.argv[1]
sys.argv = sys.argv[1:]
sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
runpy.run_path(script, run_name="__main__")
