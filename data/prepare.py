"""
Download FineWeb-Edu sample-10BT parquet files and tokenize them with the GPT-2
tokenizer into uint16 memmap shards of 100M tokens each. Documents are
separated by EOT (50256). The first shard written is the validation shard
(val.bin), the following ones are train_0000.bin, train_0001.bin, ...

Usage: python data/prepare.py --out ~/data/fineweb_edu_gpt2 --files 6 --workers 30
  6 parquet files give about 4.3B tokens (about 8.6 GB of shards).
"""
import argparse, os, time
import numpy as np
import pyarrow.parquet as pq
import tiktoken
from multiprocessing import Pool
from huggingface_hub import hf_hub_download

SHARD = 100_000_000
enc = tiktoken.get_encoding("gpt2")
EOT = enc.eot_token

def tok(doc):
    ids = enc.encode_ordinary(doc)
    ids.append(EOT)
    return np.array(ids, dtype=np.uint16)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--files", type=int, default=6)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--first_file", type=int, default=0, help="resume: first parquet file index to process")
    ap.add_argument("--start_shard", type=int, default=0, help="resume: shard index to start writing at (0 = val)")
    ap.add_argument("--cache", default=os.path.expanduser("~/data/hf_cache"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    buf = np.empty(SHARD, dtype=np.uint16); fill = 0; shard = a.start_shard; total = 0; t0 = time.time()

    def flush(n):
        nonlocal shard
        name = "val" if shard == 0 else f"train_{shard-1:04d}"
        path = os.path.join(a.out, f"{name}.bin")
        np.memmap(path, dtype=np.uint16, mode="w+", shape=(n,))[:] = buf[:n]
        print(f"wrote {path} {n/1e6:.1f}M tokens  total={total/1e9:.3f}B  {time.time()-t0:.0f}s", flush=True)
        shard += 1

    with Pool(a.workers) as pool:
        for fi in range(a.first_file, a.files):
            fname = f"sample/10BT/{fi:03d}_00000.parquet"
            local = hf_hub_download("HuggingFaceFW/fineweb-edu", fname, repo_type="dataset", cache_dir=a.cache)
            pf = pq.ParquetFile(local)
            for rg in range(pf.num_row_groups):
                docs = pf.read_row_group(rg, columns=["text"]).column("text").to_pylist()
                for ids in pool.imap(tok, docs, chunksize=64):
                    n = len(ids); total += n
                    while n > 0:
                        take = min(n, SHARD - fill)
                        buf[fill:fill+take] = ids[len(ids)-n:len(ids)-n+take]
                        fill += take; n -= take
                        if fill == SHARD:
                            flush(SHARD); fill = 0
            print(f"file {fi} done", flush=True)
    if fill > 0:
        flush(fill)
    print("done", total)

if __name__ == "__main__":
    main()
