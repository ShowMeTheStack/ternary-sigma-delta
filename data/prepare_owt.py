"""
Stream OpenWebText (Skylion007/openwebtext) and tokenize it with the GPT-2 tokenizer into
uint16 memmap shards of 100M tokens in ~/data/openwebtext_gpt2 (val.bin first, then
train_0000.bin, ...). Stops once the token target is reached (checked after every block of
2048 documents).

Usage: python data/prepare_owt.py [target_tokens]      (default 1200000000)
"""
import os, sys, time, numpy as np, tiktoken
from multiprocessing import Pool
from datasets import load_dataset
SHARD=100_000_000; enc=tiktoken.get_encoding("gpt2"); EOT=enc.eot_token
def tok(doc):
    ids=enc.encode_ordinary(doc); ids.append(EOT); return np.array(ids, dtype=np.uint16)
if __name__=="__main__":
    out=os.path.expanduser("~/data/openwebtext_gpt2"); os.makedirs(out, exist_ok=True)
    ds=load_dataset("Skylion007/openwebtext", split="train", streaming=True, trust_remote_code=True)
    buf=np.empty(SHARD, dtype=np.uint16); fill=0; shard=0; total=0; t0=time.time(); target=int(sys.argv[1]) if len(sys.argv)>1 else 1_200_000_000
    def flush(n):
        global shard
        name="val" if shard==0 else f"train_{shard-1:04d}"
        np.memmap(os.path.join(out, name+".bin"), dtype=np.uint16, mode="w+", shape=(n,))[:]=buf[:n]
        print(f"wrote {name} total={total/1e9:.3f}B {time.time()-t0:.0f}s", flush=True); shard+=1
    with Pool(8) as pool:
        batch=[]
        for ex in ds:
            batch.append(ex["text"])
            if len(batch)==2048:
                for ids in pool.imap(tok, batch, chunksize=64):
                    n=len(ids); total+=n
                    while n>0:
                        take=min(n, SHARD-fill); buf[fill:fill+take]=ids[len(ids)-n:len(ids)-n+take]; fill+=take; n-=take
                        if fill==SHARD: flush(SHARD); fill=0
                batch=[]
                if total>=target: break
    if fill>0: flush(fill)
    print("done", total)
