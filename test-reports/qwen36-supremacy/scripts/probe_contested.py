#!/usr/bin/env python3
"""Focused probe for the two contested throughput metrics, server-usage anchored:
  prefill@8192  (prefill tok/s on an ~8192-token prompt)
  decode@16384  (decode tok/s after a ~16384-token prefill)
Usage: probe_contested.py BASE MODEL LABEL OUT.json"""
import sys, json, time, random, urllib.request
BASE,MODEL,LABEL,OUT=sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]
FILLER="The quick brown fox jumps over the lazy dog. "
def prompt(approx):
    reps=max(1,int(approx*4.45/len(FILLER)))
    return f"[r{random.randint(0,10**9)}] "+FILLER*reps+"\nSummarize in one sentence."
def stream(content,maxtok,timeout=900):
    body=json.dumps({"model":MODEL,"messages":[{"role":"user","content":content}],"max_tokens":maxtok,"temperature":0,"stream":True,"stream_options":{"include_usage":True}}).encode()
    req=urllib.request.Request(BASE.rstrip("/")+"/chat/completions",data=body,headers={"Content-Type":"application/json"})
    t0=time.time();times=[];usage=None
    with urllib.request.urlopen(req,timeout=timeout) as r:
        for raw in r:
            ln=raw.decode("utf-8","ignore").strip()
            if not ln.startswith("data:"):continue
            d=ln[5:].strip()
            if d=="[DONE]":break
            try:o=json.loads(d)
            except:continue
            if o.get("usage"):usage=o["usage"]
            ch=o.get("choices") or []
            if ch:
                dl=ch[0].get("delta") or {}
                if dl.get("content") or dl.get("reasoning_content") or dl.get("reasoning"):times.append(time.time())
    return t0,times,(usage or {})
# prefill@8192
t0,tm,u=stream(prompt(8192),1)
pf=round(u.get("prompt_tokens",0)/(tm[0]-t0),1) if tm else None
# decode@16384
t0,tm,u=stream(prompt(16384),64)
dec=round(u.get("completion_tokens",0)/(tm[-1]-tm[0]),2) if (len(tm)>1 and tm[-1]>tm[0]) else None
out={"label":LABEL,"prefill_8k_tps":pf,"prefill_prompt_tokens":u.get("prompt_tokens"),"decode_16k_tps":dec}
json.dump(out,open(OUT,"w"),indent=2)
print(json.dumps(out))
