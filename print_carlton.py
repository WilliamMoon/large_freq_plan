import json, sys
n = sys.argv[1]
d = json.load(open(f"results_data/dynamic_carlton_N{n}.json"))
r = d["results"]
print(f"N={n} CARLTON (fixed):")
for k, v in sorted(r.items(), key=lambda x: float(x[0].split("_")[1])):
    sp = float(k.split("_")[1])
    m = v.get("cum_interf_prob_mean", v.get("interf_prob"))
    s = v.get("cum_interf_prob_std", 0.0)
    print(f"  v{int(sp):3d}: P_int={m:.4f} +/- {s:.4f}")
