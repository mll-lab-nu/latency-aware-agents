"""Summarise harbor model-trial results: msum.py JOBDIR... -> score, shots, reviews, tokens, minutes."""
import json, sys
K = ["points_for", "points_against", "won", "our_shots", "good_shots", "n_reviews", "n_act", "our_serves",
     "our_service_faults", "metered_output_tokens", "metered_input_tokens", "elapsed_s", "valid_trial", "revision", "level"]
for d in sys.argv[1:]:
    r = json.load(open(d.rstrip("/") + "/result.json"))
    m = list(r["stats"]["evals"].values())[0]["metrics"][0]
    print(d.rstrip("/").split("/")[-1], {k: m.get(k) for k in K})
