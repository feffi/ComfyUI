exec(open(__file__.replace("pruned.py", "param_count.py")).read().split("with torch.device")[0])
with torch.device("meta"):
    for k in (8, 32):
        m = comfy.ldm.minimax.model.MiniMaxH3Model(device="meta", dtype=torch.bfloat16, operations=ops, adaln_curve_grid=1001, time_embed_dim=k)
        report(f"MiniMaxH3 pruned k={k}", m)
