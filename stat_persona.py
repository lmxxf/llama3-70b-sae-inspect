"""统计 16 种人格条件的激活数据（100 个 topic 全量）"""
import torch
import statistics

data = torch.load("features_persona_layer50.pt", map_location="cpu")
prompt_types = ["standard", "teacher", "socratic", "child", "interviewer",
                "debugger", "critic", "eli5", "assistant", "villain",
                "drunk", "poet", "conspiracy", "novice", "expert", "guru"]

stats = {p: {"active_counts": [], "max_vals": []} for p in prompt_types}

for case in data:
    for p in prompt_types:
        key = f"{p}_features"
        if key in case:
            feat = case[key]
            stats[p]["active_counts"].append((feat > 0).sum().item())
            stats[p]["max_vals"].append(feat.max().item())

print(f"共 {len(data)} 个 topic\n")
print(f"{'人格':<15} {'平均激活数':<12} {'std':<10} {'平均max':<10} {'min激活':<10} {'max激活':<10}")
print("-" * 67)

results = []
for p in prompt_types:
    counts = stats[p]["active_counts"]
    maxes = stats[p]["max_vals"]
    avg_c = sum(counts) / len(counts)
    std_c = statistics.stdev(counts) if len(counts) > 1 else 0
    avg_m = sum(maxes) / len(maxes)
    min_c = min(counts)
    max_c = max(counts)
    results.append((p, avg_c, std_c, avg_m, min_c, max_c))

# 按平均激活数排序
results.sort(key=lambda x: x[1], reverse=True)

for p, avg_c, std_c, avg_m, min_c, max_c in results:
    print(f"{p:<15} {avg_c:<12.1f} {std_c:<10.1f} {avg_m:<10.2f} {min_c:<10} {max_c:<10}")
