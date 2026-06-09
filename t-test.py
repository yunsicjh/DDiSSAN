from scipy.stats import ttest_ind_from_stats

# 通用参数：所有模型均为5次随机种子 n=5
n = 5

# ================== Ours 统计量 ==================
# Restaurant (Res14)
our_res_acc_mean, our_res_acc_std = 87.49, 1.06
our_res_f1_mean, our_res_f1_std = 82.07, 1.64
# Laptop (LAP)
our_lap_acc_mean, our_lap_acc_std = 83.26, 0.61
our_lap_f1_mean, our_lap_f1_std = 80.21, 0.89

# ================== 1. Ours vs GPT2(base) ==================
# Restaurant
gpt2b_res_acc_mean, gpt2b_res_acc_std = 56.47, 0.82
gpt2b_res_f1_mean, gpt2b_res_f1_std = 77.59, 0.32
t_res_acc, p_res_acc = ttest_ind_from_stats(our_res_acc_mean, our_res_acc_std, n, gpt2b_res_acc_mean, gpt2b_res_acc_std, n)
t_res_f1, p_res_f1 = ttest_ind_from_stats(our_res_f1_mean, our_res_f1_std, n, gpt2b_res_f1_mean, gpt2b_res_f1_std, n)
# Laptop
gpt2b_lap_acc_mean, gpt2b_lap_acc_std = 50.65, 1.04
gpt2b_lap_f1_mean, gpt2b_lap_f1_std = 72.61, 1.03
t_lap_acc, p_lap_acc = ttest_ind_from_stats(our_lap_acc_mean, our_lap_acc_std, n, gpt2b_lap_acc_mean, gpt2b_lap_acc_std, n)
t_lap_f1, p_lap_f1 = ttest_ind_from_stats(our_lap_f1_mean, our_lap_f1_std, n, gpt2b_lap_f1_mean, gpt2b_lap_f1_std, n)

# ================== 2. Ours vs GPT2(medium) ==================
# Restaurant
gpt2m_res_acc_mean, gpt2m_res_acc_std = 60.07, 0.52
gpt2m_res_f1_mean, gpt2m_res_f1_std = 81.52, 0.80
t_res_acc2, p_res_acc2 = ttest_ind_from_stats(our_res_acc_mean, our_res_acc_std, n, gpt2m_res_acc_mean, gpt2m_res_acc_std, n)
t_res_f12, p_res_f12 = ttest_ind_from_stats(our_res_f1_mean, our_res_f1_std, n, gpt2m_res_f1_mean, gpt2m_res_f1_std, n)
# Laptop
gpt2m_lap_acc_mean, gpt2m_lap_acc_std = 53.55, 0.43
gpt2m_lap_f1_mean, gpt2m_lap_f1_std = 75.94, 0.17
t_lap_acc2, p_lap_acc2 = ttest_ind_from_stats(our_lap_acc_mean, our_lap_acc_std, n, gpt2m_lap_acc_mean, gpt2m_lap_acc_std, n)
t_lap_f12, p_lap_f12 = ttest_ind_from_stats(our_lap_f1_mean, our_lap_f1_std, n, gpt2m_lap_f1_mean, gpt2m_lap_f1_std, n)

# ================== 3. Ours vs MSPF (仅F1) ==================
# Restaurant
mspf_res_f1_mean, mspf_res_f1_std = 73.51, 0.36
t_res_f13, p_res_f13 = ttest_ind_from_stats(our_res_f1_mean, our_res_f1_std, n, mspf_res_f1_mean, mspf_res_f1_std, n)
# Laptop
mspf_lap_f1_mean, mspf_lap_f1_std = 67.03, 0.42
t_lap_f13, p_lap_f13 = ttest_ind_from_stats(our_lap_f1_mean, our_lap_f1_std, n, mspf_lap_f1_mean, mspf_lap_f1_std, n)

# 打印结果
print("=== Ours vs GPT2(base) ===")
print(f"Restaurant Acc: t={t_res_acc:.4f}, p={p_res_acc:.4f}")
print(f"Restaurant F1: t={t_res_f1:.4f}, p={p_res_f1:.4f}")
print(f"Laptop Acc: t={t_lap_acc:.4f}, p={p_lap_acc:.4f}")
print(f"Laptop F1: t={t_lap_f1:.4f}, p={p_lap_f1:.4f}")

print("\n=== Ours vs GPT2(medium) ===")
print(f"Restaurant Acc: t={t_res_acc2:.4f}, p={p_res_acc2:.4f}")
print(f"Restaurant F1: t={t_res_f12:.4f}, p={p_res_f12:.4f}")
print(f"Laptop Acc: t={t_lap_acc2:.4f}, p={p_lap_acc2:.4f}")
print(f"Laptop F1: t={t_lap_f12:.4f}, p={p_lap_f12:.4f}")

print("\n=== Ours vs MSPF ===")
print(f"Restaurant F1: t={t_res_f13:.4f}, p={p_res_f13:.4f}")
print(f"Laptop F1: t={t_lap_f13:.4f}, p={p_lap_f13:.4f}")