# -*- coding: utf-8 -*-
"""
04 表型关联分析（表型 × 结局/病原学/治疗模式）
输入: s1_phenotyped.csv（含'表型'+结局/用药/病原学等独立变量）
输出: 连续变量(Kruskal-Wallis)+分类变量(卡方)关联结果
注意: 关联变量均为【未参与聚类的独立变量】，避免循环论证
"""
import numpy as np, pandas as pd
from scipy import stats

CONT = ["住院日", "费用"]          # 连续关联变量（独立）
CAT = ["培养阳性", "抗菌使用", "广谱抗菌", "联合抗菌", "无创通气", "有创插管",
       "激素使用", "胸腔积液", "肺心病", "肺大泡"]  # 分类关联变量（独立）

def main():
    df = pd.read_csv("s1_phenotyped.csv")
    df = df[df["表型"].notna()].copy()
    order = ["P1低炎症型", "P2中间型", "P3高炎症高碳酸型"]
    print(f"样本: {df.shape}")

    print("\n=== 连续变量（Kruskal-Wallis）===")
    for c in CONT:
        if c not in df.columns: continue
        groups = [g[c].dropna().values for _, g in df.groupby("表型")]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) == 3:
            h, p = stats.kruskal(*groups)
            med = df.groupby("表型")[c].median().reindex(order).round(1).to_dict()
            print(f"  {c}: {med}  H={h:.1f}, p={p:.2e}")

    print("\n=== 分类变量（卡方，比例%）===")
    for c in CAT:
        if c not in df.columns: continue
        ct = pd.crosstab(df["表型"], df[c])
        if ct.shape[1] < 2: continue
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        rate = (df.groupby("表型")[c].mean() * 100).reindex(order).round(1).to_dict()
        print(f"  {c}: {rate}  χ²={chi2:.1f}, p={p:.2e}")

if __name__ == "__main__":
    main()
