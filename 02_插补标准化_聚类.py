# -*- coding: utf-8 -*-
"""
02 插补标准化与聚类（COPD急性加重多模态表型发现）
输入: s1_feature_matrix.csv（01步生成，每行=1次住院，含诊断/生物学特征/肺功能/结局等）
输出: s1_phenotyped.csv（含表型标签）、聚类K选择指标
方法学要点（与论文一致）:
  1. 聚类仅用9项生物学指标，不含住院日/费用（结局）
  2. 偏态指标log转换 → KNN插补 → Z-score标准化 → KMeans(K=3)
  3. 剔除白细胞极端值(>50或<1) + 核心特征完整度≥70%筛选
"""
import numpy as np, pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# ---------- 配置 ----------
IN = "s1_feature_matrix.csv"
OUT = "s1_phenotyped.csv"
# 9项聚类生物学特征（血常规分类/炎症/凝血/通气/心负荷）
BIO = ["白细胞", "中性粒细胞百分比", "淋巴细胞百分比", "嗜酸细胞百分比",
       "C反应蛋白", "D二聚体", "PCO2", "氧合指数", "NTproBNP"]
# 偏态指标（log1p转换）
SKEW = ["白细胞", "C反应蛋白", "D二聚体", "NTproBNP"]
RANDOM_STATE = 42

def main():
    df = pd.read_csv(IN)
    print(f"特征矩阵: {df.shape}")

    # 1. 限定COPD主诊断（剔除肺大疱等非COPD，需数据含'主诊断'列）
    if "主诊断" in df.columns:
        df = df[~df["主诊断"].astype(str).str.contains("肺大疱", na=False)].copy()
        print(f"剔除肺大疱后: {df.shape}")

    # 2. 取9项生物学特征
    bio = df[BIO].apply(pd.to_numeric, errors="coerce")

    # 3. 剔除白细胞极端值（>50 或 <1，白血病样异常）
    bad = (bio["白细胞"] > 50) | (bio["白细胞"] < 1)
    df = df[~bad].copy(); bio = bio[~bad]
    print(f"剔除白细胞极端值后: {df.shape}")

    # 4. 核心特征完整度≥70%筛选
    completeness = bio.notna().mean(axis=1)
    keep = completeness >= 0.7
    df = df[keep].copy(); bio = bio[keep]
    print(f"核心特征完整度≥70%后: {df.shape}  (中位完整度 {completeness[keep].median():.2f})")

    # 5. 偏态指标 log1p 转换
    bio_t = bio.copy()
    for c in SKEW:
        if c in bio_t.columns:
            bio_t[c] = np.log1p(bio_t[c].clip(lower=0))

    # 6. KNN插补 + Z-score标准化
    X = KNNImputer(n_neighbors=5).fit_transform(bio_t)
    Xs = StandardScaler().fit_transform(X)

    # 7. K选择：轮廓系数 + Calinski-Harabasz（K=2~6）
    print("\nK选择指标:")
    for k in range(2, 7):
        km = KMeans(n_clusters=k, n_init=30, random_state=RANDOM_STATE).fit(Xs)
        sizes = pd.Series(km.labels_).value_counts().sort_index().tolist()
        print(f"  K={k}: 轮廓={silhouette_score(Xs, km.labels_):.3f}, "
              f"CH={calinski_harabasz_score(Xs, km.labels_):.0f}, 簇规模={sizes}")

    # 8. K=3 聚类
    km3 = KMeans(n_clusters=3, n_init=30, random_state=RANDOM_STATE).fit(Xs)
    df["pheno_raw"] = km3.labels_

    # 9. 按中性粒细胞百分比排序命名表型（P1低/P2中/P3高炎症）
    order = df.groupby("pheno_raw")["中性粒细胞百分比"].median().sort_values().index.tolist()
    name_map = {order[0]: "P1低炎症型", order[1]: "P2中间型", order[2]: "P3高炎症高碳酸型"}
    df["表型"] = df["pheno_raw"].map(name_map)
    print("\n表型分布:")
    print(df["表型"].value_counts())

    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"\n输出: {OUT}")

if __name__ == "__main__":
    main()
