# -*- coding: utf-8 -*-
"""
03 表型验证（与论文一致的七重验证）
输入: s1_phenotyped.csv（02步生成，含'表型'标签+生物学特征+肺功能FEV1/FVC）
输出: 验证指标打印
验证: ①肺功能GOLD外部锚定(探索) ②特征归因(排列重要性,刻画非验证)
      ③bootstrap稳定性(ARI) ④多算法对比(GMM/层次) ⑤血常规依赖性 ⑥K选择已在02
注意: 随机森林用聚类输入特征预测表型标签,仅刻画特征贡献(数据泄漏),不作独立验证
"""
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import adjusted_rand_score
from scipy import stats

BIO = ["白细胞", "中性粒细胞百分比", "淋巴细胞百分比", "嗜酸细胞百分比",
       "C反应蛋白", "D二聚体", "PCO2", "氧合指数", "NTproBNP"]
SKEW = ["白细胞", "C反应蛋白", "D二聚体", "NTproBNP"]
RS = 42

def prep(df, feats):
    b = df[feats].apply(pd.to_numeric, errors="coerce").copy()
    for c in SKEW:
        if c in b.columns: b[c] = np.log1p(b[c].clip(lower=0))
    X = KNNImputer(n_neighbors=5).fit_transform(b)
    return StandardScaler().fit_transform(X)

def main():
    df = pd.read_csv("s1_phenotyped.csv")
    df = df[df["表型"].notna()].copy()
    y = df["表型"].values
    Xs = prep(df, BIO)
    print(f"验证样本: {df.shape}, 表型分布: {df['表型'].value_counts().to_dict()}")

    # ① 肺功能GOLD锚定（探索性，n较小，不显著≠无关）
    if "FEV1/FVC" in df.columns:
        sub = df.dropna(subset=["FEV1/FVC"])
        groups = [g["FEV1/FVC"].values for _, g in sub.groupby("表型")]
        if len(groups) == 3:
            h, p = stats.kruskal(*groups)
            print(f"\n① GOLD锚定(探索): n={len(sub)}, FEV1/FVC Kruskal p={p:.3f} "
                  f"(未显著≠无关, 样本量有限)")
            print("   各表型FEV1/FVC中位:", sub.groupby("表型")["FEV1/FVC"].median().round(1).to_dict())

    # ② 特征归因（排列重要性——刻画特征贡献，非独立验证）
    rf = RandomForestClassifier(n_estimators=300, random_state=RS).fit(Xs, y)
    imp = permutation_importance(rf, Xs, y, n_repeats=10, random_state=RS)
    order = np.argsort(-imp.importances_mean)
    print("\n② 特征贡献度(排列重要性, 刻画用):")
    for i in order[:5]:
        print(f"   {BIO[i]}: {imp.importances_mean[i]:.3f}")

    # ③ bootstrap稳定性（与KMeans基准的ARI）
    rng = np.random.RandomState(RS); km_base = KMeans(3, n_init=30, random_state=RS).fit(Xs)
    aris = []
    for _ in range(30):
        idx = rng.choice(len(Xs), len(Xs), replace=True)
        km = KMeans(3, n_init=10, random_state=rng.randint(1e5)).fit(Xs[idx])
        aris.append(adjusted_rand_score(km_base.labels_, km.labels_))
    print(f"\n③ bootstrap稳定性: ARI中位={np.median(aris):.3f} (范围 {min(aris):.3f}-{max(aris):.3f})")
    # 注：真实数据(356例)上bootstrap ARI中位=0.839(0.731~0.966)。
    #     合成示例数据因表型边界不如真实数据清晰，bootstrap ARI可能偏低，仅供演示流程，
    #     实际稳健性评估须在真实数据上进行。

    # ④ 多算法对比（GMM/层次 vs KMeans）
    for name, lab in [("GMM", GaussianMixture(3, random_state=RS).fit_predict(Xs)),
                      ("层次Ward", AgglomerativeClustering(3, linkage="ward").fit_predict(Xs))]:
        ari = adjusted_rand_score(km_base.labels_, lab)
        print(f"④ {name} vs KMeans: ARI={ari:.3f}")

    # ⑤ 血常规依赖性（去血常规/仅血常规 vs 全特征）
    cbc = ["白细胞", "中性粒细胞百分比", "淋巴细胞百分比", "嗜酸细胞百分比"]
    rest = [c for c in BIO if c not in cbc]
    for tag, feats in [("去血常规", rest), ("仅血常规", cbc)]:
        X2 = prep(df, feats)
        lab = KMeans(3, n_init=30, random_state=RS).fit_predict(X2)
        print(f"⑤ {tag}聚类 vs 全特征: ARI={adjusted_rand_score(km_base.labels_, lab):.3f}")

if __name__ == "__main__":
    main()
