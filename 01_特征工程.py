# -*- coding: utf-8 -*-
"""
01 特征工程：六源数据 → 特征矩阵（每行=1次住院）
⚠️ 本脚本依赖真实六源电子病历数据（隐私，不开源）。
   此处提供代码框架与数据结构说明；真实数据需经伦理审批获取。
   合成演示可用 data/合成数据示例.csv 跑通 02/03。

六源数据（每源一个表，按 病案号+住院次/PATIENT_ID 关联）:
  1. 主表(住院记录): 主诊断/年龄/性别/入院日期/出院日期/住院日/费用/病历文本
  2. 检验: 项目名/结果值/采样日期 → 入院早期(入院-1~+2天)取均值
  3. 医嘱: 药名/开始时间 → 抗菌/激素/吸入/茶碱等用药标志
  4. 微生物: 标本/菌名/药敏 → 培养阳性/病原(鲍曼/肺克/铜绿)
  5. 影像: 报告文本 → 肺气肿/肺大泡/结节/胸腔积液/肺心病
  6. 肺功能: FEV1/FVC → GOLD分级

输出特征(示例):
  - 生物学(聚类用): 白细胞/中性粒细胞百分比/淋巴细胞百分比/嗜酸细胞百分比/
                     C反应蛋白/D二聚体/PCO2/氧合指数/NTproBNP
  - 肺功能: FEV1/FVC
  - 结局/关联(非聚类输入): 住院日/费用/培养阳性/用药标志/影像征象
"""
import numpy as np, pandas as pd

def early_lab(lab_df, admit_col="入院日期", day_col="采样日期", item_col="项目名", val_col="结果值"):
    """取入院早期(-1~+2天)检验均值。lab_df需含住院标识+日期。返回每住院一行的透视表。"""
    lab_df = lab_df.copy()
    lab_df[day_col] = pd.to_datetime(lab_df[day_col], errors="coerce")
    lab_df[admit_col] = pd.to_datetime(lab_df[admit_col], errors="coerce")
    lab_df["dday"] = (lab_df[day_col] - lab_df[admit_col]).dt.days
    early = lab_df[(lab_df["dday"] >= -1) & (lab_df["dday"] <= 2)]
    early["val"] = pd.to_numeric(early[val_col], errors="coerce")
    piv = early.groupby(["病案号", "住院次", item_col])["val"].mean().unstack(item_col)
    return piv

def build_feature_matrix():
    """
    框架：读取六源 → 关联 → 特征工程 → 合并为特征矩阵。
    真实实现需按实际数据结构填充（此处为占位说明）。
    """
    raise NotImplementedError(
        "01需真实六源数据。请参考 docstring 中的数据结构与 early_lab 等函数，"
        "按实际数据填充；合成演示请直接用 data/合成数据示例.csv 作为 s1_feature_matrix.csv 跑 02/03。")

if __name__ == "__main__":
    print(__doc__)
    print("\n提示：真实数据填充 build_feature_matrix()；合成演示见 data/合成数据示例.csv")
