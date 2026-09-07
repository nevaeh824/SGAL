"""
SG-AL 固定 17 个基础学习器的唯一注册表。

参数尽量复现 ``mnsc.2023.03604/detection_model.py``。主程序每次拟合都
通过这里的工厂创建全新实例，避免 OOF 折、stacking 层或伪标签轮次之间
共享模型状态。

NCA、ITML 本身是度量学习器，不会直接输出 0/1/2 分类结果，因此用
Pipeline 把学习后的距离空间连接到 KNN。LMNN 因完整训练耗时不可接受而
删除；其替代模型 RCA 又因 143 维输入下协方差秩不足、变换结果产生 NaN
而删除。当前模型总数和 stacking 列数均为 17。
"""

from functools import partial

from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


REQUIRED_MODEL_NAMES = (
    # 判别分析、广义线性模型和朴素贝叶斯
    "LDA",
    "QDA",
    "Logit",
    "Probit",
    "NaiveBayes",
    # 三种论文命名的决策树
    "Tree-ID3",
    "Tree-C4.5",
    "Tree-CART",
    # 一至三层隐藏层的前馈神经网络
    "FNN1",
    "FNN2",
    "FNN3",
    # 三种核函数的支持向量机
    "SVM-Lin",
    "SVM-Poly",
    "SVM-RBF",
    # 原始 KNN 与三种“度量学习 + KNN”
    "KNN",
    "NCA",
    "ITML",
)


def validate_required_model_names(model_names):
    """要求名称和顺序完全一致，因为每个 stacking 列都按该顺序解释。"""
    names = tuple(model_names)
    if names != REQUIRED_MODEL_NAMES:
        missing = [name for name in REQUIRED_MODEL_NAMES if name not in names]
        extra = [name for name in names if name not in REQUIRED_MODEL_NAMES]
        raise ValueError(
            "SG-AL 必须使用当前固定的 17 个基础学习器，且顺序必须一致；"
            f"missing={missing}, extra={extra}, received={list(names)}"
        )
    return list(names)


def build_required_model(model_name):
    """按名称创建一个尚未拟合的基础学习器。"""
    if model_name == "LDA":
        return LinearDiscriminantAnalysis()
    if model_name == "QDA":
        return QuadraticDiscriminantAnalysis()
    if model_name == "Logit":
        return LogisticRegression(solver="liblinear")
    if model_name == "Probit":
        # 忠实保留参考代码：名称虽为 Probit，实际是 SAG Logistic，
        # 并不是带正态分布链接函数的严格 Probit。
        return LogisticRegression(solver="sag")
    if model_name == "NaiveBayes":
        return GaussianNB()
    if model_name == "Tree-ID3":
        return DecisionTreeClassifier(criterion="entropy")
    if model_name == "Tree-C4.5":
        # 参考实现与 ID3 一样使用 entropy。sklearn 没有原生实现
        # C4.5 的增益率划分，因此这里不是严格意义的 C4.5。
        return DecisionTreeClassifier(criterion="entropy")
    if model_name == "Tree-CART":
        return DecisionTreeClassifier(criterion="gini")
    if model_name == "FNN1":
        return MLPClassifier(
            hidden_layer_sizes=(32,), solver="sgd", alpha=0.01
        )
    if model_name == "FNN2":
        return MLPClassifier(
            hidden_layer_sizes=(32, 32), solver="sgd", alpha=0.01
        )
    if model_name == "FNN3":
        return MLPClassifier(
            hidden_layer_sizes=(32, 32, 32), solver="sgd", alpha=0.01
        )
    if model_name == "SVM-Lin":
        return SVC(kernel="linear")
    if model_name == "SVM-Poly":
        return SVC(kernel="poly")
    if model_name == "SVM-RBF":
        return SVC(kernel="rbf")
    if model_name == "KNN":
        return KNeighborsClassifier()
    if model_name == "NCA":
        # 先学习监督距离变换，再在变换后的空间用 KNN 输出类别。
        return Pipeline([
            ("metric", NeighborhoodComponentsAnalysis()),
            ("classifier", KNeighborsClassifier()),
        ])
    if model_name == "ITML":
        # metric-learn 延迟导入：只有真正创建 ITML 时才要求依赖。
        try:
            from metric_learn import ITML_Supervised
        except ImportError as exc:
            raise ImportError(
                "ITML 需要安装 metric-learn，17 模型 SG-AL 不允许跳过该模型。"
            ) from exc
        return Pipeline([
            ("metric", ITML_Supervised()),
            ("classifier", KNeighborsClassifier()),
        ])
    raise KeyError(f"未知基础学习器: {model_name}")


def required_model_factories():
    """返回 17 个零参数工厂；每次调用工厂都得到独立的新模型实例。"""
    return {
        model_name: partial(build_required_model, model_name)
        for model_name in REQUIRED_MODEL_NAMES
    }
