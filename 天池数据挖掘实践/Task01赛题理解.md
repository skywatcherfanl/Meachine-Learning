Datawhale 零基础入门数据挖掘-Task1 赛题理解
=========================================

# 1 赛题理解

**赛题：零基础入门数据挖掘 - 二手车交易价格预测**

地址：https://tianchi.aliyun.com/competition/entrance/231784/introduction?spm=5176.12281957.1004.1.38b02448ausjSX

## 1.1 赛题概况
比赛要求参赛选手根据给定的数据集，建立模型，二手汽车的交易价格。

来自 Ebay Kleinanzeigen 报废的二手车，数量超过 370,000，包含 20 列变量信息，为了保证比赛的公平性，将会从中抽取 10 万条作为训练集，5 万条作为测试集 A，5 万条作为测试集B。同时会对名称、车辆类型、变速箱、model、燃油类型、品牌、公里数、价格等信息进行脱敏。

通过这道赛题来引导大家走进 AI 数据竞赛的世界，主要针对于于竞赛新人进行自我练习、自我提高。
 
## 1.2 数据概况

---
一般而言，对于数据在比赛界面都有对应的数据概况介绍（匿名特征除外），说明列的性质特征。了解列的性质会有助于我们对于数据的理解和后续分析。
Tip:匿名特征，就是未告知数据列所属的性质的特征列。

---
**train.csv**
* name - 汽车编码
* regDate - 汽车注册时间
* model - 车型编码
* brand - 品牌
* bodyType - 车身类型
* fuelType - 燃油类型
* gearbox - 变速箱
* power - 汽车功率
* kilometer - 汽车行驶公里
* notRepairedDamage - 汽车有尚未修复的损坏
* regionCode - 看车地区编码
* seller - 销售方
* offerType - 报价类型
* creatDate - 广告发布时间
* price - 汽车价格
* v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_10', 'v_11', 'v_12', 'v_13','v_14'（根据汽车的评论、标签等大量信息得到的embedding向量）【人工构造 匿名特征】
　
 数字全都脱敏处理，都为label encoding形式，即数字形式

## 1.3 预测指标

---
**本赛题的评价标准为MAE(Mean Absolute Error):**

$$
MAE=\frac{\sum_{i=1}^{n}\left|y_{i}-\hat{y}_ {i}\right|}{n}
$$
其中$y_ {i}$代表第$i$个样本的真实值，其中$\hat{y}_ {i}$代表第$i$个样本的预测值。

---
**一般问题评价指标说明:**

什么是评估指标：

>评估指标即是我们对于一个模型效果的数值型量化。（有点类似与对于一个商品评价打分，而这是针对于模型效果和理想效果之间的一个打分）

一般来说分类和回归问题的评价指标有如下一些形式：

### 分类算法常见的评估指标如下：
* 对于二类分类器/分类算法，评价指标主要有accuracy， [Precision，Recall，F-score，Pr曲线]，ROC-AUC曲线。
* 对于多类分类器/分类算法，评价指标主要有accuracy， [宏平均和微平均，F-score]。

### 对于回归预测类常见的评估指标如下:
* 平均绝对误差（Mean Absolute Error，MAE），均方误差（Mean Squared Error，MSE），平均绝对百分误差（Mean Absolute Percentage Error，MAPE），均方根误差（Root Mean Squared Error）， R2（R-Square）

**平均绝对误差**
**平均绝对误差（Mean Absolute Error，MAE）**:平均绝对误差，其能更好地反映预测值与真实值误差的实际情况，其计算公式如下：
$$
MAE=\frac{1}{N} \sum_{i=1}^{N}\left|y_{i}-\hat{y}_ {i}\right|
$$

**均方误差**
**均方误差（Mean Squared Error，MSE）**,均方误差,其计算公式为：
$$
MSE=\frac{1}{N} \sum_{i=1}^{N}\left(y_{i}-\hat{y}_ {i}\right)^{2}
$$

**R2（R-Square）的公式为**：
残差平方和：
$$
SS_{res}=\sum\left(y_{i}-\hat{y}_ {i}\right)^{2}
$$
总平均值:
$$
SS_{tot}=\sum\left(y_{i}-\overline{y}_ {i}\right)^{2}
$$

其中$\overline{y}$表示$y$的平均值
得到$R^2$表达式为：
$$
R^{2}=1-\frac{SS_{res}}{SS_{tot}}=1-\frac{\sum\left(y_{i}-\hat{y}_ {i}\right)^{2}}{\sum\left(y_{i}-\overline{y}\right)^{2}}
$$
$R^2$用于度量因变量的变异中可由自变量解释部分所占的比例，取值范围是 0~1，$R^2$越接近1,表明回归平方和占总平方和的比例越大,回归线与各观测点越接近，用x的变化来解释y值变化的部分就越多,回归的拟合程度就越好。所以$R^2$也称为拟合优度（Goodness of Fit）的统计量。

$y_{i}$表示真实值，$\hat{y}_ {i}$表示预测值，$\overline{y}_{i}$表示样本均值。得分越高拟合效果越好。

## 1.4 分析赛题

1. 此题为传统的数据挖掘问题，通过数据科学以及机器学习深度学习的办法来进行建模得到结果。
2. 此题是一个典型的回归问题。
3. 主要应用xgb、lgb、catboost，以及pandas、numpy、matplotlib、seabon、sklearn、keras等等数据挖掘常用库或者框架来进行数据挖掘任务。
4. 通过EDA来挖掘数据的联系和自我熟悉数据。
5. 通过特征工程、特征筛选来建立模型。
6. 最后通过多模型融合提高预测效果。
