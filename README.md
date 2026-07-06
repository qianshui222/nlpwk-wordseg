# 中文分词程序

程序以词典最大匹配为规则基础，并引入小型人工标注语料训练边界分类模型，形成“规则 + 机器学习”的混合分词方案。

## 功能

- 正向最大匹配（FMM）
- 逆向最大匹配（BMM）
- 双向最大匹配（BM）
- 规则 + 机器学习分词
- `jieba` 补充对比
- 自定义词典加载
- 小型训练语料训练边界模型
- 命令行单句分词
- 文本文件批量输入输出

## 核心思路

1. 先用双向最大匹配得到规则分词结果。
2. 把句子转换为“字与字之间是否切分”的边界分类任务。
3. 为每个边界提取字符上下文、词典命中情况、FMM/BMM/BM 切分建议等规则特征。
4. 使用人工标注语料训练逻辑回归模型。
5. 用模型预测最终边界，输出规则 + 机器学习分词结果。

## 文件说明

- `segmenter.py`：主程序
- `boundary_model.py`：边界分类模型与训练逻辑
- `dictionary.txt`：词典文件
- `training_corpus.txt`：人工标注训练语料
- `input.txt`：示例输入文件
- `output.txt`：示例输出文件
- `compare_input.txt`：对比实验输入文件
- `compare_output.txt`：对比实验输出文件
- `boundary_model.pkl`：训练后的模型文件

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行方法

### 1. 训练规则 + 机器学习模型

```bash
python segmenter.py --train-ml
```

### 2. 交互模式

```bash
python segmenter.py
```

### 3. 直接分词一条句子

```bash
python segmenter.py --text "今天我去吃午饭"
```

### 4. 生成对比结果

```bash
python segmenter.py --compare-text "今天我去吃午饭"
```

### 5. 文件输入输出

```bash
python segmenter.py --input input.txt --output output.txt
```

## 示例

对于句子：

```text
今天我去吃午饭
```

程序可输出：

```text
双向最大匹配分词: 今 / 天 / 我 / 去 / 吃 / 午 / 饭
规则+机器学习分词: 今天 / 我 / 去 / 吃 / 午饭
```


