# Chinese-split Integrated

这是在原 `Chinese-split` Python 分词项目基础上融合 `WordSeg` C++ 版本后的增强版。

## 已迁移内容

- 正向最大匹配 FMM
- 逆向最大匹配 BMM
- 双向最大匹配 BM
- 正向最小匹配 MINM
- 逆向最小匹配 RMINM
- 邻近匹配 NM
- 最短路径匹配 SPM
- GUI 分词页面，提供打开文本、运行算法、显示耗时、保存结果
- 保留原来的 `jieba` 辅助混合分词
- 保留原来的规则 + 机器学习边界分类模型

## 运行命令行版本

```bash
python segmenter.py --text "他说的确实在理，从小学到中学他都是好学生。"
```

## 运行 GUI 版本

```bash
python gui.py
```

## 词典格式

仍然兼容原来的纯词表格式：

```text
中文
分词
自然语言处理
```

也支持给最短路径算法使用的词频格式：

```text
中文 100
分词 80
自然语言处理 50
```

如果没有写词频，程序会默认该词频为 `1`。
