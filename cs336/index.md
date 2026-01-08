# 斯坦福CS336：从零开始的语言模型


## 为什么系统爱好者都应该学习大模型？

在当今AI技术浪潮中，掌握大模型知识已成为系统开发者的必备技能。通过参与**斯坦福CS336大模型系统课程**，开始从零构建大模型的实践之旅。这门课程很可能在未来3年内成为系统领域的标杆课程（正如CMU 15-445数据库课程近年来的地位）。

### 作业1：构建 Transformer 语言模型

通过以下三个小节实现了一个小型语言模型。

- Tokenizer设计与实现
- 模型架构编码（含Self-Attention机制）
- 优化器开发

作业地址：
[Assignment1-Basics GitHub仓库](https://github.com/Kosthi/assignment1-basics)

接下来我将分享完成作业的一部分细节和心得。
#### 2.字节对编码（BPE）分词器

##### 2.1 Unicode 标准

**Problem**（unicode1）理解 Unicode（1 分）

(a) `chr(0)`返回什么 Unicode 字符？

NULL字符，即ASCII空字符

```python
>>> chr(0)
'\x00'
```

(b) 该字符的字符串表示（`__repr__()`）与其打印表示有何不同？

repr()函数显示转移序列'\\\x00'，打印什么都不显示，即空字符

```python
>>> repr(chr(0))
"'\\x00'"
>>> print(chr(0))

```

(c) 该字符出现在文本中时会发生什么？可在 Python 解释器中尝试以下代码验证

空字符虽然在打印时不可见，但仍作为 Python 字符串的一部分，这表明 Python 字符串可以包含不可见的字符，且这些空字符仍然会影响字符串的存储和处理。

```python
>>> chr(0)
'\x00'
>>> print(chr(0))

>>> "this is a test" + chr(0) + "string"
'this is a test\x00string'
>>> print("this is a test" + chr(0) + "string")
this is a teststring
```
##### 2.2 Unicode 编码

##### **Problem**（unicode2）Unicode 编码（1 分）

(a) 为什么优先选择在 UTF-8 编码的字节上训练分词器，而非 UTF-16 或 UTF-32？可对比不同编码对各类输入字符串的输出结果。

训练分词器时处理的是字节序列，UTF-8能更紧凑地表示常见字符，减少序列长度，这对模型训练更高效。而且UTF-8向后兼容ASCII，处理英文文本时特别高效。

```python
>>> test_string="你好,世界!"
>>> list(test_string.encode("utf-8"))
[228, 189, 160, 229, 165, 189, 44, 228, 184, 150, 231, 149, 140, 33]
>>> list(test_string.encode("utf-16"))
[255, 254, 96, 79, 125, 89, 44, 0, 22, 78, 76, 117, 33, 0]
>>> list(test_string.encode("utf-32"))
[255, 254, 0, 0, 96, 79, 0, 0, 125, 89, 0, 0, 44, 0, 0, 0, 22, 78, 0, 0, 76, 117, 0, 0, 33, 0, 0, 0]
```

(b) 以下函数意图将 UTF-8 字节串解码为 Unicode 字符串，但存在错误。该函数为何不正确？请提供一个导致错误结果的输入字节串示例。

该函数不正确，因为它逐字节解码，这对于多字节UTF-8 字符会失败。单独的字节\xe4是无效的 UTF-8字符，会引发UnicodeDecodeError 异常。

```python
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])

>>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
'hello'
>>> "你好".encode("utf-8")
b'\xe4\xbd\xa0\xe5\xa5\xbd'
>>> decode_utf8_bytes_to_str_wrong("你好".encode("utf-8"))
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
    import platform
  File "<stdin>", line 2, in decode_utf8_bytes_to_str_wrong
    import sys
    
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe4 in position 0: unexpected end of data
>>> bytes([0xe4,0xbd,0xa0]).decode("utf-8")
'你' ## 汉字“你”占用了3个字节
```

(c) 给出一个无法解码为任何 Unicode 字符的两字节序列。

`b'\x80\x81'`是无效的，因为在 UTF-8 中，任何以二进制 `10` 开头的字节（如 `0x80`）必须是延续字节，但此处它作为首字节出现，前面没有有效的前导字节。

```python
>>> b'\x80\x81'.decode("utf-8")
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
    import platform
UnicodeDecodeError: 'utf-8' codec can't decode byte 0x80 in position 0: invalid start byte
>>> b'\xe4\x80\x81'.decode("utf-8")
'䀁' ## 生僻字 䀁(yòu) UTF-8编码 E4 80 81
```

##### 2.5 BPE分词器训练实验
**1.在预分词之前移除特殊标记**

```python
special_tokens = ["<|endoftext|>", "<sep>", "[SPECIAL]"]
# 注意先转义再用"|"分割，在正则表达式中"|"表示为或
<|endoftext|>|<sep>|[SPECIAL] # "|".join(special_tokens) 结果
<\|endoftext\|>\|<sep>\|\[SPECIAL\] # Wrong: re.escape("|".join(special_tokens)) 结果
<\|endoftext\|>|<sep>|\[SPECIAL\] # True: "|".join([re.escape(token) for token in special_tokens])
```

**2.字节对编码（BPE）的优化**

文档中推荐使用的cppyy在Mac和Linux环境中有问题，为了追求高性能，使用Pybind11来绑定cpp代码，我的代码中预分词由py处理，而BPE归并过程交给cpp，实际最大的瓶颈还是预分词，可以直接用已有的代码``pretokenization_example.py``做个分块并行提升最大（8核 100s，16核30s）。

主要包含以下几个关键优化：

1. **并行化处理**

- 使用OpenMP并行处理`build_initial_counts()`函数
- 每个线程维护本地统计结果（`ThreadLocal`结构），避免频繁的锁竞争
- 最终合并各线程的结果到全局统计中

2. **惰性删除的优先级队列**

- 使用最大堆（priority_queue）快速找到最高频的token pair
- 采用"惰性删除"策略：不直接从队列中删除过期的pair
- 当从队列顶部取出pair时，检查它是否仍然有效（频率是否与当前统计匹配）
- 避免每次合并后重建队列，复杂度从O(NlogN)降低到O(KlogN)，其中K是需要跳过的过期条目数

3. **增量更新机制**

- `merge_and_update()`函数在合并token时，只更新受影响的相邻pair
- 维护`pair_positions`索引结构，记录每个pair在哪些单词的什么位置出现
- 避免每次合并后重新扫描所有单词，大幅减少计算量

4. **高效的数据结构**

- 使用整数ID（0-255）表示字节，避免频繁的字符串操作
- 自定义哈希函数`PairHash`支持pair作为unordered_map的键
- 使用`-1`标记已合并的token，避免数据移动

5. **内存友好的表示**

- 单词存储为整数向量而不是字符串
- 词汇表使用map<int, Bytes>，支持快速ID到字节串的查找
- 特殊token在训练结束时添加，不影响核心训练过程

6. **灵活的训练控制**

- 支持指定目标词汇表大小
- 支持特殊token（如`<pad>`）
- 返回完整的合并记录，便于后续编码使用

这些优化使得算法能够高效处理大规模文本数据，特别是在构建初始统计和迭代合并阶段表现出色。并行化处理加速了初始计数，惰性删除的优先级队列减少了维护开销，而增量更新机制避免了不必要的重复计算。

用一个例子说明这些优化：

**输入数据：**

```cpp
words = {"low", "lower", "widest", "newest"}
counts = [5, 2, 3, 6]
```
初始token频率统计（频率相同时按字典序比较）：

| Token Pair | 频率 | 来源                  |
| ---------- | ---- | --------------------- |
| ("e","s")  | 9    | newest(6) + widest(3) |
| ("s","t")  | 9    | newest(6) + widest(3) |
| ("w","e")  | 8    | newest(6) + lower(2)  |
| ("l","o")  | 7    | low(5) + lower(2)     |
| ("o","w")  | 7    | low(5) + lower(2)     |

**频率相同时的字典序比较：**
- `"es"` (对应("e","s")) 
- `"st"` (对应("s","t"))
- 字典序比较：`"es" < "st"`，所以`("e","s")`的优先级**低于**`("s","t")`

在最大堆中，优先级低的会下沉，所以堆顶是`("s","t")`。

**优化1：惰性删除队列的正确工作流程**

**第一次合并：合并`("s","t")`为新token 256**

1. **初始队列状态**（最大堆，堆顶优先）：
   ```
   堆顶: ("s","t"):9  <-- 将被合并
         ("e","s"):9
         ("w","e"):8
         ("l","o"):7
         ("o","w"):7
   ```

2. **合并操作的影响**：
   
   - 单词"newest"：[110,101,119,101,115,116] → [110,101,119,101,256,-1]
   - 单词"widest"：[119,105,100,101,115,116] → [119,105,100,101,256,-1]
   
3. **增量更新（而非重新计算全部）**：
   ```cpp
   // 对于"newest"：
   // 删除受影响pairs: ("e","s"):6, ("s","t"):6
   // 添加新pairs: ("e",256):6, (256,?):不存在右邻居
   
   // 对于"widest"：
   // 删除受影响pairs: ("e","s"):3, ("s","t"):3  
   // 添加新pairs: ("e",256):3
   ```

4. **队列更新（惰性方式）**：
   - 不立即删除队列中的旧条目
   - 将新pairs推入队列：`("e",256):9`
   - 队列现在包含新旧混合条目

5. **下一次获取最高频pair时**：
   ```cpp
   while (!pair_queue.empty()) {
       best_info = pair_queue.top();  // 可能拿到过期的("s","t"):9
       pair_queue.pop();
       
       auto it = pair_counts.find(best_info.pair);
       if (it != pair_counts.end() && it->second == best_info.count) {
           // 有效，使用它
           break;
       }
       // 否则，这是过期条目，继续检查下一个
   }
   ```
   - 第一次循环：拿到`("s","t"):9`，但检查发现当前`pair_counts["s","t"] = 0`，丢弃
   - 第二次循环：拿到`("e","s"):9`，检查发现`pair_counts["e","s"] = 0`，丢弃
   - 第三次循环：拿到`("e",256):9`，检查有效，使用它进行合并

**优化2：增量更新的具体数值变化**

**合并前全局统计**：

```
pair_counts = {
    ("e","s"):9,
    ("s","t"):9,
    ("w","e"):8,
    ("l","o"):7,
    ("o","w"):7,
    ("n","e"):6,
    ("e","w"):6,  // 来自"newest"
    ("w","e"):6,  // 来自"newest"（注意有两个w,e）
    ("w","i"):3,
    ("i","d"):3,
    ("d","e"):3,
    ("e","r"):2
}
```

**合并`("s","t")`后的增量更新**：

对于"newest"（频率6）：
1. 删除左相邻`("e","s")`：`pair_counts[("e","s")] -= 6` → 从9到3
2. 删除`("s","t")`自身：`pair_counts[("s","t")] -= 6` → 从9到3
3. 添加新左相邻`("e",256)`：`pair_counts[("e",256)] += 6` → 从0到6

对于"widest"（频率3）：
1. 删除左相邻`("e","s")`：`pair_counts[("e","s")] -= 3` → 从3到0
2. 删除`("s","t")`自身：`pair_counts[("s","t")] -= 3` → 从3到0
3. 添加新左相邻`("e",256)`：`pair_counts[("e",256)] += 3` → 从6到9

**最终更新后的统计**：
```
pair_counts = {
    ("e",256):9,      // 新增最高频
    ("w","e"):8,
    ("l","o"):7,
    ("o","w"):7,
    ("n","e"):6,
    ("e","w"):6,
    ("w","e"):6,
    ("w","i"):3,
    ("i","d"):3,
    ("d","e"):3,
    ("e","r"):2
    // ("e","s"):0 已删除
    // ("s","t"):0 已删除
}
```

**优化3：并行处理的优势**

假设有8个线程，处理100万个单词：

**优化前（串行）**：
- 单线程扫描100万单词，统计所有相邻pair
- 时间复杂度：O(N×M)，其中M是平均单词长度

**优化后（并行）**：
```cpp
#pragma omp parallel for schedule(static)
for (size_t i = 0; i < 1,000,000; ++i) {
    // 每个线程处理约125,000个单词
    // 线程本地统计，无锁竞争
}
// 最后合并线程本地结果
```

**性能提升**：
- 理想情况下：8线程加速≈6-7倍
- 实际考虑线程创建、合并开销：加速≈5-6倍

**优化4：内存效率的示例**

**存储优化对比**：

| 方法        | 存储"newest"                | 合并"st"后                 | 内存占用            |
| ----------- | --------------------------- | -------------------------- | ------------------- |
| 字符串数组  | `["n","e","w","e","s","t"]` | `["n","e","w","e","st"]`   | 需要移动/复制字符串 |
| 整数ID+标记 | `[110,101,119,101,115,116]` | `[110,101,119,101,256,-1]` | 只修改两个整数      |

**哈希表键优化**：
```cpp
// 比较字符串键和整数键
std::unordered_map<std::string, int> string_map;  // 键："es"（2字节+开销）
std::unordered_map<std::pair<int,int>, int, PairHash> int_map; // 键：(101,115)（8字节）

// 搜索效率：整数比较 vs 字符串比较
int_map.find({101, 115});  // O(1)，一次哈希计算+整数比较
string_map.find("es");     // O(1)，一次哈希计算+可能的多字节比较
```

**BPE实现性能对比**

测试机器：autodl Xeon(R) Platinum 8352V 32核心CPU 60GB内存，预分词使用24个核心并行工作。

|           数据           | 版本            | 预分词  | BPE归并训练 | 总时间   |
| :----------------------: | --------------- | ------- | ----------- | -------- |
| TinyStoriesV2-GPT4-train | Python          | 29.65s  | 10min++     | 不可接受 |
| TinyStoriesV2-GPT4-train | Cpp 未优化归并  | 27.337s | 366.644s    | 394.16s  |
| TinyStoriesV2-GPT4-train | Cpp 优化归并    | 26.767s | 1.081s      | 28.03s   |
| TinyStoriesV2-GPT4-train | Rust 优化预分词 | 67.261s | -           | -        |

Python 的regex库底层c语言优化太神了，CPP的regex库对unicode支持不完善，Rust性能比Python慢一倍，正如文档所说，Python的regex库更胜一筹`...but the regex package in Python is, if anything, even faster.`。

##### **问题**（train_bpe_tinystories）：在 TinyStories 上训练 BPE（2 分）

(a) 训练耗时多久、占用多少内存？词汇表中最长的令牌是什么？是否合理？

28.03s

10GB

最长的 token 是 token：" accomplishment"，对应的 id： 7159，长度： 15 个字符（包括前面的空格）

合理。

(b) 分析代码性能。分词器训练过程中哪个部分耗时最长？

- **N**：去重后的单词总数（distinct words）
- **L**：平均单词长度（字符数/初始token数）
- **V**：目标词汇表大小
- **M**：合并次数 = V - 256 - |special_tokens|（从256个字节token开始）
- **K**：特定pair的出现次数
- **P**：临时Pair频率统计表

未优化前耗时最大是BPE归并过程，需要6分钟，时间复杂度为O(M × N × L)，空间复杂度O(N × L + P)。

优化后只要1s左右，时间复杂度为O(N × L + M)，

位置索引: O(N × L)，存储每个相邻对的位置，优先级队列：O(P)，总计空间复杂度为O(N × L + P)。优化算法需要额外的位置索引存储，但避免了每轮重新统计的开销。

优化后耗时最大为预分词过程，16进程并行30s，24进程并行25s，时间复杂度O(N*L/D)，D为进程个数。

##### **问题**（train_bpe_expts_owt）：在 OpenWebText 上训练 BPE（2 分）

(a) 在 OpenWebText 数据集上训练字节级 BPE 分词器，词汇表最大大小设为 32,000。将生成的词汇表和合并序列序列化到磁盘以便后续查看。词汇表中最长的令牌是什么？是否合理？

最长的令牌列表
  ID:  25835 | 字节长度:  64 | 内容: '----------------------------------------------------------------'
  ID:  25821 | 字节长度:  64 | 内容: 'ÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ'

(b) 对比在 TinyStories 和 OpenWebText 上训练的分词器。

有一些 token 本身里面包含换行符 \n ，写文件的时候没有做转义 ，所以「一条合并规则」在文件里被拆成了多行。

```
\n The\n           ← 中间这个 \n 是 token 本身，最后那个 \n 是行结束
        （这一行是空的，因为前面那个 \n 把光标移到下一行）
The
```

##### 2.7 实验

##### **问题**（tokenizer_experiments）：分词器实验（4 分）

(a) 从 TinyStories 和 OpenWebText 中各采样 10 个文档。使用之前训练的 TinyStories 分词器（词汇表大小 10K）和 OpenWebText 分词器（词汇表大小 32K），将这些采样文档编码为整数 ID。每个分词器的压缩比（字节数 / 令牌数）是多少？

TinyStories-10K 分词器在 TinyStories 上的压缩比为 **4.14 字节/令牌**，OpenWebText-32K 分词器在 OpenWebText 上的压缩比为 **4.70 字节/令牌**。

(b) 用 TinyStories 分词器编码 OpenWebText 采样文档会发生什么？对比压缩比，或定性描述结果。

使用 TinyStories-10K 分词器编码 OpenWebText 文档时，压缩比降至 **3.26 字节/令牌**，表明更小的词汇表（10K）在面对复杂文本时会产生更多令牌，导致压缩效率降低。

(c) 估算分词器的吞吐量（如字节 / 秒）。编码 Pile 数据集（825GB 文本）需要多长时间？

TinyStories-10K 分词器吞吐量约 **626,519.6 字节/秒**，编码 825GB Pile 数据集约需 **16.4 天**；OpenWebText-32K 约 **763,734.4 字节/秒**，约需 **13.4天**。

(d) 使用 TinyStories 和 OpenWebText 分词器，分别将对应的训练集和开发集编码为整数令牌 ID（后续用于训练语言模型）。建议将令牌 ID 序列化为`uint16`类型的 NumPy 数组。为什么`uint16`是合适的选择？

两个分词器的词汇表大小（10K 和 32K）均小于 65,536（2¹⁶），因此 uint16 足以表示所有令牌 ID，且比 uint32 节省 50% 存储空间。

##### 3.6 资源核算（Resource accounting）
了解 Transformer 各组成部分的计算量和内存占用情况十分有用。我们将逐步开展基础的“浮点运算次数（FLOPs）核算”。

Transformer 的绝大多数浮点运算都来自矩阵乘法，因此核心思路很简单：

- 1.列出 Transformer 前向传播过程中所有的矩阵乘法操作；

- 2.将每个矩阵乘法转换为所需的浮点运算次数（FLOPs）。

对于第二步，以下结论会有所帮助：规则：给定矩阵 \(A \in \mathbb{R}^{m×n}\) 和 \(B \in \mathbb{R}^{n×p}\)，矩阵乘法 AB 需消耗 2mnp 个浮点运算（FLOPs）。推导如下：矩阵乘法结果 AB 的第 \([i, j]\) 个元素等于 A 的第 i 行与 B 的第 j 列的点积（即 \((AB)[i, j] = A[i, :] · B[:, j]\)），该点积运算包含 n 次加法和 n 次乘法，共 2n 个浮点运算；而矩阵 AB 共有 \(m×p\) 个元素，因此总浮点运算次数为 \((2n)×(mp) = 2mnp\)。在解决下一个问题前，建议逐一梳理 Transformer 块和 Transformer 语言模型（LM）的每个组件，列出所有矩阵乘法操作及其对应的浮点运算成本。

#### 问题（transformer_accounting）：Transformer 语言模型资源核算（5 分）

(a) 考虑 GPT-2 XL 模型，其配置如下：

- 词汇表大小（vocab_size）：50,257
- 上下文长度（context_length）：1,024
- 层数（num_layers）：48
- 模型维度（d_model）：1,600
- 注意力头数（num_heads）：25
- 前馈网络内层维度（d_ff）：6,400

若按此配置构建模型，该模型将包含多少个可训练参数？假设每个参数采用单精度浮点数（single-precision floating point）表示，仅加载该模型需要占用多少内存？

##### 1. 令牌嵌入层（Token Embedding）

作用：将整数令牌 ID 映射为 d_model 维向量，参数是嵌入矩阵。

- 矩阵维度：`vocab_size × d_model`（词汇表中每个令牌对应一个 d_model 维向量）
- 计算：50257 × 1600 = 80,411,200（约 8041 万）

##### 2. 单一及总Transformer 块的参数（共 48 层，每层参数相同）

##### （1）多头自注意力（MHA）的参数

MHA 包含 Q/K/V 投影、输出投影 4 个权重矩阵，且`d_k = d_model / num_heads = 1600 / 25 = 64`（每个头的维度）：

- Q/K/V 投影矩阵（3 个）：每个矩阵维度`d_model × d_model`（因需将 d_model 维向量拆分为 num_heads 个 d_k 维向量，等价于`d_model × (num_heads×d_k) = d_model×d_model`）
  - 单个投影矩阵参数：1600 × 1600 = 2,560,000
  - 3 个总参数：3 × 2,560,000 = 7,680,000
- 输出投影矩阵（1 个）：维度`d_model × d_model`（将 num_heads 个 d_k 维向量拼接后的 d_model 维向量，映射回 d_model 维）
  - 参数：1600 × 1600 = 2,560,000
- MHA 单一层总参数：7,680,000 + 2,560,000 = 10,240,000（1024 万）

##### （2）前馈网络（FFN，SwiGLU 架构）的参数

FFN 包含 3 个权重矩阵（W1、W2、W3），维度分别为`d_ff×d_model`、`d_model×d_ff`和`d_ff×d_model`：

- W1（输入→内层）：6400 × 1600 = 10,240,000（1024 万）

- W2（内层→输出）：1600 × 6400 = 10,240,000（1024 万）

- W3（输入→内层）：6400 × 1600 = 10,240,000（1024 万）

- FFN 单一层总参数：10,240,000 * 3 = 30,720,000（3072 万）

##### （3）2 个 RMSNorm 层参数

  - 每个 RMSNorm 的增益参数 \(g\) 维度：(d_model, )，(1600 个参数 / 层）
  - 2 个 RMSNorm 总参数：\(2×1600 = 3,200\)（3200 个 / 层）
  - 预归一化 Transformer 块包含 **2 个 RMSNorm 层**：分别在 MHA 前和 SwiGLU 前。

##### （4）Transformer 块总参数

ROPE 没有需要训练的参数，不变参数全部预计算并缓存。

计算：MHA 参数 + FFN 参数 + RMSNorm 参数 = 10,240,000 + 30,720,000 + 3,200 = 40,963,200（4096.3200 万 / 层）
48 层总参数：48 × 40,963,200 = 1,966,233,600（约 19.66 亿）

##### 3. 归一化层（RMSNorm）

- RMSNorm 的增益参数 \(g\) 维度：(d_model, )，(1600 个参数 / 层）

##### 4. 输出投影层（LM Head，与嵌入层权重不共享）

作用：将 Transformer 输出的 d_model 维向量映射到词汇表维度，预测下一个令牌。

- 矩阵维度：`vocab_size × d_model`

- 计算：50257 × 1600 = 80,411,200（约 8041 万）

将所有组件参数相加：

\(\begin{align*} \text{总参数} &= 80,411,200 + 1,966,233,600 + 1,600 + 80,411,200 \\ &= 2,127,057,600 \end{align*}\)

**ANSWER：**总可训练参数约 **21.28亿**，加载该模型需要占用 **7.92390704155 GB（约 8 GB） **。


(b) 明确完成 GPT-2 XL 型模型前向传播所需的所有矩阵乘法操作。这些矩阵乘法总共需要多少浮点运算次数（FLOPs）？假设输入序列长度等于上下文长度（context_length）。列出所有矩阵乘法操作（含描述），并给出总浮点运算次数。

| 矩阵乘法操作             | 维度说明                                                     |
| ------------------------ | ------------------------------------------------------------ |
| 词嵌入投影（Embedding）  | 通常只是查表，无矩阵乘法                                     |
| 多头注意力 Q/K/V 投影    | 每层：(seq_len, d_model) × (d_model, d_model) -> (seq_len, d_model)（Q、K、V 各 1 次，共 3 次） |
| 点积注意力分数计算       | 每层：(seq_len, d_k) × (d_k, seq_len) -> (seq_len, seq_len)  |
| 点积注意力值加权求和计算 | 每层：(seq_q, seq_k) × (seq_k, d_v) -> (seq_q, d_v)          |
| 多头注意力输出投影       | 每层：(seq_len, d_model) × (d_model, d_model) -> (seq_len, d_model) |
| 前馈网络第一层（W1）     | 每层：(seq_len, d_model) × (d_model, d_ff) -> (seq_len, d_ff) |
| 前馈网络第二层（W2）     | 每层：(seq_len, d_ff) × (d_ff, d_model) -> (seq_len, d_model) |
| 前馈网络第二层（W3）     | 每层：(seq_len, d_model) × (d_model, d_ff) -> (seq_len, d_ff) |
| 输出层投影（LM Head）    | (seq_len, d_model) × (d_model, vocab_size) -> (seq_len, vocab_size) |

当 seq_len = context_length = 1024，GPT-2 XL一次完整的前向传播（序列长度1024）大约需要 **4.20万亿次浮点运算（4.20 TFLOPs)**。

(c) 根据上述分析，模型的哪些部分消耗的浮点运算次数（FLOPs）最多？

FLOPs分布:

- 注意力Q/K/V投影: 17.96%
- 点积注意力分数计算: 0.15%
- 点积注意力权重值计算: 0.15%
- 注意力输出投影: 5.99%
- 前馈网络第一层: 23.94%
- 前馈网络第二层: 23.94%
- 前馈网络第三层: 23.94%
- 输出层: 3.92%
- 前馈网络总计: 71.83%
- 注意力总计: 24.25%

从以上计算可以看出，**前馈网络**是计算量最大的部分，在 GPT-2 XL 中约占总 FLOPs 的71.83%。这主要因为其内部有一个从 `d` 到 `d_ff`（通常4倍于`d`）的大维度矩阵乘法。其次是**注意力模块中的Q/K/V投影，**约占总 FLOPs 的17.96%。

(d) 对 GPT-2 小模型（12 层、d_model=768、12 个注意力头）、GPT-2 中模型（24 层、d_model=1024、16 个注意力头）和 GPT-2 大模型（36 层、d_model=1280、20 个注意力头）重复上述分析。随着模型规模增大，Transformer 语言模型的哪些部分占总浮点运算次数（FLOPs）的比例会增加或减少？针对每个模型，给出各组件及其对应的浮点运算次数占前向传播总浮点运算次数的比例；此外，用 1-2 句话描述模型规模变化如何影响各组件浮点运算次数的占比。

下表计算了不同规模的GPT-2模型在序列长度为 1024 时，各组件 FLOPs 占总量的比例。

| 模型组件                   | GPT-2 Small | GPT-2 Medium | GPT-2 Large | GPT-2 XL | 趋势说明                                                     |
| :------------------------- | :---------- | :----------- | :---------- | :------- | :----------------------------------------------------------- |
| **多头注意力 Q/K/V投影**   | 13.84%      | 16.51%       | 17.47%      | 17.96%   | **占比小幅增加并趋于稳定**。其计算量（`~6SLd²`）与模型维度`d²`成正比，增长速度快于与`d`成正比的组件，但慢于前馈网络。 |
| **点积注意力分数计算**     | 0.51%       | 0.34%        | 0.23%       | 0.15%    | **占比急剧下降**。其计算量（`~2S²Ld_k`）仅与序列长度`S²`和头维度`d_k`相关，当`d`增大而`S`固定时，占比被显著稀释。 |
| **点积注意力加权求和计算** | 0.51%       | 0.34%        | 0.23%       | 0.15%    | 趋势同“分数计算”，原因完全相同。                             |
| **多头注意力输出投影**     | 4.61%       | 5.50%        | 5.82%       | 5.99%    | **占比小幅增加并趋于稳定**。原因同Q/K/V投影，计算量（`~2SLd²`）与`d²`成正比。 |
| **前馈网络 (三层)**        | 55.36%      | 66.04%       | 69.89%      | 71.83%   | **占比显著且持续增加**，是总FLOPs的绝对主体。其计算量（`~2SLd·d_ff`，通常`d_ff=4d`）与`d²`成正比，且因层数`L`的累加效应，增长最快。 |
| **输出层投影**             | 25.16%      | 11.25%       | 6.35%       | 3.92%    | **占比断崖式下降**。该操作仅一次，计算量（`~2SdV`）仅与`d`线性相关，远慢于随`L`线性增长的层内计算，故相对重要性迅速降低。 |

基于上表数据，可以清晰地观察到以下趋势：

- **前馈网络的FLOPs占比显著增加**：从Small模型的 **55.36%** 增长到XL模型的 **71.83%**。这是因为前馈网络的计算成本（`~2 * S * L * d_model * d_ff`）与模型维度 `d_model` 和其内部维度 `d_ff`（通常为4倍）的乘积强相关，而 `d_model` 和层数 `L` 的同步增长会使其计算量增速超过其他部分。
- **注意力分数及加权求和计算的FLOPs占比急剧下降**：从Small模型的 **1.02%** 下降到XL模型的 **0.30%**。这是因为该部分计算成本与层数 `L`、序列长度平方（`S²`）和注意力头维度（`d_k`）相关（`~2 * S² * L * d_k`），而与模型主体维度 `d_model` 无关。由`d_k`=`d_model`/`num_heads`，可发现 3 个模型的`d_k`均为 **64**，故该部分计算成本实际仅与`L`线性相关。考虑一次前向传播，当 `d_model` 增大而 `S` 和`d_k`固定时，该值不变，其占比自然被稀释。再考虑多层，占比不变，而由于加入输出层投影导致分母增大，占比在总体中更小。
- **输出层投影的FLOPs占比大幅下降**：从Small模型的 **25.16%** 骤降至XL模型的 **3.92%**。这是因为输出层（`(S, d) × (d, V)`）仅在整个模型末端计算一次，其FLOPs（`~2 * S * d_model * V`）仅随 `d_model` 线性增长。而层内的计算（注意力、前馈）会随层数 `L` 线性增长，导致输出层的相对贡献迅速变小。

**核心结论**：随着Transformer模型规模的扩大（增加层数和维度），计算瓶颈会从**与模型维度平方相关的前馈网络**和**单次的输出投影**，快速转移到**前馈网络**上。这使得大模型在长序列推理时，前馈网络成为绝对的FLOPs消耗主体。

1. **主导部分转移**：随着**与词汇表大小`V`强相关的输出层占比越来越小**，计算主力彻底转变为**与模型维度平方`d²`强相关的前馈网络**。
2. **注意力“轻量化”**：**注意力核心计算（分数与加权）** 的占比变得微乎其微，尤其在长模型、短上下文场景下。
3. **扩展启示**：该趋势解释了为何优化大模型推理时，**降低前馈层的计算/存储开销**（如使用MoE、量化）是关键方向；而在处理极长序列时，**注意力计算的`O(S²)`复杂度**会重新成为瓶颈。

(e) 针对 GPT-2 XL 模型，将上下文长度（context_length）增加到 16,384。此时，一次前向传播的总浮点运算次数（FLOPs）会发生怎样的变化？各组件浮点运算次数的相对贡献会如何变化？

当上下文长度从 1,024 增加到 16,384 时：

- 总 FLOPs 从约 3.5T 增加到约 70.4T，增长了约 20 倍（不是简单的线性增长，因为注意力计算包含 S² 项）
- 各组件相对贡献变化：
  - 自注意力模块的 FLOPs 占比从约 24.25% 增加到约 27.58%（包含注意力分数计算部分）
  - 前馈网络（FFN）的占比从约 71.83% 下降到约 68.68%
  - 语言模型头的占比从约 3.92% 下降到约 3.75%

#### 4. 训练 Transformer 语言模型
##### 4.1 交叉熵损失
Transformer 语言模型会为长度为 \(m+1\) 的序列 x 和每个位置 \(i=1,…,m\) 定义分布 \(p_{\theta}(x_{i+1} | x_{1: i})\)。

给定由长度为m的序列组成的训练集D，我们定义标准的交叉熵（负对数似然）损失函数：

\(\ell(\theta ; D)=\frac{1}{|D|} \sum_{x \in D} \sum_{i=1}^{m}-\log p_{\theta}\left(x_{i+1} | x_{1: i}\right)\)

Transformer 的单次前向传播会同时输出所有\(i=1,…,m\)对应的\(p_{\theta}(x_{i+1} | x_{1: i})\)，其中|D|为训练集大小(batch_size)。

具体来说，Transformer 会为每个位置 i 计算对数几率（logits）\(o_{i} \in \mathbb{R}^{vocab\_size}\)，由此可得：

\(p\left(x_{i+1} | x_{1: i}\right)=\text{softmax}\left(o_{i}\right)\left[x_{i+1}\right]=\frac{\exp \left(o_{i}\left[x_{i+1}\right]\right)}{\sum_{a=1}^{vocab\_size} \exp \left(o_{i}[a]\right)}\)

交叉熵损失通常基于对数几率向量\(o_{i} \in \mathbb{R}^{vocab\_size}\)和目标值\(x_{i+1}\)定义。与 softmax 类似，实现交叉熵损失时需要注意数值稳定性问题。

我们已经实现了 softmax 的稳定版本，对其取自然对数得 log_softmax：
\[
\log\left(\frac{e^{x_i}}{\sum_j e^{x_j}}\right) = \log\left(\frac{e^{x_i-m}}{\sum_j e^{x_j-m}}\right) = (x_i - m) - \log\sum_j e^{x_j-m}
\]

**Problem（cross_entropy）：实现交叉熵**

```python
import torch
from .softmax import log_softmax


def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor):
    # inputs: (batch_size, vocab_size)
    # targets: (batch_size, )
    # (batch_size, vocab_size)
    D = inputs.shape[0]

    # 1. 计算softmax概率
    probs = log_softmax(inputs, dim=-1)

    # 2. 提取目标位置的概率
    p = probs[torch.arange(D), targets]

    # 3. 计算平均损失（除以batch_size）
    return -torch.mean(p)
```

###### **Problem**（learning_rate_tuning）：调整学习率

如前所述，学习率是影响训练效果最重要的超参数之一。在我们的简单示例中实际验证这一点：使用上述 SGD 示例，分别尝试另外三个学习率值（1e1、1e2、1e3），仅训练 10 次迭代。观察每个学习率对应的损失变化：是衰减更快、更慢，还是发散（即训练过程中损失增加）？

```bash
(cs336-basics) (base) koschei@192 assignment1-basics % uv run ./cs336_basics/sgd.py
learing rate: 10.0
step1 , loss: 20.765413284301758
step2 , loss: 13.289864540100098
step3 , loss: 9.796720504760742
step4 , loss: 7.66488790512085
step5 , loss: 6.208559036254883
step6 , loss: 5.14760684967041
step7 , loss: 4.341323375701904
step8 , loss: 3.709784507751465
step9 , loss: 3.203690767288208
step10, loss: 2.7907707691192627
(cs336-basics) (base) koschei@192 assignment1-basics % uv run ./cs336_basics/sgd.py
learing rate: 100.0
step1 , loss: 29.470537185668945
step2 , loss: 29.470535278320312
step3 , loss: 5.0563435554504395
step4 , loss: 0.12100967764854431
step5 , loss: 1.2657409207190063e-16
step6 , loss: 1.4107469187345678e-18
step7 , loss: 4.750480809550992e-20
step8 , loss: 2.8298939276754454e-21
step9 , loss: 2.427666298147088e-22
step10, loss: 2.697406997941209e-23
(cs336-basics) (base) koschei@192 assignment1-basics % uv run ./cs336_basics/sgd.py
learing rate: 1000.0
step1 , loss: 23.761384963989258
step2 , loss: 8577.859375
step3 , loss: 1481531.25
step4 , loss: 164804544.0
step5 , loss: 13349167104.0
step6 , loss: 842485268480.0
step7 , loss: 43250442305536.0
step8 , loss: 1860819142836224.0
step9 , loss: 6.858582164871578e+16
step10, loss: 2.2023668704120668e+18
```

**ANSWER: ** 学习率 10 时损失缓慢衰减；学习率 100 时损失迅速衰减，极速收敛至接近零；学习率 1000 时损失爆炸式增长，明显发散。

**Problem (adamw): 实现 AdamW**

```python
import torch
import math


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        if lr < 0:
            raise ValueError(f"无效的学习率：{lr}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            # 获取学习率等参数
            alpha = group["lr"]
            beta1, beat2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                # 获取与参数 p 相关的状态
                state = self.state[p]

                # 从状态中获取迭代次数，若无则初始化为 1
                t = state.get("t", 1)
                # 一阶矩估计
                m = state.get("m", torch.zeros_like(p))
                # 二阶矩估计
                v = state.get("v", torch.zeros_like(p))

                # 获取损失相对于p的梯度
                g = p.grad.data

                # 更新一、二阶矩估计
                m = beta1 * m + (1 - beta1) * g
                v = beat2 * v + (1 - beat2) * g * g

                # 计算当前迭代的调整后学习率 αt
                alpha_t = alpha * math.sqrt(1 - beat2**t) / (1 - beta1**t)
                # 更新参数
                p.data -= alpha_t * m / (torch.sqrt(v) + eps)
                # 应用权重衰减
                p.data -= alpha * weight_decay * p.data

                # 递增迭代次数
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v

        return loss
```

**Problem（AdamW Accounting）：AdamW 训练的资源核算**

计算运行 AdamW 所需的内存和计算资源。假设所有张量都使用 float32 精度。

**(a) 运行 AdamW 需要多少峰值内存？**

根据参数、激活值、梯度和优化器状态的内存使用情况分解答案。用批量大小（batch_size）和模型超参数（vocab_size、context_length、num_layers、d_model、num_heads）表示。假设\(d_{ff}=4 \times d_{model}\)。

为简化计算，激活值的内存使用仅考虑以下组件：

- Transformer 块
  - RMSNorm 层
  - 多头自注意力子层：QKV 投影、\(Q^{\top} K\)矩阵乘法、softmax、值的加权和、输出投影
  - 位置 - wise 前馈网络：\(W_{1}\)矩阵乘法、SiLU 激活、\(W_{2}\)矩阵乘法
- 最终的 RMSNorm
- 输出嵌入
- 对数几率的交叉熵计算

分别给出参数、激活值、梯度和优化器状态的代数表达式，以及总内存的代数表达式。

- **参数内存**：  
	$$
  M_{\text{params}} = 4 \left( 2Vd + L(16d^2 + 2d) + d \right)
  $$
  其中 \(V\) 为词表大小，\(d\) 为模型维度，\(L\) 为层数。

- **梯度内存**：  
  \[
  M_{\text{params}} = 4 \left( 2Vd + L(16d^2 + 2d) + d \right)
  \]

- **优化器状态内存**（AdamW 存储一阶矩和二阶矩）：  
  \[
  M_{\text{opt}} = 8 \left( 2Vd + L(16d^2 + 2d) + d \right)
  \]

- **激活内存**（基于中间张量的保守估计）：  
  \[
  M_{\text{act}} = 4 \times \left[ L \times (16 B T d + 2 B h T^2) + B T d + 2 B T V \right]
  \]
  其中 \(B\) 为批次大小，\(T\) 为上下文长度，\(h\) 为注意力头数。

- **总峰值内存**：
  \[
  M_{\text{total}} = 16 \left( 2Vd + L(16d^2 + 2d) + d \right) + 4 \times \left[ L \times (16 B T d + 2B h T^2) + B T d + 2 B T V \right]
  \]

参考 DeepSeek 给的思路，给出激活内存的计算过程。

**激活内存计算过程**

激活内存指前向传播中需要存储的中间变量，用于反向传播计算梯度。根据给定的组件，我们逐项计算每个组件的激活内存（以元素数量计）：

**1. Transformer 块（共 L 层）**

每个 Transformer 层包含以下部分，其激活内存计算如下：

- **RMSNorm(s)**:  
  输入和输出形状均为 `[B, T, d]`，需要存储输出（输入通常来自上一层已存储）。按存储输出计算：`B × T × d`。  
  每层有两个 RMSNorm，共 `2 × B × T × d`。
- **多头自注意力子层**:
  - **QKV 投影**：通过线性层将输入 `[B, T, d]` 投影为 Q、K、V。通常合并为一个输出张量，形状 `[B, T, 3d]`，然后拆分为三个独立的 `[B, T, d]` 张量。需存储这三个张量，共 `3 × B × T × d`。
  - **Q⊤K 矩阵乘法**：计算注意力分数，输出形状 `[B, h, T, T]`，需存储。元素数量为 `B × h × T × T`。
  - **softmax**：对注意力分数进行归一化，输出形状与输入相同，需存储。元素数量为 `B × h × T × T`。
  - **加权和**：将注意力权重与 V 相乘，得到每个头的输出，然后合并多头，输出形状 `[B, T, d]`，需存储。元素数量为 `B × T × d`。
  - **输出投影**：线性层，输入 `[B, T, d]`，输出 `[B, T, d]`，需存储。元素数量为 `B × T × d`。
- **位置前馈网络**:
  - **W1 矩阵乘法**：将输入 `[B, T, d]` 投影到 `[B, T, 4d]`（因 `d_ff = 4 × d`），需存储。元素数量为 `4 × B × T × d`。
  - **SiLU 激活**：输入和输出均为 `[B, T, 4d]`，需存储输出。元素数量为 `4 × B × T × d`。
  - ~~**W3 矩阵乘法**：将输入 `[B, T, d]` 投影到 `[B, T, 4d]`（因 `d_ff = 4 × d`），需存储。元素数量为 `4 × B × T × d`。~~
  - ~~**GLU门控**：输入和输出`[B, T, 4d]`，需存储输出。元素数量为 `4 × B × T × d`。~~
  - **W2 矩阵乘法**：将 `[B, T, 4d]` 投影回 `[B, T, d]`，需存储。元素数量为 `B × T × d`。

此外，每个 Transformer 层还有残差连接，需要存储层的输入（来自上一层输出）和最终输出。但这些通常已在其他组件中考虑或可重用，不单独计算。

汇总一个 Transformer 层的激活内存元素数量：
\[
(2 + 3 + 1 + 1 + 4 + 4 + 1) \times B \times T \times d + 2 \times B \times h \times T^2 = 16 \times B \times T \times d + 2 \times B \times h \times T^2
\]
故根据题目要求，我们采用给定表达式：
\[
\text{每层激活内存（元素）} = 16 \times B \times T \times d + 2 \times B \times h \times T^2
\]
**2. 其他组件**

- **最终 RMSNorm**：输入和输出形状 `[B, T, d]`，需存储输出。元素数量为 `B × T × d`。
- **输出嵌入**：将隐藏状态投影到词表，输出 logits 形状 `[B, T, V]`，需存储。元素数量为 `B × T × V`。
- **交叉熵损失**：需要 logits 计算损失，logits 已存储；需要计算 log_softmax，临时存储probs`B × T × V`；标签通常为整数，不占显著激活内存。

**3. 总激活内存**

综合以上，总激活内存元素数量为：
\[
L \times (16 \times B \times T \times d + 2 \times B \times h \times T^2) + B \times T \times d + 2 \times B \times T \times V
\]
转换为字节（乘以 4，因 float32 占 4 字节）：
\[
M_{\text{act}} = 4 \times \left[ L \times (16 B T d + 2 B h T^2) + B T d + 2 B T V \right]
\]
**(b) 针对 GPT-2 XL 规模的模型实例化答案，得到仅依赖于批量大小（batch_size）的表达式。在 80GB 内存中，最多可使用多大的批量大小？**

给出形如a×batch_size+b的表达式（其中a、b为数值），以及最大批量大小的数值。

总内存：
\[
M_{\text{total}} \approx 14.45 \times B + 31.70\ \text{GB}
\]
设内存上限为 80 GB：
\[
14.45B + 31.70 \leq 80 \implies B \leq \frac{80 - 31.70}{14.45} \approx 3.34
\]
最大批次大小为 **3**。

**(c) 执行一次 AdamW 步骤需要多少 FLOPs？**

给出代数表达式，并简要说明理由。

GPT-2 XL 参数总数 P = 2,127,057,600

AdamW 更新每个参数约需 10 次浮点操作：

  1. 计算一阶矩估计: 2 FLOPs (乘法和加法)
  2. 计算二阶矩估计: 2 FLOPs (乘法和加法)
  3. 偏差修正: 2 FLOPs (幂运算和除法)
  4. 参数更新: 4 FLOPs (除法、平方根、乘法、减法)
    总计: ~10 FLOPs/参数

一步 AdamW 的总 FLOPs = 10 × P = 2.13e+10 次

**(d) 模型 FLOPs 利用率（MFU）定义为观测吞吐量（每秒处理的 token 数）与硬件理论峰值 FLOP 吞吐量的比值。NVIDIA A100 GPU 的 float32 运算理论峰值为 19.5 太拉 FLOP/s。假设 MFU 为 50%，在单个 A100 上，使用批量大小 1024 训练 GPT-2 XL 400K 步骤需要多长时间？按照 Kaplan 等人和 Hoffmann 等人的假设，反向传播的 FLOPs 是前向传播的两倍。**

给出训练所需的天数，并简要说明理由。

每训练步的 FLOPs 主要来自前向与后向传播。假设：
- 前向传播 FLOPs 约为 `2 × B × T × P`（每个参数一次乘加，即 2 FLOPs）。
- 后向传播 FLOPs 约为前向的 2 倍。

每步总 FLOPs：
\[
\text{FLOPs}_{\text{step}} \approx 6 \times B \times T \times P
\]
代入 `B = 1024`，`T = 1024`，`P = 2,127,057,600`：
\[
\text{FLOPs}_{\text{step}} \approx 6 \times 1024 \times 1024 \times 2,127,057,600 \approx 1.34 \times 10^{16} \ \text{FLOPs}
\]
总步数 400k，总 FLOPs：
\[
\text{FLOPs}_{\text{total}} \approx 400,\!000 \times 13,382,289,299,865,600 = 5.35 \times 10^{21} \ \text{FLOPs}
\]
NVIDIA A100 峰值吞吐为 `19.5 TFLOPS = 1.95 × 10^13 FLOP/s`，50% MFU 下实际吞吐：
\[
\text{Throughput} = 0.5 \times 1.95 \times 10^{13} = 9.75 \times 10^{12} \ \text{FLOP/s}
\]
训练时间：
\[
t = \frac{5.35 \times 10^{21}}{9.75 \times 10^{12}} \approx 5.5 \times 10^8 \ \text{秒} \approx 6354.4 \ \text{天} \approx 17.4 \ \text{年}
\]
因此，在单 A100 上以 50% MFU 训练需约 **17.4 年**。

