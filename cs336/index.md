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

