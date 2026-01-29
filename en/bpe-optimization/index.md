# High-Performance BPE Tokenizer Optimization: From 10 Minutes to 1 Second


> This article is supplementary reading for [CS336 Assignment 1](/posts/cs336-assign1/), providing a detailed introduction to the optimized implementation of the BPE tokenizer.

## Background

The recommended cppyy in the documentation has issues in Mac and Linux environments. To pursue high performance, I used Pybind11 to bind C++ code: pre-tokenization is handled by Python, while the BPE merge process is delegated to C++. The actual biggest bottleneck is still pre-tokenization, which can be parallelized using the existing code `pretokenization_example.py` for chunked parallel processing (8 cores 100s → 16 cores 30s).

## Core Optimization Strategies

### 1. Parallelization

- Use OpenMP to parallelize the `build_initial_counts()` function
- Each thread maintains local statistics (`ThreadLocal` structure) to avoid frequent lock contention
- Finally merge each thread's results into global statistics

### 2. Lazy Deletion Priority Queue

- Use a max heap (priority_queue) to quickly find the highest-frequency token pair
- Adopt a "lazy deletion" strategy: don't directly delete expired pairs from the queue
- When extracting a pair from the top of the queue, check if it's still valid (whether the frequency matches current statistics)
- Complexity reduced from O(NlogN) to O(KlogN), where K is the number of expired entries to skip

### 3. Incremental Update Mechanism

- The `merge_and_update()` function only updates affected adjacent pairs when merging tokens
- Maintain a `pair_positions` index structure to record where each pair appears in which words and at what positions
- Avoid rescanning all words after each merge, greatly reducing computation

### 4. Efficient Data Structures

- Use integer IDs (0-255) to represent bytes, avoiding frequent string operations
- Custom hash function `PairHash` supports pairs as keys in unordered_map
- Use `-1` to mark merged tokens, avoiding data movement

### 5. Memory-Friendly Representation

- Words are stored as integer vectors instead of strings
- Vocabulary uses `map<int, Bytes>`, supporting fast ID to byte string lookup
- Special tokens are added at the end of training, not affecting the core training process

### 6. Flexible Training Control

- Support specifying target vocabulary size
- Support special tokens (like `<pad>`)
- Return complete merge records for subsequent encoding use

---

## Detailed Example: Optimized Workflow

Let's use a concrete example to illustrate how these optimizations work together.

### Input Data

```cpp
words = {"low", "lower", "widest", "newest"}
counts = [5, 2, 3, 6]
```

Initial token frequency statistics (when frequencies are equal, compare by lexicographic order):

| Token Pair | Frequency | Source                  |
| ---------- | --------- | ----------------------- |
| ("e","s")  | 9         | newest(6) + widest(3)   |
| ("s","t")  | 9         | newest(6) + widest(3)   |
| ("w","e")  | 8         | newest(6) + lower(2)    |
| ("l","o")  | 7         | low(5) + lower(2)       |
| ("o","w")  | 7         | low(5) + lower(2)       |

**Lexicographic comparison when frequencies are equal:**
- `"es"` corresponds to ("e","s")
- `"st"` corresponds to ("s","t")
- Lexicographic comparison: `"es" < "st"`, so ("e","s") has **lower** priority than ("s","t")

In the max heap, lower priority items sink, so the top of the heap is ("s","t").

### Lazy Deletion Queue Workflow

**First Merge: Merge ("s","t") into new token 256**

1. **Initial queue state** (max heap, top priority first):
   ```
   Top: ("s","t"):9  <-- will be merged
         ("e","s"):9
         ("w","e"):8
         ("l","o"):7
         ("o","w"):7
   ```

2. **Impact of merge operation**:
   - Word "newest": `[110,101,119,101,115,116]` → `[110,101,119,101,256,-1]`
   - Word "widest": `[119,105,100,101,115,116]` → `[119,105,100,101,256,-1]`

3. **Incremental update (instead of recalculating everything)**:
   ```cpp
   // For "newest":
   // Delete affected pairs: ("e","s"):6, ("s","t"):6
   // Add new pairs: ("e",256):6

   // For "widest":
   // Delete affected pairs: ("e","s"):3, ("s","t"):3
   // Add new pairs: ("e",256):3
   ```

4. **Queue update (lazy approach)**:
   - Don't immediately delete old entries from the queue
   - Push new pairs into the queue: `("e",256):9`
   - Queue now contains mixed old and new entries

5. **When getting the next highest frequency pair**:
   ```cpp
   while (!pair_queue.empty()) {
       best_info = pair_queue.top();
       pair_queue.pop();

       auto it = pair_counts.find(best_info.pair);
       if (it != pair_counts.end() && it->second == best_info.count) {
           break;  // Valid, use it
       }
       // Otherwise, this is an expired entry, continue checking next
   }
   ```

### Specific Numeric Changes in Incremental Update

**Global statistics before merge**:

```
pair_counts = {
    ("e","s"):9,
    ("s","t"):9,
    ("w","e"):8,
    ("l","o"):7,
    ("o","w"):7,
    ("n","e"):6,
    ("e","w"):6,
    ...
}
```

**Incremental update after merging ("s","t")**:

For "newest" (frequency 6):
1. Delete left adjacent ("e","s"): `pair_counts[("e","s")] -= 6` → from 9 to 3
2. Delete ("s","t") itself: `pair_counts[("s","t")] -= 6` → from 9 to 3
3. Add new left adjacent ("e",256): `pair_counts[("e",256)] += 6` → from 0 to 6

For "widest" (frequency 3):
1. Delete left adjacent ("e","s"): `pair_counts[("e","s")] -= 3` → from 3 to 0
2. Delete ("s","t") itself: `pair_counts[("s","t")] -= 3` → from 3 to 0
3. Add new left adjacent ("e",256): `pair_counts[("e",256)] += 3` → from 6 to 9

### Advantages of Parallel Processing

Suppose we have 8 threads processing 1 million words:

**Before optimization (serial)**:
- Single thread scans 1 million words, counting all adjacent pairs
- Time complexity: O(N×M), where M is average word length

**After optimization (parallel)**:
```cpp
#pragma omp parallel for schedule(static)
for (size_t i = 0; i < 1000000; ++i) {
    // Each thread processes about 125,000 words
    // Thread-local statistics, no lock contention
}
// Finally merge thread-local results
```

**Performance improvement**:
- Ideal case: 8 threads speedup ≈ 6-7x
- Actual considering thread creation/merge overhead: speedup ≈ 5-6x

### Memory Efficiency Comparison

| Method         | Store "newest"               | After merging "st"          | Memory Usage              |
| -------------- | ---------------------------- | --------------------------- | ------------------------- |
| String array   | `["n","e","w","e","s","t"]`  | `["n","e","w","e","st"]`    | Requires moving/copying strings |
| Integer ID+tag | `[110,101,119,101,115,116]`  | `[110,101,119,101,256,-1]`  | Only modify two integers  |

---

## Performance Comparison

Test machine: autodl Xeon(R) Platinum 8352V 32-core CPU with 60GB RAM, pre-tokenization uses 24 cores for parallel processing.

|           Data           | Version             | Pre-tokenization | BPE Merge Training | Total Time   |
| :----------------------: | ------------------- | ---------------- | ------------------ | ------------ |
| TinyStoriesV2-GPT4-train | Python              | 29.65s           | 10min++            | Unacceptable |
| TinyStoriesV2-GPT4-train | Cpp unoptimized merge | 27.337s        | 366.644s           | 394.16s      |
| TinyStoriesV2-GPT4-train | Cpp optimized merge | 26.767s          | 1.081s             | 28.03s       |
| TinyStoriesV2-GPT4-train | Rust optimized pre-tokenization | 67.261s | -                  | -            |

> Python's regex library has excellent underlying C optimization. C++'s regex library has incomplete Unicode support, and Rust is actually twice as slow as Python. As the documentation states: *"...but the regex package in Python is, if anything, even faster."*

## Summary

These optimizations enable the algorithm to efficiently process large-scale text data, with particularly excellent performance in building initial statistics and iterative merge stages:

- **Parallelization** speeds up initial counting
- **Lazy deletion priority queue** reduces maintenance overhead
- **Incremental update mechanism** avoids unnecessary repeated calculations

The final result is a performance improvement from over 10 minutes to about 1 second, a speedup of over **300x**.

