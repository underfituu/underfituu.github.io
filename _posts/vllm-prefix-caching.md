## 摘要

随着生成式人工智能进入大规模落地阶段，大语言模型（LLM）的推理性能优化已成为行业关注的核心。在大规模、长上下文的交互场景下，计算资源的浪费和显存带宽的瓶颈限制了系统的高效运转。前缀缓存（Prefix Caching）作为一种通过空间换时间、通过存储换计算的关键技术，正逐步从单机内存管理演变为跨节点的分布式系统优化方案。本报告将从前缀缓存的核心原理出发，重点探讨其在国产昇腾（Ascend）平台上的性能分析手段（涵盖常规输入与超长输入场景）。

## 1. 原理概述

### 1.1 什么是前缀缓存（Prefix Caching）？

前缀缓存是一种大语言模型推理优化技术，其核心思想是缓存历史对话中的 KV Cache（Key-Value 缓存），以便后续请求能够直接重用这些中间计算结果。这种机制旨在显著降低首令牌延迟（Time to First Token, TTFT），并提升整体推理效率。在多轮对话、长文档问答等高前缀复用场景中，前缀缓存的作用尤为突出。

从系统设计的角度看，前缀缓存将推理问题转化为计算与检索之间的平衡问题。在传统的 GQA（Grouped-Query Attention）模型推理流程中，预填充（Prefill）阶段对输入序列进行 **$O(L^2)$** 复杂度的注意力计算（其中 **$L$** 为序列长度）。若能通过某种检索机制在 **$O(L)$** 甚至更低的时间复杂度内找回已有的 KV 状态，则推理系统便可跳过昂贵的预填充计算。如果假设存在一个无限大的 KV 存储空间，每次推理请求只需进行检索与 KV 搬运，剩余工作全部由解码（Decode）阶段完成，这将彻底改变推理的成本结构。

> **P.S.** 网上主要讨论 Prefix Caching 对 Prefill 性能的影响，而不会减少 Decode 阶段的时间。然而，在混合部署场景中，Prefill 操作通常会打断 Decode 过程，特别是在整图模式下，某一个batch中pd请求会鱼龙混杂，因此理论上 Prefix Caching 也会直接提升TPOT；在PD分离场景中，需要传输的kvcache减少了，也能在一定程度上优化 Decode 的吞吐量。

---

### 1.2 Prefix Caching的简易设计方案

在深入 vLLM 的代码实现之前，我们不妨先忽略其复杂的工程细节，思考一个基础的前缀缓存方案。假设每个字母代表一个 Token，那么以下两个序列分别对应不同的 Token 列表：

- **Tokens 1**: ABCDE → [KV1, KV2, KV3, KV4, KV5]
- **Tokens 2**: ABCDF → [KV1, KV2, KV3, KV4, KV6]

根据自回归模型的训练原理，在推理 Token F 时，可以复用 [KV1, KV2, KV3, KV4] 这些已缓存的前缀 KV 状态。因此，我们支持Prefix Caching首先需要设计一个支持存储和检索的数据结构，如下所示：

```python
class KVCacheStore:
    def store(tokens, kv_cache_tensors):
        return
    def retrieve(tokens):
        return kv_cache_tensors

kv_cache_store = KVCacheStore()
kv_cache_store.store("ABCDE", [KV1, KV2, KV3, KV4, KV5])
print(kv_cache_store.retrieve("ABCDE"))
# 输出: [KV1, KV2, KV3, KV4]
```

从算法设计的角度看，很容易想到使用前缀树（Trie）来实现前缀匹配，并可进一步结合基数树（Radix Tree）进行优化（这也是 SGLang 的核心实现）。以下 LeetCode 题目可以帮助理解前缀树的基本操作：

- [实现 Trie（前缀树）](https://leetcode.cn/problems/implement-trie-prefix-tree/description/)
- [最长公共前缀的长度](https://leetcode.cn/problems/find-the-length-of-the-longest-common-prefix/description/)

泛化到大批量请求前缀复用的场景，即系统中有 100 个字符串，问题转化为新字符串与系统中所有字符串的最大重叠部分是什么？

我们可以维护一个基于 Token 粒度的前缀树，可以直接进行检索匹配。然而，过细的匹配粒度会导致维护成本激增，因此可以引入分块（Chunking）机制。每个块固定包含若干 Token，只记录块的缓存状态，减少维护和检索成本，这与 PagedAttention 的 Block 机制完美契合，这就是 vLLM 的原始思路。与此同时，vLLM 选择了更为简单高效的哈希映射方案，系统将前缀串联起来计算哈希值，每个哈希值对应不同长度的物理块集合，通过 **$O(1)$** 的复杂度获取对应的 KV 块。这种设计避免了 Trie 树深度遍历的开销，而且在固定块大小（如 16 Token）的背景下还达到最为均衡的性能。

```python
prefix_hash = ""
chunked_prefix_kv_maps = {}
for chunk in chunked_tokens:
    chunk_hash = hash(prefix_hash + chunk)
    prefix_hash = chunk_hash

# 存储操作
for chunk_hash, chunk_kv in zip(chunk_hashes, chunk_kvs):
    hash_store.put(chunk_hash, chunk_kv)

# 检索操作
for chunk_hash in chunk_hashes:
    chunk_kv = hash_store.get(chunk_hash)
    if chunk_kv is None:
        break
```

综上，`hash_store` 的代码设计是关键，需要方便存储与删除。vLLM 中直接使用双向链表来存储这些 Block，每个 Block 对象被封装成一个节点，节点中包含了前缀哈希标识，让我们快速进入代码走读环节。

---

### 1.3 vLLM 代码走读

我们先从基本的设计方案入手，暂不考虑多模态 `mm_cache`、`lora_extra_hash` 等衍生算法。Block 本质上是一个页表，用于存放 `block_size` 个 Token 的 KV Cache。以下代码分析基于 vLLM v0.16.0 版本，请注意，vLLM 迭代速度很快，建议读者结合实际代码仓库进行实时更新（Gemini的Deep Research你值得拥有，学吧，学无止境，太深了）。本文仅作抛砖引玉之用。

#### 1.3.1 关键数据结构

##### 1.3.1.1 class BlockPool

[Block Pool](https://github.com/vllm-project/vllm/blob/89a77b10846fd96273cce78d86d2556ea582d26e/vllm/v1/core/block_pool.py#L128) 负责管理所有 KV Cache Block，提供分配、释放和缓存 Block 的方法。它包含以下核心组件：

- 所有 `KVCacheBlock` 的列表
- 用于管理空闲块的 `FreeKVCacheBlockQueue`
- 通过 `cached_block_hash_to_block`（`BlockHashToBlockMap` 类型）维护哈希值与 Block 之间的映射关系

```python
class BlockPool:
    def __init__(self, num_gpu_blocks, enable_caching, ...):
        self.blocks = [KVCacheBlock(idx) for idx in range(num_gpu_blocks)]  # 所有 Block 列表
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)  # 空闲 Block 队列
        self.cached_block_hash_to_block = BlockHashToBlockMap()  # 哈希到 Block 的映射
```

##### 1.3.1.2 class KVCacheBlock

[KVCacheBlock](https://github.com/vllm-project/vllm/blob/89a77b10846fd96273cce78d86d2556ea582d26e/vllm/v1/core/kv_cache_utils.py#L108) 记录了物理块 ID、引用计数 `ref_cnt` 以及块哈希值。块哈希是一个关键属性，它通常由“父块哈希 + 当前块 Token 元组 + 额外哈希（如 LoRA ID、多模态哈希）”共同组成。这种链式哈希确保了缓存检索的唯一性。

- **引用计数（ref_cnt）**：块被请求使用时加 1，请求结束时减 1
- **哈希缓存**：一个 Block 被打满后会分配一个哈希值，用于后续前缀快速查找复用
- **双指针**：空闲块通过 `prev_free_block`/`next_free_block` 组成链表

```python
@dataclass
class KVCacheBlock:
    """KV-cache block metadata."""

    # Block ID, ranging from 0 to num_gpu_blocks - 1.
    block_id: int
    # Reference count.
    ref_cnt: int = 0
    # The hash key (block hash + group id) of the block, only available
    # when the block is full and cached.
    _block_hash: BlockHashWithGroupId | None = None

    # Used to construct a doubly linked list for free blocks.
    # These two attributes should only be manipulated by FreeKVCacheBlockQueue.
    prev_free_block: "KVCacheBlock | None" = None
    next_free_block: "KVCacheBlock | None" = None
```

##### 1.3.1.3 class FreeKVCacheBlockQueue

[Free Block Queue](https://github.com/vllm-project/vllm/blob/2d5be1dd5ce2e44dfea53ea03ff61143da5137eb/vllm/v1/core/kv_cache_utils.py#L156) 使用双向链表结构管理空闲块。它由 `KVCacheBlock` 组成的双向链表构成，用于维护所有空闲的 KV Cache Block。队列本身仅维护 `fake_free_list_head` 和 `fake_free_list_tail` 虚拟指针（也是 `KVCacheBlock` 对象），每个 Block 通过其 `prev_free_block` 和 `next_free_block` 字段链接。其核心价值在于支持在 **$O(1)$** 时间内添加、删除或移动任意位置的 Block。
**核心方法**：
- **初始化**：将一组 `KVCacheBlock` 连接成双向链表
- **`popleft()`**：从队头取出空闲块（分配），若队头有空闲块，直接更新队列头部指针，`num_free_blocks` 减一
- **`append()`**：将块添加到队尾（回收），回收操作和引用数强相关，归零说明暂无请求占用，防止队尾，`num_free_blocks` 加一
- **`get_all_free_blocks()`**：获取所有空闲块列表，顺序遍历，收集到名为 `ret` 的列表中返回，方便运行时打印当前空闲资源情况

当一个 Block 被释放时，会按照最近最少使用（LRU）原则放回队列。在释放请求关联的多个块时，通常会按照逆序放回队列。这意味着序列末尾的块（包含更多请求特有信息）会排在队列前端，优先被驱逐；而序列开头的块（如 System Prompt）会排在后端，留在显存中的概率更高。

```python
class FreeKVCacheBlockQueue:
    def __init__(self, blocks: list[KVCacheBlock]) -> None:
        self.num_free_blocks = len(blocks)
        # Initialize doubly links of consecutive blocks
        for i in range(self.num_free_blocks):
            if i > 0:
                blocks[i].prev_free_block = blocks[i - 1]
            if i < self.num_free_blocks - 1:
                blocks[i].next_free_block = blocks[i + 1]
        # Create a fake head and a tail block for the doubly linked list to
        # reduce branching in the code
        #
        # The implementation guaranteed that the fake head and tail
        # are NEVER got popped, so we could safely assume each real blocks
        # in the queue has prev and next blocks.
        self.fake_free_list_head = KVCacheBlock(block_id=-1)
        self.fake_free_list_tail = KVCacheBlock(block_id=-1)
        if self.num_free_blocks > 0:
            # Connect fake_head and fake_tail to the first and last block
            # respectively.
            self.fake_free_list_head.next_free_block = blocks[0]
            blocks[0].prev_free_block = self.fake_free_list_head
            self.fake_free_list_tail.prev_free_block = blocks[-1]
            blocks[-1].next_free_block = self.fake_free_list_tail
        else:
            # For empty list, simply connect the fake head and tail.
            self.fake_free_list_head.next_free_block = self.fake_free_list_tail
            self.fake_free_list_tail.prev_free_block = self.fake_free_list_head
```

##### 1.3.1.4 class KVCacheManager

[KVCacheManager](https://github.com/vllm-project/vllm/blob/2d5be1dd5ce2e44dfea53ea03ff61143da5137eb/vllm/v1/core/kv_cache_manager.py#L94) 的核心是维护请求与 Block 之间的映射关系。它以请求为粒度进行管理，而 `BlockPool` 是以 Block 为粒度进行管理。最新版本还封装了 `KVCacheCoordinator`，用于协同 `BlockPool` 以及 `SingleTypeKVCacheManager` 的管理操作，核心内容详见下文关键操作部分。

```python
class KVCacheManager:
    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        hash_block_size: int,
        enable_caching: bool = True,
        use_eagle: bool = False,
        log_stats: bool = False,
        enable_kv_cache_events: bool = False,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ) -> None:
        self.max_model_len = max_model_len

        self.enable_caching = enable_caching
        self.use_eagle = use_eagle
        self.log_stats = log_stats
        self.metrics_collector = metrics_collector
        # FIXME: make prefix cache stats conditional on log_stats. We still need
        # this comment because when the log stats is enabled there are still
        # potential configs we could expose in the future.
        self.prefix_cache_stats = PrefixCacheStats() if log_stats else None

        self.coordinator = get_kv_cache_coordinator(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            use_eagle=self.use_eagle,
            enable_caching=self.enable_caching,
            enable_kv_cache_events=enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            hash_block_size=hash_block_size,
            metrics_collector=self.metrics_collector,
        )
        self.num_kv_cache_groups = len(kv_cache_config.kv_cache_groups)
        self.block_pool = self.coordinator.block_pool
        self.kv_cache_config = kv_cache_config

        # Pre-constructed KVCacheBlocks with no blocks, callers should use this
        # via create_kv_cache_blocks instead of creating new ones to avoid GC
        # overhead.
        #
        # We use nested tuples to ensure the empty KVCacheBlocks is immutable.
        self.empty_kv_cache_blocks = KVCacheBlocks(
            tuple(() for _ in range(self.num_kv_cache_groups))
        )
```

#### 1.3.2 关键操作

在阅读函数操作之前，先看一个具体示例会非常有感觉。vLLM 官方文档中提供了一个合适的示例，限制了 10 个 Block，第三个请求到来时会发生一次驱逐。

- Example原文：[vLLM Prefix Caching Example](https://docs.vllm.ai/en/stable/design/prefix_caching/#example)
- 中文解析（源神00480351知乎）：[知乎文章](https://zhuanlan.zhihu.com/p/1896927732027335111)
- 核心操作流程（全网最细节的讲解，b站，针对v0.9.0版本）：[核心篇：vLLM 键值缓存管理器](https://www.bilibili.com/video/BV1NhFjzTEKe?spm_id_from=333.788.videopod.sections&vd_source=284d4ed414e28e59ec394ebf78406f2b)

##### 1.3.2.1 分配（Allocation）

调度器为**新请求**分配 KV Cache Block 的流程如下：

1. **调用 `kv_cache_manager.get_computed_blocks()`**：
   - 根据请求的 Prompt Tokens 进行哈希，并在缓存中查找对应的 Cache Blocks，获取已计算的 Block 序列。
2. **调用 `kv_cache_manager.allocate_slots()`**：
   - 计算当前需要分配的新 Block 数量；若可用 Block 不足，则返回；
   - 从 Free Block Queue 的队头弹出 Block；如果弹出的 Block 是缓存 Block，则同时驱逐该 Block，避免其他请求再复用；
   - 将 Token ID 写入已有 Block 和新分配的 Block 中的空槽位。如果某个 Block 被填满，则将其添加到 Cache Blocks 中以进行缓存。

##### 1.3.2.2 释放（Release）

当一个请求结束时，如果其占用的 Block 没有被其他请求使用（引用计数为 0），则释放这些 Block。在以下示例中，释放了请求 1 以及其关联的 Block 2、3、4 和 8。释放的 Blocks 会按照**逆序**添加到 Free Block Queue 的尾部。这是因为请求的最后一个 Block 通常哈希了更多的 Token，更具请求特异性，不太可能被其他请求复用，因此应当优先被淘汰。

![Block Release Example](https://wiki.huawei.com/vision-file-storage/api/file/download/upload-v2/WIKI2026030210265243/39632466/c75490340e694227826d164b434d4f5a.png)

##### 1.3.2.3 驱逐（Eviction，LRU）

当 `FreeBlockQueue` 的队头 Block（即最近最少使用的 Block）仍处于缓存状态时，在使用之前，必须将其驱逐，不然会直接阻塞其他请求继续使用新的Block进行utility。（Example原文：[vLLM Prefix Caching Example]）示例中的 Block 3 就是第一个有哈希值被驱逐的 Block。

具体的驱逐过程包括以下步骤：

- 从 `FreeBlockQueue` 的队头弹出该 Block，即要被驱逐的 LRU Block；
- 从 `cached_block_hash_to_block` 中移除该 Block 的 ID；
- 从 `KVCacheBlock` 移除该 Block 对应的哈希值。

---

### 1.4 SGLang 与 vLLM 的对比（待进一步完善）

SGLang 提出了 RadixAttention 机制，它与 vLLM 的主要区别在于使用了 Radix Tree 代替简单的 Hash Map。

##### 1.4.1 概述

| **特性** | **vLLM PagedAttention** | **SGLang RadixAttention** |
|----------|--------------------------|----------------------------|
| **数据结构** | 哈希映射（Block-based） | 前缀树（Radix Tree） |
| **缓存粒度** | 固定块（通常 16 Tokens） | 变长序列（Variable-length） |
| **共享机制** | 显式哈希匹配 | 自动最长前缀探测 |
| **检索复杂度** | 每个块 $O(1)$ | $O(L)$，$L$ 为匹配长度 |

##### 1.4.2 动态对话流的适配

SGLang 的 Radix Tree 能够优雅地处理对话分支。在复杂的 Agent 工作流中，用户可能会在同一前缀后尝试不同的指令。Radix Tree 可以将公共前缀节点作为父节点共享，而将不同的指令分支作为子节点。这种结构化管理使得 SGLang 在多轮对话场景下的吞吐量通常比 vLLM 高出 10-20%。

##### 1.4.3 匹配方案对比

- **哈希匹配（vLLM 方案）**
  
  - **块级哈希**：将输入序列分块（Block），每块包含固定数量 Token（如 16 个）。块的哈希值由三要素计算：
    - `parent_block_hash`：父块哈希值（依赖前序块）。
    - `cur_block_token_ids`：当前块内 Token ID 序列。
    - `extra_keys`：多模态或 LoRA 等扩展标识。
  - **O(1) 检索**：通过全局哈希表实现快速查找，复杂度与序列长度线性相关。
  - **适用场景**：长文本匹配，追求系统易用性。
- **树形匹配（SGLang 方案，RadixTree）**
  
  - **Trie 树结构**：每个节点代表一个 Block，通过树形结构维护前缀关系。
  - **O(L) 检索**：需遍历树深，但通过哈希优化子节点查找。
  - **适用场景**：更灵活的前缀匹配，但维护成本较高。

---

## 2. 收益分析
Prefix Caching 从本质上对于客户层面最直观的感知是首 Token 输出更快了。在混合部署场景下，Decode 被打断的时间减少，也会提升一部分 Decode 的性能。然而，该特性的原理虽然相对简单，但性能分析却较为复杂。不同的配置（如业务界面是 PD 分离还是混合部署、`chunk_size` 的设置、Prefill 是否开启图模式、`block_size` 是否与竞品对齐等）都会影响整体的性能理论分析。

**因此，下文将结合近期客户遇到的主要问题以及内部的一些问题单定位，抽丝剥茧，总结关键的问题分析思路，并持续更新，欢迎讨论。**

我们先假设所有的性能测试基准都是不开 Prefix Caching 的配置，对性能分析影响最大的因素是输入长度与 `chunk_size` 的比例，在 vLLM 的配置脚本中，这对应 `max-num-batched-tokens` 参数，不同的客户场景，不同的输入请求，不同的性能需求下，chunk_size 千变万化。

- 在长序列（如 128K 输入）场景下，`chunk_size` 的经验值设置为 2048。
- 在短序列输入（2K、3.5K）场景下，`chunk_size` 的经验值往往设置为 16384。

这两种不同场景对应的性能提升方式有较大差异，**客户层面一般只关注TTFT提升了多少**。本文将分两篇衍生文章讨论这两种场景，**只考虑单前缀，多前缀的场景只是容易触发驱逐**，本质和单前缀的性能分析手段一致。关于“短序列”和“长序列”的定义，笔者认为：当输入长度的普遍小于 `chunk_size` 时，可视为常规序列；当平均输入长度是 `chunk_size` 的两倍以上时，可视为长序列。边界条件可以进行特定分析。

---

### 2.1 常规序列 Prefix Caching 性能分析

#### 2.1.1 问题定义

再次锚定**物理边界**，即什么是**常规序列（Regular Sequence）**：

* **定义**：单序列长度 **$L \le \text{max-num-batch-tokens}$**。
* **物理特性**：Prefill 阶段为原子操作（Atomic），无需 **Chunked-Prefill**（分段预填充）或 **Context Parallel**（上下文并行）跨卡通信。

同时我们定义以下参数表达：

* **$n$**：阻塞队列中的请求总数。
* **$m$**：**无缓存**时，单 Batch 能容纳的请求数（受 max-num-batch-tokens限制）。
* **$k$**：**无缓存**时的总调度轮次，**$k = n / m$**。
* **$r$**：前缀缓存命中率（**$0 \le r < 1$**）。
* **$T$**：处理单批次（Batch）推理的耗时（假设为常数）。

#### 2.1.2 场景 A：高并发满载（排队阻塞）

##### 2.1.2.1 性能收益计算

在连续调度中，prefill阶段的调度耗时往往能忽略，$n$ 个请求被调度 $k$ 次，每一个batch共 $m$ 个请求，并且每个批次请求的输入总长度一致，我们假设第 1 批请求的 TTFT 为 $T$，$T$ 为常数，那么第 2 批有一个 $T$ 的等待耗时，则TTFT为 $2T$，以此类推。

调度轮次为 $k$，**平均 TTFT** 为：

$$
\text{TTFT}_{\text{ave}} = \frac{T + 2T + \dots + kT}{k} = \frac{(k + 1)T}{2}
$$

由于缓存命中了 $r$ 比例的 Token，单请求的计算压力变为 $(1-r)$。此时单 Batch 能容纳的请求数提升至 $m / (1-r)$，对应的有缓存调度轮次 $k'$ 为：

$$
k' = k \cdot (1 - r)
$$

我们将性能收益（Gain）定义为平均 TTFT 的下降比例：

$$
\text{Gain} = 1 - \frac{\text{TTFT}_{\text{ave, cached}}}{\text{TTFT}_{\text{ave, random}}} = 1 - \frac{k' + 1}{k + 1}
$$

将 **$k' = k(1-r)$** 代入：

$$
\text{Gain} = \frac{k \cdot r}{k + 1}
$$

**推论**：当总调度轮次越大时候（**$k \to \infty$**）时，可能因为序列长度短或者命中率高，收益趋近于命中率 **$r$**。

##### 2.1.2.2 三种典型命中率对比

假设总请求 **$n=8$**，无缓存时单 Batch 容量 **$m=2$**，则初始轮次 **$k=4$**。

| **命中率 (r)** | **计算量占比 (1−r)** | **调度轮次 (k′)** | **平均 TTFT** | **性能收益 (Gain)** |
| ---------------------- | ----------------------------- | -------------------------- | --------------------- | --------------------------- |
| **0% (Random)**     | 100%                        | 4                        | **$2.5T$**        | 0%                        |
| **25%**             | 75%                         | 3                        | **$2.0T$**        | **20%**                  |
| **50%**             | 50%                         | 2                        | **$1.5T$**        | **40%**                  |
| **75%**             | 25%                         | 1                        | **$1.0T$**        | **60%**                  |

##### 2.1.2.3 真实情况

在高并发满载下，单次调度耗时 **$T$** 其实会**微涨**。Attention 算子的kv_len输入变长了，导致整体计算耗时增加。Batch 里的新 Token 越多，要 Look-back 的 KV 越长，Attention 计算开销越大。但在“常规序列”下，这部分增加通常远小于线性层节省的开销，所以性能估算是，完全可以将$T$定义为常数。

#### 2.1.3 场景 B：缓慢加压（随来随处理）

如果队列不阻塞，来 1-2 条就处理 1-2 条，模拟真实的业务场景，此时拼的是**单次 Prefill 的绝对延迟**。

##### 2.1.3.1性能收益计算

一切回归本质，收益取决于算子在计算量减少后，耗时能否线性下降：

$$
T(L) = T_{launch} + \alpha \cdot L_{eff} + T_{bubble} 
$$

* **$T_{launch}$**：**启动开销**（Kernel Launch, 显存延迟等固化成本）。
* **$\alpha$**：**算子耗时增长率**（耗时随 Token 增加的敏感度，大部分都为线性增长）。
* **$L_{eff}$**：有效 Token 数，**$L \cdot (1-r)$**。
* **$T_{bubble}$** ：奇奇怪怪的空泡耗时。

**算子耗时直接减少的收益**：

$$
\text{Gain} \approx \frac{\alpha \cdot L \cdot r}{T_{launch} + \alpha \cdot L + T_{bubble}}
$$

**结论**：
- 如果算子耗时增长率很差，常说线性度不太行（一般是$T_{launch}$ 占比大，比如咱们的通信算子启动开销），即使命中率高，耗时也降不下来。
- 一般来说，只有在计算密集，**空泡比较小**，启动开销较低的场景下，Prefix-Caching 的延迟优化才比较明显
- 这种情况只能采集profiling分析，哪个算子性能不行，写个UT实锤就行，持续优化。

##### 2.1.3.2 非理想情况

#### 2.1.4 总结

| **维度** | **高并发满载（排队模式）**                | **缓慢加压（或prefill低载模式）**           |
| ---------------- | ------------------------------------------------- | ------------------------------------------ |
| **核心瓶颈**  | **调度总轮次 (k)**                             | **算子耗时增长率 ($\alpha$)**        |
| **收益本质**  | 猛算新 Token | **算子在养生**，通过减少调度的token来减少计算/通信量 |
| **表现特征**  | **排队时间极大减少**，平均TTFT下降，吞吐上限翻倍             | **单条请求的TTFT 缩短**，响应变快        |
| **关键制约**  | max-num-seqs是否够用                    | $T_{launch}$ 启动开销和空泡$T_{bubble}$是否过大，算子耗时下降线性度是否合理       |

---

### 2.2 长序列 Prefix Caching 性能分析

#### 2.2.1 问题定义

我们定义超长序列就是max-model-len远大于chunk_size，必须用Chunk Prefill或者Context Parallel的方式才能推理

先定义以下关键变量：

- $C$: 块大小（Chunk Size）
- $S$: 输入序列长度（Sequence Length）
- $N$: 序列 $S$ 分块后的总块数，$N = S / C$
- $K$: 因前缀缓存而可跳过的块数，满足 $1 \leq K \leq N$
- $T_{\mathrm{att}}^{(i)}$: 计算第 $i$ 个块的注意力机制耗时（变量）
- $T_{\mathrm{other}}$: 处理单个块所需的其他操作耗时（常数）

#### 2.2.2 耗时模型

##### 2.2.2.1 原始总耗时（不开prefix-caching）

处理整个序列的总耗时为各块耗时之和：

$$
T_{\mathrm{total}}^{\mathrm{(orig)}} = \sum_{i=1}^{N} T_{\mathrm{chunk}}^{(i)} = \sum_{i=1}^{N} \left( T_{\mathrm{att}}^{(i)} + T_{\mathrm{other}} \right) = \sum_{i=1}^{N} T_{\mathrm{att}}^{(i)} + N \cdot T_{\mathrm{other}}
$$

##### 2.2.2.2 有部分chunk命中后的总耗时

启用prefix-caching后，若跳过前 $K$ 个块，则计算从第 $K+1$ 个块开始：

$$
T_{\mathrm{total}}^{\mathrm{(cached)}} = \sum_{i=K+1}^{N} \left( T_{\mathrm{att}}^{(i)} + T_{\mathrm{other}} \right) = \sum_{i=K+1}^{N} T_{\mathrm{att}}^{(i)} + (N - K) \cdot T_{\mathrm{other}}
$$

#### 2.2.3 TTFT提升比分析

##### 2.2.3.1 精确公式

TTFT的节省时间为原始总耗时与缓存后总耗时之差，提升比例为节省时间与原始总耗时之比：

$$
\begin{aligned}
T_{\mathrm{prefix}} &= T_{\mathrm{total}}^{\mathrm{(orig)}} - T_{\mathrm{total}}^{\mathrm{(none-cached)}} \\
&= \left[ \sum_{i=1}^{N} T_{\mathrm{att}}^{(i)} + N \cdot T_{\mathrm{other}} \right] - \left[ \sum_{i=K+1}^{N} T_{\mathrm{att}}^{(i)} + (N - K) \cdot T_{\mathrm{other}} \right] \\
&= \sum_{i=1}^{K} T_{\mathrm{att}}^{(i)} + K \cdot T_{\mathrm{other}}
\end{aligned}
$$

因此，TTFT提升比例的通用精确公式如下，适用于任意 $T_{\mathrm{att}}^{(i)}$ 分布，作为分析基准：

$$
R_{\mathrm{prefix}} = \frac{T_{\mathrm{prefix}}}{T_{\mathrm{total}}^{\mathrm{(orig)}}} = \frac{\sum_{i=1}^{K} T_{\mathrm{att}}^{(i)} + K \cdot T_{\mathrm{other}}}{\sum_{i=1}^{N} T_{\mathrm{att}}^{(i)} + N \cdot T_{\mathrm{other}}}
$$

##### 2.2.3.2 实际案例分析

以Qwen3-32B-int8基础稠密模型为例，配置为64K的长输入、2048的块大小、TP4部署，并启用所有优化特性。我们先看看$T_{\mathrm{other}}$ 是不是固定的，随机采两个chunk的耗时：

![image](https://wiki.huawei.com/vision-file-storage/api/file/download/upload-v2/WIKI2026030210265243/39632544/fa144d0cbb2d4738a68436171602486b.png)

![image](https://wiki.huawei.com/vision-file-storage/api/file/download/upload-v2/WIKI2026030210265243/39632542/162e1682fa8340d5a8bcd1ad308c780c.png)

- 第一个块：$T_{\mathrm{other}} = 262.9 - 81.82 = 181.08$ ms
- 第二个块：$T_{\mathrm{other}} = 868.9 - 688.89 = 180.01$ ms

因此，$T_{\mathrm{other}}$ 可近似为常数180 ms。

进一步，我们将每一个chunk的耗时统计到一个折线图中，如下图所示，结果显示每一个chunk的耗时随着chunk数量的变多，也就是kv_len的逐渐增加接近线性增长（具体线性度与芯片算力及算子实现强相关）。假设64K的输入命中了50%的前缀，即缓存命中块数 $K=16$，则计算收益比：

$$
R_{\mathrm{prefix}} = \frac{2534.80 + 180 \times 16}{9416.07 + 180 \times 32} = \frac{5414}{15176} \approx 35\%
$$

![image](https://wiki.huawei.com/vision-file-storage/api/file/download/upload-v2/WIKI2026030210265243/39632545/b5f0f6f9a871479790fa973b1d11f586.png)

#### 2.2.4 近似分析

在单Batch长序列输入场景下，TTFT提升比例取决于被跳过的前 $K$ 个块的计算耗时占总耗时的比例。实践中，$T_{\mathrm{att}}^{(i)}$ 随 $i$ 增大而增加（因为kv_len在持续增长，q_len不变），常近似为线性增长：$T_{\mathrm{att}}^{(i)} = T_1 + (i-1)\Delta$，其中 $T_1 \approx T_{\mathrm{att}}^1+T_{\mathrm{other}}$ ，即第一个Chunk的计算耗时，$\Delta$ 为每块递增开销。归一化后，设 $T(x) = T_1 + m x$，其中 $x \in [0,1]$，$r = K/N$ 为命中率。

- 总计算时间：

$$
T_{\mathrm{total}} = \int_0^1 (T_1 + m x) dx = T_1 + \frac{m}{2}
$$

- 前缀部分计算时间：

$$
T_{\mathrm{prefix}} = \int_0^r (T_1 + m x)  dx = r T_1 + \frac{m}{2} r^2
$$

- TTFT提升比例：

$$
R = \frac{T_{\mathrm{prefix}}}{T_{\mathrm{total}}} = \frac{r T_1 + \frac{m}{2} r^2}{T_1 + \frac{m}{2}}
$$

##### 2.2.4.1 增长耗时占主导（m >> T1）

当注意力耗时远大于初始值时，即 $m \gg T_1$，提升比例近似为：

$$
R_{\mathrm{prefix}} \approx \frac{\frac{m}{2} r^2}{\frac{m}{2}} = r^2
$$

例如，命中率 $r = 0.5$ 时，$R_{\mathrm{prefix}} \approx (0.5)^2 = 0.25$。

##### 2.2.4.2 固定耗时占主导（m << T1）

当其他操作耗时主导时，提升比例简化为：

$$
R_{\mathrm{prefix}} \approx r = K/N
$$

#### 2.2.5 算子特例讨论

前述分析主要基于单请求线性拟合的理想模型。但在实际生产环境中，受**多 Batch 并发、序列长度不均**以及​**算子耗时随序列长度非线性增长**​（如 FIA 算子）等等复杂因素影响，性能收益往往难以准确预测。因此，需结合具体业务场景进行深入剖析，并对 Prefix Caching 的性能收益范围进行修正。

近期实践发现，部分算子在特定输入规模下的实际耗时与线性预估存在偏差，必须依赖Profiling进行精确校准。例如，在某客户案例中，当 KV 长度超过 64K 时（基于 800I-A3 芯片、560T 算力及特定 TP 切分配置），FIA 算子的耗时呈现明显的​**非线性增长趋势**​。当输入形状触及特定阈值，20 核计算单元达到算力瓶颈，导致算子耗时激增。在这种情况下，由于前 **$K$** 个分块（Chunk）在整体推理耗时中的占比相对下降，Prefix Caching 带来的加速收益也随之减少。

##### 2.2.5.1 整网 Profiling 分析

如下图所示，客户场景采集Profiling 数据发现，不同的硬件单个 Chunk 的耗时增长曲线存在显著差异：
* **非线性增长：** 在 800I-A3-560T 平台上，当 KV 长度在 64K（即 32 个 Chunk）以内时，耗时基本维持线性增长；而一旦进入 64K-80K 区间，耗时增长速率明显加快，线性斜率陡增。
* **线性增长：** 相比之下，H20 和 800T-A3-752T 平台的耗时曲线表现得更为平滑（“丝滑”），更符合线性预期。‸
![image](https://wiki.huawei.com/vision-file-storage/api/file/download/upload-v2/WIKI2026030210265243/39771684/44a9c7e727a347eca84cede8eed9cec3.png)



##### 2.2.5.2 算子 UT 性能测试

根据2.2.2性能评估公式，假设 **$T_{\mathrm{other}}$**（非计算部分耗时）保持固定，则整体性能波动的核心变量在于 ​**FIA 算子的耗时增长差异**​。下图展示了 FIA 算子在 UT测试环境下，耗时随序列长度变化的实际测试结果：
![image](https://wiki.huawei.com/vision-file-storage/api/file/download/upload-v2/WIKI2026030210265243/39632543/f21aaa88798643d5a8ed8c450213ca8a.png)

**结论：** 在评估 Prefix Caching 的最终收益时，不能简单进行线性外推。必须充分考量底层算子在不同硬件上的实际性能曲线，并结合 Profiling 数据进行针对性建模，以确保预估收益与实际表现的一致性。

#### 2.2.6 叠加PCP影响（待更新）

（注：本节内容待补充，暂时可根据以上分析方案对实际场景进行叠加pcp特性影响分析。）

---

## 3. 昇腾使用场景注意事项

- **Block Size 限制**：目前 vLLM-Ascend 版本由于底层算子原因，建议固定 `block_size` 为 128。相比 GPU 默认的 16，这意味着如果平均输入很短（如 1K 以内），末尾不满 128 Token 的块无法形成有效哈希，导致缓存命中率大幅下降。（支持 `block_size` 为 32 的算子正在灰度上库）
- **显存压力**：昇腾 A3 代际芯片在显存容量上相较于 H20 等竞品可能存在差异。由于 vLLM 原生 Prefix Caching 不支持多 DP（数据并行）间共享缓存，为了减少驱逐（Eviction），建议在显存紧张时减少 DP 数量，或直接采用池化方案，详见[MY节点的经验](https://jx.huawei.com/community/comgroup/postsDetails?postId=737a36b381534acbafb595c2a4a5ac60&noTop=true&type=freePost&previou=comments&welink_open_uri=aDU6Ly80NzE2NTE3MzE0Nzc5NTcvaHRtbC9pbmRleC5odG1sIy9qeC9kZXRhaWw/aWQ9NzM3YTM2YjM4MTUzNGFjYmFmYjU5NWMyYTRhNWFjNjAmdHlwZT1mcmVlX3Bvc3QmdXJsPQ==)、[多DP场景下池化](https://jx.huawei.com/community/comgroup/postsDetails?postId=ba0ce11fecd44a35a0e4f784172cf544&noTop=true&type=freePost&welink_open_uri=aDU6Ly80NzE2NTE3MzE0Nzc5NTcvaHRtbC9pbmRleC5odG1sIy9qeC9kZXRhaWw/aWQ9YmEwY2UxMWZlY2Q0NGEzNWEwZTRmNzg0MTcyY2Y1NDQmdHlwZT1mcmVlX3Bvc3QmdXJsPQ==)。
- **对比策略**：在给客户演示时，如果对方芯片算力较弱，有时关闭 Prefix Caching、直接发挥昇腾的暴力计算能力反而能取得更好的 Prefill 表现。

## 4. 总结

前缀缓存作为大语言模型推理加速的关键技术，通过空间换时间的方式显著提升了系统性能。本文系统性地介绍了前缀缓存的原理、vLLM 的实现细节、SGLang 的替代方案，以及在昇腾平台上的实践建议。**分级池化、语义 Caching、多实例路由、压缩 Cache** 等多种新场景的能力正在细致探究中，待后续发文。随着 Agent 时代的到来，Prefix Caching 将成为构建具有长期记忆的智能系统的核心竞争力。

（欢迎讨论交流 h00889645，共同进步，壮大 vLLM-Ascend 以及 vLLM-Omni 社区）

## 参考文献
[AI INFRA 学习 03 - Prefix Caching 原理详解](https://www.bilibili.com/video/BV1jgTRzSEjS/)
[[EP05] vllm从开源到部署，Prefix Caching和开源答疑](https://www.bilibili.com/video/BV1ZiVJziE6o/)
[RadixTree-SGLang RadixAttention基础](https://zhuanlan.zhihu.com/p/1917894424488293447)
[手撕SGLang KV Cache核心逻辑：快速理解](RadixAttentionhttps://zhuanlan.zhihu.com/p/1994495318197305400)
[vLLM的prefix cache为何零开销](https://zhuanlan.zhihu.com/p/1896927732027335111)
[[Prefill优化][万字]🔥原理&图解vLLM Automatic Prefix Cache(RadixAttention): 首Token时延优化](https://zhuanlan.zhihu.com/p/693556044)