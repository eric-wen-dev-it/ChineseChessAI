# 传统象棋引擎设计

本文档定义一个与现有 AlphaZero/MCTS 管线并行的传统象棋引擎。目标不是替换当前神经网络训练，而是先做一个可控、可调试、可参与对弈和训练数据生成的搜索型棋软核心。

## 目标

第一阶段目标：

- 复用现有 `Board`、`Move`、`MoveGenerator` 和 `ChineseChessRuleEngine`。
- 实现固定时间或固定深度的传统搜索。
- 提供稳定的 `BestMove`、分数、主变和搜索统计。
- 能作为 WPF 对弈模式、联赛对手、训练辅助对手接入。

非目标：

- 第一版不做完整 UCCI 协议引擎。
- 第一版不引入残局库。
- 第一版不修改现有 MCTS 逻辑。

## 模块结构

建议新增目录：

```text
Traditional/
  TraditionalEngine.cs
  TraditionalSearch.cs
  TraditionalEvaluator.cs
  TraditionalMoveOrdering.cs
  TranspositionTable.cs
  OpeningBook.cs
  TraditionalEngineOptions.cs
  SearchResult.cs
  SearchLimits.cs
```

### `TraditionalEngine`

对外入口。UI、联赛或训练模块只依赖这一层。

职责：

- 接收 `Board` 和搜索限制。
- 优先查询开局库。
- 调用搜索器。
- 返回最佳走法和分析信息。

建议接口：

```csharp
public sealed class TraditionalEngine
{
    public SearchResult Search(Board board, SearchLimits limits, CancellationToken cancellationToken = default);
}
```

### `TraditionalSearch`

搜索核心。第一版使用 Negamax Alpha-Beta，之后升级到 PVS。

第一版：

- Iterative deepening。
- Alpha-Beta Negamax。
- Quiescence search 只扩展吃子、将军、应将。
- Mate score。
- 超时检查。

第二版：

- PVS。
- Aspiration window。
- Null move pruning。
- Late move reduction。
- Check extension。
- Internal iterative deepening。

### `TraditionalEvaluator`

静态局面评估。返回值统一为“当前行棋方视角”，正分表示当前行棋方好。

第一版评估项：

- 子力价值。
- 子力位置分。
- 兵过河奖励。
- 车、炮、马机动性。
- 将安全。
- 被将军惩罚。
- 简单威胁：被攻击的高价值子。

基础子力分建议：

```text
帅/将: 0
车: 900
马: 400
炮: 450
仕/士: 200
相/象: 200
兵/卒: 100
过河兵/卒: 160-220
```

当前 `Board` 已维护 `RedMaterial` 和 `BlackMaterial`，但传统评估建议使用整数分重新计算，避免浮点和训练侧权重绑定。

### `TraditionalMoveOrdering`

走法排序决定 Alpha-Beta 效率。

第一版排序优先级：

1. 开局库命中走法。
2. 置换表最佳走法。
3. 将军。
4. 吃子，按 MVV-LVA 排序。
5. Killer moves。
6. History heuristic。
7. 普通走法。

大师样本可以在这里发挥作用：

- 某局面历史高频大师走法加分。
- 未命中完整开局库时，使用局面前缀统计辅助排序。

### `TranspositionTable`

使用 `Board.CurrentHash` 作为 key。

条目字段：

```csharp
public readonly struct TTEntry
{
    public ulong Hash { get; init; }
    public int Depth { get; init; }
    public int Score { get; init; }
    public Move BestMove { get; init; }
    public TTBound Bound { get; init; } // Exact, Lower, Upper
    public int Age { get; init; }
}
```

第一版容量可固定，例如 1M 或 4M 条。替换策略使用“深度优先，同深度新条目覆盖旧条目”。

### `OpeningBook`

从大师棋谱生成局面哈希到候选走法列表。

生成流程：

1. 读取 `data/master_data` 或 PGN。
2. 从初始局面按棋谱逐步 `Push`。
3. 对前 `N` 回合记录 `CurrentHash -> Move -> Count`。
4. 保存为轻量 JSON 或二进制文件。

运行时：

- 如果当前局面命中开局库，按权重选择走法。
- 可以设置 `BookMode`：
  - `Best`: 总是选最高频。
  - `Weighted`: 按频率随机。
  - `Off`: 禁用。

第一版建议只统计前 24 ply，也就是双方各 12 手，避免中后盘样本稀疏。

## 搜索流程

```text
TraditionalEngine.Search
  -> OpeningBook.TryGetMove
     -> 命中且未禁用：直接返回
  -> TraditionalSearch.IterativeDeepening
     -> depth = 1..maxDepth
        -> GenerateLegalMoves
        -> OrderMoves
        -> Negamax
        -> Quiescence
     -> 返回最后一个完整深度结果
```

搜索必须使用现有 `Board.Push(from, to)` 和 `Board.Pop()` 回退，避免频繁 `Clone()`。

## 分数约定

- 普通评估单位为 centipawn-like integer。
- `+100` 约等于一个兵。
- `MateScore = 1_000_000`。
- `MateScore - ply` 表示越快将死越好。
- `-MateScore + ply` 表示越晚被将死越好。

`SearchResult.Score` 必须是根节点当前行棋方视角。

## 与现有项目的接入点

### UI 对弈

`ChineseChessAI.Play` 当前创建 `MCTSEngine`。可以增加一个 AI 类型枚举：

```csharp
public enum AiEngineKind
{
    Mcts,
    Traditional,
    Hybrid
}
```

传统模式下调用：

```csharp
var result = _traditionalEngine.Search(board, limits, cancellationToken);
board.Push(result.BestMove.From, result.BestMove.To);
```

### 训练和联赛

传统引擎可以作为固定强度对手：

- `TraditionalDepth4`
- `TraditionalDepth6`
- `TraditionalDepth8`
- `TraditionalBookDepth6`

这样神经网络可以和稳定基准对手比赛，避免只在弱模型之间循环自博弈。

### 神经网络辅助

后续可以做混合：

- NN policy 参与 move ordering。
- NN value 替代部分静态评估。
- Alpha-Beta 搜索结果反过来生成高质量训练样本。

## 第一版实现顺序

1. 新增 `TraditionalEngineOptions`、`SearchLimits`、`SearchResult`。
2. 新增 `TraditionalEvaluator`，先做子力和简单位置分。
3. 新增 `TraditionalMoveOrdering`，先做吃子和将军优先。
4. 新增 `TraditionalSearch`，实现 iterative deepening + alpha-beta。
5. 新增 `TranspositionTable`。
6. 接入 WPF 选择项。
7. 新增开局库生成器和 `OpeningBook`。
8. 接入联赛/训练辅助对手。

## 第一版验收标准

- 初始局面固定深度 4 能稳定返回合法走法。
- 搜索过程中不会破坏 `Board`，搜索前后 `CurrentHash`、`IsRedTurn` 和棋子数组一致。
- 被将死局面能返回 mate 分。
- 超时时返回最后一个完整深度的最佳走法。
- 同一局面重复搜索结果稳定。
- 可以在 UI 中选择传统引擎与人对弈。

## 风险点

- `GenerateLegalMoves` 包含长打/长捉检测，深搜中可能有额外开销。搜索内部可默认 `skipPerpetualCheck: true`，根节点和最终落子仍使用完整规则校验。
- `Board.Push` 会维护重复局面历史。搜索回退必须严格 `try/finally Pop()`。
- 现有部分中文注释和字符串有编码损坏，不影响引擎逻辑，但后续编辑相关文件时要避免扩大编码问题。
- 第一版评估弱于成熟棋软很正常，关键是先建立搜索框架和测试基准。

## 推荐默认参数

```text
MaxDepth: 5
MoveTime: 1000 ms
QuiescenceDepth: 6
TranspositionTableSize: 1M
UseOpeningBook: true
BookMaxPly: 24
BookMode: Weighted
```

开发调试时建议先用固定深度，不用固定时间；UI 对弈再切到固定时间。
