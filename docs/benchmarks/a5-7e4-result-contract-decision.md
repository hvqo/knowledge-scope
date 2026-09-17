# A5.7e4：ResultContract 实验决策记录

## 决策

ChatBI 生产 NL2SQL 保留 SQL-only 基线：

- prompt：`a5.3-v3`；
- NL2SQL 初次生成：`reasoning="disabled"`，`max_tokens=1024`；
- NL2SQL 有界 repair：`reasoning="disabled"`，`max_tokens=1024`；
- 结果分析：`reasoning="disabled"`，`max_tokens=1024`；
- 默认不要求 `ResultContract`，生产链路仍是 SQL 生成 → A5.3 校验 → A5.4 只读执行。

这是基于 DEV 实验的配置决定，不是 TEST 性能结论。

## 实验证据

| 运行 | 配置 | 初次可解析 | 初次截断 | 终态 generation failure | Execution Accuracy | output tokens | total P95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Run #6 | disabled + 1024 | 50/50 | 0 | 0 | 21/47 = 44.68% | 6,104 | 2,807.7 ms |
| Run #7 | low + 1024 | 38/50 | 12 | 5 | 21/47 = 44.68% | — | — |
| Run #8 | high + 1024 | 39/50 | 10 | 5 | 19/47 = 40.43% | — | — |
| Run #9 | high + 2048 | 46/50 | 3 | 1 | 21/47 = 44.68% | 33,384 | 13,332.6 ms |

选择 `disabled + 1024` 的依据是：DEV 中初次生成可解析率为 100%，没有观察到初次截断或
终态 generation failure；它与 `high + 2048` 的 DEV Execution Accuracy 相同，但 token 和
延迟开销更低。`low + 1024`、`high + 1024`、`high + 2048` 不作为普通默认配置。

以上数字只描述 DEV 观察，不代表冻结 TEST 性能或 provider/model 质量。

## ResultContract 的保留边界

`a5.3-v4` 的 ResultContract + SQL 模型、prompt、解析器、语义校验、SQL 一致性校验和诊断
工具继续保留，但只能作为显式评测/诊断能力。A5.7e3 的 contract-only probe 也保留为历史
诊断证据；它不进入生产 NL2SQL，也不会成为正常请求的必需步骤。历史运行产物和旧格式读取
兼容性保持不变，不回写或重算旧文件。

## 未解决事项

Run #9 的 negative safe success 为 `0/3`。这是后续语义/安全优化工作，不在本决策中修改。
本记录不引入拒答策略变化，也不声明任何 TEST 结果。
