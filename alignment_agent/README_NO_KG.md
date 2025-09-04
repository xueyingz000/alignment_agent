# IFC Semantic Agent - No Knowledge Graph Version

这是IFC语义代理的简化版本，**不依赖知识图谱(Knowledge Graph)**，专注于使用LLM和语义对齐技术进行IFC-法规语义理解和对齐。

## 🎯 主要特点

### ✅ 优势
- **无外部依赖**: 不需要图数据库或知识图谱基础设施
- **快速启动**: 初始化速度更快，适合快速测试和开发
- **轻量级**: 内存占用更少，适合资源受限环境
- **独立运行**: 可以作为独立模块使用，不依赖复杂的图谱系统
- **直接语义对齐**: 使用LLM和规则方法进行直接的语义匹配

### ⚠️ 限制
- **无历史知识**: 不能利用预构建的知识图谱进行推理
- **有限的上下文**: 无法进行复杂的图谱遍历和路径查找
- **实时处理**: 每次查询都需要重新处理，无法复用历史分析结果

## 🚀 快速开始

### 1. 基本使用

```python
from core.ifc_semantic_agent_no_kg import IFCSemanticAgentNoKG

# 初始化代理
agent = IFCSemanticAgentNoKG()

# 处理简单查询
query = "What are the key relationships between IFC walls and building regulations?"
response = agent.process_query(query)

print(f"Answer: {response.final_answer}")
print(f"Confidence: {response.confidence_score}")
```

### 2. 使用IFC数据

```python
# 准备IFC数据
ifc_data = {
    'entities': {
        'wall_001': {
            'type': 'IfcWall',
            'attributes': {'Name': 'ExteriorWall', 'Description': 'Load bearing wall'},
            'properties': {'Width': 200, 'Height': 3000, 'Material': 'Concrete'}
        }
    },
    'relationships': []
}

# 分析IFC数据
response = agent.process_query(
    query="Analyze this IFC wall for compliance issues",
    ifc_data=ifc_data
)
```

### 3. 使用法规文本

```python
# 准备法规文本
regulatory_text = """
Building Code Section 3.2: Wall Requirements
All load-bearing walls shall have a minimum thickness of 150mm.
Exterior walls must provide adequate thermal insulation.
"""

# 分析法规要求
response = agent.process_query(
    query="Extract key requirements from building code",
    regulatory_text=regulatory_text
)
```

### 4. 综合语义对齐

```python
# 同时使用IFC数据和法规文本进行对齐分析
response = agent.process_query(
    query="Check IFC wall compliance with building code requirements",
    ifc_data=ifc_data,
    regulatory_text=regulatory_text
)

print(f"Semantic mappings found: {len(response.semantic_mappings)}")
print(f"Knowledge sources: {response.knowledge_sources}")
```

## 🔧 可用功能

### 支持的操作类型

| 操作类型 | 描述 | 输入要求 |
|---------|------|----------|
| `ANALYZE_IFC` | 分析IFC数据并提取实体 | IFC数据字典 |
| `PROCESS_REGULATORY_TEXT` | 处理法规文本并提取实体 | 法规文本字符串 |
| `EXTRACT_SEMANTICS` | 从查询中提取语义信息 | 查询字符串 |
| `ALIGN_ENTITIES` | 执行实体间的语义对齐 | 已提取的实体列表 |
| `GENERATE_MAPPING` | 生成最终的语义映射 | 对齐结果 |
| `VALIDATE_ALIGNMENT` | 验证对齐结果的质量 | 对齐结果 |
| `REFLECT_ON_RESULTS` | 反思结果并提出改进建议 | 当前分析状态 |

### 移除的功能

以下功能在无知识图谱版本中**不可用**：
- ❌ `QUERY_KNOWLEDGE_GRAPH` - 查询知识图谱
- ❌ `RESOLVE_ENTITY` - 实体消歧和解析
- ❌ 图谱遍历和路径查找
- ❌ RAG (检索增强生成) 系统
- ❌ 历史知识复用

## 📊 响应结构

```python
@dataclass
class AgentResponse:
    query: str                          # 原始查询
    final_answer: str                   # 最终答案
    confidence_score: float             # 置信度分数 (0-1)
    react_steps: List[ReActStep]        # ReAct推理步骤
    total_steps: int                    # 总步骤数
    execution_time: float               # 执行时间(秒)
    knowledge_sources: List[str]        # 使用的知识源
    semantic_mappings: List[Dict]       # 语义映射结果
```

## 🔍 ReAct推理框架

该版本仍然使用ReAct (Reasoning + Acting) 框架：

1. **Think (思考)**: 分析当前状态，确定推理类型
2. **Act (行动)**: 选择并执行适当的操作
3. **Observe (观察)**: 收集操作结果并评估
4. **Reflect (反思)**: 根据结果调整策略

### 推理类型
- `analysis`: 初始分析阶段
- `planning`: 规划后续操作
- `reflection`: 反思当前结果
- `conclusion`: 生成最终结论

## 📝 使用示例

运行完整的使用示例：

```bash
# 运行使用示例
python example_no_kg_usage.py

# 运行测试
python test_no_kg_agent.py
```

## ⚙️ 配置选项

在 `config.yaml` 中可以配置以下参数：

```yaml
react_agent:
  max_iterations: 10              # 最大推理迭代次数
  reflection_threshold: 0.3       # 反思触发阈值
  confidence_threshold: 0.7       # 置信度停止阈值

llm:
  provider: "openai"              # LLM提供商
  model: "gpt-4"                  # 使用的模型
  # ... 其他LLM配置

semantic_alignment:
  similarity_threshold: 0.6       # 语义相似度阈值
  alignment_methods: ["lexical", "semantic", "contextual"]
```

## 🔄 与完整版本的对比

| 特性 | 完整版本 (带KG) | 无KG版本 |
|------|----------------|----------|
| 知识图谱 | ✅ 支持 | ❌ 不支持 |
| RAG系统 | ✅ 支持 | ❌ 不支持 |
| 实体解析 | ✅ 图谱辅助 | ⚠️ 基础解析 |
| 历史知识 | ✅ 可复用 | ❌ 无历史 |
| 初始化速度 | ⚠️ 较慢 | ✅ 快速 |
| 内存占用 | ⚠️ 较高 | ✅ 较低 |
| 部署复杂度 | ⚠️ 复杂 | ✅ 简单 |
| 语义对齐 | ✅ 增强 | ✅ 基础 |
| ReAct框架 | ✅ 支持 | ✅ 支持 |

## 🎯 适用场景

### ✅ 推荐使用场景
- **快速原型开发**: 需要快速验证语义对齐概念
- **资源受限环境**: 内存或计算资源有限
- **独立应用**: 不需要复杂的知识图谱基础设施
- **教学演示**: 用于理解ReAct框架和语义对齐原理
- **测试开发**: 在开发过程中进行快速测试

### ⚠️ 不推荐场景
- **生产环境**: 需要高精度和复杂推理的生产系统
- **大规模数据**: 需要处理大量历史数据和复杂关系
- **复杂查询**: 需要多步推理和图谱遍历的复杂查询
- **知识积累**: 需要积累和复用领域知识的应用

## 🚀 迁移到完整版本

当需要更强大的功能时，可以轻松迁移到完整版本：

```python
# 从无KG版本
from core.ifc_semantic_agent_no_kg import IFCSemanticAgentNoKG
agent = IFCSemanticAgentNoKG()

# 迁移到完整版本
from core.ifc_semantic_agent import IFCSemanticAgent
agent = IFCSemanticAgent()  # 自动包含知识图谱功能

# API接口保持一致
response = agent.process_query(query, ifc_data, regulatory_text)
```

## 📚 相关文档

- [主项目README](README.md) - 完整版本的文档
- [API文档](docs/api.md) - 详细的API说明
- [配置指南](docs/configuration.md) - 配置选项说明

---

**注意**: 这是IFC语义代理的简化版本，专为快速开发和测试而设计。对于生产环境和复杂应用，建议使用包含知识图谱的完整版本。