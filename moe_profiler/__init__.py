"""MoE Runtime Profiler — 零侵入性能探测系统。

提供：
- Profiler: 主类，通过 wrap(engine) 注入探测代理
- ProfileReport: 结构化探测报告，支持终端摘要和 JSON 导出

用法：
    from moe_profiler import Profiler

    engine = MoeRuntimeEngine(config)
    profiler = Profiler()
    profiler.wrap(engine)          # 唯一新增的一行
    engine.build_components()
    engine.load_model()
    engine.generate(...)
    report = profiler.report()
    report.print_summary()
    report.save_json("profile.json")
"""

from .profiler import Profiler
from .reporter import ProfileReport

__all__ = ["Profiler", "ProfileReport"]
