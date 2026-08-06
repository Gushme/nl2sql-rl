"""Teacher API 客户端、轨迹采集与 SFT 数据构建。"""

from nl2sql_rl.teacher.client import LLMClient, LLMClientConfig, LLMCompletion
from nl2sql_rl.teacher.collector import CollectorConfig, TeacherAttempt, collect_trajectories

__all__ = [
    "CollectorConfig",
    "LLMClient",
    "LLMClientConfig",
    "LLMCompletion",
    "TeacherAttempt",
    "collect_trajectories",
]
