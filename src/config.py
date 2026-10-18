"""项目集中配置：路径、月份、随机种子、会话阈值等。

src/analysis.py、src/run_*.py 与 main.ipynb 共用，避免魔法数字散落。
"""
from pathlib import Path

# ── 路径 ──
# 本文件在 src/ 下，项目根目录 = src/ 的上一层。所有路径都从根目录拼出来，
# 因此无论从哪个工作目录运行，读写位置都一致。
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data'
RESULTS_DIR = ROOT / 'results'

PANEL_FILE = DATA_DIR / 'panel_7months.parquet'     # 按用户抽样的 7 个月面板
MONTH_CACHE_DIR = DATA_DIR / '_month_cache'         # 月度表缓存（notebook 用，可删可重建）

OUTPUT_TRACKING = RESULTS_DIR / 'tracked_users_list_Nov.csv'
OUTPUT_ROLLING = RESULTS_DIR / 'rolling_validation_results.csv'
OUTPUT_COHORT_LABELS = RESULTS_DIR / 'cohort_frozen_labels.csv'   # 冻结阈值重算的标签占比（标签保持/转化）

# ── 时间窗口 ──
SESSION_GAP_SECONDS = 1800  # 会话内相邻事件间隔上限（30 分钟）

# ── 多个月用户面板（滚动时间外验证 / 队列迁移分析用）──
MONTHS = ['2019-10', '2019-11', '2019-12', '2020-01', '2020-02', '2020-03', '2020-04']
BASE_MONTH = '2019-10'    # 队列迁移分析的固定基期

# ── 数据质量 ──
DEDUPE_EVENTS = True      # 构建特征前是否去除完全重复事件

# ── 随机性与模型 ──
RANDOM_STATE = 42         # 全局随机种子（GMM / LR）
LR_MAX_ITER = 2000        # LR 最大迭代次数

# ── 触达名单：优先触达比例 ──
# 名单的"打谁"由调用方决定（本项目当前只对"高潜力转化用户"打分与排序；
# 后续要对其他标签打分时，在 notebook 里再调用一次 LR_predict_rank 即可）；
# 这里只配置"名单内按购买概率排序后取前多少比例"用于标记优先触达。
POOL_TOP_RATIO = 0.5      # 名单内按购买概率排序后取前多少比例（k = ceil(名单人数 × 比例)，默认 50%）
