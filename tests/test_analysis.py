# -*- coding: utf-8 -*-
"""核心逻辑单元测试（合成数据，不依赖原始 CSV）。

运行：在项目根目录执行  python -m unittest discover -s tests -v
"""
import os
import sys
import tempfile
import unittest

# 项目根目录入 path，保证 `from src.analysis import ...` 在任何执行方式下都能找到包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.analysis import (build_features, build_month_tables,
                          cohort_migration, Friction_and_Activity, export_tracking,
                          F_and_A_scalers, silent_buyer, gmm, load_panel,
                          rolling_validation_rate_test, rolling_validation,
                          LR_predict_rank, build_user_segment,
                          HIGH_VALUE_SEGMENTS, SILENT_SEGMENT_LABEL)


def make_events(n_users: int = 6) -> pd.DataFrame:
    """合成事件表：约 1/3 用户未购（其中部分加购未买），其余已购。"""
    rows = []
    for uid in range(n_users):
        t = pd.Timestamp('2019-10-01 10:00:00') + pd.Timedelta(minutes=uid * 5)
        session = f's{uid}'
        if uid % 3 == 0:
            # 未购用户：只浏览；uid % 6 == 0 的再加购未买（高潜力候选）
            rows.append({'user_id': uid, 'user_session': session, 'event_time': t,
                         'event_type': 'view', 'product_id': 101, 'price': 0.0})
            t += pd.Timedelta(minutes=1)
            rows.append({'user_id': uid, 'user_session': session, 'event_time': t,
                         'event_type': 'view', 'product_id': 102, 'price': 0.0})
            if uid % 6 == 0:
                t += pd.Timedelta(minutes=1)
                rows.append({'user_id': uid, 'user_session': session, 'event_time': t,
                             'event_type': 'cart', 'product_id': 101, 'price': 10.0})
        else:
            for p, etype in [(101, 'view'), (102, 'view'), (101, 'cart'), (101, 'purchase')]:
                rows.append({'user_id': uid, 'user_session': session, 'event_time': t,
                             'event_type': etype, 'product_id': p, 'price': 10.0})
                t += pd.Timedelta(minutes=1)
    return pd.DataFrame(rows)


class TestGmmIntersectionThreshold(unittest.TestCase):
    def test_constant_series_returns_median(self):
        s = pd.Series([5.0] * 60)
        self.assertEqual(gmm(s), 5.0)

    def test_small_sample_returns_median(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(gmm(s), float(np.median([1, 2, 3, 4])))

    def test_zero_inflated_guard_positive(self):
        # 大量 0 值 + 少量正值的零膨胀序列：阈值应 > 0（有加购即高摩擦）
        s = pd.Series([0.0] * 100 + [0.693, 1.099, 1.386, 1.609])
        cut = gmm(s)
        self.assertGreater(cut, 0.0)
        self.assertLessEqual(cut, min(x for x in s if x > 0))


class TestComputeEngagementMetrics(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'user_id': [1, 2, 3],
            'Pages_Viewed': [0, 100, 50],
            'Estimated_Time': [0, 5000, 3000],
            'Session_Count': [1, 5, 3],
            'Purchase_Frequency': [0, 4, 2],
            'Cart_Products': [3, 10, 6],
            'Purchased_Products': [0, 6, 2],
        })

    def test_friction_definition(self):
        out = Friction_and_Activity(self.df)
        np.testing.assert_allclose(out['Friction'], [3, 4, 4])      # max(加购-购买, 0)
        self.assertTrue(np.allclose(out['Log_Friction'], np.log1p(out['Friction'])))
        self.assertNotIn('Friction_1', out.columns)

    def test_activity_definition(self):
        # Activity = log1p 后总体标准差(z-score, ddof=0)的等权均值
        out = Friction_and_Activity(self.df)
        vals = [np.log1p(self.df[c]).to_numpy() for c in
                ['Pages_Viewed', 'Estimated_Time', 'Session_Count']]
        z = [(v - v.mean()) / v.std(ddof=0) for v in vals]
        np.testing.assert_allclose(out['Activity'], np.mean(z, axis=0), atol=1e-12)

    def test_frozen_scalers_reuse(self):
        # 冻结标准化器后，同一数据两次计算的 Activity 应完全一致（冻结路径 = 基期尺子）
        scalers = F_and_A_scalers(self.df)
        out1 = Friction_and_Activity(self.df)
        out2 = Friction_and_Activity(self.df, scalers=scalers)
        np.testing.assert_allclose(out1['Activity'], out2['Activity'], atol=1e-12)

    def test_frozen_scalers_differ_from_refit(self):
        # 构造两组分布差异大的用户：冻结（基期）标准化与当月重拟合应产生不同 Activity，
        # 证明冻结确实改变了行为，且冻结路径 = 用基期 mean/std 手工变换
        base = pd.DataFrame({
            'user_id': [1, 2, 3],
            'Pages_Viewed': [1, 2, 3],
            'Estimated_Time': [10, 20, 30],
            'Session_Count': [1, 1, 1],
            'Purchase_Frequency': [0, 0, 0],
            'Cart_Products': [0, 0, 0],
            'Purchased_Products': [0, 0, 0],
        })
        new_pop = base.copy()
        new_pop['Pages_Viewed'] = [100, 200, 300]   # 分布整体平移
        scalers = F_and_A_scalers(base)
        frozen = Friction_and_Activity(new_pop, scalers=scalers)
        refit = Friction_and_Activity(new_pop)
        self.assertFalse(np.allclose(frozen['Activity'], refit['Activity'], atol=1e-6))
        cols = ['Pages_Viewed', 'Estimated_Time', 'Session_Count']
        z = []
        for col in cols:
            v = np.log1p(new_pop[col].clip(lower=0)).to_numpy()
            z.append((v - scalers[col].mean_[0]) / scalers[col].scale_[0])
        np.testing.assert_allclose(frozen['Activity'], np.mean(z, axis=0), atol=1e-12)


class TestBuildFeatures(unittest.TestCase):
    def test_columns_and_counts(self):
        events = make_events()
        obs_end = events['event_time'].max()
        feats = build_features(events, obs_end)
        for col in ['user_id', 'Purchase_Frequency', 'Total_Spending', 'Pages_Viewed',
                    'Estimated_Time', 'Recency_Days', 'Session_Count',
                    'Cart_Products', 'Purchased_Products', 'Activity', 'Friction',
                    'Log_Friction']:
            self.assertIn(col, feats.columns)
        self.assertNotIn('Friction_1', feats.columns)
        buyer = feats[feats['user_id'] == 1].iloc[0]
        self.assertEqual(int(buyer['Purchase_Frequency']), 1)
        self.assertEqual(int(buyer['Pages_Viewed']), 2)
        self.assertEqual(int(buyer['Cart_Products']), 1)
        self.assertEqual(int(buyer['Purchased_Products']), 1)
        self.assertEqual(float(buyer['Total_Spending']), 10.0)

    def test_dedupe_exact_duplicates(self):
        events = make_events()
        dup = events[(events['user_id'] == 1) & (events['event_type'] == 'purchase')].iloc[[0]]
        events = pd.concat([events, dup], ignore_index=True)
        feats = build_features(events, events['event_time'].max())
        buyer = feats[feats['user_id'] == 1].iloc[0]
        self.assertEqual(int(buyer['Purchase_Frequency']), 1)   # 重复购买行被去重
        self.assertEqual(int(buyer['Cart_Products']), 1)

    def test_build_features_frozen_scalers(self):
        # build_features 支持基期取回标准化器（return_scalers=True）、后续月复用（scalers=...），
        # 且默认调用（不冻结）仍返回单表、行为与旧版一致
        events = make_events(12)
        obs_end = events['event_time'].max()
        feats1, scalers = build_features(events, obs_end, return_scalers=True)
        self.assertIsInstance(scalers, dict)
        self.assertEqual(set(scalers), {'Pages_Viewed', 'Estimated_Time', 'Session_Count'})
        feats2 = build_features(events, obs_end, scalers=scalers)
        np.testing.assert_allclose(feats1['Activity'], feats2['Activity'], atol=1e-12)
        feats3 = build_features(events, obs_end)
        self.assertIsInstance(feats3, pd.DataFrame)
        # 冻结路径与默认（重拟合）对同一数据结果一致（同分布时两种口径等价）
        np.testing.assert_allclose(feats1['Activity'], feats3['Activity'], atol=1e-12)


class TestSegmentUsers(unittest.TestCase):
    def test_segments_exhaustive(self):
        events = make_events(8)
        feats = build_features(events, events['event_time'].max())
        seg, meta = build_user_segment(feats)
        self.assertEqual(len(seg), len(feats))
        self.assertTrue(seg['User_Segment'].isin(
            ['高潜力首购用户', '普通浏览用户', '常规已购用户', '精准直购用户',
             '深度互动用户']).all())
        for key in ['vip_value_index_cutoff', 'nonbuyer_activity_cutoff',
                    'nonbuyer_log_friction_cutoff', 'vip_activity_cutoff']:
            self.assertIn(key, meta)
        self.assertNotIn('vip_log_friction1_cutoff', meta)


class TestSilentBuyerFlag(unittest.TestCase):
    """silent_buyer：跨月沉默是独立标记，不写回 User_Segment。"""

    def setUp(self):
        # 基期分层结果（合成）：高价值两类 + 非高价值三类
        self.base = pd.DataFrame({
            'user_id': [1, 2, 3, 4, 5, 6],
            'User_Segment': ['精准直购用户', '深度互动用户', '精准直购用户',
                             '普通浏览用户', '常规已购用户', '高潜力首购用户'],
        })
        # 观察月事件：2 只浏览、2 又购买、5 只加购；1、3、4、6 完全没出现
        self.obs_events = pd.DataFrame({
            'user_id': [2, 2, 5],
            'event_type': ['view', 'purchase', 'cart'],
            'user_session': ['s2', 's2', 's5'],
            'product_id': [101, 101, 102],
            'price': [0.0, 10.0, 10.0],
        })

    def _active_ids(self, event_type=None):
        """观察月活跃 user_id 集合（可按事件类型收窄，模拟"只有某类行为"）。"""
        ev = self.obs_events
        if event_type is not None:
            ev = ev[ev['event_type'] == event_type]
        return set(ev['user_id'])

    def _flag_of(self, out, uid):
        return bool(out.loc[out['user_id'] == uid, 'High_Value_Silent_Flag'].item())

    def test_silent_when_no_activity_at_all(self):
        # 场景 1：基期高价值 + 观察月完全无活动 → True
        out = silent_buyer(self.base, self._active_ids())
        self.assertTrue(self._flag_of(out, 1))
        self.assertTrue(self._flag_of(out, 3))

    def test_view_only_is_active(self):
        # 场景 2：观察月只有浏览 → False
        out = silent_buyer(self.base, self._active_ids('view'))
        self.assertFalse(self._flag_of(out, 2))

    def test_cart_only_is_active(self):
        # 场景 3：观察月只有加购 → False
        out = silent_buyer(self.base, self._active_ids('cart'))
        self.assertFalse(self._flag_of(out, 5))
        self.assertTrue(self._flag_of(out, 2))   # 2 本月没加购，但确实没有任何行为

    def test_purchase_only_is_active(self):
        # 场景 4：观察月只有购买 → False
        out = silent_buyer(self.base, self._active_ids('purchase'))
        self.assertFalse(self._flag_of(out, 2))

    def test_non_high_value_never_silent(self):
        # 场景 5：非高价值用户即使观察月完全没有活动，也恒为 False
        out = silent_buyer(self.base, self._active_ids())
        for uid in (4, 5, 6):
            self.assertFalse(self._flag_of(out, uid))
        self.assertEqual(out['High_Value_Silent_Flag'].dtype, bool)

    def test_user_segment_is_never_overwritten(self):
        # 场景 6：执行 silent_buyer 后原始 User_Segment 完全不变
        out = silent_buyer(self.base, self._active_ids())
        pd.testing.assert_series_equal(out['User_Segment'], self.base['User_Segment'])
        self.assertNotIn(SILENT_SEGMENT_LABEL, set(out['User_Segment']))

    def test_unusable_active_ids_raise(self):
        # 场景 7：活动集合缺失 / 不可用（含误传 DataFrame）→ 报错，绝不"全员沉默"
        for bad in (None, pd.DataFrame({'user_id': [1, 2]}), set(), []):
            with self.assertRaises((ValueError, TypeError)):
                silent_buyer(self.base, bad)

    def test_missing_columns_raise(self):
        with self.assertRaises(ValueError):
            silent_buyer(self.base[['user_id']], {1})
        with self.assertRaises(TypeError):
            silent_buyer([1, 2, 3], {1})

    def test_final_segment_is_display_only(self):
        # 场景 10：Final_Segment 只在沉默高价值用户上改名，基期标签不受影响
        out = silent_buyer(self.base, self._active_ids())
        expected = ['高价值跨月沉默用户', '深度互动用户', '高价值跨月沉默用户',
                    '普通浏览用户', '常规已购用户', '高潜力首购用户']
        self.assertEqual(out['Final_Segment'].tolist(), expected)
        self.assertEqual(out['Final_Segment'].tolist().count(SILENT_SEGMENT_LABEL), 2)
        # 未沉默的高价值用户 Final_Segment 仍是原基期标签
        self.assertEqual(out.loc[out['user_id'] == 2, 'Final_Segment'].item(), '深度互动用户')

    def test_flag_depends_on_presence_not_on_outcome(self):
        # 观察月"买了"与"只浏览"都算有活动 → 两者 flag 都是 False；
        # 基期标签在两种观察月下完全一致（不因观察月结果反向改变基期身份）
        by_purchase = silent_buyer(self.base, self._active_ids('purchase'))
        by_view = silent_buyer(self.base, self._active_ids('view'))
        self.assertFalse(self._flag_of(by_purchase, 2))
        self.assertFalse(self._flag_of(by_view, 2))
        pd.testing.assert_series_equal(by_purchase['User_Segment'], by_view['User_Segment'])

    def test_pipeline_integration(self):
        # 与真实管线（build_features → build_user_segment）串联时不破坏基期标签
        events = make_events(12)
        feats = build_features(events, events['event_time'].max())
        seg, _ = build_user_segment(feats)
        obs_ids = {1, 2, 4, 5}
        flagged = silent_buyer(seg, obs_ids)
        pd.testing.assert_series_equal(flagged['User_Segment'], seg['User_Segment'])
        vip_base = set(seg[seg['User_Segment'].isin(HIGH_VALUE_SEGMENTS)]['user_id'])
        silent = set(flagged.loc[flagged['High_Value_Silent_Flag'], 'user_id'])
        # 沉默人群 = 基期高价值 - 观察月活跃用户（精确相等）
        self.assertEqual(silent, vip_base - set(obs_ids))
        self.assertGreater(len(silent), 0)
        # 非高价值用户不会被打上标记
        self.assertFalse(flagged.loc[~flagged['User_Segment'].isin(HIGH_VALUE_SEGMENTS),
                                     'High_Value_Silent_Flag'].any())
        # 未购人群的标签不受影响
        self.assertEqual(set(seg[seg['Base_Segment'] == '未购用户']['User_Segment']),
                         set(flagged[flagged['Base_Segment'] == '未购用户']['User_Segment']))


class TestRateTest(unittest.TestCase):
    def test_known_contingency(self):
        # 验证月买过的人（1、2 买了，3、4、5 没买）
        purchased_ids = {1, 2}
        res = rolling_validation_rate_test(purchased_ids, {1, 2, 3}, {4, 5}, '测试组：购买率验证')
        self.assertEqual(res['目标人数'], 3)
        self.assertEqual(res['对照人数'], 2)
        self.assertAlmostEqual(res['目标购买率'], 2 / 3)
        self.assertAlmostEqual(res['对照购买率'], 0.0)
        self.assertTrue(res['p值'] >= 0)


class TestLRPredictRank(unittest.TestCase):
    """纯工具 LR_predict_rank：对指定用户预测『预测月』购买概率并在其内部排名。"""

    def _panel(self):
        frames = []
        for mi, month in enumerate(['2019-10', '2019-11', '2019-12']):
            ev = make_events(12)
            ev['event_time'] = ev['event_time'] + pd.Timedelta(days=mi * 31)
            ev['month'] = month
            if mi == 1:   # 11 月：未购用户（0、6）转为购买（未购标签两类）；部分已购用户（uid%3==2）不买（已购标签两类）
                extra = ev[(ev['user_id'].isin([0, 6])) & (ev['event_type'] == 'cart')].copy()
                extra['event_type'] = 'purchase'
                ev = pd.concat([ev, extra], ignore_index=True)
                ev = ev[~((ev['user_id'] % 3 == 2) & (ev['event_type'] == 'purchase'))]
            if mi == 2:   # 12 月：让部分已购用户（uid%3==2）不买 → 已购训练标签两类
                ev = ev[~((ev['user_id'] % 3 == 2) & (ev['event_type'] == 'purchase'))]
            frames.append(ev)
        return pd.concat(frames, ignore_index=True)

    def _tables(self):
        """把合成面板整理成 {月份: 该月表}（与 notebook 走同一条构造路径）。"""
        return build_month_tables(self._panel(), ['2019-10', '2019-11', '2019-12'])

    def test_predict_rank_nonbuyers(self):
        # 未购用户：返回 Prob / Rank；名次在传入用户内部算（1 起步，并列取最小名次）
        tables = self._tables()
        target = [0, 3, 6, 9]
        out = LR_predict_rank(tables, '2019-11', target)
        self.assertEqual(list(out.columns), ['user_id', 'Prob', 'Rank'])
        self.assertEqual(set(out['user_id']), set(target))
        self.assertEqual(len(out), len(target))
        self.assertEqual(int(out['Rank'].min()), 1)
        self.assertLessEqual(int(out['Rank'].max()), len(target))
        self.assertTrue(out['Prob'].between(0, 1).all())
        # 名次与概率单调一致：概率更高者名次更靠前；概率相同则名次相同
        for _, r1 in out.iterrows():
            for _, r2 in out.iterrows():
                if r1['Prob'] > r2['Prob']:
                    self.assertLess(r1['Rank'], r2['Rank'])
                elif r1['Prob'] == r2['Prob']:
                    self.assertEqual(r1['Rank'], r2['Rank'])

    def test_predict_rank_buyers(self):
        # 已购用户：同一工具、同一输出契约（函数不区分人群）
        tables = self._tables()
        target = [1, 4, 7, 10]
        out = LR_predict_rank(tables, '2019-11', target)
        self.assertEqual(list(out.columns), ['user_id', 'Prob', 'Rank'])
        self.assertEqual(set(out['user_id']), set(target))
        self.assertEqual(int(out['Rank'].min()), 1)
        self.assertLessEqual(int(out['Rank'].max()), len(target))

    def test_predict_rank_subset_is_internal(self):
        # 只传子集时，名次在子集内部计算（不会因外部用户而变大），且相对次序与全集一致
        tables = self._tables()
        full = LR_predict_rank(tables, '2019-11', [0, 1, 3, 6, 9])
        sub = LR_predict_rank(tables, '2019-11', [0, 6])
        self.assertEqual(len(sub), 2)
        self.assertEqual(int(sub['Rank'].min()), 1)
        self.assertLessEqual(int(sub['Rank'].max()), 2)
        full_prob = dict(zip(full['user_id'], full['Prob']))
        sub_prob = dict(zip(sub['user_id'], sub['Prob']))
        for a in sub_prob:
            for b in sub_prob:
                self.assertEqual(full_prob[a] > full_prob[b], sub_prob[a] > sub_prob[b])

    def test_october_cannot_score(self):
        # 10 月是最早建模月，没有更早的训练月份 → 抛 ValueError
        tables = self._tables()
        with self.assertRaises(ValueError):
            LR_predict_rank(tables, '2019-10', [0, 1])

    def test_unknown_target_user_raises(self):
        # 目标用户不在打分月的特征表里 → 报错而不是静默少打分
        tables = self._tables()
        with self.assertRaises(ValueError):
            LR_predict_rank(tables, '2019-11', [99999])
        with self.assertRaises(ValueError):
            LR_predict_rank(tables, '2019-11', [])

    def test_columns(self):
        events = make_events(10)
        feats = build_features(events, events['event_time'].max())
        seg, _ = build_user_segment(feats)
        # 观察月活跃用户（0、1 有活动 → 不会被判为沉默）
        seg = silent_buyer(seg, {0, 1})
        # 概率列由打分函数产出；这里只测 export_tracking 的列契约，直接赋值即可
        seg['First_Purchase_Prob'] = 0.1
        seg['First_Purchase_Rank'] = np.nan
        seg['TopK_Flag'] = np.nan
        seg['Repurchase_Prob'] = np.nan
        seg['Repurchase_Rank'] = np.nan
        track = export_tracking(seg)
        for col in ['user_id', 'User_Segment', 'Activity', 'Friction', 'Value_Index',
                    'First_Purchase_Prob', 'First_Purchase_Rank', 'TopK_Flag',
                    'Repurchase_Prob', 'Repurchase_Rank']:
            self.assertIn(col, track.columns)
        # 默认导出全量：所有标签的用户都在名单里
        self.assertEqual(set(track['User_Segment']), set(seg['User_Segment']))
        self.assertEqual(len(track), len(seg))

    def test_segments_filter(self):
        # 传入 segments 时只导出指定**基期标签**；跨月沉默改为按 High_Value_Silent_Flag 过滤
        events = make_events(10)
        feats = build_features(events, events['event_time'].max())
        seg, _ = build_user_segment(feats)
        # 观察月活跃用户（0、1 有活动 → 不会被判为沉默）
        flagged = silent_buyer(seg, {0, 1})
        # 概率列由打分函数产出；这里只测 export_tracking 的列契约，直接赋值即可
        flagged['First_Purchase_Prob'] = 0.1
        flagged['First_Purchase_Rank'] = np.nan
        flagged['TopK_Flag'] = np.nan
        flagged['Repurchase_Prob'] = np.nan
        flagged['Repurchase_Rank'] = np.nan
        cand = export_tracking(flagged, segments=['高潜力首购用户'])
        self.assertTrue(set(cand['User_Segment']) <= {'高潜力首购用户'})
        self.assertLess(len(cand), len(flagged))
        # 跨月沉默人群：先按标记筛出，再导出（User_Segment 里不再有这个取值）
        silent_rows = flagged[flagged['High_Value_Silent_Flag']]
        self.assertGreater(len(silent_rows), 0)
        silent_track = export_tracking(silent_rows)
        self.assertTrue(set(silent_track['User_Segment']) <= set(HIGH_VALUE_SEGMENTS))
        self.assertEqual(len(silent_track), len(silent_rows))


class TestRollingValidation(unittest.TestCase):
    def test_rolling_validation_synthetic(self):
        frames = []
        for mi, month in enumerate(['2019-10', '2019-11', '2019-12', '2020-01']):
            ev = make_events(10)
            ev['event_time'] = ev['event_time'] + pd.Timedelta(days=mi * 31)
            ev['month'] = month
            frames.append(ev)
        panel = pd.concat(frames, ignore_index=True)
        months = ['2019-10', '2019-11', '2019-12', '2020-01']
        # 滚动验证直接收"月份表"（不再收 panel + cache）
        tables = build_month_tables(panel, months)
        res = rolling_validation(tables, months)
        rates = res['验证表']
        # 实验一：3 个月对（2 月窗口）；实验二：2 个月组（基期→沉默→验证，3 月窗口）
        exp1 = rates[rates['实验'] == '高潜力首购']
        exp2 = rates[rates['实验'] == '高价值跨月沉默']
        self.assertEqual(len(exp1), 3)
        self.assertEqual(len(exp2), 2)
        self.assertEqual(set(exp1['训练月']), {'2019-10', '2019-11', '2019-12'})
        self.assertEqual(set(exp2['沉默月']), {'2019-11', '2019-12'})
        for col in ['训练月', '沉默月', '验证月', '实验', '目标人数', '目标购买率',
                    '对照人数', '对照购买率', '购买率差', 'p值']:
            self.assertIn(col, rates.columns)
        # rolling_validation 只返回验证表（"基准表" 是历史版本的产物，已移除）
        self.assertNotIn('基准表', res)

    def test_load_panel(self):
        ev = make_events(6)
        ev['month'] = '2019-10'
        path = os.path.join(tempfile.gettempdir(), 'test_panel.parquet')
        ev.to_parquet(path, index=False)
        try:
            panel = load_panel(path)
            self.assertIn('month', panel.columns)
            self.assertEqual(len(panel), len(ev))
        finally:
            os.remove(path)


class TestRollingSilentGroups(unittest.TestCase):
    """滚动验证：沉默组 / 活跃组按 High_Value_Silent_Flag 划分，购买结果只取评估月。"""

    MONTHS = ['2019-10', '2019-11', '2019-12']

    def _tables(self):
        """构造 3 个月月度表；把一部分基期高价值用户从观察月表里移除，制造"沉默"用户。"""
        frames = []
        for mi, month in enumerate(self.MONTHS):
            ev = make_events(12)
            ev['event_time'] = ev['event_time'] + pd.Timedelta(days=mi * 31)
            ev['month'] = month
            frames.append(ev)
        tables = build_month_tables(pd.concat(frames, ignore_index=True), self.MONTHS)
        base = tables['2019-10']
        vip = sorted(base.loc[base['User_Segment'].isin(HIGH_VALUE_SEGMENTS), 'user_id'])
        k = max(1, len(vip) // 2) if vip else 0
        if k:
            obs = tables['2019-11']
            tables['2019-11'] = obs[~obs['user_id'].isin(vip[:k])]
        return tables

    def test_groups_are_flag_based_and_disjoint(self):
        # 场景 8：沉默组 = 基期高价值 & flag==True；对照组 = 基期高价值 & flag==False；两组互斥
        tables = self._tables()
        base, obs = tables['2019-10'], tables['2019-11']
        flagged = silent_buyer(base, set(obs['user_id']))
        vip_base = set(base.loc[base['User_Segment'].isin(HIGH_VALUE_SEGMENTS), 'user_id'])
        self.assertGreater(len(vip_base), 1)
        silent = set(flagged.loc[flagged['High_Value_Silent_Flag'], 'user_id'])
        active_mask = (flagged['User_Segment'].isin(HIGH_VALUE_SEGMENTS)
                       & ~flagged['High_Value_Silent_Flag'])
        active = set(flagged.loc[active_mask, 'user_id'])
        # 两组都非空、互斥、并集 = 基期高价值用户
        self.assertGreater(len(silent), 0)
        self.assertGreater(len(active), 0)
        self.assertEqual(silent & active, set())
        self.assertEqual(silent | active, vip_base)
        # 沉默组 = 基期高价值 - （观察月出现过的人）
        self.assertEqual(silent, vip_base - set(obs['user_id']))
        # 滚动验证表里的两组人数与上面独立算出的划分一致
        rates = rolling_validation(tables, self.MONTHS)['验证表']
        row = rates[(rates['实验'] == '高价值跨月沉默') & (rates['训练月'] == '2019-10')].iloc[0]
        self.assertEqual(row['沉默月'], '2019-11')
        self.assertEqual(row['验证月'], '2019-12')
        self.assertEqual(row['目标人数'], len(silent))
        self.assertEqual(row['对照人数'], len(active))

    def test_rates_come_from_evaluation_month_only(self):
        # 场景 9：把评估月（t+2）的购买结果清零 → 两组购买率都变成 0，
        # 说明购买结果只取评估月，基期/观察月的结果不会泄漏进分组依据
        tables = self._tables()
        eval_month = self.MONTHS[-1]
        frozen = tables[eval_month].copy()
        # 评估月表只被当作"购买结果"来源使用（Purchase_Frequency > 0 即买过），清零即无人购买
        frozen['Purchase_Frequency'] = 0
        tables[eval_month] = frozen
        rates = rolling_validation(tables, self.MONTHS)['验证表']
        row = rates[(rates['实验'] == '高价值跨月沉默') & (rates['训练月'] == '2019-10')].iloc[0]
        self.assertGreater(row['目标人数'], 0)
        self.assertGreater(row['对照人数'], 0)
        self.assertEqual(row['目标购买率'], 0.0)
        self.assertEqual(row['对照购买率'], 0.0)


class TestCohortMigration(unittest.TestCase):
    """队列迁移分析：固定基期分层 → 逐月追踪同一批用户（合成面板）。"""

    def _panel(self):
        frames = []
        for mi, month in enumerate(['2019-10', '2019-11', '2019-12', '2020-01']):
            ev = make_events(10)
            if mi > 0:                              # 后续月份只保留一半用户 → 另一半沉默
                ev = ev[ev['user_id'] % 2 == 0]
            ev['event_time'] = ev['event_time'] + pd.Timedelta(days=mi * 31)
            ev['month'] = month
            frames.append(ev)
        return pd.concat(frames, ignore_index=True)

    def test_cohort_migration_outputs(self):
        # cohort_migration 返回冻结标签占比表（基期标签 × 月份 × 冻结标签）
        panel = self._panel()
        months = ['2019-10', '2019-11', '2019-12', '2020-01']
        frozen = cohort_migration(panel, '2019-10', months=months)
        n_after = 3
        self.assertEqual(len(frozen), frozen['基期标签'].nunique() * n_after)
        self.assertIn('基期标签', frozen.columns)
        self.assertIn('month', frozen.columns)
        # 当月无任何事件的用户应记为"无任何活动"
        self.assertIn('无任何活动', frozen.columns)
        # 每个基期标签 × 月份的行内占比之和应为 1
        rows = frozen.set_index(['基期标签', 'month'])
        self.assertTrue(np.allclose(rows.sum(axis=1).to_numpy(), 1.0, atol=1e-9))

    def test_frozen_thresholds_consistency(self):
        # 冻结阈值 = 用 10 月阈值重算 11 月标签，等价于直接调用 segment_users(thresholds=...)
        panel = self._panel()
        oct_events = panel[panel['month'].eq('2019-10')]
        nov_events = panel[panel['month'].eq('2019-11')]
        seg_oct, thr = build_user_segment(build_features(oct_events, oct_events['event_time'].max()))
        seg_nov_frozen, _ = build_user_segment(build_features(nov_events, nov_events['event_time'].max()),
                                               thresholds=thr)
        self.assertNotIn('vip_log_friction1_cutoff', thr)
        self.assertEqual(len(seg_nov_frozen), len(nov_events['user_id'].unique()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
