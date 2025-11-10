import warnings
import gradio as gr
import matplotlib.font_manager as fm
import os

import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime, timedelta
import numpy as np

warnings.filterwarnings('ignore')
from model import Kronos, KronosTokenizer, KronosPredictor

MODEL_AVAILABLE = True
import yfinance as yf
import akshare as ak

AKSHARE_AVAILABLE = True

AVAILABLE_MODELS = {
    'kronos-mini': {
        'name': 'Kronos-mini',
        'model_id': 'NeoQuasar/Kronos-mini',
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-2k',
        'context_length': 2048,
        'params': '4.1M',
        'description': 'Lightweight model, suitable for fast prediction'
    },
    'kronos-small': {
        'name': 'Kronos-small',
        'model_id': 'NeoQuasar/Kronos-small',
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-base',
        'context_length': 512,
        'params': '24.7M',
        'description': 'Small model, balanced performance and speed'
    },
    'kronos-base': {
        'name': 'Kronos-base',
        'model_id': 'NeoQuasar/Kronos-base',
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-base',
        'context_length': 512,
        'params': '102.3M',
        'description': 'Base model, provides better prediction quality'
    }
}


class MarketTimeHandler:
    """市场时间处理器"""

    def __init__(self):
        # 各市场交易时间（北京时间）
        self.market_hours = {
            'A股': {
                'open_time': '09:30:00',
                'close_time': '15:00:00',
                'lunch_break_start': '11:30:00',
                'lunch_break_end': '13:00:00',
                'timezone': 'Asia/Shanghai'
            },
            '港股': {
                'open_time': '09:30:00',
                'close_time': '16:00:00',
                'lunch_break_start': '12:00:00',
                'lunch_break_end': '13:00:00',
                'timezone': 'Asia/Hong_Kong'
            },
            '美股': {
                'open_time': '21:30:00',  # 北京时间晚上9:30
                'close_time': '04:00:00',  # 次日凌晨4:00
                'timezone': 'America/New_York'
            }
        }

    def get_market_type(self, symbol):
        """根据股票代码判断市场类型"""
        if symbol.endswith(('.SH', '.SZ')):
            return 'A股'
        elif symbol.endswith('.HK'):
            return '港股'
        else:
            return '美股'

    def adjust_to_beijing_time(self, timestamp):
        """调整到北京时间"""
        # 如果时间戳有时区信息，转换为北京时间（UTC+8）
        if timestamp.tzinfo is not None:
            # 转换为UTC时间
            utc_time = timestamp.tz_convert('UTC')
            # 转换为北京时间（UTC+8）
            beijing_time = utc_time.tz_convert('Asia/Shanghai')
            # 移除时区信息
            return beijing_time.tz_localize(None)
        return timestamp

    def generate_trading_timestamps(self, start_time, interval, market_type, n_periods):
        """生成符合市场交易时间的时间戳（北京时间）"""
        if market_type not in self.market_hours:
            market_type = 'A股'  # 默认使用A股时间

        market_info = self.market_hours[market_type]

        # 解析时间间隔
        if 'm' in interval:
            minutes = int(interval.replace('m', ''))
            delta = timedelta(minutes=minutes)
        elif 'h' in interval:
            hours = int(interval.replace('h', ''))
            delta = timedelta(hours=hours)
        elif 'd' in interval:
            days = int(interval.replace('d', ''))
            delta = timedelta(days=days)
        else:
            delta = timedelta(minutes=5)  # 默认5分钟

        timestamps = []
        current_time = start_time

        for i in range(n_periods):
            # 移动到下一个交易时间段
            current_time = self._get_next_trading_time(current_time, market_info, delta)
            timestamps.append(current_time)

        return timestamps

    def _get_next_trading_time(self, current_time, market_info, delta):
        """获取下一个交易时间点"""
        next_time = current_time + delta

        # 对于日线及以上数据，跳过周末
        if delta >= timedelta(days=1):
            while next_time.weekday() >= 5:  # 5=周六, 6=周日
                next_time += timedelta(days=1)

        # 对于日内数据，检查是否在交易时间内
        if delta < timedelta(days=1):
            next_time = self._adjust_intraday_time(next_time, market_info, delta)

        return next_time

    def _adjust_intraday_time(self, time_point, market_info, delta):
        """调整日内交易时间"""
        # 获取交易时间
        open_time = datetime.strptime(market_info['open_time'], '%H:%M:%S').time()
        close_time = datetime.strptime(market_info['close_time'], '%H:%M:%S').time()

        current_date = time_point.date()
        current_time = time_point.time()

        # 检查是否在交易时间内
        if current_time < open_time:
            # 早于开盘时间，调整到当天开盘
            return datetime.combine(current_date, open_time)
        elif current_time > close_time:
            # 晚于收盘时间，调整到下一个交易日的开盘
            next_date = current_date + timedelta(days=1)
            while next_date.weekday() >= 5:  # 跳过周末
                next_date += timedelta(days=1)
            return datetime.combine(next_date, open_time)

        # 检查午休时间（A股和港股）
        if 'lunch_break_start' in market_info and 'lunch_break_end' in market_info:
            lunch_start = datetime.strptime(market_info['lunch_break_start'], '%H:%M:%S').time()
            lunch_end = datetime.strptime(market_info['lunch_break_end'], '%H:%M:%S').time()

            if lunch_start <= current_time <= lunch_end:
                # 在午休时间，调整到午休结束
                return datetime.combine(current_date, lunch_end)

        return time_point


# 创建市场时间处理器
market_time_handler = MarketTimeHandler()


class AkshareDataFetcher:
    """akshare 数据获取器"""

    def __init__(self):
        self.market_mapping = {
            '.SH': '上海证券交易所',
            '.SZ': '深圳证券交易所',
            '.BJ': '北京证券交易所',
            '.HK': '香港交易所',
            '.US': '美国交易所'
        }
        self.time_handler = MarketTimeHandler()

    def normalize_symbol(self, symbol):
        """标准化股票代码"""
        symbol = symbol.upper().strip()

        # 处理中文名称
        chinese_stocks = {
            '腾讯控股': '00700.HK',
            '贵州茅台': '600519.SH',
            '宁德时代': '300750.SZ',
            '中国平安': '601318.SH',
            '招商银行': '600036.SH',
            '比亚迪': '002594.SZ',
            '美团': '03690.HK',
            '小米集团': '01810.HK',
            '阿里巴巴': '09988.HK',
            '京东': '09618.HK'
        }

        if symbol in chinese_stocks:
            return chinese_stocks[symbol]

        # 添加默认后缀
        if not any(symbol.endswith(suffix) for suffix in ['.SH', '.SZ', '.HK', '.US']):
            if symbol.startswith(('6', '5', '9')):
                symbol += '.SH'
            elif symbol.startswith(('0', '3')):
                symbol += '.SZ'
            elif len(symbol) == 4 and symbol.isdigit():
                symbol += '.HK'

        return symbol

    def fetch_a_stock_data(self, symbol, period="5d", interval="5m"):
        """获取A股数据"""
        try:
            # 去除后缀
            clean_symbol = symbol.replace('.SH', '').replace('.SZ', '')

            # 确定交易所
            exchange = 'sh' if symbol.endswith('.SH') else 'sz'
            full_symbol = f"{exchange}{clean_symbol}"

            print(f"获取A股数据: {full_symbol}, 周期: {period}, 间隔: {interval}")

            # 根据间隔选择不同的akshare函数
            if interval in ['1m', '5m', '15m', '30m', '60m']:
                # 分钟级数据
                period_map = {
                    '1d': '1',
                    '5d': '5',
                    '1mo': '30'
                }
                period_num = period_map.get(period, '5')

                df = ak.stock_zh_a_hist_min_em(
                    symbol=clean_symbol,
                    period=interval,
                    start_date=(datetime.now() - timedelta(days=int(period_num))).strftime('%Y%m%d'),
                    end_date=datetime.now().strftime('%Y%m%d'),
                    adjust="qfq"
                )
            else:
                # 日线数据
                df = ak.stock_zh_a_hist(
                    symbol=clean_symbol,
                    period="daily",
                    start_date=(datetime.now() - timedelta(days=30)).strftime('%Y%m%d'),
                    end_date=datetime.now().strftime('%Y%m%d'),
                    adjust="qfq"
                )

            if df is not None and not df.empty:
                # 重命名列以匹配统一格式
                column_mapping = {
                    '日期': 'timestamps',
                    '时间': 'timestamps',
                    '开盘': 'open',
                    '最高': 'high',
                    '最低': 'low',
                    '收盘': 'close',
                    '成交量': 'volume',
                    '成交额': 'amount'
                }

                df = df.rename(columns=column_mapping)

                # 确保有timestamps列
                if 'timestamps' not in df.columns:
                    if '日期' in df.columns and '时间' in df.columns:
                        df['timestamps'] = df['日期'] + ' ' + df['时间']
                    elif '日期' in df.columns:
                        df['timestamps'] = df['日期']

                # 选择需要的列
                required_cols = ['timestamps', 'open', 'high', 'low', 'close', 'volume']
                available_cols = [col for col in required_cols if col in df.columns]
                df = df[available_cols]

                # 转换数据类型并调整到北京时间
                df['timestamps'] = pd.to_datetime(df['timestamps'])
                df['timestamps'] = df['timestamps'].apply(self.time_handler.adjust_to_beijing_time)

                for col in ['open', 'high', 'low', 'close']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                if 'volume' in df.columns:
                    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')

                df['data_source'] = 'akshare'
                df['market'] = 'A股'

                return df

        except Exception as e:
            print(f"A股数据获取失败 {symbol}: {e}")

        return None

    def fetch_hk_stock_data(self, symbol, period="5d", interval="5m"):
        """获取港股数据"""
        try:
            # 去除.HK后缀
            clean_symbol = symbol.replace('.HK', '')

            print(f"获取港股数据: {clean_symbol}")

            # 使用akshare获取港股数据
            df = ak.stock_hk_hist(
                symbol=clean_symbol,
                period="daily",
                start_date=(datetime.now() - timedelta(days=30)).strftime('%Y%m%d'),
                end_date=datetime.now().strftime('%Y%m%d'),
                adjust="qfq"
            )

            if df is not None and not df.empty:
                # 重命名列
                df = df.rename(columns={
                    '日期': 'timestamps',
                    '开盘': 'open',
                    '最高': 'high',
                    '最低': 'low',
                    '收盘': 'close',
                    '成交量': 'volume'
                })

                df['timestamps'] = pd.to_datetime(df['timestamps'])
                df['timestamps'] = df['timestamps'].apply(self.time_handler.adjust_to_beijing_time)

                for col in ['open', 'high', 'low', 'close']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                if 'volume' in df.columns:
                    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')

                df['data_source'] = 'akshare'
                df['market'] = '港股'

                return df

        except Exception as e:
            print(f"港股数据获取失败 {symbol}: {e}")

        return None

    def fetch_stock_data(self, symbol, period="5d", interval="5m"):
        """统一获取股票数据"""
        symbol = self.normalize_symbol(symbol)

        if symbol.endswith(('.SH', '.SZ')):
            return self.fetch_a_stock_data(symbol, period, interval)
        elif symbol.endswith('.HK'):
            return self.fetch_hk_stock_data(symbol, period, interval)
        else:
            # 非A股/港股，使用yfinance作为备用
            try:
                yf_data = yf.download(symbol, period=period, interval=interval)
                if not yf_data.empty:
                    df = yf_data.reset_index()
                    df = df.rename(columns={
                        'Datetime': 'timestamps',
                        'Open': 'open',
                        'High': 'high',
                        'Low': 'low',
                        'Close': 'close',
                        'Volume': 'volume'
                    })
                    # 调整时间到北京时间
                    df['timestamps'] = pd.to_datetime(df['timestamps'])
                    df['timestamps'] = df['timestamps'].apply(self.time_handler.adjust_to_beijing_time)
                    df['data_source'] = 'yfinance'
                    df['market'] = '其他'
                    return df
            except Exception as e:
                print(f"yfinance数据获取失败: {e}")

        return None


# 创建akshare数据获取器实例
akshare_fetcher = AkshareDataFetcher() if AKSHARE_AVAILABLE else None

# 设置中文字体
try:
    # 尝试设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
    plt.rcParams['axes.unicode_minus'] = False
    print("✅ 中文字体设置成功")
except:
    print("⚠️ 中文字体设置失败，使用默认字体")


def clean_dataframe_for_plotting(df):
    """专门为绘图清理DataFrame结构"""
    if df is None or df.empty:
        return None

    df_clean = df.copy()

    # 重置索引，避免索引级别和列名冲突
    df_clean = df_clean.reset_index(drop=True)

    # 检查是否有timestamps列
    if 'timestamps' not in df_clean.columns:
        # 尝试从索引中获取
        if hasattr(df_clean.index, 'name') and df_clean.index.name == 'timestamps':
            df_clean = df_clean.reset_index()
        elif hasattr(df_clean.index, 'names') and 'timestamps' in df_clean.index.names:
            df_clean = df_clean.reset_index()
        else:
            # 创建默认时间戳
            start_date = datetime.now() - timedelta(days=len(df_clean))
            df_clean['timestamps'] = [start_date + timedelta(hours=i) for i in range(len(df_clean))]

    # 确保timestamps是datetime类型
    df_clean['timestamps'] = pd.to_datetime(df_clean['timestamps'])

    # 确保有所有必需的列
    required_columns = ['timestamps', 'open', 'high', 'low', 'close', 'volume']
    for col in required_columns:
        if col not in df_clean.columns:
            df_clean[col] = np.nan

    # 选择需要的列
    df_clean = df_clean[required_columns]

    # 排序并重置索引
    df_clean = df_clean.sort_values('timestamps').reset_index(drop=True)

    return df_clean


def add_future_timestamps(df, m_rows, interval, symbol):
    """添加未来时间戳（使用正确的市场交易时间）"""
    result_df = df.copy()

    # 获取最后一个时间戳
    last_timestamp = result_df['timestamps'].iloc[-1]

    # 获取市场类型
    market_type = market_time_handler.get_market_type(symbol)

    # 生成符合市场交易时间的未来时间戳（北京时间）
    future_timestamps = market_time_handler.generate_trading_timestamps(
        last_timestamp, interval, market_type, m_rows
    )


    # 创建未来数据
    future_data = []
    for ts in future_timestamps:
        future_row = {
            'timestamps': ts,
            'open': None,
            'high': None,
            'low': None,
            'close': None,
            'volume': None
        }
        future_data.append(future_row)

    future_df = pd.DataFrame(future_data)

    # 合并数据
    result_df = pd.concat([result_df, future_df], ignore_index=True)
    result_df['data_type'] = 'historical'
    result_df.loc[result_df.index >= len(df), 'data_type'] = 'future'

    return result_df


# 使用同级目录下的SimHei.ttf字体文件
def setup_chinese_font():
    """设置中文字体"""
    try:
        # 检查字体文件是否存在
        font_path = "SimHei.ttf"
        if os.path.exists(font_path):
            # 注册字体
            font_prop = fm.FontProperties(fname=font_path)
            # 设置全局字体
            plt.rcParams['font.family'] = [font_prop.get_name()]
            plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
            plt.rcParams['axes.unicode_minus'] = False
            print(f"✅ 成功加载中文字体: {font_prop.get_name()}")
            return True
        else:
            print(f"❌ 字体文件不存在: {font_path}")
            return False
    except Exception as e:
        print(f"❌ 字体设置失败: {e}")
        return False

# 初始化中文字体
font_setup_success = setup_chinese_font()
# 创建字体属性对象供后续使用
if font_setup_success:
    font_zh = fm.FontProperties(fname="SimHei.ttf", size=12)
else:
    font_zh = None
    print("⚠️ 使用默认字体，中文可能显示为方块")

def convert_to_standard_format(df):
    """转换为标准格式"""
    if df is None or df.empty:
        return None

    # 确保有所有必需的列
    required_columns = ['timestamps', 'open', 'high', 'low', 'close', 'volume']
    for col in required_columns:
        if col not in df.columns:
            df[col] = None

    # 选择需要的列
    df = df[required_columns]

    # 确保时间戳格式
    df['timestamps'] = pd.to_datetime(df['timestamps'])

    # 排序
    df = df.sort_values('timestamps').reset_index(drop=True)

    return df


def create_prediction_chart(pred_df, symbol):
    """根据预测结果DataFrame绘制high和low两条线"""
    try:
        if pred_df is None or pred_df.empty:
            return "<p style='color: orange;'>预测数据为空，无法生成图表</p>"

        # 清理DataFrame结构
        df_clean = clean_dataframe_for_plotting(pred_df)
        if df_clean is None or df_clean.empty:
            return "<p style='color: orange;'>数据清理后为空，无法生成图表</p>"

        # 检查必要的列是否存在
        required_cols = ['timestamps', 'high', 'low']
        missing_cols = [col for col in required_cols if col not in df_clean.columns]
        if missing_cols:
            return f"<p style='color: red;'>缺少必要的数据列: {', '.join(missing_cols)}</p>"

        # 过滤掉NaN值
        valid_data = df_clean.dropna(subset=['high', 'low'])
        if valid_data.empty:
            return "<p style='color: orange;'>有效数据为空，无法生成图表</p>"

        # 创建图表
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(12, 6))

        # 按时间排序
        valid_data = valid_data.sort_values('timestamps')

        # 绘制high线（红色）
        ax.plot(valid_data['timestamps'], valid_data['high'],
                'r-', linewidth=2.5, label='最高价 (High)', marker='o', markersize=4, alpha=0.8)

        # 绘制low线（蓝色）
        ax.plot(valid_data['timestamps'], valid_data['low'],
                'b-', linewidth=2.5, label='最低价 (Low)', marker='s', markersize=4, alpha=0.8)

        # 填充价格区间
        ax.fill_between(valid_data['timestamps'],
                        valid_data['low'], valid_data['high'],
                        color='gray', alpha=0.2, label='价格区间')

        # 设置图表属性（使用中文字体）
        if font_setup_success:
            ax.set_title(f'{symbol} - 价格区间预测', fontproperties=font_zh, fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('时间', fontproperties=font_zh, fontsize=12)
            ax.set_ylabel('价格', fontproperties=font_zh, fontsize=12)
        else:
            ax.set_title(f'{symbol} - Price Prediction', fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('Time', fontsize=12)
            ax.set_ylabel('Price', fontsize=12)

        # 图例（使用中文或英文）
        if font_setup_success:
            ax.legend(['最高价 (High)', '最低价 (Low)', '价格区间'], prop=font_zh, loc='upper left')
        else:
            ax.legend(['High Price', 'Low Price', 'Price Range'], loc='upper left')

        ax.grid(True, alpha=0.3, linestyle='--')

        # 格式化x轴
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m/%d %H:%M'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 添加统计信息
        if not valid_data.empty:
            avg_high = valid_data['high'].mean()
            avg_low = valid_data['low'].mean()
            price_range = valid_data['high'].max() - valid_data['low'].min()
            volatility = ((valid_data['high'] - valid_data['low']) / valid_data['low']).mean() * 100

            if font_setup_success:
                stats_text = f'''统计信息:
最高价: {valid_data["high"].max():.2f}
最低价: {valid_data["low"].min():.2f}
平均高价: {avg_high:.2f}
平均低价: {avg_low:.2f}
价格区间: {price_range:.2f}
波动率: {volatility:.1f}%'''
            else:
                stats_text = f'''Statistics:
Max Price: {valid_data["high"].max():.2f}
Min Price: {valid_data["low"].min():.2f}
Avg High: {avg_high:.2f}
Avg Low: {avg_low:.2f}
Price Range: {price_range:.2f}
Volatility: {volatility:.1f}%'''

            # 使用适当的字体属性
            if font_setup_success:
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontproperties=font_zh, fontsize=9,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            else:
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        plt.tight_layout()

        # 转换为base64图片
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        plt.close()

        return f'<img src="data:image/png;base64,{img_str}" style="max-width:100%; height:auto; border:1px solid #ddd; border-radius:8px;">'

    except Exception as e:
        return f"<p style='color: red;'>图表生成失败: {str(e)}</p>"


class InferPipeline:
    def __init__(self, model_key='kronos-base'):
        try:
            device = 'cpu'
            model_config = AVAILABLE_MODELS[model_key]
            print(f"加载模型: {model_config['name']}")

            tokenizer = KronosTokenizer.from_pretrained(model_config['tokenizer_id'])
            model = Kronos.from_pretrained(model_config['model_id'])
            self.predictor = KronosPredictor(model, tokenizer, device=device,
                                             max_context=model_config['context_length'])
            self.model_loaded = True
            print("模型加载成功!")
        except Exception as e:
            print(f"模型加载失败: {e}")
            self.predictor = None
            self.model_loaded = False

    def infer(self, company_ticker, n_days, m_interval, pred_len, temperature, top_p, sample_count):
        """预测函数（使用正确的时间戳）"""
        if not self.model_loaded:
            return None, None, "错误: 模型未加载成功"

        try:
            # 使用akshare获取数据（如果可用）
            if AKSHARE_AVAILABLE and akshare_fetcher:
                print("使用akshare获取数据...")
                df = akshare_fetcher.fetch_stock_data(company_ticker, n_days, m_interval)
            else:
                print("使用yfinance获取数据...")
                yf_data = yf.download(company_ticker, period=n_days, interval=m_interval)
                df = yf_data.reset_index().rename(columns={
                    'Datetime': 'timestamps',
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                })
                df['data_source'] = 'yfinance'

            if df is None or df.empty:
                return None, None, "错误: 无法获取股票数据"

            # 转换为标准格式
            df = convert_to_standard_format(df)

            # 添加未来时间戳（使用正确的市场交易时间）
            df = add_future_timestamps(df, pred_len, m_interval, company_ticker)

            # 分割数据
            lookback = len(df) - pred_len
            if lookback <= 0:
                return None, None, "错误: 历史数据不足"

            historical_df = df.iloc[:lookback].copy()
            x_df = historical_df[['open', 'high', 'low', 'close', 'volume']].copy()
            x_timestamp = historical_df['timestamps'].copy()
            y_timestamp = df.iloc[lookback:lookback + pred_len]['timestamps'].copy()

            # 调用预测器（使用正确的时间戳）
            pred_df = self.predictor.predict(
                df=x_df,
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=pred_len,
                T=temperature,
                top_p=top_p,
                sample_count=sample_count
            )

            if pred_df is not None:
                # 添加时间戳信息（已经是正确的时间，不需要再修改）
                pred_df['timestamps'] = y_timestamp.values
                pred_df['data_source'] = df['data_source'].iloc[0] if 'data_source' in df.columns else 'unknown'

                # 创建预测结果图表
                kline_chart = create_prediction_chart(pred_df, company_ticker)

                return pred_df, kline_chart, "预测完成!"
            else:
                return None, None, "预测失败"

        except Exception as e:
            return None, None, f"预测错误: {str(e)}"


# 创建Gradio界面
def create_akshare_interface():
    """创建基于akshare的Gradio界面"""
    pipeline = InferPipeline()

    with gr.Blocks(theme=gr.themes.Soft(), title="量化之神") as demo:
        gr.Markdown(f"""
        # 📈 实时股票价格预测系统
        **数据源**: {'✅ akshare (实时A股/港股数据)' if AKSHARE_AVAILABLE else '❌ yfinance (有延迟)'}
        **时间标准**: 所有时间均为北京时间

        ### 支持的市场：
        - 🇨🇳 **A股**: 600519.SH (贵州茅台), 300750.SZ (宁德时代) - 交易时间: 9:30-15:00
        - 🇭🇰 **港股**: 00700.HK (腾讯控股), 09988.HK (阿里巴巴) - 交易时间: 9:30-16:00  
        - 🇺🇸 **美股**: AAPL, TSLA (使用yfinance) - 交易时间: 21:30-04:00 (北京时间)
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 参数设置")
                company_ticker = gr.Textbox(label="股票代码/名称", value="600519.SH", info="输入股票代码")
                n_days = gr.Dropdown(label="历史周期", choices=["1d", "3d", "5d", "1wk", "1mo"], value="5d")
                m_interval = gr.Dropdown(label="时间间隔", choices=["1m", "5m", "15m", "30m", "60m", "1d"], value="1m")
                pred_len = gr.Slider(label="预测长度", minimum=1, maximum=50, value=10, step=1)
                temperature = gr.Slider(label="温度参数", minimum=0.1, maximum=2.0, value=1.0, step=0.1)
                top_p = gr.Slider(label="Top-p采样", minimum=0.1, maximum=1.0, value=0.9, step=0.1)
                sample_count = gr.Slider(label="采样次数", minimum=1, maximum=10, value=1, step=1)
                predict_btn = gr.Button("开始预测", variant="primary")

            with gr.Column(scale=2):
                gr.Markdown("### 预测结果")
                message_output = gr.Textbox(label="状态信息", interactive=False)
                kline_chart = gr.HTML(label="价格预测图表",
                                      value="<p style='text-align: center; color: #666;'>预测结果将显示在这里</p>")
                result_table = gr.Dataframe(label="预测结果明细",
                                            headers=["时间戳", "开盘价", "最高价", "最低价", "收盘价", "成交量",
                                                     "数据源"], interactive=False, wrap=True)

        def predict_wrapper(company_ticker, n_days, m_interval, pred_len, temperature, top_p, sample_count):
            """包装预测函数"""
            pred_df, kline_chart_html, message = pipeline.infer(
                company_ticker, n_days, m_interval, pred_len, temperature, top_p, sample_count
            )

            if pred_df is not None:
                # 格式化数值
                for col in ['open', 'high', 'low', 'close']:
                    if col in pred_df.columns:
                        pred_df[col] = pred_df[col].round(4)
                if 'volume' in pred_df.columns:
                    pred_df['volume'] = pred_df['volume'].round(0)

                return message, kline_chart_html, pred_df
            else:
                return message, "<p style='color: red;'>预测失败，无法生成图表</p>", pd.DataFrame()

        predict_btn.click(
            fn=predict_wrapper,
            inputs=[company_ticker, n_days, m_interval, pred_len, temperature, top_p, sample_count],
            outputs=[message_output, kline_chart, result_table]
        )

    return demo


if __name__ == "__main__":
    if not AKSHARE_AVAILABLE:
        print("⚠️  建议安装akshare以获得更好的A股/港股数据: pip install akshare")

    demo = create_akshare_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, inbrowser=True)