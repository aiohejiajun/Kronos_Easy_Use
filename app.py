import pandas as pd
from datetime import datetime, timedelta
import warnings
import gradio as gr
warnings.filterwarnings('ignore')
from model import Kronos, KronosTokenizer, KronosPredictor
MODEL_AVAILABLE = True
import yfinance as yf
import akshare as ak


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

    def get_market_info(self, symbol):
        """获取市场信息"""
        if symbol.endswith('.SH'):
            return '上海证券交易所', 'A股'
        elif symbol.endswith('.SZ'):
            return '深圳证券交易所', 'A股'
        elif symbol.endswith('.HK'):
            return '香港交易所', '港股'
        elif symbol.endswith('.US'):
            return '美国交易所', '美股'
        else:
            return '未知市场', '未知'

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

                # 转换数据类型
                df['timestamps'] = pd.to_datetime(df['timestamps'])
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
                    df['data_source'] = 'yfinance'
                    df['market'] = '其他'
                    return df
            except Exception as e:
                print(f"yfinance数据获取失败: {e}")

        return None


# 创建akshare数据获取器实例
akshare_fetcher = AkshareDataFetcher() if AKSHARE_AVAILABLE else None


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


def add_future_timestamps(df, m_rows, interval):
    """添加未来时间戳"""
    result_df = df.copy()

    # 获取最后一个时间戳
    last_timestamp = result_df['timestamps'].iloc[-1]

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
        delta = timedelta(minutes=5)

    # 生成未来时间戳
    future_timestamps = []
    current_time = last_timestamp

    for i in range(m_rows):
        current_time += delta

        # 对于日线数据跳过周末
        if 'd' in interval:
            while current_time.weekday() >= 5:
                current_time += timedelta(days=1)

        future_timestamps.append(current_time)

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


class InferPipeline:
    def __init__(self, model_key='kronos-small'):
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
        """预测函数"""
        if not self.model_loaded:
            return None, "错误: 模型未加载成功"

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
                return None, "错误: 无法获取股票数据"

            # 转换为标准格式
            df = convert_to_standard_format(df)

            # 添加未来时间戳
            df = add_future_timestamps(df, pred_len, m_interval)

            # 分割数据
            lookback = len(df) - pred_len
            if lookback <= 0:
                return None, "错误: 历史数据不足"

            x_df = df.iloc[:lookback][['open', 'high', 'low', 'close', 'volume']].copy()
            x_timestamp = df.iloc[:lookback]['timestamps'].copy()
            y_timestamp = df.iloc[lookback:lookback + pred_len]['timestamps'].copy()

            # 调用预测器
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
                # 添加时间戳信息
                pred_df['timestamps'] = y_timestamp.values
                pred_df['data_source'] = df['data_source'].iloc[0] if 'data_source' in df.columns else 'unknown'

                return pred_df, "预测完成!"
            else:
                return None, "预测失败"

        except Exception as e:
            return None, f"预测错误: {str(e)}"


# 创建Gradio界面
def create_akshare_interface():
    """创建基于akshare的Gradio界面"""

    pipeline = InferPipeline()

    with gr.Blocks(theme=gr.themes.Soft(), title="实时股票预测系统") as demo:
        gr.Markdown(f"""
        # 📈 实时股票价格预测系统
        **数据源**: {'✅ akshare (实时A股/港股数据)' if AKSHARE_AVAILABLE else '❌ yfinance (有延迟)'}

        ### 支持的市场：
        - 🇨🇳 **A股**: 600519.SH (贵州茅台), 300750.SZ (宁德时代)
        - 🇭🇰 **港股**: 00700.HK (腾讯控股), 09988.HK (阿里巴巴)
        - 🇺🇸 **美股**: AAPL, TSLA (使用yfinance)

        ### 也支持中文名称：贵州茅台、腾讯控股、宁德时代
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 参数设置")

                company_ticker = gr.Textbox(
                    label="股票代码/名称",
                    value="贵州茅台",
                    info="输入股票代码或中文名称"
                )

                n_days = gr.Dropdown(
                    label="历史周期",
                    choices=["1d", "3d", "5d", "1wk", "1mo"],
                    value="5d"
                )

                m_interval = gr.Dropdown(
                    label="时间间隔",
                    choices=["1m", "5m", "15m", "30m", "60m", "1d"],
                    value="15m"
                )

                pred_len = gr.Slider(
                    label="预测长度",
                    minimum=1,
                    maximum=50,
                    value=10,
                    step=1
                )

                temperature = gr.Slider(
                    label="温度参数",
                    minimum=0.1,
                    maximum=2.0,
                    value=1.0,
                    step=0.1
                )

                top_p = gr.Slider(
                    label="Top-p采样",
                    minimum=0.1,
                    maximum=1.0,
                    value=0.9,
                    step=0.1
                )

                sample_count = gr.Slider(
                    label="采样次数",
                    minimum=1,
                    maximum=10,
                    value=1,
                    step=1
                )

                predict_btn = gr.Button("开始预测", variant="primary")

            with gr.Column(scale=2):
                gr.Markdown("### 预测结果")

                message_output = gr.Textbox(
                    label="状态信息",
                    interactive=False
                )

                result_table = gr.Dataframe(
                    label="预测结果",
                    headers=["时间戳", "开盘价", "最高价", "最低价", "收盘价", "成交量", "数据源"],
                    interactive=False,
                    wrap=True
                )

        # 示例
        examples = gr.Examples(
            examples=[
                ["贵州茅台", "5d", "15m", 10, 1.0, 0.9, 1],
                ["腾讯控股", "3d", "30m", 5, 0.8, 0.95, 1],
                ["300750.SZ", "5d", "60m", 8, 1.2, 0.85, 1],
                ["AAPL", "5d", "1d", 15, 1.0, 0.9, 1]
            ],
            inputs=[company_ticker, n_days, m_interval, pred_len, temperature, top_p, sample_count]
        )

        def predict_wrapper(company_ticker, n_days, m_interval, pred_len, temperature, top_p, sample_count):
            """包装预测函数"""
            pred_df, message = pipeline.infer(
                company_ticker, n_days, m_interval, pred_len, temperature, top_p, sample_count
            )

            if pred_df is not None:
                # 格式化数值
                for col in ['open', 'high', 'low', 'close']:
                    if col in pred_df.columns:
                        pred_df[col] = pred_df[col].round(4)
                if 'volume' in pred_df.columns:
                    pred_df['volume'] = pred_df['volume'].round(0)

                return message, pred_df
            else:
                return message, pd.DataFrame()

        predict_btn.click(
            fn=predict_wrapper,
            inputs=[company_ticker, n_days, m_interval, pred_len, temperature, top_p, sample_count],
            outputs=[message_output, result_table]
        )

    return demo


if __name__ == "__main__":
    # 安装检查
    if not AKSHARE_AVAILABLE:
        print("⚠️  建议安装akshare以获得更好的A股/港股数据: pip install akshare")

    # 创建并启动界面
    demo = create_akshare_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    )