# 可以自己import我们平台支持的第三方python模块，比如pandas、numpy等。
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


# 在这个方法中编写任何的初始化逻辑。context对象将会在你的算法策略的任何方法之间做传递。
def init(context):
    # 在context中保存全局变量
    # context.s1 = "000001.XSHE"
    # 实时打印日志
    # logger.info("RunInfo: {}".format(context.run_info))

    # 初始化股票池
    context.hs300 = index_components('000300.XSHG')
    # 选股的数量
    context.stock_num = 20
    # 设置权重
    weight = np.mat(np.array(np.array(
        [0.02953221, -0.04920124, -0.10791485, 0.00801783, -0.03613599, 0.1310877, -0.03030564, 0.40286239,
         -0.30166898])))
    context.weight = weight.T
    # 每月进行一次调仓
    # 设置定时器
    scheduler.run_monthly(MyLinearRegression, tradingday=1)


def three_sigma(data):
    """
    基于3sigma 原则离散值处理
    """
    up = data.mean() + 3 * data.std()
    low = data.mean() - 3 * data.std()
    data = np.where(data > up, up, data)
    data = np.where(data < low, low, data)
    return data


def stand_sca(data):
    """
    标准差标准化
    """
    data = (data - data.mean()) / data.std()
    return data


def deal_data(data):
    """
    因子数据处理
    """
    # 1、缺失值处理 直接删除
    data.dropna(axis=0, how='any', inplace=True)

    for col in data.columns:
        # 2、离散值处理
        data.loc[:, col] = three_sigma(data.loc[:, col])
        data.loc[:, col] = stand_sca(data.loc[:, col])

        if col != 'market_cap':  # market_cap 市值因子
            # 市值因子中性化处理
            x = data.loc[:, 'market_cap'].values.reshape((-1, 1))
            y = data.loc[:, col].values
            lr = LinearRegression()
            lr.fit(x, y)  # 特征值至少二维，目标值必须是一维
            y_predict = lr.predict(x)
            data.loc[:, col] = y - y_predict

    return data


def tiao_cang(context):
    """
    调仓
    """
    # 获取所有仓位的股票
    for tmp in context.portfolio.positions.keys():
        # 卖出不在stock_list中的股票
        if tmp not in context.stock_list:
            order_target_percent(tmp, 0)

    for s in context.stock_list:
        order_target_percent(s, 1 / len(context.stock_list))


def MyLinearRegression(context, bar_dict):  #
    """
    每月执行的逻辑函数 基于线性回归
    """
    # 1、获取因子元素  选取9个因子
    q = query(
        fundamentals.eod_derivative_indicator.pe_ratio,
        fundamentals.eod_derivative_indicator.pb_ratio,
        fundamentals.eod_derivative_indicator.market_cap,
        fundamentals.financial_indicator.ev,
        fundamentals.financial_indicator.return_on_asset_net_profit,
        fundamentals.financial_indicator.du_return_on_equity,
        fundamentals.financial_indicator.earnings_per_share,
        fundamentals.income_statement.revenue,
        fundamentals.income_statement.total_expense
    ).filter(fundamentals.stockcode.in_(context.hs300))
    fund = get_fundamentals(q)
    context.fator = fund.T
    # print(context.fator)

    # 2、因子数据处理
    context.fator = deal_data(context.fator)

    # 3、因子 * 权重 + B = 预测收益
    # (300, 9) * (9, 1) = (300, 1)
    context.fator.loc[:, 'factor_return'] = np.dot(context.fator, context.weight)

    # 4、根据预测收益进行排序，降序，选取前stock_num股
    context.stock_list = context.fator.sort_values(by='factor_return', ascending=False).head(context.stock_num).index

    # 5、调仓
    tiao_cang(context)


# before_trading此函数会在每天策略交易开始前被调用，当天只会被调用一次
def before_trading(context):
    pass


# 你选择的证券的数据更新将会触发此段逻辑，例如日或分钟历史数据切片或者是实时数据切片更新
def handle_bar(context, bar_dict):
    pass
    # 开始编写你的主要的算法逻辑

    # bar_dict[order_book_id] 可以拿到某个证券的bar信息
    # context.portfolio 可以拿到现在的投资组合信息

    # 使用order_shares(id_or_ins, amount)方法进行落单

    # TODO: 开始编写你的算法吧！
    # order_shares(context.s1, 1000)


# after_trading函数会在每天交易结束后被调用，当天只会被调用一次
def after_trading(context):
    pass
