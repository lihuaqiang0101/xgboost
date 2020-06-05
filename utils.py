import pandas as pd
import numpy as np

#去除特殊符号
def qx(data):
    try:
        if '，' in data:
            data = data.replace('，', '')
    except:
        pass
    if '。' in data:
        data = data.replace('。','')
    if '-' in data:
        data = data.replace('-','')
    if '*' in data:
        data = data.replace('*', '')
    if ' ' in data:
        data = data.replace(' ', '')
    if '.' in data:
        data = data.replace('.', '')
    if '"' in data:
        data = data.replace('"', '')
    return data

def is_contain_chinese(check_str):

    """

    判断字符串中是否包含中文

    :param check_str: {str} 需要检测的字符串

    :return: {bool} 包含返回True， 不包含返回False

    """

    for ch in check_str:

        if u'\u4e00' <= ch <= u'\u9fff':

            return True

    return False

"""
功能：
厂牌转换函数
输入：
原始厂牌列表
输出：
转换后的厂牌列表
"""
def convertchangpai(Datas):
    Dict1 = {}
    L = []
    with open('file\厂牌修正表20200312.csv') as f:
        datas = f.readlines()
        for data in datas:
            data = data.strip()
            data = data.split(',')
            if qx(data[0]) not in Dict1:
                Dict1[qx(data[0])] = qx(data[1])
    for i in range(len(Datas)):
        Datas[i] = Datas[i].upper()
        try:
            cx = Dict1[qx(Datas[i])]
        except:
            cx = qx(Datas[i])
        if '宝马' in Datas[i]:
            cx = '宝马'
        elif '马自达' in Datas[i]:
            cx = '马自达'
        elif '奔驰' in Datas[i]:
            cx = '奔驰'
        elif '奥迪' in Datas[i]:
            cx = '奥迪'
        elif '保时捷' in Datas[i]:
            cx = '保时捷'
        elif '标致' in Datas[i]:
            cx = '标致'
        elif '哈弗' in Datas[i]:
            cx = '哈弗'
        elif '宝骏' in Datas[i]:
            cx = '宝骏汽车'
        elif '大众' in Datas[i]:
            cx = '大众'
        elif not is_contain_chinese(Datas[i]):
            cx = '无'
        elif '4轮电动车' in Datas[i]:
            cx = '无'
        elif '自定义' in Datas[i] or '标准' in Datas[i]:
            cx = '无'
        L.append(cx)
    return L

#车系去括号
def qxcx(data):
    if '【' in data:
        index = data.find('【')
        data = data[:index]
    if '[' in data:
        index = data.find('[')
        data = data[:index]
    if '（' in data:
        index = data.find('（')
        data = data[:index]
    if '(' in data:
        index = data.find('(')
        data = data[:index]
    return data

"""
功能：
车系转换函数
输入：
Datas1:转换后的厂牌列表
Datas2:转换前的车系列表
输出：
转换后的车系列表
"""
def convertchexi(Datas1,Datas2):#D1,D2分别是厂牌和车系
    L = []
    datas = pd.read_csv('file\车系修正表20200312.csv',encoding='gbk', engine='python')
    fit_name = datas['brand']
    fit_name0 = datas['auto_series_chinaname']
    chexi = datas['auto_series_chinaname0']
    DIc = {}
    for i in range(len(fit_name)):
        try:
            DIc[qx(fit_name[i])+qx(fit_name0[i])] = qx(chexi[i])
        except:
            pass
    for i in range(len(Datas1)):
        Datas1[i] = Datas1[i].upper()
        Datas2[i] = Datas2[i].upper()
        try:
            cx = DIc[qx(Datas1[i])+qx(qxcx(Datas2[i]))]
            L.append(cx.upper())
        except:
                if '自定义' in Datas2[i] or '标准' in Datas2[i]:
                    L.append('无')
                else:
                    L.append(qx(qxcx(Datas2[i])).upper())
    return L

#清洗项目函数
def qxxm(data):
    data = data.upper()
    # if '前' in data:
    #     if not '前围' in data and not '前门' in data:
    #         data = data.replace('前', '')
    # if '后' in data:
    #     if not '后视镜' in data and not '后围' in data:
    #         data = data.replace('后', '')
    # if '左' in data:
    #     data = data.replace('左', '')
    # if '右' in data:
    #     data = data.replace('右', '')
    # if '上' in data:
    #     data = data.replace('上', '')
    # if '下' in data:
    #     data = data.replace('下', '')
    if '（' in data:
        data = data.replace('（','')
    if '）' in data:
        data = data.replace('）','')
    if '(' in data:
        data = data.replace('(','')
    if ')' in data:
        data = data.replace(')','')
    if '"' in data:
        data = data.replace('"', '')
    if '喷漆' in data:
        data = data.replace('喷漆', '')
    if '喷塑' in data:
        data = data.replace('喷塑', '')
    if '修复' in data:
        data = data.replace('修复', '')
    if '含拆装' in data:
        data = data.replace('含拆装', '')
    if '拆装' in data:
        data = data.replace('拆装', '')
    if '油漆' in data:
        data = data.replace('油漆', '')
    if '打包' in data:
        data = data.replace('打包', '')
    if '钣金' in data:
        data = data.replace('钣金', '')
    if '塑喷' in data:
        data = data.replace('塑喷', '')
    if '更换' in data:
        data = data.replace('更换', '')
    if '校修' in data:
        data = data.replace('校修', '')
    if '半喷' in data:
        data = data.replace('打包', '')
    if '切割' in data:
        data = data.replace('切割', '')
    if '焊接' in data:
        data = data.replace('焊接', '')
    if '做漆' in data:
        data = data.replace('做漆', '')
    if '翻新' in data:
        data = data.replace('翻新', '')
    if '打包' in data:
        data = data.replace('打包', '')
    if '处理' in data:
        data = data.replace('处理', '')
    if '半漆' in data:
        data = data.replace('半漆', '')
    if '烤漆' in data:
        data = data.replace('烤漆', '')
    if '塑修' in data:
        data = data.replace('塑修', '')
    if '矫正' in data:
        data = data.replace('矫正', '')
    if '钣喷' in data:
        data = data.replace('钣喷', '')
    if '喷底漆' in data:
        data = data.replace('喷底漆', '')
    if '工时' in data:
        data = data.replace('工时', '')
    if '维修' in data:
        data = data.replace('维修', '')
    if '无法点选' in data:
        data = data.replace('无法点选', '')
    if '镀晶' in data:
        data = data.replace('镀晶', '')
    if '整形' in data:
        data = data.replace('整形', '')
    if '更换' in data:
        data = data.replace('更换', '')
    if '调校' in data:
        data = data.replace('调校', '')
    if '补漆' in data:
        data = data.replace('补漆', '')
    if '半喷' in data:
        data = data.replace('半喷', '')
    if '+' in data:
        data = data.replace('+', '')
    if ':' in data:
        data = data.replace(':', '')
    if '*' in data:
        data = data.replace('*', '')
    if '。' in data:
        data = data.replace('。', '')
    if '.' in data:
        data = data.replace('.', '')
    if '-' in data:
        data = data.replace('-', '')
    if ',' in data:
        data = data.replace(',', '')
    if '，' in data:
        data = data.replace('，', '')
    if data.strip() == '车门':
        data = data.replace('车', '')
    if '翼子板' in data:
        if '后' in data:
            # data = data.replace('翼子板', '叶子板')
            data = '叶子板(后)'
        else:
            data = '叶子板(前 )'
    if '拆' in data:
        data = data.replace('拆', '')
    if '做漆' in data:
        data = data.replace('做漆', '')
    if '本体' in data:
        data = data.replace('本体', '')
    if '补损' in data:
        data = data.replace('补损', '')
    if '碰花' in data:
        data = data.replace('碰花', '')
    if '半' in data:
        data = data.replace('半', '')
    if '现场' in data:
        data = data.replace('现场', '')
    if '部' in data:
        if not '前' in data and not '后' in data and not '中' in data:
            data = data.replace('部', '')
    if '银色' in data:
        data = data.replace('银色', '')
    if '护杠' in data:
        data = '护杠' + '\n'
    if '校正' in data:
        data = data.replace('校正', '')
    if '补充' in data:
        data = data.replace('补充', '')
    if '镀络' in data:
        data = data.replace('镀络', '')
    if '喷绘' in data:
        data = data.replace('喷绘', '')
    if '侧侧' in data:
        data = data.replace('侧侧', '侧')
    if '图喷' in data:
        data = data.replace('图喷', '')
    if '喷字' in data:
        data = data.replace('喷字', '')
    if '防锈漆' in data:
        data = data.replace('防锈漆', '')
    if '外修' in data:
        data = data.replace('外修', '')
    if '喷素' in data:
        data = data.replace('喷素', '')
    if '所有' in data:
        data = data.replace('所有', '')
    if '漆' in data:
        data = data.replace('漆', '')
    if '你' in data:
        data = data.replace('你', '')
    if '金额' in data:
        data = data.replace('金额', '')
    if '段' in data:
        data = data.replace('段', '')
    if '差' in data:
        if not '差速器' in data:
            data = '无'
    if '费' in data:
        data = '无'
    if '工时' in data:
        data = '无'
    if '自定义' in data:
        data = '无'
    if '标准' in data:
        data = '无'
    data = data.replace(' ', '')
    for i in range(10):
        data = data.replace(str(i), '')
    if len(data) == 0:
        data ='无'
    if '及' in data:
        data = '无'
    if '抛光' in data:
        data = '无'
    if '，' in data:
        data = '无'
    if '追加' in data:
        data = '无'
    if '、' in data:
        data = '无'
    if '工时' in data:
        data = '无'
    if '其' in data:
        data = '无'
    if '和' in data:
        data = '无'
    if '事故' in data:
        data = '无'
    if '材料' in data:
        data = '无'
    if '三者' in data:
        data = '无'
    if '定损' in data:
        data = '无'
    if '缺额' in data:
        data = '无'
    if '含' in data:
        data = '无'
    if '补' in data:
        data = '无'
    if '跟单' in data:
        data = '无'
    if '增加' in data:
        data = '无'
    if '整案' in data:
        data = '无'

    if '喷漆' in data:
        data = data.replace('喷漆', '')
    if '喷塑' in data:
        data = data.replace('喷塑', '')
    if '修复' in data:
        data = data.replace('修复', '')
    if '拆装' in data:
        data = data.replace('拆装', '')
    if '油漆' in data:
        data = data.replace('油漆', '')
    if '打包' in data:
        data = data.replace('打包', '')
    if '钣金' in data:
        data = data.replace('钣金', '')
    if '塑喷' in data:
        data = data.replace('塑喷', '')
    if '更换' in data:
        data = data.replace('更换', '')
    if '校修' in data:
        data = data.replace('校修', '')
    if '半喷' in data:
        data = data.replace('半喷', '')
    if '切割' in data:
        data = data.replace('切割', '')
    if '焊接' in data:
        data = data.replace('焊接', '')
    if '做漆' in data:
        data = data.replace('做漆', '')
    if '翻新' in data:
        data = data.replace('翻新', '')
    if '打包' in data:
        data = data.replace('打包', '')
    if '半漆' in data:
        data = data.replace('半漆', '')
    if '烤漆' in data:
        data = data.replace('烤漆', '')
    if '塑修' in data:
        data = data.replace('塑修', '')
    if '矫正' in data:
        data = data.replace('矫正', '')
    if '钣喷' in data:
        data = data.replace('钣喷', '')
    if '喷底漆' in data:
        data = data.replace('喷底漆', '')
    if '维修' in data:
        data = data.replace('维修', '')
    if '镀晶' in data:
        data = data.replace('镀晶', '')
    if '整形' in data:
        data = data.replace('整形', '')
    if '调校' in data:
        data = data.replace('调校', '')
    if '补漆' in data:
        data = data.replace('补漆', '')
    if '半喷' in data:
        data = data.replace('半喷', '')

    if data.strip() == '总成':
        data = '无'
    if '杠' in data:
        if '保险杠' in data:
            if '皮' in data:
                data = '保险杠外皮'
            elif '保险杠骨架' in data:
                data = '保险杠骨架'
            elif '饰板' in data:
                data = '保险杠饰板'
            else:
                if '眉' not in data:
                    data = '保险杠'
        elif '杠包角' in data:
            data = '保险杠包角'
        elif '饰板' in data:
            data = '保险杠饰板'
        else:
            if '眉' not in data:
                data = '保险杠'
    if '裙' in data:
        if not '杠' in data:
            data = '底大边'
    if data.strip() in [
        '门门',
        '三者车',
        '侧侧',
        '喷绘',
        '镀络',
        '材料',
        '事故',
        '全车漆车顶是全景天窗',
        '总成',
        '跟单定损',
        '三者出租车叶喷字',
        '缺额',
        '市场监管徽标',
        '配件录入',
        '门饰条叶饰条',
        '门饰板   轮眉',
        '特殊理赔政策',
        '位'
    ]:
        data = '无'
    return data

#定损项目转换函数
def convertxiangmu(Datas):
    datas = pd.read_excel('file\要替换的配件名称3.31.xlsx')
    fit_name = datas['fit_name']
    fit_name0 = datas['fit_name3']
    DIc = {}
    L = []
    for i in range(len(fit_name0)):
        try:
            DIc[qx(fit_name[i])] = fit_name0[i]
        except:
            pass
    for i in range(len(Datas)):
        Datas[i] = Datas[i].upper()
        try:
            a = DIc[qx(Datas[i])]
            if '裙' in a:
                if not '杠' in a:
                    a = '底大边'
            if '叶子板' == a:
                a = '叶子板(前)'
            L.append(a.upper())#qxxm(
        except:
            # try:
                a = qxxm(Datas[i])
                if '叶子板' == a:
                    a = '叶子板(前)'
                L.append(a.upper())
            # except:
            #     L.append('无')
    return L



#转换国别
def convertguobie(data):
    if not '中国' in data:
        data = '进口'
    return data

"""
输入：
Dsdh：定损单号的列表
Dsxmmc：未转换前的定损项目名称列表
输出：
含补差价的整案定损单号列表
"""
def buchajia(Dsdh,Dsxmmc):
    L2 = []
    for i in range(len(Dsdh)):
        if '差' in Dsxmmc[i] or '增加' in Dsxmmc[i] or '增补' in Dsxmmc[i] or '补偿' in Dsxmmc[i]:
            if not '差速' in Dsxmmc[i]:
                L2.append(Dsdh[i])
    L2 = list(set(L2))
    return L2

"""
输入：
Dsdh：定损单号的列表
Buchalist：含补差价的整案定损单号列表
输出：
定损单号的列表中每个项目是否为补差案件
"""
def sfbc(Dsdh,Buchalist):
    L = []
    for i in range(len(Dsdh)):
        if Dsdh[i] in Buchalist:
            L.append(1)
        else:
            L.append(0)
    return L

def isornot_fitting_barbarism(data):
    if '差' in data:
        if not '差速器' in data:
            data = '无'
    if '费' in data:
        data = '无'
    if '工时' in data:
        data = '无'
    if '自定义' in data:
        data = '无'
    if '标准' in data:
        data = '无'
    data = data.replace(' ', '')
    for i in range(10):
        data = data.replace(str(i), '')
    if len(data) == 0:
        data = '无'
    if '及' in data:
        data = '无'
    if '抛光' in data:
        data = '无'
    if '，' in data:
        data = '无'
    if '追加' in data:
        data = '无'
    if '、' in data:
        data = '无'
    if '工时' in data:
        data = '无'
    if '其' in data:
        data = '无'
    if '和' in data:
        data = '无'
    if '事故' in data:
        data = '无'
    if '材料' in data:
        data = '无'
    if '三者' in data:
        data = '无'
    if '定损' in data:
        data = '无'
    if '缺额' in data:
        data = '无'
    if '含' in data:
        data = '无'
    if '补' in data:
        data = '无'
    if '跟单' in data:
        data = '无'
    if '增加' in data:
        data = '无'
    if '整案' in data:
        data = '无'
    if data == '无':
        return True
    else:
        return False

"""
输入:
Changpai:转换后的厂牌列表
Chexi：转换后的车系列表
输出：
该案件是否厂牌车系录入不规范
"""
def is_brand_invalid(Changpai,Chexi):
    L = []
    for i in range(len(Chexi)):
        if Changpai[i] == '无' or Chexi[i] == '无' or '货车' in Changpai[i] or '摩托' in Changpai[i]:
            L.append(1)
        else:
            L.append(0)
    return L

def classyfichexi(datas1):
    datas2 = pd.read_excel('file\分类结果v2.xlsx',encoding='gbk')
    pp1 = datas1['厂牌']
    cx1 = datas1['车系']
    pp2 = datas2['brand']
    cx2 = datas2['auto_series_chinaname']
    x1 = datas2['品牌分类']
    x2 = datas2['车系调整分类']
    dict1 = {}#品牌
    dict2 = {}#车系
    for i in range(len(pp2)):
        dict1[pp2[i]] = x1[i]
        dict2[pp2[i] + '#' + cx2[i]] = x2[i]
    L1 = []#分类
    # L2 = []#新车系
    for i in range(len(pp1)):
        try:
            L1.append(str(dict2[pp1[i] + '#' + cx1[i]])+cx1[1])
            # L2.append(cx1[i])
        except:
            try:
                L1.append(str(dict1[pp1[i]])+cx1[i])
                # L2.append(pp1[i])
            except:
                # L1.append(0)
                L1.append('空')
                # L2.append(pp1[i])
    return L1

def get_mean(changpai,values):
    Dict = {}
    Ch = list(set(changpai))
    Ch_dict = {}
    c = 0
    for i in range(len(Ch)):
        Ch_dict[Ch[i]] = c
        c += 1
    num = np.ones(len(Ch))
    for i in range(len(changpai)):
        if values[i] > 0:
            if changpai[i] not in Dict:
                Dict[changpai[i]] = values[i]
            else:
                Dict[changpai[i]] += values[i]
                num[Ch_dict[changpai[i]]] += 1
        else:
            Dict[changpai[i]] = 0
    for d in Dict:
        try:
            Dict[d] /= num[Ch_dict[d]]
        except:
            print(Dict[d],num[Ch_dict[d]])
    return Dict

"""
统计频次函数
"""
def all_list(arr):
    result = {}
    for data in arr:
        if not data in result:
            result[data] = 1
        else:
            result[data] += 1
    return result

#————————获取生成喷漆均值————————————
def get_mean_pengqi(changpai, values, pqlx):
    Dict = {}
    Ch = list(set(changpai))
    Ch_dict = {}
    c = 0
    for i in range(len(Ch)):
        Ch_dict[Ch[i]] = c
        c += 1
    num = np.ones(len(Ch))
    for i in range(len(changpai)):
        if '全漆' in pqlx[i]:
            if values[i] < 999999:
                if changpai[i] not in Dict:
                    Dict[changpai[i]] = values[i]
                else:
                    Dict[changpai[i]] += values[i]
                    num[Ch_dict[changpai[i]]] += 1
    for d in Dict:
        Dict[d] /= num[Ch_dict[d]]
    v = []
    for d in changpai:
        try:
            v.append(round(Dict[d] / 100) * 100)
        except KeyError:
            v.append(-1)
    return v

"""
获取喷漆众数
"""
def get_zhengshu_pengqi(changpai, values,pqlx):
    Ch = list(set(changpai))
    Ch_dict = {}
    def get_num(L):
        x = dict((a, L.count(a)) for a in L)
        y = [k for k, v in x.items() if max(x.values()) == v]
        return np.mean(np.array(y))
    for cls in Ch:
        num = []
        for i in range(len(changpai)):
            if '全漆' in pqlx[i]:
                if cls == changpai[i] and values[i] < 999999:
                    num.append(float(values[i]))
        if num != []:
            Ch_dict[cls] = get_num(num)
    v = []
    for d in changpai:
        try:
            v.append(round(Ch_dict[d] / 100) * 100)
        except:
            v.append(-1)
    return v

"""
获取拆装费的均值
"""
def get_mean_caizhaung(changpai,values):
    Dict = {}
    Ch = list(set(changpai))
    Ch_dict = {}
    c = 0
    for i in range(len(Ch)):
        Ch_dict[Ch[i]] = c
        c += 1
    num = np.ones(len(Ch))
    for i in range(len(changpai)):
        if values[i] < 1000:
            if changpai[i] not in Dict:
                Dict[changpai[i]] = values[i]
            else:
                Dict[changpai[i]] += values[i]
                num[Ch_dict[changpai[i]]] += 1
    for d in Dict:
        Dict[d] /= num[Ch_dict[d]]
    v = []
    for d in changpai:
        try:
            v.append(round(Dict[d] / 10) * 10)
        except KeyError:
            v.append(-1)
    return v

"""
获取拆装费的众数
"""
def get_zhengshu_chaizhaung(changpai, values):
    Ch = list(set(changpai))
    Ch_dict = {}
    def get_num(L):
        x = dict((a, L.count(a)) for a in L)
        y = [k for k, v in x.items() if max(x.values()) == v]
        return np.mean(np.array(y))
    for cls in Ch:
        num = []
        for i in range(len(changpai)):
            if cls == changpai[i] and values[i] < 1000:
                num.append(values[i])
        if num != []:
            Ch_dict[cls] = get_num(num)
    v = []
    for d in changpai:
        try:
            v.append(round(Ch_dict[d] / 10) * 10)
        except KeyError:
            v.append(-1)
    return v

"""
获取维修费的均值
"""
def get_mean_weixiu(changpai,values):
    Dict = {}
    Ch = list(set(changpai))
    Ch_dict = {}
    c = 0
    for i in range(len(Ch)):
        Ch_dict[Ch[i]] = c
        c += 1
    num = np.ones(len(Ch))
    for i in range(len(changpai)):
            if values[i] < 10000:
                if changpai[i] not in Dict:
                    Dict[changpai[i]] = values[i]
                else:
                    Dict[changpai[i]] += values[i]
                    num[Ch_dict[changpai[i]]] += 1
    for d in Dict:
        Dict[d] /= num[Ch_dict[d]]
    v = []
    for d in changpai:
        try:
            v.append(round(Dict[d] / 10) * 10)
        except KeyError:
            v.append(-1)
    return v

"""
获取维修费的众数
"""
def get_zhengshu_weixiu(changpai, values):
    Ch = list(set(changpai))
    Ch_dict = {}
    def get_num(L):
        x = dict((a, L.count(a)) for a in L)
        y = [k for k, v in x.items() if max(x.values()) == v]
        return np.mean(np.array(y))
    for cls in Ch:
        num = []
        for i in range(len(changpai)):
                if cls == changpai[i] and values[i] < 10000:
                    num.append(float(values[i]))
        if num != []:
            Ch_dict[cls] = get_num(num)
    v = []
    for d in changpai:
        try:
            v.append(round(Ch_dict[d] / 10) * 10)
        except:
            v.append(-1)
    return v

"""
获取各个省的品牌价格字典
"""
def get_changpai_price(sheng):
    datas1 = pd.read_excel('file\各机构品牌标准价格.xlsx')
    n = datas1['dptname']
    x = datas1['brand_name']
    y = datas1['avg_total_bz']
    Dict1 = {}
    for i in range(len(x)):
        if sheng in n[i]:
            Dict1[x[i]] = y[i]
    return Dict1

"""
获取各个省的车系价格字典
"""
def get_chexi_price(sheng):
    datas1 = pd.read_excel('file\各机构车系标准价格.xlsx')
    n = datas1['dptname']
    x1 = datas1['brand_name']
    x2 = datas1['auto_series_chinaname']
    y = datas1['avg_total_dpt_bz']
    Dict2 = {}
    for i in range(len(x1)):
        if sheng in n[i]:
            Dict2[x1[i] + x2[i]] = y[i]
    return Dict2