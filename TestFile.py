import pandas as pd
import csv
import numpy as np
import os
import json
from utils import convertchangpai
from utils import convertchexi
from utils import convertxiangmu
from utils import convertguobie
from utils import buchajia
from utils import sfbc
from utils import isornot_fitting_barbarism
from utils import classyfichexi
from utils import all_list
from utils import get_mean_pengqi
from utils import get_mean_caizhaung
from utils import get_mean_weixiu
from utils import get_changpai_price
from utils import get_chexi_price
import xgboost as xgb
from math import *

"""
生成测试文件类
传入参数：
task：任务类型，可选参数：（喷漆、拆装、维修）
sheng：机构省份名，如：四川、陕西等
starttime：起始时间如（2019-04、2018-12、2020-02）等
endtime：截至时间
方法名及功能：
get_fac_palce：获取修理厂编号对应的地级市填充到fac_place
all_list1：统计两个关联列表的频次返回一个字典
create_edition_one：生成第一版文件（给各个条目打上（规范或者不规范）标记）
create_edition_two：生成第二版文件（加上跟单，删除不规范的条目，标记车的价格类别）
get_mean1cp：获取厂牌的均值
get_meancx：获取车系的均值
create_edition_three：生成第三版文件（加载均值众数等）
create_edition_four：生成第四版文件（加入F值）
make：生成最终测试文件

例子：
testfile = TestFile(task='喷漆',sheng='四川',starttime='2020-03',endtime='2020-04')#创建一个测试文件对象
testfile.make()#生成测试文件
testfile.eval()#进行测试
"""
class TestFile():
    def __init__(self,task,sheng,starttime,endtime):
        if not os.path.exists('testfile'):
            os.mkdir('testfile')
        if not os.path.exists('testfile\{}'.format(task)):
            os.mkdir('testfile\{}'.format(task))
        if not os.path.exists('testfile\{}\{}'.format(task,sheng)):
            os.mkdir('testfile\{}\{}'.format(task,sheng))
        self.trainfile = '201811_to_202004_all.csv'
        self.outputfile = 'testfile\{}\{}\{}预测数据.csv'.format(task,sheng,sheng + task)
        self.sheng = sheng
        self.task = task
        self.fac_place = {}
        self.fac_city = {}
        if not os.path.exists(self.outputfile):
            self.dataset = pd.read_csv(self.trainfile)
            self.get_fac_palce()
            self.dataset = self.dataset.loc[(self.dataset['关联机构'] == '{}分公司'.format(sheng)) & (self.dataset['末核损通过时间'] >= starttime) & (self.dataset['末核损通过时间'] < endtime)]
            self.dataset['修理厂编码'] = self.dataset['修理厂编码'].fillna('空')
            self.dataset['车系'] = self.dataset['车系'].fillna('空')
            self.dataset['厂牌'] = self.dataset['厂牌'].fillna('空')
            self.dataset['定损项目名称'] = self.dataset['定损项目名称'].fillna('空')
            self.dataset['工时折扣率'] = self.dataset['工时折扣率'].fillna(0)
            self.dataset['合作类型'] = self.dataset['合作类型'].fillna('无')
            if task == '喷漆':
                self.dataset = self.dataset.loc[self.dataset['折后喷漆费'] > 0]
                self.dataset['喷漆类型'] = self.dataset['喷漆类型'].fillna('空')
            elif task == '拆装':
                self.dataset = self.dataset.loc[self.dataset['折后拆装费'] > 0]
            elif task == '维修':
                self.dataset = self.dataset.loc[self.dataset['折后维修费'] > 0]
                self.dataset['折后拆装费'] = self.dataset['折后拆装费'].fillna(0)
                self.dataset['维修程度'] = self.dataset['维修程度'].fillna('空')
                self.dataset['配件外修费'] = self.dataset['配件外修费'].fillna(0)
            else:
                print('task error!')
                exit()

    #————————————————构建特征列表——————————————
            if task == '维修':
                self.Pjwxf = []
                for data in self.dataset['配件外修费']:
                    self.Pjwxf.append(data)
                self.Wxcd = []
                for data in self.dataset['维修程度']:
                    self.Wxcd.append(data)
            self.Dsdh = []
            for data in self.dataset['定损单号']:
                self.Dsdh.append(data)
            self.Dsxmmc = []
            for data in self.dataset['定损项目名称']:
                self.Dsxmmc.append(data)
            self.Dingsunxiangmu = convertxiangmu(self.Dsxmmc)
            self.Buchalist = buchajia(self.Dsdh, self.Dsxmmc)
            self.Sfbc = sfbc(self.Dsdh, self.Buchalist)
            self.Hzlx = []
            for data in self.dataset['合作类型']:
                self.Hzlx.append(data)
            self.Xlcbm = []
            for data in self.dataset['修理厂编码']:
                self.Xlcbm.append(data)
            self.Xlcmc = []
            for data in self.dataset['修理厂名称']:
                self.Xlcmc.append(data)
            self.Czlx = []
            for data in self.dataset['操作类型']:
                self.Czlx.append(data)
            if task == '喷漆':
                self.Pqlx = []
                for data in self.dataset['喷漆类型']:
                    self.Pqlx.append(data)
            self.Gb = []
            for data in self.dataset['国别']:
                self.Gb.append(convertguobie(data))
            self.Xlclx = []
            for data in self.dataset['修理厂类型']:
                self.Xlclx.append(data)
            self.Gsdjlx = []
            for data in self.dataset['工时单价类型']:
                self.Gsdjlx.append(data)
            self.Sfcxcp = []
            for data in self.dataset['是否承修厂牌']:
                self.Sfcxcp.append(data)
            self.Zhpqf = []
            if task == '喷漆':
                for data in self.dataset['折后喷漆费']:
                    self.Zhpqf.append(float(data))
            elif task == '拆装':
                for data in self.dataset['折后拆装费']:
                    self.Zhpqf.append(float(data))
            elif task == '维修':
                self.Chai = []
                for data in self.dataset['折后拆装费']:
                    self.Chai.append(float(data))
                self.Wei = []
                for data in self.dataset['折后维修费']:
                    self.Wei.append(float(data))
                self.Zhpqf = []
                for i in range(len(self.Chai)):
                    self.Zhpqf.append(self.Wei[i] - self.Chai[i])
            self.Cp = []
            for data in self.dataset['厂牌']:
                self.Cp.append(data)
            self.Cx = []
            for data in self.dataset['车系']:
                self.Cx.append(data)
            self.Gszkl = []
            for data in self.dataset['工时折扣率']:
                self.Gszkl.append(float(data))
            self.Dsygh = []
            for data in self.dataset['定损员工号']:
                self.Dsygh.append(data)
            self.Hsygh = []
            for data in self.dataset['核损员工号']:
                self.Hsygh.append(data)
            self.Hsyxm = []
            for data in self.dataset['核损员姓名']:
                self.Hsyxm.append(data)
            self.Dsyxm = []
            for data in self.dataset['定损员名称']:
                self.Dsyxm.append(data)
            self.Zdy = []
            for data in self.dataset['配件来源']:
                self.Zdy.append(data)

            self.pc_dict = all_list(self.Dingsunxiangmu)#统计转换后的定损项目的频次
            self.Changpai = convertchangpai(self.Cp)#转换厂牌
            self.cp_dict = all_list(self.Changpai)#计算转换后的厂牌的频次
            self.Chexi = convertchexi(self.Changpai, self.Cx)#转换车系
            self.cx_dict = self.all_list1(self.Changpai, self.Chexi)#统计车系的频次

            self.A = []  # 机构、品牌、车系、工时价格类型，工时项目
            self.B = []  # 品牌、车系、工时价格类型，工时项目
            for i in range(len(self.Changpai)):
                self.A.append(sheng + self.Changpai[i] + self.Chexi[i] + self.Gsdjlx[i] + self.Dingsunxiangmu[i])
                self.B.append(self.Changpai[i] + self.Chexi[i] + self.Gsdjlx[i] + self.Dingsunxiangmu[i])
            if task == '喷漆':
                self.A1 = get_mean_pengqi(self.A, self.Zhpqf, self.Pqlx)  # 机构、品牌、车系、工时价格类型，工时项目的平均价格
                self.B1 = get_mean_pengqi(self.B, self.Zhpqf, self.Pqlx)  # 品牌、车系、工时价格类型，工时项目的平均价格
            elif task == '拆装':
                self.A1 = get_mean_caizhaung(self.A, self.Zhpqf)  # 机构、品牌、车系、工时价格类型，工时项目的平均价格
                self.B1 = get_mean_caizhaung(self.B, self.Zhpqf)  # 品牌、车系、工时价格类型，工时项目的平均价格
            elif task == '维修':
                self.A1 = get_mean_weixiu(self.A, self.Zhpqf)  # 机构、品牌、车系、工时价格类型，工时项目的平均价格
                self.B1 = get_mean_weixiu(self.B, self.Zhpqf)  # 品牌、车系、工时价格类型，工时项目的平均价格
            self.A2 = all_list(self.A)  # 机构、品牌、车系、工时价格类型，工时项目的平均数量
            self.B2 = all_list(self.B)  # 品牌、车系、工时价格类型，工时项目的平均数量

            self.Dict_duty = {}
            with open('file\查勘责任比例.txt') as f:
                datas = f.readlines()
                for data in datas:
                    data = data.strip()
                    data = data.split('|')
                    self.Dict_duty[data[0]] = data[-1]
            self.LS1 = ['杠', '杆', '轮', '叶', '翼', '灯', '盖', '门', '钢圈', 'A', 'B', 'C']

    #—————字段名称列表————————
            self.L1 = []#定损单号
            self.L2 = []#原始定损项目名称
            self.L3 = []#定损项目名称
            self.L4 = []#合作类型
            self.L5 = []#修理厂编码
            self.L6 = []#操作类型
            self.L7 = []#喷漆类型
            self.L9 = []#厂牌
            self.L10 = []#车系
            self.L11 = []#修理厂类
            self.L12 = []#工时单价类型
            self.L13 = []#是否承修厂牌
            self.L14 = []#折后喷漆费
            self.L15 = []#国别
            self.L16 = []#修理厂地址
            self.L17 = []#修理厂名称
            self.L18 = []#工时折扣率
            self.L19 = []#除以工时折扣率的喷漆费
            self.L20 = []#维修程度
            self.L21 = []#维修费
            self.L22 = []#拆装费
            self.L23 = []#定损员工号
            self.L24 = []#核损员工号
            self.L25 = []#核损员姓名
            self.L26 = []#定损员姓名
            self.L27 = []#是否含有补差价
            self.L28 = []#是否自定义
            self.L29 = []#定损项目不规范
            self.L30 = []#品牌车系录入不规范
            self.L31 = []#训练数据量不足
            self.L32 = []#定损项目金额过低
            self.L33 = []#是否单个项目补差
            self.L34 = []#责任标记
            self.L35 = []#定损项目金额过高
            self.L36 = []#外修费
            self.L37 = []  # 定损价格排序过高过低的标记，跟单标记

    #——————填充各个字段——————
            for i in range(len(self.Dingsunxiangmu)):
                if self.Gszkl[i] > 0 and self.Zhpqf[i] > 0:
                    if task == '维修':
                        if self.Pjwxf[i] > 0:
                            self.L36.append(1)
                        else:
                            self.L36.append(0)
                    if '差' in self.Dsxmmc[i]:
                        if not '差速器' in self.Dsxmmc[i]:
                            self.L35.append(1)
                        else:
                            self.L35.append(0)
                    else:
                        self.L35.append(0)
                    data = self.Dsdh[i].split('-')
                    try:
                        if self.Dict_duty[data[0]] == '同责':
                            if data[1] == '0202':
                                self.L34.append(1)
                            else:
                                self.L34.append(0)
                        elif self.Dict_duty[data[0]] == '次责':
                            self.L34.append(1)
                        else:
                            self.L34.append(0)
                    except:
                        self.L34.append(0)
                    if '自定义' in self.Zdy[i]:
                        self.L33.append(1)
                    else:
                        self.L33.append(0)
                    if self.Changpai[i] == '无' or self.Chexi[i] == '无' or '货车' in self.Changpai[i] or '摩托' in self.Changpai[i]:
                        self.L29.append(1)
                    else:
                        self.L29.append(0)
                    if self.cp_dict[self.Changpai[i]] < 10 or self.pc_dict[self.Dingsunxiangmu[i]] <= 2 or self.cx_dict[self.Changpai[i] + self.Chexi[i]] < 5:
                        self.L30.append(1)
                    else:
                        self.L30.append(0)
                    try:
                        o = self.fac_place[self.Xlcbm[i]]
                    except:
                        o = '无'
                    p = self.Xlcmc[i]
                    self.L1.append(self.Dsdh[i])
                    self.L2.append(self.Dsxmmc[i])
                    c = 0
                    for s in self.LS1:
                        if s in self.Dsxmmc[i]:
                            c += 1
                            if c >= 2:
                                break
                    if c >= 2:
                        self.L28.append(1)
                    else:
                        if isornot_fitting_barbarism(self.Dsxmmc[i]):
                            self.L28.append(1)
                        else:
                            self.L28.append(0)
                    self.L3.append(self.Dingsunxiangmu[i])
                    self.L4.append(self.Hzlx[i])
                    self.L5.append(self.Xlcbm[i])  # 不作为训练依据
                    if task == '喷漆':
                        self.L6.append(self.Czlx[i])
                        self.L7.append(self.Pqlx[i])
                    self.L15.append(self.Gb[i])
                    self.L9.append(self.Changpai[i])
                    self.L10.append(self.Chexi[i])
                    if self.Xlclx[i] == '4S店':
                        self.L11.append('4S店')
                    else:
                        self.L11.append('综合修理厂')
                    if self.Gsdjlx[i] == '4S店':
                        self.L12.append('4S店')
                    else:
                        self.L12.append('综合修理厂')
                    self.L13.append(self.Sfcxcp[i])
                    self.L14.append(self.Zhpqf[i])
                    self.L16.append(o)
                    self.L17.append(p)
                    self.L18.append(self.Gszkl[i])
                    self.L19.append(float(self.Zhpqf[i]))  # / float(Gszkl[i]) * 100
                    if task == '喷漆':
                        if float(self.Zhpqf[i]) < 10:  # / float(Gszkl[i]) * 100
                            self.L31.append(1)
                        else:
                            self.L31.append(0)
                        if float(self.Zhpqf[i]) >= 999999:  # / float(Gszkl[i]) * 100
                            self.L32.append(1)
                        else:
                            self.L32.append(0)
                    elif task == '拆装':
                        if float(self.Zhpqf[i]) < 5:  # / float(Gszkl[i]) * 100
                            self.L31.append(1)
                        else:
                            self.L31.append(0)
                        if float(self.Zhpqf[i]) >= 1000:  # / float(Gszkl[i]) * 100
                            self.L32.append(1)
                        else:
                            self.L32.append(0)
                    elif task == '维修':
                        self.L20.append(self.Wxcd[i])
                        self.L21.append(self.Wei[i])
                        self.L22.append(self.Chai[i])
                        if float(self.Zhpqf[i]) < 5:  # / float(Gszkl[i]) * 100
                            self.L31.append(1)
                        else:
                            self.L31.append(0)
                        if float(self.Zhpqf[i]) >= 10000:  # / float(Gszkl[i]) * 100
                            self.L32.append(1)
                        else:
                            self.L32.append(0)
                    self.L23.append(self.Dsygh[i])
                    self.L24.append(self.Hsygh[i])
                    self.L25.append(self.Hsyxm[i])
                    self.L26.append(self.Dsyxm[i])
                    self.L27.append(self.Sfbc[i])
                    try:
                        if self.A2[sheng + self.Changpai[i] + self.Chexi[i] + self.Gsdjlx[i] + self.Dingsunxiangmu[
                            i]] >= 5 and float(self.Zhpqf[i]) / self.A1[i] >= 3:
                            self.L37.append(1)
                        elif self.A1[i] / float(self.Zhpqf[i]) >= 3:
                            self.L37.append(1)
                        elif float(self.Zhpqf[i]) / self.B1[i] >= 2 or self.B1[i] / float(self.Zhpqf[i]) >= 3:
                            self.L37.append(1)
                        else:
                            self.L37.append(0)
                    except:
                        self.L37.append(1)


#——————获取修理厂编号对应的地级市——————
    def get_fac_palce(self):
        datas = pd.read_excel('file\全量修理厂清单.xlsx')
        code_factory = datas['修理厂代码']
        address_factory = datas['地级市']
        for i in range(len(code_factory)):
            self.fac_place[code_factory[i]] = address_factory[i]

# ——————获取一个列表的个数字典（两个参数）————————
    def all_list1(self,arr1, arr2):
        result = {}
        for i in range(len(arr1)):
            if not arr1[i] + arr2[i] in result:
                result[arr1[i] + arr2[i]] = 1
            else:
                result[arr1[i] + arr2[i]] += 1
        return result

#——————生成第一版文件————————
    def create_edition_one(self):
        outputs = open(self.outputfile, 'w', newline='')
        csv_writer = csv.writer(outputs, dialect='excel')
        if self.task == '喷漆':
            csv_writer.writerow(['定损单号', '原始定损项目名称', '定损项目名称', '合作类型', '操作类型', '喷漆类型', '国别', '厂牌', '车系', '修理厂类型', \
                                 '工时单价类型', '是否承修厂牌', '除以工时折扣率的喷漆费', '折后喷漆费', '修理厂编码', '修理厂地址', '修理厂名称', '工时折扣率', '定损员工号',
                                 '定损员姓名', '核损员工号', '核损员姓名', '是否含有补差价', '是否单个项目补差', '是否自定义', '定损项目不规范', '品牌车系录入不规范',
                                 '训练数据量不足', '定损项目金额过低', '定损项目金额过高', '跟单标记', '定损价格排序过高过低'])
        elif self.task == '拆装':
            csv_writer.writerow(['定损单号', '原始定损项目名称', '定损项目名称', '合作类型', '国别', '厂牌', '车系', '修理厂类型', \
                                 '工时单价类型', '是否承修厂牌', '折前拆装费', '折后拆装费', '修理厂编码', '修理厂地址', '修理厂名称', '工时折扣率', '定损员工号',
                                 '定损员姓名', '核损员工号', '核损员姓名', '是否含有补差价', '是否单个项目补差', '是否自定义', '定损项目不规范', '品牌车系录入不规范',
                                 '训练数据量不足', '定损项目金额过低', '定损项目金额过高', '跟单标记', '定损价格排序过高过低'])
        elif self.task == '维修':
            csv_writer.writerow(['定损单号', '原始定损项目名称', '定损项目名称', '合作类型', '国别', '厂牌', '车系', '修理厂类型', \
                                 '工时单价类型', '是否承修厂牌', '维修程度', '折扣前的费用', '折后维修费减去折后拆装费', '修理厂编码', \
                                 '修理厂地址', '修理厂名称', '折后维修费', '折后拆装费', '工时折扣率', '定损员工号', '定损员姓名', '核损员工号', '核损员姓名',
                                 '是否含有补差价', '是否单个项目补差', '是否自定义', '定损项目不规范', '品牌车系录入不规范', '训练数据量不足', '定损项目金额过低',
                                 '定损项目金额过高', '跟单标记', '是否含有配件外修费','定损价格排序过高过低'])

        for i in range(len(self.L2)):
            try:
                data = []
                data.append(self.L1[i])
                data.append(self.L2[i])
                data.append(self.L3[i])
                data.append(self.L4[i])
                if self.task == '喷漆':
                    data.append(self.L6[i])
                    data.append(self.L7[i])
                data.append(self.L15[i])
                data.append(self.L9[i])
                data.append(self.L10[i])
                data.append(self.L11[i])
                data.append(self.L12[i])
                data.append(self.L13[i])
                if self.task == '维修':
                    data.append(self.L20[i])
                data.append(self.L19[i])
                data.append(self.L14[i])
                data.append(self.L5[i])
                data.append(self.L16[i])
                data.append(self.L17[i])
                if self.task == '维修':
                    data.append(self.L21[i])
                    data.append(self.L22[i])
                data.append(self.L18[i])
                data.append(self.L23[i])
                data.append(self.L26[i])
                data.append(self.L24[i])
                data.append(self.L25[i])
                data.append(self.L27[i])
                data.append(self.L35[i])
                data.append(self.L33[i])
                data.append(self.L28[i])
                data.append(self.L29[i])
                data.append(self.L30[i])
                data.append(self.L31[i])
                data.append(self.L32[i])
                data.append(self.L34[i])
                if self.task == '维修':
                    data.append(self.L36[i])
                data.append(self.L37[i])
                csv_writer.writerow(data)
            except UnicodeEncodeError:
                print(data)

# ——————生成第二版文件————————
    def create_edition_two(self):
#—————标记跟单案件———————————
        Dict_duty = {}
        filename = self.outputfile
        with open('file\查勘责任比例.txt') as f:
            datas = f.readlines()
            for data in datas:
                data = data.strip()
                data = data.split('|')
                Dict_duty[data[0]] = data[-1]
        Anhui = pd.read_csv(filename,encoding='gbk')
        dsdh = Anhui['定损单号']
        L = []
        for data in dsdh:
            data = data.split('-')
            try:
                if Dict_duty[data[0]] == '同责':
                    if data[1] == '0202':
                        L.append(1)
                    else:
                        L.append(0)
                elif Dict_duty[data[0]] == '次责':
                    L.append(1)
                else:
                    L.append(0)
            except:
                L.append(0)
        Anhui['跟单案件'] = np.c_[L]
        Anhui.to_csv(filename,index=0,encoding='gbk')
#——————根据频次计算价格————————
        i = 0
        Dict = {}
        Dict1 = {}
        filename1 = self.outputfile
        filename2 = 'temp.csv'
        if self.task == '喷漆':
            with open(filename1) as f:
                datas = f.readlines()
                for data in datas:
                    data = data.strip()
                    data = data.split(',')
                    if i > 0:
                        if len(data) == 33:
                            l1 = data[:12] + data[14:]
                            l1 = [x + '#' for x in l1]
                            l2 = ''
                            for x in l1:
                                l2 += x
                            l2 += '#' + str(data[13])
                            l = data[2] + '#' + data[5] + '#' + data[7] + '#' + data[8] + '#' + data[9]
                            if not l in Dict:
                                Dict[l] = 1
                            else:
                                Dict[l] += 1

                            if not l in Dict1:
                                Dict1[l] = [l2]
                            else:
                                Dict1[l].append(l2)
                    i += 1
            outputs = open(filename2, 'w', newline='')
            import csv
            csv_write = csv.writer(outputs, dialect='excel')
            csv_write.writerow(
                ['定损单号', '原始定损项目', '定损项目名称', '合作类型', '操作类型', '喷漆类型', '国别', '厂牌', '车系', '修理厂类型', '工时单价类型', '是否承修厂牌', '修理厂编码',
                 '修理厂地址',
                 '修理厂名称', '工时折扣率', '定损员工号', '定损员姓名', '核损员工号', '核损员姓名', '是否含有补差价', '是否单个项目补差', '是否自定义', '定损项目不规范', '品牌车系录入不规范',
                 '训练数据量不足', '定损项目金额过低', '定损项目金额过高', '跟单标记','定损价格排序过高过低','跟单案件', '不要1', '不要2', '价格', 'n', 'i', 'rank'])
            for key in Dict1:
                L = Dict1[key]
                n = Dict[key]
                D = {}
                for l in L:
                    data = l.split('#')
                    l1 = data[:-1]
                    l1 = [x + '#' for x in l1]
                    k = ''
                    for x in l1:
                        k += x
                    v = float(data[-1])
                    D[k] = v
                D = sorted(D.items(), key=lambda x: x[1])
                i = 1
                for d in D:
                    data = []
                    data.extend(d[0].split('#'))
                    data.append(d[1])
                    data.append(n)
                    data.append(i)
                    data.append(i - n / 2)
                    csv_write.writerow(data)
                    i += 1
        elif self.task == '拆装':
            with open(filename1) as f:
                datas = f.readlines()
                for data in datas:
                    data = data.strip()
                    data = data.split(',')
                    if i > 0:
                        if len(data) == 31:
                            l1 = data[:10] + data[12:]
                            l1 = [x + '#' for x in l1]
                            l2 = ''
                            for x in l1:
                                l2 += x
                            l2 += '#' + str(data[10])
                            l = data[2] + '#' + data[5] + '#' + data[6] + '#' + data[7]
                            if not l in Dict:
                                Dict[l] = 1
                            else:
                                Dict[l] += 1

                            if not l in Dict1:
                                Dict1[l] = [l2]
                            else:
                                Dict1[l].append(l2)
                    i += 1
            import csv
            outputs = open(filename2, 'w', newline='')
            csv_write = csv.writer(outputs, dialect='excel')
            csv_write.writerow(
                ['定损单号', '原始定损项目', '定损项目名称', '合作类型', '国别', '厂牌', '车系', '修理厂类型', '工时单价类型', '是否承修厂牌', '修理厂编码', '修理厂地址',
                 '修理厂名称', '工时折扣率', '定损员工号', '定损员姓名', '核损员工号', '核损员姓名', '是否含有补差价', '是否单个项目补差', '是否自定义', '定损项目不规范', '品牌车系录入不规范',
                 '训练数据量不足', '定损项目金额过低', '定损项目金额过高', '跟单标记','定损价格排序过高过低', '跟单案件', '不要1', '不要2', '价格', 'n', 'i', 'rank'])
            for key in Dict1:
                L = Dict1[key]
                n = Dict[key]
                D = {}
                for l in L:
                    data = l.split('#')
                    l1 = data[:-1]
                    l1 = [x + '#' for x in l1]
                    k = ''
                    for x in l1:
                        k += x
                    v = float(data[-1])
                    D[k] = v
                D = sorted(D.items(), key=lambda x: x[1])
                i = 1
                for d in D:
                    data = []
                    data.extend(d[0].split('#'))
                    data.append(d[1])
                    data.append(n)
                    data.append(i)
                    data.append(i - n / 2)
                    csv_write.writerow(data)
                    i += 1
        elif self.task == '维修':
            with open(filename1) as f:
                datas = f.readlines()
                for data in datas:
                    data = data.strip()
                    data = data.split(',')
                    if i > 0:
                        if len(data) == 35:
                            l1 = data[:11] + data[13:]
                            l1 = [x + '#' for x in l1]
                            l2 = ''
                            for x in l1:
                                l2 += x
                            l2 += '#' + str(data[11])
                            l = data[2] + '#' + data[5] + '#' + data[6] + '#' + data[7]
                            if not l in Dict:
                                Dict[l] = 1
                            else:
                                Dict[l] += 1

                            if not l in Dict1:
                                Dict1[l] = [l2]
                            else:
                                Dict1[l].append(l2)
                    i += 1
            import csv
            outputs = open(filename2, 'w', newline='')
            csv_write = csv.writer(outputs, dialect='excel')
            csv_write.writerow(
                ['定损单号', '原始定损项目', '定损项目名称', '合作类型', '国别', '厂牌', '车系', '修理厂类型', '工时单价类型', '是否承修厂牌', '维修程度', '修理厂编码', '修理厂地址',
                 '修理厂名称', '折后维修费','折后拆装费','工时折扣率', '定损员工号', '定损员姓名', '核损员工号', '核损员姓名', '是否含有补差价', '是否单个项目补差', '是否自定义', '定损项目不规范', '品牌车系录入不规范',
                 '训练数据量不足', '定损项目金额过低', '定损项目金额过高', '跟单标记', '是否含有外修费','定损价格排序过高过低', '跟单案件', '不要1', '不要2', '价格', 'n', 'i', 'rank'])
            for key in Dict1:
                L = Dict1[key]
                n = Dict[key]
                D = {}
                for l in L:
                    data = l.split('#')
                    l1 = data[:-1]
                    l1 = [x + '#' for x in l1]
                    k = ''
                    for x in l1:
                        k += x
                    v = float(data[-1])
                    D[k] = v
                D = sorted(D.items(), key=lambda x: x[1])
                i = 1
                for d in D:
                    data = []
                    data.extend(d[0].split('#'))
                    data.append(d[1])
                    data.append(n)
                    data.append(i)
                    data.append(i - n / 2)
                    csv_write.writerow(data)
                    i += 1
        datas = pd.read_csv(filename2, encoding='gbk')
        datas.drop(['不要1', '不要2'], axis=1, inplace=True)
        datas.to_csv(filename2, encoding='gbk', index=0)
#———————得到标记————————
        datas = pd.read_csv(filename2,encoding='gbk',error_bad_lines=False)
        N = datas['n']
        I = datas['i']
        L = []
        for i in range(len(I)):
            if N[i] * 0.1 >= I[i] or N[i] * 0.9 <= I[i]:
                if I[i] <= 3 or I[i] >= N[i] - 3:
                    if N[i] >= 10:
                        L.append(1)
                    else:
                        L.append(0)
                else:
                    L.append(0)
            else:
                L.append(0)
        datas['标记'] = np.c_[L]
        if self.task == '喷漆':
            datas = datas.drop(columns = ['操作类型'])
#——————获得车系类别——————————————
        class_chexi = classyfichexi(datas)
        datas['车系类别'] = np.c_[class_chexi]
        datas.to_csv(filename1,index=0,encoding='gbk')

# ——————生成第三版文件（加入均值众数）————————
    def get_mean1cp(self,X):
        L = []
        for x in X:
            try:
                L.append(self.Dict1[x])
            except:
                L.append(-1)
        return L

    def get_meancx(self,X, Y):
        L = []
        for i in range(len(X)):
            try:
                L.append(self.Dict2[X[i] + Y[i]])
            except:
                try:
                    L.append(self.Dict1[X[i]])
                except:
                    L.append(-1)
        return L

    def create_edition_three(self):
        filename = self.outputfile
        datas = pd.read_csv(filename, encoding='gbk')
        self.Dict1 = get_changpai_price(self.sheng)
        self.Dict2 = get_chexi_price(self.sheng)

        Dingsunxm = list(datas['定损项目名称'])
        Hezuoleix = list(datas['合作类型'])
        Changpai = list(datas['厂牌'])
        Chexi = list(datas['车系'])
        Xiulichanglx = list(datas['修理厂类型'])
        mean_changpai = self.get_mean1cp(Changpai)
        mean_chexi = self.get_meancx(Changpai, Chexi)
        onehot_path = 'json/{}/{}'.format(self.task,self.sheng)
        with open('{}/项目均值.json'.format(onehot_path)) as f:
            M_xm = json.load(f)
        with open('{}/项目众数.json'.format(onehot_path)) as f:
            Z_xm = json.load(f)
        with open('{}/修理厂均值.json'.format(onehot_path)) as f:
            M_xlc = json.load(f)
        with open('{}/修理厂众数.json'.format(onehot_path)) as f:
            Z_xlc = json.load(f)
        with open('{}/合作类型均值.json'.format(onehot_path)) as f:
            M_hzlx = json.load(f)
        with open('{}/合作类型众数.json'.format(onehot_path)) as f:
            Z_hzlx = json.load(f)
        mean_peijian = []
        mean_hzlx = []
        mean_xiulichang = []
        zhong_hzlx = []
        zhong_peijian = []
        zhong_xiulichang = []
        for i in range(len(Dingsunxm)):
            try:
                mean_peijian.append(M_xm[Dingsunxm[i]])
            except:
                mean_peijian.append(0)
            try:
                mean_xiulichang.append(M_xlc[Xiulichanglx[i]])
            except:
                mean_xiulichang.append(0)
            try:
                mean_hzlx.append(M_hzlx[Hezuoleix[i]])
            except:
                mean_hzlx.append(0)
            try:
                zhong_peijian.append(Z_xm[Dingsunxm[i]])
            except:
                zhong_peijian.append(0)
            try:
                zhong_xiulichang.append(Z_xlc[Xiulichanglx[i]])
            except:
                zhong_xiulichang.append(0)
            try:
                zhong_hzlx.append(Z_hzlx[Hezuoleix[i]])
            except:
                zhong_hzlx.append(0)
        datas['厂牌均值'] = np.c_[mean_changpai]
        datas['车系均值'] = np.c_[mean_chexi]
        datas['合作类型均值'] = np.c_[mean_hzlx]
        datas['项目均值'] = np.c_[mean_peijian]
        datas['修理厂均值'] = np.c_[mean_xiulichang]
        datas['合作类型众数'] = np.c_[zhong_hzlx]
        datas['项目众数'] = np.c_[zhong_peijian]
        datas['修理厂众数'] = np.c_[zhong_xiulichang]
        datas.to_csv(filename, encoding='gbk', index=0)

# ——————生成第四版文件（加入F值）————————
    def create_edition_four(self):
        filename = self.outputfile
        fivefeature = 'trainfile\{}\{}\五个特征.csv'.format(self.task, self.sheng)
        datas = pd.read_csv(filename, encoding='gbk')

        f1 = datas['定损项目名称']
        f2 = datas['厂牌']
        f3 = datas['车系']
        f4 = datas['修理厂类型']
        f0 = datas['修理厂地址']

        A1 = {}
        A2 = {}
        B1 = {}
        B2 = {}
        C1 = {}
        C2 = {}
        D1 = {}
        D2 = {}
        E1 = {}
        E2 = {}
        F_1 = {}
        F_2 = {}
        with open(fivefeature) as f:
            datas1 = f.readlines()
            for data in datas1:
                data = data.strip()
                data = data.split(',')
                x1 = data[0] + data[1] + data[2] + data[3] + data[4]
                x2 = data[0] + data[1] + data[2] + data[4]
                x3 = data[1] + data[2] + data[3] + data[4]
                x4 = data[1] + data[2] + data[4]
                x5 = data[0] + data[1] + data[4]
                x6 = data[1] + data[4]
                try:
                    A1[x1] = float(data[5])
                    A2[x1] = float(data[6])
                    B1[x2] = float(data[7])
                    B2[x2] = float(data[8])
                    C1[x3] = float(data[9])
                    C2[x3] = float(data[10])
                    D1[x4] = float(data[11])
                    D2[x4] = float(data[12])
                    E1[x5] = float(data[13])
                    E2[x5] = float(data[14])
                    F_1[x6] = float(data[15])
                    F_2[x6] = float(data[16])
                except ValueError:
                    pass
        F0 = []
        F1 = []
        F2 = []
        F3 = []
        F4 = []
        F5 = []
        F6 = []
        for i in range(len(f1)):
            X1 = f0[i] + f1[i] + f2[i] + f3[i] + f4[i]
            X2 = f0[i] + f1[i] + f2[i] + f4[i]
            X3 = f1[i] + f2[i] + f3[i] + f4[i]
            X4 = f1[i] + f2[i] + f4[i]
            X5 = f0[i] + f1[i] + f4[i]
            X6 = f1[i] + f4[i]
            if A2.get(X1) is not None and A2[X1] >= 8:
                f0_ = A1[X1]
                f1_ = 1
                f2_ = 0
                f3_ = 0
                f4_ = 0
                f5_ = 0
                f6_ = 0
            elif B2.get(X2) is not None and B2[X2] >= 8:
                f0_ = B1[X2]
                f1_ = 0
                f2_ = 1
                f3_ = 0
                f4_ = 0
                f5_ = 0
                f6_ = 0
            elif C2.get(X3) is not None and C2[X3] >= 8:
                f0_ = C1[X3]
                f1_ = 0
                f2_ = 0
                f3_ = 1
                f4_ = 0
                f5_ = 0
                f6_ = 0
            elif D2.get(X4) is not None and D2[X4] >= 8:
                f0_ = D1[X4]
                f1_ = 0
                f2_ = 0
                f3_ = 0
                f4_ = 1
                f5_ = 0
                f6_ = 0
            elif E2.get(X5) is not None and E2[X5] >= 8:
                f0_ = E1[X5]
                f1_ = 0
                f2_ = 0
                f3_ = 0
                f4_ = 0
                f5_ = 1
                f6_ = 0
            elif F_2.get(X6) is not None and F_2[X6] >= 8:
                f0_ = F_1[X6]
                f1_ = 0
                f2_ = 0
                f3_ = 0
                f4_ = 0
                f5_ = 0
                f6_ = 1
            else:
                f0_ = -1
                f1_ = 0
                f2_ = 0
                f3_ = 0
                f4_ = 0
                f5_ = 0
                f6_ = 0

            F0.append(f0_)
            F1.append(f1_)
            F2.append(f2_)
            F3.append(f3_)
            F4.append(f4_)
            F5.append(f5_)
            F6.append(f6_)
        datas['F0'] = np.c_[F0]
        datas['F1'] = np.c_[F1]
        datas['F2'] = np.c_[F2]
        datas['F3'] = np.c_[F3]
        datas['F4'] = np.c_[F4]
        datas['F5'] = np.c_[F5]
        datas['F6'] = np.c_[F6]
        datas = datas.drop(columns=['训练数据量不足'])

        datas['机构名'] = np.c_[[self.sheng for i in range(len(datas))]]
        if self.task == '喷漆':
            order = ['定损单号', '原始定损项目', '定损项目名称', '合作类型', '喷漆类型', '国别', '厂牌', '车系类别', '修理厂类型', '工时单价类型', \
                     '是否承修厂牌', '厂牌均值', '车系均值', '合作类型均值', '项目均值', '修理厂均值', '合作类型众数', '项目众数', \
                     '修理厂众数', 'F0', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', '价格', '修理厂编码','机构名', '修理厂地址', '修理厂名称', '工时折扣率', '定损员工号',
                     '定损员姓名','核损员工号', '核损员姓名','车系']
        elif self.task == '拆装':
            order = ['定损单号', '原始定损项目', '定损项目名称', '合作类型', '国别', '厂牌', '车系类别', '修理厂类型', '工时单价类型', \
                     '是否承修厂牌', '厂牌均值', '车系均值', '合作类型均值', '项目均值', '修理厂均值', '合作类型众数', '项目众数', \
                     '修理厂众数', 'F0', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', '价格', '修理厂编码', '机构名', '修理厂地址', '修理厂名称', '工时折扣率',
                     '定损员工号','定损员姓名', '核损员工号', '核损员姓名', '车系']
        elif self.task == '维修':
            order = ['定损单号', '原始定损项目', '定损项目名称', '合作类型','维修程度', '国别', '厂牌', '车系类别', '修理厂类型', '工时单价类型', \
                     '是否承修厂牌', '厂牌均值', '车系均值', '合作类型均值', '项目均值', '修理厂均值', '合作类型众数', '项目众数', \
                     '修理厂众数', 'F0', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', '价格', '修理厂编码', '机构名', '修理厂地址', '修理厂名称', '工时折扣率',
                     '定损员工号','定损员姓名', '核损员工号', '核损员姓名', '车系']
        datas = datas[order]
        datas.to_csv(filename, index=0, encoding='gbk')

    def make(self):
        self.create_edition_one()
        self.create_edition_two()
        self.create_edition_three()
        self.create_edition_four()

    def eval(self):
        np.set_printoptions(threshold=5000)
        onehot_path = 'json/{}/{}'.format(self.task, self.sheng)
        test_file = self.outputfile
        model_name = 'trainfile/{}/{}/{}.model'.format(self.task, self.sheng, self.sheng + self.task)
        test_output = 'testfile/{}/{}/{}预测数据.xlsx'.format(self.task,self.sheng,self.sheng + self.task)
        dataset = pd.read_csv(test_file, encoding='gbk')  # 注意自己数据路径
        results1 = []
        Acc = []
        if self.task == '喷漆':
            for i in range(len(dataset)):
                try:
                    test = dataset.iloc[i, 2:11].values
                    test2 = dataset.iloc[i, 11:26].values
                    labels = dataset.iloc[i, 26]

                    with open('{}/定损项目名称.json'.format(onehot_path)) as f:
                        D = json.load(f)
                    L = []
                    for a in D[test[0]]:
                        try:
                            L.append(int(a))
                        except:
                            pass
                    intermediary0 = np.array(L)

                    with open('{}/合作类型.json'.format(onehot_path)) as f:
                        D = json.load(f)
                    L = []
                    for a in D[test[1]]:
                        try:
                            L.append(int(a))
                        except:
                            pass
                    intermediary1 = np.array(L)

                    with open('{}/喷漆类型.json'.format(onehot_path)) as f:
                        D = json.load(f)
                    L = []
                    for a in D['全漆']:  # test[2]
                        try:
                            L.append(int(a))
                        except:
                            pass
                    intermediary3 = np.array(L)

                    with open('{}/国别.json'.format(onehot_path)) as f:
                        D = json.load(f)
                    L = []
                    for a in D[test[3]]:
                        try:
                            L.append(int(a))
                        except:
                            pass
                    intermediary4 = np.array(L)

                    with open('{}/厂牌.json'.format(onehot_path)) as f:
                        D = json.load(f)
                    L = []
                    for a in D[test[4]]:
                        try:
                            L.append(int(a))
                        except:
                            pass
                    intermediary5 = np.array(L)

                    with open('{}/车系.json'.format(onehot_path)) as f:
                        D = json.load(f)
                    L = []
                    for a in D[test[5]]:
                        try:
                            L.append(int(a))
                        except:
                            pass
                    intermediary6 = np.array(L)

                    with open('{}/修理厂类型.json'.format(onehot_path)) as f:
                        D = json.load(f)
                    L = []
                    for a in D[test[6]]:
                        try:
                            L.append(int(a))
                        except:
                            pass
                    intermediary7 = np.array(L)

                    with open('{}/修理厂类型.json'.format(onehot_path)) as f:
                        D = json.load(f)
                    L = []
                    for a in D[test[7]]:
                        try:
                            L.append(int(a))
                        except:
                            pass
                    intermediary8 = np.array(L)

                    with open('{}/是否承修厂牌.json'.format(onehot_path)) as f:
                        D = json.load(f)
                    L = []
                    for a in D[test[8]]:
                        try:
                            L.append(int(a))
                        except:
                            pass
                    intermediary9 = np.array(L)

                    num = []
                    num.append(len(intermediary0.tolist()))
                    num.append(len(intermediary1.tolist()))
                    num.append(len(intermediary3.tolist()))
                    num.append(len(intermediary4.tolist()))
                    num.append(len(intermediary5.tolist()))
                    num.append(len(intermediary6.tolist()))
                    num.append(len(intermediary7.tolist()))
                    num.append(len(intermediary8.tolist()))
                    num.append(len(intermediary9.tolist()))
                    trains = np.zeros(sum(num))
                    trains[:int(num[0])] = intermediary0
                    trains[int(num[0]):int(sum(num[:2]))] = intermediary1
                    trains[int(sum(num[:2])):int(sum(num[:3]))] = intermediary3
                    trains[int(sum(num[:3])):int(sum(num[:4]))] = intermediary4
                    trains[int(sum(num[:4])):int(sum(num[:5]))] = intermediary5
                    trains[int(sum(num[:5])):int(sum(num[:6]))] = intermediary6
                    trains[int(sum(num[:6])):int(sum(num[:7]))] = intermediary7
                    trains[int(sum(num[:7])):int(sum(num[:8]))] = intermediary8
                    trains[int(sum(num[:8])):int(sum(num))] = intermediary9
                    trainss = np.concatenate((trains, test2))
                    labels = np.expand_dims(np.array(labels), axis=0)
                    xgtest = xgb.DMatrix(np.expand_dims(trainss,axis=0), label=labels)
                    model = xgb.Booster(model_file=model_name)
                    preds = model.predict(xgtest)
                    acc = exp(-abs(labels[0] - preds[0]) / labels[0] * 2)
                    results1.append(preds[0])
                    Acc.append(acc)
                    print(labels[0], preds[0], acc)
                except:
                    results1.append('错误')
                    Acc.append(0)

        elif self.task == '拆装':
            for i in range(len(dataset)):
                try:
                    test = dataset.iloc[i, 2:10].values
                    test2 = dataset.iloc[i, 10:25].values
                    labels = dataset.iloc[i, 25]
                    with open('{}/定损项目名称.json'.format(onehot_path)) as f:
                        D = json.load(f)
                    L = []
                    for a in D[test[0]]:
                        try:
                            L.append(int(a))
                        except:
                            pass
                    intermediary0 = np.array(L)

                    with open('{}/合作类型.json'.format(onehot_path)) as f:
                        D = json.load(f)
                    L = []
                    for a in D[test[1]]:
                        try:
                            L.append(int(a))
                        except:
                            pass
                    intermediary1 = np.array(L)

                    with open('{}/国别.json'.format(onehot_path)) as f:
                        D = json.load(f)
                    L = []
                    for a in D[test[2]]:
                        try:
                            L.append(int(a))
                        except:
                            pass
                    intermediary2 = np.array(L)

                    with open('{}/厂牌.json'.format(onehot_path)) as f:
                        D = json.load(f)
                    L = []
                    for a in D[test[3]]:
                        try:
                            L.append(int(a))
                        except:
                            pass
                    intermediary3 = np.array(L)

                    with open('{}/车系.json'.format(onehot_path)) as f:
                        D = json.load(f)
                    L = []
                    for a in D[test[4]]:
                        try:
                            L.append(int(a))
                        except:
                            pass
                    intermediary4 = np.array(L)

                    with open('{}/修理厂类型.json'.format(onehot_path)) as f:
                        D = json.load(f)
                    L = []
                    for a in D[test[5]]:
                        try:
                            L.append(int(a))
                        except:
                            pass
                    intermediary5 = np.array(L)

                    with open('{}/修理厂类型.json'.format(onehot_path)) as f:
                        D = json.load(f)
                    L = []
                    for a in D[test[6]]:
                        try:
                            L.append(int(a))
                        except:
                            pass
                    intermediary6 = np.array(L)

                    with open('{}/是否承修厂牌.json'.format(onehot_path)) as f:
                        D = json.load(f)
                    L = []
                    for a in D[test[7]]:
                        try:
                            L.append(int(a))
                        except:
                            pass
                    intermediary7 = np.array(L)

                    num = []
                    num.append(len(intermediary0.tolist()))
                    num.append(len(intermediary1.tolist()))
                    num.append(len(intermediary2.tolist()))
                    num.append(len(intermediary3.tolist()))
                    num.append(len(intermediary4.tolist()))
                    num.append(len(intermediary5.tolist()))
                    num.append(len(intermediary6.tolist()))
                    num.append(len(intermediary7.tolist()))
                    trains = np.zeros(sum(num))
                    trains[:int(num[0])] = intermediary0
                    trains[int(num[0]):int(sum(num[:2]))] = intermediary1
                    trains[int(sum(num[:2])):int(sum(num[:3]))] = intermediary2
                    trains[int(sum(num[:3])):int(sum(num[:4]))] = intermediary3
                    trains[int(sum(num[:4])):int(sum(num[:5]))] = intermediary4
                    trains[int(sum(num[:5])):int(sum(num[:6]))] = intermediary5
                    trains[int(sum(num[:6])):int(sum(num[:7]))] = intermediary6
                    trains[int(sum(num[:7])):int(sum(num))] = intermediary7
                    trainss = np.concatenate((trains, test2))
                    labels = np.expand_dims(np.array(labels), axis=0)
                    xgtest = xgb.DMatrix(np.expand_dims(trainss, axis=0), label=labels)
                    model = xgb.Booster(model_file=model_name)
                    preds = model.predict(xgtest)
                    acc = exp(-abs(labels[0] - preds[0]) / labels[0] * 2)
                    results1.append(preds[0])
                    Acc.append(acc)
                    print(labels[0], preds[0], acc)
                except:
                    results1.append('错误')
                    Acc.append(0)
        elif self.task == '维修':
            for i in range(len(dataset)):
                try:
                    test = dataset.iloc[i, 2:11].values
                    test2 = dataset.iloc[i, 11:26].values
                    labels = dataset.iloc[i, 26]
                    with open('{}/定损项目名称.json'.format(onehot_path)) as f:
                        D = json.load(f)
                    L = []
                    for a in D[test[0]]:
                        try:
                            L.append(int(a))
                        except:
                            pass
                    intermediary0 = np.array(L)

                    with open('{}/合作类型.json'.format(onehot_path)) as f:
                        D = json.load(f)
                    L = []
                    for a in D[test[1]]:
                        try:
                            L.append(int(a))
                        except:
                            pass
                    intermediary1 = np.array(L)

                    with open('{}/维修程度.json'.format(onehot_path)) as f:
                        D = json.load(f)
                    L = []
                    for a in D[test[2]]:
                        try:
                            L.append(int(a))
                        except:
                            pass
                    intermediary2 = np.array(L)

                    with open('{}/国别.json'.format(onehot_path)) as f:
                        D = json.load(f)
                    L = []
                    for a in D[test[3]]:
                        try:
                            L.append(int(a))
                        except:
                            pass
                    intermediary3 = np.array(L)

                    with open('{}/厂牌.json'.format(onehot_path)) as f:
                        D = json.load(f)
                    L = []
                    for a in D[test[4]]:
                        try:
                            L.append(int(a))
                        except:
                            pass
                    intermediary4 = np.array(L)

                    with open('{}/车系.json'.format(onehot_path)) as f:
                        D = json.load(f)
                    L = []
                    for a in D[test[5]]:
                        try:
                            L.append(int(a))
                        except:
                            pass
                    intermediary5 = np.array(L)

                    with open('{}/修理厂类型.json'.format(onehot_path)) as f:
                        D = json.load(f)
                    L = []
                    for a in D[test[6]]:
                        try:
                            L.append(int(a))
                        except:
                            pass
                    intermediary6 = np.array(L)

                    with open('{}/修理厂类型.json'.format(onehot_path)) as f:
                        D = json.load(f)
                    L = []
                    for a in D[test[7]]:
                        try:
                            L.append(int(a))
                        except:
                            pass
                    intermediary7 = np.array(L)

                    with open('{}/是否承修厂牌.json'.format(onehot_path)) as f:
                        D = json.load(f)
                    L = []
                    for a in D[test[8]]:
                        try:
                            L.append(int(a))
                        except:
                            pass
                    intermediary8 = np.array(L)

                    num = []
                    num.append(len(intermediary0.tolist()))
                    num.append(len(intermediary1.tolist()))
                    num.append(len(intermediary2.tolist()))
                    num.append(len(intermediary3.tolist()))
                    num.append(len(intermediary4.tolist()))
                    num.append(len(intermediary5.tolist()))
                    num.append(len(intermediary6.tolist()))
                    num.append(len(intermediary7.tolist()))
                    num.append(len(intermediary8.tolist()))
                    trains = np.zeros(sum(num))
                    trains[:int(num[0])] = intermediary0
                    trains[int(num[0]):int(sum(num[:2]))] = intermediary1
                    trains[int(sum(num[:2])):int(sum(num[:3]))] = intermediary2
                    trains[int(sum(num[:3])):int(sum(num[:4]))] = intermediary3
                    trains[int(sum(num[:4])):int(sum(num[:5]))] = intermediary4
                    trains[int(sum(num[:5])):int(sum(num[:6]))] = intermediary5
                    trains[int(sum(num[:6])):int(sum(num[:7]))] = intermediary6
                    trains[int(sum(num[:7])):int(sum(num[:8]))] = intermediary7
                    trains[int(sum(num[:8])):int(sum(num))] = intermediary8
                    trainss = np.concatenate((trains, test2))
                    labels = np.expand_dims(np.array(labels), axis=0)
                    xgtest = xgb.DMatrix(np.expand_dims(trainss, axis=0), label=labels)
                    model = xgb.Booster(model_file=model_name)
                    preds = model.predict(xgtest)
                    acc = exp(-abs(labels[0] - preds[0]) / labels[0] * 2)
                    results1.append(preds[0])
                    Acc.append(acc)
                    print(labels[0], preds[0], acc)
                except:
                    results1.append('错误')
                    Acc.append(0)
        dataset['预测折后喷漆费'] = np.c_[results1]
        dataset['准确率'] = np.c_[Acc]
        dataset.to_excel(test_output)