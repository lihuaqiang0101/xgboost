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
from utils import get_mean
from utils import all_list
from utils import get_mean_pengqi
from utils import get_mean_caizhaung
from utils import get_mean_weixiu
from utils import get_changpai_price
from utils import get_chexi_price
from utils import get_zhengshu_pengqi
from utils import get_zhengshu_chaizhaung
from utils import get_zhengshu_weixiu
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
from sklearn.metrics import r2_score
from math import *

"""
生成训练文件类
传入参数：
task：任务类型，可选参数：（喷漆、拆装、维修）
sheng：机构省份名，如：四川、陕西等

方法名及功能：
get_fac_palce：获取修理厂编号对应的地级市填充到fac_place
all_list1：统计两个关联列表的频次返回一个字典
create_edition_one：生成第一版文件（给各个条目打上（规范或者不规范）标记）
create_edition_two：生成第二版文件（加上跟单，删除不规范的条目，标记车的价格类别）
get_mean1cp：获取厂牌的均值
get_meancx：获取车系的均值
create_edition_three：生成第三版文件（计算均值众数等）
create_five_feature_file：生成F值特征要用到的五个特征的数据表
create_edition_four：生成第四版文件（加入F值）
write_mean_mode：将均值众数等写入json文件
make：生成最终训练文件

例子：
trainfile = TrainFile(task='拆装',sheng='大连')#创建一个训练文件对象
trainfile.make()#调用make方法生成训练文件
trainfile.train(thread=8)#调用train方法使用8个线程开始训练
"""
class TrainFile():
    def __init__(self,task,sheng):
        if not os.path.exists('trainfile'):
            os.mkdir('trainfile')
        if not os.path.exists('trainfile\{}'.format(task)):
            os.mkdir('trainfile\{}'.format(task))
        if not os.path.exists('trainfile\{}\{}'.format(task,sheng)):
            os.mkdir('trainfile\{}\{}'.format(task,sheng))
        self.trainfile = '201811_to_202004_all.csv'
        self.outputfile = 'trainfile\{}\{}\{}训练数据.csv'.format(task,sheng,sheng + task)
        self.sheng = sheng
        self.task = task
        self.fac_place = {}
        self.fac_city = {}
        if not os.path.exists(self.outputfile):
            self.dataset = pd.read_csv(self.trainfile)
            self.get_fac_palce()
            self.dataset = self.dataset.loc[self.dataset['关联机构'] == '{}分公司'.format(sheng)]
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

#————————生成五个特征的数据表————————
    def create_five_feature_file(self):
        filename = self.outputfile
        fivefeature = 'trainfile\{}\{}\五个特征.csv'.format(self.task,self.sheng)
        datas = pd.read_csv(filename, encoding='gbk')

        f0 = datas['修理厂地址']
        f1 = datas['定损项目名称']
        if self.task == '喷漆':
            f2 = datas['喷漆类型']
        f3 = datas['厂牌']
        f5 = datas['车系']
        f4 = datas['修理厂类型']
        f6 = datas['价格']
        x1 = []
        x2 = []
        x3 = []
        x4 = []
        x5 = []
        x6 = []
        y = []
        for i in range(len(f1)):
            if self.task == '喷漆':
                if f2[i] == '全漆':
                    x1.append(f0[i] + ',' + f1[i] + ',' + f3[i] + ',' + f5[i] + ',' + f4[i])  # + f2[i] + ','
                    x2.append(f0[i] + ',' + f1[i] + ',' + f3[i] + ',' + f4[i])
                    x3.append(f1[i] + ',' + f3[i] + ',' + f5[i] + ',' + f4[i])
                    x4.append(f1[i] + ',' + f3[i] + ',' + f4[i])
                    x5.append(f0[i] + ',' + f1[i] + ',' + f4[i])
                    x6.append(f1[i] + ',' + f4[i])
                    y.append(f6[i])
            else:
                x1.append(f0[i] + ',' + f1[i] + ',' + f3[i] + ',' + f5[i] + ',' + f4[i])  # + f2[i] + ','
                x2.append(f0[i] + ',' + f1[i] + ',' + f3[i] + ',' + f4[i])
                x3.append(f1[i] + ',' + f3[i] + ',' + f5[i] + ',' + f4[i])
                x4.append(f1[i] + ',' + f3[i] + ',' + f4[i])
                x5.append(f0[i] + ',' + f1[i] + ',' + f4[i])
                x6.append(f1[i] + ',' + f4[i])
                y.append(f6[i])
        A1 = get_mean(x1, y)
        A2 = all_list(x1)
        B1 = get_mean(x2, y)
        B2 = all_list(x2)
        C1 = get_mean(x3, y)
        C2 = all_list(x3)
        D1 = get_mean(x4, y)
        D2 = all_list(x4)
        E1 = get_mean(x5, y)
        E2 = all_list(x5)
        F1 = get_mean(x6, y)
        F2 = all_list(x6)
        outputs = open(fivefeature, 'w', newline='')
        csv_write = csv.writer(outputs, dialect='excel')
        csv_write.writerow(
            ['地级市', '定损项目', '厂牌', '车系', '修理厂类型', 'a1', 'a2', 'b1', 'b2', 'c1', 'c2', 'd1', 'd2', 'e1', 'e2', 'f1',
             'f2'])
        for key in x1:
            data = []
            k = key.split(',')
            data.extend(k)
            k1 = k[0] + ',' + k[1] + ',' + k[2] + ',' + k[4]
            k2 = k[1] + ',' + k[2] + ',' + k[3] + ',' + k[4]
            k3 = k[1] + ',' + k[2] + ',' + k[4]
            k4 = k[0] + ',' + k[1] + ',' + k[4]
            k5 = k[1] + ',' + k[4]
            try:
                a1 = round(A1[key] / 10) * 10
            except:
                a1 = -1
            try:
                a2 = A2[key] / 10 * 10
            except:
                a2 = 0
            try:
                b1 = round(B1[k1] / 10) * 10
            except:
                b1 = -1
            try:
                b2 = B2[k1] / 10 * 10
            except:
                b2 = 0
            try:
                c1 = round(C1[k2] / 10) * 10
            except:
                c1 = -1
            try:
                c2 = C2[k2] / 10 * 10
            except:
                c2 = 0
            try:
                d1 = round(D1[k3] / 10) * 10
            except:
                d1 = -1
            try:
                d2 = D2[k3] / 10 * 10
            except:
                d2 = 0
            try:
                e1 = round(E1[k4] / 10) * 10
            except:
                e1 = -1
            try:
                e2 = E2[k4] / 10 * 10
            except:
                e2 = 0
            try:
                f1 = round(F1[k5] / 10) * 10
            except:
                f1 = -1
            try:
                f2 = F2[k5] / 10 * 10
            except:
                f2 = 0
            data.extend([a1, a2, b1, b2, c1, c2, d1, d2, e1, e2, f1, f2])
            csv_write.writerow(data)

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
                            # l1 = data[2] + '#' + data[3] + '#' + data[4] + '#' + data[5] + '#' + data[6] + '#' + data[7] + '#' + data[8] + '#' + data[9] + '#' + str(data[12])
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
                    # k = data[0] + '#'+ data[1] + '#'+ data[2] + '#'+ data[3] + '#'+ data[4] + '#'+ data[5] + '#'+ data[6]+ '#'+ data[7]
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
#——————筛选掉不和规则的数据————————————
            datas = datas.loc[ datas['喷漆类型'] == '全漆' ]
        datas = datas.loc[datas['是否含有补差价'] == 0]
        datas = datas.loc[datas['定损项目不规范'] == 0]
        datas = datas.loc[datas['品牌车系录入不规范'] == 0]
        datas = datas.loc[datas['训练数据量不足'] == 0]
        datas = datas.loc[datas['定损项目金额过低'] == 0]
        datas = datas.loc[datas['定损项目金额过高'] == 0]
        datas = datas.loc[datas['定损价格排序过高过低'] == 0]
        datas = datas.loc[datas['标记'] == 0]
        if self.task == '维修':
            datas = datas.loc[datas['是否含有外修费'] == 0]
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
        if self.task == '喷漆':
            Penqilx = list(datas['喷漆类型'])
        Price = list(datas['价格'])
        Hezuoleix = list(datas['合作类型'])
        Changpai = list(datas['厂牌'])
        Chexi = list(datas['车系'])
        Xiulichanglx = list(datas['修理厂类型'])
        mean_changpai = self.get_mean1cp(Changpai)
        mean_chexi = self.get_meancx(Changpai, Chexi)
        if self.task == '喷漆':
            mean_hzlx = get_mean_pengqi(Hezuoleix, Price, Penqilx)
            mean_peijian = get_mean_pengqi(Dingsunxm, Price, Penqilx)
            mean_xiulichang = get_mean_pengqi(Xiulichanglx, Price, Penqilx)
            zhong_hzlx = get_zhengshu_pengqi(Hezuoleix, Price, Penqilx)
            zhong_peijian = get_zhengshu_pengqi(Dingsunxm, Price, Penqilx)
            zhong_xiulichang = get_zhengshu_pengqi(Xiulichanglx, Price, Penqilx)
        elif self.task == '拆装':
            mean_hzlx = get_mean_caizhaung(Hezuoleix, Price)
            mean_peijian = get_mean_caizhaung(Dingsunxm, Price)
            mean_xiulichang = get_mean_caizhaung(Xiulichanglx, Price)
            zhong_hzlx = get_zhengshu_chaizhaung(Hezuoleix, Price)
            zhong_peijian = get_zhengshu_chaizhaung(Dingsunxm, Price)
            zhong_xiulichang = get_zhengshu_chaizhaung(Xiulichanglx, Price)
        elif self.task == '维修':
            mean_hzlx = get_mean_weixiu(Hezuoleix, Price)
            mean_peijian = get_mean_weixiu(Dingsunxm, Price)
            mean_xiulichang = get_mean_weixiu(Xiulichanglx, Price)
            zhong_hzlx = get_zhengshu_weixiu(Hezuoleix, Price)
            zhong_peijian = get_zhengshu_weixiu(Dingsunxm, Price)
            zhong_xiulichang = get_zhengshu_weixiu(Xiulichanglx, Price)
        datas['厂牌均值'] = np.c_[mean_changpai]
        datas['车系均值'] = np.c_[mean_chexi]
        datas['合作类型均值'] = np.c_[mean_hzlx]
        datas['项目均值'] = np.c_[mean_peijian]
        datas['修理厂均值'] = np.c_[mean_xiulichang]
        datas['合作类型众数'] = np.c_[zhong_hzlx]
        datas['项目众数'] = np.c_[zhong_peijian]
        datas['修理厂众数'] = np.c_[zhong_xiulichang]
        datas = datas.loc[datas['厂牌均值'] != -1]
        datas = datas.loc[datas['车系均值'] != -1]
        datas = datas.loc[datas['合作类型均值'] != -1]
        datas = datas.loc[datas['项目均值'] != -1]
        datas = datas.loc[datas['修理厂均值'] != -1]
        datas = datas.loc[datas['合作类型众数'] != -1]
        datas = datas.loc[datas['项目众数'] != -1]
        datas = datas.loc[datas['修理厂众数'] != -1]
        datas.to_csv(filename, encoding='gbk', index=0)

# ——————生成第四版文件（加入F值）————————
    def create_edition_four(self):
        s = self.sheng
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

        datas = datas.loc[datas['F0'] != -1]
        datas = datas.loc[datas['修理厂地址'] != '无']
        datas = datas.loc[datas['F0'] != -1]
        datas = datas.loc[datas['F1'] != -1]
        datas = datas.loc[datas['F2'] != -1]
        datas = datas.loc[datas['F3'] != -1]
        datas = datas.loc[datas['F4'] != -1]
        datas = datas.loc[datas['F5'] != -1]
        datas = datas.loc[datas['F6'] != -1]
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

    def write_mean_mode(self):
        if not os.path.exists('json'):
            os.mkdir('json')
        if not os.path.exists('json\{}'.format(self.task)):
            os.mkdir('json\{}'.format(self.task))
        if not os.path.exists('json\{}\{}'.format(self.task,self.sheng)):
            os.mkdir('json\{}\{}'.format(self.task,self.sheng))

        datas = pd.read_csv(self.outputfile, encoding='gbk')
        xm = datas['定损项目名称']
        xlc = datas['修理厂类型']
        hzlx = datas['合作类型']
        m_xm = datas['项目均值']
        z_xm = datas['项目众数']
        m_xlc = datas['修理厂均值']
        z_xlc = datas['修理厂众数']
        m_hzlx = datas['合作类型均值']
        z_hzlx = datas['合作类型众数']

        M_xm = {}
        Z_xm = {}
        M_xlc = {}
        Z_xlc = {}
        M_hzlx = {}
        Z_hzlx = {}

        for i in range(len(datas)):
            if not xm[i] in M_xm:
                M_xm[xm[i]] = m_xm[i]
                Z_xm[xm[i]] = z_xm[i]
            if not xlc[i] in M_xlc:
                M_xlc[xlc[i]] = m_xlc[i]
                Z_xlc[xlc[i]] = z_xlc[i]
            if not hzlx[i] in M_hzlx:
                M_hzlx[hzlx[i]] = m_hzlx[i]
                Z_hzlx[hzlx[i]] = z_hzlx[i]

        onehot_path = 'json/{}/{}'.format(self.task,self.sheng)
        with open('{}/项目均值.json'.format(onehot_path), 'w') as f:
            json.dump(M_xm, f)
        with open('{}/项目众数.json'.format(onehot_path), 'w') as f:
            json.dump(Z_xm, f)
        with open('{}/修理厂均值.json'.format(onehot_path), 'w') as f:
            json.dump(M_xlc, f)
        with open('{}/修理厂众数.json'.format(onehot_path), 'w') as f:
            json.dump(Z_xlc, f)
        with open('{}/合作类型均值.json'.format(onehot_path), 'w') as f:
            json.dump(M_hzlx, f)
        with open('{}/合作类型众数.json'.format(onehot_path), 'w') as f:
            json.dump(Z_hzlx, f)

    def make(self):
        self.create_edition_one()
        self.create_edition_two()
        self.create_edition_three()
        self.create_five_feature_file()
        self.create_edition_four()
        self.write_mean_mode()

    def train(self,thread):
        enc = OneHotEncoder()
        np.set_printoptions(threshold=5000)
        onehot_path = 'json/{}/{}'.format(self.task,self.sheng)
        train_file = self.outputfile
        train_output = 'trainfile/{}/{}/{}训练结果.xlsx'.format(self.task,self.sheng,self.sheng + self.task)
        thread = thread
        model_name = 'trainfile/{}/{}/{}.model'.format(self.task,self.sheng,self.sheng + self.task)

        x1 = []
        x2 = []
        x3 = []
        x4 = []
        x5 = []
        x6 = []
        x7 = []
        x8 = []
        x9 = []
        x10 = []
        x11 = []
        if self.task == '喷漆':
            with open(train_file, encoding='gbk') as f:
                datas = csv.reader(f)
                i = 0
                for data in datas:
                    if i > 0:
                        x1.append(data[2])
                        x2.append(data[3])
                        x3.append(data[4])
                        x4.append(data[5])
                        x5.append(data[6])
                        x6.append(data[7])
                        x7.append(data[8])
                        x9.append(data[10])
                    i += 1

            X1 = [[data] for data in list(set(x1))]
            X2 = [[data] for data in list(set(x2))]
            X3 = [[data] for data in list(set(x3))]
            X4 = [[data] for data in list(set(x4))]
            X5 = [[data] for data in list(set(x5))]
            X6 = [[data] for data in list(set(x6))]
            X7 = [[data] for data in list(set(x7))]
            X9 = [[data] for data in list(set(x9))]

            enc.fit(X1)
            D = {}
            values = enc.transform(X1).toarray()
            for i in range(len(X1)):
                D[X1[i][0]] = str(values[i]).replace('\n', '')
            with open('{}/定损项目名称.json'.format(onehot_path), 'w') as f:
                json.dump(D, f)

            enc.fit(X2)
            D = {}
            values = enc.transform(X2).toarray()
            for i in range(len(X2)):
                D[X2[i][0]] = str(values[i]).replace('\n', '')
            with open('{}/合作类型.json'.format(onehot_path), 'w') as f:
                json.dump(D, f)

            enc.fit(X3)
            D = {}
            values = enc.transform(X3).toarray()
            for i in range(len(X3)):
                D[X3[i][0]] = str(values[i]).replace('\n', '')
            with open('{}/喷漆类型.json'.format(onehot_path), 'w') as f:
                json.dump(D, f)

            enc.fit(X4)
            D = {}
            values = enc.transform(X4).toarray()
            for i in range(len(X4)):
                D[X4[i][0]] = str(values[i]).replace('\n', '')
            with open('{}/国别.json'.format(onehot_path), 'w') as f:
                json.dump(D, f)

            enc.fit(X5)
            D = {}
            values = enc.transform(X5).toarray()
            for i in range(len(X5)):
                D[X5[i][0]] = str(values[i]).replace('\n', '')
            with open('{}/厂牌.json'.format(onehot_path), 'w') as f:
                json.dump(D, f)

            enc.fit(X6)
            D = {}
            values = enc.transform(X6).toarray()
            for i in range(len(X6)):
                D[X6[i][0]] = str(values[i]).replace('\n', '')
            with open('{}/车系.json'.format(onehot_path), 'w') as f:
                json.dump(D, f)

            enc.fit(X7)
            D = {}
            values = enc.transform(X7).toarray()
            for i in range(len(X7)):
                D[X7[i][0]] = str(values[i]).replace('\n', '')
            with open('{}/修理厂类型.json'.format(onehot_path), 'w') as f:
                json.dump(D, f)

            enc.fit(X9)
            D = {}
            values = enc.transform(X9).toarray()
            for i in range(len(X9)):
                D[X9[i][0]] = str(values[i]).replace('\n', '')
            with open('{}/是否承修厂牌.json'.format(onehot_path), 'w') as f:
                json.dump(D, f)

            dataset = pd.read_csv(train_file, encoding='gbk')  # 注意自己数据路径
            train = dataset.iloc[:, 2:11].values
            train2 = dataset.iloc[:, 11:26].values
            labels = dataset.iloc[:, 26].values

            with open('{}/定损项目名称.json'.format(onehot_path)) as f:
                D = json.load(f)
            intermediary0 = np.array([np.array(
                [int(n) for n in list(D[data].replace('.', '').replace(' ', '').replace('[', '').replace(']', ''))]) for
                                      data in train[:, 0]])

            with open('{}/合作类型.json'.format(onehot_path)) as f:
                D = json.load(f)
            intermediary1 = np.array([np.array(
                [int(n) for n in list(D[data].replace('.', '').replace(' ', '').replace('[', '').replace(']', ''))]) for
                                      data in train[:, 1]])

            with open('{}/喷漆类型.json'.format(onehot_path)) as f:
                D = json.load(f)
            intermediary3 = np.array([np.array(
                [int(n) for n in list(D[data].replace('.', '').replace(' ', '').replace('[', '').replace(']', ''))]) for
                                      data in train[:, 2]])

            with open('{}/国别.json'.format(onehot_path)) as f:
                D = json.load(f)
            intermediary4 = np.array([np.array(
                [int(n) for n in list(D[data].replace('.', '').replace(' ', '').replace('[', '').replace(']', ''))]) for
                                      data in train[:, 3]])

            with open('{}/厂牌.json'.format(onehot_path)) as f:
                D = json.load(f)
            intermediary5 = np.array([np.array(
                [int(n) for n in list(D[data].replace('.', '').replace(' ', '').replace('[', '').replace(']', ''))]) for
                                      data in train[:, 4]])

            with open('{}/车系.json'.format(onehot_path)) as f:
                D = json.load(f)
            intermediary6 = np.array([np.array(
                [int(n) for n in list(D[data].replace('.', '').replace(' ', '').replace('[', '').replace(']', ''))]) for
                                      data in train[:, 5]])

            with open('{}/修理厂类型.json'.format(onehot_path)) as f:
                D = json.load(f)
            intermediary7 = np.array([np.array(
                [int(n) for n in list(D[data].replace('.', '').replace(' ', '').replace('[', '').replace(']', ''))]) for
                                      data in train[:, 6]])

            with open('{}/修理厂类型.json'.format(onehot_path)) as f:
                D = json.load(f)
            intermediary8 = np.array([np.array(
                [int(n) for n in list(D[data].replace('.', '').replace(' ', '').replace('[', '').replace(']', ''))]) for
                                      data in train[:, 7]])

            with open('{}/是否承修厂牌.json'.format(onehot_path)) as f:
                D = json.load(f)
            intermediary9 = np.array([np.array(
                [int(n) for n in list(D[data].replace('.', '').replace(' ', '').replace('[', '').replace(']', ''))]) for
                                      data in train[:, 8]])

            num = []
            num.append(len(intermediary0[0]))
            num.append(len(intermediary1[0]))
            num.append(len(intermediary3[0]))
            num.append(len(intermediary4[0]))
            num.append(len(intermediary5[0]))
            num.append(len(intermediary6[0]))
            num.append(len(intermediary7[0]))
            num.append(len(intermediary8[0]))
            num.append(len(intermediary9[0]))

            trains = np.zeros(shape=(len(intermediary0), sum(num)))
            for i in range(len(intermediary0)):
                trains[i, :num[0]] = intermediary0[i]
                trains[i, num[0]:sum(num[:2])] = intermediary1[i]
                trains[i, sum(num[:2]):sum(num[:3])] = intermediary3[i]
                trains[i, sum(num[:3]):sum(num[:4])] = intermediary4[i]
                trains[i, sum(num[:4]):sum(num[:5])] = intermediary5[i]
                trains[i, sum(num[:5]):sum(num[:6])] = intermediary6[i]
                trains[i, sum(num[:6]):sum(num[:7])] = intermediary7[i]
                trains[i, sum(num[:7]):sum(num[:8])] = intermediary8[i]
                trains[i, sum(num[:8]):sum(num)] = intermediary9[i]
        elif self.task == '拆装':
            with open(train_file, encoding='gbk') as f:
                datas = csv.reader(f)
                i = 0
                for data in datas:
                    if i > 0:
                        x1.append(data[2])
                        x2.append(data[3])
                        x5.append(data[4])
                        x6.append(data[5])
                        x7.append(data[6])
                        x8.append(data[7])
                        x9.append(data[8])
                        x10.append(data[9])
                    i += 1

            X1 = [[data] for data in list(set(x1))]
            X2 = [[data] for data in list(set(x2))]
            X5 = [[data] for data in list(set(x5))]
            X6 = [[data] for data in list(set(x6))]
            X7 = [[data] for data in list(set(x7))]
            X8 = [[data] for data in list(set(x8))]
            X9 = [[data] for data in list(set(x9))]
            X10 = [[data] for data in list(set(x10))]

            enc.fit(X1)
            D = {}
            values = enc.transform(X1).toarray()
            for i in range(len(X1)):
                D[X1[i][0]] = str(values[i]).replace('\n', '')
            with open('{}/定损项目名称.json'.format(onehot_path), 'w') as f:
                json.dump(D, f)

            enc.fit(X2)
            D = {}
            values = enc.transform(X2).toarray()
            for i in range(len(X2)):
                D[X2[i][0]] = str(values[i]).replace('\n', '')
            with open('{}/合作类型.json'.format(onehot_path), 'w') as f:
                json.dump(D, f)

            enc.fit(X5)
            D = {}
            values = enc.transform(X5).toarray()
            for i in range(len(X5)):
                D[X5[i][0]] = str(values[i]).replace('\n', '')
            with open('{}/国别.json'.format(onehot_path), 'w') as f:
                json.dump(D, f)

            enc.fit(X6)
            D = {}
            values = enc.transform(X6).toarray()
            for i in range(len(X6)):
                D[X6[i][0]] = str(values[i]).replace('\n', '')
            with open('{}/厂牌.json'.format(onehot_path), 'w') as f:
                json.dump(D, f)

            enc.fit(X7)
            D = {}
            values = enc.transform(X7).toarray()
            for i in range(len(X7)):
                D[X7[i][0]] = str(values[i]).replace('\n', '')
            with open('{}/车系.json'.format(onehot_path), 'w') as f:
                json.dump(D, f)

            enc.fit(X8)
            D = {}
            values = enc.transform(X8).toarray()
            for i in range(len(X8)):
                D[X8[i][0]] = str(values[i]).replace('\n', '')
            with open('{}/修理厂类型.json'.format(onehot_path), 'w') as f:
                json.dump(D, f)

            enc.fit(X10)
            D = {}
            values = enc.transform(X10).toarray()
            for i in range(len(X10)):
                D[X10[i][0]] = str(values[i]).replace('\n', '')
            with open('{}/是否承修厂牌.json'.format(onehot_path), 'w') as f:
                json.dump(D, f)

            dataset = pd.read_csv(train_file, encoding='gbk')  # 注意自己数据路径
            train = dataset.iloc[:, 2:10].values
            train2 = dataset.iloc[:, 10:25].values
            labels = dataset.iloc[:, 25].values
            with open('{}/定损项目名称.json'.format(onehot_path)) as f:
                D = json.load(f)
            intermediary0 = np.array([np.array(
                [int(n) for n in list(D[data].replace('.', '').replace(' ', '').replace('[', '').replace(']', ''))]) for
                                      data in train[:, 0]])

            with open('{}/合作类型.json'.format(onehot_path)) as f:
                D = json.load(f)
            intermediary1 = np.array([np.array(
                [int(n) for n in list(D[data].replace('.', '').replace(' ', '').replace('[', '').replace(']', ''))]) for
                                      data in train[:, 1]])

            with open('{}/国别.json'.format(onehot_path)) as f:
                D = json.load(f)
            intermediary4 = np.array([np.array(
                [int(n) for n in list(D[data].replace('.', '').replace(' ', '').replace('[', '').replace(']', ''))]) for
                                      data in train[:, 2]])

            with open('{}/厂牌.json'.format(onehot_path)) as f:
                D = json.load(f)
            intermediary5 = np.array([np.array(
                [int(n) for n in list(D[data].replace('.', '').replace(' ', '').replace('[', '').replace(']', ''))]) for
                                      data in train[:, 3]])

            with open('{}/车系.json'.format(onehot_path)) as f:
                D = json.load(f)
            intermediary6 = np.array([np.array(
                [int(n) for n in list(D[data].replace('.', '').replace(' ', '').replace('[', '').replace(']', ''))]) for
                                      data in train[:, 4]])

            with open('{}/修理厂类型.json'.format(onehot_path)) as f:
                D = json.load(f)
            intermediary7 = np.array([np.array(
                [int(n) for n in list(D[data].replace('.', '').replace(' ', '').replace('[', '').replace(']', ''))]) for
                                      data in train[:, 5]])

            with open('{}/修理厂类型.json'.format(onehot_path)) as f:
                D = json.load(f)
            intermediary8 = np.array([np.array(
                [int(n) for n in list(D[data].replace('.', '').replace(' ', '').replace('[', '').replace(']', ''))]) for
                                      data in train[:, 6]])

            with open('{}/是否承修厂牌.json'.format(onehot_path)) as f:
                D = json.load(f)
            intermediary9 = np.array([np.array(
                [int(n) for n in list(D[data].replace('.', '').replace(' ', '').replace('[', '').replace(']', ''))]) for
                                      data in train[:, 7]])

            num = []
            num.append(len(intermediary0[0]))
            num.append(len(intermediary1[0]))
            num.append(len(intermediary4[0]))
            num.append(len(intermediary5[0]))
            num.append(len(intermediary6[0]))
            num.append(len(intermediary7[0]))
            num.append(len(intermediary8[0]))
            num.append(len(intermediary9[0]))

            trains = np.zeros(shape=(len(intermediary0), sum(num)))
            for i in range(len(intermediary0)):
                trains[i, :num[0]] = intermediary0[i]
                trains[i, num[0]:sum(num[:2])] = intermediary1[i]
                trains[i, sum(num[:2]):sum(num[:3])] = intermediary4[i]
                trains[i, sum(num[:3]):sum(num[:4])] = intermediary5[i]
                trains[i, sum(num[:4]):sum(num[:5])] = intermediary6[i]
                trains[i, sum(num[:5]):sum(num[:6])] = intermediary7[i]
                trains[i, sum(num[:6]):sum(num[:7])] = intermediary8[i]
                trains[i, sum(num[:7]):sum(num)] = intermediary9[i]
        elif self.task == '维修':
            with open(train_file, encoding='gbk') as f:
                datas = csv.reader(f)
                i = 0
                for data in datas:
                    if i > 0:
                        x1.append(data[2])
                        x2.append(data[3])
                        x5.append(data[4])
                        x6.append(data[5])
                        x7.append(data[6])
                        x8.append(data[7])
                        x9.append(data[8])
                        x10.append(data[9])
                        x11.append(data[10])
                    i += 1

            X1 = [[data] for data in list(set(x1))]
            X2 = [[data] for data in list(set(x2))]
            X5 = [[data] for data in list(set(x5))]
            X6 = [[data] for data in list(set(x6))]
            X7 = [[data] for data in list(set(x7))]
            X8 = [[data] for data in list(set(x8))]
            X9 = [[data] for data in list(set(x9))]
            X10 = [[data] for data in list(set(x10))]
            X11 = [[data] for data in list(set(x11))]

            enc.fit(X1)
            D = {}
            values = enc.transform(X1).toarray()
            for i in range(len(X1)):
                D[X1[i][0]] = str(values[i]).replace('\n', '')
            with open('{}/定损项目名称.json'.format(onehot_path), 'w') as f:
                json.dump(D, f)

            enc.fit(X2)
            D = {}
            values = enc.transform(X2).toarray()
            for i in range(len(X2)):
                D[X2[i][0]] = str(values[i]).replace('\n', '')
            with open('{}/合作类型.json'.format(onehot_path), 'w') as f:
                json.dump(D, f)

            enc.fit(X5)
            D = {}
            values = enc.transform(X5).toarray()
            for i in range(len(X5)):
                D[X5[i][0]] = str(values[i]).replace('\n', '')
            with open('{}/维修程度.json'.format(onehot_path), 'w') as f:
                json.dump(D, f)

            enc.fit(X6)
            D = {}
            values = enc.transform(X6).toarray()
            for i in range(len(X6)):
                D[X6[i][0]] = str(values[i]).replace('\n', '')
            with open('{}/国别.json'.format(onehot_path), 'w') as f:
                json.dump(D, f)

            enc.fit(X7)
            D = {}
            values = enc.transform(X7).toarray()
            for i in range(len(X7)):
                D[X7[i][0]] = str(values[i]).replace('\n', '')
            with open('{}/厂牌.json'.format(onehot_path), 'w') as f:
                json.dump(D, f)

            enc.fit(X8)
            D = {}
            values = enc.transform(X8).toarray()
            for i in range(len(X8)):
                D[X8[i][0]] = str(values[i]).replace('\n', '')
            with open('{}/车系.json'.format(onehot_path), 'w') as f:
                json.dump(D, f)

            enc.fit(X9)
            D = {}
            values = enc.transform(X9).toarray()
            for i in range(len(X9)):
                D[X9[i][0]] = str(values[i]).replace('\n', '')
            with open('{}/修理厂类型.json'.format(onehot_path), 'w') as f:
                json.dump(D, f)

            enc.fit(X11)
            D = {}
            values = enc.transform(X11).toarray()
            for i in range(len(X11)):
                D[X11[i][0]] = str(values[i]).replace('\n', '')
            with open('{}/是否承修厂牌.json'.format(onehot_path), 'w') as f:
                json.dump(D, f)

            dataset = pd.read_csv(train_file, encoding='gbk')  # 注意自己数据路径
            train = dataset.iloc[:, 2:11].values
            train2 = dataset.iloc[:, 11:26].values
            labels = dataset.iloc[:, 26].values

            with open('{}/定损项目名称.json'.format(onehot_path)) as f:
                D = json.load(f)
            intermediary0 = np.array([np.array(
                [int(n) for n in list(D[data].replace('.', '').replace(' ', '').replace('[', '').replace(']', ''))]) for
                                      data in train[:, 0]])

            with open('{}/合作类型.json'.format(onehot_path)) as f:
                D = json.load(f)
            intermediary1 = np.array([np.array(
                [int(n) for n in list(D[data].replace('.', '').replace(' ', '').replace('[', '').replace(']', ''))]) for
                                      data in train[:, 1]])

            with open('{}/维修程度.json'.format(onehot_path)) as f:
                D = json.load(f)
            intermediary4 = np.array([np.array(
                [int(n) for n in list(D[data].replace('.', '').replace(' ', '').replace('[', '').replace(']', ''))]) for
                                      data in train[:, 2]])

            with open('{}/国别.json'.format(onehot_path)) as f:
                D = json.load(f)
            intermediary5 = np.array([np.array(
                [int(n) for n in list(D[data].replace('.', '').replace(' ', '').replace('[', '').replace(']', ''))]) for
                                      data in train[:, 3]])

            with open('{}/厂牌.json'.format(onehot_path)) as f:
                D = json.load(f)
            intermediary6 = np.array([np.array(
                [int(n) for n in list(D[data].replace('.', '').replace(' ', '').replace('[', '').replace(']', ''))]) for
                                      data in train[:, 4]])

            with open('{}/车系.json'.format(onehot_path)) as f:
                D = json.load(f)
            intermediary7 = np.array([np.array(
                [int(n) for n in list(D[data].replace('.', '').replace(' ', '').replace('[', '').replace(']', ''))]) for
                                      data in train[:, 5]])

            with open('{}/修理厂类型.json'.format(onehot_path)) as f:
                D = json.load(f)
            intermediary8 = np.array([np.array(
                [int(n) for n in list(D[data].replace('.', '').replace(' ', '').replace('[', '').replace(']', ''))]) for
                                      data in train[:, 6]])

            with open('{}/修理厂类型.json'.format(onehot_path)) as f:
                D = json.load(f)
            intermediary9 = np.array([np.array(
                [int(n) for n in list(D[data].replace('.', '').replace(' ', '').replace('[', '').replace(']', ''))]) for
                                      data in train[:, 7]])

            with open('{}/是否承修厂牌.json'.format(onehot_path)) as f:
                D = json.load(f)
            intermediary10 = np.array([np.array(
                [int(n) for n in list(D[data].replace('.', '').replace(' ', '').replace('[', '').replace(']', ''))]) for
                                       data in train[:, 8]])

            num = []
            num.append(len(intermediary0[0]))
            num.append(len(intermediary1[0]))
            num.append(len(intermediary4[0]))
            num.append(len(intermediary5[0]))
            num.append(len(intermediary6[0]))
            num.append(len(intermediary7[0]))
            num.append(len(intermediary8[0]))
            num.append(len(intermediary9[0]))
            num.append(len(intermediary10[0]))

            trains = np.zeros(shape=(len(intermediary0), sum(num)))
            for i in range(len(intermediary0)):
                trains[i, :num[0]] = intermediary0[i]
                trains[i, num[0]:sum(num[:2])] = intermediary1[i]
                trains[i, sum(num[:2]):sum(num[:3])] = intermediary4[i]
                trains[i, sum(num[:3]):sum(num[:4])] = intermediary5[i]
                trains[i, sum(num[:4]):sum(num[:5])] = intermediary6[i]
                trains[i, sum(num[:5]):sum(num[:6])] = intermediary7[i]
                trains[i, sum(num[:6]):sum(num[:7])] = intermediary8[i]
                trains[i, sum(num[:7]):sum(num[:8])] = intermediary9[i]
                trains[i, sum(num[:8]):sum(num)] = intermediary10[i]
        trainss = np.concatenate((trains, train2), axis=1)
        print('开始')
        xgtrain = xgb.DMatrix(trainss, label=labels)
        params = {
            'booster': 'gbtree',
            # 这里手写数字是0-9，是一个多类的问题，因此采用了multisoft多分类器，
            'objective': 'reg:squarederror',
            'gamma': 0.05,  # 在树的叶子节点下一个分区的最小损失，越大算法模型越保守 。[0:]
            'max_depth': 7,  # 构建树的深度 [1:]
            # 'lambda':450,  # L2 正则项权重
            'subsample': 0.5,  # 采样训练数据，设置为0.5，随机选择一般的数据实例 (0:1]
            'colsample_bytree': 1,  # 对特征的采样比例用来控制每棵随机采样的列数的占比(每一列是一个特征)
            # 'min_child_weight':12, # 节点的最少特征数
            'silent': 0,
            'eta': 0.008,  # 如同学习率
            'seed': 710,
            'nthread': thread,  # cpu 线程数,根据自己U的个数适当调整
        }
        plst = list(params.items())
        num_rounds = 800  # 迭代你次数
        model = xgb.train(plst, xgtrain, num_rounds)
        preds = model.predict(xgtrain, ntree_limit=model.best_iteration)
        Acc = []
        for i in range(len(preds)):
            acc = exp(-abs(labels[i] - preds[i]) / labels[i] * 2)
            Acc.append(acc)
        Acc = np.array(Acc)
        print('平均准确率：',r2_score(labels, preds))  # [offset:]
        dataset['预测费用'] = np.c_[preds]
        dataset['准确率'] = np.c_[Acc]
        dataset.to_excel(train_output)
        model.save_model(model_name)