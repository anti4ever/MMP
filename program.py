from matplotlib import mlab
from terminaltables import AsciiTable
import xlrd
import math, random, numpy
import matplotlib.pyplot as plt


def def_lambda_ef ():
    book = xlrd.open_workbook('data_book.xlsx')
    sheet = book.sheet_by_index(0)
    a0 = sheet.cell(5,0).value
    a1 = sheet.cell(5,1).value
    l1 = sheet.cell(5,2).value
    l2 = sheet.cell(5,4).value
    a2 = sheet.cell(5,5).value
    l3 = sheet.cell(5,6).value
    a3 = sheet.cell(5,7).value
    l4 = sheet.cell(5,8).value
    a4 = sheet.cell(5,9).value
    l5 = sheet.cell(5,10).value
    a5 = sheet.cell(5,11).value
    l6 = sheet.cell(5,12).value
    a_cem = sheet.cell(5,13).value
    lamb_ef = (1 / ( (1/l1)*math.log((a1/a0),math.e) + (1/l2)*math.log((a2/a1),math.e) + (1/l3)*math.log((a3/a2),math.e) + (1/l4)*math.log((a4/a3),math.e) + (1/l5)*math.log((a5/a4),math.e) + (1/l6)*math.log((a_cem/a5),math.e) )) * math.log((a_cem/a0),math.e)
    return lamb_ef

#print (def_lambda_ef())
lambda_ef = def_lambda_ef()

a0 = 0.057 # Внутренний радиус НКТ, м
a1 = 0.1095 # Наружный радиус НКТ с учетом теплоизоляции, м
### a2 = # 
a = 0.2 # Наружный радиус цементного кольца за внешней колонной скважины, м
#### l_izol = # Протяженность теплоизолированного участка, м
#### lambda_izol = # Коэффициент теплопроводности слоя теплоизоляции НКТ, Вт/(м*град)
lambda_gp = 0.52 # Коэффициент теплопроводности газовой прослойкимежду НКТ и экспл. колонной, Вт/(м*град)
lambda_cem = 0.6 # Коэффициент теплопроводности цементного камня, Вт/(м*град)
t0 = -5 # Среднегодовая температура поверхности массива мерзлых пород, град. цельсия
tf = 20 # Температура добываемого флюида, град. цкльсия
C = 0.5772

h = 50 #Мощность расчитываемого интервала, м

#### Задается для каждого расчетного интервала:

z = 50 # Мощность слоя

plotn_sk = 1800 # плотность скелета мерзлой породы, кг/м3
w_tot = 0.3 # Сумарная весовая влажность мерзлой породы, доля единиц (д.е.)
w_w = 0 # Весовая влажность мерзлой породы за счет незамерзшей воды (д.е)
lambda_t = 1.4 # Коэффициент теплопроводности пород в талом состоянии, Вт/(м*град)
lambda_m = 1.8 # Коэффициент теплопроводности пород в мерзлом состоянии, Вт/(м*град)
tfi = 0 # Температура начала оттаивания мерзлых пород, град. цкльсия / Температура начала фазовых переходов

tau = 30 # Время эксплуатации, годы
speed = 0.001 #Точность вычисления / скорость

class Layer(object):
    def __init__(self, z, plotn_sk, w_tot, w_w, lambda_t, lambda_m, tfi, lambda_ef):
        self.z = z
        self.plotn_sk = plotn_sk
        self.w_tot = w_tot
        self.w_w = w_w
        self.lambda_t = lambda_t
        self.lambda_m = lambda_m
        self.tfi = tfi
        self.lambda_ef = lambda_ef #((1 / (((1/lambda_gp)*math.log((a1/a0),math.e)+((1/lambda_cem)*math.log((a/a1),math.e)))))*math.log((a/a0),math.e))
        self.ps_i = (math.log((a/a0),math.e))/(math.log((2*h/a),math.e))
        self.tc = tf*((1+(lambda_m*t0*self.ps_i/self.lambda_ef/tf))/(1+(self.lambda_t*self.ps_i/self.lambda_ef)))
        self.beta = -1*(self.lambda_m*(t0-self.tfi))/(self.lambda_t*(self.tc-self.tfi))
        self.alfa = (335000 * self.plotn_sk * (self.w_tot-self.w_w)) / (31100000 * self.lambda_t * (self.tc - self.tfi))

    def info(self):
        table_data = [
        ['H слоя, м', 'plotn_sk', 'w_tot', 'w_w', 'lambda_t', 'lanbda_m','tfi'],
        [str(self.z), str(self.plotn_sk), str(self.w_tot), str(self.w_w), str(self.lambda_t), str(self.lambda_m), str(self.tfi)],    
        ]
        table = AsciiTable(table_data)
        table.title = 'Layer 1'
        print (table.table)
    
    def r_r(self,z):
        z = z
        rl=a*(math.pow((2*z/a),(1/(1+self.beta))))
        print('rl:',z,rl)
        r = a + 0.01
        tau0 = 0
        d_frz_0 = -1 * (1-(C/math.log((2*z/a),math.e)))/(r*math.log((2*z/a),math.e))
        frz_0 = 1 - ((math.log((r/a),math.e) / (math.log((2*z/a),math.e))) * (1 - (C / (math.log((2*z/a),math.e)))))
        int_0 = self.alfa * (1 / (d_frz_0 * (1/(frz_0-1)+self.beta/frz_0)))
        while tau >= tau0:
            d_frz = -1 * (1-(C/math.log((2*z/a),math.e)))/(r*math.log((2*z/a),math.e))
            frz = 1 - ((math.log((r/a),math.e) / (math.log((2*z/a),math.e))) * (1 - (C / (math.log((2*z/a),math.e)))))
            int_1 = self.alfa *  (1 / (d_frz * (1/(frz-1)+self.beta/frz)))
            tau_d = ((int_0+int_1)/2) * speed #Точность вычисления / скорость
            int_0 = int_1
            r = r + speed #Точность вычисления / скорость
            tau0 = tau0 + tau_d
            if r >= rl:
                r = rl
                break
        return r
   
    def mas_rr(self):
        i = 1
        mas_rad = []
        while i <= self.z:
            mas_rad.append(self.r_r(i))
            print(mas_rad[i-1]) 
            i = i+1
        return mas_rad

Layer1 = Layer(z,plotn_sk,w_tot,w_w,lambda_t,lambda_m,tfi,lambda_ef)

mas_plot_rr = Layer1.mas_rr()

plot_z = numpy.arange(1,z+1)

plt.grid(True)
plt.gca().invert_yaxis()
plt.plot(mas_plot_rr,plot_z)
Layer1.info()
plt.show()
