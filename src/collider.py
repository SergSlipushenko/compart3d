#!/usr/bin/env python
# -*- coding: utf-8 -*-

almostzero = 1e-10
FIGTYPES = ['png', 'eps']

import matplotlib
font = {'size': 26}

matplotlib.rc('font', **font)
matplotlib.rc('text', usetex=True)


import copy
import numpy as np
import matplotlib.pyplot as plt


def figsave(name):
    for figtype in FIGTYPES:
        plt.savefig('%s.%s' % (name, figtype))

class Particle:
    def __init__(self, m=1., R=1., v=(0, 0, 0), r = (0, 0, 0)):
        self.m = float(m)
        self.R = float(R)
        self.v = np.array(v, float)
        self.r = np.array(r, float)

    def __repr__(self):
        return "mass = {} radius = {} \nr = {} \nv = {}\n".format(
            self.m,self.R,self.r,self.v)

def tracepair(p0,p1):
    '''
    Вычисляется время до столкновения частиц.
    Если столкновения не найдено возвращается False
    '''
    dv = p0.v - p1.v
    dr = p0.r - p1.r
    drdv = np.dot(dr,dv)
    dr2 = np.dot(dr,dr)
    dv2 = np.dot(dv,dv)
    if dv2 < almostzero : return False
    dR2 = (p0.R + p1.R)**2
    delta2 = drdv**2 - dv2*(dr2 - dR2)
    if delta2 < 0 : return False
    t0 = (- drdv + delta2**0.5)/dv2
    t1 = (- drdv - delta2**0.5)/dv2
    if t0 >= almostzero :
        if t1 > almostzero : return min(t0,t1)
        else:
            return t0
    else:
        if t1 >= almostzero : return t1
        else:
            return False


def collide(p0, p1):
    n = p0.r - p1.r
    # вектор между центрами
    n = n/((n**2).sum())**.5
    # нашли продольные скорости
    v0 = np.dot(p0.v,n)
    v1 = np.dot(p1.v,n)
    # вычислили поперечные компоненты скоростей. они сохраняются.
    v0_c = p0.v - n*v0
    v1_c = p1.v - n*v1
    M = p0.m + p1.m
    # вычислили продольные новые скорости
    u0 = (p0.m - p1.m)/M*v0 + 2*p1.m/M*v1
    u1 = 2*p0.m/M*v0 + (p1.m - p0.m)/M*v1
    # вычислили новые вектора
    p0.v = v0_c + n*u0
    p1.v = v1_c + n*u1
    return p0, p1

def traceall (particles, structure):
    '''
    Выполняет расчет времени до столкновения всех частиц в системе.
    С учетом структуры перебираются все возможные пары.
    В матрицу-результат попадают только положительные времена.
    Элементы в матрице-результате ниже диалонали маскируются.
    '''
    n = len(particles)
    flighttime = np.ma.array(np.zeros((n, n)), mask=np.ones((n, n)))
    for i,p0 in enumerate(particles):
        for j,p1 in enumerate(particles[i + 1:]):
            if structure[i,j + i + 1] == 1 :
                t = tracepair(p0,p1)
                if t :
                    flighttime.mask[i,j + i + 1] = False
                    flighttime.data[i,j + i + 1] = t
    return flighttime

def tracechanged (particles,structure,_flighttime,pairs, clean = False):
    '''
    Выполняет расчет времени до столкновения частиц в системе,
    изменивших скорость после столкновения.
    В матрицу-результат попадают только положительные времена.
    Элементы в матрице-результате ниже диалонали маскируются.
    '''
    if clean : flighttime = copy.copy(_flighttime)
    else : flighttime = _flighttime
    changed = np.array(pairs).flatten()
    n = len(particles)
    for i,p0 in enumerate(particles):
        for j,p1 in enumerate(particles[i + 1:]):
            if (i in changed) or (j + i + 1 in changed) :
                if structure[i,j + i + 1] == 1 :
                    t = tracepair(p0, p1)
                    if t :
                        flighttime.mask[i, j + i + 1] = False
                        flighttime.data[i, j + i + 1] = t
                    else :
                        flighttime.mask[i, j + i + 1] = True
    return flighttime

def move(_particles, _flighttime, clean = False):
    '''
    Двигаем частицы до первого столкновения.
    Возвращаем :
    передвинутые частицы
    новый массив времен
    время полета до столкновения
    кортеж пар столкнувшихся частиц
    '''
    if clean :
        particles = copy.copy(_particles)
        flighttime = copy.copy(_flighttime)
    else :
        particles = _particles
        flighttime = _flighttime
    # Движение
    t = flighttime.min()

    for p in particles:
        p.r += p.v*t

    flighttime -= t
    # Ищем столкнувшиеся пары
    z = flighttime.flatten() < almostzero
    indx =  np.where(np.logical_and(z.data,np.logical_not(z.mask)))[0]
    n = len(particles)
    # Оптимизация на наиболее частого случая
    if len(indx) == 1:
        return particles, flighttime, t, [np.unravel_index(indx[0], (n,n)),]
    pairs = [np.unravel_index(i,(n,n)) for i in indx]
    collidingparts = np.array(pairs).flatten()
    unique_collidingparts = np.unique(collidingparts)
    if len(collidingparts) <> len(unique_collidingparts):
        raise ArithmeticError, "Multibody collision detected!!!"
    return particles, flighttime,t,pairs


def get_totals(particles):
    total_m = total_mi = 0
    total_e = 0
    total_p = np.array((0, 0, 0), float)
    total_l = np.array((0, 0, 0), float)
    for n, p in enumerate(particles):
        total_e += p.m*(p.v**2).sum()/2.
        total_p += p.v*p.m
        total_l += p.m*np.cross(p.v, p.r)
        total_m += p.m
        if n != 0:
            total_mi += p.m
    v_c = total_p/total_m

    return total_m, total_mi, total_e, total_p, total_l, v_c


def run_experiment(particles, steps):
    inner_collide_counter = 0
    # Для оболочек указывается отрицательный радиус!!!
    n_part = len(particles)
    structure = np.ones((n_part, n_part), int)
    r_s = [np.empty((0, 3), float) for i in xrange(n_part)]
    e_s = [[] for i in xrange(n_part)]
    total_m, total_mi, total_e, total_p, total_l, v_c = get_totals(particles)
    f_time = traceall(particles, structure)
    for i in xrange(steps):
        # сбор данных
        r_c = np.array((0, 0, 0), float)
        for k, p in enumerate(particles):
            e_s[k].append(p.m*((p.v - v_c)**2).sum()/2.)
            r_c += p.m*p.r/total_m
            r_s[k] = np.vstack((r_s[k], p.r - r_c))
        # логика столкновений
        particles, f_time, t, pairs = move(particles, f_time, clean=False)
        if pairs[0][0] != 0:
            inner_collide_counter += 1
        for pair in pairs:
            particles[pair[0]], particles[pair[1]] = collide(
                particles[pair[0]], particles[pair[1]])
        # ftime = traceall(particles,structure)
        f_time = tracechanged(particles, structure, f_time, pairs, clean=False)
    print 1.*inner_collide_counter/steps
    return r_s, e_s

def main():

    # particles = [
    #     Particle(m=1., R=-1., v=(0., 0.0, 0.0), r=(0., 0., 0.)),
    #     # Particle(m=0.25, R=0.1, v=(0., 1., 1.), r=(0.1, 0., 0.)),
    #     # Particle(m=0.25, R=0.1, v=(0., 0., 1.), r=(-0.5, 0., 0.)),
    #     Particle(m=1, R=0.1, v=(1., 0., 0.), r=(0.4, 0., 0.)),
    #     Particle(m=1, R=0.1, v=(0., 1., 0.), r=(0.0, 0.1, 0.0)),
    # ]
    # r_s, e_s = run_experiment(particles, steps=50)
    #
    # plt.subplots_adjust(left=0.15, bottom=0.15)
    # plt.ylabel('$E_i$')
    # plt.xlabel('$n$')
    # styles = iter(('-', ':', '--', '-.'))
    # for e_i in e_s:
    #     plt.plot(e_i, color='k', linestyle=next(styles))
    # figsave('energy_ch')
    # plt.clf()

    # particles = [
    #     Particle(m=4., R=-1., v=(0., 0.0, 0.0), r=(0., 0., 0.)),
    #     # Particle(m=1, R=0.1, v=(1., 1., 0.), r=(-0.4, 0., 0.)),
    #     Particle(m=1, R=0.1, v=(1., 0., 0.), r=(0.4, 0., 0.)),
    #     Particle(m=1, R=0.1, v=(0., 1., 0.), r=(0.0, 0.1, 0.0)),
    # ]
    # r_s, e_s = run_experiment(particles, steps=200)
    #
    # plt.subplots_adjust(left=0.15, bottom=0.15)
    # plt.ylabel('$E_i$')
    # plt.xlabel('$n$')
    # styles = iter(('-', ':', '--', '-.'))
    # for e_i in e_s:
    #     plt.plot(e_i, color='k', linestyle=next(styles))
    # figsave('energy_int')
    # plt.clf()

    # particles = [
    #     Particle(m=5., R=-1., v=(0., 0.0, 0.0), r=(0., 0., 0.)),
    #     Particle(m=0.5, R=0.1, v=(0., 1., 1.), r=(0.1, 0., 0.)),
    #     Particle(m=0.5, R=0.1, v=(0., 0., 1.), r=(-0.5, 0., 0.)),
    # ]
    # r_s, e_s = run_experiment(particles, steps=100000)
    # total_m, total_mi, _, _, _, _ = get_totals(particles)
    #
    # e_mean = np.empty_like(e_s)
    # for k, total_e in enumerate(e_s):
    #     mean_sum = 0.
    #     for i, e in enumerate(total_e):
    #         mean_sum += e
    #         e_mean[k, i] = mean_sum/(1 + i)
    #
    # plt.plot([0, len(e_mean[0])],
    #          [particles[0].m/total_mi, particles[0].m/total_mi],
    #          color='k', linestyle='-')
    # styles = iter(('-', '--', '-.', ':'))
    # for total_e in e_mean[1:]:
    #     E_frac_mean = np.empty_like(total_e)
    #     for i, e in enumerate(total_e):
    #         E_frac_mean[i] = total_e[i]/e_mean[0,i]
    #     plt.plot(E_frac_mean, linestyle=next(styles), color='k')
    #
    # plt.subplots_adjust(left=0.15, bottom=0.15)
    # plt.ylabel('${E_i}/{E_0}$')
    # plt.xlabel('$n$')
    # figsave('result2')
    # plt.clf()
    #
    # particles = [
    #     Particle(m=5., R=-1., v=(0., 0.0, 0.0), r=(0., 0., 0.)),
    #     Particle(m=0.3333, R=0.1, v=(1., 0., 0.), r=(0.3, 0., 0.)),
    #     Particle(m=0.3333, R=0.1, v=(0., 0., 1.), r=(-0.5, 0., 0.)),
    #     Particle(m=0.3333, R=0.1, v=(0., 1., 0.), r=(0.0, 0.1, 0.0)),
    # ]
    # r_s, e_s = run_experiment(particles, steps=100000)
    # total_m, total_mi, _, _, _, _ = get_totals(particles)
    #
    # e_mean = np.empty_like(e_s)
    # for k, total_e in enumerate(e_s):
    #     mean_sum = 0.
    #     for i, e in enumerate(total_e):
    #         mean_sum += e
    #         e_mean[k, i] = mean_sum/(1 + i)
    #
    # plt.plot([0, len(e_mean[0])],
    #          [particles[0].m/total_mi, particles[0].m/total_mi],
    #          color='k', linestyle='-')
    # styles = iter(('-', '--', '-.', ':'))
    # for total_e in e_mean[1:]:
    #     E_frac_mean = np.empty_like(total_e)
    #     for i, e in enumerate(total_e):
    #         E_frac_mean[i] = total_e[i]/e_mean[0,i]
    #     plt.plot(E_frac_mean, linestyle=next(styles), color='k')
    #
    # plt.subplots_adjust(left=0.15, bottom=0.15)
    # plt.ylabel('${E_i}/{E_0}$')
    # plt.xlabel('$n$')
    # figsave('result3')
    # plt.clf()

    particles = [
        Particle(m=5., R=-1., v=(0., 0.0, 0.0), r=(0., 0., 0.)),
        Particle(m=0.25, R=0.1, v=(1., 0., 1.), r=(0.1, 0., 0.3)),
        Particle(m=0.25, R=0.1, v=(0., 0., 0.), r=(-0.5, 0., 0.)),
        Particle(m=0.25, R=0.1, v=(0., 0., 1.), r=(0.2, 0., -0.1)),
        Particle(m=0.25, R=0.1, v=(0., 1., 0.), r=(0.0, 0.3, 0.0)),
    ]
    r_s, e_s = run_experiment(particles, steps=200000)
    total_m, total_mi, _, _, _, _ = get_totals(particles)

    e_mean = np.empty_like(e_s)
    for k, total_e in enumerate(e_s):
        mean_sum = 0.
        for i, e in enumerate(total_e):
            mean_sum += e
            e_mean[k, i] = mean_sum/(1 + i)

    plt.plot([0, len(e_mean[0])],
             [particles[0].m/total_mi, particles[0].m/total_mi],
             color='k', linestyle='-')
    styles = iter(('-', '--', '-.', ':'))
    for total_e in e_mean[1:]:
        E_frac_mean = np.empty_like(total_e)
        for i, e in enumerate(total_e):
            E_frac_mean[i] = total_e[i]/e_mean[0,i]
        plt.plot(E_frac_mean, linestyle=next(styles), color='k')

    plt.subplots_adjust(left=0.15, bottom=0.15)
    plt.ylabel('${E_i}/{E_0}$')
    plt.xlabel('$n$')
    figsave('result4')
    plt.clf()


if __name__ == '__main__':
    main()
