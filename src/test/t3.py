#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


def main():
    x_start, x_stop = 0, 2*np.pi
    point_cnt = 100
    x_step = (x_stop-x_start)/(point_cnt-1)
    xs = np.linspace(x_start, x_stop, point_cnt)
    x_value = x_stop
    series_sin, series_cos = np.sin(xs), np.cos(xs)

    # 打开交互模式（非阻塞，代码可以继续执行）
    plt.ion()
    fig, ax = plt.subplots()

    def draw():
        ax.imshow(np.random.random(size=(256, 256, 3)))
        # ax.plot(xs, series_sin, label=r'$y=sin(x)$')
        # ax.plot(xs, series_cos, label=r'$y=cos(x)$')
        # ax.xaxis.set_major_locator(MultipleLocator(1))
        # ax.xaxis.set_minor_locator(MultipleLocator(0.2))
        # ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        # ax.yaxis.set_major_locator(MultipleLocator(1))
        # ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        # ax.set_title('Matplotlib in PySide2 demo')
        # ax.set_xlabel(r'$x$')
        # ax.set_ylabel(r'$y$')
        # ax.legend()
        # ax.grid()
        # ax.axis('equal')

    # 第一次绘制
    draw()

    for _ in range(100):
        # 清除上一次的图表
        ax.cla()

        xs = np.delete(xs, 0)
        series_sin = np.delete(series_sin, 0)
        series_cos = np.delete(series_cos, 0)
        x_value += x_step
        xs = np.append(xs, x_value)
        series_sin = np.append(series_sin, np.sin(x_value))
        series_cos = np.append(series_cos, np.cos(x_value))

        # 重新绘制
        draw()

        # 暂停之前会更新和显示图表
        plt.pause(0.01)

    # 关闭交互模式
    plt.ioff()
    # 需要调用show，否则直接退出
    plt.show()


if __name__ == '__main__':
    main()
