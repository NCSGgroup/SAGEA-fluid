# -*- coding: utf-8 -*-
# @Author  : Weihang Zhang
# @Email   : zhang.17371@osu.edu
# @File    : Plot_Xover_1D.py
# @Create  : 10/24/26 21:11
# @Project : SWOT
# @Software: PyCharm
"""
Description:
"""
import os
from pysrc.SWOT.KaRIn.L2LRSSH.SSH_Expert import Load_SSH
from pysrc.SWOT.Settings.SWOTClasses import TrackType
from pysrc.SWOT.SWOTKits.Statistics import round_away_zero,round_remove_decimal
import numpy as np
import pygmt

OutputPath = "../../../result/SWOT/Fig_Nov26/"
if not os.path.exists(OutputPath):
    os.mkdir(OutputPath)

def Process_Xover(cNo=30,pNo=10,index_num=[10],track_type=TrackType.Along_Track):
    along_index,cross_index = [43],[4573] # along is the columns, range from 0-68, cross is the rows, range from 0-9865
    along_index = index_num
    cross_index = index_num
    dataset = (Load_SSH(InputPath="/Users/zhang.17371/Documents/").
               get_ssh_pass(cycle_No=cNo, pass_No=pNo))

    xover = dataset["height_cor_xover"]
    xover_qual = dataset["height_cor_xover_qual"]
    lat,lon = dataset['lat'],dataset['lon']
    print(f"\nxover,xover_qual,lat,lon:"
          f"{xover.shape},{xover_qual.shape},"
          f"{lat.shape},{lon.shape}")
    x_sets,y_sets,y_q_sets = [],[],[]
    fig_region = []
    if track_type is TrackType.Along_Track:
        for i in np.arange(len(along_index)):
            x_axis = lat[:,along_index[i]]
            y_axis = xover[:,along_index[i]]
            y_qual = xover_qual[:,along_index[i]]
            x_sets.append(x_axis)
            y_sets.append(y_axis)
            y_q_sets.append(y_qual)
        fig_y = [y_sets,y_q_sets]
        if len(along_index)==1:
            fig_region = [[round_remove_decimal(num=np.nanmin(y_axis)) + 0.5, round_away_zero(num=np.nanmax(y_axis)) + 0.5,
                           -90, 90],
                           [-1, 3, -90, 90]]
        else:
            fig_region = [[-1.5,1.5,-90, 90],
                          [-1, 3, -90, 90]]
    else:

        for i in np.arange(len(cross_index)):
            x_axis = lon[cross_index[i],:]
            y_axis = xover[cross_index[i],:]
            y_qual = xover_qual[cross_index[i],:]
            x_sets.append(x_axis)
            y_sets.append(y_axis)
            y_q_sets.append(y_qual)
            # fig_region.append(region)
        # print(f"{cross_index}th {track_type.name} info x/y/y_qual:{x_axis.shape},{y_axis.shape},{y_qual.shape}")
        fig_y = [y_sets,y_q_sets]
        if len(along_index)==1:
            # fig_region = [[-1.5,1.5,-90, 90],]
            fig_region = [[round_away_zero(num=np.nanmin(x_axis) - 1), round_away_zero(num=np.nanmax(x_axis)),
                           round_away_zero(num=np.nanmin(y_axis)), round_away_zero(num=np.nanmax(y_axis))],
                          [round_away_zero(num=np.nanmin(x_axis) - 1), round_away_zero(num=np.nanmax(x_axis)),
                           -1, 3]]
        else:
            fig_region = [[np.nanmin(np.array(x_sets))-0.1,
                           np.nanmax(np.array(x_sets))+0.1,
                           -3,3],
                          [np.nanmin(np.array(x_sets))-0.1,
                           np.nanmax(np.array(x_sets))+0.1,
                           -1,3]]


    print(f"\n{track_type.name} regions is:{fig_region}")
    print(f"{along_index}th {track_type.name} info x/y/y_qual:{x_axis.shape},{y_axis.shape},{y_qual.shape}")
    print(f"regions: [{np.nanmin(x_axis)},{np.nanmax(x_axis)},{np.nanmin(y_axis)},{np.nanmax(y_axis)}]")
    return x_sets,fig_y,fig_region
    # return x_axis,fig_y,fig_region

def Plot_Xover_Along(cNo=30,pNo=10,index_num=43):
    """set the plotting info"""
    x_axis,fig_y,fig_regions = Process_Xover(cNo=cNo,pNo=pNo,index_num=index_num,
                                             track_type=TrackType.Along_Track)
    fig_colors = ['red', 'blue', 'green', 'orange', 'purple', 'pink']
    fig_frames = [["WSne", "xa0.5f0.1g0.1+l(m)", "ya30f10g5+l Latitude"],
                  ["wSnE", "xa1f1g1+l (Dimensionless)", "ya30f10g5+l Latitude"],]
    fig_texts = [f"(a) Cycle{cNo}-Pass{pNo}", "(b) Quality flag, zero means good"]

    fig = pygmt.Figure()
    pygmt.config(
        FONT_LABEL='15p,5,black', FONT_ANNOT_PRIMARY='15p,5,black',
        FONT_TITLE='20,5,black',
        COLOR_NAN='grey',
        MAP_TITLE_OFFSET='0.2c', MAP_FRAME_TYPE='plain',
        MAP_TICK_LENGTH_PRIMARY='8p', MAP_FRAME_PEN='2p,black',
        MAP_TICK_LENGTH_SECONDARY='5p',
        MAP_TICK_PEN_PRIMARY='1.5p,black', MAP_TICK_PEN_SECONDARY='1.5p,black',
        MAP_GRID_PEN_PRIMARY='gray,dashed'
    )
    for j in np.arange(len(fig_y)):
        fig.basemap(region=fig_regions[j], projection="X10c/20c",
                    frame=fig_frames[j])
        # for i in np.arange(len(y_series)):
        for i in np.arange(len(fig_y[j])):
            with pygmt.config(IO_NAN_RECORDS="pass"):
                fig.plot(y=x_axis[i], x=fig_y[j][i], pen=f"1.5p,{fig_colors[i]}",
                         label=f"Colunm No.{index_num[i]}")
        fig.legend(position="BL", box=False)
        fig.text(position="TL", text=f"{fig_texts[j]}",
                 offset="0.1c/-0.1c", font='17p,5-Bold,black')
        fig.shift_origin(xshift='11c')

    fig.savefig(f"{OutputPath}/Xover_Along_Cyc{cNo}_Pas{pNo}_Column{index_num}.png")
    fig.show()
    pass

def Plot_Xover_Cross(cNo=30,pNo=10,index_num=[4321]):
    """set the plotting info"""
    x_axis, fig_y, fig_regions = Process_Xover(cNo=cNo, pNo=pNo, index_num=index_num,
                                               track_type=TrackType.Cross_Track)
    fig_colors = ['red', 'blue', 'green', 'orange', 'purple', 'pink']
    fig_frames = [["WSne", "xa0f0.1g0.1", "ya1f0.2g0.2+l (meter)"],
                  ["WSne", "xa0.2f0.1g0.1", "ya1f1g1+l (Dimensionless)"]]

    fig_texts = [f"(a) Cycle{cNo}-Pass{pNo}", "(b) Quality flag, zero means good"]

    fig = pygmt.Figure()
    pygmt.config(
        FONT_LABEL='15p,5,black', FONT_ANNOT_PRIMARY='15p,5,black',
        FONT_TITLE='20,5,black',
        COLOR_NAN='grey',
        MAP_TITLE_OFFSET='0.2c', MAP_FRAME_TYPE='plain',
        MAP_TICK_LENGTH_PRIMARY='8p', MAP_FRAME_PEN='2p,black',
        MAP_TICK_LENGTH_SECONDARY='5p',
        MAP_TICK_PEN_PRIMARY='1.5p,black', MAP_TICK_PEN_SECONDARY='1.5p,black',
        MAP_GRID_PEN_PRIMARY='gray,dashed'
    )
    for j in np.arange(len(fig_y)):
        print(f"j:{j},length is:{len(fig_y)}")
        fig.basemap(region=fig_regions[j], projection="X20c/8c",
                    frame=fig_frames[j])
        for i in np.arange(len(fig_y[j])):
            print(f"j:{j},i:{i}")
            # print(f"fig_colors: {fig_color_colors[i]},length is:{len(fig_colors)}")
            with pygmt.config(IO_NAN_RECORDS="pass"):
                fig.plot(x=x_axis[i], y=fig_y[j][i], pen=f"1.5p,{fig_colors[i]}",
                         label=f"Row No.{index_num[i]}")  # 绘制 SSH 随纬度变化的折线
        fig.legend(position="BL", box=False)
        fig.text(position="TR", text=f"{fig_texts[j]}",
                 offset="-0.1c/-0.1c", font='17p,5-Bold,black')
        fig.shift_origin(yshift='-9c')

    fig.savefig(f"{OutputPath}/Xover_Cross_Cyc{cNo}_Pas{pNo}_Row{index_num}.png")
    fig.show()

def demo_Plot(plottype=TrackType.Along_Track):
    cNo,pNo=30,11
    along_index,cross_index = [20,27,30,34,45,50],[4035,4120,4332,4453]
    if plottype==TrackType.Along_Track:
        Plot_Xover_Along(cNo=cNo, pNo=pNo,index_num=along_index)
    else:
        Plot_Xover_Cross(cNo=cNo, pNo=pNo, index_num=cross_index)

if __name__ == "__main__":
    demo_Plot(TrackType.Cross_Track)
