# -*- coding: utf-8 -*-
# @Author  : Weihang Zhang
# @Email   : zhang.17371@osu.edu
# @File    : Plot_1D_DRV.py
# @Create  : 11/4/26 18:01
# @Project : PySWOT
# @Software: PyCharm
"""
Description:
"""
import os
import numpy as np
import pygmt
from pysrc.SWOT.KaRIn.L2LRSSH.SSH_Expert import Load_SSH
from pysrc.SWOT.SWOTKits.Statistics import round_remove_decimal,round_away_zero
from pysrc.SWOT.Process.PreProcess import data_filter,data_unfliter
from pysrc.SWOT.SWOTKits.Derivation import derivative_2D,derivative_1D
from pysrc.SWOT.Settings.SWOTClasses import TrackType


OutputPath = "../../../result/SWOT/Fig_Nov26/"
if not os.path.exists(OutputPath):
    os.makedirs(OutputPath)

def Plot_Along(cNo=30,pNo=10,index_num=43):
    """set the plotting info"""
    datasets = data_filter(cNo=cNo,pNo=pNo)
    x_sets,y_sets,mss_set,mdt_set,ged_set = [],[],[],[],[]

    x_axis = datasets['lat'][:,index_num]
    y_ssh = datasets['ssh'][:,index_num]
    y_mss = datasets['mss'][:,index_num]
    y_mdt = datasets['mdt'][:,index_num]
    y_ged = datasets['geoid'][:,index_num]
    fig_y = [y_ssh,y_mss,y_ged,y_mdt]
    fig_regions = []
    for i in np.arange(len(fig_y)):
        region = [round_away_zero(np.nanmin(fig_y[i])),
                  round_away_zero(np.nanmax(fig_y[i])),
                  -90,90]
        print(f"region: {region}")
        fig_regions.append(region)
    fig_colors = ['red', 'blue', 'green', 'orange', 'purple', 'pink']
    fig_frames = [["WSne", "xa20f5g5+l m", "ya30f5g5+l Latitudes"],
                  ["wSne", "xa20f5g5+l m", "ya30f5g5"],
                  ["wSne", "xa20f5g5+l m", "ya30f5g5"],
                  ["wSnE", "xa1f0.2g0.2+l m", "ya30f10g5+l Latitudes"]]
    fig_texts = ["(a) SSH", "(b) MSS","(c) EGM2008", "(d) MDT"]

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
        with pygmt.config(IO_NAN_RECORDS="pass"):
            fig.plot(y=x_axis, x=fig_y[j], pen=f"1.5p,{fig_colors[j]}",
                     label=f"Colunm No.{index_num}")
        fig.legend(position="BL", box=False)
        fig.text(position="TL", text=f"{fig_texts[j]}",
                 offset="0.1c/-0.1c", font='17p,5-Bold,black')
        fig.shift_origin(xshift='11c')

    fig.savefig(f"{OutputPath}/Drv_Org_Along_Cyc{cNo}_Pas{pNo}_column{index_num}.png")
    fig.show()
    pass

def Plot_Cross(cNo=30,pNo=10,index_num=4321):
    """set the plotting info"""
    """set the plotting info"""
    datasets = data_filter(cNo=cNo, pNo=pNo)
    x_axis = datasets['lon'][index_num,:]
    y_ssh = datasets['ssh'][index_num,:]
    y_mss = datasets['mss'][index_num,:]
    y_mdt = datasets['mdt'][index_num,:]
    y_ged = datasets['geoid'][index_num,:]
    fig_y = [y_ssh, y_mss, y_ged, y_mdt]
    fig_regions = []
    for i in np.arange(len(fig_y)):
        region = [round_away_zero(np.nanmin(x_axis)-1),
                  round_away_zero(np.nanmax(x_axis)),
                  round_away_zero(np.nanmin(fig_y[i])-1),
                  round_away_zero(np.nanmax(fig_y[i])+1),]
        if i == 3:
            region = [round_away_zero(np.nanmin(x_axis) - 1),
                      round_away_zero(np.nanmax(x_axis)),
                      round_away_zero(np.nanmin(fig_y[i]))-0.5,
                      round_away_zero(np.nanmax(fig_y[i])), ]

        print(f"region: {region}")
        fig_regions.append(region)

    fig_colors = ['red', 'blue', 'green', 'orange', 'purple', 'pink']
    fig_frames = [["WSne", "xa0f0.1g0.1", "ya1f0.2g0.2+l m"],
                  ["WSne", "xa0f0.1g0.1", "ya1f0.2g0.2+l m"],
                  ["WSne", "xa0f0.1g0.1", "ya1f0.2g0.2+l m"],
                  ["WSne", "xa0.2f0.1g0.1+l Longitude", "ya0.1f0.02g0.02+l m"]]
    fig_texts = ["(a) SSH", "(b) MSS", "(c) EGM2008", "(d) MDT"]

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
        fig.basemap(region=fig_regions[j], projection="X24c/8c",
                    frame=fig_frames[j])
        # for i in np.arange(len(y_series)):
        with pygmt.config(IO_NAN_RECORDS="pass"):
            fig.plot(x=x_axis, y=fig_y[j], pen=f"1.5p,{fig_colors[j]}",
                     label=f"Colunm No.{index_num}")
        fig.legend(position="BL", box=False)
        fig.text(position="TL", text=f"{fig_texts[j]}",
                 offset="0.1c/-0.1c", font='17p,5-Bold,black')
        fig.shift_origin(yshift='-9c')
    fig.savefig(f"{OutputPath}/Drv_Org_Cross_Cyc{cNo}_Pas{pNo}_row{index_num}.png")
    fig.show()

def Plot_Along_Dif(cNo=30,pNo=10,index_num=43):
    """set the plotting info"""
    datasets = data_filter(cNo=cNo, pNo=pNo)
    x_sets, y_sets, mss_set, mdt_set, ged_set = [], [], [], [], []
    x_axis = datasets['lat'][:, index_num]
    y_ssh = datasets['ssh'][:, index_num]
    y_mss = datasets['mss'][:, index_num]
    y_mdt = datasets['mdt'][:, index_num]
    y_ged = datasets['geoid'][:, index_num]

    y_axis = y_ssh-y_ged-y_mdt
    y_drv = derivative_1D(data=y_axis,spacing=0.02,pNo=pNo)

    fig_y = [y_drv]
    fig_regions = []
    for i in np.arange(len(fig_y)):
        region = [round_away_zero(np.nanmin(fig_y[i])),
                  round_away_zero(np.nanmax(fig_y[i])),
                  -90, 90]
        print(f"region: {region}")
        fig_regions.append(region)
    fig_colors = ['red', 'blue', 'green', 'orange', 'purple', 'pink']
    fig_frames = [["WSne", "xa1f0.2g0.2+l@~\264@~10@+-5@+", "ya30f5g5+l Latitudes"],
                  ["wSne", "xa20f5g5+l m", "ya30f5g5"],
                  ["wSne", "xa20f5g5+l m", "ya30f5g5"],
                  ["wSnE", "xa1f0.2g0.2+l m", "ya30f10g5+l Latitudes"]]
    fig_texts = ["@~\266@~@~\104@~/@~\266@~a", "(b) MSS", "(c) EGM2008", "(d) MDT"]

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
        with pygmt.config(IO_NAN_RECORDS="pass"):
            fig.plot(y=x_axis, x=fig_y[j], pen=f"1.5p,{fig_colors[j]}",
                     label=f"Colunm No.{index_num}")
        fig.legend(position="BL", box=False)
        fig.text(position="TL", text=f"{fig_texts[j]}",
                 offset="0.1c/-0.1c", font='17p,5-Bold,black')
        fig.shift_origin(xshift='11c')

    fig.savefig(f"{OutputPath}/Drv_Along_Cyc{cNo}_Pas{pNo}_column{index_num}.png")
    fig.show()
    pass

def Plot_Cross_Dif(cNo=30,pNo=10,index_num=4321):
    """set the plotting info"""
    """set the plotting info"""
    datasets = data_filter(cNo=cNo, pNo=pNo)
    x_axis = datasets['lon'][index_num, :]
    y_ssh = datasets['ssh'][index_num, :]
    y_mss = datasets['mss'][index_num, :]
    y_mdt = datasets['mdt'][index_num, :]
    y_ged = datasets['geoid'][index_num, :]

    y_axis = y_ssh-y_ged-y_mdt
    y_drv = derivative_1D(data=y_axis, spacing=0.02, pNo=pNo)
    fig_y = [y_drv]
    fig_regions = []
    for i in np.arange(len(fig_y)):
        region = [round_away_zero(np.nanmin(x_axis) - 1),
                  round_away_zero(np.nanmax(x_axis)),
                  round_away_zero(np.nanmin(fig_y[i]) - 1),
                  round_away_zero(np.nanmax(fig_y[i]) + 1), ]
        if i == 3:
            region = [round_away_zero(np.nanmin(x_axis) - 1),
                      round_away_zero(np.nanmax(x_axis)),
                      round_away_zero(np.nanmin(fig_y[i])) - 0.5,
                      round_away_zero(np.nanmax(fig_y[i])), ]

        print(f"region: {region}")
        fig_regions.append(region)

    fig_colors = ['red', 'blue', 'green', 'orange', 'purple', 'pink']
    fig_frames = [["WSne", "xa0.2f0.1g0.1+lLatitude", "ya1f0.2g0.2+l@~\264@~10@+-5@+"],
                  ["WSne", "xa0f0.1g0.1", "ya1f0.2g0.2+l m"],
                  ["WSne", "xa0f0.1g0.1", "ya1f0.2g0.2+l m"],
                  ["WSne", "xa0.2f0.1g0.1+l Longitude", "ya0.1f0.02g0.02+l m"]]
    fig_texts = ["@~\266@~@~\104@~/@~\266@~c", "(b) MSS", "(c) EGM2008", "(d) MDT"]

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
        fig.basemap(region=fig_regions[j], projection="X24c/8c",
                    frame=fig_frames[j])
        # for i in np.arange(len(y_series)):
        with pygmt.config(IO_NAN_RECORDS="pass"):
            fig.plot(x=x_axis, y=fig_y[j], pen=f"1.5p,{fig_colors[j]}",
                     label=f"Colunm No.{index_num}")
        fig.legend(position="BL", box=False)
        fig.text(position="TL", text=f"{fig_texts[j]}",
                 offset="0.1c/-0.1c", font='17p,5-Bold,black')
        fig.shift_origin(yshift='-9c')
    fig.savefig(f"{OutputPath}/Drv_Cross_Cyc{cNo}_Pas{pNo}_row{index_num}.png")
    fig.show()

def demo(typetrack=TrackType.Cross_Track):
    cNo,pNo=30,10
    index_along  =43
    index_cross = 4312
    if typetrack == TrackType.Along_Track:
        Plot_Along(cNo=cNo,pNo=pNo,index_num=index_along)
    else:
        Plot_Cross(cNo=cNo,pNo=pNo,index_num=index_cross)

def demo_dif(typetrack=TrackType.Cross_Track):
    cNo,pNo=30,10
    index_along  =43
    index_cross = 4312
    if typetrack == TrackType.Along_Track:
        Plot_Along_Dif(cNo=cNo,pNo=pNo,index_num=index_along)
    else:
        Plot_Cross_Dif(cNo=cNo,pNo=pNo,index_num=index_cross)


if __name__ == "__main__":
    # demo()
    demo_dif()