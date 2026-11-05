# -*- coding: utf-8 -*-
# @Author  : Weihang Zhang
# @Email   : zhang.17371@osu.edu
# @File    : Plot_2D.py
# @Create  : 10/20/26 16:26
# @Project : PySWOT
# @Software: PyCharm
"""
Description:
"""
import os
# from pysrc.Analysis.Computation import First_Derivation
from pysrc.SWOT.Process.PreProcess import data_filter,data_unfliter
import numpy as np
import pygmt
import xarray as xr
from pysrc.SWOT.Process.ProjectionEN import projection_EN
from pysrc.SWOT.Settings.SWOTClasses import TrackType

OutputPath = "../../../result/SWOT/Fig_Nov26/"
if not os.path.exists(OutputPath):
    os.mkdir(OutputPath)

def Plot(cNo=30,pNo=10):
    dataset = data_filter(cNo=cNo,pNo=pNo,InputPath="/Users/zhang.17371/Documents/")
    lat,lon = dataset['lat'],dataset['lon']
    ssh = dataset['ssh']
    print(f"{dataset['lat'].shape},{dataset['lon'].shape},{dataset['ssh'].shape},{dataset['ssh_da'].shape},{dataset['ssh_dc'].shape}")

    data_y = [dataset['ssh']-dataset['mss']-dataset['mdt'],
              dataset['ssh_da']-dataset['mss_da']-dataset['mdt_da'],
              dataset['ssh_dc']-dataset['mss_dc']-dataset['mdt_dc']]
    grid_y = []
    """Not for difference setup"""
    fig_name = ["(a) SSH",
                "(b) @~\266@~ssh/@~\266@~a",
                "(c) @~\266@~ssh/@~\266@~c"]
    fig_frames = [["a2f1"], ["ya0f1", "xa2f1"], ["ya0f1", "xa2f1"]]
    fig_colorticks = ["xa10f1+lSSH (m)",
                      "xa5f0.5+l@~\264@~10@+-5@+ (Dimensionless)",
                      "xa5f0.5+l@~\264@~10@+-5@+ (Dimensionless)"]
    fig_series = [[-80,-40],[-10,10],[-10,10]]
    """For difference setup"""
    fig_name = ["(a) SSH-MSS-MDT",
                "(b) @~D@~@~\266@~ssh/@~\266@~a",
                "(c) @~D@~@~\266@~ssh/@~\266@~c"]
    fig_frames = [["a2f1"], ["ya0f1", "xa2f1"], ["ya0f1", "xa2f1"]]
    fig_colorticks = ["xa0.5f0.1+lm",
                      "xa0.5f0.1+l@~\264@~10@+-5@+ (Dimensionless)",
                      "xa0.5f0.1+l@~\264@~10@+-5@+ (Dimensionless)"]
    fig_series = [[-1, 1], [-1, 1], [-1, 1]]
    region = [55, 65, 10, 20]  # west, east, south, north
    for i in np.arange(len(data_y)):
        valid = (
                np.isfinite(lon)
                & np.isfinite(lat)
                & np.isfinite(data_y[i])
                & (lon >= region[0]) & (lon <= region[1])
                & (lat >= region[2]) & (lat <= region[3])
        )
        # print(f"valid: {valid.shape}")
        # # 二维布尔索引会提取出彼此对应的一维数组
        # x,y,v = lon[valid],lat[valid],ssh[valid]
        x,y,v = lon[valid],lat[valid],data_y[i][valid]
        print(f" > > {fig_name[i]}")
        print(f"x,y,v shapes are {x.shape, y.shape, v.shape}")
        # print(f"{fig_name[i]}:{np.nanmin(data_y[i])}~{np.nanmax(data_y[i])}")
        print(f"{fig_name[i]}:{np.nanmin(v)}~{np.nanmax(v)}")
        print(f"Effective counts of values:{np.sum(np.isfinite(data_y[i]))}")
        print(f"============================================================\n")
        grid = pygmt.xyz2grd(
            x=x,
            y=y,
            z=v,
            region=region,
            spacing="2k/2k",
            registration="p",
            coltypes="g",
        )
        grid_y.append(grid)
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
    for i in np.arange(len(grid_y)):
        pygmt.makecpt(cmap="panoply",series=fig_series[i])
        fig.grdimage(
            grid=grid_y[i],
            region=region,
            projection="M10c",
            cmap=True,
            interpolation="n",
            frame=fig_frames[i],
        )
        fig.coast(shorelines="0.5p", land="gray90")
        fig.text(position="TL", text=f"{fig_name[i]}",
                 offset="0.1c/-0.1c", font='17p,5-Bold,black')
        fig.colorbar(frame=fig_colorticks[i],
                     position="JBC+o0c/1c+w8c/0.3c+h+e0.5c")
        fig.shift_origin(xshift="11c")
    fig.savefig(f"{OutputPath}/2D_Dif_cycle{cNo}_pass{pNo}_Zhang.png")
    fig.show()

def Plot_Single(cNo=30,TotalPass=584,trackType=TrackType.Cross_Track):
    # region = [0, 360, -90, 90]
    region = [55, 65, 10, 20]
    # region = [120, 135, 30, 45]
    grid_data = projection_EN(cNo=cNo,TotalPass=TotalPass,region=region,
                              tracktype=trackType)
    # grid_data = xr.open_dataset("../../result/tempdata/delta_ssh_030_584_Along_Track.nc")
    text_name = "@~\266@~@~\104@~/@~\266@~c"

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

    pygmt.makecpt(cmap="panoply",series=[-1,1])
    fig.grdimage(
        grid=grid_data,
        region=region,
        # projection="Q10c",
        # frame=["ya30f10", "xa60f10"],
        projection="M10c",
        frame=["ya5f1","xa5f1"],
        cmap=True,
        interpolation="n",

    )
    fig.coast(shorelines="0.5p", land="gray90")
    fig.text(position="TL", text=f"{text_name}",
             offset="0.1c/-0.1c", font='17p,5-Bold,black')
    fig.colorbar(frame="xa0.5f0.1+l@~\264@~10@+-5@+ (Dimensionless)",
                 position="JBC+o0c/1c+w8c/0.3c+h+e0.5c")
    fig.shift_origin(xshift="11c")
    fig.savefig(f"{OutputPath}/2D_Dif_cycle{cNo}_{TotalPass}_{trackType.name}.png")
    print(f"Saved Figure in: {OutputPath}/2D_Dif_cycle{cNo}_{TotalPass}_{trackType.name}.png")
    fig.show()


if __name__ == "__main__":
    Plot_Single()
    pass