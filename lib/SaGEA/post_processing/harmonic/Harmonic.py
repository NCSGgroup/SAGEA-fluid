import numpy as np

from lib.SaGEA.auxiliary.preference.EnumClasses import PhysicalDimensions
from lib.SaGEA.auxiliary.aux_tool.MathTool import MathTool


class Harmonic:
    """
    Harmonic analysis and synthesis: Ordinary 2D integration for computing Spherical Harmonic coefficients
    """

    def __init__(self, lat, lon, lmax: int, option=0):
        """

        :param lat: Co-latitude if option=0, unit[rad]; geophysical latitude if option = others, unit[degree]
        :param lon: If option=0, unit[rad]; else unit[degree]
        :param lmax: int, max degree/order
        :param option:
        """
        if option != 0:
            self.lat, self.lon = MathTool.get_colat_lon_rad(lat, lon)
        else:
            self.lat, self.lon = lat, lon

        self.lmax = lmax

        self._prepare()

    def _prepare(self):
        """pre-process some parameters of the 'two-step' method."""
        self.nlat, self.nlon = len(self.lat), len(self.lon)

        self.pilm = MathTool.get_Legendre(self.lat, self.lmax)
        # Associative Legendre polynomials, indexes stand for (co-lat[rad], degree l, order m)
        # 3-d array [theta, l, m] shape: (nlat * (lmax + 1) * (lmax + 1))

        # self.si = np.sin(self.lat) * np.pi / (2 * self.lmax + 1)

        # get Neumann weights
        # x_mat = np.array([np.cos(self.lat) ** i for i in range(self.nlat)])
        # r_vec = np.ones(self.nlat) * 2 / np.arange(1, self.nlat + 1, 1)
        # r_vec[np.arange(1, self.nlat + 1, 2)] = 0
        # self.wi = np.linalg.pinv(x_mat) @ r_vec

        m = np.arange(self.lmax + 1)
        self.g = m[:, None] @ self.lon[None, :]

        self.factor1 = np.ones((self.nlat, self.lmax + 1))
        self.factor1[:, 0] += 1
        self.factor1 = 1 / (self.factor1 * self.nlon)

        self.factor2 = np.ones((self.lmax + 1, self.lmax + 1))
        self.factor2[:, 0] += 1
        self.factor2 *= np.pi / (2 * self.nlat)

        self.factor3 = np.ones((self.lmax + 1, self.lmax + 1))
        self.factor3[:, 0] += 1
        self.factor3 /= 2
        pass

    def analysis(self, gqij: np.ndarray, special_type: PhysicalDimensions = None):
        assert len(gqij.shape) in (2, 3)

        single = (len(gqij.shape) == 2)
        if single:
            gqij = np.array([gqij])

        assert special_type in (
            PhysicalDimensions.HorizontalDisplacementEast, PhysicalDimensions.HorizontalDisplacementNorth, None)

        if special_type in (
                PhysicalDimensions.HorizontalDisplacementEast, PhysicalDimensions.HorizontalDisplacementNorth):
            assert False, "Horizontal Displacement is not supported yet."

        g = self.g
        co = np.cos(g)  # cos(m phi)
        so = np.sin(g)  # sin(m phi)

        am = np.einsum('pij,mj->pim', gqij, co, optimize='greedy') * self.factor1
        bm = np.einsum('pij,mj->pim', gqij, so, optimize='greedy') * self.factor1


        if special_type is None:
            cqlm = np.einsum('pim,ilm,i->plm', am, self.pilm, np.sin(self.lat), optimize='greedy') * self.factor2
            sqlm = np.einsum('pim,ilm,i->plm', bm, self.pilm, np.sin(self.lat), optimize='greedy') * self.factor2

            # cqlm = np.einsum('pim,ilm,i->plm', am, self.pilm, self.wi, optimize='greedy') * self.factor3
            # sqlm = np.einsum('pim,ilm,i->plm', bm, self.pilm, self.wi, optimize='greedy') * self.factor3

            # cqlm[:, :, 0] = (np.einsum('pim,ilm->plm', am * self.wi[:, None], self.pilm,
            #                               optimize='greedy') * self.factor3)[:, :, 0]
            # sqlm[:, :, 0] = (np.einsum('pim,ilm->plm', bm * self.wi[:, None], self.pilm,
            #                            optimize='greedy') * self.factor3)[:, :, 0]

            # cqlm = np.einsum('pim,ilm->plm', am * self.wi[:, None], self.pilm,
            #                  optimize='greedy') * self.factor3
            # sqlm = np.einsum('pim,ilm->plm', bm * self.wi[:, None], self.pilm,
            #                  optimize='greedy') * self.factor3

        elif special_type == PhysicalDimensions.HorizontalDisplacementNorth:
            """do NOT use this code!"""

            pilm_derivative = MathTool.get_Legendre_derivative(self.lat, self.lmax)

            cqlm = np.einsum('pim,ilm,i->plm', -am, pilm_derivative, np.sin(self.lat)) * self.factor2
            sqlm = np.einsum('pim,ilm,i->plm', -bm, pilm_derivative, np.sin(self.lat)) * self.factor2

        elif special_type == PhysicalDimensions.HorizontalDisplacementEast:
            """do NOT use this code!"""

            mrange = np.arange(self.lmax + 1)
            pilm_divide_sin_theta = self.pilm / np.sin(self.lat)[:, None, None]
            m_pilm_divide_sin_theta = np.einsum("m,ilm->ilm", mrange, pilm_divide_sin_theta)

            am_east = np.einsum('pij,mj->pim', gqij, -so, optimize='greedy') * self.factor1
            bm_east = np.einsum('pij,mj->pim', gqij, co, optimize='greedy') * self.factor1

            cqlm = np.einsum('pim,ilm,i->plm', am_east, m_pilm_divide_sin_theta, np.sin(self.lat)) * self.factor2
            sqlm = np.einsum('pim,ilm,i->plm', bm_east, m_pilm_divide_sin_theta, np.sin(self.lat)) * self.factor2

        else:
            assert False

        if single:
            assert cqlm.shape[0] == 1 and sqlm.shape[0] == 1
            return cqlm[0], sqlm[0]

        else:
            return cqlm, sqlm

    def synthesis(self, cqlm: np.ndarray, sqlm: np.ndarray, special_type: PhysicalDimensions = None):
        assert cqlm.shape == sqlm.shape
        assert len(cqlm.shape) in (2, 3)
        single = (len(cqlm.shape) == 2)
        if single:
            cqlm = np.array([cqlm])
            sqlm = np.array([sqlm])

        assert special_type in (
            PhysicalDimensions.HorizontalDisplacementEast, PhysicalDimensions.HorizontalDisplacementNorth, None)

        cqlm = np.array(cqlm)
        sqlm = np.array(sqlm)

        if special_type is None:
            am = np.einsum('ijk,ljk->ilk', cqlm, self.pilm)
            bm = np.einsum('ijk,ljk->ilk', sqlm, self.pilm)

        elif special_type is PhysicalDimensions.HorizontalDisplacementNorth:
            pilm_derivative = MathTool.get_Legendre_derivative(self.lat, self.lmax)

            am = np.einsum('ijk,ljk->ilk', -cqlm, pilm_derivative)
            bm = np.einsum('ijk,ljk->ilk', -sqlm, pilm_derivative)
        elif special_type is PhysicalDimensions.HorizontalDisplacementEast:
            pilm_divide_sin_theta = self.pilm / np.sin(self.lat)[:, None, None]
            mrange = np.arange(self.lmax + 1)

            am = np.einsum('ijk,k,ljk->ilk', sqlm, mrange, pilm_divide_sin_theta)
            bm = np.einsum('ijk,k,ljk->ilk', -cqlm, mrange, pilm_divide_sin_theta)
        else:
            assert False

        co = np.cos(self.g)
        so = np.sin(self.g)

        gqij = am @ co + bm @ so
        if single:
            assert gqij.shape[0] == 1
            return gqij[0]
        else:
            return gqij

    def synthesis_1D(self,cqlm: np.ndarray):

        SHF = self.get_spherical_harmonic_function()
        Upsilon = SHF["Upsilon"]
        # print(f"Upsilon shape is: {Upsilon.shape},{SH.value.shape}")
        grid = np.einsum("wjl,gl->gwj", Upsilon, cqlm)
        return grid

    def get_spherical_harmonic_function(self):
        co = np.cos(self.g)
        so = np.sin(self.g)
        Upsilon_co = np.einsum("wnm,mj->wjnm", self.pilm, co)
        Upsilon_so = np.einsum("wnm,mj->wjnm", self.pilm, so)
        N = int((self.lmax+2)*(self.lmax+1)/2+(self.lmax)*(self.lmax+1)/2)
        # print(N)
        Upsilon = np.zeros((len(Upsilon_co[:,0,0,0]),len(Upsilon_co[0,:,0,0]),N))
        index = 0
        for n in np.arange(self.lmax+1):
            for m in np.arange(-n,n+1):
                if m<0:
                    Upsilon[:,:,index] = Upsilon_so[:,:,n,np.abs(m)]
                    index+=1
                else:
                    Upsilon[:,:,index] = Upsilon_co[:,:,n,m]
                    index+=1
        SHF = {"Upsilon":Upsilon, "Upsilon_co":Upsilon_co,
               "Upsilon_so":Upsilon_so}
        return SHF


    def synthesis_steps(self,cqlm,sqlm):
        co = np.cos(self.g)
        so = np.sin(self.g)
        Upsilon_co = np.einsum("wnm,mj->wjnm", self.pilm, co)
        Upsilon_so = np.einsum("wnm,mj->wjnm", self.pilm, so)

        Am = np.einsum("inm,wjnm->iwj", cqlm, Upsilon_co)
        Bm = np.einsum("inm,wjnm->iwj", sqlm, Upsilon_so)

        grid = Am+Bm
        return grid



def demo():
    c = np.ones()
    lat = np.arange(90.1,-90.1,-1)
    lon = np.arange(0,360,0.5)
    a = Harmonic(lat=lat,lon=lon,lmax=180,option=1)
    gqij = np.ones((181,361,720))
    print(gqij)
    C,S = a.analysis(gqij=gqij)
    # print(C)
    print('\nSynthesis:\n')
    print(a.synthesis(cqlm=C,sqlm=S))


def quick_fig(grid,lat=None,lon=None,maxvalue=2,savefile=None,unit="cm"):
    import pygmt
    import xarray as xr
    print(f"data of figure max/min:{np.max(grid)},{np.min(grid)}")

    fig_data = xr.DataArray(data=grid, dims=['lat', 'lon'], coords={'lat': lat, 'lon': lon})

    fig = pygmt.Figure()
    pygmt.config(FONT_ANNOT_PRIMARY="15p",FONT_LABEL="15p",MAP_FRAME_TYPE="plain",MAP_TITLE_OFFSET="-0.3c")
    pygmt.makecpt(cmap="haxby",series=[-maxvalue,maxvalue,maxvalue/10])
    fig.grdimage(grid=fig_data,projection="Q10c",cmap=True,frame=["a60f30"])
    fig.coast(shorelines="1/0.5p,black",resolution="f")
    fig.colorbar(position='JBC+o0c/1c+w8c+h',
                 frame=f"xa{maxvalue / 2}+lEWH ({unit})")
    # fig.text(position="BR",text=f"{var}",offset='-0.1c/0.2c', font='15p,Helvetica-Bold,black')
    if savefile:
        fig.savefig(savefile)
    fig.show()

def demo1():
    from pysrc.ancillary.load_file.LoadCS import LoadCS
    from lib.SaGEA.data_class.DataClass import SHC

    resolution = 1

    C,S = LoadCS().get_CS().get_cs2d()
    SH = SHC(c=C,s=S)
    lat,lon = MathTool.get_global_lat_lon_range(resolution=resolution)

    H = Harmonic(lat=lat,lon=lon,lmax=60,option=1)
    grid1 = H.synthesis(cqlm=C,sqlm=S)
    grid2 = H.synthesis_1D(cqlm=SH.value)


    # grid2 = H.synthesis_steps(cqlm=C,sqlm=S)

    # print(f"Shapes are: {grid1.shape},{grid2.shape}")

    # Cq,Sq = np.zeros_like(C),np.zeros_like(S)
    # Cq[0,0,0] = C[0,0,0]
    # Cq[0,2,1] = C[0,2,1]
    # # Cq[0,2,2] = C[0,2,2]
    # Cq[0,2,0] = C[0,2,0]
    # Sq[0,2,1] = S[0,2,1]
    # Sq[0,2,2] = S[0,2,2]
    #
    # lat1 = np.arange(90 - resolution / 2, -90 - resolution / 2, -resolution)
    # print(f"C,S:{C[0,0,0],S[0,0,0]}")

    # H2 = Harmonic(lat=lat1,lon=lon, lmax=60, option=1)
    # grid1 = H1.synthesis(cqlm=Cq,sqlm=Sq)
    # grid2 = H2.synthesis(cqlm=C,sqlm=S)
    # print(grid1.shape,grid2.shape)
    quick_fig(grid=grid1[0]*100,lat=lat,lon=lon,maxvalue=5)
    quick_fig(grid=grid2[0]*100,lat=lat,lon=lon,maxvalue=5)
    quick_fig(grid=grid1[0]-grid2[0],lat=lat,lon=lon,maxvalue=1)




if __name__ == '__main__':
    demo1()
