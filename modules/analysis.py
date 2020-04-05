import numpy as np
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import lmfit as lm
import matplotlib.gridspec as gridspec

class analysis():
    fontSize = 14

    def __init__(self, imageData, xPixSize, yPixSize, minX, maxX, minY, maxY, savePath, date, time):
        # %% Reading in the data
        self.data = imageData.T
        self.pixelX = np.arange(0, np.shape(imageData)[1] + 1, 1)
        self.pixelY = np.arange(0, np.shape(imageData)[0] + 1, 1)

        self.distY = self.pixelX * float(xPixSize)  # Pixel in µm
        self.distX = self.pixelY * float(yPixSize)

        # ROI from main class
        self.xMin = minX
        self.yMin = minY

        self.xMax = maxX
        self.yMax = maxY

        self.name = date + "_" + time
        self.path = savePath
        self.totalPlot()
        self.fitGauss()

    def fireice(self):
        """Returns a self defined analog of the colormap fireice"""
        cdict = {'red': [(0.0, 0.75, 0.75),
                         (1 / 6, 0, 0),
                         (2 / 6, 0, 0),
                         (3 / 6, 0, 0),
                         (4 / 6, 1.0, 1.0),
                         (5 / 6, 1.0, 1.0),
                         (1.0, 1.0, 1.0)],

                 'green': [(0.0, 1, 1),
                           (1 / 6, 1, 1),
                           (2 / 6, 0, 0),
                           (3 / 6, 0, 0),
                           (4 / 6, 0, 0),
                           (5 / 6, 1.0, 1.0),
                           (1.0, 1.0, 1.0)],

                 'blue': [(0.0, 1, 1),
                          (1 / 6, 1, 1),
                          (2 / 6, 1, 1),
                          (3 / 6, 0, 0),
                          (4 / 6, 0, 0),
                          (5 / 6, 0, 0),
                          (1.0, 0.75, 0.75)]}
        fireice = matplotlib.colors.LinearSegmentedColormap('fireice', cdict)
        return (fireice)

    def calcGridBoxes(self, grid):
        """ calculates the size of a grid cell and the left and right boundaries of each gridcell in a monotonous but
          possibly nonlinear grid

           Parameters
           ----------
               vector : 1D numpy array
                        numpy array containing a monotonous grid with points
           Returns
           ------
           in that order
               delta : 1D numpy array of same length as vector
                       the distance to the left and right neighbor devided by 2
               l : 1D numpy array of same length as vector
                   left boundaries of the grid cell
               r : 1D numpy array of same length as vector
                   right boundaries of the grid cell
           Example
           -------
               >>> delta,right,left = calcGridBoxes([0,1,2,4])
                   (array([ 1. ,  1. ,  1.5,  2. ]),
                    array([-0.5,  0.5,  1.5,  3. ]),
                    array([ 0.5,  1.5,  3. ,  5. ]))"""

        delta = np.zeros(len(grid))
        r = np.zeros(len(grid))
        l = np.zeros(len(grid))
        for n in range(len(grid)):
            if (n == 0):
                delta[n] = grid[n + 1] - grid[n]
                l[n] = grid[n] - delta[n] / 2
                r[n] = np.mean([grid[n], grid[n + 1]])
            elif n == (len(grid) - 1):
                delta[n] = grid[n] - grid[n - 1]
                l[n] = np.mean([grid[n], grid[n - 1]])
                r[n] = grid[n] + delta[n] / 2
            else:
                l[n] = np.mean([grid[n], grid[n - 1]])
                r[n] = np.mean([grid[n], grid[n + 1]])
                delta[n] = np.abs(r[n] - l[n])
        return delta, l, r

    def setROI2D(self,xAxis, yAxis, Matrix, xMin, xMax, yMin, yMax):
        """ selects a rectangular region of intrest ROI from a 2D Matrix based on the
        passed boundaries xMin,xMax, yMin and yMax. x stands for the columns of the
        Matrix, y for the rows

        Parameters
        ----------
            xAxis, yAxis : 1D numpy arrays
                numpy arrays containing the x and y grid respectively
            Matrix : 2D numpy array
                2D array with the shape (len(yAxis),len(xAxis))
            xMin,xMax,yMin,yMax : inclusive Boundaries for the ROI

        Returns
        ------
        in that order
            xROI : 1D numpy array slice of xAxis between xMin and xMax

            yROI : 1D numpy array slice of yAxis between yMin and yMax

            ROI : 2D numpy array of same length as vector

            xIntegral : 1D numpy arrays with the same length as xROI
                array containing the sum of ROI over the y direction

            yIntegral : 1D numpy arrays with the same length as yROI
                array containing the sum of ROI over the x direction

        Example
        -------
            >>> qzCut,qxCut,ROI,xIntegral,yIntegral = setROIMatrix(qzGrid,qxGrid,RSMQ,2.1,2.2,-0.5,0.5)"""

        selectX = np.logical_and(xAxis >= xMin, xAxis <= xMax)
        selectY = np.logical_and(yAxis >= yMin, yAxis <= yMax)

        xROI = xAxis[selectX]
        yROI = yAxis[selectY]

        ROI = Matrix[selectY, :]
        ROI = ROI[:, selectX]
        xIntegral = np.sum(ROI, 0)
        yIntegral = np.sum(ROI, 1)
        return xROI, yROI, ROI, xIntegral, yIntegral

    def finderA(self,array, key):
        """This little function returns the index of the array where the array value is closest to the key

        Parameters
        ----------
        array : 1D numpy array
                numpy array with values
        key :   float value
                The value that one looks for in the array

        Returns
        ------
        index :     integer
                    index of the array value closest to the key

        Example
        -------
        >>> index = finderA(np.array([1,2,3]),3)
        will return 2"""
        index = (np.abs(array - key)).argmin()
        return index

    def calcMoments(self,xAxis, yValues):
        """ calculates the Center of Mass, standard Deviation and Integral of a given Distribution

        Parameters
        ----------
            xAxis : 1D numpy array
                numpy array containing the x Axis
            yValues : 1D numpy array
                numpy array containing the according y Values

        Returns
        ------
        in that order
            COM : float
                xValue of the Center of Mass of the data

            STD : float
                xValue for the standard deviation of the data around the center of mass

            integral :
                integral of the data

        Example
        -------
            >>> COM,std,integral = calcMoments([1,2,3],[1,1,1])
                sould give a COM of 2, a std of 1 and an integral of 3 """

        COM = np.average(xAxis, axis=0, weights=yValues)
        STD = np.sqrt(np.average((xAxis - COM) ** 2, weights=yValues))
        delta = self.calcGridBoxes(xAxis)[0]
        integral = np.sum(yValues * delta)
        return COM, STD, integral

    def totalPlot(self):
        X, Y = np.meshgrid(self.distX, self.distY)

        plt.figure(1, figsize=(8, 8 * np.size(self.pixelY) / np.size(self.pixelX)), linewidth=2)

        pl = plt.pcolormesh(X, Y, self.data, cmap=self.fireice(), vmin=1, vmax=255, norm=matplotlib.colors.LogNorm())
        plt.axis([0, X.max(), 0, Y.max()])
        plt.xlabel(r'x ($\mathrm{\mu{}}$m)', fontsize=self.fontSize)
        pl1 = plt.ylabel(r'y ($\mathrm{\mu{}}$m)', fontsize=self.fontSize)

        plt.colorbar()
        plt.savefig(self.path + "\\" + self.name + "_Image.png", dpi=300)

    def fitGauss(self):
        xROI, yROI, dataROI, xIntegral, yIntegral = self.setROI2D(self.distX[1:], self.distY[1:], self.data, self.xMin,
                                                             self.xMax, self.yMin, self.yMax)

        X, Y = np.meshgrid(xROI - self.xMin, yROI - self.yMin)

        dX = self.xMax - self.xMin
        dY = self.yMax - self.yMin

        imax = self.finderA(yIntegral, np.max(yIntegral))
        self.sliceY = dataROI[imax, :]

        imax = self.finderA(xIntegral, np.max(xIntegral))
        self.sliceX = dataROI[:, imax]

        # %% Fitting the resulting Parameters
        model = lm.models.GaussianModel() + lm.models.LinearModel()
        parsX = lm.Parameters()
        parsY = lm.Parameters()

        COMx, STDx, Ix = self.calcMoments(xROI - self.xMin, xIntegral)
        COMy, STDy, Iy = self.calcMoments(yROI - self.yMin, yIntegral)
        # Here you can set the initial values and possible boundaries on the fitting parameters
        # Name, Value, Vary, Min, Max

        parsX.add_many(('center', COMx, True),
                       ('sigma', STDx, True),
                       ('amplitude', 500, True),
                       ('slope', 0, False),
                       ('intercept', 0, True))

        parsY.add_many(('center', COMy, True),
                       ('sigma', STDy, True),
                       ('amplitude', 800, True),
                       ('slope', 0, False),
                       ('intercept', 0, True))
        ## Fitting takes place here
        resultX = model.fit(xIntegral / np.max(xIntegral), parsX, x=xROI - self.xMin)
        resultY = model.fit(yIntegral / np.max(yIntegral), parsY, x=yROI - self.yMin)

        self.resultXslice = model.fit(self.sliceX / np.max(self.sliceX), parsX, x=yROI - self.yMin)
        self.resultYslice = model.fit(self.sliceY / np.max(self.sliceY), parsY, x=xROI - self.xMin)

        ## Writing the results into the peaks dictionary takes place here
        self.FWHMx = 2.35482 * resultX.values["sigma"]  # in micron
        self.FWHMy = 2.35482 * resultY.values["sigma"]  # in micron

        self.FWHMxslice = 2.35482 * self.resultXslice.values["sigma"]  # in micron
        self.FWHMyslice = 2.35482 * self.resultYslice.values["sigma"]  # in micron

        self.plotROI(xROI, yROI, xIntegral, yIntegral, resultX, resultY, X, Y, dataROI)

    def plotROI(self, xROI, yROI, xIntegral, yIntegral, resultX, resultY, X, Y, dataROI):
        plt.figure(2, figsize=(6, 6), linewidth=2)
        gs = gridspec.GridSpec(2, 2,
                               width_ratios=[3, 1],
                               height_ratios=[1, 3],
                               wspace=0.0,
                               hspace=0.0)

        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])
        ax3 = plt.subplot(gs[2])
        ax4 = plt.subplot(gs[3])

        ## Plot 1) Top Left: Horizontal Profile ##
        ax1.xaxis.tick_top()
        ax1.xaxis.set_label_position("top")

        ax1.text(0, 0.8, 'FWHMx =\n' + str(int(round(self.FWHMx))) + r' $\mathrm{\mu{}m}$', fontsize=6)
        ax1.text(0, 0.3, 'FWHMxslice =\n' + str(int(round(self.FWHMxslice))) + r' $\mathrm{\mu{}m}$', fontsize=6)

        ax1.plot(xROI - self.xMin, xIntegral / np.max(xIntegral), '-k', lw=2, label='Integral')
        ax1.plot(xROI - self.xMin, resultX.best_fit, '-', color='orange', lw=1, label="Fit Integral")
        ax1.plot(xROI - self.xMin, self.sliceY / np.max(self.sliceY), '-r', lw=1, label='slice')
        ax1.plot(xROI - self.xMin, self.resultYslice.best_fit, '--', color='green', lw=1, label="Fit Slice")

        ax1.set_ylabel('I (a.u.)', fontsize=self.fontSize)
        ax1.set_xlabel(r'x ($\mathrm{\mu{}}$m)', fontsize=self.fontSize)
        ax1.set_ylim([0, 1.1])
        ax1.set_yticks(np.arange(0.25, 1.25, .25))

        ## Plot 3) Bottom Left: Colormap of the Profile#############
        pl = ax3.pcolormesh(X, Y, dataROI, cmap=self.fireice(), vmin=1, vmax=150, norm=matplotlib.colors.LogNorm())
        ax3.axis([0, self.xMax - self.xMin, 0, self.yMax - self.yMin])
        ax3.set_xlabel(r'x ($\mathrm{\mu{}}$m)', fontsize=self.fontSize)
        ax3.set_ylabel(r'y ($\mathrm{\mu{}}$m)', fontsize=self.fontSize)
        # ax3.text(0,0,"Pin  =" + str(np.round(Power,4)) + "mW \n-> F at 90° = " + str(F90/10) + "mJ/cm^2 \n-> F at "+str(int(angle))+"° = " + str(np.round(Fangle/10,2)) + "mJ/cm^2",fontsize = 6)
        ### Colorbar placement #############
        axins3 = insert_axes(ax3,
                            width="60%",  # width = 10% of parent_bbox width
                            height="5%",  # height : 50%                   )
                            loc=4)
        ax3.add_patch(Rectangle((0.35, 0.018), 0.65, 0.14, edgecolor="none", facecolor="white", alpha=0.75,
                                transform=ax3.transAxes))
        cbar = plt.colorbar(pl, cax=axins3, orientation="horizontal")
        cbar.ax.tick_params(labelsize=8)
        # cbar.set_label('norm. Reflectivity ')
        axins3.xaxis.set_ticks_position("top")
        axins3.xaxis.set_label_position("top")
        cl = plt.getp(cbar.ax, 'xmajorticklabels')
        plt.setp(cl, color="black")
        # ax1.legend(loc = (1.05,0.0))
        ax2.axis('off')

        ##Plot 4) Bottom Right Vertical Profile
        ax4.plot(yIntegral / np.max(yIntegral), yROI - self.yMin, '-k', lw=2, label='Integral')
        ax4.plot(resultY.best_fit, yROI - self.yMin, '-', color='orange', lw=1, label="Fit Integral")
        ax4.plot(self.sliceX / np.max(self.sliceX), yROI - self.yMin, '-r', lw=1, label='slice')
        ax4.plot(self.resultXslice.best_fit, yROI - self.yMin, '--', color='green', lw=1, label="Fit Slice")

        ax4.set_ylabel(r'y ($\mathrm{\mu{}}$m)', fontsize=self.fontSize)
        ax4.set_xlabel('I (a.u.)', fontsize=self.fontSize)
        ax4.yaxis.tick_right()
        ax4.yaxis.set_label_position("right")
        ax4.legend(loc=(0.05, 1.05), frameon=False)
        ax4.set_ylim(0, self.yMax - self.yMin)

        ax4.text(0.25, 100, 'FWHMy =\n' + str(int(round(self.FWHMy))) + r' $\mathrm{\mu{}m}$', fontsize=8)
        ax4.text(0.25, 350, 'FWHMyslice =\n' + str(int(round(self.FWHMyslice))) + r' $\mathrm{\mu{}m}$', fontsize=8)
        # ax4.set_xlabel("Normalized X-Ray Reflectivity")
        ax4.set_xticks([0.5, 1.1])
        ax4.grid('off')
        plt.savefig(self.path + "\\" + self.name + "_FitResult.png", bbox_inches='tight', dpi=300)
        plt.show()

def calculateFluenceFromPower(power, anglePump, repitionRate, FWHMx, FWHMy, chopper):
    """
    Calculate the Fluence from a given Power value (behind the chopper, in
    mW). Calculates x0 and y0 as the value of 1/e instead of 1/2 to not
    overestimate the fluence.
    Pulse Energy [Ws] = Power [W] / (Repitition Rate [Hz])
    Area of Pump [µm^2] = pi * y0 [µm] * x0 [µm]
    F [J/m^2] = Pulse Energy [Ws] /
    ([Area [µm^2] * 10^-12] [m^2] * sin(angle))

    FmJ [mJ/cm^2] = F [J/m^2] * 10^-3 [mJ] * 10^4 [cm^2]
    = F [J/m^2] + 10
    :param power: in mW after chopper
    :return: FmJ: Fluence in mJ/cm^2
    """
    if chopper:
        power = power * 2
    x0, y0 = calculateX0(FWHMx, FWHMy)
    Ep = (power) / (repitionRate * 1000);
    Area = np.pi * x0 * y0
    if Area != 0:
        F = Ep / (Area * 1e-12) * np.sin(np.deg2rad(anglePump))
        FmJ = (F) / 10
        return FmJ
    else:
        #print('Cant calculate Fluence, Area of Spot is 0')
        return 0


def calculateX0(FWHMx, FWHMy):
    """
    Calculate the 1/e values from the measured Full Width Half Maximum
    values
    :param FWHMx: µm value fitted from Beamprofile
    :param FWHMy: µm value fitted from Beamprofile
    :return: 1/e x, 1/e y
    """

    Variante2Factor = 2 * np.sqrt(np.log(2))
    x0 = float(FWHMx) / Variante2Factor
    y0 = float(FWHMy) / Variante2Factor
    return x0, y0