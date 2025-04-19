import numpy as np
from pprint import pprint
import math
import plotly
import plotly.graph_objs as go
import plotly.express as px
import matplotlib.pyplot as plt
import pickle as pck
from sklearn.linear_model import LinearRegression



class CyclicRegression():

    def __init__(self, boundary=96):
        self.boundary =  boundary

    def cyclic_distance(self, x1, x2):
        poss_values = self.boundary
        diff = abs(x1-x2)
        distance = min(poss_values - diff, diff)
        return distance


    def rotateData(self, xdata, rotation=10):
        rotatedXdata = []
        for x in xdata:
            if (x+rotation) >= self.boundary:
                rotatedX = (x+rotation)-self.boundary
                rotatedXdata.append(rotatedX)
            else:
                rotatedXdata.append(x+rotation)
        return rotatedXdata

    def checkBoundary(self, xdata):
        for x in xdata:
            diff = abs(xdata[0] - x)
            if (self.boundary-diff) < diff:
                return True
        return False

    def findClusterEdges(self, xdata):
        diffs = list(map(self.cyclic_distance, [xdata[0]]*len(xdata), xdata))
        firstEdge = xdata[diffs.index(max(diffs))]

        diffs = list(map(self.cyclic_distance, [firstEdge]*len(xdata), xdata))
        secondEdge = xdata[diffs.index(max(diffs))]
        return min(firstEdge, secondEdge), max(firstEdge, secondEdge)

    def shiftCluster(self, xdata, firstEdge, secondEdge):
        shiftedX = []

        for x in xdata:
            if x>=secondEdge:
                shiftedX.append(x-secondEdge)
            else:
                shiftedX.append(x+(self.boundary-secondEdge))
        return shiftedX

    def LinReg(self, featureData, ydata):
        # Linear Regression using sklearn
        transformed_data = np.array(featureData[0]).reshape(-1,1)

#        xtrain, xtest, ytrain, ytest = transformed_data[:int(0.8*len(featureData[0]))], transformed_data[int(0.2*len(featureData[0])):], ydata[:int(0.8*len(featureData[0]))], ydata[int(0.2*len(featureData[0])):]


        LR = LinearRegression()
#    sintransform = np.array(sin_time).reshape(-1,1)
        LR.fit(transformed_data, ydata)
        preds = LR.predict(transformed_data)
        m = LR.coef_
        c = LR.intercept_

        return preds, m, c

    def cyclicRegression(self, xdata, ydata):
        if self.checkBoundary(xdata):
            firstEdge, secondEdge = self.findClusterEdges(xdata)
            shiftedX = self.shiftCluster(xdata, firstEdge, secondEdge)
            preds, m, c = self.LinReg([shiftedX], ydata)
#            ys = [(m*x+c) for x in shiftedX]
        else:
            preds, m, c = self.LinReg([xdata], ydata)
#            ys = [(m*x+c) for x in xdata]
#        fig, axes =  plt.subplots(1,1, figsize = (10,10))
#        axesList = fig.get_axes()
#        axes.scatter(xdata, ydata, c='blue', s=2)
#        axes.scatter(xdata, ys, color='red', linewidth=5)
#        axes.set_xlim(0,96)
#        fig.savefig('rotatedSamples.pdf')
        return m, c

    def rotateCyclicData(self, xdata, ydata):
        for i in range(len(features)):
            featureData = [p[i] for p in xdata]
            if self.checkBoundary(featureData):
                firstEdge, secondEdge = self.findClusterEdges(featureData)
                shiftedX = self.shiftCluster(featureData, firstEdge, secondEdge)

                for j, x in enumerate(xdata):
                    x[i] = featureData[j]
        return xdata

    def plotCircularData(self, xdata, ydata, ys, fig, color):
        if not fig:
            layout = go.Layout(
                    margin={'l': 0, 'r': 0, 'b': 0, 't': 0}
                    )
            fig = go.Figure(data=[], layout=layout)

        sin_time = np.sin(np.array(xdata)*2*np.pi/96)
        cos_time = np.cos(np.array(xdata)*2*np.pi/96)

        fig.add_trace(go.Scatter3d(
            x=sin_time,
            y=cos_time,
            z=ydata,
            mode='markers',
            marker={
                'size': 5,
                'opacity': 0.8,
                'color': color}
        ))
        lindata = np.array(ys).flatten()
        fig.add_trace(go.Scatter3d(
            x=sin_time,
            y=cos_time,
            z=lindata,
            mode='markers',
            marker={'size': 12, 'color': color}
        ))

        return fig
