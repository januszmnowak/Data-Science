import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 100)

#style setpup
plt.style.use('seaborn-pastel') #fivethirtyeight', 'seaborn-pastel', 'seaborn-whitegrid', 'ggplot', 'grayscale', 'dark_background'

#Simple chart - object oriented interface
fig = plt.figure() #figure to zbior wykresow
ax= plt.axes() #ax to pojedynczy wykres
ax.plot(x, np.sin(x))
ax.plot(x, np.cos(x))
plt.show()

#Simple chart - Matlab interface
plt.plot(x, np.sin(x))
plt.plot(x, np.cos(x))
plt.show()

#Subplots
fig=plt.figure() # tworze jeden zbior wykresow (fig)
ax1 = fig.add_subplot(2, 2, 1) #dodaje wykresy (ax1, ...), w subplots definiuje rozmieszczenie wykresow (rows, columns, aktywny subplot)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4)
ax1.plot(x, np.sin(x))
ax4.plot(x, np.cos(x))
plt.title("Subplots")
plt.show()

#Saving charts to file
fig.savefig('my_figure.jpg')

#Line Colors and Styles
plt.plot(x, np.sin(x - 0), color='blue') # specify color by name
plt.plot(x, np.sin(x - 1), color='g') # short color code (rgbcmyk)
plt.plot(x, np.sin(x - 2), color='0.75') # Grayscale between 0 and 1
plt.plot(x, np.sin(x - 3), color='#FFDD44') # Hex code (RRGGBB from 00 to FF)
plt.plot(x, np.sin(x - 4), color=(1.0,0.2,0.3)) # RGB tuple, values 0 and 1
plt.plot(x, np.sin(x - 5), color='chartreuse'); # all HTML color names supported
plt.plot(x, x + 4, linestyle='-') # solid
plt.plot(x, x + 5, linestyle='--') # dashed
plt.plot(x, x + 6, linestyle='-.') # dashdot
plt.plot(x, x + 7, linestyle=':'); # dotted
plt.title("Colors and styles")
plt.show()

#axes limits, title, labels, legends, text annotations and arrows
plt.plot(x, np.sin(x),'-g',label='sin(x)')
plt.plot(x, np.cos(x),':b',label='cos(x)')
plt.xlim(-1, 11)
plt.ylim(-1.5, 1.5)
plt.title("Labels, legends, limits")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(loc='upper right', frameon=False, ncol=2)
plt.text(5,-1.2,"Minimum",ha='center',size=10, color='gray') #text label - takes an x position, a y position, a string, and then optional keywords specifying the color, size, style, alignment, etc.
plt.annotate('Maximum', xy=(8, 1), xytext=(10, 1), arrowprops=dict(facecolor='black', shrink=0.05))
plt.show()

#log scales
fig = plt.figure()
ax = plt.axes(xscale='log', yscale='log')
x=[10,100,1000,10000]
y=[20,200,2000,20000]
plt.plot(x,y)
plt.title('Log scales')
plt.show()


#scatter plots
rng = np.random.RandomState(0)
x = rng.randn(100)
y = rng.randn(100)
colors = rng.rand(100)
sizes = 1000 * rng.rand(100)
plt.scatter(x, y, c=colors, s=sizes, alpha=0.5, cmap='viridis')
plt.colorbar()
plt.title("Scatter plot")
for area in [50, 100, 150]: #legend bubble size
    plt.scatter([],[],c='k',alpha=0.3,s=area, label=str(area))
plt.legend(scatterpoints=1, frameon=False, labelspacing=1, title='Legend bubble size')
plt.show()

#error bars
x = np.linspace(0, 10, 50)
dy = 0.8
y = np.sin(x) + dy * np.random.randn(50)
plt.errorbar(x, y, yerr=dy, fmt='.k',ecolor='lightgray')
plt.title("Error bars")
plt.show()

#contour charts
x = np.linspace(0, 5, 50)
y = np.linspace(0, 5, 40)
def f(x, y):
    return np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)
X, Y = np.meshgrid(x, y) #zapewnia ciaglosc danych
Z = f(X, Y)
plt.contourf(X, Y, Z, 20, cmap='RdGy')
plt.colorbar()
plt.title("Contour")
plt.show()

#histograms
x1 = np.random.normal(0, 0.8, 1000)
x2 = np.random.normal(-2, 1, 1000)
plt.hist(x1, histtype='stepfilled', alpha=0.3, bins=40)
plt.hist(x2, histtype='stepfilled', alpha=0.3, bins=40)
plt.title("Histograms")
plt.show()

#3D charts - simple
fig = plt.figure()
ax = plt.axes(projection='3d')
# Data for a three-dimensional line
zline = np.linspace(0, 15, 1000)
xline = np.sin(zline)
yline = np.cos(zline)
ax.plot3D(xline, yline, zline, 'gray')
# Data for three-dimensional scattered points
zdata = 15 * np.random.random(100)
xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')
plt.title('3D simple')
plt.show()

#3D charts - contour
def f(x, y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))
x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 50, cmap='rainbow')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.title('3D contour')
plt.show()

#Maps
# 0. Prepare data
import pandas as pd
cities=pd.read_csv('california_cities.csv')
lat = cities['latd'].values
lon = cities['longd'].values
population = cities['population_total'].values
area = cities['area_total_km2'].values

# 1. Draw the map background
from mpl_toolkits.basemap import Basemap
fig = plt.figure(figsize=(8, 8))
m = Basemap(projection='lcc', resolution='i', lat_0=37.5, lon_0=-119, width=1E6, height=1.2E6)
m.shadedrelief()
m.drawcoastlines(color='gray')
m.drawcountries(color='gray')
m.drawstates(color='gray')

# 2. scatter city data, with color reflecting population and size reflecting area
m.scatter(lon, lat, latlon=True, c=np.log10(population), s=area, cmap='Reds', alpha=0.5)

# 3. create colorbar and legend
plt.colorbar(label=r'$\log_{10}({\rm population})$')
plt.clim(3, 7)

# make legend with dummy points
for a in [100, 300, 500]:
    plt.scatter([], [], c='k', alpha=0.5, s=a, label=str(a) + ' km$^2$')
plt.legend(scatterpoints=1, frameon=False, labelspacing=1, loc='lower left')
