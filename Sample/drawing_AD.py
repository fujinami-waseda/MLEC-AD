import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""
define class
"""

class DrawingAD():
    def __init__(self, inppath, savepath):
        self.mol_radius = {"H":31, "He":28, "Li":128, "Be":96, "B":84, "C":76, "N":71, "O":66, "F":57, "Ne":58, "Na":166, "Mg":141, "Al":121, "Si":111, "P":107, "S":105, "Cl":102, "Ar":106, "K":203, "Ca":176} # Covalent radius (2008 values) unit: pm
#        self.mol_radius_angs = {"H":0.31, "He":0.28, "Li":1.28, "Be":0.96, "B":0.84, "C":0.76, "N":0.71, "O":0.66, "F":0.57, "Ne":0.58, "Na":1.66, "Mg":1.41, "Al":1.21, "Si":1.11, "P":1.07, "S":1.05, "Cl":1.02, "Ar":1.06, "K":2.03, "Ca":1.76}
        self.mol_radius_angs = {"H":0.28, "He":0.28, "Li":0.58, "Be":0.58, "B":0.58, "C":0.58, "N":0.58, "O":0.58, "F":0.58, "Ne":0.58, "Na":1.06, "Mg":1.06, "Al":1.06, "Si":1.06, "P":1.06, "S":1.06, "Cl":1.06, "Ar":1.06, "K":2.03, "Ca":1.76}
        # source: Beatriz Cordero, Veronica Gomez, Ana E. Platero-Prats, Marc Reves, Jorge Echeverria, Eduard Cremades, Flavia Barragan and Santiago Alvarez, in "Covalent radii revisited", Dalton Trans., 2008, [DOI: 10.1039/b801115j].
        self.colordict = {"H":"white", "C":"black", "N":"blue", "O":"red", "F":"greenyellow", "S":"yellow", "Cl":"green", "P":"orange", "Si":"cadetblue", "Li":"purple", "B":"pink", "Al":"plum"} # color of atom
        self.inppath = inppath
        self.savepath = savepath

    def loading_molecule(self, mol):
        molecule = pd.read_csv(self.inppath+"/"+mol+".xyz", header=None, sep="\s+", names=["atom", "x", "y", "z"], skiprows=2)
        return molecule

    def prepare_sphere(self, i):
        atom, x0, y0, z0 = self.moldata.iloc[i, 0], self.moldata.iloc[i, 1], self.moldata.iloc[i, 2], self.moldata.iloc[i, 3]
        r = self.mol_radius_angs[atom]
        theta_1_0 = np.linspace(0, np.pi, 1000)
        theta_2_0 = np.linspace(0, 2*np.pi, 1000)
        theta_1, theta_2 = np.meshgrid(theta_1_0, theta_2_0)
        x = np.cos(theta_2) * np.sin(theta_1) * r
        y = np.sin(theta_2) * np.sin(theta_1) * r
        z = np.cos(theta_1) * r
        x += x0
        y += y0
        z += z0

        return atom, x, y, z

    def draw(self, mol, stdpath, coordpath):
        self.mol = mol
        self.moldata = self.loading_molecule(self.mol)
        moldata = self.moldata
        fig = plt.figure(figsize=(12,12), dpi=600)
        #ax = Axes3D(fig)
        ax = fig.add_subplot(111, projection="3d")
        ax.xaxis.set_pane_color((0,0,1,0.3))
        ax.yaxis.set_pane_color((0,0,1,0.3))
        ax.zaxis.set_pane_color((0,0,1,0.3))

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        x_max, x_min = moldata["x"].max(), moldata["x"].min()
        y_max, y_min = moldata["y"].max(), moldata["y"].min()
        z_max, z_min = moldata["z"].max(), moldata["z"].min()

        num, stdlist = self.write_plot_ens2(stdpath, self.mol, coordpath)

        x_max_data, y_max_data, z_max_data, x_min_data, y_min_data, z_min_data = find_max_min(stdlist)

        xmax, ymax, zmax = max([x_max, x_max_data]), max([y_max, y_max_data]), max([z_max, z_max_data])
        xmin, ymin, zmin = min([x_min, x_min_data]), min([y_min, y_min_data]), min([z_min, z_min_data])

        maxval = max([x_max, y_max, z_max, x_max_data, y_max_data, z_max_data])
        minval = min([x_min, y_min, z_min, x_min_data, y_min_data, z_min_data])

        xave, yave, zave = (xmax+xmin)/2, (ymax+ymin)/2, (zmax+zmin)/2

        ax.set_xlim(minval-0.5, maxval+0.5)
        ax.set_ylim(minval-0.5, maxval+0.5)
        ax.set_zlim(minval-0.5, maxval+0.5)
        ax.view_init(elev=10, azim=35)

        for i in range(len(moldata)):
            atom, x, y, z = self.prepare_sphere(i)
            ax.plot_surface(x-xave, y-yave, z-zave, alpha=0.45, color=self.colordict[atom])

        for i in range(len(num)):
            tmp = stdlist[i]
            ax.scatter(tmp[:, 0]-xave, tmp[:, 1]-yave, tmp[:, 2]-zave, marker=".", color="tomato", alpha=0.5, label=num)
            plt.subplots_adjust(wspace=0.4)
            plt.subplots_adjust(left=0, right=0.95, bottom=0.1, top=0.95)
            plt.savefig(self.savepath+"/"+mol+"_"+str(i)+".png",bbox_inches='tight', pad_inches=0)
        plt.close()

    def write_plot_ens2(self, stdpath, mol, coordpath):
        coord = np.loadtxt(coordpath+"/"+mol+".csv", delimiter=",", skiprows=1)
        coord = coord[:, [0, 1, 2]]
        coord *= 0.529177 #bohr -> angs
        std = np.loadtxt(stdpath+"/"+mol+"_std.csv")
        std = std.reshape(-1,1)
        coord_std = np.hstack([coord, std])
        num = [10**(-1.75)] #thresh

        stdlist = []
        print(num)
        print("Data max: {0}, Data min: {1}".format(coord_std[:, 3].max(), coord_std[:, 3].min()))
        for i in range(len(num)):
            if i == 0:
                print("{0} <= coord_std".format(num[i]))
                tmp = coord_std[np.where((num[i] <= coord_std[:, 3]))]
            else:
                print("{0} <= coord_std < {1}".format(num[i], num[i-1]))

                tmp = coord_std[np.where((num[i] <= coord_std[:, 3]) & (coord_std[:, 3] < num[i-1]))]
            try:
                print("Max: {0}, Min: {1}".format(tmp[:, 3].max(), tmp[:, 3].min()))
                print("The size is {0}".format(len(tmp)))
            except ValueError:
                print("zero-size array has been got")
                print("The size is 0")
            stdlist.append(tmp)
        return num, stdlist

def find_max_min(inplist):
    x_max, y_max, z_max = [], [], []
    x_min, y_min, z_min = [], [], []
    for i in range(len(inplist)):
        print(i)
        tmp = inplist[i]
        try:
          tmpx_max, tmpx_min = tmp[:, 0].max(), tmp[:, 0].min()
        except ValueError:
          continue
        tmpy_max, tmpy_min = tmp[:, 1].max(), tmp[:, 1].min()
        tmpz_max, tmpz_min = tmp[:, 2].max(), tmp[:, 2].min()
        x_max.append(tmpx_max)
        y_max.append(tmpy_max)
        z_max.append(tmpz_max)
        x_min.append(tmpx_min)
        y_min.append(tmpy_min)
        z_min.append(tmpz_min)
    return max(x_max), max(y_max), max(z_max), \
           min(x_min), min(y_min), min(z_min)

def main():
    inppath = "./plot_AD"
    coordpath = "./data"
    savepath = "./plot_AD"
    stdpath = "./ens"
    mol = "1"

    draw = DrawingAD(inppath, savepath)
    draw.draw(mol, stdpath, coordpath)

if __name__ == "__main__":
    main()
