import numpy as np
import matplotlib.pyplot as plt
import configparser

config = configparser.ConfigParser()
config.read('SimulationParams.ini')
simulation_filename = config['Technical data']['heat_filename']

model_area_x = int(config['Model Geometry']['model area x [mm]'])
model_area_z = int(config['Model Geometry']['model area z [mm]'])
data_x_size = int(config['Model Geometry']['data x size'])
data_z_size = int(config['Model Geometry']['data z size'])
#body_temp = float(config['Tissue params for heat']['initial body temperature [k]'])
duration = float(config['Heat modelling params']['modelling duration [s]'])

class IndexTracker:
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('use scroll wheel to set time')
        ax.set_xlabel('mm')
        ax.set_xticks([i*(data_x_size-1)/5 for i in range(6)])
        ax.set_xticklabels([i*model_area_x/5 for i in range(6)])
        ax.set_ylabel('mm')
        ax.set_yticks([i*(data_z_size-1)/5 for i in range(6)])
        ax.set_yticklabels([i*model_area_z/5 for i in range(6)])
        self.X = X
        self.min_val = np.amin(X)
        self.max_val = np.amax(X)
        self.slices = X.shape[1]
        self.ind = self.slices//2

        self.im = ax.imshow(self.X[:, self.ind, :],vmin=self.min_val, vmax=self.max_val)
        cbar = plt.colorbar(self.im)
        self.update()

    def on_scroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = int(self.ind + self.slices/50+1)
            if self.ind >= self.slices:
               self.ind = self.slices-1
        else:
            self.ind = int(self.ind - self.slices/50)
            if self.ind < 0:
                self.ind = 0
        self.update()

    def update(self):
        self.im.set_data(self.X[:, self.ind, :])
        self.ax.set_title('elapsed time %f s' % (duration/self.slices*self.ind))
        self.im.axes.figure.canvas.draw()

if __name__ == "__main__":
    fig, ax = plt.subplots(1, 1)

    X = np.load(simulation_filename)
    print(X.shape)
    tracker = IndexTracker(ax, X)


    fig.canvas.mpl_connect('scroll_event', tracker.on_scroll)
    plt.show()