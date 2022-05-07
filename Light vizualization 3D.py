import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import configparser

config = configparser.ConfigParser()
config.read('SimulationParams.ini')

model_area_x = int(config['Model Geometry']['model area x [mm]'])
model_area_y = int(config['Model Geometry']['model area y [mm]'])
model_area_z = int(config['Model Geometry']['model area z [mm]'])
data_x_size = int(config['Model Geometry']['data x size'])
data_y_size = int(config['Model Geometry']['data y size'])
data_z_size = int(config['Model Geometry']['data z size'])

filename = config['Technical data']['absorbed light filename']

def blockwise_average_3D(A,scaler):
    S=(scaler,scaler,scaler)
    m,n,r = np.array(A.shape)//S
    return A[:m*S[0],:n*S[1],:r*S[2]].reshape(m,S[0],n,S[1],r,S[2]).sum((1,3,5))

class Visualisation3D:
    def __init__(self, ax, data) -> None:
        self.ax = ax
        self.init_data = data
        self.current_data = blockwise_average_3D(data, 10)
        self.reducer = 1
        self.threshold = 2

        z, y, x = self.current_data.nonzero()
        self.scat = self.ax.scatter(x, y, z, s=self.current_data[z,y,x]/50, c=self.current_data[z,y,x], alpha=0.3)
        self.cbar = plt.colorbar(self.scat, ax=self.ax)

    def update_reduce(self, val):
        self.reducer = val
        self.ax.clear()
        self.current_data=blockwise_average_3D(self.init_data, self.reducer)
        print('New data shape ',self.current_data.shape)
        display_data=np.copy(self.current_data)
        self.update(display_data)
        

    def update_thresh(self, val):
        self.threshold = val
        self.ax.clear()
        display_data=np.copy(self.current_data)
        self.update(display_data)
        
    
    def update(self,display_data):
        with np.nditer(display_data, op_flags=['readwrite']) as it:
            for value in it:
                if value < self.threshold:
                    value[...] = 0

        z, y, x = display_data.nonzero()
        max_val = display_data.max()/50
        self.scat = self.ax.scatter(x, y, z, s=display_data[z,y,x]/max_val, c=display_data[z,y,x], alpha=0.3)
        self.cbar.update_normal(self.scat)
        
        self.ax.set_xlim(0, display_data.shape[2])
        self.ax.set_ylim(0, display_data.shape[1])
        self.ax.set_zlim(display_data.shape[0], 0)
        ax.set_xlabel('X [mm]')
        ax.set_xticks([i*display_data.shape[0]/5 for i in range(6)])
        ax.set_xticklabels([i*model_area_x/5 for i in range(6)])
        ax.set_ylabel('Y [mm]')
        ax.set_yticks([i*display_data.shape[1]/5 for i in range(6)])
        ax.set_yticklabels([i*model_area_y/5 for i in range(6)])
        ax.set_zlabel('Z [mm]')
        ax.set_zticks([i*display_data.shape[0]/5 for i in range(6)])
        ax.set_zticklabels([i*model_area_z/5 for i in range(6)])
        fig.canvas.draw_idle()
    
if __name__ == "__main__":
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # adjust the main plot to make room for the sliders
    plt.subplots_adjust(left=0.0, bottom=0.25)

    # Make a horizontal slider to control the frequency.
    axthresh = plt.axes([0.1, 0.1, 0.8, 0.03])
    axreduce = plt.axes([0.1, 0.05, 0.8, 0.03])

    threshold_slider = Slider(
        ax=axthresh,
        label='Display threshold',
        valmin=1,
        valmax=100,
        valinit=2,
        valstep=1
    )

    reduce_slider = Slider(
        ax=axreduce,
        label='Reduce data size',
        valmin=1,
        valmax=10,
        valinit=5,
        valstep=1
    )

    data = np.load(filename)
    print('Shape of loaded data ', data.shape)
    vizualizator = Visualisation3D(ax,data)
    vizualizator.update_reduce(5)

    threshold_slider.on_changed(vizualizator.update_thresh)
    reduce_slider.on_changed(vizualizator.update_reduce)

    plt.show()