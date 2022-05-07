#Модуль расчёта фотонагрева ткани.
#Разработана при финансовой поддержке Российского Научного Фонда (грант № 20-72-00081).
#Антон Попов 2022: a.popov.fizteh@gmail.com

import numpy as np
import numexpr as ne
import configparser
import matplotlib.pyplot as plt
import tqdm
from datetime import datetime
import scipy.interpolate as interp
import os.path

params = dict()

#Siluation requires SimulationParams.ini file for operation

def load_config():
    """Load simulation parameters"""

    print('')
    print('Start loading simulation params from SimulationParams.ini...')
    print('')
    config = configparser.ConfigParser()
    config.read('SimulationParams.ini')

    params['light distribution filename'] = config['Technical data']['absorbed light filename']

    spec_config = configparser.ConfigParser()
    spec_config.read(params['light distribution filename'][:-4] + '.ini')

    #load physical heat parameters
    k = float(config['Tissue params for heat']['thermal conductivity [w/(m*k)]'])
    params['k'] = k
    print('Parameter loaded: Tissue thermal conductivity =', k, 'W/(m*k)')
    ro = int(config['Tissue params for heat']['density [kg/m^3]'])
    params['ro'] = ro
    print('Parameter loaded: Tissue density =', ro, 'kg/m**3')
    C_tissue = int(config['Tissue params for heat']['specific heat [j/(kg*k)]'])
    params['C_tissue'] = C_tissue
    print('Parameter loaded: Tissue specific heat =', C_tissue, 'j/(kg*k)')
    body_temp = float(config['Tissue params for heat']['initial body temperature [k]'])
    print('Parameter loaded: Initial tissue temperature =', body_temp, 'k')
    params['h'] = float(config['Tissue params for heat']['convection heat loss [W/(m**2*k)]'])
    print('Parameter loaded: Convection surface heat loss =', params['h'], 'W/(m**2*k)')

    #load laser parameters
    Laser_power = float(config['Illumination params']['power [W]'])
    params['Laser power'] = Laser_power
    print('Parameter loaded: Laser_power =', Laser_power, 'W')
    Laser_duration = float(config['Heat modelling params']['heating duration [s]'])
    print('Parameter loaded: Heat diffusion modelling duration =', Laser_duration, 's')
    Total_duration = float(config['Heat modelling params']['modelling duration [s]'])
    if config['Illumination params']['center y'] == 'center':
        params['laser y pos'] = float(config['Model Geometry']['model area y [mm]'])/2
    else:
        params['laser y pos'] = float(config['Illumination params']['center y'])

     #load light distribution parameters
    photon_count = int(config['MonteCarlo Params']['photon count'])
    print('Parameter loaded: Amount of simulated photons =', photon_count)
    if config['Illumination params']['neglect side bottom escape'] == '0':
        transmitted_light = round(
            1 - float(config['Technical data']['reflected light'])
            - (
                float(spec_config['Technical data']['escaped top photons'])
                + float(spec_config['Technical data']['escaped side photons'])
                + float(spec_config['Technical data']['escaped bottom photons'])
              ) / float(config['MonteCarlo Params']['photon count']),2
            )
        print('Parameter loaded: Fraction of photons absorbed in tissue =', transmitted_light*100, '%')
    else:
        transmitted_light = round(
            1 - float(spec_config['Technical data']['reflected light'])
            - (
                float(spec_config['Technical data']['escaped top photons'])
              ) / float(config['MonteCarlo Params']['photon count']),2
            )
        print('Parameter loaded: Fraction of photons absorbed in tissue =', transmitted_light*100, '%')
    data = np.load(params['light distribution filename'], allow_pickle=True)
    print('Data loaded: absorbed light distribution array with shape', data.shape)
    print('Max photons at position (ZYX)', np.unravel_index(np.argmax(data), data.shape))

    #load model geometry
    data_x_size = int(float(config['Model Geometry']['data x size']))
    print('Parameter loaded: Amount of data points along x axis =', data_x_size)
    data_y_size = int(float(config['Model Geometry']['data y size']))
    print('Parameter loaded: Amount of data points along y axis =', data_y_size)
    data_z_size = int(float(config['Model Geometry']['data z size']))
    print('Parameter loaded: Amount of data points along z axis =', data_z_size)
    grid_step = float(config['Model Geometry']['grid step [um]']) / 1000000 # in m
    print('Parameter loaded: grid step =', grid_step, 'm')
    params['model area y'] = float(config['Model Geometry']['model area y [mm]'])

    #load heating curve positions and save params
    params['save full data'] = int(config['Heat save params']['save full data'])
    heating_curve_positions = [] #curve 1 position [save,x,y,z]
    for i in range(3):
        save_curve = 'save heating curve ' + str(i+1)
        heating_curve_positions.append(int(config['Heat save params'][save_curve]))
        if heating_curve_positions[4*i] == 1:
            curve_x = 'heating curve ' + str(i+1) + ' x position [mm]'
            curve_y = 'heating curve ' + str(i+1) + ' y position [mm]' 
            curve_z = 'heating curve ' + str(i+1) + ' z position [mm]' 
            if config['Heat save params'][curve_x] == 'center':
                heating_curve_positions.append(int(float(config['Model Geometry']['model area x [mm]'])/2/grid_step/1000))
            else:
                heating_curve_positions.append(int(data_x_size/2 - float(config['Heat save params'][curve_x])/grid_step/1000))
            if config['Heat save params'][curve_y] == 'center':
                heating_curve_positions.append(int(float(config['Model Geometry']['model area y [mm]'])/2/grid_step/1000))
            else:
                heating_curve_positions.append(int(data_y_size/2 - float(config['Heat save params'][curve_y])/grid_step/1000))
            heating_curve_positions.append(int(float(config['Heat save params'][curve_z])/grid_step/1000))
            print('Heating curve at x = ', round(heating_curve_positions[4*i+1]*grid_step*1000,1), 'y = ', round(heating_curve_positions[4*i+2]*grid_step*1000,1), 'z = ', round(heating_curve_positions[4*i+3]*grid_step*1000,1), 'will be saved')
        else:
            heating_curve_positions.append(0)
            heating_curve_positions.append(0)
            heating_curve_positions.append(0)

    params['heating curve y pos'] = float(config['Heat save params']['heating curve 1 y position [mm]'])

    #load optimization parameters
    params['heat iteration mode'] = config['Heat iteration']['mode']
    if params['heat iteration mode'] == 'iteration':
        params['iterate thermal conductivity'] = int(config['Heat iteration']['iterate thermal conductivity'])
        params['iterate density'] = int(float(config['Heat iteration']['iterate density']))
        params['iterate specific heat'] = int(config['Heat iteration']['iterate specific heat'])
        params['heat iterations'] = int(float(config['Heat iteration']['iterations per variable']))
        params['heat iteration range'] = float(config['Heat iteration']['iteration range [%]'])
        params['ref curve x pos'] = config['Heat iteration']['reference curve x position [mm]']
        params['ref curve y pos'] = config['Heat iteration']['reference curve y position [mm]']
        params['ref curve z pos'] = config['Heat iteration']['reference curve z position [mm]']
        if config['Heat iteration']['reference curve x position [mm]'] == 'center':
            heating_curve_positions[1] = int(float(config['Model Geometry']['model area x [mm]'])/2/grid_step/1000)
        else:
            heating_curve_positions[1] = int(data_x_size/2 - float(config['Heat iteration']['reference curve x position [mm]'])/grid_step/1000)
        heating_curve_positions[2] = int(data_y_size/2 - float(config['Heat iteration']['reference curve y position [mm]'])/grid_step/1000)
        heating_curve_positions[3] = int(float(config['Heat iteration']['reference curve y position [mm]'])/grid_step/1000)

    params['reference curve filename'] = config['Technical data']['reference curve filename']

    return(k,ro,C_tissue,body_temp,Laser_power,Laser_duration,Total_duration,data_x_size,data_y_size,data_z_size,
            grid_step,photon_count,transmitted_light,data,heating_curve_positions)

def energy(data, dt, Laser_power, transmitted_light, photon_count, C_tissue, grid_step, ro):
    """returns an array with distribution of absorbed photons converted to temperature increment"""
    
    Q = Laser_power*dt*transmitted_light/photon_count #absorbed energy during dt per photon (unit of data array)
    T = Q/C_tissue/grid_step**3/ro #temperature increase per photon per one cell volume
    return(data*T)

def CD_solver(u,ut,dt,lt,nt,D,grid_step, h_t, body_temp, heating_curve_positions):
    """returns 3D array with ZX slices and time in Y axis.
    u - temperature array,
    ut - temperature increment per dt (source) array must be of the same shape as u,
    dt - time step,
    nt - amount of time steps,
    D - diffusion coefficient,
    grid_step - spatial step.
    """
    ut0 = np.zeros_like(ut) # will be used temp increment when laser is off
    steps_per_second = int(1/dt)
    saved_time_points = int(nt*dt)+2

    u_snap = np.full((u.shape[0],saved_time_points,u.shape[2]),np.amin(u)) #array for storage plane sections in time
    saved_curves = np.zeros((saved_time_points,6)) #array for heating curves [time,temperature]
    u_snap[:,0,:] = u[:,int(u.shape[1]/2),:].copy()
    #saved_curves[0,0]=0
    #saved_curves[0,1] = u[heating_curve_positions[3],heating_curve_positions[2],heating_curve_positions[1]].copy()
    for l in tqdm.tqdm(range(1,nt+1)):
        un=u.copy() #data[z,y,x] возможно стоит вернуть .copy()
        if l == (lt-1):
            ut = ut0
        uijk=u[1:-1,1:-1,1:-1]
        unijk=un[1:-1,1:-1,1:-1]
        un2ijk = un[2:,1:-1,1:-1]
        un_2ijk = un[:-2,1:-1,1:-1]
        uni2jk = un[1:-1,2:,1:-1]
        uni_2jk = un[1:-1,:-2,1:-1]
        unij2k = un[1:-1,1:-1,2:]
        unij_2k = un[1:-1,1:-1,:-2]
        utijk = ut[1:-1,1:-1,1:-1]
        uijk = ne.evaluate("unijk+D*dt/grid_step**2*((un2ijk-2*unijk+un_2ijk) + (uni2jk-2*unijk+uni_2jk) + (unij2k-2*unijk+unij_2k)) + utijk")
        u[1:-1,1:-1,1:-1] = uijk
        #test adding surface cooling
        #print(np.amax(h_t*(un[0,1:-1,1:-1]-body_temp)))
        u[0,1:-1,1:-1] = un[0,1:-1,1:-1]+D*dt/grid_step**2*(
                            2*(un[1,1:-1,1:-1]-un[0,1:-1,1:-1]) +
                            (un[0,2:,1:-1]-2*un[0,1:-1,1:-1]+un[0,:-2,1:-1]) +
                            (un[0,1:-1,2:]-2*un[0,1:-1,1:-1]+un[0,1:-1,:-2])
                            ) + ut[0,1:-1,1:-1] - h_t*(un[0,1:-1,1:-1]-body_temp) #surface Z = 0
        u[-1,1:-1,1:-1] = un[-1,1:-1,1:-1]+D*dt/grid_step**2*(
                            2*(un[-2,1:-1,1:-1]-un[-1,1:-1,1:-1]) +
                            (un[-1,2:,1:-1]-2*un[-1,1:-1,1:-1]+un[-1,:-2,1:-1]) +
                            (un[-1,1:-1,2:]-2*un[-1,1:-1,1:-1]+un[-1,1:-1,:-2])
                            ) + ut[-1,1:-1,1:-1] - h_t*(un[-1,1:-1,1:-1]-body_temp) #surface Z = -1
        u[1:-1,0,1:-1] = un[1:-1,0,1:-1]+D*dt/grid_step**2*(
                            (un[2:,0,1:-1]-2*un[1:-1,0,1:-1]+un[:-2,0,1:-1]) +
                            2*(un[1:-1,1,1:-1]-un[1:-1,0,1:-1]) +
                            (un[1:-1,0,2:]-2*un[1:-1,0,1:-1]+un[1:-1,0,:-2])
                            ) + ut[1:-1,0,1:-1] - h_t*(un[1:-1,0,1:-1]-body_temp) #surface Y = 0
        u[1:-1,-1,1:-1] = un[1:-1,-1,1:-1]+D*dt/grid_step**2*(
                            (un[2:,-1,1:-1]-2*un[1:-1,-1,1:-1]+un[:-2,-1,1:-1]) +
                            2*(un[1:-1,-2,1:-1]-un[1:-1,-1,1:-1]) +
                            (un[1:-1,-1,2:]-2*un[1:-1,-1,1:-1]+un[1:-1,-1,:-2])
                            ) + ut[1:-1,-1,1:-1] - h_t*(un[1:-1,-1,1:-1]-body_temp) #surface Y = -1
        u[1:-1,1:-1,0] = un[1:-1,1:-1,0]+D*dt/grid_step**2*(
                            (un[2:,1:-1,0]-2*un[1:-1,1:-1,0]+un[:-2,1:-1,0]) +
                            (un[1:-1,2:,0]-2*un[1:-1,1:-1,0]+un[1:-1,:-2,0]) +
                            2*(un[1:-1,1:-1,1]-un[1:-1,1:-1,0])
                            ) + ut[1:-1,1:-1,0] - h_t*(un[1:-1,1:-1,0]-body_temp) #surface X = 0
        u[1:-1,1:-1,-1] = un[1:-1,1:-1,-1]+D*dt/grid_step**2*(
                            (un[2:,1:-1,-1]-2*un[1:-1,1:-1,-1]+un[:-2,1:-1,-1]) +
                            (un[1:-1,2:,-1]-2*un[1:-1,1:-1,-1]+un[1:-1,:-2,-1]) +
                            2*(un[1:-1,1:-1,-2]-un[1:-1,1:-1,-1])
                            ) + ut[1:-1,1:-1,-1] - h_t*(un[1:-1,1:-1,-1]-body_temp) #surface X = -1
        u[0,0,1:-1] = un[0,0,1:-1]+D*dt/grid_step**2*(
                            2*(un[1,0,1:-1]-un[0,0,1:-1]) +
                            2*(un[0,1,1:-1]-un[0,0,1:-1]) +
                            (un[0,0,2:]-2*un[0,0,1:-1]+un[0,0,:-2])
                            ) + ut[0,0,1:-1] - 2*h_t*(un[0,0,1:-1]-body_temp) #border Z = 0, Y = 0
        u[0,-1,1:-1] = un[0,-1,1:-1]+D*dt/grid_step**2*(
                            2*(un[1,-1,1:-1]-un[0,-1,1:-1]) +
                            2*(un[0,-2,1:-1]-un[0,-1,1:-1]) +
                            (un[0,-1,2:]-2*un[0,-1,1:-1]+un[0,-1,:-2])
                            ) + ut[0,-1,1:-1] - 2*h_t*(un[0,-1,1:-1]-body_temp) #border Z = 0, Y = -1
        u[0,1:-1,0] = un[0,1:-1,0]+D*dt/grid_step**2*(
                            2*(un[1,1:-1,0]-un[0,1:-1,0]) +
                            (un[0,2:,0]-2*un[0,1:-1,0]+un[0,:-2,0]) +
                            2*(un[0,1:-1,1]-un[0,1:-1,0])
                            ) + ut[0,1:-1,0] - 2*h_t*(un[0,1:-1,0]-body_temp) #border Z = 0, X = 0
        u[0,1:-1,-1] = un[0,1:-1,-1]+D*dt/grid_step**2*(
                            2*(un[1,1:-1,-1]-un[0,1:-1,-1]) +
                            (un[0,2:,-1]-2*un[0,1:-1,-1]+un[0,:-2,-1]) +
                            2*(un[0,1:-1,-2]-un[0,1:-1,-1])
                            ) + ut[0,1:-1,-1] - 2*h_t*(un[0,1:-1,-1]-body_temp) #border Z = 0, X = -1
        u[-1,0,1:-1] = un[-1,0,1:-1]+D*dt/grid_step**2*(
                            2*(un[-2,0,1:-1]-un[-1,0,1:-1]) +
                            2*(un[-1,1,1:-1]-un[-1,0,1:-1]) +
                            (un[-1,0,2:]-2*un[-1,0,1:-1]+un[-1,0,:-2])
                            ) + ut[-1,0,1:-1] - 2*h_t*(un[-1,0,1:-1]-body_temp) #border Z = -1, Y = 0
        u[-1,-1,1:-1] = un[-1,-1,1:-1]+D*dt/grid_step**2*(
                            2*(un[-2,-1,1:-1]-un[-1,-1,1:-1]) +
                            2*(un[-1,-2,1:-1]-un[-1,-1,1:-1]) +
                            (un[-1,-1,2:]-2*un[-1,-1,1:-1]+un[-1,-1,:-2])
                            ) + ut[-1,-1,1:-1] - 2*h_t*(un[-1,-1,1:-1]-body_temp) #border Z = -1, Y = -1
        u[-1,1:-1,0] = un[-1,1:-1,0]+D*dt/grid_step**2*(
                            2*(un[-2,1:-1,0]-un[-1,1:-1,0]) +
                            (un[-1,2:,0]-2*un[-1,1:-1,0]+un[-1,:-2,0]) +
                            2*(un[-1,1:-1,1]-un[-1,1:-1,0])
                            ) + ut[-1,1:-1,0] - 2*h_t*(un[-1,1:-1,0]-body_temp) #border Z = -1, X = 0
        u[-1,1:-1,-1] = un[-1,1:-1,-1]+D*dt/grid_step**2*(
                            2*(un[-2,1:-1,-1]-un[-1,1:-1,-1]) +
                            (un[-1,2:,-1]-2*un[-1,1:-1,-1]+un[-1,:-2,-1]) +
                            2*(un[-1,1:-1,-2]-un[-1,1:-1,-1])
                            ) + ut[-1,1:-1,-1] - 2*h_t*(un[-1,1:-1,-1]-body_temp) #border Z = -1, X = -1
        u[1:-1,0,0] = un[1:-1,0,0]+D*dt/grid_step**2*(
                            (un[2:,0,0]-2*un[1:-1,0,0]+un[:-2,0,0]) +
                            2*(un[1:-1,1,0]-un[1:-1,0,0]) +
                            2*(un[1:-1,0,1]-un[1:-1,0,0])
                            ) + ut[1:-1,0,0] - 2*h_t*(un[1:-1,0,0]-body_temp) #border Y = 0, X = 0
        u[1:-1,0,-1] = un[1:-1,0,-1]+D*dt/grid_step**2*(
                            (un[2:,0,-1]-2*un[1:-1,0,-1]+un[:-2,0,-1]) +
                            2*(un[1:-1,1,-1]-un[1:-1,0,-1]) +
                            2*(un[1:-1,0,-2]-un[1:-1,0,-1])
                            ) + ut[1:-1,0,-1] - 2*h_t*(un[1:-1,0,-1]-body_temp) #border Y = 0, X = -1
        u[1:-1,-1,0] = un[1:-1,-1,0]+D*dt/grid_step**2*(
                            (un[2:,-1,0]-2*un[1:-1,-1,0]+un[:-2,-1,0]) +
                            2*(un[1:-1,-2,0]-un[1:-1,-1,0]) +
                            2*(un[1:-1,-1,1]-un[1:-1,-1,0])
                            ) + ut[1:-1,-1,0] - 2*h_t*(un[1:-1,-1,0]-body_temp) #border Y = -1, X = 0
        u[1:-1,-1,-1] = un[1:-1,-1,-1]+D*dt/grid_step**2*(
                            (un[2:,-1,-1]-2*un[1:-1,-1,-1]+un[:-2,-1,-1]) +
                            2*(un[1:-1,-2,-1]-un[1:-1,-1,-1]) +
                            2*(un[1:-1,-1,-2]-un[1:-1,-1,-1])
                            ) + ut[1:-1,-1,-1] - 2*h_t*(un[1:-1,-1,-1]-body_temp) #border Y = 0, X = -1
        u[0,0,0] = un[0,0,0]+D*dt/grid_step**2*(
                            2*(un[1,0,0]-un[0,0,0]) +
                            2*(un[0,1,0]-un[0,0,0]) +
                            2*(un[0,0,1]-un[0,0,0])
                            ) + ut[0,0,0] - 3*h_t*(un[0,0,0]-body_temp) #point Z = 0, Y = 0, X = 0
        u[0,0,-1] = un[0,0,-1]+D*dt/grid_step**2*(
                            2*(un[1,0,-1]-un[0,0,-1]) +
                            2*(un[0,1,-1]-un[0,0,-1]) +
                            2*(un[0,0,-2]-un[0,0,-1])
                            ) + ut[0,0,-1] - 3*h_t*(un[0,0,-1]-body_temp) #point Z = 0, Y = 0, X = -1
        u[0,-1,0] = un[0,-1,0]+D*dt/grid_step**2*(
                            2*(un[1,-1,0]-un[0,-1,0]) +
                            2*(un[0,-2,0]-un[0,-1,0]) +
                            2*(un[0,-1,1]-un[0,-1,0])
                            ) + ut[0,-1,0] - 3*h_t*(un[0,-1,0]-body_temp) #point Z = 0, Y = -1, X = 0
        u[0,-1,-1] = un[0,-1,-1]+D*dt/grid_step**2*(
                            2*(un[1,-1,-1]-un[0,-1,-1]) +
                            2*(un[0,-2,-1]-un[0,-1,-1]) +
                            2*(un[0,-1,-2]-un[0,-1,-1])
                            ) + ut[0,-1,-1] - 3*h_t*(un[0,-1,-1]-body_temp) # point Z = 0, Y = -1, X = -1
        u[-1,0,0] = un[-1,0,0]+D*dt/grid_step**2*(
                            2*(un[-2,0,0]-un[-1,0,0]) +
                            2*(un[-1,1,0]-un[-1,0,0]) +
                            2*(un[-1,0,1]-un[-1,0,0])
                            ) + ut[-1,0,0] - 3*h_t*(un[-1,0,0]-body_temp) #point Z = -1, Y = 0, X = 0
        u[-1,0,-1] = un[-1,0,-1]+D*dt/grid_step**2*(
                            2*(un[-2,0,-1]-un[-1,0,-1]) +
                            2*(un[-1,1,-1]-un[-1,0,-1]) +
                            2*(un[-1,0,-2]-un[-1,0,-1])
                            ) + ut[-1,0,-1] - 3*h_t*(un[-1,0,-1]-body_temp) #point Z = -1, Y = 0, X = -1
        u[-1,-1,0] = un[-1,-1,0]+D*dt/grid_step**2*(
                            2*(un[-2,-1,0]-un[-1,-1,0]) +
                            2*(un[-1,-2,0]-un[-1,-1,0]) +
                            2*(un[-1,-1,1]-un[-1,-1,0])
                            ) + ut[-1,-1,0] - 3*h_t*(un[-1,-1,0]-body_temp) #point Z = -1, Y = -1, X = 0
        u[-1,-1,-1] = un[-1,-1,-1]+D*dt/grid_step**2*(
                            2*(un[-2,-1,-1]-un[-1,-1,-1]) +
                            2*(un[-1,-2,-1]-un[-1,-1,-1]) +
                            2*(un[-1,-1,-2]-un[-1,-1,-1])
                            ) + ut[-1,-1,-1] - 3*h_t*(un[-1,-1,-1]-body_temp) # point Z = 0, Y = -1, X = -1
        #print('time = ',round(l*dt,2), 's, Temp_min = ', np.amin(u), 'C, Temp_max = ', round(np.amax(u),2), 'C')
        if (l%steps_per_second) == 0 and int(l/steps_per_second)<u_snap.shape[1]:
            u_snap[:,int(l/steps_per_second),:] = u[:,int(u.shape[1]/2),:].copy()
            for i in range(3):
                saved_curves[int(l/steps_per_second),2*i]=l*dt
                saved_curves[int(l/steps_per_second),2*i+1] = u[heating_curve_positions[4*i+3],heating_curve_positions[4*i+2],heating_curve_positions[4*i+1]].copy()-u_snap[0,0,0].copy()
    return u_snap, saved_curves

def save_config(dt,lt,nt,heat_filename,curves_filename):
    """Save configuration parameters to MonteCarlo.ini"""

    config = configparser.ConfigParser(comment_prefixes='/', allow_no_value=True)
    config.read('SimulationParams.ini')
    config.set('Heat modelling params', 'heating duration [s]', str(dt*lt))
    config.set('Heat modelling params', 'modelling duration [s]', str(dt*nt))
    config.set('Technical data', 'heat_filename', heat_filename)
    config.set('Technical data', 'Heating curves filename',curves_filename)

    with open('SimulationParams.ini', 'w') as configfile:
        config.write(configfile)
        print('Config is saved to: SimulationParams.ini')

def save_data(data_snap, saved_curves):

    temp_filename = params['light distribution filename']
    start_i = temp_filename.find('_')
    end_i = temp_filename.find('_', temp_filename.find('beam'))
    filename_p2 = temp_filename[start_i+1:end_i]
    filename_p1 = 'modeling results/Heat2D' + filename_p2 + '_k' + str(params['k']) + '_C' + str(params['C_tissue'])\
         + '_p' + str(params['ro']) + '_h' + str(params['h']) + '_Power' + str(params['Laser power']) + '_run'

    i = 1
    while (os.path.exists(filename_p1 + str(i) + '.npy')):
        i += 1
    filename = filename_p1 + str(i)

    if params['save full data'] == 1:
        np.save(filename, data_snap)

    y_dist = round((params['heating curve y pos'] + params['laser y pos'] - params['model area y']/2),1)
    heat_filename_p1 = 'modeling results/HeatCurve_Y-dist' + str(y_dist) + filename_p2 + '_k' + str(params['k'])\
         + '_C' + str(params['C_tissue']) + '_p' + str(params['ro']) + '_h' + str(params['h']) + '_Power' + str(params['Laser power']) + '_run'

    i = 1
    while (os.path.exists(heat_filename_p1 + str(i) + '.txt')):
        i += 1
    heat_filename = heat_filename_p1 + str(i) + '.txt'
    np.savetxt(heat_filename, saved_curves)

    print('\nResults are saved to ', heat_filename, '\nand to', saved_curves, '\n')

    return(filename + '.npy', heat_filename)

if __name__ == "__main__":

    print('\nМодуль расчёта фотонагрева ткани.')
    print('Разработан при финансовой поддержке Российского Научного Фонда (грант № 20-72-00081).')
    print('Антон Попов 2022: a.popov.fizteh@gmail.com')

    #Load parameters from SimulationParams.ini
    k, ro, C_tissue, body_temp, Laser_power, Laser_duration, Total_duration, data_x_size, data_y_size, data_z_size,\
            grid_step, photon_count, transmitted_light, data, heating_curve_positions = load_config()

    if params['heat iteration mode'] == 'single':
        dt = grid_step**2*ro*C_tissue/(6*k)/3
        print('')
        print('Calculated parameter: Time step dt = ', dt, ' s')

        lt = int(Laser_duration/dt)
        nt = int(Total_duration/dt)
        print('Calculated parameter: Amount of time points nt = ', nt)

        ut = energy(data.copy(),dt, Laser_power, transmitted_light, photon_count, C_tissue, grid_step, ro)
        
        u = np.full_like(ut,body_temp) #array with initial body temperature
        D = k/ro/C_tissue #Diffusion coefficient
        h_t = dt*params['h']/(C_tissue*ro*grid_step) # dT = h_t*(T_surf-T_body) in celsium
        print('Calculated parameter: Diffusion coefficient D = ', D, ' m^2/s')

        data_snap, saved_curves = CD_solver(u,ut,dt,lt,nt,D,grid_step, h_t, body_temp, heating_curve_positions)
        heat_filename, curves_filename = save_data(data_snap, saved_curves)
        
        save_config(dt,lt,nt,heat_filename,curves_filename)

    elif params['heat iteration mode'] == 'iteration':
        print('\nStart of heat parameters optimization procedure...')
        ref_data = np.loadtxt(params['reference curve filename'])
        print('Reference curve loaded with data shape', ref_data.shape)
        start_time_index = 0
        for i in range(50):
            if (ref_data[i+7,4] - ref_data[i,4]) > 1:
                start_time_index = i
                break
        print('Start heating index =', start_time_index, ' start heating time =', ref_data[start_time_index,0])
        finish_time_index = (np.abs(ref_data[:,0] - Total_duration - ref_data[start_time_index,0] - 1)).argmin()
        print('Finish time index =', finish_time_index, 'Finish time =', ref_data[finish_time_index,0])
        ref_data[:,0] -= ref_data[start_time_index,0]
        
        if (params['iterate specific heat'] + params['iterate density'] + params['iterate thermal conductivity']) == 2:
            print('Optimization of 2 variables will be performed:')
            if params['iterate specific heat'] == 1:
                C_tissue_0 = C_tissue*(1-params['heat iteration range']/200)
                dC = C_tissue*params['heat iteration range']/100/(params['heat iterations']-1)
                print('Slow variable: Specific heat. Start value C0 = ', C_tissue_0, 'step dC = ', dC)
                
                if params['iterate thermal conductivity'] == 1:
                    k0 = k*(1-params['heat iteration range']/200)
                    dk = k*params['heat iteration range']/100/(params['heat iterations']-1)
                    print('Fast variable: Thermal conductivity. Start value k0 = ', k0, 'step dk = ', dk)
                    
                    opt_arr = np.zeros((params['heat iterations'],params['heat iterations']))
                    opt_curve = np.empty((int(Total_duration)+1,3))
                    C_opt = C_tissue_0
                    k_opt = k0
                    for i in range(params['heat iterations']):
                        for j in range(params['heat iterations']):
                            dt = grid_step**2*ro*(C_tissue_0+i*dC)/(6*(k0+j*dk))/3
                            print('')
                            print('Calculated parameter: Time step dt = ', dt, ' s')

                            lt = int(Laser_duration/dt)
                            nt = int(Total_duration/dt)
                            print('Calculated parameter: Amount of time points nt = ', nt)
                            print('Iterated parameters: C = ', C_tissue_0+i*dC, 'k = ', k0+j*dk)

                            ut = energy(data.copy(),dt, Laser_power, transmitted_light, photon_count, (C_tissue_0+i*dC), grid_step, ro)
                            
                            u = np.full_like(ut,body_temp) #array with initial body temperature
                            D = (k0+j*dk)/ro/(C_tissue_0+i*dC) #Diffusion coefficient
                            print('Calculated parameter: Diffusion coefficient D = ', D, ' m^2/s')

                            data_snap, saved_curves = CD_solver(u,ut,dt,lt,nt,D,grid_step, heating_curve_positions)
                            ref_data_interp = interp.interp1d(ref_data[start_time_index:finish_time_index,0],ref_data[start_time_index:finish_time_index,4])
                            ref_for_compare = ref_data_interp(saved_curves[:,0])
                            sqrt_dev = (sum((saved_curves[:,1]-ref_for_compare[:])**2))**0.5
                            print('sqrt_dev = ', sqrt_dev)
                            opt_arr[i,j] = sqrt_dev
                            minval = np.min(opt_arr[opt_arr.nonzero()])

                            if i == 0:
                                opt_curve[:,:2] = saved_curves[:,:2]
                                opt_curve[:,2] = ref_for_compare[:]
                            
                            if sqrt_dev < minval:
                                opt_curve[:,:2] = saved_curves[:,:2]
                                opt_curve[:,2] = ref_for_compare[:]
                                C_opt = C_tissue_0+i*dC
                                k_opt = k0+j*dk
                                print('Opt curve updated at C_opt = ', C_opt, 'k_opt = ', k_opt)
                            print(opt_arr)

                    np.savetxt('opt_arr.txt',opt_arr)
                    np.savetxt('opt_heating_curve_noNPs_x' + params['ref curve x pos'] + 'y' + params['ref curve y pos'] + 'z' + params['ref curve z pos'] + 'Copt=' + str(C_opt) + 'kopt=' + str(k_opt) + '.txt', opt_curve)                
                else:
                    print('Fast variable: Density')
                    pass
            else:
                print('Slow variable: Thermal conductivity')
                print('Fast variable: Density')
                pass
        
        elif (params['iterate specific heat'] + params['iterate density'] + params['iterate thermal conductivity']) == 1:
            print('Optimization of 1 variables will be performed:')
            if params['iterate specific heat'] == 1:
                C_tissue_0 = C_tissue*(1-params['heat iteration range']/200)
                dC = C_tissue*params['heat iteration range']/100/(params['heat iterations']-1)
                print('Iteration variable: Specific heat. Start value C0 = ', C_tissue_0, 'step dC = ', dC)
                    
                opt_arr = np.zeros((params['heat iterations']))
                opt_curve = np.empty((int(Total_duration)+1,3))
                C_opt = C_tissue_0

                for i in range(params['heat iterations']):
                    dt = grid_step**2*ro*(C_tissue_0+i*dC)/(6*k)/3
                    print('')
                    print('Calculated parameter: Time step dt =', dt, ' s')

                    lt = int(Laser_duration/dt)
                    nt = int(Total_duration/dt)
                    print('Calculated parameter: Amount of time points nt =', nt)
                    print('Iterated parameter: C =', C_tissue_0+i*dC)

                    ut = energy(data.copy(),dt, Laser_power, transmitted_light, photon_count, (C_tissue_0+i*dC), grid_step, ro)
                    
                    u = np.full_like(ut,body_temp) #array with initial body temperature
                    D = k/ro/(C_tissue_0+i*dC) #Diffusion coefficient
                    print('Calculated parameter: Diffusion coefficient D =', D, ' m^2/s')

                    data_snap, saved_curves = CD_solver(u,ut,dt,lt,nt,D,grid_step, heating_curve_positions)

                    ref_data_interp = interp.interp1d(ref_data[start_time_index:finish_time_index,0],ref_data[start_time_index:finish_time_index,4])
                    ref_for_compare = ref_data_interp(saved_curves[:,0])
                    sqrt_dev = (sum((saved_curves[:,1]-ref_for_compare[:])**2))**0.5

                    print('sqrt_dev =', sqrt_dev)
                    
                    if i == 0:
                        opt_curve[:,:2] = saved_curves[:,:2]
                        opt_curve[:,2] = ref_for_compare[:]

                    opt_arr[i] = sqrt_dev
                    minval = np.min(opt_arr[opt_arr.nonzero()])
                    if sqrt_dev < minval:
                        opt_curve[:,:2] = saved_curves[:,:2]
                        opt_curve[:,2] = ref_for_compare[:]
                        C_opt = C_tissue_0+i*dC
                        print('Opt curve updated at C_opt =', C_opt)
                    print(opt_arr)

                np.savetxt('C_opt_arr.txt',opt_arr)
                np.savetxt('C_opt_heating_curve_x' + params['ref curve x pos'] + 'y' + params['ref curve y pos'] + 'z' + params['ref curve z pos'] + 'Copt=' + str(C_opt) + '.txt', opt_curve)                

            elif params['iterate thermal conductivity'] == 1:
                k0 = k*(1-params['heat iteration range']/200)
                dk = k*params['heat iteration range']/100/(params['heat iterations']-1)
                print('Iteration variable: Thermal conductivity. Start value k0 = ', k0, 'step dk = ', dk)
                    
                opt_arr = np.zeros((params['heat iterations']))
                opt_curve = np.empty((int(Total_duration)+1,3))
                k_opt = k0

                for i in range(params['heat iterations']):
                    dt = grid_step**2*ro*C_tissue/(6*(k0+i*dk))/3
                    print('')
                    print('Calculated parameter: Time step dt = ', dt, ' s')

                    lt = int(Laser_duration/dt)
                    nt = int(Total_duration/dt)
                    print('Calculated parameter: Amount of time points nt = ', nt)
                    print('Iterated parameter: k = ', k0+i*dk)

                    ut = energy(data.copy(),dt, Laser_power, transmitted_light, photon_count, C_tissue, grid_step, ro)
                    
                    u = np.full_like(ut,body_temp) #array with initial body temperature
                    D = (k0+i*dk)/ro/C_tissue #Diffusion coefficient
                    print('Calculated parameter: Diffusion coefficient D = ', D, ' m^2/s')

                    data_snap, saved_curves = CD_solver(u,ut,dt,lt,nt,D,grid_step, heating_curve_positions)

                    ref_data_interp = interp.interp1d(ref_data[start_time_index:finish_time_index,0],ref_data[start_time_index:finish_time_index,4])
                    ref_for_compare = ref_data_interp(saved_curves[:,0])
                    sqrt_dev = (sum((saved_curves[:,1]-ref_for_compare[:])**2))**0.5

                    print('sqrt_dev = ', sqrt_dev)

                    if i == 0:
                        opt_curve[:,:2] = saved_curves[:,:2]
                        opt_curve[:,2] = ref_for_compare[:]


                    opt_arr[i] = sqrt_dev
                    minval = np.min(opt_arr[opt_arr.nonzero()])
                    if sqrt_dev < minval:
                        opt_curve[:,:2] = saved_curves[:,:2]
                        opt_curve[:,2] = ref_for_compare[:]
                        k_opt = k0+i*dk
                        print('Opt curve updated at k_opt = ', k_opt)
                    print(opt_arr)

                np.savetxt('k_opt_arr.txt',opt_arr)
                np.savetxt('k_opt_heating_curve_x' + params['ref curve x pos'] + 'y' + params['ref curve y pos'] + 'z' + params['ref curve z pos'] + 'kopt=' + str(k_opt) + '.txt', opt_curve)

            elif params['iterate density'] == 1:
                ro0 = ro*(1-params['heat iteration range']/200)
                dro = ro*params['heat iteration range']/100/(params['heat iterations']-1)
                print('Iteration variable: Density. Start value ro0 = ', ro0, 'step d(ro) = ', dro)
                    
                opt_arr = np.zeros((params['heat iterations']))
                opt_curve = np.empty((int(Total_duration)+1,3))
                ro_opt = ro0

                for i in range(params['heat iterations']):
                    dt = grid_step**2*(ro0+i*dro)*C_tissue/(6*k)/3
                    print('')
                    print('Calculated parameter: Time step dt = ', dt, ' s')

                    lt = int(Laser_duration/dt)
                    nt = int(Total_duration/dt)
                    print('Calculated parameter: Amount of time points nt = ', nt)
                    print('Iterated parameter: ro = ', ro0+i*dro)

                    ut = energy(data.copy(),dt, Laser_power, transmitted_light, photon_count, C_tissue, grid_step, (ro0+i*dro))
                    
                    u = np.full_like(ut,body_temp) #array with initial body temperature
                    D = k/(ro0+i*dro)/C_tissue #Diffusion coefficient
                    print('Calculated parameter: Diffusion coefficient D = ', D, ' m^2/s')

                    data_snap, saved_curves = CD_solver(u,ut,dt,lt,nt,D,grid_step, heating_curve_positions)

                    ref_data_interp = interp.interp1d(ref_data[start_time_index:finish_time_index,0],ref_data[start_time_index:finish_time_index,4])
                    ref_for_compare = ref_data_interp(saved_curves[:,0])
                    sqrt_dev = (sum((saved_curves[:,1]-ref_for_compare[:])**2))**0.5

                    print('sqrt_dev = ', sqrt_dev)

                    if i == 0:
                        opt_curve[:,:2] = saved_curves[:,:2]
                        opt_curve[:,2] = ref_for_compare[:]

                    opt_arr[i] = sqrt_dev
                    minval = np.min(opt_arr[opt_arr.nonzero()])
                    if sqrt_dev < minval:
                        opt_curve[:,:2] = saved_curves[:,:2]
                        opt_curve[:,2] = ref_for_compare[:]
                        ro_opt = ro0+i*dro
                        print('Opt curve updated at ro_opt = ', ro_opt)
                    print(opt_arr)

                np.savetxt('ro_opt_arr.txt',opt_arr)
                np.savetxt('ro_opt_heating_curve_x' + params['ref curve x pos'] + 'y' + params['ref curve y pos'] + 'z' + params['ref curve z pos'] + 'kopt=' + str(ro_opt) + '.txt', opt_curve)
                
            else:
                print('Error! Iteration variable cannot be identified!')
                
    else:
        print('Error! Unknown heat iteration mode!')