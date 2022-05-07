#Модуль расчёта пространсвенного распределения поглощенного света.
#Разработана при финансовой поддержке Российского Научного Фонда (грант № 20-72-00081).
#Антон Попов 2022: a.popov.fizteh@gmail.com

import enum
import random
import math
import numpy as np
import tqdm
import multiprocessing as mp
import configparser
from datetime import datetime
import os.path

params = dict()

def load_config():
    """Load simulation parameters"""

    print('')
    print('Start loading simulation params from SimulationParams.ini...')
    print('')
    config = configparser.ConfigParser()
    config.read('SimulationParams.ini')

    #load general monte carlo parameters
    params['simulation_mode'] = config['MonteCarlo Params']['simulation mode']
    print('Parameter loaded: simulation_mode =', params['simulation_mode'])

    params['iter_start_z'] = float(config['MonteCarlo Params']['start z [mm]'])
    params['iter_stop_z'] = float(config['MonteCarlo Params']['stop z [mm]'])
    params['iter_step_z'] = float(config['MonteCarlo Params']['step z [mm]'])
    
    params['iter_start_tissue_abs'] = float(config['MonteCarlo Params']['start tissue abs [1/cm]'])
    params['iter_stop_tissue_abs'] = float(config['MonteCarlo Params']['stop tissue abs [1/cm]'])
    params['iter_step_tissue_abs'] = float(config['MonteCarlo Params']['step tissue abs [1/cm]'])

    params['iter_start_tissue_sca'] = float(config['MonteCarlo Params']['start tissue sca [1/cm]'])
    params['iter_stop_tissue_sca'] = float(config['MonteCarlo Params']['stop tissue sca [1/cm]'])
    params['iter_step_tissue_sca'] = float(config['MonteCarlo Params']['step tissue sca [1/cm]'])

    params['iter_start_NPs_abs'] = float(config['MonteCarlo Params']['start nps abs [1/cm]'])
    params['iter_stop_NPs_abs'] = float(config['MonteCarlo Params']['stop nps abs [1/cm]'])
    params['iter_step_NPs_abs'] = float(config['MonteCarlo Params']['step nps abs [1/cm]'])

    params['max_iterations'] = int(float(config['MonteCarlo Params']['max iterations']))
    print('Parameter loaded: max_iterations =', params['max_iterations'])

    #load model geometry
    params['model_area_x'] = float(config['Model Geometry']['model area x [mm]'])
    print('Parameter loaded: model_area_x =', params['model_area_x'], 'mm')
    params['model_area_y'] = float(config['Model Geometry']['model area y [mm]'])
    print('Parameter loaded: model_area_y =', params['model_area_y'], 'mm')
    params['model_area_z'] = float(config['Model Geometry']['model area z [mm]'])
    print('Parameter loaded: model_area_z =', params['model_area_z'], 'mm')

    params['grid_step'] = float(config['Model Geometry']['grid step [um]'])
    print('Parameter loaded: grid_step =', params['grid_step'], '[um]')

    params['data_x_size'] = int(1000*params['model_area_x'] / params['grid_step'])
    print('Parameter calculated: data_x_size =', params['data_x_size'])
    params['data_y_size'] = int(1000*params['model_area_y'] / params['grid_step'])
    print('Parameter calculated: data_y_size =', params['data_y_size'])
    params['data_z_size'] = int(1000*params['model_area_z'] / params['grid_step'])
    print('Parameter calculated: data_z_size =', params['data_z_size'])

    #load laser beam parameters
    params['photon_count'] = int(float(config['MonteCarlo Params']['photon count']))
    print('Parameter loaded: photon_count =', params['photon_count'])
    params['beam_width'] = float(config['Illumination params']['beam width [mm]'])
    print('Parameter loaded: beam_width (SD) =', params['beam_width'], '[mm]')
    if (config['Illumination params']['center x']) == 'center':
        params['beam center x'] = params['model_area_x']/2
    else:
        params['beam center x'] = float(config['Illumination params']['center x'])
    print('Parameter loaded: beam center x =', params['beam center x'], 'mm')
    if (config['Illumination params']['center y']) == 'center':
        params['beam center y'] = params['model_area_y']/2
    else:
        params['beam center y'] = float(config['Illumination params']['center y'])
    print('Parameter loaded: beam center y =', params['beam center y'], 'mm')

    #load  tissue optical parameters
    params['tissue_abs_coef'] = float(config['Tissue optical params']['tissue abs coef [1/cm]'])
    print('Parameter loaded: tissue_abs_coef =', params['tissue_abs_coef'], '[1/cm]')
    params['tissue_sca_coef'] = float(config['Tissue optical params']['tissue sca coef [1/cm]'])
    print('Parameter loaded: tissue_sca_coef =', params['tissue_sca_coef'], '[1/cm]')
    params['g'] = float(config['Tissue optical params']['g'])
    print('Parameter loaded: g =', params['g'])
    params['n_tissue'] = float(config['Tissue optical params']['n tissue'])
    print('Parameter loaded: n_tissue =', params['n_tissue'])
    params['n_ext'] = float(config['External medium optical params']['n ext'])
    print('Parameter loaded: n_ext =', params['n_ext'])

    #load NPs parameters
    params['np distribution'] = config['Nanoparticles params']['np distribution']
    print('\nNPs distribution:', params['np distribution'])

    if params['np distribution'] == 'sphere':
        params['np distribution radius'] = float(config['Nanoparticles params']['distribution radius [mm]'])
        print('Parameter loaded: radius =', params['np distribution radius'], '[mm]')
        if config['Nanoparticles params']['x position of distribution center [mm]'] == 'center':
            params['np spherical x position'] = params['model_area_x']/2
            print('Parameter loaded: x position of distribution center =', params['np spherical x position'], '[mm]')
        else:
            params['np spherical x position'] = int(float(config['Nanoparticles params']['x position of distribution center [mm]']))
            print('Parameter loaded: x position of distribution center =', params['np spherical x position'], '[mm]')
        if config['Nanoparticles params']['y position of distribution center [mm]'] == 'center':
            params['np spherical y position'] = params['model_area_y']/2
            print('Parameter loaded: y position of distribution center =', params['np spherical y position'], '[mm]')
        else:
            params['np spherical y position'] = int(float(config['Nanoparticles params']['y position of distribution center [mm]']))
            print('Parameter loaded: y position of distribution center =', params['np spherical y position'], '[mm]')
        if config['Nanoparticles params']['z position of distribution center [mm]'] == 'center':
            params['np spherical z position'] = params['model_area_z']/2
            print('Parameter loaded: z position of distribution center =', params['np spherical z position'], '[mm]')
        else:
            params['np spherical z position'] = int(float(config['Nanoparticles params']['z position of distribution center [mm]']))
            print('Parameter loaded: z position of distribution center =', params['np spherical z position'], '[mm]')
        params['np_abs_coef'] = float(config['Nanoparticles params']['np abs coef [1/cm]'])
        print('Parameter loaded: np_abs_coef =', params['np_abs_coef'], '[1/cm]')
        params['np_sca_coef'] = float(config['Nanoparticles params']['np sca coef [1/cm]'])
        print('Parameter loaded: np_sca_coef =', params['np_sca_coef'], '[1/cm]')
    elif params['np distribution'] == 'layer':
        params['np layer depth'] = float(config['Nanoparticles params']['np layer depth [mm]'])
        print('The layer is located', params['np layer depth'], 'under the top surface')
        params['np_abs_coef'] = float(config['Nanoparticles params']['np abs coef [1/cm]'])
        print('Parameter loaded: np_abs_coef =', params['np_abs_coef'], '[1/cm]')
        params['np_sca_coef'] = float(config['Nanoparticles params']['np sca coef [1/cm]'])
        print('Parameter loaded: np_sca_coef =', params['np_sca_coef'], '[1/cm]')
    elif params['np distribution'] == 'none':
        params['np_abs_coef'] = 0
        params['np_sca_coef'] = 0
        print('No NPs are present')
    return ()

def save_config(runs, top_photons, side_photons, bottom_photons, absorbed_photons, filename):
    """Save configuration parameters"""

    config = configparser.ConfigParser(comment_prefixes='/', allow_no_value=True)
    config.read('SimulationParams.ini')
    config.set('MonteCarlo Params', 'photon count', str(runs*mp.cpu_count()))
    config.set('Technical data', 'escaped top photons', str(top_photons))
    config.set('Technical data', 'escaped side photons', str(side_photons))
    config.set('Technical data', 'escaped bottom photons', str(bottom_photons))
    config.set('Technical data', 'absorbed photons', str(absorbed_photons))
    config.set('Model Geometry', 'data x size', str(params['data_x_size']))
    config.set('Model Geometry', 'data y size', str(params['data_y_size']))
    config.set('Model Geometry', 'data z size', str(params['data_z_size']))
    config.set('Technical data', 'absorbed light filename', filename)
    config.set('Technical data', 'reflected light', str((params['n_tissue']-params['n_ext'])**2/(params['n_tissue']+params['n_ext'])**2))

    ini_filename = filename[:-4] + '.ini'
    with open(ini_filename, 'w') as configfile:
        config.write(configfile)
        print('Config is saved to:', ini_filename)
    
    with open('SimulationParams.ini', 'w') as configfile:
        config.write(configfile)

def save_data(data_np):
    if params['np distribution'] == 'none':
        filename_p2 = '_noNPs'
    elif params['np distribution'] == 'sphere':
        filename_p2 = '_NPs-Abs-' + str(params['np_abs_coef']) + '_Sphere-X' + str(params['np spherical x position']) + 'Y' + str(params['np spherical y position']) + 'Z' + str(params['np spherical z position'])
    elif params['np distribution'] == 'layer':
        filename_p2 = '_NPs-Abs-' + str(params['np_abs_coef']) + '_Layer-Z' + str(params['np layer depth'])
    
    filename_p1 = 'modeling results/Light' + filename_p2 + '_Tissue-Abs-' + str(params['tissue_abs_coef']) + '-Sca-' + str(params['tissue_sca_coef']) + '_beam-X' + str(params['beam center x']) + 'Y' + str(params['beam center y']) + '_run'

    i = 1
    while (os.path.exists(filename_p1 + str(i) + '.npy')):
        i += 1
    filename = filename_p1 + str(i)
    
    np.save(filename,data_np)
    return(filename + '.npy')

def save_iter_log(save_params):
    
    if params['simulation_mode'] == 'calculate absorbed' and params['np distribution'] == 'sphere':
        save_string = (
            '\nSphere, '
            + 'Tissue Sca=' + str(save_params['tissue_sca']) + ' '
            + 'Abs=' + str(save_params['tissue_abs']) + ' '
            + 'NPs Abs=' + str(save_params['NPs_abs']) + ' '
            + 'Z=' + str(save_params['NPs_distrib_z_pos']) + ' '
            + 'Distribution Radius=' + str(save_params['NPs_distrib_radius']) + ': '
            + 'Relative photons increase=' + str(save_params['relative_photons_increase']) + ', '
            + 'Fractional increase=' + str(save_params['fractional_photons_increase']) + ' '
            + 'Fraction absorbed=' + str(save_params['fraction_wo_escape'])
            )

    elif params['simulation_mode'] == 'calculate absorbed' and params['np distribution'] == 'layer':
        save_string = (
            '\nLayer '
            + 'Tissue Sca=' + str(save_params['tissue_sca']) + ' '
            + 'Abs=' + str(save_params['tissue_abs']) + ' '
            + 'NPs Abs=' + str(save_params['NPs_abs']) + ' '
            + 'Z=' + str(save_params['NPs_distrib_z_pos']) + ': '
            + 'Relative photons increase=' + str(save_params['relative_photons_increase']) + ', '
            + 'Fractional increase=' + str(save_params['fractional_photons_increase']) + ' '
            + 'Fraction absorbed=' + str(save_params['fraction_wo_escape'])
            )

    with open('modeling results/Iteration log.txt', 'a') as log_file:
        log_file.write(save_string)

class Photon:
    x = 0 # initial coordinates
    y = 0
    z = 0
    x_prev = 0
    y_prev = 0
    z_prev = 0
    cos_x = 0 # cos of the photon along x
    cos_y = 0 # cos of the photon along y
    cos_z = 1 # cos of the photon along z
    l = 0 # step size
    l_remain = 0 #remaining of the step in case of boundary refraction
    refr_count = 0 #amount of refractions
    state = 'init'
    step_number = 0
    
    def __init__(self,x0,y0,z0,cos_x0,cos_y0, cos_z0):
        self.x = x0
        self.y = y0
        self.z = z0
        self.cos_x = cos_x0
        self.cos_y = cos_y0
        self.cos_z = cos_z0
    
    def step(self, mu_s, mu_a, params):
        self.x_prev = self.x
        self.y_prev = self.y
        self.z_prev = self.z
        mu_t = (mu_s + mu_a)
        self.l = - math.log(random.random())/mu_t

        self.x += self.l*self.cos_x
        self.y += self.l*self.cos_y
        self.z += self.l*self.cos_z
        
        #print('coordinates before step xyz', self.x,self.y,self.z)
        self.step_number += 1

        if self.z < 0:
            if self.z < 0 and self.step_number == 1:
                print('SELF.Z = ', self.z, 'self.z_prev = ', self.z_prev, 'cos_z = ', self.cos_z)
            R = fresnel(self.cos_x,self.cos_y, self.cos_z, params['n_tissue'], params['n_ext'], 'z_min')
            if random.random()<R:
                l_border = abs(self.z/self.cos_z)
                if l_border < 0:
                    print('!!! Error in frensel for z<0. l_border = ', l_border)
                
                x = self.x - l_border*self.cos_x #
                y = self.y - l_border*self.cos_y #возвращаем фотон на границу
                z = 0.00001 #
                if x >= 0 and x <= params['model_area_x'] and y >= 0 and y <= params['model_area_y'] and z >=0 and z <= params['model_area_z']:
                    self.refr_count +=1
                    self.x=x
                    self.y=y
                    self.z=z
                    self.cos_z = - self.cos_z #Отражение фотона от границы
                    self.l_remain += l_border #непройденная длина
            else:
                self.state = 'out top'
                return

        if self.z > params['model_area_z']:
            R = fresnel(self.cos_x,self.cos_y, self.cos_z, params['n_tissue'], params['n_ext'], 'z_max')
            if random.random()<R:
                l_border = (self.z - params['model_area_z'])/self.cos_z
                if l_border < 0:
                    print('!!! Error in frensel for z>0. cos_x = ', self.cos_x, 'cos_y = ', self.cos_y, 'cos_z = ', self.cos_z, 'l = ', self.l)
                    print('x = ', self.x, 'y = ', self.y, 'z = ', self.z, 'l_remain = ', self.l_remain, 'refr_count = ', self.refr_count)
                x = self.x - l_border*self.cos_x #
                y = self.y - l_border*self.cos_y #возвращаем фотон на границу
                z = params['model_area_z'] - 0.0000001
                if x >= 0 and x <= params['model_area_x'] and y >= 0 and y <= params['model_area_y'] and z >=0 and z <= params['model_area_z']:
                    self.refr_count +=1
                    self.x=x
                    self.y=y
                    self.z=z
                    self.cos_z = - self.cos_z #Отражение фотона от границы
                    self.l_remain += l_border #непройденная длина
            else:
                self.state = 'out down'
                return
        
        if self.x<0:
            R = fresnel(self.cos_x,self.cos_y, self.cos_z, params['n_tissue'], params['n_ext'], 'x_min')
            if random.random()<R:
                l_border = abs(self.x/self.cos_x)
                if l_border < 0:
                    print('!!! Error in frensel for x<0. l_border = ', l_border)
                x = 0.0000001
                y = self.y - l_border*self.cos_y #возвращаем фотон на границу
                z = self.z - l_border*self.cos_z #
                if x >= 0 and x <= params['model_area_x'] and y >= 0 and y <= params['model_area_y'] and z >=0 and z <= params['model_area_z']:
                    self.refr_count +=1
                    self.x=x
                    self.y=y
                    self.z=z
                    self.cos_x = - self.cos_x #Отражение фотона от границы
                    self.l_remain += l_border #непройденная длина
            else:
                self.state = 'out side'
                return
        
        if self.x > params['model_area_x']:
            R = fresnel(self.cos_x,self.cos_y, self.cos_z, params['n_tissue'], params['n_ext'], 'x_max')
            if random.random()<R:
                l_border = (self.x - params['model_area_x'])/self.cos_x
                if l_border < 0:
                    print('!!! Error in frensel for x>0.  cos_x = ', self.cos_x, 'cos_y = ', self.cos_y, 'cos_z = ', self.cos_z, 'l = ', self.l)
                    print('x = ', self.x, 'y = ', self.y, 'z = ', self.z, 'l_remain = ', self.l_remain, 'refr_count = ', self.refr_count)
                x = params['model_area_x'] - 0.0000001
                y = self.y - l_border*self.cos_y #возвращаем фотон на границу
                z = self.z - l_border*self.cos_z #
                if x >= 0 and x <= params['model_area_x'] and y >= 0 and y <= params['model_area_y'] and z >=0 and z <= params['model_area_z']:
                    self.refr_count +=1
                    self.x=x
                    self.y=y
                    self.z=z
                    self.cos_x = - self.cos_x #Отражение фотона от границы
                    self.l_remain += l_border #непройденная длина
            else:
                self.state = 'out side'
                return

        if self.y<0:
            R = fresnel(self.cos_x,self.cos_y, self.cos_z, params['n_tissue'], params['n_ext'], 'y_min')
            if random.random()<R:
                l_border = abs(self.y/self.cos_y)
                if l_border < 0:
                    print('!!! Error in frensel for y<0. l_border = ', l_border)
                x = self.x - l_border*self.cos_x #
                y = 0.0000001                    #возвращаем фотон на границу
                z = self.z - l_border*self.cos_z #
                if x >= 0 and x <= params['model_area_x'] and y >= 0 and y <= params['model_area_y'] and z >=0 and z <= params['model_area_z']:
                    self.refr_count +=1
                    self.x=x
                    self.y=y
                    self.z=z
                    self.cos_y = - self.cos_y #Отражение фотона от границы
                    self.l_remain += l_border #непройденная длина
            else:
                self.state = 'out side'
                return
        
        if self.y > params['model_area_y']:
            R = fresnel(self.cos_x,self.cos_y, self.cos_z, params['n_tissue'], params['n_ext'], 'y_max')
            if random.random()<R:
                l_border = (self.y - params['model_area_y'])/self.cos_y
                if l_border < 0:
                    print('!!! Error in frensel for y>0.  cos_x = ', self.cos_x, 'cos_y = ', self.cos_y, 'cos_z = ', self.cos_z, 'l = ', self.l)
                    print('x = ', self.x, 'y = ', self.y, 'z = ', self.z, 'l_remain = ', self.l_remain, 'refr_count = ', self.refr_count)
                x = self.x - l_border*self.cos_x #
                y = params['model_area_y'] - 0.0000001     #возвращаем фотон на границу
                z = self.z - l_border*self.cos_z #
                if x >= 0 and x <= params['model_area_x'] and y >= 0 and y <= params['model_area_y'] and z >=0 and z <= params['model_area_z']:
                    self.refr_count +=1
                    self.x=x
                    self.y=y
                    self.z=z
                    self.cos_y = - self.cos_y #Отражение фотона от границы
                    self.l_remain += l_border #непройденная длина
            else:
                self.state = 'out side'
                return

        if params['g'] != 0: #случай с анизотропией
            cos_theta = 1/(2*params['g'])*(1+params['g']**2-((1-params['g']**2)/(1-params['g']+2*params['g']*random.random()))**2)
        else:
            cos_theta = random.uniform(-1,1)

        phi = random.uniform(0,math.pi*2)
        sin_phi = math.sin(phi)
        cos_phi = math.cos(phi)
        sin_theta = math.sin(math.acos(cos_theta))

        if abs(self.cos_z)>0.999: #Special case when direction is very close to vertical
            self.cos_x = sin_theta*cos_phi
            self.cos_y = sin_theta*sin_phi
            if self.cos_z>0:
                self.cos_z = cos_theta
            else:
                self.cos_z = - cos_theta
        else:
            cos_x = sin_theta/math.sqrt(1.0-self.cos_z**2)*(self.cos_x*self.cos_z*cos_phi-self.cos_y*sin_phi)+self.cos_x*cos_theta
            cos_y = sin_theta/math.sqrt(1.0-self.cos_z**2)*(self.cos_y*self.cos_z*cos_phi+self.cos_x*sin_phi)+self.cos_y*cos_theta
            cos_z = -sin_theta*cos_phi*math.sqrt(1.0-self.cos_z**2)+self.cos_z*cos_theta

            self.cos_x = cos_x
            self.cos_y = cos_y
            self.cos_z = cos_z

        if abs(self.cos_x) > 1 or abs(self.cos_y) > 1 or abs(self.cos_z) > 1:
            print('ERROR!!!!!!!')

def fresnel (cos_x, cos_y, cos_z, n_tissue, n_ext, exit_plane):
    """Returns refraction probability from boundary plane for unpolarized light"""

    n1 = n_tissue
    n2 = n_ext
    if exit_plane == 'z_min' or exit_plane == 'z_max':
        if math.sin(math.acos(cos_z))>(n2/n1): #total internal reflection
            return 1
        
        cos_i = abs(cos_z)
        sin_i = math.sqrt(1-cos_i**2)
        sin_t = sin_i*n1/n2
        cos_t = math.sqrt(1-sin_t**2)
    
    if exit_plane == 'x_min' or exit_plane == 'x_max':
        if math.sin(math.acos(cos_x))>(n2/n1): #total internal reflection
            return 1
        
        cos_i = abs(cos_x)
        sin_i = math.sqrt(1-cos_i**2)
        sin_t = sin_i*n1/n2
        cos_t = math.sqrt(1-sin_t**2)

    if exit_plane == 'y_min' or exit_plane == 'y_max':
        if math.sin(math.acos(cos_y))>(n2/n1): #total internal reflection
            return 1
        
        cos_i = abs(cos_y)
        sin_i = math.sqrt(1-cos_i**2)
        sin_t = sin_i*n1/n2
        cos_t = math.sqrt(1-sin_t**2)

    # Reflection coefficients for s and p polarizations
    R_s = ((n1*cos_i-n2*cos_t)/(n1*cos_i+n2*cos_t))**2
    R_p = ((n1*cos_t-n2*cos_i)/(n1*cos_t+n2*cos_i))**2

    # Reflection coefficient for unpolarized light
    R = (R_s + R_p)/2

    if R>1:
        print('Bad R = ', R)

    return R

def fiber_init(params, z, cos_x, cos_y, cos_z):
    """Returns photon with starting position according to vertically oriented collimated gaussian beam."""
    
    pht = Photon(
        (random.gauss(params['beam center x'], params['beam_width'])),
        (random.gauss(params['beam center y'], params['beam_width'])),
        z,
        cos_x,
        cos_y,
        cos_z
    )
    return pht

def distribution(photon, params):
    """Returns optical properties in photon coordinates. Currently NPs are evenly distributed in a sphere with center in x0,y0,z0 and radius."""
    
    #conversion of optical parameters to 1/mm
    abs_tissue = params['tissue_abs_coef']/10
    sca_tissue = params['tissue_sca_coef']/10
    abs_NP = params['np_abs_coef']/10
    sca_NP = params['np_sca_coef']/10
    
    if params['np distribution'] == 'sphere':
        if (((photon.x-params['np spherical x position'])**2+(photon.y-params['np spherical y position'])**2+(photon.z-params['np spherical z position'])**2)<params['np distribution radius']**2):
            abs_coeff = abs_NP + abs_tissue
            sca_coeff = sca_NP + sca_tissue
        else:
            abs_coeff = abs_tissue
            sca_coeff = sca_tissue
    
    elif params['np distribution'] == 'none':
        abs_coeff = abs_tissue
        sca_coeff = sca_tissue
    
    elif params['np distribution'] == 'layer':
        if photon.z > params['np layer depth']:
            abs_coeff = abs_NP + abs_tissue
            sca_coeff = sca_NP + sca_tissue
        else:
            abs_coeff = abs_tissue
            sca_coeff = sca_tissue
    
    return abs_coeff,sca_coeff

def nps_location():
    """Returns masked array with NPs spherical location"""

    arr = np.zeros((params['data_z_size'],params['data_y_size'],params['data_x_size']))

    if params['np distribution'] == 'sphere':
        #calculate coorinate of distribtuion center
        x_center = int(params['np spherical x position']/params['grid_step']*1000)
        y_center = int(params['np spherical y position']/params['grid_step']*1000)
        z_center = int(params['np spherical z position']/params['grid_step']*1000)

        #calculate distribution radius in data steps
        radius = int(params['np distribution radius']/params['grid_step']*1000)

        #generate open meshgrid centered at distribution center
        centered_axes = np.ogrid[
            -z_center:(params['data_z_size'] - z_center),
            -y_center:(params['data_y_size'] - y_center),
            -x_center:(params['data_x_size'] - x_center)
            ]

        #calculate distance from distribution center
        for axes in centered_axes:
            arr += (axes/radius)**2
    
    elif params['np distribution'] == 'layer':

        #calculate layer depth in data steps
        depth = int(params['np layer depth']/params['grid_step']*1000)

        arr[:depth,:,:] = 2
        

    #values within sphere will be <=1    
    return arr <=1

def run_photon(out_top_counter, out_side_counter, out_bottom_counter, absorbed_counter,
                data, lock, runs, params):
    """Function for calculation of photons destiny. Can be used for parallel run."""

    for j in tqdm.tqdm(range(runs)):
        pht = fiber_init(params, z = 0, cos_x = 0, cos_y = 0, cos_z = 1)
        abs_coeff,sca_coeff = distribution(pht, params)
        if pht.x < 0 or pht.x > params['model_area_x'] or pht.y < 0 or pht.y > params['model_area_y'] or pht.z < 0 or pht.z > params['model_area_z']:
            pht.state = 'out side'
        for i in range(params['max_iterations']): # Max kол-во шагов фотона 
            if pht.state == 'init': 
                abs_coeff,sca_coeff = distribution(pht, params)               
                pht.step(sca_coeff, abs_coeff, params)

            if pht.x < 0 or pht.x > params['model_area_x'] or pht.y < 0 or pht.y > params['model_area_y'] or pht.z < 0 or pht.z > params['model_area_z']:
                if pht.state == 'init':
                    print('!!! Error in photon step xyz', pht.x, pht.y, pht.z, 'state = ', pht.state)
                    print('Step number = ', pht.step_number, 'Refractions count', pht.refr_count, 'cos_x = ', pht.cos_x, 'cos_y = ', pht.cos_y, 'cos_z = ', pht.cos_z, 'l = ', pht.l)

            if pht.state == 'out top':
                with lock:
                    out_top_counter.value += 1
                break

            if pht.state == 'out down':
                with lock:
                    out_bottom_counter.value += 1
                break

            if pht.state == 'out side':
                with lock:
                    out_side_counter.value += 1
                break
            if pht.l_remain > pht.l:
                print('!!! Error in l_remain = ', pht.l_remain, 'l = ', pht.l, 'delta = ',pht.l_remain-pht.l, 'refr_count = ', pht.refr_count, 'step number = ', pht.step_number, 'state = ', pht.state)
                print('x = ', pht.x,'y = ', pht.y, 'z = ', pht.z)
                print('x_prev = ', pht.x_prev, 'y_prev = ', pht.y_prev, 'z_prev = ', pht.z_prev)
                print('cos_x = ', pht.cos_x, 'cos_y = ', pht.cos_y, 'cos_z = ', pht.cos_z)
            if random.random() < (abs_coeff/(abs_coeff + sca_coeff)*(pht.l-pht.l_remain)/pht.l):
                idx = int(pht.x/params['model_area_x'] * (params['data_x_size'] - 1))
                idy = int(pht.y/params['model_area_y'] * (params['data_y_size'] - 1))
                idz = int(pht.z/params['model_area_z'] * (params['data_z_size'] - 1))
                index = params['data_x_size'] * params['data_y_size'] * idz + params['data_x_size'] * idy + idx
                with lock:
                    absorbed_counter.value += 1
                    data[index] += 1 #счётчик фотонов в точке
                break
            if i == (params['max_iterations'] - 1):
                idx = int(pht.x/params['model_area_x'] * (params['data_x_size'] - 1))
                idy = int(pht.y/params['model_area_y'] * (params['data_y_size'] - 1))
                idz = int(pht.z/params['model_area_z'] * (params['data_z_size'] - 1))
                index = params['data_x_size'] * params['data_y_size'] * idz + params['data_x_size'] * idy + idx
                with lock:
                    absorbed_counter.value += 1
                    data[index] += 1 #счётчик фотонов в точке
                    
            if pht.l_remain != 0:
                pht.l_remain = 0

def run_simulation ():

    data_np = np.zeros(params['data_x_size']*params['data_y_size']*params['data_z_size'], np.int8)
    runs = 0
    top_photons = 0
    side_photons = 0
    bottom_photons = 0
    absorbed_photons = 0

    if __name__ == "__main__": #required for running parallel calculations

        #define shared variables and array for absorbed photon storage
        out_top_counter = mp.Value('i', 0)
        out_side_counter = mp.Value('i', 0)
        out_bottom_counter = mp.Value('i', 0)
        absorbed_counter = mp.Value('i', 0)
        lock = mp.Lock()

        data = mp.Array('i', params['data_x_size']*params['data_y_size']*params['data_z_size'])

        runs = int(params['photon_count']/mp.cpu_count()) #defines amount of iterations per CPU unit
        procs = [] #list for storing processes

        #running parallel calculation
        for i in range(mp.cpu_count()):
            p = mp.Process(target=run_photon, args=(out_top_counter, out_side_counter, out_bottom_counter, absorbed_counter,
                                                    data, lock, 
                                                    runs, params))
            p.start()
            procs.append(p)
        # wait for all processes to stop
        for p in procs: p.join()

        top_photons=out_top_counter.value
        side_photons=out_side_counter.value
        bottom_photons=out_bottom_counter.value
        absorbed_photons=absorbed_counter.value

        print('photons escaped top = ', top_photons/runs/mp.cpu_count()*100,'%,', 'photons out to sides =', side_photons/runs/mp.cpu_count()*100,'% ', 
                'photons out down = ', bottom_photons/runs/mp.cpu_count()*100,'% ', 'photons_absorbed = ', absorbed_photons/runs/mp.cpu_count()*100,'% ', 
                'total processed photons = ', top_photons+side_photons+bottom_photons+absorbed_photons,)
        data_np = np.ctypeslib.as_array(data.get_obj())
    return data_np.reshape(params['data_x_size'], params['data_y_size'], params['data_z_size'], order='F').T, runs, top_photons, side_photons, bottom_photons, absorbed_photons

if __name__ == "__main__":

    print('\nМодуль расчёта пространсвенного распределения поглощенного света.')
    print('Разработан при финансовой поддержке Российского Научного Фонда (грант № 20-72-00081).')
    print('Антон Попов: a.popov.fizteh@gmail.com')
    #Load simulation parameters from SimulationParams.ini
    load_config()


    if(params['simulation_mode'] =='single'):
        print('\nSimulation is starting with params:')
        print(
            'Photon beam is a collimated beam with 2D gaussian distibution with sigma =',
            params['beam_width'],'mm. The beam comes vertically from top side at X =', 
            params['beam center x'], ' Y =', params['beam center y'], '.'
            )
        if params['np distribution'] == 'none':
            print('NO NANOPARTICLES ARE PRESENT!')
        elif params['np distribution'] == 'sphere':
            print(
                'NPs are spherically distributed. Distribution radius =',
                params['np distribution radius'],
                'Position of the distribution center x =',
                params['np spherical x position'],
                'y =', params['np spherical y position'], 'z =', 
                params['np spherical z position']
                )
        elif params['np distribution'] == 'layer':
            print(
                'NPs are distributed in a layer, which is located ', 
                params['np layer depth'],
                'mm under the top surface.'
                )
        else:
            print('Error! Unknown NPs distribution type!')
            exit()
        
        data_np, runs, top_photons, side_photons, bottom_photons, absorbed_photons = run_simulation()

        filename = save_data(data_np)
        save_config(runs, top_photons, side_photons, bottom_photons, absorbed_photons, filename)

    if(params['simulation_mode']=='calculate absorbed'):
        print('\nIncrease in absorption of photons in the area with NPs will be calculated!')
        print('\nSimulation is starting with params:')
        print(
            'Photon beam is a collimated beam with 2D gaussian distibution with sigma =',
            params['beam_width'], 'mm. The beam comes vertically from top side at X =', 
            params['beam center x'], ' Y =', params['beam center y'], '.'
            )
        if params['np distribution'] == 'sphere':
            print(
                'NPs are spherically distributed. Distribution radius =',
                params['np distribution radius'],
                'Position of the distribution center x =',
                params['np spherical x position'],
                'y =', params['np spherical y position'], 
                'z will be varied in the simulation'
                )
        elif params['np distribution'] == 'layer':
            print(
                'NPs are distributed in a layer, which position will ',
                'be varied in the simulation.'
                )
        else:
            print('Error! Wrong NPs distribution type = ', params['np distribution'])
            exit()

        save_params = dict()

        #create iteration arrays
        iter_z = np.arange(params['iter_start_z'],params['iter_stop_z'],params['iter_step_z'])
        iter_tissue_abs = np.arange(params['iter_start_tissue_abs'],params['iter_stop_tissue_abs'],params['iter_step_tissue_abs'])
        iter_tissue_sca = np.arange(params['iter_start_tissue_sca'],params['iter_stop_tissue_sca'],params['iter_step_tissue_sca'])
        iter_nps_abs = np.arange(params['iter_start_NPs_abs'],params['iter_stop_NPs_abs'],params['iter_step_NPs_abs'])


        for k, t_sca in enumerate(iter_tissue_sca):
            params['tissue_sca_coef'] = t_sca
            print('Tissue scattering coef set to =', t_sca)
            for j, t_abs in enumerate(iter_tissue_abs):
                params['tissue_abs_coef'] = t_abs
                print('Tissue absorbtion coef set to =', t_abs)

                #temporary NPs optical params
                np_abs_coef = params['np_abs_coef']
                np_sca_coef = params['np_sca_coef']

                # no NPs for reference calculation
                params['np_abs_coef'] = 0
                params['np_sca_coef'] = 0

                print('\nCalculating reference photons distribution...')
                data_np_ref, runs_ref, top_photons_ref, side_photons_ref, bottom_photons_ref, absorbed_photons_ref = run_simulation()
                #filename_ref = save_data(data_np_ref)
                print('...Done!')

                # restore NPs optical params
                params['np_abs_coef'] = np_abs_coef
                params['np_sca_coef'] = np_sca_coef

                for l, np_abs in enumerate(iter_nps_abs):
                    params['np_abs_coef'] = np_abs
                    print('NPs absorbption coef set to =', np_abs)
                    for i, z in enumerate(iter_z):
                        if params['np distribution'] == 'sphere':
                            params['np spherical z position'] = z
                            print(
                                '\nCalculating actual photons distribution with center at Z=',
                                z, 'mm under top surface...'
                                )
                        elif params['np distribution'] == 'layer':
                            params['np layer depth'] = z
                            print(
                                '\nCalculating actual photons distribution located at Z>',
                                z, 'mm under top surface...'
                                )
                        #masked array with NPs location
                        nps_distribution = nps_location()

                        data_np, runs, top_photons, side_photons, bottom_photons, absorbed_photons = run_simulation()
                        #filename = save_data(data_np)
                        print('...Done!')
                        
                        target_photons_ref = np.sum(data_np_ref*nps_distribution)
                        target_photons = np.sum(data_np*nps_distribution)
                        print('Reference photons absorbed in NPs location =', target_photons_ref)
                        print('Photons absorbed in NPs location =', target_photons)

                        save_params['NPs_distrib_z_pos'] = z
                        save_params['tissue_abs'] = t_abs
                        save_params['tissue_sca'] = t_sca
                        save_params['NPs_abs'] = round(np_abs, 2)
                        save_params['NPs_distrib_radius'] = params.get('np distribution radius')
                        save_params['relative_photons_increase'] = round(target_photons/target_photons_ref,2)
                        save_params['fractional_photons_increase'] = round(
                            (target_photons - target_photons_ref)/absorbed_photons, 4
                            )
                        save_params['fraction_wo_escape'] = round(
                            (target_photons)/(absorbed_photons), 4
                            )
                        save_iter_log(save_params)

        save_config(runs, top_photons, side_photons, bottom_photons, absorbed_photons, filename)