# -*- coding: utf-8 -*-

#%% Librerias

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
from linearFIR import filter_design
from Filtro_wavelet import filtrar_senal
from os import listdir
from os.path import isfile, join
import math
import librosa
import librosa.display
from IPython import get_ipython

#%% Funciones


"""
La función cargar_filtrar se encarga de cargar el archivo de audio 
y a la vez aplicar los filtros necesarios para obtenerlas frecuencias 
entre 100 y 2000 Hz justo antes de que pase al filtrado por wavelet
"""
def cargar_filtrar(filename):
    
    file, sr = librosa.load(filename);
    
    order, lowpass = filter_design(sr, locutoff = 0, hicutoff = 2000, revfilt = 0);
    #frecuencia de muestreo sr
    order, highpass = filter_design(sr, locutoff = 100, hicutoff = 0, revfilt = 1);
    
    file_paltas = signal.filtfilt(highpass, 1, file);
    
    file_pbajas = signal.filtfilt(lowpass, 1,file_paltas);
    
    file_filt = np.asfortranarray(file_pbajas);
    
    return file_filt, sr


"""
la funcion obtener_ciclos recibe los archivos de audio, de texto y la frecuencia
de muestreo y retorna los ciclos respiratorios.  
"""
def obtener_ciclos(file_audio, file_texto, sr):
    
    file_txt = np.loadtxt(file_texto); #carga el archivo txt
    
    y, x = file_txt.shape;
    
    ciclos_resp = []; #Esta lista se usará para guardar los datos del txt
    
    for i in range(y):
        
        first = math.floor(file_txt[i, 0]*sr);
        
        last = math.ceil(file_txt[i, 1]*sr);
        
        ciclo=file_audio[first:last];
        
        ciclos_resp.append([ciclo, file_txt[i, 2], file_txt[i, 3]]);
        
    return ciclos_resp

"""
La funcion calcular recibe el ciclo y entrega el rango del ciclo, la varianza,
la suma de los promedios y el promedio del espectro. lo que hemos llamado
suma de promedios, hace referencia al promedio de los promedio móviles,
lo hemos realizado con el promedio fino con base en la asesosría del viernes
5 de junio. 
"""
def calcular(ciclo):
    
    muestras = 800;
    
    corrimiento = 100;
    
    rango_ciclo = np.abs(np.max(ciclo) - np.min(ciclo));
    
    varianza = np.var(ciclo);
    
    ubicacion = np.arange(0, len(ciclo)-muestras, corrimiento);
    
    datos = [];
    
    for i in ubicacion:
        datos.append(np.mean([ciclo[i:i + muestras]]));
        
    datos.append(np.mean([ciclo[ubicacion[len(ubicacion)-1]:]]));
    
    suma_prom = np.mean(datos);
    
    f, Pxx = signal.periodogram(ciclo);
    
    espectro = np.mean(Pxx);
    
    return rango_ciclo, varianza, suma_prom, espectro
   
#%% Principal

#Se cargan los archivos 
Directorio = (r'C:\Users\Personal\Desktop\2019-2\Señales\Proyecto 3\Respiratory_Sound_Database\audio_and_txt_files')
#Se discriminan los archivos de audio y lo de texto
audio_files = [file for file in listdir(Directorio) if file.endswith(".wav") if isfile(join(Directorio, file))]
text_files = [file for file in listdir(Directorio) if file.endswith(".txt") if isfile(join(Directorio, file))]

print('--------------------------------------------------------------------');
print('Comenzando análisis de datos...');
print();
print();
print('Analizando... Esto puede tardar unos minutos...');
print();
print();


for i in range(len(audio_files)):
    print('Archivos restantes: '+str(920-i)+' - Archivo actual:'+audio_files[i]);
    print();
    
    audio_filtrado, sr = cargar_filtrar(audio_files[i]);
    
    """
    Despues de realizar pruebas aparte se determino que la mejor configuración
    para el wavelet en este caso sería 3,2,2 que refiere a: 
    poderación: Multilevel, tipo de filtrado: Soft y umbral: Minimax 
    """
    wavelet = filtrar_senal(audio_filtrado,3,2,2);
    
    audio_final = audio_filtrado - wavelet[0:len(audio_filtrado)];
    
    datos = obtener_ciclos(wavelet, text_files[i], sr);
    
    #Se crea el Data Frame
    
    data_frame = pd.DataFrame(columns=['Ciclo Respiratorio','Sin Patologia', 'Crepitancia','Sibilancia','Varianza', 'Rango', 'Suma de Promedios', 'Promedio de Espectros'])
    
    #Se rellenan las columnas
    for j in range(len(datos)):
        
        ciclo = float(j + 1);
        
        crepitancias = datos[j][1];
        
        sibilancias = datos[j][2];
        
        if crepitancias == 0 and sibilancias == 0:
            normal = 1;
        else: 
            normal = 0
        
        rango, varianza, promedios, espectro = calcular(datos[j][0]);
        
        data = pd.DataFrame({'Ciclo Respiratorio':[ciclo], 
                           'Sin Patologia':[normal],
                           'Crepitancia':[crepitancias], 
                           'Sibilancia':[sibilancias], 
                           'Rango':[rango], 
                           'Varianza':[varianza], 
                           'Suma de Promedios':[promedios], 
                           'Promedio de Espectros':[espectro]})
        
        data_frame = pd.concat([data_frame, data], ignore_index=True);
        
print('Finalizando...');
Frames=[];
Frames.append(data_frame);
data_final = pd.concat(Frames, keys=audio_files);
data_final

data_final.to_csv('datos_estudio.csv');
print();
print('Fin')

#%% PRUEBAS INDIVIDUALES

# get_ipython().run_line_magic('matplotlib', 'qt')
# plt.figure();
# plt.subplot(3,1,1);
# librosa.display.waveplot(audio_filtrado,sr=sr);
# plt.subplot(3,1,2); 
# librosa.display.waveplot(wavelet,sr=sr);
# plt.subplot(3,1,3);
# librosa.display.waveplot(audio_final,sr=sr);

#librosa.output.write_wav('audio_filtrado_24.wav', Audio_filtrado, sr)
#librosa.output.write_wav('prueba_920_322.wav', Audio_final, sr);


#plt.plot(Audio_filtrado[1],wavelet[1]); 


