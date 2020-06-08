# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 20:21:48 2020

@author: Personal
"""


import numpy as np
# Funciones para aplicar t_haar y denoising
    
def descomponer_senal(senal,nivel_actual,nivel_final,senal_trans):
#A esta funcion se le ingresa un nivel inicial que varía (nivel actual), además un nivel final que se calcula con formula
#senal_trans corresponde a la trasnformada haar. La funcion realiza la descomposición de la señal 
    
    wavelet = [-1/np.sqrt(2) , 1/np.sqrt(2)];
    scale = [1/np.sqrt(2) , 1/np.sqrt(2)];
    
    if (nivel_actual <= nivel_final):
        if (senal.shape[0] %2) != 0:
            senal = np.append(senal, 0)
   
    scale_senal = np.convolve(senal, scale, 'full');
    aprox_senal = scale_senal[1::2];


    wavelet_senal = np.convolve(senal, wavelet, 'full');
    detail_senal = wavelet_senal[1::2];


    senal_trans.append(detail_senal);
    
    if (nivel_actual < nivel_final):
       return descomponer_senal(aprox_senal,nivel_actual+1,nivel_final,senal_trans);
    senal_trans.append(aprox_senal);
    
    
    return senal_trans
 


#La función umbral_señal recibe los valores de los indices de tres comboBox diferentes en donde se
#escogen el nivel de filtrado(one, single, multiple), la forma de filtrado(duro o suave) y el tipo de umbral(universal,minimax,sure)
def umbrales_senal(senal_trans,nivel,forma,umbral):

    #primero el tipo de umbral
    n = len(senal_trans);
    if (umbral == 1):
        lamda = np.sqrt(2*np.log(n));
    if (umbral == 2): 
        lamda = 0.3936 + 0.1829*(np.log(n)/np.log(2));
    if (umbral == 3): 
        sx2 = []
        risks = []
        for i in range(len(senal_trans)):
            sx2 = np.append(sx2,senal_trans[i])
        sx2 = np.power(np.sort(np.abs(sx2)),2);
        risks = n-(2*np.arange(1,n+1))+(np.cumsum(sx2[0:n])+np.multiply(np.arange(n,0,-1),sx2[0:n]))/n;
        best = int(round(np.min(risks)));
        lamda = np.sqrt(sx2[best]);
    
    #determinar la ponderacion (nivel)
    sigma = np.zeros(len(senal_trans));
    if (nivel == 1):
        sigma[:] = 1;
    if (nivel == 2): 
        sigma_detalle = (np.median(np.absolute(senal_trans[0])))/0.6745;
        sigma[:] = sigma_detalle;
    if (nivel == 3): 
        for i in range(len(senal_trans)):
            sigma_detalle_all = (np.median(np.absolute(senal_trans[i])))/0.6745;
            sigma[:] = sigma_detalle_all;
                
    umbral = sigma*lamda; 
    
    #duro o suave
    if (forma == 1):          
        for i in range(len(senal_trans)):
            for j in range(len(senal_trans[i])):
                if np.abs(senal_trans[i][j]) < umbral[i]:
                    senal_trans[i][j] = 0
                else:
                    senal_trans[i][j] = senal_trans[i][j]
        
    if (forma == 2):
        
        for i in range(len(senal_trans)):
            for j in range(len(senal_trans[i])):
                if np.abs(senal_trans[i][j]) < umbral[i]:
                    senal_trans[i][j] = 0
                else:
                    signo = np.sign(senal_trans[i][j])
                    resta = np.abs(senal_trans[i][j]) - umbral[i]
                    senal_trans[i][j] = signo*resta
    return senal_trans

#La función recomponer señal aplica la transformada inversa de haar, se la aplica a la señal que entrega la función
#umbrales señal 
def recomponer_senal(senal_trans,nivel_actual,nivel_final,senal):
    wavelet_inv = [1/np.sqrt(2) , -1/np.sqrt(2)];
    scale_inv = [1/np.sqrt(2) , 1/np.sqrt(2)];

    longitud_senal = len(senal_trans)
    detalle = senal_trans[longitud_senal - 1 - nivel_actual]
    
    if (nivel_actual <= nivel_final):
        if (nivel_actual==1):
            longitud_aprox = len(senal_trans[len(senal_trans)-1])
            aprox_inv = np.zeros(2*longitud_aprox)
            aprox_inv[0::2] = senal_trans[longitud_senal-1]                
    
        else:
            if (len(senal) > len(detalle)):
                senal = senal[0:len(detalle)]
            longitud_aprox = len(senal)
            aprox_inv = np.zeros(2*longitud_aprox)
            aprox_inv[0::2] = senal
        
        aprox_senal = np.convolve(aprox_inv, scale_inv, 'full')
        
        detalle_inv = np.zeros(2*longitud_aprox)
        detalle_inv[0::2] = detalle
        
        detalle_senal = np.convolve(detalle_inv, wavelet_inv, 'full')
        
        senal = aprox_senal + detalle_senal
     
        return recomponer_senal(senal_trans, nivel_actual+1, nivel_final, senal)
    return senal

#La función filtrar señal llama las funciones anteriormente creadas, de manera que 
#integra la obtencion de los umbrales, la descomposicion, la filtracion y la recomposicion
#de la señal inicial, entregando la señal filtrada
def filtrar_senal(senal,nivel,forma,umbral):
    nivel_final = np.floor(np.log2(senal.shape[0]/2) -1)
    senal_trans = descomponer_senal(senal,1,nivel_final,[])
    umbral_final = umbrales_senal(senal_trans,nivel,forma,umbral)
    recomponer = recomponer_senal(umbral_final,1,nivel_final,senal)
    return recomponer
    

    
    

    
       
   
    