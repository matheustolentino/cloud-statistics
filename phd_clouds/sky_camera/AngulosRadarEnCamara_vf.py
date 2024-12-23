# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 20:16:30 2024
Este script ejecuta un ejemplo para aplicar la función "dibujarAngulos" que es capaz de localizar y dibujar sobre una imagen de cielo dada, 
los puntos donde se encuentra una lista de ángulos dadas por el azimuth y el zenith. Ademas ejecuta un modelo de segmentación que identifica 
la clase de cada pixel (nube, sol, cielo despejado...) y con ello devuelve si en el ángulo introducido en la imagen de la cámara se ve
una nube o no.

Codigo de descarga de imagenes de la carpeta Cameras-HDR en SKYNAS a través de url. Se descarga en la misma carpeta donde este este código, con la misma estructura de carpetas que hay en la página web (ID_Camara/Año/Mes/Dia)
Modificar: USUARIO y CONTRASEÑA, fecha_camara, zenith y azimuth
 
@author: Roberto Román & Celia Herrero del Barrio
@email: robertor@goa.uva.es; celia@goa.uva.es
"""

import numpy as np
import os
import matplotlib.pyplot  as plt
import cv2
from keras.models import load_model
import requests
from datetime import datetime
import xml.etree.ElementTree as ET

def download_jpgs_from_folder(camera, date, base_url_list, base_url_download, username, password, ruta_base):

    session = requests.Session()
    session.auth = (username, password)

    folder_url_list = f"{base_url_list}{camera}/{date.strftime('%Y/%m/%d')}/"  #Url completa para buscar la carpeta del servidor con nº cámara/año/mes/dia
    folder_url_download = f"{base_url_download}{camera}/{date.strftime('%Y/%m/%d')}/"  #Url completa para descargar los ficheros del servidor con nº cámara/año/mes/dia
    local_folder_path = os.path.join(ruta_base, camera, date.strftime('%Y/%m/%d'))   #Directorio local donde guardar las descargas

    # Crear la carpeta local si no existe
    os.makedirs(local_folder_path, exist_ok=True)

    # Listar archivos en la carpeta del servidor usando WebDAV
    headers = {'Depth': '1'}   # significa que el servidor devolverá los archivos dentro de la carpeta solicitada, pero no buscará recursivamente en subcarpetas.
    try:
        #La solicitud PROPFIND es utilizada por WebDAV para buscar o listar recursos (como archivos) en un servidor. La respuesta a esta solicitud contendrá los archivos disponibles en la carpeta.
        response = session.request("PROPFIND", folder_url_list, headers=headers)
        
        # Verificar si la solicitud fue exitosa
        if response.status_code == 207:    #El código de estado HTTP 207 indica que la solicitud PROPFIND fue exitosa y que se han encontrado recursos en la carpeta.
            root = ET.fromstring(response.text)   #Convierte la respuesta XML del servidor en un árbol de elementos utilizando el módulo ElementTree de Python, que facilita el acceso y manipulación de datos XML.
            namespaces = {'d': 'DAV:'}   #Define un espacio de nombres para manejar las etiquetas en el XML

            for elem in root.findall('.//d:response', namespaces):
                href = elem.find('d:href', namespaces).text
                file_name = href.split('/')[-1] #La URL obtenida en el paso anterior se divide utilizando / como delimitador. Esto crea una lista de partes de la URL.[-1] toma el último elemento de la lista, que es el nombre del archivo

                # Descargar solo archivos .jpg
                if file_name.endswith('.jpg'):
                    try:
                        #Busca la hora en el nombre de los archivos
                        file_time = datetime.strptime(file_name.split('_')[-1].split('.')[0], "%H%M").time()
                    except ValueError:
                        print(f"No se pudo determinar la hora del archivo: {file_name}")
                        continue

                        # Verificar si el archivo está dentro del rango de tiempo especificado
                    if file_time == date.time():
                        # Crear URL de descarga directa
                        file_url = f"{folder_url_download}{file_name}"
                        local_file_path = os.path.join(local_folder_path, file_name)

                        # Descargar el archivo si no existe en el sistema local
                        if not os.path.exists(local_file_path):
                            file_response = session.get(file_url)
                            if file_response.status_code == 200:   #Verifica que la descarga haya sido exitosa (código de estado 200 significa éxito).
                                with open(local_file_path, "wb") as file:  #Abre un archivo en modo binario (porque es una imagen no texto) de escritura ("wb") en la ubicación local definida previamente.
                                    file.write(file_response.content)
                                print(f"{file_name} descargado en {local_file_path}")
                            else:
                                print(f"Error al descargar {file_name}: {file_response.status_code}")
                        else:
                            print(f"{file_name} ya existe en {local_file_path}, omitiendo descarga.")
        else:
            print(f"No se encontraron archivos en la carpeta {folder_url_list} (código de respuesta: {response.status_code})")
    except Exception as e:
        print(f"Error al acceder a {folder_url_list}: {e}")

    return local_file_path



#ESTA FUNCION DETERMINA LAS COORDENADAS (x,y) DEL PIXEL MÁS CERCANO A UN VALOR DE azimuth Y zenith A PARTIR DE LOS PARAMETROS
# pol_r, desfase, x0 e y0 OBTENIDOS EN LA CALIBRACION GEOMETRICA
def select_pixel_fast(azimut_input, zenit_input, pol_r, desfase, x0, y0):
    r = np.polyval(pol_r, zenit_input)
    rads = np.radians(azimut_input + desfase)
    y = np.rint(x0 - r * np.sin(rads)).astype(int)
    x = np.rint(y0 - r * np.cos(rads)).astype(int)
    pixel_coordinates = (y, x)

    return pixel_coordinates #LAS COORDENADAS x e y DEL PIXEL MAS CERCANO

#ESTA FUNCION EJECUTA PARA UNA IMAGEN "im_ori" EL MODELO DEL GOA QUE SEGMENTA LAS IMAGENES DE CIELO, DANDO PARA CADA PIXEL DE UNA IMAGEN LA CLASE
# A LA QUE CORRESPONDE:
        # 0 EN LOS PIXELES QUE SON EDIFICIOS O NO CORRESPODEN A CIELO ABIERTO
        # 1 EN LOS PIXELES QUE SON CIELO CLARO
        # 2 EN LOS PIXELES QUE SON SOL DESPEJADO
        # 3 EN LOS PIXELES QUE SON NUBES OPACAS
        # 4 EN LOS PIXELES QUE SON NUBES MÁS FINAS O TRASLUCIDAS O QUE ES DUDOSO SI ES UNA NUBE O NO
def mascara_nubesGOA(im_ori):

# SE CARGA EL MODELO QUE SE ENCUENTRA EN LA CARPETA "modelos"
    model_name='modelos//MultiUnetBACKUP256weight_10_100_1_100_25.h5'
    model=load_model(model_name,compile=False)
    
# SE CARGA LA MASCARA PARA QUITAR AQUELLOS PIXELES POR DEBAJO DEL HORIZONTE Y QUE VEN EDIFIFCIOS Y OTROS
    mascaraCamara='mascara//mascara_C011_Granada.png'
    maskCamara=cv2.imread(mascaraCamara)
    maskCamara=maskCamara[:,:,0]

    # DIMENSIONES A LAS QUE HAY QUE REDIMENSIONAR PARA QUE LA PROCESE EL MODELO
    IMG_WIDTH = 256
    IMG_HEIGHT = 256
    
    # SE CALCULAN LAS DIMENSIONES DE LA IMAGEN ORIGINAL
    size=im_ori.shape
    
    # SE REDIMENSIONA LA IMAGEN A LAS DIMENSIONES QUE ADMITE EL MODELO
    im_resized = cv2.resize(im_ori, (IMG_HEIGHT, IMG_WIDTH))
    
    # SE EJECUTA EL MODELO Y SE ALMACENA LA PREDICCION EN LA VARIABLE prediccion
    # LA VARIABLE prediccion MUESTRA UN:
        # 0 EN LOS PIXELES QUE SON MASCARA
        # 1 EN LOS PIXELES QUE SON CIELO CLARO
        # 2 EN LOS PIXELES QUE SON SOL DESPEJADO
        # 3 EN LOS PIXELES QUE SON NUBES OPACAS
        # 4 EN LOS PIXELES QUE SON NUBES MÁS FINAS O TRASLUCIDAS
    prediccion1=np.squeeze(model.predict(np.expand_dims(im_resized,0)))
    prediccion=np.argmax(prediccion1, axis=2)
    
    # LA MASCARA SE REDIMENSIONA A LAS DIMENSIONES ORIGINALES DE LA IMAGEN Y SE GUARDA EN LA VARIABLE pre_resized
    pre_resized = cv2.resize(prediccion,(size[1],size[0]),interpolation = cv2.INTER_NEAREST)
    pre_resized[maskCamara==0]=0

    # SE CREA UNA IMAGEN NUEVA (im_nueva) QUE VA A SER IGUAL QUE LA ORIGINAL PERO QUE SE VA A COLOREAR DE TAL MANERA:
        # NEGRO (0,0,0) EN LOS PIXELES QUE SON MASCARA (predicción=0)
        # AZUL (0, 0, 255) EN LOS PIXELES QUE SON CIELO CLARO (predicción=1)
        # AMARILLO (255, 255, 0) EN LOS PIXELES QUE SON SOL DESPEJADO (predicción=2)
        # BLANCO (255, 255, 255) EN LOS PIXELES QUE SON NUBES OPACAS (predicción=3)
        # GRIS (125, 125, 125) EN LOS PIXELES QUE SON NUBES MÁS FINAS O TRASLUCIDAS (predicción=4)
    im_nueva=np.copy(im_ori)
    #Selección de los canales rgb de la imagen copia
    a0=im_nueva[:,:,0]
    a1=im_nueva[:,:,1]
    a2=im_nueva[:,:,2]
    #Modifica los canales rgb en función del color asignado a la predicción para cada pixel
    a0[pre_resized==0]=0
    a1[pre_resized==0]=0
    a2[pre_resized==0]=0

    a0[pre_resized==1]=0
    a1[pre_resized==1]=0
    a2[pre_resized==1]=255


    a0[pre_resized==4]=125
    a1[pre_resized==4]=125
    a2[pre_resized==4]=125


    a0[pre_resized==3]=255
    a1[pre_resized==3]=255
    a2[pre_resized==3]=255

    a0[pre_resized==2]=255
    a1[pre_resized==2]=255
    a2[pre_resized==2]=0

    im_nueva[:,:,0]=a0
    im_nueva[:,:,1]=a1
    im_nueva[:,:,2]=a2
     
    return pre_resized,im_nueva #DEVUELVE LA MATRIZ EN LA QUE SE ENCUENTRA LA CLASE DE CADA PIXEL  Y LA IMAGEN MAS VISUAL DONDE SE VE LAS NUBES BLANCAS ETC.

#ESTA ES LA FUNCION PRINCIPAL, DIBUJA EN UNA IMAGEN (ruta=path_img) LOS PUNTOS EN LOS QUE APUNTA A LOS VALORES DE 
# ZENITH (zenith) Y AZIMUTH (azimuth). COMO ENTRADA NECESITA LOS VALORES DE LA CALIBRACION GEOEMTRICA DE LA CAMARA:desfase,pol_r,x0,y0    
def dibujarAngulos(path_img,zenith,azimuth,desfase,pol_r,x0,y0):
    # INICIALIZA ALGUNAS VARIABLES 
    coordenadas=[]
    nube=[]
    paso=12 #ESTE VALOR HACE LOS PUNTOS ROJOS/VERDES MAS GRANDES O MAS PEQUEÑOS EN LA IMAGEN A REPRESENTAR
    # LEE LA IMAGEN
    im_ori = cv2.cvtColor(cv2.imread(path_img), cv2.COLOR_BGR2RGB)
    # PASA EL MODELO PARA SEGMENTAR LA IMAGEN
    segmentacion,im_seg=mascara_nubesGOA(im_ori)

    # INICIALIZA LAS IMAGENES EN LAS QUE SE VAN A DIBUJAR LOS PUNTOS 
    im_puntos=np.copy(im_ori)
    im_seg_puntos=np.copy(im_seg)
    
    # BUCLE RECORRIENDO LA LISTA DE AZIMUTS
    for n in range(0,len(zenith)):
        # BUSCA EL PIXEL QUE CORRESPONDE A ESAS COORDENADAS
        pc=select_pixel_fast(azimuth[n], zenith[n], pol_r, desfase, x0, y0)
        # CALCULA LA CLASE A LA QUE CORRESPONDE ESE PIXEL (NUBE, SOL, ETC)
        clase=segmentacion[pc[0],pc[1]]
        
        # EN BASE A LA CLASE DEL PIXEL SE ALAMACENARA LA INFORMACION
        
        if clase==3: #EN EL CASO DE QUE SEA NUBE OPACA. EL ELEMENTO n DEL VECTOR NUBE ES 1 Y SE PINTA DE COLOR ROJO
            nube.append(1)
            color=[255,0,0]
        elif clase==4: #EN EL CASO DE QUE SEA NUBE MAS FINA, TRASLUCIDA O DUDOSA. EL ELEMENTO n DEL VECTOR NUBE ES 0.5 Y SE PINTA DE COLOR ROSA
            nube.append(0.5)
            color=[255,0,255]
        else: #SI NO HAY NUBE EL ELEMENTO n DEL VECTOR NUBE ES 0 Y SE PINTA DE COLOR VERDE
            nube.append(0)
            color=[0,255,0]
            # SE DIBUJAN LOS PUNTOS EN LA IMAGEN UNO A UNO 
        im_puntos[pc[0]-paso:pc[0]+paso,pc[1]-paso:pc[1]+paso,:]=color
        im_seg_puntos[pc[0]-paso:pc[0]+paso,pc[1]-paso:pc[1]+paso,:]=color

        # SE ALMACENAN LAS COORDENADAS EN LA VARIABLE coordenadas
        coordenadas.append(pc)
        
    # DIBUJA LOS RESULTADOS EN 4 SUBPLOTS
    plt.figure(1)
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    plt.subplot(2, 2, 1)    
    plt.imshow(im_ori)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2, 2, 2)    
    plt.imshow(im_puntos)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2, 2, 3)    
    plt.imshow(im_seg)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2, 2, 4)    
    plt.imshow(im_seg_puntos)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    
    return pc,nube #DEVUELVE EL VECTOR CON LAS POSICIONES DE LOS PIXELES (pc) Y EL VECTOR QUE DICE SI ES NUBE (1) O NO (0) O DUDOSO/FINA (0.5)


#COMIENZA EL SCRIPT PARA EJECUTAR LA FUNCION CON UN EJEMPLO
# Credenciales: CAMBIAR USUARIO Y CONTRASEÑA POR UNOS REALES
username, password = "gfat", "forevergfat"  # Credenciales de acceso al servidor
# URL de descarga del archivo
base_url_list = 'https://skynas.opt.cie.uva.es/remote.php/dav/files/'+username+'/'
base_url_download = "https://skynas.opt.cie.uva.es/remote.php/webdav/"

#SELECCIONAR FECHA DE INTERÉS
fecha_camara = {"fecha": "20241104 14:00"}  #Instante del que descargar la imagen

camera = "C011"  #ID de la cámara de GRANADA
# ESTOS VALORES SON DE LA CALIBRACION GEOMETRICA
desfase=-148.960468
x0=989.644
y0=1007.887
pol_r=[-0.011975053897187376, 11.883106086004899, -9.394133194587406]

# Ruta para las carpetas de descarga locales
ruta_base = 'imagenes'

date = datetime.strptime(fecha_camara['fecha'], '%Y%m%d %H:%M')

path_img=download_jpgs_from_folder(camera, date, base_url_list, base_url_download, username, password, ruta_base)

# SE ELIGEN LOS PUNTOS DE ZENITH Y AZIMUTH QUE SE QUIEREN
zenith=np.linspace(0, 70, 100)
azimuth=np.ones(100)*0
# zenith=[30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30]
# azimuth=[5,10,15,30,50,70,90,110,130,150,-5,-10,-15,-30,-50,-70,-90,-110,-130,-150]

# SE LLAMA A LA FUNCION PRINCIPAL QUE ES LA DE dibujarAngulos Y QUE DEVUELVE LAS COORDENADAS DE LOS PIXELES CORRESPONDIENTES A LOS ANGULOS 
# INTRODUCIDOS. ADEMAS LA VARIABLE nube INDICA SI EN LA CAMARA SE VE NUBE O NO O NO ESTA MUY CLARO. TAMBIEN ENTRAN COMO ENTRADA LOS PARAMETROS DE LA CALIBRACION GEOMETRICA
pixels,nube=dibujarAngulos(path_img,zenith,azimuth,desfase,pol_r,x0,y0)

