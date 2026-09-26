# KinderNet

KinderNet es una aplicación web para enseñar y aprender sobre redes neuronales, enfocado especialmente a alumnas y alumnos de primaria y secundaria. El objetivo es que puedan entrenar su propia red neuronal para reconocer cosas que se presenten a la camara web, de una forma interactiva. Se puede jugar y experimentar con el proceso de entrenamiento y prueba de redes neuronales, cambiando el tamaño de la red, cantidad y tipos de cosas a clasificar. La red es sencilla pero puede aprender a distinguir cosas con muy pocos ejemplos. 

La aplicación corre completamente en el navegador, es decir que no hay transferencia de datos a internet (tanto las imágenes como los modelos entrenados quedan en la PC localmente y se borran al cerrar o reiniciar la aplicación) 

La aplicación está funcionando en [este link](https://sinc-lab.github.io/kindernet/). La versión de desarrollo (para pruebas) está en [este otro link](https://sinc-lab.github.io/kindernet/dev/).

## Para instalar localmente

1. Instalar [node.js](https://nodejs.org/en/download/).

2. Moverse a la carpeta raiz kindernet-page e instalar los paquetes necesarios con el siguiente comando. Estos paquetes se instalarán localmente
    ``` 
    npm install 
    ```
3. Correr desde la carpeta raiz (en windows puede ser necesario configurar permisos de ejecución de scripts)
    ```
    npm run start
    ```

La aplicación debería abrirse automáticamente en [http://localhost:3000](http://localhost:3000). Si el puerto 3000 esta ocupado, en esta consola se sugerirá otro puerto.

    

L. Bugnon, D. Milone, J. Raad, G. Stegmayer, C. Yones.   
www.sinc.unl.edu.ar