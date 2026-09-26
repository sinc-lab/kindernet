import React from 'react';
import './App.css';
import Webcam from "react-webcam";
import {Link, Toolbar, Dialog, DialogTitle, IconButton, Typography, Button,  Card, Box, AppBar, TextField,
    Grid, CssBaseline, Switch,  FormControlLabel, FormLabel, Radio, RadioGroup, DialogContent, List, ListItem, ListItemText, ListItemButton} from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import DeleteIcon from '@mui/icons-material/Delete';
import SaveIcon from '@mui/icons-material/Save';
import FolderOpenIcon from '@mui/icons-material/FolderOpen';
import CategoryList from "./CategoryList"
import ImagesList from "./ImagesList"
import { Network } from './NeuralNetwork';
import {height, unit_sep, use_timer, base_timer} from './constants';
import Avatar from '@mui/material/Avatar';
import logo from "./ia.png"
import sinclogo from "./sinc-logo.png"
import * as tf from '@tensorflow/tfjs'
import * as mobilenet from '@tensorflow-models/mobilenet';


var TEST_SAMPLES = 2
var MIN_SAMPLES = 5

// event listener
class EventListener extends React.Component{
    componentDidMount() {
        window.addEventListener("keyup",this.props.onKeyUp)
    }
    componentWillUnmount() {
        window.removeEventListener("keyup",this.props.onKeyUp)
    }
    render(){
        return null;
    }
}

// Kindernet ==========================================
class KinderNet extends React.Component{
    constructor(props){
        super(props);
        this.state={
            img_size: 64,
            is_training: false,
            is_adding_pic: false,
            category: -1,
            classifying: false,
            net_size: 0, // mayor valor, mas compleja la red
            category_names: ["Cosa 1", "Cosa 2"],
            images: [Array(2)],
            accuracy: [0, 0],
            scores: [0, 0],
            n_samples : [0,0], // n_samples  de la clase actual durante el entrenamiento
            output_on: -1,
            listen_keys: false,
            output_ypos: [0, 0],
            help: false,
            show_images: false,
            config: false,
            about: false,
            save_state: false,
            load_state: false,
            saved_states: [],
            save_name: ""
        };
        this.response = null
        this.captureGlobalEvent = this.captureGlobalEvent.bind(this);
        this.handleTransitionEnd = this.handleTransitionEnd.bind(this);
        this.handleTimerOut = this.handleTimerOut.bind(this);
        this.captureCategoryNames = this.captureCategoryNames.bind(this);
        
        this.handleTrain = this.handleTrain.bind(this);
        this.handleClassifierChange = this.handleClassifierChange.bind(this);
        this.handleAddCategory = this.handleAddCategory.bind(this);
        this.handleRemoveCategory = this.handleRemoveCategory.bind(this);
        this.handleKeyListen = this.handleKeyListen.bind(this);
        this.handleDeleteImage = this.handleDeleteImage.bind(this);
        this.handleSaveState = this.handleSaveState.bind(this);
        this.handleLoadState = this.handleLoadState.bind(this);
        this.handleDeleteSavedState = this.handleDeleteSavedState.bind(this);
        
        this.classifyPic = this.classifyPic.bind(this);
    }
    setRef = webcam => {
        window.webcam = webcam;
    };

  
    defineNet(net_size, nclasses){
        
        let classifier = tf.sequential();
        if(net_size === 0){
            classifier.add(tf.layers.conv2d({filters: 16, kernelSize: 3, activation: 'relu', inputShape: [this.state.img_size, this.state.img_size, 3]}))
            classifier.add(tf.layers.batchNormalization())
            classifier.add(tf.layers.maxPooling2d({poolSize: 2}))
            classifier.add(tf.layers.conv2d({filters: 32, kernelSize: 5, activation: 'relu'}))
            classifier.add(tf.layers.batchNormalization())
            classifier.add(tf.layers.maxPooling2d({poolSize: 2}))
            classifier.add(tf.layers.flatten())
            classifier.add(tf.layers.dense({units: nclasses, activation: 'softmax'}))
            classifier.compile({loss: 'categoricalCrossentropy', optimizer: 'adam', metrics: ['accuracy']});
        }
        if(net_size === 2){
            classifier.add(tf.layers.dense({units: nclasses, activation: 'softmax', inputShape: 1024}))
            classifier.compile({loss: 'categoricalCrossentropy', optimizer: 'sgd', metrics: ['accuracy']});
        }
        
        return classifier
    }

    resetValues(){
        window.train_tensors = tf.zeros([0, this.state.img_size, this.state.img_size, 3])
        window.train_features = tf.zeros([0, 1024])
        window.train_labels = tf.zeros([0, 2])
        window.test_tensors = tf.zeros([0, this.state.img_size, this.state.img_size, 3])
        window.test_features = tf.zeros([0, 1024])
        window.test_labels = tf.zeros([0, 2])
        // Inicializa el timer
        if(use_timer)
            setTimeout(this.handleTimerOut, base_timer)
        window.classifier = this.defineNet(0, 2)
        this.setState({net_size: 0, category: -1, output_on: -1, classifying: false, accuracy: [0, 0],
        images: [[], []], n_samples: [0,0], n_outputs: [0, 0], category_names: ["Cosa 1", "Cosa 2"]})
        
    }

    componentDidMount() {
        // create async function to load model
        async function loadModel() {
            return await mobilenet.load();
        }
        // load model
        loadModel().then((mobilenet) => {window.mobilenet=mobilenet; this.setState({listen_keys: true})});

        this.resetValues()
        this.loadSavedStatesList()
    }

    // Save/Load State functionality
    async handleSaveState() {
        if (!window.classifier || this.state.n_samples.reduce((a, b) => a + b, 0) === 0) {
            alert("No hay modelo entrenado o imágenes para guardar")
            return
        }

        const stateName = this.state.save_name.trim() || `Estado ${new Date().toLocaleString()}`
        
        try {
            // Save model to IndexedDB
            const modelKey = `kindernet_model_${Date.now()}`
            await window.classifier.save(`indexeddb://${modelKey}`)
            
            // Convert tensors to arrays for serialization
            const trainTensorsData = window.train_tensors ? await window.train_tensors.array() : null
            const trainFeaturesData = window.train_features ? await window.train_features.array() : null
            const trainLabelsData = window.train_labels ? await window.train_labels.array() : null
            const testTensorsData = window.test_tensors ? await window.test_tensors.array() : null
            const testFeaturesData = window.test_features ? await window.test_features.array() : null
            const testLabelsData = window.test_labels ? await window.test_labels.array() : null

            // Get tensor shapes
            const tensorShapes = {
                train_tensors: window.train_tensors ? window.train_tensors.shape : null,
                train_features: window.train_features ? window.train_features.shape : null,
                train_labels: window.train_labels ? window.train_labels.shape : null,
                test_tensors: window.test_tensors ? window.test_tensors.shape : null,
                test_features: window.test_features ? window.test_features.shape : null,
                test_labels: window.test_labels ? window.test_labels.shape : null
            }

            // Prepare state data
            const stateData = {
                name: stateName,
                modelKey: modelKey,
                category_names: this.state.category_names,
                images: this.state.images,
                n_samples: this.state.n_samples,
                accuracy: this.state.accuracy,
                net_size: this.state.net_size,
                img_size: this.state.img_size,
                tensorShapes: tensorShapes,
                trainTensorsData: trainTensorsData,
                trainFeaturesData: trainFeaturesData,
                trainLabelsData: trainLabelsData,
                testTensorsData: testTensorsData,
                testFeaturesData: testFeaturesData,
                testLabelsData: testLabelsData,
                savedAt: new Date().toISOString()
            }

            // Save to localStorage
            const savedStates = JSON.parse(localStorage.getItem('kindernet_saved_states') || '[]')
            savedStates.push({
                name: stateName,
                key: modelKey,
                data: stateData,
                savedAt: stateData.savedAt
            })
            localStorage.setItem('kindernet_saved_states', JSON.stringify(savedStates))

            this.loadSavedStatesList()
            this.setState({ save_state: false, save_name: "" })
            alert(`Estado "${stateName}" guardado exitosamente`)
        } catch (error) {
            console.error("Error saving state:", error)
            alert("Error al guardar el estado: " + error.message)
        }
    }

    async handleLoadState(stateKey, stateData) {
        try {
            // Stop any ongoing operations
            this.setState({ 
                is_training: false, 
                classifying: false, 
                listen_keys: false,
                load_state: false 
            })

            // Dispose old tensors to free memory
            if (window.train_tensors) window.train_tensors.dispose()
            if (window.train_features) window.train_features.dispose()
            if (window.train_labels) window.train_labels.dispose()
            if (window.test_tensors) window.test_tensors.dispose()
            if (window.test_features) window.test_features.dispose()
            if (window.test_labels) window.test_labels.dispose()
            if (window.classifier) window.classifier.dispose()

            // Load model from IndexedDB
            const modelKey = stateData.modelKey || stateKey
            window.classifier = await tf.loadLayersModel(`indexeddb://${modelKey}`)
            
            // Recompile the model (TensorFlow.js doesn't always preserve compilation state)
            // Use the same compilation settings as when the model was created
            if (stateData.net_size === 0) {
                window.classifier.compile({
                    loss: 'categoricalCrossentropy', 
                    optimizer: 'adam', 
                    metrics: ['accuracy']
                })
            } else if (stateData.net_size === 2) {
                window.classifier.compile({
                    loss: 'categoricalCrossentropy', 
                    optimizer: 'sgd', 
                    metrics: ['accuracy']
                })
            } else {
                // Default compilation for other net sizes
                window.classifier.compile({
                    loss: 'categoricalCrossentropy', 
                    optimizer: 'adam', 
                    metrics: ['accuracy']
                })
            }
            
            // Verify model is compiled
            if (!window.classifier.optimizer) {
                throw new Error("Model failed to compile after loading")
            }

            // Restore tensors from arrays
            if (stateData.trainTensorsData && stateData.tensorShapes && stateData.tensorShapes.train_tensors) {
                window.train_tensors = tf.tensor(stateData.trainTensorsData, stateData.tensorShapes.train_tensors)
            } else {
                window.train_tensors = tf.zeros([0, stateData.img_size, stateData.img_size, 3])
            }

            if (stateData.trainFeaturesData && stateData.tensorShapes && stateData.tensorShapes.train_features) {
                window.train_features = tf.tensor(stateData.trainFeaturesData, stateData.tensorShapes.train_features)
            } else {
                window.train_features = tf.zeros([0, 1024])
            }

            if (stateData.trainLabelsData && stateData.tensorShapes && stateData.tensorShapes.train_labels) {
                window.train_labels = tf.tensor(stateData.trainLabelsData, stateData.tensorShapes.train_labels)
            } else {
                window.train_labels = tf.zeros([0, stateData.category_names.length])
            }

            if (stateData.testTensorsData && stateData.tensorShapes && stateData.tensorShapes.test_tensors) {
                window.test_tensors = tf.tensor(stateData.testTensorsData, stateData.tensorShapes.test_tensors)
            } else {
                window.test_tensors = tf.zeros([0, stateData.img_size, stateData.img_size, 3])
            }

            if (stateData.testFeaturesData && stateData.tensorShapes && stateData.tensorShapes.test_features) {
                window.test_features = tf.tensor(stateData.testFeaturesData, stateData.tensorShapes.test_features)
            } else {
                window.test_features = tf.zeros([0, 1024])
            }

            if (stateData.testLabelsData && stateData.tensorShapes && stateData.tensorShapes.test_labels) {
                window.test_labels = tf.tensor(stateData.testLabelsData, stateData.tensorShapes.test_labels)
            } else {
                window.test_labels = tf.zeros([0, stateData.category_names.length])
            }

            // Restore state
            this.setState({
                category_names: stateData.category_names,
                images: stateData.images,
                n_samples: stateData.n_samples,
                accuracy: stateData.accuracy || Array(stateData.category_names.length).fill(0),
                net_size: stateData.net_size,
                img_size: stateData.img_size,
                category: -1,
                output_on: -1,
                classifying: false,
                listen_keys: true
            })

            alert(`Estado "${stateData.name || 'Sin nombre'}" cargado exitosamente`)
        } catch (error) {
            console.error("Error loading state:", error)
            alert("Error al cargar el estado: " + error.message)
            this.setState({ listen_keys: true })
        }
    }

    handleDeleteSavedState(stateKey, event) {
        event.stopPropagation()
        if (!window.confirm("¿Estás seguro de que quieres eliminar este estado guardado?")) {
            return
        }

        try {
            // Remove from localStorage
            const savedStates = JSON.parse(localStorage.getItem('kindernet_saved_states') || '[]')
            const filteredStates = savedStates.filter(state => state.key !== stateKey)
            localStorage.setItem('kindernet_saved_states', JSON.stringify(filteredStates))

            // Try to delete model from IndexedDB (best effort)
            // Note: IndexedDB cleanup might need manual intervention in browser dev tools
            this.loadSavedStatesList()
        } catch (error) {
            console.error("Error deleting state:", error)
            alert("Error al eliminar el estado: " + error.message)
        }
    }

    loadSavedStatesList() {
        try {
            const savedStates = JSON.parse(localStorage.getItem('kindernet_saved_states') || '[]')
            this.setState({ saved_states: savedStates })
        } catch (error) {
            console.error("Error loading saved states list:", error)
            this.setState({ saved_states: [] })
        }
    }

    handleClassifierChange(net_size){
        window.classifier = this.defineNet(net_size, this.state.category_names.length)
        this.setState({net_size: net_size, category: -1, output_on: -1, classifying: false, accuracy: Array(this.state.category_names.length).fill(0)})
        
        return
    }
    handleAddCategory(){
        let category_names = this.state.category_names
        category_names.push("Cosa " + (category_names.length + 1))
        let n_samples = this.state.n_samples
        n_samples.push(0)
        let zeros = Array(category_names.length).fill(0)
        let images = this.state.images
        images.push([])
        
        // add column to labels tensor 
        if(window.train_labels){
            let new_column = tf.zeros([window.train_labels.shape[0], 1])
            window.train_labels = tf.concat([window.train_labels, new_column], 1)
        }

        if(window.test_labels){
            let new_column = tf.zeros([window.test_labels.shape[0], 1])
            window.test_labels = tf.concat([window.test_labels, new_column], 1)
        }

        window.classifier = this.defineNet(this.state.net_size, category_names.length)
        this.setState({category_names, n_samples, category: -1, output_on: -1, 
            classifying: false, accuracy: zeros, images})
        return
    }
    handleRemoveCategory(category){
        let category_names = this.state.category_names
        let n_samples = this.state.n_samples
        let images = this.state.images        
        category_names.splice(category, 1)
        images.splice(category, 1)

        n_samples.splice(category, 1)
        
        // remove files and columns of category
        if(window.train_labels){
            let ind = []
            let labels_data = window.train_labels.arraySync()
            for(let i = 0; i < window.train_labels.shape[0]; i++)
                if(labels_data[i][category] === 0)
                ind.push(i)
            ind = tf.tensor1d(ind, 'int32');

            window.train_labels = window.train_labels.gather(ind)
            window.train_features = window.train_features.gather(ind)
            window.train_tensors = window.train_tensors.gather(ind)

            ind = tf.tensor1d(Array.from(Array(window.train_labels.shape[1]).keys()).filter(x => x !== category), 'int32')
            window.train_labels = window.train_labels.gather(ind, 1)
        }              
        
        if(window.test_labels){
            let ind = []
            let labels_data = window.test_labels.arraySync()
            for(let i = 0; i < window.test_labels.shape[0]; i++)
                if(labels_data[i][category] === 0)
                ind.push(i)
            ind = tf.tensor1d(ind, 'int32');

            window.test_labels = window.test_labels.gather(ind)
            window.test_features = window.test_features.gather(ind)
            window.test_tensors = window.test_tensors.gather(ind)

            ind = tf.tensor1d(Array.from(Array(window.test_labels.shape[1]).keys()).filter(x => x !== category), 'int32')
            window.test_labels = window.test_labels.gather(ind, 1)
        }

        window.classifier = this.defineNet(this.state.net_size, n_samples.length)
        this.setState({n_samples, images, category_names, category: -1, 
            output_on: -1, classifying: false, accuracy: Array(this.state.category_names.length).fill(0)})

        return
    }
    
    handleDeleteImage(category, imageIndex){
        // No permitir eliminar si no hay imágenes
        if(this.state.n_samples[category] === 0 || imageIndex >= this.state.n_samples[category]){
            return
        }

        let images = this.state.images
        let n_samples = this.state.n_samples
        
        // Determinar si la imagen está en test o train
        const isTestImage = imageIndex < TEST_SAMPLES
        
        // Eliminar la imagen del array
        images[category].splice(imageIndex, 1)
        n_samples[category] -= 1

        // Calcular el índice en los tensores correspondientes
        // Primero contar cuántas imágenes de test/train hay en categorías anteriores
        let testTensorIndex = 0
        let trainTensorIndex = 0
        
        for(let cat = 0; cat < category; cat++){
            const catSamples = this.state.n_samples[cat]
            for(let i = 0; i < catSamples; i++){
                if(i < TEST_SAMPLES){
                    testTensorIndex++
                } else {
                    trainTensorIndex++
                }
            }
        }
        
        // Agregar el offset dentro de la categoría actual
        if(isTestImage){
            testTensorIndex += imageIndex
        } else {
            trainTensorIndex += (imageIndex - TEST_SAMPLES)
        }
        
        // Eliminar de los tensores correspondientes usando gather
        // No usar tf.tidy() aquí porque estamos asignando a variables globales
        if(isTestImage && window.test_tensors && window.test_tensors.shape[0] > testTensorIndex && testTensorIndex >= 0){
            const totalTest = window.test_tensors.shape[0]
            const indices = Array.from(Array(totalTest).keys())
                .filter(i => i !== testTensorIndex)
            
            if(indices.length > 0){
                const indicesTensor = tf.tensor1d(indices, 'int32')
                // Crear nuevos tensores primero
                const oldTestTensors = window.test_tensors
                const oldTestFeatures = window.test_features
                const oldTestLabels = window.test_labels
                
                window.test_tensors = oldTestTensors.gather(indicesTensor)
                window.test_features = oldTestFeatures.gather(indicesTensor)
                window.test_labels = oldTestLabels.gather(indicesTensor)
                
                // Dispose de los tensores antiguos
                oldTestTensors.dispose()
                oldTestFeatures.dispose()
                oldTestLabels.dispose()
                indicesTensor.dispose()
            } else {
                // Si no quedan elementos, crear tensores vacíos
                if(window.test_tensors) window.test_tensors.dispose()
                if(window.test_features) window.test_features.dispose()
                if(window.test_labels) window.test_labels.dispose()
                
                window.test_tensors = tf.zeros([0, this.state.img_size, this.state.img_size, 3])
                window.test_features = tf.zeros([0, 1024])
                window.test_labels = tf.zeros([0, this.state.category_names.length])
            }
        }
        
        if(!isTestImage && window.train_tensors && window.train_tensors.shape[0] > trainTensorIndex && trainTensorIndex >= 0){
            const totalTrain = window.train_tensors.shape[0]
            const indices = Array.from(Array(totalTrain).keys())
                .filter(i => i !== trainTensorIndex)
            
            if(indices.length > 0){
                const indicesTensor = tf.tensor1d(indices, 'int32')
                // Crear nuevos tensores primero
                const oldTrainTensors = window.train_tensors
                const oldTrainFeatures = window.train_features
                const oldTrainLabels = window.train_labels
                
                window.train_tensors = oldTrainTensors.gather(indicesTensor)
                window.train_features = oldTrainFeatures.gather(indicesTensor)
                window.train_labels = oldTrainLabels.gather(indicesTensor)
                
                // Dispose de los tensores antiguos
                oldTrainTensors.dispose()
                oldTrainFeatures.dispose()
                oldTrainLabels.dispose()
                indicesTensor.dispose()
            } else {
                if(window.train_tensors) window.train_tensors.dispose()
                if(window.train_features) window.train_features.dispose()
                if(window.train_labels) window.train_labels.dispose()
                
                window.train_tensors = tf.zeros([0, this.state.img_size, this.state.img_size, 3])
                window.train_features = tf.zeros([0, 1024])
                window.train_labels = tf.zeros([0, this.state.category_names.length])
            }
        }
        
        // Resetear accuracy ya que el modelo necesita reentrenarse
        this.setState({
            images: images,
            n_samples: n_samples,
            accuracy: Array(this.state.category_names.length).fill(0),
            is_training: false
        })
        
        // Si hay suficientes muestras, reentrenar automáticamente
        this.trainClassifier()
    }
    
    handleTimerOut(){

        if(!this.state.classifying){
            setTimeout(this.handleTimerOut, base_timer)
            return
        }
        
        this.classifyPic()

        setTimeout(this.handleTimerOut, base_timer)
        return
    }
    handleTransitionEnd(){
        this.setState({output_on: -1})
    }
    handleTrain(category){
        this.addPic(category)
        this.trainClassifier()
    }
    handleKeyListen(is_enabled){
        this.setState({listen_keys: is_enabled})
    }


    argmax(array){ return array.map((x, i) => [x, i]).reduce((r, a) => (a[0] > r[0] ? a : r))[1]}

    classifyPic(){
        const base64ImageData =  window.webcam.getScreenshot()

        // Create a new Image object
        const image = new Image();

        // Set the source of the image as the Base64 image data
        image.src = base64ImageData;

        // Wait for the image to load
        image.onload = () => {
            const canvas = document.createElement('canvas');
            canvas.width = this.state.img_size;
            canvas.height = this.state.img_size;
            const context = canvas.getContext('2d');
            context.translate(this.state.img_size, 0);
            context.scale(-1, 1);
            context.drawImage(image, 0, 0, canvas.width, canvas.height);
            const imageData = context.getImageData(0, 0, canvas.width, canvas.height);
            
            var output = null
            var scores = null
            tf.tidy(() => {
                var tensor = tf.browser.fromPixels(imageData).expandDims(0);
                
                if(this.state.net_size === 2){ 
                    // MobileNet preprocessing
                    output = window.classifier.predict(window.mobilenet.infer(tensor, true))
                }
                else{
                    output = window.classifier.predict(tensor)
                }
                scores = output.arraySync()[0]
            }
            )
            
            const argmax = this.argmax(scores)
            this.setState({scores: scores, category: argmax, output_on: argmax})   
        };        
        
       
        }

    trainClassifier(){
        // count number of samples per category in train_labels

        let enoughSamples = true
        for(let i = 0; i < this.state.n_samples.length; i++)
            if(this.state.n_samples[i]<MIN_SAMPLES) enoughSamples = false 
    
        // Fit the model only if there are at least N samples per category, the model is not training
        if(enoughSamples && !this.state.is_training){

            this.setState({is_training: true})
            
            
            tf.tidy(() => {

                let train_input, test_input
                if(this.state.net_size<2){
                    train_input = window.train_tensors
                    test_input = window.test_tensors
                }
                else{
                    train_input = window.train_features
                    test_input = window.test_features
                }
                
                window.classifier.fit(train_input, window.train_labels, {
                    batchSize: 8,
                    epochs: 10,
                    shuffle: true,
                    //validationData: [test_input, window.test_labels],
                    //callbacks: {
                    //    onEpochEnd: (epoch, logs) => {
                    //    console.log(`Epoch ${epoch + 1} loss: ${logs.loss.toFixed(2)} acc: ${logs.acc.toFixed(2)} val_loss: ${logs.val_loss.toFixed(2)} val_acc: ${logs.val_acc.toFixed(2)}`);
                    //    //console.log(`Epoch ${epoch + 1} loss: ${logs.loss.toFixed(2)} acc: ${logs.acc.toFixed(2)}`);
                    //}
                    //},
                    //yieldEvery: 'never',
                }).then(() => {

                    // convert one-hot encoding to integer labels
                    let test_labels_int = []
                    for(let i = 0; i < window.test_labels.shape[0]; i++){
                        test_labels_int.push(this.argmax(window.test_labels.arraySync()[i]))
                    }
                    
                    // get avg score per class on test:
                    let avgscore = Array(this.state.category_names.length).fill(0)
                    let n_samples = Array(this.state.category_names.length).fill(0)
                    
                    tf.tidy(() => {
                        let predictions = window.classifier.predict(test_input).arraySync()

                        for(let i = 0; i < predictions.length; i++){
                            avgscore[test_labels_int[i]] += predictions[i][test_labels_int[i]]
                            n_samples[test_labels_int[i]] += 1}

                        })
                    for(let i = 0; i < avgscore.length; i++){
                        if(n_samples[i] > 0) avgscore[i] = avgscore[i]/n_samples[i]
                    }

                    this.setState({accuracy: avgscore, is_training: false})

                });
                
            });
        }
    }
    
    
    addPic(category){

        if(this.state.output_on === -1 && !this.state.is_adding_pic){
            this.setState({is_adding_pic: true})

            let images = this.state.images
            let n_samples = this.state.n_samples
            n_samples[category] += 1
            
            const image = new Image();
            image.src = window.webcam.getScreenshot();
            
            image.onload = () => {
                const canvas = document.createElement('canvas');
                canvas.width = this.state.img_size;
                canvas.height = this.state.img_size;
                const context = canvas.getContext('2d');
                context.translate(this.state.img_size, 0);
                context.scale(-1, 1);
                context.drawImage(image, 0, 0, canvas.width, canvas.height);
                const imageData = context.getImageData(0, 0, canvas.width, canvas.height);

                images[category].push(canvas.toDataURL('image/png'));

                var tensor = tf.browser.fromPixels(imageData).expandDims(0);
                let feature = window.mobilenet.infer(tensor, true)

                
                let label = Array(this.state.category_names.length).fill(0)
                label[category] = 1
                label = tf.tensor(label).expandDims(0)

                if(n_samples[category] <= TEST_SAMPLES){
                    window.test_tensors = tf.concat([window.test_tensors, tensor])
                    window.test_features = tf.concat([window.test_features, feature])
                    window.test_labels = tf.concat([window.test_labels, label])}
                else{
                    window.train_tensors = tf.concat([window.train_tensors, tensor])
                    window.train_features = tf.concat([window.train_features, feature])
                    window.train_labels = tf.concat([window.train_labels, label])}

                this.setState({n_samples,  output_on: category, images, is_adding_pic: false})

              };        
              
           
           
            }
    }

    captureGlobalEvent(e) {
        if(this.state.listen_keys){
            // entrenamiento
            if (e.key <= this.state.category_names.length) {
                const category = Number(e.key)-1
                this.addPic(category)
                this.trainClassifier()
            }
            if (e.key === "c") 
                this.setState({classifying: !this.state.classifying})
        }        
        

    }

    captureCategoryNames(i, name){
        let names = this.state.category_names
        names[i] = name
        this.setState({category_names: names}) 
    }
        

    render(){
        const videoConstraints = {
            width: 350,
            height: 350,
            facingMode: "user"
        };

        let pred_message
        if(!this.state.classifying)
            pred_message = ""
        else
            if(this.state.category!==-1)
                pred_message = "¡Es '" + this.state.category_names[this.state.category] + "'!"
           
        let ypos = []
        for (let i = 0; i <this.state.n_samples.length; i++) 
            ypos[i] = height / 2 + unit_sep[2] * (i - this.state.n_samples.length / 2)

        return(
            <Box className="noselect">

                <AppBar position="static">
                    <Toolbar>
                    <IconButton>
                    <Avatar alt="Kindernet logo" src={logo} />
                    </IconButton>
                    <Typography variant="h5" component="div" sx={{ flexGrow: 1 }}>
                        KinderNet: ¡Enseñemos a la compu a ver!
                    </Typography>

                    <Button disabled={this.state.n_samples.reduce((a, b)=>a+b)===0} onClick={()=>{this.setState({show_images: true, listen_keys: false, classifying: false})}} color="inherit" >Imágenes</Button>
                    <Button startIcon={<SaveIcon />} disabled={this.state.n_samples.reduce((a, b)=>a+b)===0 || this.state.is_training} onClick={()=>{this.setState({save_state: true, listen_keys: false, classifying: false})}} color="inherit">Guardar Estado</Button>
                    <Button startIcon={<FolderOpenIcon />} onClick={()=>{this.loadSavedStatesList(); this.setState({load_state: true, listen_keys: false, classifying: false})}} color="inherit">Cargar Estado</Button>
                    <Button onClick={()=>{this.setState({config: true, listen_keys: false, classifying: false})}} color="inherit">Configuración</Button>
                    <Button onClick={()=>{this.setState({help: true, listen_keys: false, classifying: false})}} color="inherit">Ayuda</Button>
                    <Button onClick={()=>{this.setState({about: true, listen_keys: false, classifying: false})}} color="inherit">Acerca de</Button>
                    </Toolbar>
                    
                </AppBar>

                <Dialog maxWidth="lg" maxHeight="80%" onClose={()=>{this.setState({about: false, listen_keys: true})}} open={this.state.about}>
                    <DialogTitle>KinderNet</DialogTitle>
                    <DialogContent >
                        <Typography align="justify">
                            Este es un proyecto de aplicación web desarrollado  desde el <Link href="http://www.sinc.unl.edu.ar">sinc(i)</Link> para aprender sobre redes neuronales con alumnos de primaria y secundaria. El objetivo es que 
                            los alumnos puedan entrenar su propia red neuronal para reconocer cosas que se presenten a la camara web, de una forma interactiva. 
                            Los alumnos puedan jugar y experimentar con el proceso de entrenamiento y prueba de redes neuronales, cambiando el tamaño de la red, 
                            cantidad y tipos de clases. La red es sencilla pero puede aprender a discriminar cosas con muy pocos ejemplos.
                            <br/> <br/>
                            Más detalles en el <Link href="https://github.com/sinc-lab/kindernet">repositorio del proyecto</Link>.
                        </Typography>
                            
                    </DialogContent>
                </Dialog>

                <Dialog onClose={()=>{this.setState({show_images: false, listen_keys: true})}} open={this.state.show_images}>
                    <DialogTitle>Imágenes</DialogTitle>
                    <DialogContent>
                    <ImagesList images = {this.state.images} category_names={this.state.category_names} n_samples={this.state.n_samples} onDeleteImage={this.handleDeleteImage} />  
                    </DialogContent>
                </Dialog>


                <Dialog onClose={()=>{this.setState({help: false, listen_keys: true})}} open={this.state.help}>
                    <DialogTitle>Instrucciones</DialogTitle>
                    <DialogContent >
                        <Typography align="justify">
                            Antes de comenzar, definamos las cosas que vamos a clasificar. Por ejemplo: "manzana" y "banana" en lugar de "Cosa 1" y "Cosa 2".
                            
                            <br/> <br/>
                        
                            Para comenzar a entrenar la red neuronal, ubicar la primer cosa en la webcam y apretar el 1 o hacer click en la neurona correspondiente de la derecha. Se va a tomar una foto que se pasará a la red para que vaya aprendiendo.

                            <br/> <br/>
                        
                            Hacer lo mismo con la otra cosa hasta que tenga al menos {MIN_SAMPLES} ejemplos cada una. Las barras de la derecha indican qué tan bien la red está aprendiendo cada clase. Si la barra está llena, la red ya aprendió todo lo que puede de esa cosa. Si la barra está vacía, la red no sabe nada de esa cosa.

                            <br/> <br/>

                            Una vez que la red haya aprendido, se puede probar haciendo click en el botón de la izquierda para que indique "Probando". La cámara tomará fotos y las pasará por la red para que indique la cosa que reconoce. 
    
                            <br/> <br/>

                            Podés sumar más cosas haciendo click en el botón <AddIcon/> a la derecha. También podés borrar una cosas haciendo click en el botón <DeleteIcon/>. Refrescando la página (F5) se borra todo y se vuelve a empezar. 

                            

                        </Typography>
                            
                    </DialogContent>
                </Dialog>

                <Dialog onClose={()=>{this.setState({save_state: false, save_name: "", listen_keys: true})}} open={this.state.save_state}>
                    <DialogTitle>Guardar Estado</DialogTitle>
                    <DialogContent>
                        <Grid container spacing={2} sx={{ mt: 1 }}>
                            <Grid item xs={12}>
                                <TextField 
                                    fullWidth
                                    label="Nombre del estado" 
                                    value={this.state.save_name}
                                    onChange={(e) => {this.setState({save_name: e.target.value})}}
                                    placeholder={`Estado ${new Date().toLocaleString()}`}
                                    variant="standard"
                                />
                            </Grid>
                            <Grid item xs={12}>
                                <Typography variant="body2" color="text.secondary">
                                    Se guardará: modelo entrenado, {this.state.category_names.length} categoría(s), {this.state.n_samples.reduce((a, b) => a + b, 0)} imagen(es)
                                </Typography>
                            </Grid>
                            <Grid item xs={12}>
                                <Button 
                                    fullWidth 
                                    variant="contained" 
                                    startIcon={<SaveIcon />}
                                    onClick={this.handleSaveState}
                                    disabled={this.state.is_training}
                                >
                                    Guardar
                                </Button>
                            </Grid>
                        </Grid>
                    </DialogContent>
                </Dialog>

                <Dialog onClose={()=>{this.setState({load_state: false, listen_keys: true})}} open={this.state.load_state} maxWidth="sm" fullWidth>
                    <DialogTitle>Cargar Estado</DialogTitle>
                    <DialogContent>
                        {this.state.saved_states.length === 0 ? (
                            <Typography variant="body2" color="text.secondary" sx={{ py: 2 }}>
                                No hay estados guardados. Guarda un estado primero.
                            </Typography>
                        ) : (
                            <List>
                                {this.state.saved_states.map((savedState, index) => (
                                    <ListItem 
                                        key={savedState.key || index}
                                        disablePadding
                                        secondaryAction={
                                            <IconButton 
                                                edge="end" 
                                                onClick={(e) => this.handleDeleteSavedState(savedState.key, e)}
                                                color="error"
                                            >
                                                <DeleteIcon />
                                            </IconButton>
                                        }
                                    >
                                        <ListItemButton onClick={() => this.handleLoadState(savedState.key, savedState.data)}>
                                            <ListItemText 
                                                primary={savedState.name || `Estado ${index + 1}`}
                                                secondary={
                                                    `${savedState.data.category_names.length} categoría(s), ` +
                                                    `${savedState.data.n_samples.reduce((a, b) => a + b, 0)} imagen(es) - ` +
                                                    `${new Date(savedState.savedAt).toLocaleString()}`
                                                }
                                            />
                                        </ListItemButton>
                                    </ListItem>
                                ))}
                            </List>
                        )}
                    </DialogContent>
                </Dialog>

                <Dialog onClose={()=>{this.resetValues(); this.setState({config: false, listen_keys: true});}}  open={this.state.config}>
                    <DialogTitle>Configuración</DialogTitle>
                    <DialogContent >
                    <Grid container justifyContent='center' alignItems='center'>
                        <TextField error={this.state.img_size<16 || this.state.img_size>224} 
                        helperText={this.state.img_size<16 || this.state.img_size>224 ? 'Usar imágenes entre 16 y 224 píxeles': ''} id="standard-basic" type="number" label="Tamaño de imagen" defaultValue={this.state.img_size} variant="standard" 
                        onChange={(e) => {this.setState({img_size: Number(e.target.value)})}}/>
                    </Grid> 
                            
                    </DialogContent>
                </Dialog>

                <EventListener onKeyUp={this.captureGlobalEvent}/>
                <Grid  container pt={10} justifyContent='center' textAlign='center'>
                    
                    <Grid item pt={15} sm={4} lg={2}>
                        <Webcam videoConstraints = {videoConstraints} audio={false} ref={this.setRef} screenshotFormat="image/png" quality={1} className="Webcam"/> 
                        <Card variant="outlined">
                            <h2>Panel de control</h2>
                            <Grid container justifyContent='center' alignItems='center'>
                                <Grid style={{color:this.state.classifying? "black":"gray"}}><h3>Probando</h3></Grid>
                                <Switch checked={!this.state.classifying} onChange={()=>{this.setState({classifying: !this.state.classifying})}} />
                                <Grid style={{color:this.state.classifying? "gray":"black"}}><h3>Aprendiendo</h3></Grid>
                            </Grid> 
                            <FormLabel id="radio-buttons-size">Tamaño de la red neuronal</FormLabel>
                            <Grid container justifyContent='center' alignItems='center'>
                                <RadioGroup aria-labelledby="radio-buttons-size" value={this.state.net_size === 0 ? "Pequeña" : "Grande"}>
                                    <FormControlLabel value="Pequeña" control={<Radio onChange={()=>{this.handleClassifierChange(0)}}/>} 
                                    label="Pequeña" />
                                    <FormControlLabel value="Grande" control={<Radio onChange={()=>{this.handleClassifierChange(2)}}/>} 
                                    label="Grande" />
                                </RadioGroup>
                            </Grid> 
                        </Card>
                        
                    </Grid>

                    <Grid item sm={4}>
                        
                        <Network is_enabled={this.state.listen_keys} onClick={this.handleTrain} category = {this.state.output_on} onTransitionEnd = {this.handleTransitionEnd}
                            size = {this.state.net_size} n_outputs = {this.state.category_names.length}
                            classifying = {this.state.classifying} />        
                         
                        <h1>{pred_message}</h1>   
            
                    </Grid>

                    <Grid item sm={4} lg={2}>
                        <CategoryList images = {this.state.images.map(last_img => last_img?last_img[last_img.length - 1]:null)}  scores={this.state.classifying?this.state.scores:this.state.accuracy} ypos={ypos} enableKeys={this.handleKeyListen} category_names={this.state.category_names} n_samples={this.state.n_samples} 
            get_category_names={this.captureCategoryNames} handleAddCategory={this.handleAddCategory} handleRemoveCategory={this.handleRemoveCategory}/>     
                    </Grid>

                </Grid>
                
                <Grid  container pr={10} mt={-10} justifyContent='right' textAlign='center'>
                <a href="http://sinc.unl.edu.ar">
                    <img src={sinclogo} style={{height: 70}} alt={"sinc(i) logo"} />
                </a>
                </Grid>
            </Box>
        );
    }

}



function App() {
  return (
      <React.Fragment>
          <CssBaseline />
          <KinderNet />
      </React.Fragment>

  );
}

export default App;
