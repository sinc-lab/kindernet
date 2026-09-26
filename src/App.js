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
import {height, unit_sep, use_timer, base_timer, batch_size, train_epochs, train_debounce, use_shape_uniforms, feature_chunk, data_tensors, bn_momentum, capture_cooldown} from './constants';
import Avatar from '@mui/material/Avatar';
import logo from "./ia.png"
import sinclogo from "./sinc-logo.png"
import * as tf from '@tensorflow/tfjs'
import * as mobilenet from '@tensorflow-models/mobilenet';


var TEST_SAMPLES = 2
var MIN_SAMPLES = 5

// elegido por el usuario o, por defecto, activado en máquinas modestas
function defaultLowPerfMode(){
    const saved = localStorage.getItem('kindernet_low_perf')
    if(saved !== null)
        return saved === 'true'
    return Boolean((navigator.deviceMemory && navigator.deviceMemory <= 4) || (navigator.hardwareConcurrency && navigator.hardwareConcurrency <= 2))
}

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
            low_perf: defaultLowPerfMode(),
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
            save_name: "",
            gpu_error: false
        };
        this.response = null
        // estado del entrenamiento (fuera de this.state: setState es asíncrono)
        this.training = false
        this.train_pending = false
        this.train_timer = null
        this.classify_timer = null
        this.output_timer = null
        this.last_pic = 0
        this.gpu_recovery_tried = false
        this.pending_dispose = []
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
        this.handleLowPerfChange = this.handleLowPerfChange.bind(this);
        this.handleTrainNow = this.handleTrainNow.bind(this);
        
        this.classifyPic = this.classifyPic.bind(this);
    }
    setRef = webcam => {
        window.webcam = webcam;
    };

  
    defineNet(net_size, nclasses){
        
        let classifier = tf.sequential();
        if(net_size === 0){
            // los píxeles entran en 0-255 y la red los lleva a 0-1
            classifier.add(tf.layers.rescaling({scale: 1/255, inputShape: [this.state.img_size, this.state.img_size, 3]}))
            classifier.add(tf.layers.conv2d({filters: 16, kernelSize: 3, activation: 'relu'}))
            classifier.add(tf.layers.batchNormalization({momentum: bn_momentum}))
            classifier.add(tf.layers.maxPooling2d({poolSize: 2}))
            classifier.add(tf.layers.conv2d({filters: 32, kernelSize: 5, activation: 'relu'}))
            classifier.add(tf.layers.batchNormalization({momentum: bn_momentum}))
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

    replaceClassifier(net_size, nclasses){
        if(window.classifier){
            window.classifier.stopTraining = true
            this.disposeLater(window.classifier)
        }
        window.classifier = this.defineNet(net_size, nclasses)
    }

    // La GPU puede dejar de devolver resultados sin avisar (ni error ni contexto perdido): todo se lee como cero.
    // Esta sonda lo detecta; sin ella la app muestra 0% como si la red no hubiera aprendido nada.
    gpuIsHealthy(){
        let healthy = false
        try{
            healthy = tf.tidy(() => tf.tensor1d([1, 2, 3]).square().sum().arraySync() === 14)
        }catch(error){
            console.error("Error al comprobar la GPU:", error)
        }
        if(!healthy)
            console.error("Diagnóstico de la GPU:", JSON.stringify(this.gpuDiagnostics()))
        return healthy
    }

    gpuDiagnostics(){
        const flags = ['WEBGL_VERSION', 'WEBGL_RENDER_FLOAT32_ENABLED', 'WEBGL_DOWNLOAD_FLOAT_ENABLED',
            'WEBGL_FORCE_F16_TEXTURES', 'WEBGL_BUFFER_SUPPORTED', 'WEBGL_FENCE_API_ENABLED', 'WEBGL_PACK']
        const info = {backend: tf.getBackend(), memoria: tf.memory().numTensors}
        flags.forEach(f => { try{ info[f] = tf.env().get(f) }catch(error){ info[f] = 'error' } })
        try{
            const gl = tf.backend().gpgpu.gl
            info.contextLost = gl.isContextLost()
            info.glError = gl.getError()
            const ext = gl.getExtension('WEBGL_debug_renderer_info')
            info.renderer = ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : 'desconocido'
        }catch(error){
            info.gl = 'sin acceso al contexto'
        }
        return info
    }

    // Recrea el backend y rearma los datos desde las fotos guardadas. Devuelve false si no se pudo recuperar.
    async recoverFromGpuFailure(){
        console.warn("La GPU dejó de devolver resultados: recreando el backend")
        this.cancelScheduledTraining()
        this.pending_dispose = []
        try{
            const factory = tf.findBackendFactory('webgl')
            tf.removeBackend('webgl')
            if(factory)
                tf.registerBackend('webgl', factory, 2)
            await tf.setBackend(factory ? 'webgl' : 'cpu')
            await tf.ready()
        }catch(error){
            console.error("No se pudo recrear el backend:", error)
            return false
        }
        if(!this.gpuIsHealthy())
            return false
        window.classifier = this.defineNet(this.state.net_size, this.state.category_names.length)
        await this.rebuildFromImages()
        return this.gpuIsHealthy()
    }

    // Rearma los tensores desde los PNG del estado. Los anteriores ya murieron con el backend, no se liberan.
    async rebuildFromImages(){
        const size = this.state.img_size
        const nclasses = this.state.category_names.length
        const frames = {train: [], test: []}
        for(let cat = 0; cat < this.state.images.length; cat++)
            for(let i = 0; i < this.state.images[cat].length; i++)
                frames[i < TEST_SAMPLES ? 'test' : 'train'].push({data: await this.decodeImage(this.state.images[cat][i], size), cat})

        for(const set of ['train', 'test']){
            window[set + '_features'] = tf.zeros([0, 1024])
            if(frames[set].length === 0){
                window[set + '_tensors'] = tf.zeros([0, size, size, 3])
                window[set + '_labels'] = tf.zeros([0, nclasses])
                continue
            }
            const [tensors, labels] = tf.tidy(() => [
                tf.stack(frames[set].map(f => tf.browser.fromPixels(f.data).toFloat())),
                tf.oneHot(frames[set].map(f => f.cat), nclasses).toFloat()
            ])
            window[set + '_tensors'] = tensors
            window[set + '_labels'] = labels
        }
    }

    decodeImage(dataUrl, size){
        return new Promise((resolve, reject) => {
            const image = new Image()
            image.onload = () => {
                const canvas = document.createElement('canvas')
                canvas.width = size
                canvas.height = size
                const context = canvas.getContext('2d')
                context.drawImage(image, 0, 0, size, size)
                resolve(context.getImageData(0, 0, size, size))
            }
            image.onerror = reject
            image.src = dataUrl
        })
    }

    // libera un tensor o modelo; si hay un fit() en curso, lo difiere hasta que termine
    disposeLater(x){
        if(!x) return
        if(this.training)
            this.pending_dispose.push(x)
        else
            x.dispose()
    }

    flushDisposals(){
        this.pending_dispose.forEach(x => x.dispose())
        this.pending_dispose = []
    }

    resetValues(){
        this.cancelScheduledTraining()
        data_tensors.forEach(name => this.disposeLater(window[name]))
        window.train_tensors = tf.zeros([0, this.state.img_size, this.state.img_size, 3])
        window.train_features = tf.zeros([0, 1024])
        window.train_labels = tf.zeros([0, 2])
        window.test_tensors = tf.zeros([0, this.state.img_size, this.state.img_size, 3])
        window.test_features = tf.zeros([0, 1024])
        window.test_labels = tf.zeros([0, 2])
        // Inicializa el timer (uno solo)
        if(use_timer){
            clearTimeout(this.classify_timer)
            this.classify_timer = setTimeout(this.handleTimerOut, base_timer)
        }
        this.replaceClassifier(0, 2)
        this.setState({net_size: 0, category: -1, output_on: -1, classifying: false, accuracy: [0, 0],
        images: [[], []], n_samples: [0,0], n_outputs: [0, 0], category_names: ["Cosa 1", "Cosa 2"]})
    }

    componentDidMount() {
        // las formas van como uniforms en los shaders: menos programas WebGL para compilar
        if(use_shape_uniforms){
            try{
                tf.env().set('WEBGL_USE_SHAPES_UNIFORMS', true)
            }catch(error){
                console.warn("No se pudo activar WEBGL_USE_SHAPES_UNIFORMS:", error)
            }
        }

        mobilenet.load().then((net) => {
            window.mobilenet = net
            this.setState({listen_keys: true, gpu_error: !this.gpuIsHealthy()})
        })

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
            
            this.ensureFeatures()
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

            // Liberar tensores y modelo anteriores
            this.cancelScheduledTraining()
            data_tensors.forEach(name => this.disposeLater(window[name]))
            if (window.classifier) {
                window.classifier.stopTraining = true
                this.disposeLater(window.classifier)
            }

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
        this.replaceClassifier(net_size, this.state.category_names.length)
        this.setState({net_size: net_size, category: -1, output_on: -1, classifying: false, accuracy: Array(this.state.category_names.length).fill(0)})
        this.scheduleTraining()
    }
    handleAddCategory(){
        let category_names = this.state.category_names
        category_names.push("Cosa " + (category_names.length + 1))
        let n_samples = this.state.n_samples
        n_samples.push(0)
        let zeros = Array(category_names.length).fill(0)
        let images = this.state.images
        images.push([])
        
        // agrega una columna a las etiquetas
        for(const name of ['train_labels', 'test_labels']){
            const old = window[name]
            if(!old) continue
            window[name] = tf.tidy(() => tf.concat([old, tf.zeros([old.shape[0], 1])], 1))
            this.disposeLater(old)
        }

        this.cancelScheduledTraining()
        this.replaceClassifier(this.state.net_size, category_names.length)
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
        
        // saca las fotos y la columna de la categoría
        this.removeCategoryFromSet('train', category)
        this.removeCategoryFromSet('test', category)

        this.replaceClassifier(this.state.net_size, n_samples.length)
        this.setState({n_samples, images, category_names, category: -1,
            output_on: -1, classifying: false, accuracy: Array(this.state.category_names.length).fill(0)})
        this.scheduleTraining()
    }

    removeCategoryFromSet(set, category){
        const labels = window[set + '_labels'].arraySync()
        const rows = []
        for(let i = 0; i < labels.length; i++)
            if(labels[i][category] === 0) rows.push(i)
        this.keepRowsInSet(set, rows, category)
    }

    // deja solo las filas `rows` del conjunto; con `drop_column` saca esa columna de las etiquetas
    keepRowsInSet(set, rows, drop_column = -1){
        const old = [window[set + '_tensors'], window[set + '_features'], window[set + '_labels']]
        const gather = (x, r) => r.length > 0 ? x.gather(tf.tensor1d(r, 'int32')) : tf.zeros([0, ...x.shape.slice(1)])
        const [tensors, features, labels] = tf.tidy(() => {
            let labels = gather(old[2], rows)
            if(drop_column !== -1){
                const cols = Array.from(Array(old[2].shape[1]).keys()).filter(c => c !== drop_column)
                labels = rows.length > 0 ? labels.gather(tf.tensor1d(cols, 'int32'), 1) : tf.zeros([0, cols.length])
            }
            // los rasgos pueden faltar para las últimas filas
            return [gather(old[0], rows), gather(old[1], rows.filter(i => i < old[1].shape[0])), labels]
        })
        old.forEach(t => this.disposeLater(t))
        window[set + '_tensors'] = tensors
        window[set + '_features'] = features
        window[set + '_labels'] = labels
    }

    handleDeleteImage(category, imageIndex){
        if(this.state.n_samples[category] === 0 || imageIndex >= this.state.n_samples[category])
            return

        let images = this.state.images
        let n_samples = this.state.n_samples

        // por categoría, primero van las fotos de prueba y después las de entrenamiento
        const test_labels = window.test_labels.arraySync()
        const n_test = test_labels.filter(label => label[category] === 1).length
        const set = imageIndex < n_test ? 'test' : 'train'
        const position = imageIndex < n_test ? imageIndex : imageIndex - n_test
        const labels = set === 'test' ? test_labels : window.train_labels.arraySync()
        let row = -1, count = 0
        for(let i = 0; i < labels.length && row === -1; i++){
            if(labels[i][category] !== 1) continue
            if(count === position) row = i
            count++
        }
        if(row !== -1)
            this.keepRowsInSet(set, Array.from(Array(labels.length).keys()).filter(i => i !== row))

        images[category].splice(imageIndex, 1)
        n_samples[category] -= 1
        this.setState({images, n_samples, accuracy: Array(this.state.category_names.length).fill(0)})
        this.scheduleTraining()
    }

    handleTimerOut(){
        if(this.state.classifying)
            this.classifyPic()
        this.classify_timer = setTimeout(this.handleTimerOut, base_timer)
    }
    handleTransitionEnd(){
        clearTimeout(this.output_timer)
        this.setState({output_on: -1})
    }
    handleTrain(category){
        this.addPic(category)
    }
    handleKeyListen(is_enabled){
        this.setState({listen_keys: is_enabled})
    }

    argmax(array){ return array.map((x, i) => [x, i]).reduce((r, a) => (a[0] > r[0] ? a : r))[1]}

    // cuadro actual de la cámara (espejado) en un canvas de img_size x img_size, o null si todavía no hay video
    captureFrame(){
        const video = window.webcam ? window.webcam.video : null
        if(!video || video.readyState < 2)
            return null
        const size = this.state.img_size
        const canvas = document.createElement('canvas')
        canvas.width = size
        canvas.height = size
        const context = canvas.getContext('2d')
        context.translate(size, 0)
        context.scale(-1, 1)
        context.drawImage(video, 0, 0, size, size)
        return {canvas, imageData: context.getImageData(0, 0, size, size)}
    }

    classifyPic(){
        const frame = this.captureFrame()
        if(!frame || !window.classifier || !window.mobilenet)
            return

        const scores = tf.tidy(() => {
            const pixels = tf.browser.fromPixels(frame.imageData).expandDims(0)
            const input = this.state.net_size === 2 ? window.mobilenet.infer(pixels, true) : pixels.toFloat()
            return window.classifier.predict(input).arraySync()[0]
        })

        const argmax = this.argmax(scores)
        this.setState({scores: scores, category: argmax, output_on: argmax})
    }

    // El entrenamiento se dispara train_debounce ms después de la última foto, una vez por ráfaga

    scheduleTraining(){
        clearTimeout(this.train_timer)
        if(this.state.low_perf)
            return
        this.train_timer = setTimeout(() => this.trainClassifier(), train_debounce)
    }

    cancelScheduledTraining(){
        clearTimeout(this.train_timer)
        this.train_timer = null
        this.train_pending = false
    }

    enoughSamples(){
        return this.state.n_samples.length > 0 && this.state.n_samples.every(n => n >= MIN_SAMPLES)
    }

    // completa hasta un múltiplo de batch_size repitiendo ejemplos: lotes de forma fija, sin recompilar shaders
    padToBatch(x, y){
        const n = x.shape[0]
        const padded = Math.ceil(n / batch_size) * batch_size
        const ind = Array.from(Array(n).keys())
        for(let i = n; i < padded; i++)
            ind.push(Math.floor(Math.random() * n))
        return tf.tidy(() => {
            const idx = tf.tensor1d(ind, 'int32')
            return [x.gather(idx), y.gather(idx)]
        })
    }

    async trainClassifier(){
        if(!this.enoughSamples())
            return
        if(this.training){
            this.train_pending = true
            return
        }

        this.training = true
        this.train_pending = false
        this.setState({is_training: true})

        const model = window.classifier
        const net_size = this.state.net_size
        if(net_size === 2)
            this.ensureFeatures()
        const train_input = net_size < 2 ? window.train_tensors : window.train_features
        const [train_x, train_y] = this.padToBatch(train_input, window.train_labels)

        try{
            await model.fit(train_x, train_y, {batchSize: batch_size, epochs: train_epochs, shuffle: true})
            // si la red cambió durante el fit, el resultado ya no sirve
            if(model === window.classifier){
                if(this.gpuIsHealthy())
                    this.setState({accuracy: this.evaluate(model, net_size), gpu_error: false})
                else{
                    // se intenta recuperar una sola vez: si vuelve a fallar, no tiene sentido insistir
                    const recovered = !this.gpu_recovery_tried && await this.recoverFromGpuFailure()
                    this.gpu_recovery_tried = true
                    this.setState({gpu_error: !recovered})
                    this.train_pending = recovered
                }
            }
        }catch(error){
            console.error("Error durante el entrenamiento:", error)
        }finally{
            train_x.dispose()
            train_y.dispose()
            this.training = false
            this.flushDisposals()
            this.setState({is_training: false})
            if(this.train_pending)
                this.trainClassifier()
        }
    }

    // rasgos de MobileNet que falten, de a feature_chunk fotos
    ensureFeatures(){
        for(const set of ['train', 'test']){
            const tensors = window[set + '_tensors']
            while(window[set + '_features'].shape[0] < tensors.shape[0]){
                const old = window[set + '_features']
                const start = old.shape[0]
                const n = Math.min(feature_chunk, tensors.shape[0] - start)
                window[set + '_features'] = tf.tidy(() => {
                    const chunk = tensors.slice([start, 0, 0, 0], [n, -1, -1, -1])
                    return tf.concat([old, window.mobilenet.infer(chunk, true)])
                })
                this.disposeLater(old)
            }
        }
    }

    // puntaje promedio por clase sobre las imágenes de prueba
    evaluate(model, net_size){
        const nclasses = this.state.category_names.length
        const avgscore = Array(nclasses).fill(0)
        const counts = Array(nclasses).fill(0)
        const test_input = net_size < 2 ? window.test_tensors : window.test_features
        if(test_input.shape[0] === 0)
            return avgscore

        tf.tidy(() => {
            const labels = window.test_labels.arraySync()
            const predictions = model.predict(test_input).arraySync()
            for(let i = 0; i < predictions.length; i++){
                const c = this.argmax(labels[i])
                avgscore[c] += predictions[i][c]
                counts[c] += 1
            }
        })
        return avgscore.map((score, i) => counts[i] > 0 ? score / counts[i] : 0)
    }

    addPic(category){
        // cooldown corto: antes se esperaba toda la animación de la neurona y en una ráfaga se perdían fotos
        const now = Date.now()
        if(now - this.last_pic < capture_cooldown)
            return
        const frame = this.captureFrame()
        if(!frame)
            return
        this.last_pic = now

        let images = this.state.images
        let n_samples = this.state.n_samples
        n_samples[category] += 1
        images[category].push(frame.canvas.toDataURL('image/png'))

        // las primeras TEST_SAMPLES fotos van al conjunto de prueba
        const set = n_samples[category] <= TEST_SAMPLES ? 'test' : 'train'
        const old = [window[set + '_tensors'], window[set + '_labels']]
        const [tensors, labels] = tf.tidy(() => {
            const pixels = tf.browser.fromPixels(frame.imageData).expandDims(0).toFloat()
            const label = tf.oneHot(category, this.state.category_names.length).toFloat().expandDims(0)
            return [tf.concat([old[0], pixels]), tf.concat([old[1], label])]
        })
        old.forEach(t => this.disposeLater(t))
        window[set + '_tensors'] = tensors
        window[set + '_labels'] = labels

        // output_on: -1 y después la categoría, para que la animación se reinicie en fotos seguidas
        this.setState({n_samples, images, output_on: -1}, () => this.setState({output_on: category}))
        clearTimeout(this.output_timer)
        this.output_timer = setTimeout(this.handleTransitionEnd, 1500)
        this.scheduleTraining()
    }

    handleLowPerfChange(enabled){
        localStorage.setItem('kindernet_low_perf', enabled)
        this.setState({low_perf: enabled}, () => enabled ? this.cancelScheduledTraining() : this.scheduleTraining())
    }

    handleTrainNow(){
        this.cancelScheduledTraining()
        this.trainClassifier()
    }

    captureGlobalEvent(e) {
        if(this.state.listen_keys){
            // entrenamiento
            if (/^[1-9]$/.test(e.key) && Number(e.key) <= this.state.category_names.length)
                this.addPic(Number(e.key) - 1)
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

        let pred_message = ""
        if(this.state.classifying){
            if(this.state.category !== -1)
                pred_message = "¡Es '" + this.state.category_names[this.state.category] + "'!"
        }
        else if(this.state.is_training)
            pred_message = "La red está aprendiendo..."

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

                            <br/> <br/>

                            Si la computadora es lenta, activá "Modo bajo rendimiento" en el panel de control: las fotos se guardan sin entrenar y la red aprende una sola vez cuando tocás el botón Entrenar.

                            

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
                            <Grid container justifyContent='center' alignItems='center'>
                                <FormControlLabel labelPlacement="start" sx={{mx: 0}} label="Modo bajo rendimiento"
                                    control={<Switch checked={this.state.low_perf} onChange={(e)=>{this.handleLowPerfChange(e.target.checked)}}/>} />
                            </Grid>
                            {this.state.low_perf &&
                                <Box px={1.5} pb={1.5}>
                                    <Typography variant="caption" display="block" color="text.secondary">Las fotos se guardan sin entrenar. Cuando termines, tocá Entrenar.</Typography>
                                    <Button fullWidth variant="contained" sx={{mt: 1}} onClick={this.handleTrainNow} disabled={!this.enoughSamples() || this.state.is_training}>
                                        {this.state.is_training ? "Aprendiendo..." : "Entrenar"}
                                    </Button>
                                    {!this.enoughSamples() &&
                                        <Typography variant="caption" display="block" color="text.secondary" sx={{mt: 0.5}}>Cada cosa necesita al menos {MIN_SAMPLES} fotos</Typography>}
                                </Box>} 
                        </Card>
                        
                    </Grid>

                    <Grid item sm={4}>
                        
                        <Network is_enabled={this.state.listen_keys} onClick={this.handleTrain} category = {this.state.output_on} onTransitionEnd = {this.handleTransitionEnd}
                            size = {this.state.net_size} n_outputs = {this.state.category_names.length}
                            classifying = {this.state.classifying} />        
                         
                        <h1>{pred_message}</h1>
                        {this.state.gpu_error &&
                            <Typography color="error" px={2}>
                                El navegador dejó de responder en la placa de video, así que la red no puede aprender.
                                Recargá la página (F5) y, si vuelve a pasar, activá "Modo bajo rendimiento".
                            </Typography>}
            
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
