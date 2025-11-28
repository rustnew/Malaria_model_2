//! Architecture CNN équilibrée pour la classification d'images de frottis sanguins

use burn::{
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig, MaxPool2d, MaxPool2dConfig},
        BatchNorm, BatchNormConfig, Dropout, DropoutConfig, Linear, LinearConfig, Relu,
    },
    tensor::{backend::Backend, Tensor},
    train::{TrainOutput, TrainStep, ValidStep},
};
use crate::data::dataset::MalariaBatch;
use crate::config::model_config::ModelConfig;
use burn_tensor::loss::cross_entropy_with_logits;

/// Réseau neuronal convolutif équilibré pour la détection du paludisme
#[derive(Module, Debug)]
pub struct MalariaCNN<B: Backend> {
    // Bloc convolutif 1 avec BatchNorm
    conv1: Conv2d<B>,
    bn1: BatchNorm<B>,
    
    // Bloc convolutif 2 avec BatchNorm  
    conv2: Conv2d<B>,
    bn2: BatchNorm<B>,
    
    // Bloc convolutif 3 avec BatchNorm
    conv3: Conv2d<B>,
    bn3: BatchNorm<B>,
    
    // Pooling
    pool1: MaxPool2d,
    pool2: MaxPool2d,
    pool3: MaxPool2d,
    
    // Pooling adaptatif pour s'adapter à différentes tailles
    adaptive_pool: AdaptiveAvgPool2d,
    
    // Classificateur
    dropout: Dropout,
    fc1: Linear<B>,
    fc2: Linear<B>,
    fc3: Linear<B>,
    
    // Activation
    relu: Relu,
}

impl<B: Backend> MalariaCNN<B> {
    /// Crée une nouvelle instance du CNN équilibré
    pub fn new(device: &B::Device, config: &ModelConfig) -> Self {
        let conv1 = Conv2dConfig::new([config.image_channels, config.conv1_filters], [3, 3])
            .with_padding(burn::nn::PaddingConfig2d::Same)
            .init(device);
            
        let conv2 = Conv2dConfig::new([config.conv1_filters, config.conv2_filters], [3, 3])
            .with_padding(burn::nn::PaddingConfig2d::Same)
            .init(device);
            
        let conv3 = Conv2dConfig::new([config.conv2_filters, config.conv3_filters], [3, 3])
            .with_padding(burn::nn::PaddingConfig2d::Same)
            .init(device);

        // Batch Normalization pour une convergence plus rapide et stable
        let bn1 = BatchNormConfig::new(config.conv1_filters).init(device);
        let bn2 = BatchNormConfig::new(config.conv2_filters).init(device);
        let bn3 = BatchNormConfig::new(config.conv3_filters).init(device);

        let pool1 = MaxPool2dConfig::new([2, 2])
            .with_strides([2, 2])
            .init();
            
        let pool2 = MaxPool2dConfig::new([2, 2])
            .with_strides([2, 2])
            .init();
            
        let pool3 = MaxPool2dConfig::new([2, 2])
            .with_strides([2, 2])
            .init();

        let adaptive_pool = AdaptiveAvgPool2dConfig::new([4, 4]).init();

        let dropout = DropoutConfig::new(config.dropout_rate).init();
        
        // Calcul de la taille d'entrée pour la première couche fully-connected
        let fc_input_size = config.conv3_filters * 4 * 4; // Après adaptive pooling 4x4
        
        let fc1 = LinearConfig::new(fc_input_size as usize, config.fc1_units)
            .init(device);
            
        let fc2 = LinearConfig::new(config.fc1_units, config.fc2_units)
            .init(device);
            
        let fc3 = LinearConfig::new(config.fc2_units, config.num_classes)
            .init(device);

        let relu = Relu::new();

        Self {
            conv1,
            bn1,
            conv2,
            bn2,
            conv3,
            bn3,
            pool1,
            pool2,
            pool3,
            adaptive_pool,
            dropout,
            fc1,
            fc2,
            fc3,
            relu,
        }
    }

    /// Passe avant complète du réseau
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        // Bloc 1: Conv -> BatchNorm -> ReLU -> Pool
        let x = self.conv1.forward(x);
        let x = self.bn1.forward(x);
        let x = self.relu.forward(x);
        let x = self.pool1.forward(x);

        // Bloc 2: Conv -> BatchNorm -> ReLU -> Pool  
        let x = self.conv2.forward(x);
        let x = self.bn2.forward(x);
        let x = self.relu.forward(x);
        let x = self.pool2.forward(x);

        // Bloc 3: Conv -> BatchNorm -> ReLU -> Pool
        let x = self.conv3.forward(x);
        let x = self.bn3.forward(x);
        let x = self.relu.forward(x);
        let x = self.pool3.forward(x);

        // Pooling adaptatif pour taille fixe
        let x = self.adaptive_pool.forward(x);

        // Reshape pour les couches fully-connected
        let [batch_size, channels, height, width] = x.dims();
        let x = x.reshape([batch_size, channels * height * width]);

        // Classificateur avec dropout
        let x = self.dropout.forward(self.fc1.forward(x));
        let x = self.relu.forward(x);
        
        let x = self.dropout.forward(self.fc2.forward(x));
        let x = self.relu.forward(x);
        
        // Dernière couche (pas d'activation pour les logits)
        self.fc3.forward(x)
    }

    /// Passe avant pour l'inférence (sans dropout)
    pub fn infer(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        // Bloc 1: Conv -> BatchNorm -> ReLU -> Pool
        let x = self.conv1.forward(x);
        let x = self.bn1.forward(x);
        let x = self.relu.forward(x);
        let x = self.pool1.forward(x);

        // Bloc 2: Conv -> BatchNorm -> ReLU -> Pool  
        let x = self.conv2.forward(x);
        let x = self.bn2.forward(x);
        let x = self.relu.forward(x);
        let x = self.pool2.forward(x);

        // Bloc 3: Conv -> BatchNorm -> ReLU -> Pool
        let x = self.conv3.forward(x);
        let x = self.bn3.forward(x);
        let x = self.relu.forward(x);
        let x = self.pool3.forward(x);

        // Pooling adaptatif pour taille fixe
        let x = self.adaptive_pool.forward(x);

        // Reshape pour les couches fully-connected
        let [batch_size, channels, height, width] = x.dims();
        let x = x.reshape([batch_size, channels * height * width]);

        // Classificateur sans dropout pour l'inférence
        let x = self.fc1.forward(x);
        let x = self.relu.forward(x);
        
        let x = self.fc2.forward(x);
        let x = self.relu.forward(x);
        
        // Dernière couche (pas d'activation pour les logits)
        self.fc3.forward(x)
    }
}

// Structure de sortie pour la classification
#[derive(Debug, Clone)]
pub struct ClassificationOutput<B: Backend> {
    /// Perte calculée
    pub loss: Tensor<B, 1>,
    /// Sortie du modèle (logits)
    pub output: Tensor<B, 2>,
    /// Labels cibles
    pub targets: Tensor<B, 1, burn::tensor::Int>,
}

// Implémentation du trait ItemLazy pour ClassificationOutput
impl<B: Backend> burn::train::metric::ItemLazy for ClassificationOutput<B> {
    type ItemSync = Self;

    fn sync(self) -> Self::ItemSync {
        self
    }
}

// Implémentation du trait Adaptor pour les métriques
impl<B: Backend> burn::train::metric::Adaptor<burn::train::metric::LossInput<B>> for ClassificationOutput<B> {
    fn adapt(&self) -> burn::train::metric::LossInput<B> {
        burn::train::metric::LossInput::new(self.loss.clone())
    }
}

impl<B: Backend> burn::train::metric::Adaptor<burn::train::metric::AccuracyInput<B>> for ClassificationOutput<B> {
    fn adapt(&self) -> burn::train::metric::AccuracyInput<B> {
        let predictions = self.output.clone().argmax(1);
        let predictions_float = predictions.float();
        burn::train::metric::AccuracyInput::new(predictions_float, self.targets.clone())
    }
}

// Implémentation des traits d'entraînement
impl<B: burn::tensor::backend::AutodiffBackend> TrainStep<MalariaBatch<B>, ClassificationOutput<B>> for MalariaCNN<B> {
    fn step(&self, batch: MalariaBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let output = self.forward(batch.images);
        
        // Convertir les labels en one-hot encoding pour la cross entropy
        let targets = batch.labels
            .clone()
            .float()
            .one_hot(2); // 2 classes
    
        let loss = cross_entropy_with_logits(output.clone(), targets);

        // Calcul des gradients
        let grads = loss.backward();
        
        TrainOutput::new(
            self,
            grads,
            ClassificationOutput {
                loss: loss.clone(),
                output: output.detach(),
                targets: batch.labels,
            }
        )
    }
}

impl<B: Backend> ValidStep<MalariaBatch<B>, ClassificationOutput<B>> for MalariaCNN<B> {
    fn step(&self, batch: MalariaBatch<B>) -> ClassificationOutput<B> {
        let output = self.infer(batch.images);
        
        // Convertir les labels en one-hot encoding pour la cross entropy
        let targets = batch.labels
            .clone()
            .float()
            .one_hot(2); // 2 classes
    
        let loss = cross_entropy_with_logits(output.clone(), targets);

        ClassificationOutput {
            loss,
            output,
            targets: batch.labels,
        }
    }
}