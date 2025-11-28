//! Trainer équilibré pour l'entraînement du modèle CNN

use burn::{
    data::{dataloader::DataLoaderBuilder, dataset::Dataset},
    optim::{AdamConfig, decay::WeightDecayConfig},
    tensor::backend::AutodiffBackend,
    train::{
        metric::{AccuracyMetric, LossMetric},
        LearnerBuilder,
    },
    record::{BinFileRecorder, FullPrecisionSettings, Recorder},
    module::Module,
};

use crate::{
    config::model_config::ModelConfig,
    data::dataset::{MalariaBatcher, MalariaDataset},
    model::malaria_cnn::MalariaCNN,
};

/// Entraîneur équilibré pour le modèle de détection du paludisme
pub struct MalariaTrainer<B: AutodiffBackend> {
    config: ModelConfig,
    _backend: std::marker::PhantomData<B>,
}

impl<B: AutodiffBackend> MalariaTrainer<B> {
    /// Crée un nouvel entraîneur
    pub fn new(config: ModelConfig) -> Self {
        Self {
            config,
            _backend: std::marker::PhantomData,
        }
    }

    /// Exécute l'entraînement équilibré du modèle
    pub async fn run(&self) -> anyhow::Result<()> {
        println!("🚀 Démarrage de l'entraînement ÉQUILIBRÉ...");

        // Création du modèle
        let device = B::Device::default();
        let model: MalariaCNN<B> = MalariaCNN::new(&device, &self.config);
        println!("✅ Modèle CNN équilibré créé avec succès");

        // Chargement des données
        println!("📁 Chargement du dataset depuis: data/");
        let full_dataset = MalariaDataset::from_directory("data")?;
        
        // Split du dataset
        let (train_dataset, valid_dataset) = full_dataset.split(0.8);
        
        println!("📊 Configuration des datasets:");
        println!("   - Train: {} images", train_dataset.len());
        println!("   - Validation: {} images", valid_dataset.len());
        println!("   - Batch size: {}", self.config.batch_size);
        println!("   - Workers: {}", self.config.num_workers);
        
        // Création des batchers
        let batcher_train = MalariaBatcher::new(
            self.config.image_height,
            self.config.image_width,
        );
        
        let batcher_valid = MalariaBatcher::new(
            self.config.image_height, 
            self.config.image_width,
        );

        // Création des data loaders
        let dataloader_train = DataLoaderBuilder::new(batcher_train)
            .batch_size(self.config.batch_size)
            .shuffle(42)
            .num_workers(self.config.num_workers)
            .build(train_dataset);

        let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
            .batch_size(self.config.batch_size)
            .num_workers(self.config.num_workers)
            .build(valid_dataset);

        // Configuration de l'optimiseur Adam avec weight decay
        let optim = AdamConfig::new()
            .with_weight_decay(Some(WeightDecayConfig::new(1e-4))); // Régularisation L2

        println!("⚡ Configuration de l'apprentissage ÉQUILIBRÉ:");
        println!("   - Époques: {}", self.config.num_epochs);
        println!("   - Batch size: {}", self.config.batch_size);
        println!("   - Taux d'apprentissage: {}", self.config.learning_rate);
        println!("   - Dropout: {}", self.config.dropout_rate);
        println!("   - Workers: {}", self.config.num_workers);
        println!("   - Cache: DÉSACTIVÉ (pour vitesse)");
        println!("   - Device: {:?}", device);

        println!("🎯 Lancement de l'entraînement...");
        
        // Construction du learner
        let learner = LearnerBuilder::new("./malaria-model-balanced")
            .metric_train_numeric(LossMetric::new())
            .metric_valid_numeric(LossMetric::new())
            .metric_train_numeric(AccuracyMetric::new())
            .metric_valid_numeric(AccuracyMetric::new())
            .with_file_checkpointer(BinFileRecorder::<FullPrecisionSettings>::new())
            .num_epochs(self.config.num_epochs)
            .grads_accumulation(self.config.grad_accum_steps)
            .summary()
            .build(model, optim.init(), self.config.learning_rate);

        // Démarrage de l'entraînement
        let model_trained = learner.fit(dataloader_train, dataloader_valid);

        println!("💾 Sauvegarde du modèle entraîné...");
        
        // Sauvegarde du modèle final
        BinFileRecorder::<FullPrecisionSettings>::new()
            .record(model_trained.model.into_record(), "./malaria-model-balanced".into())?;

        println!("✅ Entraînement équilibré terminé avec succès!");
        println!("📁 Modèle sauvegardé dans: ./malaria-model-balanced");
        
        Ok(())
    }
}