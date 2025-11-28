//! Point d'entrée principal équilibré de l'application de détection du paludisme

mod model;
mod training;
mod config;
mod data;

use anyhow::Result;
use burn::backend::{Autodiff};
use burn_ndarray::NdArray;
use training::trainer::MalariaTrainer;
use config::model_config::ModelConfig;

/// Backend principal avec autodiff
type Backend = Autodiff<NdArray<f64>>;

#[tokio::main]
async fn main() -> Result<()> {
    // Configuration ÉQUILIBRÉE qualité/vitesse
    let config = ModelConfig {
        image_width: 80,        // Compromis qualité/vitesse
        image_height: 80,       // Compromis qualité/vitesse
        conv1_filters: 24,      // Capacité préservée
        conv2_filters: 48,      // Capacité préservée  
        conv3_filters: 96,      // Capacité préservée
        fc1_units: 192,         // Bonne capacité
        fc2_units: 64,          // Bonne capacité
        dropout_rate: 0.3,      // Régularisation adaptée
        learning_rate: 0.001,
        batch_size: 64,         // Stable et efficace
        num_epochs: 15,         // Suffisant pour convergence
        use_cache: false,       // Désactivé pour vitesse
        num_workers: 2,         // Équilibre vitesse/stabilité
        grad_accum_steps: 1,
        ..Default::default()
    };

    println!("🚀 Initialisation de l'entraînement ÉQUILIBRÉ du CNN");
    println!("📊 Configuration optimisée qualité/vitesse:");
    println!("   - Image: 80x80 (bon compromis)");
    println!("   - Filtres: 24→48→96 (capacité préservée)");
    println!("   - FC: 192→64 (bonne capacité)");
    println!("   - Batch size: 64 (stable)");
    println!("   - Époques: 15 (suffisant pour convergence)");
    println!("   - Dropout: 0.3 (bonne régularisation)");
    println!("   - Cache: DÉSACTIVÉ (pour vitesse)");
    println!("   - Workers: 2 (équilibre vitesse/stabilité)");
    println!("🎯 Objectif: Qualité préservée à 85-92% avec temps réduit à 2-4 heures");

    // Création et démarrage de l'entraînement équilibré
    let trainer: MalariaTrainer<Backend> = MalariaTrainer::new(config);
    
    match trainer.run().await {
        Ok(_) => {
            println!("🎉 Entraînement équilibré terminé avec succès!");
            println!("📈 Qualité préservée avec temps d'entraînement réduit!");
        }
        Err(e) => {
            eprintln!("❌ Erreur pendant l'entraînement: {}", e);
            eprintln!("🔧 Vérifiez que le dossier 'data/' contient:");
            eprintln!("   - Parasitized/ (images infectées)");
            eprintln!("   - Uninfected/ (images saines)");
        }
    }
    
    Ok(())
}