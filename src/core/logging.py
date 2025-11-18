"""
Configuration du système de logging pour l'assistant de recherche.
Permet de tracer les événements importants (infos, erreurs, avertissements, etc.)
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

# --- Création du dossier des logs ---
log_directory = Path("logs")
log_directory.mkdir(exist_ok=True)

# --- Fonction de configuration du logger ---
def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Configure et retourne un logger complet avec console et fichiers rotatifs.

    Args:
        name (str): Nom du logger (ex: 'research_assistant')
        level (int): Niveau minimal de logging (par défaut: INFO)

    Returns:
        logging.Logger: Logger configuré
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Éviter les doublons si setup_logger() est appelé plusieurs fois
    if logger.hasHandlers():
        logger.handlers.clear()

    # Formatage lisible du message de log
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # --- Handler Console (affichage terminal) ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)  # plus verbeux pour la console
    console_handler.setFormatter(formatter)

    # --- Handler Fichier (logs persistants) ---
    file_handler = RotatingFileHandler(
        log_directory / f"{name}.log",
        maxBytes=5 * 1024 * 1024,  # 5 Mo
        backupCount=5,             # garder 5 fichiers d'historique
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)  # moins de bruit dans les fichiers
    file_handler.setFormatter(formatter)

    # --- Ajout des handlers au logger ---
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # Message de démarrage du logger
    logger.info("Logger initialisé avec succès.")
    return logger


# --- Exemple d’utilisation du logger ---
app_logger = setup_logger("research_assistant")

if __name__ == "__main__":
    app_logger.debug("Ceci est un message DEBUG (utile pour le débogage).")
    app_logger.info("Démarrage de l'application de recherche...")
    app_logger.warning("Avertissement : connexion lente à la base de données.")
    app_logger.error("Erreur : impossible de charger un fichier de configuration.")
    app_logger.critical("ERREUR CRITIQUE : application arrêtée.")
    app_logger.info("Application terminée.")