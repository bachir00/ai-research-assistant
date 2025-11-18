#!/usr/bin/env python3
"""
Script de test pour vérifier les connexions aux APIs
AI Research Assistant - Test de configuration

Usage:
    python test_apis.py

Ce script teste:
1. La connexion à l'API Groq (LLM)
2. Les APIs de recherche web disponibles (Serper, Tavily, Brave)
"""

import os
import sys
from dotenv import load_dotenv
import requests
import json
from datetime import datetime
from src.core.logging import setup_logger

logger = setup_logger("api_test_logger")

# Couleurs pour l'affichage terminal
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_status(message, status="info"):
    """Affiche un message avec une couleur selon le statut"""
    if status == "success":
        logger.info(f"{Colors.GREEN}✅ {message}{Colors.END}")
    elif status == "error":
        logger.info(f"{Colors.RED}❌ {message}{Colors.END}")
    elif status == "warning":
        logger.info(f"{Colors.YELLOW}⚠️  {message}{Colors.END}")
    elif status == "info":
        logger.info(f"{Colors.BLUE}ℹ️  {message}{Colors.END}")
    else:
        logger.info(f"   {message}")

def load_environment():
    """Charge les variables d'environnement depuis .env"""
    print_status("Chargement des variables d'environnement...", "info")
    
    # Chercher le fichier .env
    env_path = ".env"
    if not os.path.exists(env_path):
        print_status(f"Fichier .env non trouvé dans {os.getcwd()}", "error")
        return False
    
    load_dotenv(env_path)
    print_status(f"Fichier .env chargé depuis {os.path.abspath(env_path)}", "success")
    return True

def test_groq_api():
    """Test de l'API Groq"""
    print_status("Test de l'API Groq...", "info")
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print_status("GROQ_API_KEY non trouvée dans .env", "error")
        return False
    
    if api_key == "your_groq_api_key_here":
        print_status("GROQ_API_KEY n'est pas configurée (valeur par défaut)", "error")
        return False
    
    try:
        # Test avec l'API Groq
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": os.getenv("LLM_MODEL", "llama3-8b-8192"),
            "messages": [
                {
                    "role": "user", 
                    "content": "Dis simplement 'Hello' pour tester la connexion."
                }
            ],
            "max_tokens": 10,
            "temperature": 0.1
        }
        
        print_status(f"Envoi de requête test à Groq avec le modèle {data['model']}...", "info")
        response = requests.post(url, headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            message = result['choices'][0]['message']['content']
            print_status(f"API Groq fonctionne ! Réponse: '{message.strip()}'", "success")
            return True
        else:
            print_status(f"Erreur API Groq: {response.status_code} - {response.text}", "error")
            return False
            
    except requests.exceptions.Timeout:
        print_status("Timeout lors de l'appel à l'API Groq", "error")
        return False
    except Exception as e:
        print_status(f"Erreur lors du test Groq: {str(e)}", "error")
        return False

def test_serper_api():
    """Test de l'API Serper"""
    print_status("Test de l'API Serper...", "info")
    
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key or api_key == "your_serper_api_key_here":
        print_status("SERPER_API_KEY non configurée", "warning")
        return False
    
    try:
        url = "https://google.serper.dev/search"
        headers = {
            "X-API-KEY": api_key,
            "Content-Type": "application/json"
        }
        
        data = {
            "q": "artificial intelligence test",
            "num": 3
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=15)
        
        if response.status_code == 200:
            result = response.json()
            num_results = len(result.get('organic', []))
            print_status(f"API Serper fonctionne ! {num_results} résultats trouvés", "success")
            return True
        else:
            print_status(f"Erreur API Serper: {response.status_code}", "error")
            return False
            
    except Exception as e:
        print_status(f"Erreur lors du test Serper: {str(e)}", "error")
        return False

def test_tavily_api():
    """Test de l'API Tavily"""
    print_status("Test de l'API Tavily...", "info")
    
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key or api_key == "your_tavily_api_key_here":
        print_status("TAVILY_API_KEY non configurée", "warning")
        return False
    
    try:
        url = "https://api.tavily.com/search"
        headers = {
            "Content-Type": "application/json"
        }
        
        data = {
            "api_key": api_key,
            "query": "artificial intelligence test",
            "max_results": 3
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=15)
        
        if response.status_code == 200:
            result = response.json()
            num_results = len(result.get('results', []))
            print_status(f"API Tavily fonctionne ! {num_results} résultats trouvés", "success")
            return True
        else:
            print_status(f"Erreur API Tavily: {response.status_code}", "error")
            return False
            
    except Exception as e:
        print_status(f"Erreur lors du test Tavily: {str(e)}", "error")
        return False

def main():
    """Fonction principale"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}")
    print("=" * 60)
    print("   AI RESEARCH ASSISTANT - TEST DES APIS")
    print("=" * 60)
    print(f"{Colors.END}\n")
    
    print_status(f"Démarrage des tests - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "info")
    print()
    
    # Chargement de l'environnement
    if not load_environment():
        sys.exit(1)
    print()
    
    # Tests des APIs
    results = {}
    
    # Test Groq (requis)
    results['groq'] = test_groq_api()
    print()
    
    # Test des APIs de recherche
    results['serper'] = test_serper_api()
    print()
    
    results['tavily'] = test_tavily_api()
    print()
    
    
    if results['groq']:
        print_status("Groq (LLM) - Configuré et fonctionnel", "success")
    else:
        print_status("Groq (LLM) - ÉCHEC (REQUIS)", "error")
    
    search_apis = []
    if results['serper']:
        search_apis.append("Serper")
        print_status("Serper (Recherche) - Fonctionnel", "success")
    
    if results['tavily']:
        search_apis.append("Tavily")
        print_status("Tavily (Recherche) - Fonctionnel", "success")
    
    if not search_apis:
        print_status("Aucune API de recherche configurée", "warning")
        print_status("Configurez au moins Serper ou Tavily dans .env", "info")
    else:
        print_status(f"APIs de recherche disponibles: {', '.join(search_apis)}", "success")
    
    print()
    
    # Statut final
    if results['groq'] and search_apis:
        print_status("🎉 Toutes les APIs nécessaires sont configurées !", "success")
        print_status("Vous pouvez commencer à développer les agents", "info")
        return True
    elif results['groq']:
        print_status("⚠️  Groq fonctionne mais configurez une API de recherche", "warning")
        return False
    else:
        print_status("❌ Configuration incomplète - Groq requis", "error")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)