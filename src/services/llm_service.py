"""
Service LLM pour l'intégration avec Groq et autres fournisseurs.
Gère les appels aux modèles de langage pour le résumé et l'analyse.
"""

import asyncio
import aiohttp
import json
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import time

from config.settings import api_config
from src.core.logging import setup_logger
import traceback


class LLMError(Exception):
    """Exception pour les erreurs LLM."""
    pass


class LLMRateLimitError(LLMError):
    """Exception pour les erreurs de limite de taux."""
    pass


class LLMService:
    """
    Service pour les appels aux modèles de langage.
    
    Fonctionnalités:
    - Support de Groq API
    - Gestion des limites de taux
    - Retry automatique avec backoff
    - Streaming optionnel
    - Validation des réponses
    """
    
    def __init__(self):
        self.config = api_config
        self.logger = setup_logger("llm_service")
        
        # Configuration Groq
        self.groq_api_key = self.config.GROQ_API_KEY
        self.groq_base_url = "https://api.groq.com/openai/v1"
        self.default_model = getattr(self.config, 'GROQ_MODEL', "llama-3.1-8b-instant")
        
        # Gestion des limites de taux
        self.rate_limit_requests = 30  # Requêtes par minute
        self.rate_limit_tokens = 6000  # Tokens par minute
        self.request_timestamps = []
        
        # Configuration par défaut
        self.default_params = {
            "temperature": 0.3,
            "max_tokens": 2000,
            "top_p": 0.9,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.1
        }
        
        # Headers pour les requêtes
        self.headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }
    
    async def generate_completion(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Génère une complétion de texte.
        
        Args:
            prompt: Prompt utilisateur
            system_prompt: Prompt système optionnel
            model: Modèle à utiliser (défaut: config)
            **kwargs: Paramètres supplémentaires pour l'API
            
        Returns:
            Réponse générée par le modèle
            
        Raises:
            LLMError: En cas d'erreur API
            LLMRateLimitError: En cas de dépassement de limite
        """
        # Préparer les messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Paramètres de la requête
        params = {**self.default_params, **kwargs}
        payload = {
            "model": model or self.default_model,
            "messages": messages,
            **params
        }
        
        # Gestion des limites de taux
        await self._check_rate_limits()
        
        # Appel API avec retry
        return await self._make_api_call(payload)
    
    async def generate_batch_completions(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        max_concurrent: int = 3,
        **kwargs
    ) -> List[str]:
        """
        Génère plusieurs complétions en parallèle.
        
        Args:
            prompts: Liste des prompts
            system_prompt: Prompt système optionnel
            model: Modèle à utiliser
            max_concurrent: Nombre maximum de requêtes simultanées
            **kwargs: Paramètres supplémentaires
            
        Returns:
            Liste des réponses dans le même ordre que les prompts
        """
        self.logger.info(f"Génération batch de {len(prompts)} complétions")
        
        # Créer un semaphore pour limiter la concurrence
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def generate_single(prompt: str, index: int) -> tuple:
            async with semaphore:
                try:
                    # Délai pour éviter le rate limiting
                    await asyncio.sleep(index * 0.5)
                    
                    result = await self.generate_completion(
                        prompt, system_prompt, model, **kwargs
                    )
                    return index, result
                except Exception as e:
                    self.logger.error(f"Erreur completion {index}: {e}")
                    return index, f"ERREUR: {str(e)}"
        
        # Lancer toutes les tâches
        tasks = [generate_single(prompt, i) for i, prompt in enumerate(prompts)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Réorganiser les résultats dans l'ordre
        ordered_results = [""] * len(prompts)
        for result in results:
            if isinstance(result, tuple):
                index, content = result
                ordered_results[index] = content
            else:
                # Exception - la placer à la fin
                ordered_results.append(f"EXCEPTION: {str(result)}")
        
        success_count = sum(1 for r in ordered_results if not r.startswith("ERREUR"))
        self.logger.info(f"Batch terminé: {success_count}/{len(prompts)} succès")
        
        return ordered_results
    
    async def _make_api_call(self, payload: Dict[str, Any], max_retries: int = 3) -> str:
        """Effectue l'appel API avec retry automatique."""
        url = f"{self.groq_base_url}/chat/completions"
        
        for attempt in range(max_retries + 1):
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
                    async with session.post(url, json=payload, headers=self.headers) as response:
                        
                        # Enregistrer la requête pour rate limiting
                        self.request_timestamps.append(time.time())
                        
                        if response.status == 200:
                            data = await response.json()
                            content = data["choices"][0]["message"]["content"]
                            
                            # Validation de base
                            if not content or content.strip() == "":
                                raise LLMError("Réponse vide du modèle")
                            
                            return content.strip()
                            
                        elif response.status == 429:
                            # Rate limit atteint
                            retry_after = int(response.headers.get("retry-after", 60))
                            self.logger.warning(f"Rate limit atteint, attente {retry_after}s")
                            
                            if attempt < max_retries:
                                await asyncio.sleep(retry_after)
                                continue
                            else:
                                raise LLMRateLimitError("Limite de taux API dépassée")
                                
                        else:
                            # Autres erreurs HTTP
                            error_text = await response.text()
                            error_msg = f"Erreur API {response.status}: {error_text}"
                            
                            if attempt < max_retries:
                                self.logger.warning(f"{error_msg} - Tentative {attempt + 1}/{max_retries}")
                                await asyncio.sleep(2 ** attempt)  # Backoff exponentiel
                                continue
                            else:
                                raise LLMError(error_msg)
                                
            except asyncio.TimeoutError:
                if attempt < max_retries:
                    self.logger.warning(f"Timeout API - Tentative {attempt + 1}/{max_retries}")
                    await asyncio.sleep(2 ** attempt)
                    continue
                else:
                    raise LLMError("Timeout API après plusieurs tentatives")
                    
            except Exception as e:
                if attempt < max_retries:
                    self.logger.warning(f"Erreur réseau: {e} - Tentative {attempt + 1}/{max_retries}")
                    await asyncio.sleep(2 ** attempt)
                    continue
                else:
                    raise LLMError(f"Erreur de connexion: {str(e)}")
        
        raise LLMError("Toutes les tentatives ont échoué")
    
    async def _check_rate_limits(self):
        """Vérifie et applique les limites de taux."""
        current_time = time.time()
        
        # Nettoyer les timestamps anciens (plus de 1 minute)
        self.request_timestamps = [
            ts for ts in self.request_timestamps 
            if current_time - ts < 60
        ]
        
        # Vérifier si on dépasse la limite
        if len(self.request_timestamps) >= self.rate_limit_requests:
            oldest_request = min(self.request_timestamps)
            wait_time = 60 - (current_time - oldest_request)
            
            if wait_time > 0:
                self.logger.info(f"Rate limit: attente {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
    
    def estimate_tokens(self, text: str) -> int:
        """Estime le nombre de tokens dans un texte."""
        # Approximation: 1 token ≈ 4 caractères pour l'anglais/français
        return len(text) // 4
    
    def validate_input_length(self, text: str, max_tokens: int = 6000) -> bool:
        """Valide que le texte ne dépasse pas la limite de tokens."""
        estimated_tokens = self.estimate_tokens(text)
        return estimated_tokens <= max_tokens
    
    def truncate_text(self, text: str, max_tokens: int = 6000) -> str:
        """Tronque un texte pour respecter la limite de tokens."""
        estimated_tokens = self.estimate_tokens(text)
        
        if estimated_tokens <= max_tokens:
            return text
        
        # Calculer le ratio de troncature
        ratio = max_tokens / estimated_tokens
        target_length = int(len(text) * ratio * 0.9)  # Marge de sécurité
        
        # Tronquer en préservant les phrases
        sentences = text.split('. ')
        truncated = ""
        
        for sentence in sentences:
            if len(truncated) + len(sentence) + 2 <= target_length:
                truncated += sentence + ". "
            else:
                break
        
        self.logger.info(f"Texte tronqué: {len(text)} → {len(truncated)} caractères")
        return truncated.strip()
    
    async def test_connection(self) -> bool:
        """Teste la connexion à l'API."""
        try:
            result = await self.generate_completion(
                "Test de connexion. Réponds juste 'OK'.",
                system_prompt="Tu es un assistant de test."
            )
            
            if "ok" in result.lower():
                self.logger.info("Test de connexion LLM réussi")
                return True
            else:
                self.logger.warning(f"Test de connexion étrange: {result}")
                return False
                
        except Exception as e:
            self.logger.error(f"Test de connexion LLM échoué: {e}")
            return False


class LLMManager:
    """
    Gestionnaire de services LLM avec stratégies multiples.
    """
    
    def __init__(self):
        self.logger = setup_logger("llm_manager")
        self.primary_service = LLMService()
        self.services = {
            "groq": self.primary_service
        }
    
    async def get_completion(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        service: str = "groq",
        **kwargs
    ) -> str:
        """
        Obtient une complétion en utilisant le service spécifié.
        
        Args:
            prompt: Prompt utilisateur
            system_prompt: Prompt système
            service: Service LLM à utiliser
            **kwargs: Paramètres supplémentaires
            
        Returns:
            Réponse du modèle
        """
        if service not in self.services:
            raise ValueError(f"Service LLM inconnu: {service}")
        
        llm_service = self.services[service]
        return await llm_service.generate_completion(prompt, system_prompt, **kwargs)
    
    async def get_batch_completions(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        service: str = "groq",
        **kwargs
    ) -> List[str]:
        """Obtient des complétions en batch."""
        if service not in self.services:
            raise ValueError(f"Service LLM inconnu: {service}")
        
        llm_service = self.services[service]
        return await llm_service.generate_batch_completions(
            prompts, system_prompt, **kwargs
        )
    
    async def test_all_services(self) -> Dict[str, bool]:
        """Teste tous les services LLM disponibles."""
        results = {}
        
        for name, service in self.services.items():
            try:
                results[name] = await service.test_connection()
            except Exception as e:
                self.logger.error(f"Test service {name} échoué: {e}")
                results[name] = False
        
        return results
    
# Exemple d'utilisation du service LLM

async def example_usage():
    """Exemple d'utilisation du service LLM."""
    
    # 1. Test de connexion simple
    print("=== Test de connexion ===")
    llm_service = LLMService()
    
    connection_ok = await llm_service.test_connection()
    print(f"Connexion LLM: {'✓ OK' if connection_ok else '✗ Échec'}")
    
    if not connection_ok:
        print("Impossible de continuer sans connexion")
        return
    
    # 2. Génération simple
    print("\n=== Génération simple ===")
    try:
        response = await llm_service.generate_completion(
            prompt="Explique-moi en 2 phrases ce qu'est l'intelligence artificielle.",
            system_prompt="Tu es un expert en IA qui explique simplement."
        )
        print(f"Réponse: {response}")
    except Exception as e:
        print(f"Erreur: {e}")
    
    # 3. Génération avec paramètres personnalisés
    print("\n=== Génération avec paramètres ===")
    try:
        response = await llm_service.generate_completion(
            prompt="Écris un haiku sur la technologie.",
            system_prompt="Tu es un poète spécialisé dans les haikus.",
            temperature=0.8,
            max_tokens=100
        )
        print(f"Haiku: {response}")
    except Exception as e:
        print(f"Erreur: {e}")
    
    # 4. Génération en batch
    print("\n=== Génération en batch ===")
    prompts = [
        "Qu'est-ce que Python?",
        "Qu'est-ce que JavaScript?",
        "Qu'est-ce que Rust?"
    ]
    
    try:
        responses = await llm_service.generate_batch_completions(
            prompts=prompts,
            system_prompt="Réponds en une phrase courte.",
            max_concurrent=2
        )
        
        for i, (prompt, response) in enumerate(zip(prompts, responses)):
            print(f"{i+1}. {prompt}")
            print(f"   → {response}\n")
    except Exception as e:
        print(f"Erreur batch: {e}")
    
    # 5. Test des utilitaires
    print("\n=== Test des utilitaires ===")
    long_text = "Ceci est un texte très long. " * 1000
    print(f"Texte original: {len(long_text)} caractères")
    print(f"Tokens estimés: {llm_service.estimate_tokens(long_text)}")

    is_valid = llm_service.validate_input_length(long_text, max_tokens=7000)
    print(f"Texte valide (7000 tokens max): {is_valid}")
    
    if not is_valid:
        truncated = llm_service.truncate_text(long_text, max_tokens=7000)
        print(f"Texte tronqué: {len(truncated)} caractères")
        print(f"Contenu: {truncated[:200]}...")

# Test avec le gestionnaire LLM
async def example_manager_usage():
    """Exemple d'utilisation du gestionnaire LLM."""
    
    print("\n=== Test du gestionnaire LLM ===")
    
    manager = LLMManager()
    
    # Test de tous les services
    service_status = await manager.test_all_services()
    print("État des services:")
    for service, status in service_status.items():
        print(f"  {service}: {'✓' if status else '✗'}")
    
    # Utilisation via le gestionnaire
    try:
        response = await manager.get_completion(
            prompt="Salut! Comment ça va?",
            system_prompt="Tu es un assistant amical.",
            service="groq"
        )
        print(f"\nRéponse du gestionnaire: {response}")
    except Exception as e:
        print(f"Erreur gestionnaire: {e}")

# Fonction principale pour tester
async def main():
    """Fonction principale de test."""
    try:
        await example_usage()
        await example_manager_usage()
    except KeyboardInterrupt:
        print("\n\nTest interrompu par l'utilisateur")
    except Exception as e:
        print(f"\nErreur inattendue: {e}")
        traceback.print_exc()

# Pour exécuter le test
if __name__ == "__main__":
    print("🚀 Démarrage du test du service LLM...")
    asyncio.run(main())