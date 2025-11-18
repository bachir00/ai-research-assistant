"""
Agent Researcher - Premier agent du pipeline.
Effectue la recherche web sur un sujet donné et retourne des sources pertinentes.
"""

from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime

from src.agents.base_agent import BaseAgent
from src.models.research_models import ResearchQuery, ResearchOutput, SearchResult
from src.models.state_models import AgentType
from src.services.search_api import SearchAPIManager, SearchAPIError
from src.services.llm_service import LLMService, LLMError
from src.core.logging import setup_logger
from config.prompts import RESEARCHER_PROMPT, SYSTEM_PROMPTS, KEYWORD_EXTRACTION_PROMPT


class ResearcherAgent(BaseAgent[ResearchQuery, ResearchOutput]):
    """
    Agent de recherche web.
    
    Responsabilités:
    - Recevoir une requête de recherche
    - Effectuer des recherches sur le web via des APIs
    - Analyser et filtrer les résultats
    - Retourner une liste de sources pertinentes
    """
    
    def __init__(
        self,
        name: str = "researcher",
        max_retries: int = 3,
        timeout: float = 120.0  # 2 minutes pour la recherche
    ):
        super().__init__(
            agent_type=AgentType.RESEARCHER,
            name=name,
            max_retries=max_retries,
            timeout=timeout
        )
        
        # Initialisation du gestionnaire de recherche
        try:
            self.search_manager = SearchAPIManager()
            self.logger.info(f"APIs disponibles: {self.search_manager.get_available_apis()}")
        except Exception as e:
            self.logger.error(f"Impossible d'initialiser le gestionnaire de recherche: {e}")
            raise
        
        # Initialisation du service LLM pour l'extraction de mots-clés
        try:
            self.llm_service = LLMService()
            self.logger.info("Service LLM initialisé pour l'extraction de mots-clés")
        except Exception as e:
            self.logger.error(f"Impossible d'initialiser le service LLM: {e}")
            raise
        
        # Configuration de recherche
        self.default_search_params = {
            "preferred_api": "tavily",
            "search_depth": "basic",
            "include_answer": True
        }
    
    def validate_input(self, input_data: ResearchQuery) -> bool:
        """
        Valide la requête de recherche.
        
        Args:
            input_data: Requête de recherche à valider
            
        Returns:
            True si la requête est valide
        """
        if not input_data.topic or len(input_data.topic.strip()) < 3:
            self.logger.error("Le sujet de recherche doit contenir au moins 3 caractères")
            return False
        
        if input_data.max_results <= 0 or input_data.max_results > 20:
            self.logger.error("Le nombre de résultats doit être entre 1 et 20")
            return False
        
        return True
    
    async def process(self, input_data: ResearchQuery) -> ResearchOutput:
        """
        Traite la requête de recherche.
        
        Args:
            input_data: Requête de recherche
            
        Returns:
            Résultats de recherche structurés
        """
        start_time = datetime.now()
        self.logger.info(f"Début de recherche pour: '{input_data.topic}'")
        
        # Préparation de la requête
        search_query = self._prepare_search_query(input_data)
        self.logger.info(f"Requête préparée: '{search_query}'")
        
        # Configuration des paramètres de recherche
        search_params = {
            **self.default_search_params,
            "search_depth": input_data.search_depth,
            "max_results": input_data.max_results
        }
        
        try:
            # Recherche principale
            results = await self.search_manager.search(
                query=search_query,
                **search_params
            )
            
            # Post-traitement des résultats
            filtered_results = self._filter_and_rank_results(
                results, 
                input_data.topic,
                input_data.keywords
            )
            
            # Limitation au nombre demandé
            final_results = filtered_results[:input_data.max_results]
            
            # Calcul du temps de recherche
            search_time = (datetime.now() - start_time).total_seconds()
            
            # Création de l'output
            research_output = ResearchOutput(
                query=input_data,
                results=final_results,
                total_found=len(results),
                search_time=search_time,
                search_engine=search_params["preferred_api"],
                timestamp=datetime.now()
            )
            
            self.logger.info(
                f"Recherche terminée: {len(final_results)} résultats finaux "
                f"sur {len(results)} trouvés en {search_time:.2f}s"
            )
            
            return research_output
            
        except SearchAPIError as e:
            self.logger.error(f"Erreur de recherche: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Erreur inattendue lors de la recherche: {e}")
            raise
    
    def _prepare_search_query(self, query: ResearchQuery) -> str:
        """
        Prépare la requête de recherche en optimisant les mots-clés.
        
        Args:
            query: Requête originale
            
        Returns:
            Requête optimisée pour la recherche
        """
        # Commencer par le sujet principal
        search_terms = [query.topic]
        
        # Ajouter les mots-clés s'ils existent
        if query.keywords:
            # Éviter la redondance avec le sujet principal
            unique_keywords = [
                kw for kw in query.keywords 
                if kw.lower() not in query.topic.lower()
            ]
            search_terms.extend(unique_keywords)
        
        # Joindre avec des espaces
        search_query = " ".join(search_terms)
        
        ##################### A Améliorer selon ce qu'on veut rechercher #################################
                 # Optimisations spécifiques selon la profondeur 
        ##################################################################################################
        if query.search_depth == "advanced":
            # Pour les recherches avancées, ajouter des termes de contexte
            if "intelligence artificielle" in search_query.lower() or "ia" in search_query.lower():
                search_query += " 2024 2025 récent"
            if "emploi" in search_query.lower() or "travail" in search_query.lower():
                search_query += " marché impact"
        
        return search_query.strip()
    
    def _filter_and_rank_results(
        self, 
        results: List[SearchResult], 
        topic: str,
        keywords: List[str]
    ) -> List[SearchResult]:
        """
        Filtre et classe les résultats par pertinence.
        
        Args:
            results: Résultats bruts de la recherche
            topic: Sujet de recherche original
            keywords: Mots-clés de recherche
            
        Returns:
            Résultats filtrés et classés
        """
        if not results:
            return []
        
        # Mots-clés pour le scoring (topic + keywords)
        scoring_terms = [topic.lower()] + [kw.lower() for kw in keywords]
        
        # Calcul du score de pertinence pour chaque résultat
        scored_results = []
        for result in results:
            score = self._calculate_relevance_score(result, scoring_terms)
            
            # Mise à jour du score dans le résultat
            result.score = score
            scored_results.append(result)
        
        # Tri par score décroissant
        scored_results.sort(key=lambda x: x.score or 0, reverse=True)
        
        # Filtrage des résultats de faible qualité
        min_score = 0.1  # Score minimum acceptable
        filtered_results = [r for r in scored_results if (r.score or 0) >= min_score]
        
        self.logger.info(f"Filtrage: {len(filtered_results)} résultats conservés sur {len(results)}")
        
        return filtered_results
    
    #Améiorer le score selon le site 
    # EX: if result.url.endswith(".edu") or result.url.endswith(".gov"):
    # score += 0.1
    def _calculate_relevance_score(
        self, 
        result: SearchResult, 
        scoring_terms: List[str]
    ) -> float:
        """
        Calcule un score de pertinence pour un résultat.
        
        Args:
            result: Résultat à scorer
            scoring_terms: Termes de référence pour le scoring
            
        Returns:
            Score entre 0 et 1
        """
        score = 0.0
        
        # Texte à analyser (titre + snippet)
        text_to_analyze = f"{result.title} {result.snippet}".lower()
        
        # Score basé sur la présence des termes de recherche
        term_matches = 0
        for term in scoring_terms:
            if term in text_to_analyze:
                term_matches += 1
        
        if scoring_terms:
            term_score = term_matches / len(scoring_terms)
            score += term_score * 0.6  # 60% du score
        
        # Bonus pour les titres pertinents
        title_matches = sum(1 for term in scoring_terms if term in result.title.lower())
        if scoring_terms:
            title_score = title_matches / len(scoring_terms)
            score += title_score * 0.3  # 30% du score
        
        # Bonus pour les sources récentes (si date disponible)
        if result.published_date:
            days_old = (datetime.now() - result.published_date.replace(tzinfo=None)).days
            if days_old <= 365:  # Moins d'un an
                recency_score = max(0, 1 - (days_old / 365))
                score += recency_score * 0.1  # 10% du score
        
        # Score existant de l'API (si disponible)
        if result.score and result.score > 0:
            score = (score + result.score) / 2  # Moyenne avec le score API
        
        return min(score, 1.0)  # Cap à 1.0
    
    async def extract_keywords_with_llm(self, topic: str) -> List[str]:
        """
        Extrait automatiquement des mots-clés pertinents à partir du sujet
        en utilisant le service LLM.
        
        Args:
            topic: Sujet de recherche
            
        Returns:
            Liste de mots-clés extraits
        """
        try:
            self.logger.info(f"Extraction de mots-clés pour: '{topic}'")
            
            # Préparation du prompt avec le template
            prompt = KEYWORD_EXTRACTION_PROMPT.format(topic=topic)
            
            # Appel au service LLM
            response = await self.llm_service.generate_completion(
                prompt=prompt,
                system_prompt="Tu es un expert en analyse sémantique spécialisé dans l'extraction de mots-clés pour la recherche web.",
                temperature=0.3,  # Faible température pour plus de cohérence
                max_tokens=150    # Limite pour les mots-clés
            )
            
            # Parsing de la réponse
            keywords = self._parse_keywords_response(response)
            
            self.logger.info(f"Mots-clés extraits: {keywords}")
            return keywords
            
        except LLMError as e:
            self.logger.error(f"Erreur LLM lors de l'extraction de mots-clés: {e}")
            # Fallback: extraction simple basée sur le sujet
            return self._extract_keywords_fallback(topic)
        except Exception as e:
            self.logger.error(f"Erreur inattendue lors de l'extraction de mots-clés: {e}")
            return self._extract_keywords_fallback(topic)
    
    def _parse_keywords_response(self, response: str) -> List[str]:
        """
        Parse la réponse du LLM pour extraire les mots-clés.
        
        Args:
            response: Réponse brute du LLM
            
        Returns:
            Liste de mots-clés nettoyés
        """
        # Nettoyer la réponse
        response = response.strip()
        
        # Supprimer les préfixes potentiels
        for prefix in ["mots-clés:", "keywords:", "réponse:", "voici:", "liste:"]:
            if response.lower().startswith(prefix):
                response = response[len(prefix):].strip()
        
        # Séparer par virgules
        keywords = [kw.strip() for kw in response.split(",")]
        
        # Nettoyer et filtrer
        cleaned_keywords = []
        for kw in keywords:
            # Supprimer les numéros et tirets
            kw = kw.strip("0123456789.-\t\n ")
            
            # Filtrer les mots trop courts ou vides
            if len(kw) >= 2 and kw.lower() not in ["et", "ou", "le", "la", "les", "de", "du", "des"]:
                cleaned_keywords.append(kw)
        
        # Limiter le nombre de mots-clés
        return cleaned_keywords[:7]
    
    def _extract_keywords_fallback(self, topic: str) -> List[str]:
        """
        Méthode de fallback pour extraire des mots-clés simples.
        
        Args:
            topic: Sujet de recherche
            
        Returns:
            Liste de mots-clés basiques
        """
        self.logger.info("Utilisation du fallback pour l'extraction de mots-clés")
        
        # Mots communs à ignorer
        stop_words = {
            "le", "la", "les", "de", "du", "des", "et", "ou", "sur", "dans", 
            "avec", "pour", "par", "en", "à", "un", "une", "ce", "cette", "ces"
        }
        
        # Extraction simple basée sur les mots significatifs
        words = topic.lower().split()
        keywords = [word for word in words if len(word) >= 3 and word not in stop_words]
        
        return keywords[:5]  # Limiter à 5 mots-clés max
    
    async def search_with_fallback(
        self, 
        query: str, 
        max_results: int = 5
    ) -> List[SearchResult]:
        """
        Méthode utilitaire pour recherche simple avec fallback.
        
        Args:
            query: Requête de recherche simple
            max_results: Nombre de résultats souhaités
            
        Returns:
            Liste des résultats
        """
        research_query = ResearchQuery(
            topic=query,
            max_results=max_results
        )
        
        output = await self.process(research_query)
        return output.results
    
    def get_search_stats(self) -> Dict[str, Any]:
        """
        Retourne les statistiques de recherche de l'agent.
        
        Returns:
            Dictionnaire avec les statistiques
        """
        base_stats = self.get_status()
        search_stats = {
            "available_apis": self.search_manager.get_available_apis(),
            "search_params": self.default_search_params
        }
        
        return {**base_stats, **search_stats}
    

# Fonctions utilitaires pour la sauvegarde
def save_research_output(output: ResearchOutput, filename: str = None) -> str:
    """
    Sauvegarde un ResearchOutput dans un fichier JSON.
    
    Args:
        output: Sortie de recherche à sauvegarder
        filename: Nom du fichier (optionnel)
        
    Returns:
        Nom du fichier sauvegardé
    """
    import json
    from datetime import datetime
    
    if not filename:
        # Générer un nom de fichier basé sur le sujet et timestamp
        clean_topic = "".join(c for c in output.query.topic if c.isalnum() or c in (' ', '-', '_')).rstrip()
        clean_topic = clean_topic.replace(' ', '_')[:30]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"research_output_{clean_topic}_{timestamp}.json"
    
    try:
        # Conversion en dictionnaire avec sérialisation des dates
        output_dict = output.model_dump(mode='json')
        
        # Sauvegarde dans le fichier
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output_dict, f, indent=2, ensure_ascii=False)
        
        return filename
        
    except Exception as e:
        raise Exception(f"Erreur lors de la sauvegarde: {e}")


def load_research_output(filename: str) -> ResearchOutput:
    """
    Charge un ResearchOutput depuis un fichier JSON.
    
    Args:
        filename: Nom du fichier à charger
        
    Returns:
        ResearchOutput chargé
    """
    import json
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Reconstruction du ResearchOutput
        return ResearchOutput(**data)
        
    except Exception as e:
        raise Exception(f"Erreur lors du chargement: {e}")


# Ecrire un main pour tester ici la classe
if __name__ == "__main__":
    import asyncio
    import json
    from datetime import datetime
    from src.core.logging import setup_logger
    logger = setup_logger("researcher_agent_test")
    
    async def main():
        agent = ResearcherAgent()
        
        # Test 1: Extraction automatique de mots-clés avec LLM
        topic = "impact de l'intelligence artificielle sur le marché de l'emploi"
        logger.info(f"=== Test d'extraction de mots-clés pour: {topic} ===")
        
        try:
            # Extraction automatique des mots-clés
            keywords = await agent.extract_keywords_with_llm(topic)
            logger.info(f"Mots-clés extraits automatiquement: {keywords}")
            
            # Création de la requête avec les mots-clés extraits
            query = ResearchQuery(
                topic=topic,
                keywords=keywords,  # Utilisation des mots-clés extraits automatiquement
                max_results=2,
                search_depth="basic"
            )   
            
            if agent.validate_input(query):
                logger.info("=== Début de la recherche avec mots-clés automatiques ===")
                output = await agent.process(query)
                logger.info(f"Résultats obtenus: {len(output.results)}")
                
                # Affichage des résultats
                for i, res in enumerate(output.results, 1):
                    logger.info(f"{i}. {res.title}")
                    logger.info(f"   URL: {res.url}")
                    logger.info(f"   Score: {res.score:.3f}")
                    logger.info(f"   Snippet: {res.snippet[:100]}...")
                    logger.info("")
                
                # === SAUVEGARDE DU RESEARCHOUTPUT ===
                logger.info("=== Sauvegarde du ResearchOutput ===")
                
                try:
                    filename = save_research_output(output)
                    logger.info(f"✅ ResearchOutput sauvegardé dans: {filename}")
                    
                    # Affichage du contenu sauvegardé
                    logger.info("📄 Contenu sauvegardé:")
                    logger.info(f"  • Sujet: {output.query.topic}")
                    logger.info(f"  • Mots-clés: {output.query.keywords}")
                    logger.info(f"  • Nombre de résultats: {len(output.results)}")
                    logger.info(f"  • Temps de recherche: {output.search_time:.2f}s")
                    logger.info(f"  • Moteur utilisé: {output.search_engine}")
                    logger.info(f"  • Timestamp: {output.timestamp}")
                    
                    # Test de chargement pour vérifier l'intégrité
                    logger.info("=== Test de chargement ===")
                    loaded_output = load_research_output(filename)
                    logger.info(f"✅ ResearchOutput rechargé avec succès")
                    logger.info(f"  • Vérification: {len(loaded_output.results)} résultats chargés")
                    
                    # Comparaison des données
                    if loaded_output.query.topic == output.query.topic:
                        logger.info("✅ Intégrité des données vérifiée")
                    else:
                        logger.error("❌ Erreur d'intégrité des données")
                    
                    # Affichage du format JSON pour référence
                    logger.info("\n📋 EXEMPLE DE FORMAT JSON SAUVEGARDÉ:")
                    logger.info("-" * 50)
                    
                    # Créer un exemple compact pour l'affichage
                    example_output = {
                        "query": {
                            "topic": output.query.topic,
                            "keywords": output.query.keywords[:3],  # Limiter pour l'affichage
                            "max_results": output.query.max_results,
                            "search_depth": output.query.search_depth
                        },
                        "results": [
                            {
                                "title": res.title,
                                "url": str(res.url),
                                "snippet": res.snippet[:100] + "...",
                                "score": res.score
                            } for res in output.results[:2]  # Limiter à 2 résultats pour l'affichage
                        ],
                        "total_found": output.total_found,
                        "search_time": output.search_time,
                        "search_engine": output.search_engine,
                        "timestamp": output.timestamp.isoformat()
                    }
                    
                    print(json.dumps(example_output, indent=2, ensure_ascii=False))
                    
                except Exception as save_error:
                    logger.error(f"❌ Erreur lors de la sauvegarde: {save_error}")
                
            else:
                logger.error("Requête invalide.")
                
        except Exception as e:
            logger.error(f"Erreur lors du test: {e}")
    
    # Fonction utilitaire pour tester la sauvegarde indépendamment
    async def test_save_load():
        """Test spécifique de sauvegarde/chargement."""
        logger.info("=== TEST SAUVEGARDE/CHARGEMENT SEUL ===")
        
        # Créer un ResearchOutput factice pour le test
        from datetime import datetime
        
        fake_results = [
            SearchResult(
                title="Test Article 1",
                url="https://example.com/test1",
                snippet="Ceci est un test de snippet pour l'article 1",
                score=0.85
            ),
            SearchResult(
                title="Test Article 2", 
                url="https://example.com/test2",
                snippet="Ceci est un test de snippet pour l'article 2",
                score=0.78
            )
        ]
        
        fake_query = ResearchQuery(
            topic="test sauvegarde",
            keywords=["test", "sauvegarde", "json"],
            max_results=2
        )
        
        fake_output = ResearchOutput(
            query=fake_query,
            results=fake_results,
            total_found=2,
            search_time=1.5,
            search_engine="test",
            timestamp=datetime.now()
        )
        
        try:
            # Test de sauvegarde
            filename = save_research_output(fake_output, "test_research_output.json")
            logger.info(f"✅ Test sauvegarde réussi: {filename}")
            
            # Test de chargement
            loaded = load_research_output(filename)
            logger.info(f"✅ Test chargement réussi: {len(loaded.results)} résultats")
            
        except Exception as e:
            logger.error(f"❌ Test sauvegarde/chargement échoué: {e}")
    
    # Choix du test à exécuter
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test-save":
        asyncio.run(test_save_load())
    else:
        asyncio.run(main())